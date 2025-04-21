import inspect
import os
import json
import typing
from functools import partial
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QLineEdit, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot

# Try to import Gemini API, but don't fail if it's not available
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class ParameterCache:
    """Cache for parameter range inferences to avoid repeated API calls"""
    
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".parameter_tuner_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def get_cache_path(self, function_name):
        return self.cache_dir / f"{function_name}_params.json"
        
    def get_cached_params(self, function_name):
        cache_path = self.get_cache_path(function_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
        
    def cache_params(self, function_name, params_info):
        cache_path = self.get_cache_path(function_name)
        with open(cache_path, 'w') as f:
            json.dump(params_info, f)

def infer_parameter_ranges(func, param_info):
    """Use Gemini to infer reasonable parameter ranges"""
    if not GEMINI_AVAILABLE:
        return default_parameter_ranges(param_info)
    
    # Cache implementation
    cache = ParameterCache()
    func_name = func.__name__
    cached_params = cache.get_cached_params(func_name)
    
    if cached_params:
        return cached_params
    
    # If no cache available, prepare to call Gemini API
    try:
        # Set up Gemini API
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return default_parameter_ranges(param_info)
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create prompt for Gemini
        prompt = f"""
        For the Python function named '{func_name}' with the following parameters:
        {param_info}
        
        Return a JSON object where each key is a parameter name and each value is an object with suggested:
        - min_value: minimum recommended value
        - max_value: maximum recommended value
        - step: recommended step size for sliders
        - datatype: the Python data type (int, float, bool, str, list, etc.)
        
        Only include parameters from the input. Format as valid, parseable JSON.
        """
        
        # Call Gemini API with structured output
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        if hasattr(response, 'text'):
            try:
                # Try to parse the JSON response
                params_ranges = json.loads(response.text)
                cache.cache_params(func_name, params_ranges)
                return params_ranges
            except Exception:
                pass
                
    except Exception:
        pass
    
    # Fall back to default ranges if API call fails
    return default_parameter_ranges(param_info)

def default_parameter_ranges(param_info):
    """Create default parameter ranges based on parameter types"""
    ranges = {}
    
    for name, info in param_info.items():
        param_type = info.get('type')
        default = info.get('default')
        
        if param_type == int or (param_type is None and isinstance(default, int)):
            ranges[name] = {
                'min_value': 0,
                'max_value': 100 if default is None else max(100, default * 2),
                'step': 1,
                'datatype': 'int'
            }
        elif param_type == float or (param_type is None and isinstance(default, float)):
            ranges[name] = {
                'min_value': 0.0,
                'max_value': 1.0 if default is None else max(1.0, default * 2.0),
                'step': 0.01,
                'datatype': 'float'
            }
        elif param_type == bool or (param_type is None and isinstance(default, bool)):
            ranges[name] = {
                'datatype': 'bool'
            }
        elif param_type == str or (param_type is None and isinstance(default, str)):
            ranges[name] = {
                'datatype': 'str'
            }
        elif param_type in (list, tuple) or isinstance(default, (list, tuple)):
            ranges[name] = {
                'datatype': 'list',
                'values': default if default else []
            }
        else:
            # Default case for unknown types
            ranges[name] = {
                'datatype': 'unknown'
            }
            
    return ranges

class ParameterTunerWindow(QMainWindow):
    """Main window for parameter tuning"""
    
    parameter_changed = Signal()
    
    def __init__(self, target_function, fixed_parameters, map_function, feedback_function, parent=None):
        super().__init__(parent)
        self.target_function = target_function
        self.fixed_parameters = fixed_parameters or {}
        self.map_function = map_function
        self.feedback_function = feedback_function
        
        self.setWindowTitle(f"Parameter Tuner - {target_function.__name__}")
        self.resize(800, 600)
        
        # Create main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Add controls area
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        layout.addWidget(self.controls_widget)
        
        # Add feedback area
        self.feedback_widget = QWidget()
        self.feedback_layout = QVBoxLayout(self.feedback_widget)
        layout.addWidget(self.feedback_widget)
        
        # Add a separator between controls and feedback
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Add run/apply button
        self.apply_button = QPushButton("Apply Parameters")
        self.apply_button.clicked.connect(self.update_output)
        layout.addWidget(self.apply_button)
        
        # Extract parameter information
        self.param_info = self.get_parameter_info()
        self.param_widgets = {}
        
        # Create UI elements for parameters
        self.create_parameter_widgets()
        
        # Connect parameter changed signal
        self.parameter_changed.connect(self.update_output)
        
        # Initial update
        self.update_output()
        
    def get_parameter_info(self):
        """Extract parameter information from target function"""
        param_info = {}
        sig = inspect.signature(self.target_function)
        
        for name, param in sig.parameters.items():
            # Skip fixed parameters
            if name in self.fixed_parameters:
                continue
                
            # Get parameter type annotation
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
            
            # Get default value
            default = param.default if param.default != inspect.Parameter.empty else None
            
            param_info[name] = {
                'type': param_type,
                'default': default
            }
            
        return param_info
        
    def create_parameter_widgets(self):
        """Create UI widgets for each parameter"""
        # Get parameter ranges from Gemini or defaults
        param_ranges = infer_parameter_ranges(self.target_function, self.param_info)
        
        for name, info in self.param_info.items():
            default = info.get('default')
            param_range = param_ranges.get(name, {})
            datatype = param_range.get('datatype', 'unknown')
            
            # Create a container for each parameter
            param_container = QWidget()
            param_layout = QHBoxLayout(param_container)
            param_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add label
            label = QLabel(f"{name}:")
            param_layout.addWidget(label)
            
            # Create appropriate widget based on parameter type
            if datatype == 'int':
                min_val = param_range.get('min_value', 0)
                max_val = param_range.get('max_value', 100)
                step = param_range.get('step', 1)
                
                widget = QSpinBox()
                widget.setMinimum(min_val)
                widget.setMaximum(max_val)
                widget.setSingleStep(step)
                widget.setValue(default if default is not None else min_val)
                widget.valueChanged.connect(lambda: self.parameter_changed.emit())
                
            elif datatype == 'float':
                min_val = param_range.get('min_value', 0.0)
                max_val = param_range.get('max_value', 1.0)
                step = param_range.get('step', 0.01)
                
                widget = QDoubleSpinBox()
                widget.setMinimum(min_val)
                widget.setMaximum(max_val)
                widget.setSingleStep(step)
                widget.setValue(default if default is not None else min_val)
                widget.valueChanged.connect(lambda: self.parameter_changed.emit())
                
            elif datatype == 'bool':
                widget = QCheckBox()
                widget.setChecked(default if default is not None else False)
                widget.stateChanged.connect(lambda: self.parameter_changed.emit())
                
            elif datatype == 'list':
                widget = QComboBox()
                values = param_range.get('values', [])
                if values:
                    widget.addItems([str(v) for v in values])
                    if default is not None and default in values:
                        widget.setCurrentText(str(default))
                widget.currentIndexChanged.connect(lambda: self.parameter_changed.emit())
                
            else:  # str or unknown
                widget = QLineEdit()
                widget.setText(str(default) if default is not None else "")
                widget.textChanged.connect(lambda: self.parameter_changed.emit())
            
            param_layout.addWidget(widget)
            self.param_widgets[name] = widget
            self.controls_layout.addWidget(param_container)
            
    def get_current_parameters(self):
        """Get current parameter values from widgets"""
        params = {}
        
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                params[name] = widget.text()
                
        return params
        
    @Slot()
    def update_output(self):
        """Update the output based on current parameters"""
        # Get current parameter values
        current_params = self.get_current_parameters()
        
        # Combine with fixed parameters
        all_params = {**self.fixed_parameters, **current_params}
        
        try:
            # Call target function with parameters
            result = self.target_function(**all_params)
            
            # Apply mapping function
            mapped_result = self.map_function(result)
            
            # Clear feedback layout
            while self.feedback_layout.count():
                item = self.feedback_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            # Create feedback widget
            feedback_container = QWidget()
            feedback_layout = QVBoxLayout(feedback_container)
            
            # Apply feedback function to create feedback
            feedback_widget = self.feedback_function(mapped_result)
            feedback_layout.addWidget(feedback_widget)
            
            self.feedback_layout.addWidget(feedback_container)
            
        except Exception as e:
            # Show error in feedback area
            error_label = QLabel(f"Error: {str(e)}")
            error_label.setStyleSheet("color: red")
            
            # Clear feedback layout
            while self.feedback_layout.count():
                item = self.feedback_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.feedback_layout.addWidget(error_label)

def interactive_parameter_tuner(target_function, fixed_parameters=None, map_function=None, feedback_function=None):
    """
    Provide interactive parameter tuning for any Python function.
    
    Parameters:
        target_function: Function whose parameters will be tuned
        fixed_parameters: Dictionary of parameter names and their fixed values
        map_function: Function to map the output of target_function to input for feedback_function
        feedback_function: Function that provides feedback for the current parameter values
    """
    # Default mapping function (identity)
    if map_function is None:
        map_function = lambda x: x
        
    # Default feedback function (create a text label)
    if feedback_function is None:
        def default_feedback(result):
            label = QLabel(str(result))
            return label
        feedback_function = default_feedback
        
    # Create Qt application if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        
    # Create and show the parameter tuner window
    window = ParameterTunerWindow(target_function, fixed_parameters, map_function, feedback_function)
    window.show()
    
    # Run the application
    app.exec()
