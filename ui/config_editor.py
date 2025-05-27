import gradio as gr
import os
from typing import Dict, Any, List
from .base import BaseUI
from .components import (
    create_component, get_from_nested_dict, set_in_nested_dict,
    load_config, save_config
)
from config.path_config import SUPPORTED_CONFIG_EXTENSIONS

# Define parameter paths and UI components with defaults
PARAMS = [
    # Image Processing
    {"section": "Image Processing", "name": "min_dimension_threshold", "path": "image_processing.min_dimension_threshold", "type": "number", "label": "Minimum Dimension Threshold", "default": 800, "precision": 0},
    {"section": "Image Processing", "name": "upscale_factor", "path": "image_processing.upscale_factor", "type": "number", "label": "Upscale Factor", "default": 2.0, "precision": 1},
    
    # Text Detection
    {"section": "Text Detection", "name": "bin_thresh", "path": "text_detection.bin_thresh", "type": "slider", "label": "Binary Threshold", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
    {"section": "Text Detection", "name": "box_thresh", "path": "text_detection.box_thresh", "type": "slider", "label": "Box Threshold", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
    
    # Text Removal
    {"section": "Text Removal", "name": "doctr_score_thresh", "path": "text_removal.doctr_score_thresh", "type": "slider", "label": "Doctr Score Threshold", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
    {"section": "Text Removal", "name": "remove_contour_overlap_thresh", "path": "text_removal.remove_contour_overlap_thresh", "type": "slider", "label": "Contour Overlap Threshold", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    
    # Shape Detection
    {"section": "Shape Detection", "name": "verbose", "path": "shape_detection.verbose", "type": "checkbox", "label": "Verbose", "default": True},
    {"section": "Shape Detection", "name": "fill_circle_thresh", "path": "shape_detection.fill_circle_enclosing_threshold", "type": "slider", "label": "Fill Circle Enclosing Threshold", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
    {"section": "Shape Detection", "name": "fill_rect_thresh", "path": "shape_detection.fill_rect_enclosing_threshold", "type": "slider", "label": "Fill Rectangle Enclosing Threshold", "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05},
    {"section": "Shape Detection", "name": "erosion_kernel_size", "path": "shape_detection.erosion_kernel_size", "type": "text", "label": "Erosion Kernel Size (format: [w, h])", "default": "[3, 3]", "parse": "list"},
    {"section": "Shape Detection", "name": "min_stable_length", "path": "shape_detection.min_stable_length", "type": "number", "label": "Minimum Stable Length", "default": 3, "precision": 0},
    {"section": "Shape Detection", "name": "max_erosion_iter", "path": "shape_detection.max_erosion_iterations", "type": "number", "label": "Maximum Erosion Iterations", "default": 30, "precision": 0},
    {"section": "Shape Detection", "name": "classify_circle_thresh", "path": "shape_detection.classify_circle_overlap_threshold", "type": "slider", "label": "Circle Classification Threshold", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
    {"section": "Shape Detection", "name": "classify_rect_thresh", "path": "shape_detection.classify_rect_overlap_threshold", "type": "slider", "label": "Rectangle Classification Threshold", "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05},
    {"section": "Shape Detection", "name": "remove_nodes_kernel", "path": "shape_detection.remove_nodes_dilation_kernel_size", "type": "text", "label": "Remove Nodes Dilation Kernel Size (format: [w, h])", "default": "[3, 3]", "parse": "list"},
    {"section": "Shape Detection", "name": "remove_nodes_iter", "path": "shape_detection.remove_nodes_dilation_iterations", "type": "number", "label": "Remove Nodes Dilation Iterations", "default": 3, "precision": 0},
    
    # Connection Processing - Hough Line Transform
    {"section": "Connection Processing", "subsection": "Hough Line Transform", "name": "hough_rho", "path": "connection_processing.hough_rho", "type": "number", "label": "Rho", "default": 1, "precision": 0},
    {"section": "Connection Processing", "subsection": "Hough Line Transform", "name": "hough_theta", "path": "connection_processing.hough_theta_degrees", "type": "number", "label": "Theta (degrees)", "default": 1.0, "precision": 1},
    {"section": "Connection Processing", "subsection": "Hough Line Transform", "name": "hough_threshold", "path": "connection_processing.hough_threshold", "type": "number", "label": "Threshold", "default": 15, "precision": 0},
    {"section": "Connection Processing", "subsection": "Hough Line Transform", "name": "hough_min_line_len", "path": "connection_processing.hough_min_line_length", "type": "number", "label": "Minimum Line Length", "default": 10, "precision": 0},
    {"section": "Connection Processing", "subsection": "Hough Line Transform", "name": "hough_max_line_gap", "path": "connection_processing.hough_max_line_gap", "type": "number", "label": "Maximum Line Gap", "default": 25, "precision": 0},
    
    # Connection Processing - Hough Bundler
    {"section": "Connection Processing", "subsection": "Bundler Settings", "name": "hough_bundler_min_dist", "path": "connection_processing.hough_bundler_min_distance", "type": "number", "label": "Bundler Min Distance", "default": 10.0, "precision": 1},
    {"section": "Connection Processing", "subsection": "Bundler Settings", "name": "hough_bundler_min_angle", "path": "connection_processing.hough_bundler_min_angle", "type": "number", "label": "Bundler Min Angle", "default": 5.0, "precision": 1},
    
    # Connection Processing - Arrowhead Detection
    {"section": "Connection Processing", "subsection": "Arrowhead Detection", "name": "arrow_conf_threshold", "path": "connection_processing.arrowhead_confidence_threshold_percent", "type": "number", "label": "Confidence Threshold (%)", "default": 10.0, "precision": 1},
    
    # Connection Processing - Path Finding
    {"section": "Connection Processing", "subsection": "Path Finding", "name": "path_proximity_threshold", "path": "connection_processing.path_finding.proximity_threshold", "type": "number", "label": "Proximity Threshold", "default": 30.0, "precision": 1},
    {"section": "Connection Processing", "subsection": "Path Finding", "name": "path_dot_product_weight", "path": "connection_processing.path_finding.dot_product_weight", "type": "number", "label": "Dot Product Weight", "default": 0.6, "precision": 1},
    {"section": "Connection Processing", "subsection": "Path Finding", "name": "path_distance_line_weight", "path": "connection_processing.path_finding.distance_to_line_weight", "type": "number", "label": "Distance to Line Weight", "default": 0.2, "precision": 1},
    {"section": "Connection Processing", "subsection": "Path Finding", "name": "path_endpoint_distance_weight", "path": "connection_processing.path_finding.endpoint_distance_weight", "type": "number", "label": "Endpoint Distance Weight", "default": 0.2, "precision": 1},
    
    # Connection Processing - Other thresholds
    {"section": "Connection Processing", "subsection": "Proximity Settings", "name": "proximity_thres_place", "path": "connection_processing.proximity_thres_place", "type": "number", "label": "Place Proximity Threshold", "default": 1.5, "precision": 1},
    {"section": "Connection Processing", "subsection": "Proximity Settings", "name": "proximity_thres_trans_height", "path": "connection_processing.proximity_thres_trans_height", "type": "number", "label": "Transition Height Proximity Threshold", "default": 1.4, "precision": 1},
    {"section": "Connection Processing", "subsection": "Proximity Settings", "name": "proximity_thres_trans_width", "path": "connection_processing.proximity_thres_trans_width", "type": "number", "label": "Transition Width Proximity Threshold", "default": 3.0, "precision": 1},
    {"section": "Connection Processing", "subsection": "Proximity Settings", "name": "arrowhead_proximity_threshold", "path": "connection_processing.arrowhead_proximity_threshold", "type": "number", "label": "Arrowhead Proximity Threshold", "default": 40, "precision": 0},
    {"section": "Connection Processing", "subsection": "Proximity Settings", "name": "text_linking_threshold", "path": "connection_processing.text_linking_threshold", "type": "number", "label": "Text Linking Threshold", "default": 25.0, "precision": 1},
]

class ConfigEditor(BaseUI):
    """Configuration Editor UI component"""
    
    def __init__(self):
        super().__init__()
        self.components = {}
        self.config_data = None
        self.loaded_path = None
        self.status = None
        self.file_info = None
        self.custom_filename = None
        self.save_btn = None
        self.download_file = None
        self.success_indicator = None
        self.config_file_upload = None
    
    def create_interface(self) -> gr.TabItem:
        """Create the Configuration Editor interface"""
        with gr.TabItem("Configuration Editor") as tab:
            gr.Markdown("## Petri Net Configuration Editor")
            
            # File upload for configuration
            with gr.Row():
                self.config_file_upload = self.create_file_upload(
                    "Upload Configuration File", 
                    list(SUPPORTED_CONFIG_EXTENSIONS)
                )
                self.file_info = gr.Textbox(label="Selected File Path", interactive=False)

            self.status = gr.Textbox(label="Status", interactive=False)
            
            # Hidden state for the config UI
            self.config_data = self.create_state_var("config_data", {})
            self.loaded_path = self.create_state_var("loaded_path", "")
            
            # Create parameter sections
            self._create_parameter_sections()
            
            # Save controls
            with gr.Row():
                self.custom_filename = gr.Textbox(
                    label="Save as (optional)", 
                    placeholder="Enter new filename (leave empty for auto timestamp)",
                    interactive=True
                )
            
            self.save_btn = self.create_button("Save Configuration", variant="primary")
            self.download_file = gr.File(label="Download Configuration", visible=False, interactive=True, type="filepath")
            self.success_indicator = gr.Markdown("")
            
            self.setup_event_handlers()
            
        return tab
    
    def _create_parameter_sections(self):
        """Create the parameter configuration sections"""
        # Organize sections and subsections
        sections = {}
        for param in PARAMS:
            section = param["section"]
            if section not in sections:
                sections[section] = {"subsections": {}, "params": []}
            
            if "subsection" in param:
                subsection = param["subsection"]
                if subsection not in sections[section]["subsections"]:
                    sections[section]["subsections"][subsection] = []
                sections[section]["subsections"][subsection].append(param)
            else:
                sections[section]["params"].append(param)
        
        # Create components
        with gr.Accordion("Parameter Configuration", open=True):
            for section_name, section_data in sections.items():
                with gr.Accordion(section_name, open=False):
                    # Handle parameters directly in section
                    for i in range(0, len(section_data["params"]), 2):
                        with gr.Row():
                            param = section_data["params"][i]
                            self.components[param["name"]] = create_component(param)
                            
                            if i + 1 < len(section_data["params"]):
                                param = section_data["params"][i + 1]
                                self.components[param["name"]] = create_component(param)
                    
                    # Handle subsections
                    for subsec_name, subsec_params in section_data["subsections"].items():
                        with gr.Accordion(subsec_name, open=False):
                            for i in range(0, len(subsec_params), 2):
                                with gr.Row():
                                    param = subsec_params[i]
                                    self.components[param["name"]] = create_component(param)
                                    
                                    if i + 1 < len(subsec_params):
                                        param = subsec_params[i + 1]
                                        self.components[param["name"]] = create_component(param)
    
    def setup_event_handlers(self):
        """Setup event handlers for the Configuration Editor"""
        # Wire up events for config UI
        self.config_file_upload.upload(
            fn=self._browse_and_load,
            inputs=[self.config_file_upload],
            outputs=[self.config_data, self.loaded_path, self.status, self.file_info] + list(self.components.values())
        )
        
        # Create a list of inputs in the correct order
        save_inputs = [self.config_data, self.loaded_path, self.custom_filename] + list(self.components.values())
        
        self.save_btn.click(
            fn=self._handle_save_positional,
            inputs=save_inputs,
            outputs=[self.status, self.download_file]
        ).then(
            fn=lambda msg: "✅ Success!" if "Error" not in msg else "❌ Error!",
            inputs=[self.status],
            outputs=[self.success_indicator]
        )
    
    def _browse_and_load(self, file_obj):
        """Load selected configuration file from upload"""
        if file_obj is None:
            return {}, "", "No file selected", "", *[self.components[param["name"]].value for param in PARAMS]

        file_path = file_obj.name
        try:
            # Load the configuration
            config, success, message = load_config(file_path)
            
            if not success:
                # Keep current values if loading fails, but update status and path
                current_values = [self.components[param["name"]].value for param in PARAMS]
                return {}, "", message, file_path, *current_values
            
            # Get values for all components based on parameter definitions
            values = []
            for param in PARAMS:
                value = get_from_nested_dict(config, param["path"], param["default"])
                
                # Format values for display
                if param.get("parse") == "list" and isinstance(value, list):
                    value = str(value)
                
                values.append(value)
            
            return config, file_path, message, file_path, *values
        except Exception as e:
            # Keep current values on error, update status
            current_values = [self.components[param["name"]].value for param in PARAMS]
            return {}, "", f"Error loading file: {str(e)}", file_path if file_path else "", *current_values
    
    def _handle_save_positional(self, config_data, config_path, custom_filename, *param_values):
        """A wrapper for handle_save that uses positional arguments instead of keywords"""
        if not config_path:
            return "No configuration file loaded. Please load a configuration first.", gr.update(visible=False)
        
        try:
            # Map parameters to values
            param_dict = {}
            for i, param in enumerate(PARAMS):
                param_dict[param["name"]] = param_values[i]
            
            # Update config with values from UI
            for param in PARAMS:
                value = param_dict[param["name"]]
                
                # Parse special types
                if param.get("parse") == "list" and isinstance(value, str):
                    try:
                        value = eval(value)
                    except:
                        value = [3, 3]  # Default if parsing fails
                
                # Set the value in the config
                config_data = set_in_nested_dict(config_data, param["path"], value)
            
            # Save to file
            success, message, saved_path = save_config(config_data, config_path, custom_filename)
            
            if success and saved_path:
                filename = os.path.basename(saved_path)
                return message, gr.update(value=saved_path, visible=True, label=f"Download {filename}")
            else:
                return message, gr.update(visible=False)
                
        except Exception as e:
            return f"Error updating configuration: {str(e)}", gr.update(visible=False) 