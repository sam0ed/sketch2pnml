import gradio as gr
import yaml
import os
from datetime import datetime
from typing import Dict, Any, Union, List
from config.path_config import ensure_directories_exist, get_config_download_path, WORKING_DIR


class ConfigEditor:
    """Simplified Configuration Editor for Petri Net Algorithm Parameters"""
    
    def __init__(self):
        self.config_data = {}
        self.error_message = ""
        
        # Parameter definitions with validation rules
        self.param_groups = {
            "Image Processing": {
                "min_dimension_threshold": {
                    "type": "number", "default": 800, "min": 100, "max": 2000,
                    "label": "Min Dimension Threshold", 
                    "help": "Minimum image dimension before upscaling (pixels)"
                },
                "upscale_factor": {
                    "type": "number", "default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1,
                    "label": "Upscale Factor",
                    "help": "Factor to upscale small images"
                }
            },
            
            "Text Detection": {
                "bin_thresh": {
                    "type": "slider", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Binary Threshold",
                    "help": "Binarization threshold for text detection"
                },
                "box_thresh": {
                    "type": "slider", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Box Threshold",
                    "help": "Confidence threshold for text box detection"
                }
            },
            
            "Shape Detection": {
                "fill_circle_enclosing_threshold": {
                    "type": "slider", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Circle Fill Threshold",
                    "help": "Threshold for filling detected circles"
                },
                "fill_rect_enclosing_threshold": {
                    "type": "slider", "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Rectangle Fill Threshold", 
                    "help": "Threshold for filling detected rectangles"
                },
                "erosion_kernel_size": {
                    "type": "text", "default": "[3, 3]",
                    "label": "Erosion Kernel Size",
                    "help": "Kernel size for erosion [width, height]"
                },
                "min_stable_length": {
                    "type": "number", "default": 3, "min": 1, "max": 10,
                    "label": "Min Stable Length",
                    "help": "Minimum iterations for shape stability"
                },
                "max_erosion_iterations": {
                    "type": "number", "default": 30, "min": 5, "max": 100,
                    "label": "Max Erosion Iterations",
                    "help": "Maximum erosion iterations"
                },
                "classify_circle_overlap_threshold": {
                    "type": "slider", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Circle Classification Threshold",
                    "help": "Overlap threshold for circle classification"
                },
                "classify_rect_overlap_threshold": {
                    "type": "slider", "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Rectangle Classification Threshold",
                    "help": "Overlap threshold for rectangle classification"
                },
                "remove_nodes_dilation_kernel_size": {
                    "type": "text", "default": "[3, 3]",
                    "label": "Dilation Kernel Size",
                    "help": "Kernel size for node removal [width, height]"
                },
                "remove_nodes_dilation_iterations": {
                    "type": "number", "default": 3, "min": 1, "max": 10,
                    "label": "Dilation Iterations",
                    "help": "Number of dilation iterations for node removal"
                }
            },
            
            "Connection Processing - Hough Transform": {
                "hough_rho": {
                    "type": "number", "default": 1, "min": 1, "max": 5,
                    "label": "Rho (distance resolution)",
                    "help": "Distance resolution in pixels"
                },
                "hough_theta_degrees": {
                    "type": "number", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "label": "Theta (angle resolution)",
                    "help": "Angle resolution in degrees"
                },
                "hough_threshold": {
                    "type": "number", "default": 15, "min": 5, "max": 50,
                    "label": "Threshold",
                    "help": "Accumulator threshold for line detection"
                },
                "hough_min_line_length": {
                    "type": "number", "default": 10, "min": 1, "max": 50,
                    "label": "Min Line Length",
                    "help": "Minimum length of line segments"
                },
                "hough_max_line_gap": {
                    "type": "number", "default": 25, "min": 1, "max": 100,
                    "label": "Max Line Gap",
                    "help": "Maximum gap to merge line segments"
                }
            },
            
            "Connection Processing - Advanced": {
                "hough_bundler_min_distance": {
                    "type": "number", "default": 10.0, "min": 1.0, "max": 50.0, "step": 0.5,
                    "label": "Bundler Min Distance",
                    "help": "Minimum distance for line bundling"
                },
                "hough_bundler_min_angle": {
                    "type": "number", "default": 3.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "label": "Bundler Min Angle",
                    "help": "Minimum angle for line bundling"
                },
                "arrowhead_confidence_threshold_percent": {
                    "type": "number", "default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "label": "Arrowhead Confidence (%)",
                    "help": "Confidence threshold for arrowhead detection"
                },
                "proximity_thres_place": {
                    "type": "number", "default": 1.5, "min": 0.5, "max": 5.0, "step": 0.1,
                    "label": "Place Proximity Threshold",
                    "help": "Proximity threshold for places (×radius)"
                },
                "proximity_thres_trans_height": {
                    "type": "number", "default": 1.4, "min": 0.5, "max": 5.0, "step": 0.1,
                    "label": "Transition Height Threshold",
                    "help": "Height proximity threshold for transitions"
                },
                "proximity_thres_trans_width": {
                    "type": "number", "default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1,
                    "label": "Transition Width Threshold",
                    "help": "Width proximity threshold for transitions"
                },
                "arrowhead_proximity_threshold": {
                    "type": "number", "default": 40, "min": 10, "max": 100,
                    "label": "Arrowhead Proximity",
                    "help": "Distance threshold for arrowhead linking"
                },
                "text_linking_threshold": {
                    "type": "number", "default": 25.0, "min": 5.0, "max": 100.0, "step": 1.0,
                    "label": "Text Linking Threshold",
                    "help": "Distance threshold for text linking"
                }
            },
            
            "Path Finding": {
                "proximity_threshold": {
                    "type": "number", "default": 30.0, "min": 5.0, "max": 100.0, "step": 1.0,
                    "label": "Proximity Threshold",
                    "help": "Distance threshold for path segment connection"
                },
                "dot_product_weight": {
                    "type": "slider", "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Dot Product Weight",
                    "help": "Weight for direction similarity in scoring"
                },
                "distance_to_line_weight": {
                    "type": "slider", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Distance to Line Weight",
                    "help": "Weight for distance to line in scoring"
                },
                "endpoint_distance_weight": {
                    "type": "slider", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "label": "Endpoint Distance Weight", 
                    "help": "Weight for endpoint distance in scoring"
                }
            }
        }
        
        self.components = {}
        
    def create_component(self, param_name: str, param_def: Dict) -> Any:
        """Create appropriate Gradio component for parameter"""
        if param_def["type"] == "slider":
            return gr.Slider(
                minimum=param_def["min"],
                maximum=param_def["max"], 
                step=param_def["step"],
                value=param_def["default"],
                label=param_def["label"],
                info=param_def["help"]
            )
        elif param_def["type"] == "number":
            return gr.Number(
                minimum=param_def.get("min"),
                maximum=param_def.get("max"),
                step=param_def.get("step", 1),
                value=param_def["default"],
                label=param_def["label"],
                info=param_def["help"]
            )
        elif param_def["type"] == "text":
            return gr.Textbox(
                value=param_def["default"],
                label=param_def["label"],
                info=param_def["help"]
            )
    
    def validate_parameter(self, param_name: str, value: Any) -> tuple[Any, str]:
        """Validate parameter value, return (validated_value, error_message)"""
        # Find parameter definition
        param_def = None
        for group_params in self.param_groups.values():
            if param_name in group_params:
                param_def = group_params[param_name]
                break
        
        if not param_def:
            return value, f"Unknown parameter: {param_name}"
        
        try:
            # Handle list parameters (kernel sizes)
            if param_def["type"] == "text" and "kernel" in param_name.lower():
                if isinstance(value, str):
                    # Parse string representation of list
                    parsed = eval(value.strip())
                    if not isinstance(parsed, list) or len(parsed) != 2:
                        return None, f"{param_def['label']}: Must be a list of 2 numbers [w, h]"
                    if not all(isinstance(x, (int, float)) and x > 0 for x in parsed):
                        return None, f"{param_def['label']}: Values must be positive numbers"
                    return parsed, ""
                return value, ""
            
            # Validate numeric ranges
            if param_def["type"] in ["number", "slider"]:
                if not isinstance(value, (int, float)):
                    return None, f"{param_def['label']}: Must be a number"
                
                min_val = param_def.get("min")
                max_val = param_def.get("max")
                
                if min_val is not None and value < min_val:
                    return None, f"{param_def['label']}: Must be ≥ {min_val}"
                if max_val is not None and value > max_val:
                    return None, f"{param_def['label']}: Must be ≤ {max_val}"
            
            return value, ""
            
        except Exception as e:
            return None, f"{param_def['label']}: {str(e)}"
    
    def load_config_file(self, file_obj):
        """Load configuration from uploaded file"""
        if file_obj is None:
            return self._get_current_values() + ("No file selected",)
        
        try:
            with open(file_obj.name, 'r') as f:
                self.config_data = yaml.safe_load(f)
            
            # Update component values
            values = []
            for group_name, group_params in self.param_groups.items():
                for param_name, param_def in group_params.items():
                    value = self._get_config_value(param_name, param_def["default"])
                    values.append(value)
            
            return tuple(values) + (f"Loaded: {os.path.basename(file_obj.name)}",)
            
        except Exception as e:
            return self._get_current_values() + (f"Error loading file: {str(e)}",)
    
    def _get_config_value(self, param_name: str, default: Any) -> Any:
        """Get parameter value from loaded config with proper path handling"""
        # Handle path finding parameters specially
        if param_name in ["proximity_threshold", "dot_product_weight", "distance_to_line_weight", "endpoint_distance_weight"]:
            return self.config_data.get("connection_processing", {}).get("path_finding", {}).get(param_name, default)
        
        # Handle other connection_processing parameters
        elif param_name.startswith(("hough_", "arrowhead_", "proximity_", "text_")):
            return self.config_data.get("connection_processing", {}).get(param_name, default)
        
        # Handle shape detection parameters
        elif param_name in ["fill_circle_enclosing_threshold", "fill_rect_enclosing_threshold", "erosion_kernel_size", 
                           "min_stable_length", "max_erosion_iterations", "classify_circle_overlap_threshold",
                           "classify_rect_overlap_threshold", "remove_nodes_dilation_kernel_size", "remove_nodes_dilation_iterations"]:
            value = self.config_data.get("shape_detection", {}).get(param_name, default)
            # Format lists as strings for text inputs
            if isinstance(value, list):
                return str(value)
            return value
        
        # Handle text detection parameters
        elif param_name in ["bin_thresh", "box_thresh"]:
            return self.config_data.get("text_detection", {}).get(param_name, default)
        
        # Handle image processing parameters
        elif param_name in ["min_dimension_threshold", "upscale_factor"]:
            return self.config_data.get("image_processing", {}).get(param_name, default)
        
        return default
    
    def _get_current_values(self) -> tuple:
        """Get current values from all components"""
        values = []
        for group_params in self.param_groups.values():
            for param_name, param_def in group_params.items():
                values.append(param_def["default"])
        return tuple(values)
    
    def save_config(self, *param_values) -> tuple:
        """Save configuration to temporary file for download"""
        try:
            # Validate all parameters
            config = {
                "image_processing": {},
                "text_detection": {},
                "shape_detection": {},
                "connection_processing": {"path_finding": {}}
            }
            
            errors = []
            value_index = 0
            
            for group_name, group_params in self.param_groups.items():
                for param_name, param_def in group_params.items():
                    value = param_values[value_index]
                    validated_value, error = self.validate_parameter(param_name, value)
                    
                    if error:
                        errors.append(error)
                    else:
                        # Place value in correct config section
                        if param_name in ["proximity_threshold", "dot_product_weight", "distance_to_line_weight", "endpoint_distance_weight"]:
                            config["connection_processing"]["path_finding"][param_name] = validated_value
                        elif param_name.startswith(("hough_", "arrowhead_", "proximity_", "text_")):
                            config["connection_processing"][param_name] = validated_value
                        elif param_name in ["fill_circle_enclosing_threshold", "fill_rect_enclosing_threshold", "erosion_kernel_size", 
                                           "min_stable_length", "max_erosion_iterations", "classify_circle_overlap_threshold",
                                           "classify_rect_overlap_threshold", "remove_nodes_dilation_kernel_size", "remove_nodes_dilation_iterations"]:
                            config["shape_detection"][param_name] = validated_value
                        elif param_name in ["bin_thresh", "box_thresh"]:
                            config["text_detection"][param_name] = validated_value
                        elif param_name in ["min_dimension_threshold", "upscale_factor"]:
                            config["image_processing"][param_name] = validated_value
                    
                    value_index += 1
            
            if errors:
                return "Validation errors:\n" + "\n".join(errors), gr.update(visible=False)
            
            # Generate filename for display purposes

            
            # Use centralized path management
            ensure_directories_exist()
            temp_filepath = get_config_download_path()
            
            # Save file to temporary location in working directory (overwrites previous version)
            with open(temp_filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            return f"Configuration prepared for download: {temp_filepath.split('/')[-1]}", gr.update(value=temp_filepath, visible=True)
            
        except Exception as e:
            return f"Error saving configuration: {str(e)}", gr.update(visible=False)
    
    def create_interface(self) -> gr.TabItem:
        """Create the Configuration Editor interface"""
        with gr.TabItem("Simple Config Editor") as tab:
            gr.Markdown("## Algorithm Configuration Editor")
            gr.Markdown("Upload an existing config file or modify parameters below, then save.")
            
            # File upload
            with gr.Row():
                file_upload = gr.File(
                    label="Upload Configuration File (.yaml/.yml)",
                    file_types=[".yaml", ".yml"]
                )
            
            status = gr.Textbox(label="Status", interactive=False)
            
            # Create parameter sections
            components_list = []
            with gr.Accordion("Configuration Parameters", open=True):
                for group_name, group_params in self.param_groups.items():
                    with gr.Accordion(group_name, open=False):
                        for param_name, param_def in group_params.items():
                            component = self.create_component(param_name, param_def)
                            self.components[param_name] = component
                            components_list.append(component)
            
            # Save section
            with gr.Row():
                save_btn = gr.Button("Save Configuration", variant="primary")
            
            download_file = gr.File(label="Download", visible=False)
            
            # Event handlers
            file_upload.upload(
                fn=self.load_config_file,
                inputs=[file_upload],
                outputs=components_list + [status]
            )
            
            save_btn.click(
                fn=self.save_config,
                inputs=components_list,
                outputs=[status, download_file]
            )
            
        return tab 