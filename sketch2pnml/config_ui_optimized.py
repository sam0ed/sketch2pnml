import os
import yaml
import gradio as gr
from datetime import datetime
from typing import Dict, Any, List, Optional

def load_config(config_path: str) -> tuple:
    """Load configuration from yaml file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f), True, f"Successfully loaded {config_path}"
    except Exception as e:
        return {}, False, f"Error loading config: {str(e)}"

def save_config(config: Dict, config_path: str, custom_filename: Optional[str] = None) -> tuple:
    """Save configuration to yaml file with timestamp or custom name"""
    try:
        base_dir = os.path.dirname(config_path)
        
        if custom_filename:
            # Use custom filename if provided
            if not custom_filename.endswith(('.yaml', '.yml')):
                custom_filename += '.yaml'
            new_path = os.path.join(base_dir, custom_filename)
        else:
            # Otherwise use timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.splitext(config_path)[0]
            ext = os.path.splitext(config_path)[1]
            new_path = f"{base_path}_{timestamp}{ext}"
        
        with open(new_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        return True, f"Configuration saved to {new_path}", new_path
    except Exception as e:
        return False, f"Error saving config: {str(e)}", None

def get_config_files(directory: str) -> List[str]:
    """Get list of yaml files in directory"""
    try:
        files = [f for f in os.listdir(directory) if f.endswith(('.yaml', '.yml'))]
        return files
    except Exception:
        return []

def get_from_nested_dict(data: Dict, path: str, default: Any = None) -> Any:
    """Access nested dictionary using dot notation path"""
    keys = path.split('.')
    result = data
    
    for key in keys:
        if not isinstance(result, dict) or key not in result:
            return default
        result = result[key]
    
    return result

def set_in_nested_dict(data: Dict, path: str, value: Any) -> Dict:
    """Set value in nested dictionary using dot notation path"""
    keys = path.split('.')
    current = data
    
    # Navigate to the final dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    return data

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
    {"section": "Connection Processing", "subsection": "Arrowhead Detection", "name": "arrow_project_id", "path": "connection_processing.arrowhead_api.project_id", "type": "text", "label": "Roboflow Project ID", "default": ""},
    {"section": "Connection Processing", "subsection": "Arrowhead Detection", "name": "arrow_version", "path": "connection_processing.arrowhead_api.version", "type": "number", "label": "Roboflow Version", "default": 1, "precision": 0},
    {"section": "Connection Processing", "subsection": "Arrowhead Detection", "name": "arrow_api_key", "path": "connection_processing.arrowhead_api.api_key", "type": "password", "label": "Roboflow API Key", "default": ""},
    {"section": "Connection Processing", "subsection": "Arrowhead Detection", "name": "arrow_conf_threshold", "path": "connection_processing.arrowhead_api.confidence_threshold_percent", "type": "number", "label": "Confidence Threshold (%)", "default": 10.0, "precision": 1},
    
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

def create_component(param_def: Dict) -> Any:
    """Create appropriate Gradio component based on parameter definition"""
    comp_type = param_def.get("type", "text")
    
    if comp_type == "slider":
        return gr.Slider(
            label=param_def["label"],
            minimum=param_def.get("min", 0),
            maximum=param_def.get("max", 1),
            step=param_def.get("step", 0.1),
            value=param_def.get("default")
        )
    elif comp_type == "checkbox":
        return gr.Checkbox(label=param_def["label"], value=param_def.get("default", False))
    elif comp_type == "number":
        return gr.Number(label=param_def["label"], value=param_def.get("default", 0), precision=param_def.get("precision", 0))
    elif comp_type == "password":
        return gr.Textbox(label=param_def["label"], value=param_def.get("default", ""), type="password")
    else:  # text is the default
        return gr.Textbox(label=param_def["label"], value=param_def.get("default", ""))

def create_ui():
    # Set theme
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="indigo",
    ).set(
        body_background_fill="white",
        block_radius="10px",
        block_shadow="0px 5px 10px rgba(0, 0, 0, 0.1)",
    )
    
    with gr.Blocks(theme=theme, title="Petri Net Config Editor") as app:
        gr.Markdown("# Petri Net Configuration Editor")
        
        # Replace file selection with a single Browse button
        with gr.Row():
            # load_btn = gr.Button("Browse Configuration File") # Removed
            # file_info = gr.Textbox(label="Selected File", interactive=False) # Removed
            config_file_upload = gr.File(label="Upload Configuration File", file_types=['.yaml', '.yml'])
            file_info = gr.Textbox(label="Selected File Path", interactive=False) # To display the path of the uploaded file

        status = gr.Textbox(label="Status", interactive=False)
        
        # Hidden state
        config_data = gr.State({})
        loaded_path = gr.State("")
        
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
        components = {}
        
        with gr.Accordion("Parameter Configuration", open=True) as param_section:
            for section_name, section_data in sections.items():
                with gr.Accordion(section_name, open=False):
                    # Handle parameters directly in section
                    for i in range(0, len(section_data["params"]), 2):
                        with gr.Row():
                            param = section_data["params"][i]
                            components[param["name"]] = create_component(param)
                            
                            if i + 1 < len(section_data["params"]):
                                param = section_data["params"][i + 1]
                                components[param["name"]] = create_component(param)
                    
                    # Handle subsections
                    for subsec_name, subsec_params in section_data["subsections"].items():
                        with gr.Accordion(subsec_name, open=False):
                            for i in range(0, len(subsec_params), 2):
                                with gr.Row():
                                    param = subsec_params[i]
                                    components[param["name"]] = create_component(param)
                                    
                                    if i + 1 < len(subsec_params):
                                        param = subsec_params[i + 1]
                                        components[param["name"]] = create_component(param)
        
        # Save controls
        with gr.Row():
            custom_filename = gr.Textbox(
                label="Save as (optional)", 
                placeholder="Enter new filename (leave empty for auto timestamp)",
                interactive=True
            )
        
        save_btn = gr.Button("Save Configuration")
        download_file = gr.File(label="Download Configuration", visible=False, interactive=True, type="filepath")
        success_indicator = gr.Markdown("")
        
        # ===== Event Handlers =====
        def browse_and_load(file_obj):
            """Load selected configuration file from upload"""
            if file_obj is None:
                return {}, "", "No file selected", "", *[components[param["name"]].value for param in PARAMS]

            file_path = file_obj.name
            try:
                # Load the configuration
                config, success, message = load_config(file_path)
                
                if not success:
                    # Keep current values if loading fails, but update status and path
                    current_values = [components[param["name"]].value for param in PARAMS]
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
                current_values = [components[param["name"]].value for param in PARAMS]
                return {}, "", f"Error loading file: {str(e)}", file_path if file_path else "", *current_values
        
        # Wire up events
        config_file_upload.upload(
            fn=browse_and_load,
            inputs=[config_file_upload],
            outputs=[config_data, loaded_path, status, file_info] + list(components.values())
        )
        
        def handle_save_positional(config_data, config_path, custom_filename, *param_values):
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
        
        # Create a list of inputs in the correct order
        save_inputs = [config_data, loaded_path, custom_filename] + list(components.values())
        
        save_btn.click(
            fn=handle_save_positional,
            inputs=save_inputs,
            outputs=[status, download_file]
        ).then(
            fn=lambda msg: "✅ Success!" if "Error" not in msg else "❌ Error!",
            inputs=[status],
            outputs=[success_indicator]
        )
        
    return app

# Main script
if __name__ == "__main__":
    app = create_ui()
    app.launch() 