import os
import sys
import gradio as gr
import yaml
import base64
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure we can import from the current directory structure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all necessary modules from the three applications
from endpoints.converter import fix_petri_net, render_diagram_to, render_to_graphviz, render_to_json, run_and_save_pipeline
from config.path_config import (
    DEMOS_DIR as DEMO_IMAGES_DIR, VISUALIZATIONS_DIR, 
    SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_CONFIG_EXTENSIONS, 
    ensure_directories_exist, get_working_image_path, get_working_config_path, 
    get_output_file_path, OUTPUT_PNML_PATH, OUTPUT_PETRIOBJ_PATH, 
    OUTPUT_JSON_PATH, OUTPUT_PNG_PATH, OUTPUT_GV_PATH
)

#############################################################
# Constants and utility functions from each application
#############################################################

# Ensure directories exist
ensure_directories_exist()

# Get list of demo images (with fallback if empty)
try:
    demo_images = [os.path.join(DEMO_IMAGES_DIR, f) for f in os.listdir(DEMO_IMAGES_DIR) 
                 if f.endswith(SUPPORTED_IMAGE_EXTENSIONS)]
    if not demo_images:
        print("No demo images found in the demos directory. Please add some images to", DEMO_IMAGES_DIR)
        demo_images = []
except Exception as e:
    print(f"Error loading demo images: {e}")
    demo_images = []

#############################################################
# Utility functions from config_ui_optimized.py
#############################################################

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
            if not custom_filename.endswith(SUPPORTED_CONFIG_EXTENSIONS):
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

#############################################################
# Utility functions from app.py
#############################################################

def save_as_working_image(image_path):
    """Save the selected image as the working image"""
    # Ensure the extension is preserved when copying
    _, ext = os.path.splitext(image_path)
    working_image_path = get_working_image_path(ext)
    
    shutil.copy(image_path, working_image_path)
    return working_image_path

def save_as_working_yaml(yaml_path):
    """Save the uploaded YAML as the working configuration"""
    _, ext = os.path.splitext(yaml_path)
    working_yaml_path = get_working_config_path(ext)
    
    shutil.copy(yaml_path, working_yaml_path)
    return working_yaml_path

def clear_visualizations():
    """Clear the visualizations directory"""
    # Create the directory if it doesn't exist
    if not os.path.exists(VISUALIZATIONS_DIR):
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Remove existing files
    for file in os.listdir(VISUALIZATIONS_DIR):
        file_path = os.path.join(VISUALIZATIONS_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def process_files(image_path, yaml_path):
    """Process the uploaded files and run the pipeline"""
    if image_path is None or yaml_path is None:
        return None, "Both image and YAML configuration file are required."
    
    # Clear previous visualization results
    clear_visualizations()
    
    # Run the pipeline
    try:
        run_and_save_pipeline(config_path=yaml_path, image_path=image_path)
        
        # Get visualization results
        vis_files = [os.path.join(VISUALIZATIONS_DIR, f) for f in os.listdir(VISUALIZATIONS_DIR) if f.endswith('.png')]
        
        if not vis_files:
            return None, "No visualization results were generated."
        
        return vis_files, "Processing completed successfully!"
    except Exception as e:
        return None, f"Error during processing: {str(e)}"

def select_demo_image(evt: gr.SelectData, current_yaml=None, working_yaml=None):
    """Handle selection of a demo image"""
    image_path = demo_images[evt.index]
    if image_path:
        working_image_path = save_as_working_image(image_path)
        status = "Demo image selected."
        if working_yaml:
            status += " Ready to process!"
        else:
            status += " Please upload a YAML configuration file."
        return image_path, working_image_path, current_yaml, working_yaml, status
    return None, None, current_yaml, working_yaml, ""

def upload_image(file_obj, current_yaml=None, working_yaml=None):
    """Handle upload of a custom image"""
    if file_obj is not None:
        # Save the uploaded file to working image
        _, ext = os.path.splitext(file_obj.name)
        working_image_path = get_working_image_path(ext)
        shutil.copy(file_obj.name, working_image_path)
        
        status = "Custom image uploaded."
        if working_yaml:
            status += " Ready to process!"
        else:
            status += " Please upload a YAML configuration file."
        return file_obj.name, working_image_path, current_yaml, working_yaml, status
    return None, None, current_yaml, working_yaml, ""

def upload_yaml(file_obj, current_image, working_image):
    """Handle upload of a YAML file"""
    if file_obj is not None:
        # Save the uploaded YAML as the working config
        yaml_path = file_obj.name
        working_yaml_path = save_as_working_yaml(yaml_path)
        
        if working_image is not None:
            return current_image, working_image, yaml_path, working_yaml_path, "Both files uploaded. Ready to process!"
        else:
            return None, None, yaml_path, working_yaml_path, "YAML file uploaded. Please select or upload an image."
    return current_image, working_image, None, None, ""

#############################################################
# Function from gradio_app.py
#############################################################

def process_and_display():
    """Run the converter pipeline and return results for display"""
    try:
        # Ensure output directories exist
        ensure_directories_exist()
        
        # Run the pipeline functions
        fix_petri_net()
        render_diagram_to("pnml")
        render_diagram_to("petriobj")
        render_to_graphviz()
        render_to_json()
        
        # Get the output files
        pnml_path = OUTPUT_PNML_PATH
        petriobj_path = OUTPUT_PETRIOBJ_PATH
        json_path = OUTPUT_JSON_PATH
        png_path = OUTPUT_PNG_PATH
        gv_path = OUTPUT_GV_PATH
        
        # Read file contents
        with open(pnml_path, "r", encoding="utf-8") as f:
            pnml_content = f.read()
        
        with open(petriobj_path, "r", encoding="utf-8") as f:
            petriobj_content = f.read()
        
        with open(json_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        
        with open(gv_path, "r", encoding="utf-8") as f:
            gv_content = f.read()
        
        # Return all results for the tabs
        return pnml_content, petriobj_content, json_content, png_path, gv_content, pnml_path, petriobj_path, json_path, gv_path, png_path
    except Exception as e:
        error_message = f"Error during processing: {str(e)}"
        print(error_message)
        empty_path = ""
        return f"Error: {error_message}", f"Error: {error_message}", f"Error: {error_message}", None, f"Error: {error_message}", empty_path, empty_path, empty_path, empty_path, empty_path

def create_download_file(content: str, filename: str, file_extension: str) -> str:
    """Create a temporary file with the given content for download"""
    try:
        # Ensure output directory exists
        ensure_directories_exist()
        
        # Create filename with proper extension
        if not filename.endswith(file_extension):
            filename = f"{filename}{file_extension}"
        
        file_path = get_output_file_path(filename)
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return file_path
    except Exception as e:
        print(f"Error creating download file: {e}")
        return ""

def download_pnml(content: str) -> str:
    """Create downloadable PNML file from current content"""
    return create_download_file(content, "edited_output.pnml", ".pnml")

def download_petriobj(content: str) -> str:
    """Create downloadable PetriObj file from current content"""
    return create_download_file(content, "edited_output.petriobj", ".petriobj")

def download_json(content: str) -> str:
    """Create downloadable JSON file from current content"""
    return create_download_file(content, "edited_output.json", ".json")

def download_gv(content: str) -> str:
    """Create downloadable GraphViz file from current content"""
    return create_download_file(content, "edited_output.gv", ".gv")

#############################################################
# Parameters for Configuration UI (from config_ui_optimized.py)
#############################################################

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

#############################################################
# Main Application
#############################################################

with gr.Blocks(title="Petri Net Converter Suite") as app:
    gr.Markdown("# Petri Net Converter Suite")
    
    with gr.Tabs() as tabs:
        #############################################################
        # App 1: Configuration UI (config_ui_optimized.py)
        #############################################################
        with gr.TabItem("Configuration Editor"):
            gr.Markdown("## Petri Net Configuration Editor")
            
            # File upload for configuration
            with gr.Row():
                config_file_upload = gr.File(label="Upload Configuration File", file_types=list(SUPPORTED_CONFIG_EXTENSIONS))
                file_info = gr.Textbox(label="Selected File Path", interactive=False)

            status = gr.Textbox(label="Status", interactive=False)
            
            # Hidden state for the config UI
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
            
            # Wire up events for config UI
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
            
        #############################################################
        # App 2: Image Processing App (app.py)
        #############################################################
        with gr.TabItem("Image Processor"):
            gr.Markdown("## Sketch to PNML Converter")
            
            # State variables for the app
            current_image = gr.State(None)  # Original image path
            working_image = gr.State(None)  # Working image path
            current_yaml = gr.State(None)   # Original YAML path
            working_yaml = gr.State(None)   # Working YAML path
            status_text = gr.State("")      # Status text
            
            # Demo section at the top
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Demo Images")
                    if demo_images:
                        demo_gallery = gr.Gallery(
                            value=demo_images, 
                            columns=5,
                            object_fit="contain", 
                            height="150px",
                            show_label=False
                        )
                    else:
                        gr.Markdown(f"*No demo images available. Place image files in the '{DEMO_IMAGES_DIR}' directory.*")
            
            # Input and output sections below
            with gr.Row():
                # Left column for inputs
                with gr.Column():
                    # Image upload section
                    gr.Markdown("### Or Upload Your Own Image")
                    image_upload = gr.File(
                        label="Upload Image", 
                        file_types=["image"],
                        file_count="single"
                    )
                    
                    # YAML config upload section
                    gr.Markdown("### Upload YAML Configuration")
                    yaml_upload = gr.File(
                        label="Upload YAML Config", 
                        file_types=list(SUPPORTED_CONFIG_EXTENSIONS),
                        file_count="single"
                    )
                    
                    # Process button
                    process_btn = gr.Button("Process Files", variant="primary", size="lg")
                
                # Right column for outputs
                with gr.Column():
                    gr.Markdown("### Output Visualizations")
                    output_gallery = gr.Gallery(
                        label="Processing Results", 
                        columns=1,
                        height=400,
                        show_label=True
                    )
                    output_message = gr.Markdown("")
            
            # Event handlers
            if demo_images:
                demo_gallery.select(
                    select_demo_image, 
                    inputs=[current_yaml, working_yaml],
                    outputs=[current_image, working_image, current_yaml, working_yaml, status_text]
                )
            
            image_upload.change(
                upload_image, 
                inputs=[image_upload, current_yaml, working_yaml], 
                outputs=[current_image, working_image, current_yaml, working_yaml, status_text]
            )
            
            yaml_upload.change(
                upload_yaml, 
                inputs=[yaml_upload, current_image, working_image], 
                outputs=[current_image, working_image, current_yaml, working_yaml, status_text]
            )
            
            process_btn.click(
                process_files, 
                inputs=[working_image, working_yaml], 
                outputs=[output_gallery, output_message]
            )
            
            # Update status message when it changes
            status_text.change(lambda x: x, inputs=[status_text], outputs=[output_message])

        #############################################################
        # App 3: Petri Net Converter (gradio_app.py)
        #############################################################
        with gr.TabItem("Petri Net Converter"):
            gr.Markdown("## Petri Net Sketch Converter")
            gr.Markdown("Press the Translate button to convert the working image to a Petri net")
            
            with gr.Row():
                translate_button = gr.Button("Translate", variant="primary", size="lg")
            
            # Hidden components to store file paths for downloads
            pnml_path_comp = gr.State("")
            petriobj_path_comp = gr.State("")
            json_path_comp = gr.State("")
            gv_path_comp = gr.State("")
            png_path_comp = gr.State("")
            
            with gr.Tabs():
                with gr.TabItem("PNML"):
                    pnml_output = gr.Code(label="PNML Output", lines=20, max_lines=25, interactive=True, language="html")
                    with gr.Row():
                        pnml_download_btn = gr.Button("Download PNML", variant="secondary")
                        pnml_download = gr.File(label="Download PNML File", visible=False, interactive=False)
                
                with gr.TabItem("PetriObj"):
                    petriobj_output = gr.Code(label="PetriObj Output", lines=20, max_lines=25, interactive=True, language="c")
                    with gr.Row():
                        petriobj_download_btn = gr.Button("Download PetriObj", variant="secondary")
                        petriobj_download = gr.File(label="Download PetriObj File", visible=False, interactive=False)
                
                with gr.TabItem("JSON"):
                    json_output = gr.Code(label="JSON Output", lines=20, max_lines=25, interactive=True, language="json")
                    with gr.Row():
                        json_download_btn = gr.Button("Download JSON", variant="secondary")
                        json_download = gr.File(label="Download JSON File", visible=False, interactive=False)
                
                with gr.TabItem("Visualization"):
                    image_output = gr.Image(label="Petri Net Visualization", type="filepath")
                    png_download = gr.File(label="Download PNG File", interactive=False)
                
                with gr.TabItem("GraphViz"):
                    gv_output = gr.Code(label="GraphViz Output", lines=20, max_lines=25, interactive=True, language="markdown")
                    with gr.Row():
                        gv_download_btn = gr.Button("Download GraphViz", variant="secondary")
                        gv_download = gr.File(label="Download GraphViz File", visible=False, interactive=False)
            
            # Connect the button to the processing function
            translate_button.click(
                fn=process_and_display,
                outputs=[
                    pnml_output, petriobj_output, json_output, image_output, gv_output,
                    pnml_path_comp, petriobj_path_comp, json_path_comp, gv_path_comp, png_path_comp
                ]
            ).then(
                lambda path: path,
                inputs=png_path_comp,
                outputs=png_download
            )
            
            # Connect download buttons to download functions
            pnml_download_btn.click(
                fn=download_pnml,
                inputs=[pnml_output],
                outputs=[pnml_download]
            ).then(
                lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
                inputs=[pnml_download],
                outputs=[pnml_download]
            )
            
            petriobj_download_btn.click(
                fn=download_petriobj,
                inputs=[petriobj_output],
                outputs=[petriobj_download]
            ).then(
                lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
                inputs=[petriobj_download],
                outputs=[petriobj_download]
            )
            
            json_download_btn.click(
                fn=download_json,
                inputs=[json_output],
                outputs=[json_download]
            ).then(
                lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
                inputs=[json_download],
                outputs=[json_download]
            )
            
            gv_download_btn.click(
                fn=download_gv,
                inputs=[gv_output],
                outputs=[gv_download]
            ).then(
                lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
                inputs=[gv_download],
                outputs=[gv_download]
            )

if __name__ == "__main__":
    app.launch() 