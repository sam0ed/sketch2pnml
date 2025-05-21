import os
import yaml
import gradio as gr
import shutil
from datetime import datetime
from pathlib import Path

def load_config(config_path):
    """Load configuration from yaml file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f), True, f"Successfully loaded {config_path}"
    except Exception as e:
        return {}, False, f"Error loading config: {str(e)}"

def save_config(config, config_path, custom_filename=None):
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
            
        return True, f"Configuration saved to {new_path}"
    except Exception as e:
        return False, f"Error saving config: {str(e)}"

def get_config_files(directory):
    """Get list of yaml files in directory"""
    try:
        files = [f for f in os.listdir(directory) if f.endswith(('.yaml', '.yml'))]
        return files
    except Exception:
        return []

def update_section(config, section, values):
    """Update a configuration section with new values"""
    if section not in config:
        config[section] = {}
    
    for key, value in values.items():
        # Handle nested sections (like connection_processing.path_finding)
        if '.' in key:
            parts = key.split('.')
            sub_section = parts[0]
            sub_key = parts[1]
            
            if sub_section not in config[section]:
                config[section][sub_section] = {}
            
            config[section][sub_section][sub_key] = value
        else:
            config[section][key] = value
    
    return config

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
        
        with gr.Row():
            config_dir = gr.Textbox(
                label="Configuration Directory", 
                placeholder="Enter path to configuration directory",
                value="sketch2pnml/data/",
                interactive=True
            )
            refresh_btn = gr.Button("Refresh Files")
        
        with gr.Row():
            config_files = gr.Dropdown(
                label="Select Config File", 
                choices=get_config_files("sketch2pnml/data/"),
                interactive=True
            )
            load_btn = gr.Button("Load Configuration")
        
        status = gr.Textbox(label="Status", interactive=False)
        
        # Config data (hidden)
        config_data = gr.State({})
        loaded_path = gr.State("")
        
        with gr.Accordion("Parameter Configuration", open=True) as param_section:
            # Image Processing Section
            with gr.Accordion("Image Processing", open=False):
                with gr.Row():
                    img_min_dim = gr.Number(label="Minimum Dimension Threshold", precision=0)
                    img_upscale = gr.Number(label="Upscale Factor", precision=1)
            
            # Text Detection Section
            with gr.Accordion("Text Detection", open=False):
                with gr.Row():
                    text_bin_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Binary Threshold", step=0.05)
                    text_box_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Box Threshold", step=0.05)
            
            # Text Removal Section
            with gr.Accordion("Text Removal", open=False):
                with gr.Row():
                    text_doctr_score = gr.Slider(minimum=0.0, maximum=1.0, label="Doctr Score Threshold", step=0.05)
                    text_overlap_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Contour Overlap Threshold", step=0.05)
            
            # Shape Detection Section
            with gr.Accordion("Shape Detection", open=False):
                with gr.Row():
                    shape_verbose = gr.Checkbox(label="Verbose")
                with gr.Row():
                    fill_circle_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Fill Circle Enclosing Threshold", step=0.05)
                    fill_rect_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Fill Rectangle Enclosing Threshold", step=0.05)
                with gr.Row():
                    erosion_kernel_size = gr.Textbox(label="Erosion Kernel Size (format: [w, h])")
                    min_stable_length = gr.Number(label="Minimum Stable Length", precision=0)
                    max_erosion_iter = gr.Number(label="Maximum Erosion Iterations", precision=0)
                with gr.Row():
                    classify_circle_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Circle Classification Threshold", step=0.05)
                    classify_rect_thresh = gr.Slider(minimum=0.0, maximum=1.0, label="Rectangle Classification Threshold", step=0.05)
                with gr.Row():
                    remove_nodes_kernel = gr.Textbox(label="Remove Nodes Dilation Kernel Size (format: [w, h])")
                    remove_nodes_iter = gr.Number(label="Remove Nodes Dilation Iterations", precision=0)
            
            # Connection Processing Section
            with gr.Accordion("Connection Processing", open=False):
                # Hough Line Transform
                with gr.Accordion("Hough Line Transform", open=False):
                    with gr.Row():
                        hough_rho = gr.Number(label="Rho", precision=0)
                        hough_theta = gr.Number(label="Theta (degrees)", precision=1)
                    with gr.Row():
                        hough_threshold = gr.Number(label="Threshold", precision=0)
                        hough_min_line_len = gr.Number(label="Minimum Line Length", precision=0)
                        hough_max_line_gap = gr.Number(label="Maximum Line Gap", precision=0)
                
                # Hough Bundler
                with gr.Row():
                    hough_bundler_min_dist = gr.Number(label="Bundler Min Distance", precision=1)
                    hough_bundler_min_angle = gr.Number(label="Bundler Min Angle", precision=1)
                
                # Arrowhead Detection
                with gr.Accordion("Arrowhead Detection", open=False):
                    with gr.Row():
                        arrow_project_id = gr.Textbox(label="Roboflow Project ID")
                        arrow_version = gr.Number(label="Roboflow Version", precision=0)
                    with gr.Row():
                        arrow_api_key = gr.Textbox(label="Roboflow API Key", type="password")
                        arrow_conf_threshold = gr.Number(label="Confidence Threshold (%)", precision=1)
                
                # Linking Nodes to Lines
                with gr.Row():
                    proximity_thres_place = gr.Number(label="Place Proximity Threshold", precision=1)
                    proximity_thres_trans_height = gr.Number(label="Transition Height Proximity Threshold", precision=1)
                    proximity_thres_trans_width = gr.Number(label="Transition Width Proximity Threshold", precision=1)
                
                # Path Finding
                with gr.Accordion("Path Finding", open=False):
                    with gr.Row():
                        path_proximity_threshold = gr.Number(label="Proximity Threshold", precision=1)
                        path_dot_product_weight = gr.Number(label="Dot Product Weight", precision=1)
                        path_distance_line_weight = gr.Number(label="Distance to Line Weight", precision=1)
                        path_endpoint_distance_weight = gr.Number(label="Endpoint Distance Weight", precision=1)
                
                # Linking Arrowheads
                with gr.Row():
                    arrowhead_proximity_threshold = gr.Number(label="Arrowhead Proximity Threshold", precision=1)
                    text_linking_threshold = gr.Number(label="Text Linking Threshold", precision=1)
        
        # Add filename field before save button
        with gr.Row():
            custom_filename = gr.Textbox(
                label="Save as (optional)", 
                placeholder="Enter new filename (leave empty for auto timestamp)",
                interactive=True
            )
        
        save_btn = gr.Button("Save Configuration")
        success_indicator = gr.Markdown("")
        
        # Set up events
        def update_file_list(directory):
            return gr.Dropdown(choices=get_config_files(directory))
        
        refresh_btn.click(
            fn=update_file_list,
            inputs=[config_dir],
            outputs=[config_files]
        )
        
        def load_configuration(directory, filename):
            if not filename:
                return {}, "", "Please select a configuration file", {}
            
            path = os.path.join(directory, filename)
            config, success, message = load_config(path)
            
            if not success:
                return {}, "", message, {}
            
            # Prepare values for UI
            values = {
                # Image Processing
                "img_min_dim": config.get("image_processing", {}).get("min_dimension_threshold", 800),
                "img_upscale": config.get("image_processing", {}).get("upscale_factor", 2),
                
                # Text Detection
                "text_bin_thresh": config.get("text_detection", {}).get("bin_thresh", 0.3),
                "text_box_thresh": config.get("text_detection", {}).get("box_thresh", 0.1),
                
                # Text Removal
                "text_doctr_score": config.get("text_removal", {}).get("doctr_score_thresh", 0.7),
                "text_overlap_thresh": config.get("text_removal", {}).get("remove_contour_overlap_thresh", 0.5),
                
                # Shape Detection
                "shape_verbose": config.get("shape_detection", {}).get("verbose", True),
                "fill_circle_thresh": config.get("shape_detection", {}).get("fill_circle_enclosing_threshold", 0.8),
                "fill_rect_thresh": config.get("shape_detection", {}).get("fill_rect_enclosing_threshold", 0.95),
                "erosion_kernel_size": str(config.get("shape_detection", {}).get("erosion_kernel_size", [3, 3])),
                "min_stable_length": config.get("shape_detection", {}).get("min_stable_length", 3),
                "max_erosion_iter": config.get("shape_detection", {}).get("max_erosion_iterations", 30),
                "classify_circle_thresh": config.get("shape_detection", {}).get("classify_circle_overlap_threshold", 0.8),
                "classify_rect_thresh": config.get("shape_detection", {}).get("classify_rect_overlap_threshold", 0.85),
                "remove_nodes_kernel": str(config.get("shape_detection", {}).get("remove_nodes_dilation_kernel_size", [3, 3])),
                "remove_nodes_iter": config.get("shape_detection", {}).get("remove_nodes_dilation_iterations", 3),
                
                # Connection Processing
                "hough_rho": config.get("connection_processing", {}).get("hough_rho", 1),
                "hough_theta": config.get("connection_processing", {}).get("hough_theta_degrees", 1),
                "hough_threshold": config.get("connection_processing", {}).get("hough_threshold", 15),
                "hough_min_line_len": config.get("connection_processing", {}).get("hough_min_line_length", 10),
                "hough_max_line_gap": config.get("connection_processing", {}).get("hough_max_line_gap", 25),
                "hough_bundler_min_dist": config.get("connection_processing", {}).get("hough_bundler_min_distance", 10),
                "hough_bundler_min_angle": config.get("connection_processing", {}).get("hough_bundler_min_angle", 5),
                
                # Arrowhead Detection
                "arrow_project_id": config.get("connection_processing", {}).get("arrowhead_api", {}).get("project_id", ""),
                "arrow_version": config.get("connection_processing", {}).get("arrowhead_api", {}).get("version", 1),
                "arrow_api_key": config.get("connection_processing", {}).get("arrowhead_api", {}).get("api_key", ""),
                "arrow_conf_threshold": config.get("connection_processing", {}).get("arrowhead_api", {}).get("confidence_threshold_percent", 10.0),
                
                # Linking Nodes
                "proximity_thres_place": config.get("connection_processing", {}).get("proximity_thres_place", 1.5),
                "proximity_thres_trans_height": config.get("connection_processing", {}).get("proximity_thres_trans_height", 1.4),
                "proximity_thres_trans_width": config.get("connection_processing", {}).get("proximity_thres_trans_width", 3),
                
                # Path Finding
                "path_proximity_threshold": config.get("connection_processing", {}).get("path_finding", {}).get("proximity_threshold", 30.0),
                "path_dot_product_weight": config.get("connection_processing", {}).get("path_finding", {}).get("dot_product_weight", 0.6),
                "path_distance_line_weight": config.get("connection_processing", {}).get("path_finding", {}).get("distance_to_line_weight", 0.2),
                "path_endpoint_distance_weight": config.get("connection_processing", {}).get("path_finding", {}).get("endpoint_distance_weight", 0.2),
                
                # Linking Arrowheads
                "arrowhead_proximity_threshold": config.get("connection_processing", {}).get("arrowhead_proximity_threshold", 40),
                "text_linking_threshold": config.get("connection_processing", {}).get("text_linking_threshold", 25.0),
            }
            
            # Return values directly for UI updates
            return (
                config, path, message, 
                values.get("img_min_dim", 800),
                values.get("img_upscale", 2),
                values.get("text_bin_thresh", 0.3),
                values.get("text_box_thresh", 0.1),
                values.get("text_doctr_score", 0.7),
                values.get("text_overlap_thresh", 0.5),
                values.get("shape_verbose", True),
                values.get("fill_circle_thresh", 0.8),
                values.get("fill_rect_thresh", 0.95),
                values.get("erosion_kernel_size", "[3, 3]"),
                values.get("min_stable_length", 3),
                values.get("max_erosion_iter", 30),
                values.get("classify_circle_thresh", 0.8),
                values.get("classify_rect_thresh", 0.85),
                values.get("remove_nodes_kernel", "[3, 3]"),
                values.get("remove_nodes_iter", 3),
                values.get("hough_rho", 1),
                values.get("hough_theta", 1),
                values.get("hough_threshold", 15),
                values.get("hough_min_line_len", 10),
                values.get("hough_max_line_gap", 25),
                values.get("hough_bundler_min_dist", 10),
                values.get("hough_bundler_min_angle", 5),
                values.get("arrow_project_id", ""),
                values.get("arrow_version", 1),
                values.get("arrow_api_key", ""),
                values.get("arrow_conf_threshold", 10.0),
                values.get("proximity_thres_place", 1.5),
                values.get("proximity_thres_trans_height", 1.4),
                values.get("proximity_thres_trans_width", 3),
                values.get("path_proximity_threshold", 30.0),
                values.get("path_dot_product_weight", 0.6),
                values.get("path_distance_line_weight", 0.2),
                values.get("path_endpoint_distance_weight", 0.2),
                values.get("arrowhead_proximity_threshold", 40),
                values.get("text_linking_threshold", 25.0)
            )
        
        load_btn.click(
            fn=load_configuration,
            inputs=[config_dir, config_files],
            outputs=[
                config_data, loaded_path, status,
                img_min_dim, img_upscale,
                text_bin_thresh, text_box_thresh,
                text_doctr_score, text_overlap_thresh,
                shape_verbose, fill_circle_thresh, fill_rect_thresh,
                erosion_kernel_size, min_stable_length, max_erosion_iter,
                classify_circle_thresh, classify_rect_thresh,
                remove_nodes_kernel, remove_nodes_iter,
                hough_rho, hough_theta, hough_threshold, 
                hough_min_line_len, hough_max_line_gap,
                hough_bundler_min_dist, hough_bundler_min_angle,
                arrow_project_id, arrow_version, arrow_api_key, arrow_conf_threshold,
                proximity_thres_place, proximity_thres_trans_height, proximity_thres_trans_width,
                path_proximity_threshold, path_dot_product_weight, 
                path_distance_line_weight, path_endpoint_distance_weight,
                arrowhead_proximity_threshold, text_linking_threshold
            ]
        )
        
        def save_configuration(
            config_data, config_path, custom_filename,
            img_min_dim, img_upscale,
            text_bin_thresh, text_box_thresh,
            text_doctr_score, text_overlap_thresh,
            shape_verbose, fill_circle_thresh, fill_rect_thresh,
            erosion_kernel_size, min_stable_length, max_erosion_iter,
            classify_circle_thresh, classify_rect_thresh,
            remove_nodes_kernel, remove_nodes_iter,
            hough_rho, hough_theta, hough_threshold, 
            hough_min_line_len, hough_max_line_gap,
            hough_bundler_min_dist, hough_bundler_min_angle,
            arrow_project_id, arrow_version, arrow_api_key, arrow_conf_threshold,
            proximity_thres_place, proximity_thres_trans_height, proximity_thres_trans_width,
            path_proximity_threshold, path_dot_product_weight, 
            path_distance_line_weight, path_endpoint_distance_weight,
            arrowhead_proximity_threshold, text_linking_threshold
        ):
            if not config_path:
                return "No configuration file loaded. Please load a configuration first."
            
            try:
                # Convert string lists to actual lists
                try:
                    erosion_kernel = eval(erosion_kernel_size)
                    remove_kernel = eval(remove_nodes_kernel)
                except:
                    erosion_kernel = [3, 3]
                    remove_kernel = [3, 3]
                
                # Update image processing
                config_data = update_section(config_data, "image_processing", {
                    "min_dimension_threshold": int(img_min_dim),
                    "upscale_factor": float(img_upscale)
                })
                
                # Update text detection
                config_data = update_section(config_data, "text_detection", {
                    "bin_thresh": float(text_bin_thresh),
                    "box_thresh": float(text_box_thresh)
                })
                
                # Update text removal
                config_data = update_section(config_data, "text_removal", {
                    "doctr_score_thresh": float(text_doctr_score),
                    "remove_contour_overlap_thresh": float(text_overlap_thresh)
                })
                
                # Update shape detection
                config_data = update_section(config_data, "shape_detection", {
                    "verbose": bool(shape_verbose),
                    "fill_circle_enclosing_threshold": float(fill_circle_thresh),
                    "fill_rect_enclosing_threshold": float(fill_rect_thresh),
                    "erosion_kernel_size": erosion_kernel,
                    "min_stable_length": int(min_stable_length),
                    "max_erosion_iterations": int(max_erosion_iter),
                    "classify_circle_overlap_threshold": float(classify_circle_thresh),
                    "classify_rect_overlap_threshold": float(classify_rect_thresh),
                    "remove_nodes_dilation_kernel_size": remove_kernel,
                    "remove_nodes_dilation_iterations": int(remove_nodes_iter)
                })
                
                # Update connection processing
                # Main section
                config_data = update_section(config_data, "connection_processing", {
                    "hough_rho": int(hough_rho),
                    "hough_theta_degrees": float(hough_theta),
                    "hough_threshold": int(hough_threshold),
                    "hough_min_line_length": int(hough_min_line_len),
                    "hough_max_line_gap": int(hough_max_line_gap),
                    "hough_bundler_min_distance": float(hough_bundler_min_dist),
                    "hough_bundler_min_angle": float(hough_bundler_min_angle),
                    "proximity_thres_place": float(proximity_thres_place),
                    "proximity_thres_trans_height": float(proximity_thres_trans_height),
                    "proximity_thres_trans_width": float(proximity_thres_trans_width),
                    "arrowhead_proximity_threshold": float(arrowhead_proximity_threshold),
                    "text_linking_threshold": float(text_linking_threshold)
                })
                
                # Arrowhead API subsection
                if "arrowhead_api" not in config_data["connection_processing"]:
                    config_data["connection_processing"]["arrowhead_api"] = {}
                
                config_data["connection_processing"]["arrowhead_api"] = {
                    "project_id": arrow_project_id,
                    "version": int(arrow_version),
                    "api_key": arrow_api_key,
                    "confidence_threshold_percent": float(arrow_conf_threshold)
                }
                
                # Path finding subsection
                if "path_finding" not in config_data["connection_processing"]:
                    config_data["connection_processing"]["path_finding"] = {}
                    
                config_data["connection_processing"]["path_finding"] = {
                    "proximity_threshold": float(path_proximity_threshold),
                    "dot_product_weight": float(path_dot_product_weight),
                    "distance_to_line_weight": float(path_distance_line_weight),
                    "endpoint_distance_weight": float(path_endpoint_distance_weight)
                }
                
                # Save configuration with optional custom filename
                success, message = save_config(config_data, config_path, custom_filename)
                return message if success else f"Error: {message}"
                
            except Exception as e:
                return f"Error updating configuration: {str(e)}"
        
        save_btn.click(
            fn=save_configuration,
            inputs=[
                config_data, loaded_path, custom_filename,
                img_min_dim, img_upscale,
                text_bin_thresh, text_box_thresh,
                text_doctr_score, text_overlap_thresh,
                shape_verbose, fill_circle_thresh, fill_rect_thresh,
                erosion_kernel_size, min_stable_length, max_erosion_iter,
                classify_circle_thresh, classify_rect_thresh,
                remove_nodes_kernel, remove_nodes_iter,
                hough_rho, hough_theta, hough_threshold, 
                hough_min_line_len, hough_max_line_gap,
                hough_bundler_min_dist, hough_bundler_min_angle,
                arrow_project_id, arrow_version, arrow_api_key, arrow_conf_threshold,
                proximity_thres_place, proximity_thres_trans_height, proximity_thres_trans_width,
                path_proximity_threshold, path_dot_product_weight, 
                path_distance_line_weight, path_endpoint_distance_weight,
                arrowhead_proximity_threshold, text_linking_threshold
            ],
            outputs=[status]
        ).then(
            # Update success indicator based on status message
            fn=lambda msg: "✅ Success!" if "Error" not in msg else "❌ Error!",
            inputs=[status],
            outputs=[success_indicator]
        )
        
    return app

# Main script
if __name__ == "__main__":
    app = create_ui()
    app.launch() 