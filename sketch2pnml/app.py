import gradio as gr
import os
import shutil
from pathlib import Path
from endpoints.converter import run_and_save_pipeline, here

# Define paths
DATA_DIR = here("data")
DEMO_IMAGES_DIR = os.path.join(DATA_DIR, "demos")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
WORKING_IMAGE_PATH = os.path.join(DATA_DIR, "working_image.png")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(DEMO_IMAGES_DIR, exist_ok=True)  # Create demos directory if it doesn't exist

# Get list of demo images (with fallback if empty)
try:
    demo_images = [os.path.join(DEMO_IMAGES_DIR, f) for f in os.listdir(DEMO_IMAGES_DIR) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not demo_images:  # If demo directory exists but is empty
        print("No demo images found in the demos directory. Please add some images to", DEMO_IMAGES_DIR)
        # Use a placeholder message in the gallery
        demo_images = []
except Exception as e:
    print(f"Error loading demo images: {e}")
    demo_images = []

def save_as_working_image(image_path):
    """Save the selected image as the working image"""
    # Ensure the extension is preserved when copying
    _, ext = os.path.splitext(image_path)
    working_image_path = os.path.join(DATA_DIR, f"working_image{ext}")
    
    shutil.copy(image_path, working_image_path)
    return working_image_path

def save_as_working_yaml(yaml_path):
    """Save the uploaded YAML as the working configuration"""
    _, ext = os.path.splitext(yaml_path)
    working_yaml_path = os.path.join(DATA_DIR, f"working_config{ext}")
    
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
        working_image_path = os.path.join(DATA_DIR, f"working_image{ext}")
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

# Create the Gradio interface
with gr.Blocks(title="Sketch to PNML Converter", theme=gr.themes.Base(primary_hue="zinc")) as app:
    # State variables - define these BEFORE they are used in any components
    current_image = gr.State(None)  # Original image path
    working_image = gr.State(None)  # Working image path
    current_yaml = gr.State(None)   # Original YAML path
    working_yaml = gr.State(None)   # Working YAML path
    
    # Status text (hidden but used to track status)
    status_text = gr.State("")
    
    # Demo section at the top
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Demo Images")
            if demo_images:
                demo_gallery = gr.Gallery(
                    value=demo_images, 
                    columns=5,
                    object_fit="contain", 
                    height="150px",
                    show_label=False
                )
            else:
                gr.Markdown("*No demo images available. Place image files in the 'data/demos' directory.*")
    
    # Input and output sections below
    with gr.Row():
        # Left column for inputs
        with gr.Column():
            # Image upload section
            gr.Markdown("## Or Upload Your Own Image")
            image_upload = gr.File(
                label="Upload Image", 
                file_types=["image"],
                file_count="single"
            )
            
            # YAML config upload section
            gr.Markdown("## Upload YAML Configuration")
            yaml_upload = gr.File(
                label="Upload YAML Config", 
                file_types=[".yaml", ".yml"],
                file_count="single"
            )
            
            # Process button
            process_btn = gr.Button("Process Files", variant="primary", size="lg")
        
        # Right column for outputs
        with gr.Column():
            gr.Markdown("## Output Visualizations")
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

if __name__ == "__main__":
    app.launch() 