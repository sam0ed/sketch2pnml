import gradio as gr
import os
import shutil
from typing import Optional
from .base import BaseUI
from config.path_config import (
    DEMOS_DIR as DEMO_IMAGES_DIR, VISUALIZATIONS_DIR, 
    SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_CONFIG_EXTENSIONS, 
    ensure_directories_exist, get_working_image_path, get_working_config_path
)
from endpoints.converter import run_and_save_pipeline

class ImageProcessor(BaseUI):
    """Image Processor UI component"""
    
    def __init__(self):
        super().__init__()
        self.demo_images = self._load_demo_images()
        
        # UI components
        self.demo_gallery = None
        self.image_upload = None
        self.yaml_upload = None
        self.process_btn = None
        self.output_gallery = None
        self.output_message = None
        
        # State variables
        self.current_image = None
        self.working_image = None
        self.current_yaml = None
        self.working_yaml = None
        self.status_text = None
    
    def _load_demo_images(self):
        """Load demo images from the demos directory"""
        try:
            demo_images = [os.path.join(DEMO_IMAGES_DIR, f) for f in os.listdir(DEMO_IMAGES_DIR) 
                         if f.endswith(SUPPORTED_IMAGE_EXTENSIONS)]
            if not demo_images:
                print("No demo images found in the demos directory. Please add some images to", DEMO_IMAGES_DIR)
                return []
            return demo_images
        except Exception as e:
            print(f"Error loading demo images: {e}")
            return []
    
    def create_interface(self) -> gr.TabItem:
        """Create the Image Processor interface"""
        with gr.TabItem("Image Processor") as tab:
            gr.Markdown("## Sketch to PNML Converter")
            
            # State variables for the app
            self.current_image = self.create_state_var("current_image", None)
            self.working_image = self.create_state_var("working_image", None)
            self.current_yaml = self.create_state_var("current_yaml", None)
            self.working_yaml = self.create_state_var("working_yaml", None)
            self.status_text = self.create_state_var("status_text", "")
            
            # Demo section at the top
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Demo Images")
                    if self.demo_images:
                        self.demo_gallery = gr.Gallery(
                            value=self.demo_images, 
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
                    self.image_upload = self.create_file_upload(
                        "Upload Image", 
                        ["image"]
                    )
                    
                    # YAML config upload section
                    gr.Markdown("### Upload YAML Configuration")
                    self.yaml_upload = self.create_file_upload(
                        "Upload YAML Config", 
                        list(SUPPORTED_CONFIG_EXTENSIONS)
                    )
                    
                    # Process button
                    self.process_btn = self.create_button("Process Files", variant="primary", size="lg")
                
                # Right column for outputs
                with gr.Column():
                    gr.Markdown("### Output Visualizations")
                    self.output_gallery = gr.Gallery(
                        label="Processing Results", 
                        columns=1,
                        height=400,
                        show_label=True
                    )
                    self.output_message = gr.Markdown("")
            
            self.setup_event_handlers()
            
        return tab
    
    def setup_event_handlers(self):
        """Setup event handlers for the Image Processor"""
        # Event handlers
        if self.demo_images and self.demo_gallery:
            self.demo_gallery.select(
                self._select_demo_image, 
                inputs=[self.current_yaml, self.working_yaml],
                outputs=[self.current_image, self.working_image, self.current_yaml, self.working_yaml, self.status_text]
            )
        
        self.image_upload.change(
            self._upload_image, 
            inputs=[self.image_upload, self.current_yaml, self.working_yaml], 
            outputs=[self.current_image, self.working_image, self.current_yaml, self.working_yaml, self.status_text]
        )
        
        self.yaml_upload.change(
            self._upload_yaml, 
            inputs=[self.yaml_upload, self.current_image, self.working_image], 
            outputs=[self.current_image, self.working_image, self.current_yaml, self.working_yaml, self.status_text]
        )
        
        self.process_btn.click(
            self._process_files, 
            inputs=[self.working_image, self.working_yaml], 
            outputs=[self.output_gallery, self.output_message]
        )
        
        # Update status message when it changes
        self.status_text.change(lambda x: x, inputs=[self.status_text], outputs=[self.output_message])
    
    def _save_as_working_image(self, image_path):
        """Save the selected image as the working image"""
        # Ensure the extension is preserved when copying
        _, ext = os.path.splitext(image_path)
        working_image_path = get_working_image_path(ext)
        
        shutil.copy(image_path, working_image_path)
        return working_image_path
    
    def _save_as_working_yaml(self, yaml_path):
        """Save the uploaded YAML as the working configuration"""
        _, ext = os.path.splitext(yaml_path)
        working_yaml_path = get_working_config_path(ext)
        
        shutil.copy(yaml_path, working_yaml_path)
        return working_yaml_path
    
    def _clear_visualizations(self):
        """Clear the visualizations directory"""
        # Create the directory if it doesn't exist
        if not os.path.exists(VISUALIZATIONS_DIR):
            os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        
        # Remove existing files
        for file in os.listdir(VISUALIZATIONS_DIR):
            file_path = os.path.join(VISUALIZATIONS_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    def _process_files(self, image_path, yaml_path):
        """Process the uploaded files and run the pipeline"""
        if image_path is None or yaml_path is None:
            return None, "Both image and YAML configuration file are required."
        
        # Clear previous visualization results
        self._clear_visualizations()
        
        # Run the pipeline
        try:
            run_and_save_pipeline(config_path=yaml_path, image_path=image_path)
            
            # Get visualization results
            vis_files = [os.path.join(VISUALIZATIONS_DIR, f) for f in os.listdir(VISUALIZATIONS_DIR) if f.endswith('.png')]
            
            if not vis_files:
                return None, "No visualization results were generated."
            
            return vis_files, "Processing completed successfully!"
        except Exception as e:
            return None, self.handle_error(e, "during processing")
    
    def _select_demo_image(self, evt: gr.SelectData, current_yaml=None, working_yaml=None):
        """Handle selection of a demo image"""
        image_path = self.demo_images[evt.index]
        if image_path:
            working_image_path = self._save_as_working_image(image_path)
            status = "Demo image selected."
            if working_yaml:
                status += " Ready to process!"
            else:
                status += " Please upload a YAML configuration file."
            return image_path, working_image_path, current_yaml, working_yaml, status
        return None, None, current_yaml, working_yaml, ""
    
    def _upload_image(self, file_obj, current_yaml=None, working_yaml=None):
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
    
    def _upload_yaml(self, file_obj, current_image, working_image):
        """Handle upload of a YAML file"""
        if file_obj is not None:
            # Save the uploaded YAML as the working config
            yaml_path = file_obj.name
            working_yaml_path = self._save_as_working_yaml(yaml_path)
            
            if working_image is not None:
                return current_image, working_image, yaml_path, working_yaml_path, "Both files uploaded. Ready to process!"
            else:
                return None, None, yaml_path, working_yaml_path, "YAML file uploaded. Please select or upload an image."
        return current_image, working_image, None, None, "" 