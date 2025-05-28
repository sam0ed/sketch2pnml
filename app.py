import os
import sys
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure we can import from the current directory structure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the refactored UI components
from ui import ConfigEditor, ImageProcessor, PetriConverter
from config.path_config import ensure_directories_exist

def create_app():
    """Create and configure the main Gradio application"""
    # Ensure directories exist
    ensure_directories_exist()
    
    # Create UI component instances
    config_editor = ConfigEditor()
    image_processor = ImageProcessor()
    petri_converter = PetriConverter()
    
    # Create the main application
    with gr.Blocks(title="Petri Net Converter Suite") as app:
        gr.Markdown("# Petri Net Converter Suite")
        
        with gr.Tabs():
            # Create each tab using the modular components
            config_editor.create_interface()
            image_processor.create_interface()
            petri_converter.create_interface()
    
    return app

def main():
    """Main entry point for the application"""
    app = create_app()
    app.launch()

if __name__ == "__main__":
    main() 