import gradio as gr
import os
from .base import BaseUI
from endpoints.converter import (
    fix_petri_net, render_diagram_to, render_to_graphviz, render_to_json
)
from config.path_config import (
    OUTPUT_PNML_PATH, OUTPUT_PETRIOBJ_PATH, OUTPUT_JSON_PATH, 
    OUTPUT_PNG_PATH, OUTPUT_GV_PATH, OUTPUT_DIR, ensure_directories_exist
)

def create_download_file(content: str, filename: str, file_extension: str, output_dir: str) -> str:
    """Create a temporary file with the given content for download"""
    try:
        # Create filename with proper extension
        if not filename.endswith(file_extension):
            filename = f"{filename}{file_extension}"
        
        file_path = os.path.join(output_dir, filename)
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return file_path
    except Exception as e:
        print(f"Error creating download file: {e}")
        return ""

class PetriConverter(BaseUI):
    """Petri Net Converter UI component"""
    
    def __init__(self):
        super().__init__()
        
        # UI components
        self.translate_button = None
        self.pnml_output = None
        self.petriobj_output = None
        self.json_output = None
        self.image_output = None
        self.gv_output = None
        
        # Download components
        self.pnml_download_btn = None
        self.pnml_download = None
        self.petriobj_download_btn = None
        self.petriobj_download = None
        self.json_download_btn = None
        self.json_download = None
        self.gv_download_btn = None
        self.gv_download = None
        self.png_download = None
        
        # Hidden state components for file paths
        self.pnml_path_comp = None
        self.petriobj_path_comp = None
        self.json_path_comp = None
        self.gv_path_comp = None
        self.png_path_comp = None
    
    def create_interface(self) -> gr.TabItem:
        """Create the Petri Net Converter interface"""
        with gr.TabItem("Petri Net Converter") as tab:
            gr.Markdown("## Petri Net Sketch Converter")
            gr.Markdown("Press the Translate button to convert the working image to a Petri net")
            
            with gr.Row():
                self.translate_button = self.create_button("Translate", variant="primary", size="lg")
            
            # Hidden components to store file paths for downloads
            self.pnml_path_comp = self.create_state_var("pnml_path", "")
            self.petriobj_path_comp = self.create_state_var("petriobj_path", "")
            self.json_path_comp = self.create_state_var("json_path", "")
            self.gv_path_comp = self.create_state_var("gv_path", "")
            self.png_path_comp = self.create_state_var("png_path", "")
            
            with gr.Tabs():
                with gr.TabItem("PNML"):
                    self.pnml_output = gr.Code(label="PNML Output", lines=20, max_lines=25, interactive=True, language="html")
                    with gr.Row():
                        self.pnml_download_btn = self.create_button("Download PNML")
                        self.pnml_download = gr.File(label="Download PNML File", visible=False, interactive=False)
                
                with gr.TabItem("PetriObj"):
                    self.petriobj_output = gr.Code(label="PetriObj Output", lines=20, max_lines=25, interactive=True, language="c")
                    with gr.Row():
                        self.petriobj_download_btn = self.create_button("Download PetriObj")
                        self.petriobj_download = gr.File(label="Download PetriObj File", visible=False, interactive=False)
                
                with gr.TabItem("JSON"):
                    self.json_output = gr.Code(label="JSON Output", lines=20, max_lines=25, interactive=True, language="json")
                    with gr.Row():
                        self.json_download_btn = self.create_button("Download JSON")
                        self.json_download = gr.File(label="Download JSON File", visible=False, interactive=False)
                
                with gr.TabItem("Visualization"):
                    self.image_output = gr.Image(label="Petri Net Visualization", type="filepath")
                    self.png_download = gr.File(label="Download PNG File", interactive=False)
                
                with gr.TabItem("GraphViz"):
                    self.gv_output = gr.Code(label="GraphViz Output", lines=20, max_lines=25, interactive=True, language="markdown")
                    with gr.Row():
                        self.gv_download_btn = self.create_button("Download GraphViz")
                        self.gv_download = gr.File(label="Download GraphViz File", visible=False, interactive=False)
            
            self.setup_event_handlers()
            
        return tab
    
    def setup_event_handlers(self):
        """Setup event handlers for the Petri Net Converter"""
        # Connect the button to the processing function
        self.translate_button.click(
            fn=self._process_and_display,
            outputs=[
                self.pnml_output, self.petriobj_output, self.json_output, self.image_output, self.gv_output,
                self.pnml_path_comp, self.petriobj_path_comp, self.json_path_comp, self.gv_path_comp, self.png_path_comp
            ]
        ).then(
            lambda path: path,
            inputs=self.png_path_comp,
            outputs=self.png_download
        )
        
        # Connect download buttons to download functions
        self.pnml_download_btn.click(
            fn=self._download_pnml,
            inputs=[self.pnml_output],
            outputs=[self.pnml_download]
        ).then(
            lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
            inputs=[self.pnml_download],
            outputs=[self.pnml_download]
        )
        
        self.petriobj_download_btn.click(
            fn=self._download_petriobj,
            inputs=[self.petriobj_output],
            outputs=[self.petriobj_download]
        ).then(
            lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
            inputs=[self.petriobj_download],
            outputs=[self.petriobj_download]
        )
        
        self.json_download_btn.click(
            fn=self._download_json,
            inputs=[self.json_output],
            outputs=[self.json_download]
        ).then(
            lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
            inputs=[self.json_download],
            outputs=[self.json_download]
        )
        
        self.gv_download_btn.click(
            fn=self._download_gv,
            inputs=[self.gv_output],
            outputs=[self.gv_download]
        ).then(
            lambda path: gr.update(visible=True, value=path) if path else gr.update(visible=False),
            inputs=[self.gv_download],
            outputs=[self.gv_download]
        )
    
    def _process_and_display(self):
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
            error_message = self.handle_error(e, "during processing")
            print(error_message)
            empty_path = ""
            return f"Error: {error_message}", f"Error: {error_message}", f"Error: {error_message}", None, f"Error: {error_message}", empty_path, empty_path, empty_path, empty_path, empty_path
    
    def _download_pnml(self, content: str) -> str:
        """Create downloadable PNML file from current content"""
        return create_download_file(content, "edited_output.pnml", ".pnml", OUTPUT_DIR)
    
    def _download_petriobj(self, content: str) -> str:
        """Create downloadable PetriObj file from current content"""
        return create_download_file(content, "edited_output.petriobj", ".petriobj", OUTPUT_DIR)
    
    def _download_json(self, content: str) -> str:
        """Create downloadable JSON file from current content"""
        return create_download_file(content, "edited_output.json", ".json", OUTPUT_DIR)
    
    def _download_gv(self, content: str) -> str:
        """Create downloadable GraphViz file from current content"""
        return create_download_file(content, "edited_output.gv", ".gv", OUTPUT_DIR) 