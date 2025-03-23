import gradio as gr
import sys
import os
import tempfile
from PIL import Image
import threading
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap
from io import BytesIO

# Import the Capture class from the existing file
from Capturer import Capture

class GradioCapturerWrapper:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.capture_result = None
        self.app = None
        self.init_qt_app()
        
    def init_qt_app(self):
        # Initialize Qt application if not already running
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
    
    def capture_screen(self):
        """Use the existing Capturer to capture a screen region"""
        # We need to run this in a separate thread to avoid blocking Gradio
        thread = threading.Thread(target=self._run_capture)
        thread.daemon = True
        thread.start()
        
        # Wait for capture to complete (with timeout)
        max_wait = 30  # 30 seconds max wait
        start_time = time.time()
        while self.capture_result is None and time.time() - start_time < max_wait:
            time.sleep(0.5)
            
        if self.capture_result is None:
            return None, "Capture timed out or was cancelled."
            
        # Save to a temporary file
        temp_path = os.path.join(self.temp_dir, "capture.png")
        self.capture_result.save(temp_path)
        
        return temp_path, "Screen region captured successfully!"
    
    def _run_capture(self):
        """Run the capture in a separate thread"""
        # Create a dummy main window (needed by Capture but won't be shown)
        class DummyMainWindow:
            def __init__(self):
                self.label = None
                
            def hide(self):
                pass
                
            def show(self):
                pass
        
        try:
            dummy_window = DummyMainWindow()
            capturer = Capture(dummy_window)
            capturer.show()
            
            # Process Qt events until capture is complete
            while capturer.isVisible():
                self.app.processEvents()
                time.sleep(0.1)
            
            # Store the result
            self.capture_result = capturer.imgmap
            
        except Exception as e:
            print(f"Error in capture: {str(e)}")
            self.capture_result = None
    
    def save_image(self, image, save_path):
        """Save the captured image to the specified path"""
        if image is None:
            return "No image to save!"
        
        # Ensure the path has an extension
        if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            save_path += '.png'
        
        # Save the image
        try:
            if isinstance(image, str):
                # If it's a file path
                img = Image.open(image)
                img.save(save_path)
            else:
                # If it's already an image or numpy array
                Image.fromarray(image).save(save_path)
            
            return f"Image saved to {save_path}"
        except Exception as e:
            return f"Error saving image: {str(e)}"
    
    def build_interface(self):
        """Build and launch the Gradio interface"""
        with gr.Blocks(title="Screen Region Capturer", theme=gr.themes.Monochrome()) as interface:
            gr.Markdown("# Screen Region Capturer")
            
            with gr.Row():
                capture_btn = gr.Button("Capture Region", variant="primary")
            
            message_box = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                image_display = gr.Image(label="Captured Image", type="filepath")
            
            with gr.Row():
                save_path = gr.Textbox(label="Save Path (including filename)")
                save_btn = gr.Button("Save", variant="primary")
            
            # Connect the buttons to the respective functions
            capture_btn.click(
                self.capture_screen,
                inputs=[],
                outputs=[image_display, message_box]
            )
            
            save_btn.click(
                self.save_image,
                inputs=[image_display, save_path],
                outputs=[message_box]
            )
        
        return interface

def main():
    wrapper = GradioCapturerWrapper()
    interface = wrapper.build_interface()
    interface.launch()

if __name__ == "__main__":
    main()
