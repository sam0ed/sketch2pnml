import cv2
import numpy as np
from typing import Union, Tuple, Optional

class ImageViewer:
    """
    Interactive image viewer with zoom and pan capabilities using OpenCV.
    """
    def __init__(self, window_name: str = "Image Viewer", 
                 initial_width: int = 768, initial_height: int = 432):
        """
        Initialize the image viewer.
        
        Args:
            window_name: Name of the display window
            initial_width: Initial window width in pixels
            initial_height: Initial window height in pixels
        """
        self.window_name = window_name
        self.win_width = initial_width
        self.win_height = initial_height
        
        # Viewing state
        self.scale = 1.0
        self.center_x = 0.0
        self.center_y = 0.0
        
        # Panning state
        self.panning = False
        self.prev_x = 0
        self.prev_y = 0
        
        # Image data
        self.img = None
        self.img_height = 0
        self.img_width = 0
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zooming and panning."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.panning = True
            self.prev_x = x
            self.prev_y = y
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.panning = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.panning:
                delta_x = x - self.prev_x
                delta_y = y - self.prev_y
                self.center_x -= delta_x / self.scale
                self.center_y -= delta_y / self.scale
                self.prev_x, self.prev_y = x, y
                
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Determine zoom direction
            zoom_factor = 1.1 if flags > 0 else 0.9
            new_scale = self.scale * zoom_factor
            new_scale = max(min(new_scale, 10.0), 0.1)  # Limit scale
            
            # Calculate image coordinates under cursor
            img_x = self.center_x + (x - self.win_width/2) / self.scale
            img_y = self.center_y + (y - self.win_height/2) / self.scale
            
            # Update center to maintain cursor position
            self.center_x = img_x - (x - self.win_width/2) / new_scale
            self.center_y = img_y - (y - self.win_height/2) / new_scale
            self.scale = new_scale
    
    def _create_display_image(self) -> np.ndarray:
        """Generate the image to be displayed based on current view settings."""
        # Choose interpolation based on scale
        interpolation = cv2.INTER_NEAREST if self.scale >= 1.0 else cv2.INTER_LINEAR
        
        # Generate scaled image
        scaled_w, scaled_h = int(self.img_width * self.scale), int(self.img_height * self.scale)
        
        # Handle edge case of zero dimensions
        if scaled_w == 0 or scaled_h == 0:
            return np.zeros((self.win_height, self.win_width, 3), dtype=np.uint8)
        
        # Resize the image
        scaled_img = cv2.resize(self.img, (scaled_w, scaled_h), interpolation=interpolation)
        
        # Calculate visible area
        x_center = self.center_x * self.scale
        y_center = self.center_y * self.scale
        x1 = max(0, int(x_center - self.win_width//2))
        y1 = max(0, int(y_center - self.win_height//2))
        x2 = x1 + self.win_width
        y2 = y1 + self.win_height
        
        # Clamp to image boundaries
        x1, x2 = max(0, x1), min(scaled_w, x2)
        y1, y2 = max(0, y1), min(scaled_h, y2)
        visible = scaled_img[y1:y2, x1:x2]
        
        if len(visible.shape) == 2:  # if grayscale
            visible = cv2.cvtColor(visible, cv2.COLOR_GRAY2BGR)
        # Create display image
        display_img = np.zeros((self.win_height, self.win_width, 3), dtype=np.uint8)
        h, w = visible.shape[:2]
        display_img[:h, :w] = visible
        
        return display_img
            
    def _handle_keys(self, key: int) -> bool:
        """
        Handle keyboard inputs.
        
        Returns:
            True if viewing should continue, False if it should end
        """
        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.center_x, self.center_y = self.img_width/2, self.img_height/2
        return True
    
    def _initialize_view(self):
        """Initialize the scale and center position based on image and window size."""
        self.scale = min(self.win_width/self.img_width, self.win_height/self.img_height)
        self.center_x, self.center_y = self.img_width/2, self.img_height/2
    
    def show(self, image: Union[str, np.ndarray]) -> None:
        """
        Display an image with interactive controls.
        
        Args:
            image: Either a path to an image file or a numpy array containing image data
        """
        # Load or set the image
        if isinstance(image, str):
            self.img = cv2.imread(image)
            if self.img is None:
                raise ValueError(f"Could not read image from: {image}")
        else:
            self.img = image.copy()
            
        self.img_height, self.img_width = self.img.shape[:2]
        
        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_width, self.win_height)
        cv2.setMouseCallback(self.window_name, lambda *args: self._mouse_callback(*args))
        
        # Initialize view parameters
        self._initialize_view()
        
        print("Instructions:")
        print("- Mouse wheel: Zoom in/out")
        print("- Left drag: Pan")
        print("- Press 'c': Center image")
        print("- Press 'q': Quit")
        
        while True:
            # Update window size if it has changed
            current_win_rect = cv2.getWindowImageRect(self.window_name)
            if current_win_rect[2] > 0 and current_win_rect[3] > 0:
                self.win_width, self.win_height = current_win_rect[2], current_win_rect[3]
            
            # Create and display the image
            display_img = self._create_display_image()
            cv2.imshow(self.window_name, display_img)
            
            # Handle keys
            key = cv2.waitKey(1)
            if not self._handle_keys(key):
                break
        
        cv2.destroyWindow(self.window_name)

# Exportable function for easy use
def view_image(image: Union[str, np.ndarray], 
               window_name: str = "Image Viewer",
               width: int = 768, 
               height: int = 432) -> None:
    """
    Display an image with interactive zoom and pan capabilities.
    
    Args:
        image: Either a path to an image file or a numpy array containing image data
        window_name: Title of the viewer window
        width: Initial window width
        height: Initial window height
    """
    viewer = ImageViewer(window_name, width, height)
    viewer.show(image)

# Example usage
if __name__ == "__main__":
    # Using the class directly
    # viewer = ImageViewer("Custom Viewer")
    # viewer.show(r"c:\Users\samoed\Documents\GitHub\diploma_bachelor\assets\local\simple_petri_2.png")
    
    # Or using the simplified function
    # view_image(r"c:\Users\samoed\Documents\GitHub\diploma_bachelor\assets\local\simple_petri_2.png")

    img = cv2.imread(r"c:\Users\samoed\Documents\GitHub\diploma_bachelor\assets\local\simple_petri_2.png")    # view_image(np.zeros((100, 100, 3), dtype=np.uint8), "Test Image")
    view_image(img, "Test Image")