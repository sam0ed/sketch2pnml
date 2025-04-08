import cv2
import numpy as np

# Global variables to track viewer state
scale = 1.0
center_x = 0.0
center_y = 0.0
win_width = 768
win_height = 432
panning = False
prev_x = 0
prev_y = 0
img = None
img_height = 0
img_width = 0

def mouse_callback(event, x, y, flags, param):
    global scale, center_x, center_y, panning, prev_x, prev_y, win_width, win_height
    if event == cv2.EVENT_LBUTTONDOWN:
        panning = True
        prev_x = x
        prev_y = y
    elif event == cv2.EVENT_LBUTTONUP:
        panning = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if panning:
            delta_x = x - prev_x
            delta_y = y - prev_y
            center_x -= delta_x / scale
            center_y -= delta_y / scale
            prev_x, prev_y = x, y
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Zoom handling
        zoom_factor = 1.1 if flags > 0 else 0.9
        new_scale = scale * zoom_factor
        new_scale = max(min(new_scale, 10.0), 0.1)  # Limit scale
        
        # Calculate image coordinates under cursor
        img_x = center_x + (x - win_width/2) / scale
        img_y = center_y + (y - win_height/2) / scale
        
        # Update center to maintain cursor position
        center_x = img_x - (x - win_width/2) / new_scale
        center_y = img_y - (y - win_height/2) / new_scale
        scale = new_scale

def image_viewer(image_path, initial_win_width=768, initial_win_height=432):
    global img, img_height, img_width, scale, center_x, center_y, win_width, win_height
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
    img_height, img_width = img.shape[:2]
    
    # Setup window
    win_name = "Image Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, initial_win_width, initial_win_height)
    cv2.setMouseCallback(win_name, mouse_callback)
    
    # Initial scale and center
    win_width, win_height = initial_win_width, initial_win_height
    scale = min(win_width/img_width, win_height/img_height)
    center_x, center_y = img_width/2, img_height/2
    
    print("Instructions:")
    print("- Mouse wheel: Zoom in/out")
    print("- Left drag: Pan")
    print("- Press 'c': Center image")
    print("- Press 'q': Quit")

    prev_win_size = (win_width, win_height)
    
    while True:
        # Update window size
        current_win_rect = cv2.getWindowImageRect(win_name)
        if current_win_rect[2] > 0 and current_win_rect[3] > 0:
            win_width, win_height = current_win_rect[2], current_win_rect[3]
        
        # Choose interpolation
        interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_LINEAR
        
        # Generate scaled image
        scaled_w, scaled_h = int(img_width*scale), int(img_height*scale)
        if scaled_w == 0 or scaled_h == 0:
            display_img = np.zeros((win_height, win_width, 3), dtype=np.uint8)
        else:
            scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=interpolation)
            
            # Calculate visible area
            x_center = center_x * scale
            y_center = center_y * scale
            x1 = max(0, int(x_center - win_width//2))
            y1 = max(0, int(y_center - win_height//2))
            x2 = x1 + win_width
            y2 = y1 + win_height
            
            # Clamp to image boundaries
            x1, x2 = max(0, x1), min(scaled_w, x2)
            y1, y2 = max(0, y1), min(scaled_h, y2)
            visible = scaled_img[y1:y2, x1:x2]
            
            # Create display image
            display_img = np.zeros((win_height, win_width, 3), dtype=np.uint8)
            h, w = visible.shape[:2]
            display_img[:h, :w] = visible
        
        cv2.imshow(win_name, display_img)
        
        # Handle keys
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            center_x, center_y = img_width/2, img_height/2
    
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_viewer(r"c:\Users\samoed\Documents\GitHub\diploma_bachelor\assets\local\simple_petri_2.png")