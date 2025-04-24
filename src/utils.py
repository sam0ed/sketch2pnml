import os
import cv2
import inspect
import numpy as np
import shutil
from typing import Callable
import matplotlib.pyplot as plt

def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable][0]

def rename_files_in_folder(folder_path, new_name):
    """
    Renames all files in the specified folder to new_name + sequential number.
    
    Args:
        folder_path (str): Path to the folder containing files to rename
        new_name (str): Base name to use for renamed files
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    # Get all files in the directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Sort files to ensure consistent numbering
    files.sort()
    
    # Rename each file
    for i, filename in enumerate(files, 1):
        # Get file extension
        _, ext = os.path.splitext(filename)
        
        # Create new filename
        new_filename = f"{new_name}{i}{ext}"
        
        # Create full paths
        src_path = os.path.join(folder_path, filename)
        dst_path = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(src_path, dst_path)
        print(f"Renamed: {filename} -> {new_filename}")

def move_filtered_files(source_dir: str, destination_dir: str, file_filter: Callable[[str], bool]):
    """
    Move files from source directory to destination directory based on a filter function.
    
    Args:
        source_dir: Path to the source directory
        destination_dir: Path to the destination directory
        file_filter: Function that returns True for files to be moved
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Get all files in the source directory
    try:
        files = os.listdir(source_dir)
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
        return
    
    # Filter files according to the provided function
    selected_files = [f for f in files if file_filter(f)]
    
    if not selected_files:
        print(f"No matching files found in {source_dir}")
        return
    
    # Move each file
    for file in selected_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(destination_dir, file)
        shutil.move(source_path, dest_path)
        print(f"Moved {file} to {destination_dir}")
    
    print(f"Moved {len(selected_files)} files.")

def interactive_tuner(func, image, param_ranges=None, window_name="Tuner"):
    """
    Creates an interactive window with a trackbar for each hyperparameter of `func`.
    
    Parameters:
      func (function): A function that accepts an image as its first parameter 
                       and hyperparameters as keyword arguments.
      image (numpy.ndarray): The image on which `func` will operate.
      param_ranges (dict): Optional dictionary mapping parameter names to (min, max) tuples.
                           If not provided, a default range is chosen based on the default value.
      window_name (str): Name of the created window.
    """
    # Use inspect to get the function signature.
    sig = inspect.signature(func)
    params = list(sig.parameters.items())
    
    # Assume the first parameter is the image, the rest are hyperparameters.
    hyperparams = params[1:]
    
    # Create a dictionary for hyperparameters with their default values.
    current_values = {}
    for name, param in hyperparams:
        if param.default is not inspect.Parameter.empty:
            current_values[name] = param.default
        else:
            # If no default is provided, assume a starting value of 0.
            current_values[name] = 0

    # Create the main window.
    cv2.namedWindow(window_name)
    
    def nothing(x):
        pass
    
    # Create a trackbar for each hyperparameter.
    for name, value in current_values.items():
        # If the caller provided a range, use it.
        if param_ranges and name in param_ranges:
            min_val, max_val = param_ranges[name]
        else:
            # Choose a default range.
            # If the default is an integer less than 255, use 0-255.
            # Otherwise, use 0 to double the default value.
            if isinstance(value, int):
                if value < 255:
                    min_val, max_val = 0, 255
                else:
                    min_val, max_val = 0, value * 2
            else:
                try:
                    int_val = int(value)
                    if int_val < 255:
                        min_val, max_val = 0, 255
                    else:
                        min_val, max_val = 0, int_val * 2
                except Exception:
                    min_val, max_val = 0, 255
        cv2.createTrackbar(name, window_name, value, max_val, nothing)
    
    # Main loop: update the image every time a trackbar is changed.
    while True:
        # Read current hyperparameter values from trackbars.
        for name in current_values.keys():
            current_values[name] = cv2.getTrackbarPos(name, window_name)
        
        # Call the provided function with the image and current hyperparameter values.
        result = func(image, **current_values)
        cv2.imshow(window_name, result)
        
        # Exit when the user presses the ESC key.
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def interactive_tuner_demo():
        # Load an image from file
    image_path = "./assets/local/simple_petri_2.png"  # Update this with your actual image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    

    if image is None:
        print("Error loading image!")
        exit(1)
    
    # ## Canny Edge Detector Example
    # Wrap cv2.Canny into a function that accepts keyword arguments.
    def canny_filter(img, threshold1=100, threshold2=200, apertureSize=3):
        return cv2.Canny(img, threshold1, threshold2, apertureSize=apertureSize)
    
    # Optionally, you can define custom ranges for each parameter.
    custom_ranges = {
        "threshold1": (0, 255),
        "threshold2": (0, 255),
        "apertureSize": [i for i in range(3, 7) if i%2!=0 ]  # Make sure this range only includes odd numbers if needed.
    }
    
    interactive_tuner(canny_filter, image, param_ranges=custom_ranges)

    ## Hough Transform Example
    def hough_circle_visualizer(img, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0):
        # Detect circles
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp, minDist,
                                param1=param1, param2=param2,
                                minRadius=minRadius, maxRadius=maxRadius)
        
        # Create a color image for display
        output = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        
        # If circles were detected, draw them
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # Draw the outer circle
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        return output
    
    interactive_tuner(hough_circle_visualizer, image)

def rescale_image(image, target_size=640):
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def display_images(images, rows=1, figsize=(15, 10)):
    """
    Display multiple images in a grid without titles.
    
    Args:
        images: List of images to display
        rows: Number of rows in the grid
        figsize: Figure size
    """
    cols = len(images) // rows + (1 if len(images) % rows != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, image in enumerate(images):
        if len(image.shape) == 2:  # Grayscale
            axes[i].imshow(image, cmap='gray')
        else:  # Color
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass
    # Move files that have "_temp" from data/local to data/upscaled
    # move_filtered_files("data/local", "data/upscaled", lambda filename: "_temp" in filename)
