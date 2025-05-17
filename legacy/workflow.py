# %% [markdown]
# # Petri Net Recognition Workflow

# %% [markdown]
# ## 1. Imports and Setup

# %% 
# Basic imports
import cv2
import numpy as np
import math
import os
import yaml # For loading config
import matplotlib.pyplot as plt
import largestinteriorrectangle as lir
from PIL import Image # Keep if needed for other display/manipulation

# Import custom modules
from models import Point, Line, Place, Transition, Text # Add Text import
from data_loading import load_and_preprocess_image
# Import the correct functions
from recognize_text import detect_text, remove_text_contours 
from recognize_node import fill_empty_nodes, get_nodes_mask, detect_shapes, remove_nodes
# %% [markdown]
# ## 2. Configuration Loading

# %% 
CONFIG_PATH = 'config.yaml'
config = {}

try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {CONFIG_PATH}") # Kept simple confirmation
except FileNotFoundError:
    print(f"Error: {CONFIG_PATH} not found. Using empty config.")
except Exception as e:
    print(f"Error loading or parsing {CONFIG_PATH}: {e}. Using empty config.")

# %% [markdown]
# ## 3. Data Loading

# %% 
INPUT_IMAGE_PATH = 'data/local/mid_petri_2.png' # Example relative path

preprocessed_img = None
img_color_resized = None
img_gray_resized = None

try:
    # Load and preprocess
    preprocessed_img, img_color_resized, img_gray_resized = load_and_preprocess_image(INPUT_IMAGE_PATH, config)
    print(f"Image loaded and preprocessed from: {INPUT_IMAGE_PATH}") # Kept simple confirmation

    if img_color_resized is not None:
        # Convert BGR (OpenCV) to RGB (PIL) for color images
        Image.fromarray(cv2.cvtColor(img_color_resized, cv2.COLOR_BGR2RGB)).show(title="Resized Color")
    
    if img_gray_resized is not None:
        Image.fromarray(img_gray_resized).show(title="Resized Gray")

    if preprocessed_img is not None:
        Image.fromarray(preprocessed_img).show(title="Inverted + Thresholded")
    
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    # Ensure variables are None if loading fails
    preprocessed_img = None 
    img_color_resized = None
    img_gray_resized = None

# %% [markdown]
# ## 4. Text Detection and Removal (Contour-based)

# %% 
detected_text_list = []
img_no_text = None # Renamed variable for clarity

# Proceed only if data loading was successful
if img_color_resized is not None and preprocessed_img is not None: # Need preprocessed_img now
    # print("\nStarting text detection...") # Removed
    try:
        # 1. Detect Text (using color image)
        detected_text_list = detect_text(img_color_resized, config)
        
        print(f"Detected {len(detected_text_list)} text elements:") # Keep summary
        if detected_text_list:
            for text_obj in detected_text_list:
                print(f"  - {text_obj}") # Keep details of detected text
        else:
            print("  (No text detected)") # Redundant if list is empty
            
        # 2. Remove Text Contours based on detections (using thresholded image)
        img_no_text = remove_text_contours(preprocessed_img, detected_text_list)


        if preprocessed_img is not None:
            Image.fromarray(preprocessed_img).show(title="Original Thresholded Image")
        
        if img_no_text is not None:
            Image.fromarray(img_no_text).show(title="Image with Text Contours Removed")

    except ImportError as e:
        print(f"ImportError during text processing: {e}")
        print("Please ensure doctr and its backend (TensorFlow or PyTorch) are installed.")
    except Exception as e:
        print(f"An unexpected error occurred during text processing: {e}")
else:
    print("\nSkipping text detection/removal because data loading failed or was skipped.")

# %% [markdown]
# ## 5. Node Isolation and Shape Classification

img_empty_nodes_filled = None
nodes_mask = None
detected_circles = []
detected_rectangles = []
img_no_shapes = None

# Proceed only if the image after text removal is available
if img_no_text is not None:
    try:
        print("\nFilling empty nodes...")
        img_empty_nodes_filled = fill_empty_nodes(img_no_text, config) # Pass img_no_text here
        Image.fromarray(img_empty_nodes_filled).show(title="After Filling Empty Nodes")

        print("\nIsolating nodes mask...")
        nodes_mask = get_nodes_mask(img_empty_nodes_filled, config) 
        Image.fromarray(nodes_mask).show(title="Isolated Nodes Mask")

        print("\nDetecting shapes (Places/Transitions)...")
        # Store the results for later use
        detected_circles, detected_rectangles = detect_shapes(nodes_mask, config)
        print(f"Detected {len(detected_circles)} potential Places (circles).")
        print(f"Detected {len(detected_rectangles)} potential Transitions (rectangles).")

        # Optional: Visualize detected shapes on the color image (create a copy)
        img_shapes_viz = img_color_resized.copy() if img_color_resized is not None else None
        if img_shapes_viz is not None:
            for (x,y,radius) in detected_circles:
                cv2.circle(img_shapes_viz, (x,y), radius, (0,255,0), 2) # Green for circles
            for rect in detected_rectangles: # Assuming lir rectangles
                # Use lir helpers pt1 and pt2 to get diagonal corners for cv2.rectangle
                pt1 = lir.pt1(rect) 
                pt2 = lir.pt2(rect)
                cv2.rectangle(img_shapes_viz, pt1, pt2, (0,0,255), 2) # Red for rectangles
            Image.fromarray(cv2.cvtColor(img_shapes_viz, cv2.COLOR_BGR2RGB)).show(title="Detected Shapes")


        print("\nRemoving detected nodes for connection processing...")
        # Use the image *before* empty node filling but *after* text removal for removing shapes
        img_no_shapes = remove_nodes(img_no_text, detected_circles, detected_rectangles, config)
        Image.fromarray(img_no_shapes).show(title="Image with Nodes Removed")

    except Exception as e:
        print(f"An error occurred during shape detection/removal: {e}")
        # Ensure subsequent steps know these might be invalid
        detected_circles = []
        detected_rectangles = []
        img_no_shapes = None
else:
    print("\nSkipping shape detection because text removal failed or was skipped.")

# Old direct calls - replaced by the block above
# img_empty_nodes_filled = fill_empty_nodes(preprocessed_img)
# Image.fromarray(img_empty_nodes_filled).show()

# nodes_mask = get_nodes_mask(img_empty_nodes_filled) 
# Image.fromarray(nodes_mask).show()

# circles, rectangles = detect_shapes(nodes_mask)
# img_no_shapes = remove_nodes(preprocessed_img, circles, rectangles)
# Image.fromarray(img_no_shapes).show()

# %% 
# Results from this section to be used later:
# - detected_circles: List of (x, y, radius) tuples for Places
# - detected_rectangles: List of lir rectangle objects for Transitions
# - img_no_shapes: Image ready for connection processing


# %% [markdown]
# ## 6. Connection Processing (Arcs) (To be added)

# %% 
# Placeholder for connection processing logic
# print("\nPlaceholder for Connection Processing module...") # Removed placeholder print


# %% [markdown]
# ## 7. Petri Net Construction (To be added)

# %% 
# Placeholder for Petri Net construction logic
# print("\nPlaceholder for Petri Net Construction module...") # Removed placeholder print


# %% [markdown]
# ## 8. Output/Export (To be added)

# %% 
# Placeholder for output/export logic
# print("\nPlaceholder for Output/Export module...") # Removed placeholder print
