import cv2
import numpy as np
import os

def preprocess(img):
    """Applies Otsu's thresholding to the input image."""
    # Ensure input is grayscale if it's not already
    gray_img = img
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh_otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_otsu

def load_and_preprocess_image(image_path: str, config: dict):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Could not read image file: {image_path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # --- Upscaling Heuristic ---
    cfg_proc = config.get('image_processing', {})
    min_dimension_threshold = cfg_proc.get('min_dimension_threshold', 800)
    upscale_factor = cfg_proc.get('upscale_factor', 2)

    h, w = img_gray.shape
    img_color_resized = img_color
    img_gray_resized = img_gray

    if h < min_dimension_threshold or w < min_dimension_threshold:
        print(f"Image dimensions ({w}x{h}) below threshold ({min_dimension_threshold}px). Upscaling by {upscale_factor}x.")
        new_w, new_h = w * upscale_factor, h * upscale_factor
        img_gray_resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        img_color_resized = cv2.resize(img_color, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # --- Initial Preprocessing (Inversion + Thresholding) ---
    img_inverted = cv2.bitwise_not(img_gray_resized)
    preprocessed_img = preprocess(img_inverted) 

    return preprocessed_img, img_color_resized, img_gray_resized 