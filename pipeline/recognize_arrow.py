import base64
import requests
import cv2
import numpy as np
import supervision as sv
import os

# Function to load configuration (can be moved to a central utils.py if used elsewhere)
# For now, keep it simple. The main caller (notebook) will load and pass the config.

def detect_arrowheads(
    image: np.ndarray,
    config: dict # Expects the full loaded YAML config
    # image_path: str = None # This was unused and can be removed if image is always passed as np.ndarray
) -> dict:
    """
    Detects objects (arrowheads) in an image using the Roboflow API.
    Configuration for the API (project_id, version, api_key, confidence)
    is expected to be in the passed config dictionary.
    """

    api_config = config.get('connection_processing', {}).get('arrowhead_api', {})
    project_id = api_config.get('project_id')
    version = api_config.get('version')
    
    # Load API key from environment variable instead of config
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")
    
    # Roboflow API expects confidence as a percentage (0-100)
    confidence = api_config.get('confidence_threshold_percent', 10.0) # Default if not found

    if not all([project_id, version]):
        raise ValueError("Missing Roboflow API configuration (project_id or version) in config.")

    # Encode the image
    # The original code had a commented-out section for reading from image_path.
    # Sticking to encoding the provided numpy array.
    success, encoded_image_bytes = cv2.imencode(".png", image) # Using .png as it's lossless; .jpg was also an option
    if not success:
        raise ValueError("Could not encode image to PNG format.")
    
    # Base64-encode the image bytes
    b64_encoded_image = base64.b64encode(encoded_image_bytes.tobytes()).decode("utf-8")

    # Build the request URL with query parameters
    # Note: The confidence parameter in the URL is the threshold.
    url = (
        f"https://detect.roboflow.com/{project_id}/{version}"
        f"?api_key={api_key}"
        f"&confidence={confidence}"  # This should be the percentage value
        "&format=json"
        # Consider adding other parameters like overlap, stroke, labels if needed,
        # and managing them via config.
    )

    # Send the POST request with the base64-encoded image
    headers = {"Content-Type": "application/x-www-form-urlencoded"} # Roboflow expects this for base64 data
    response = requests.post(url, data=b64_encoded_image, headers=headers)
    response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

    return response.json()


def show_arrows(img, arrowhead_result):
    img_drawn = img.copy()
    detections = sv.Detections.from_inference(arrowhead_result)

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=img_drawn, detections=detections)

    sv.plot_image(annotated_image)