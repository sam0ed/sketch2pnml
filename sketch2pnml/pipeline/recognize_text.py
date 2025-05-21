import cv2
import numpy as np
from doctr.models import ocr_predictor
from .models import Text, Point, Place, Transition, Arc # Import the Text class
from .commons import filter_enclosed_contours, minmaxToContours, remove_contours, find_closest_distance_to_contour

def _geometry_to_absolute_coords(relative_geom: tuple[tuple[float, float], tuple[float, float]], 
                                 img_width: int, img_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Converts doctr's relative coordinates to absolute integer coordinates."""
    (xmin_rel, ymin_rel), (xmax_rel, ymax_rel) = relative_geom
    
    xmin_abs = int(xmin_rel * img_width)
    ymin_abs = int(ymin_rel * img_height)
    xmax_abs = int(xmax_rel * img_width)
    ymax_abs = int(ymax_rel * img_height)
    
    return ((xmin_abs, ymin_abs), (xmax_abs, ymax_abs))

def detect_text(img_color_resized: np.ndarray, config: dict) -> list[Text]:
    """
    Detects text using doctr and returns a list of Text objects with absolute coordinates.
    (Implementation remains the same)
    """
    predictor_params = config.get('text_detection', {})
    predictor = ocr_predictor(
        det_arch='db_resnet50', 
        reco_arch='crnn_vgg16_bn', 
        pretrained=True,
    )
    predictor.det_predictor.model.postprocessor.bin_thresh = predictor_params.get('bin_thresh', 0.3)
    predictor.det_predictor.model.postprocessor.box_thresh = predictor_params.get('box_thresh', 0.1)

    out = predictor([img_color_resized])

    detected_texts: list[Text] = []
    img_height, img_width = img_color_resized.shape[:2]

    if out.pages:
        for block in out.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    abs_geom = _geometry_to_absolute_coords(word.geometry, img_width, img_height)
                    text_obj = Text(value=word.value, 
                                      geometry_abs=abs_geom, 
                                      confidence=word.confidence)
                    detected_texts.append(text_obj)
    
    return detected_texts

def get_img_no_text(preprocessed_img: np.ndarray, detected_texts: list[Text]) -> np.ndarray:
    """
    Removes text from the preprocessed image by finding contours within text bounding boxes
    and applying a mask using bitwise_and (similar to original notebook).

    Args:
        preprocessed_img: The thresholded image (e.g., from Otsu).
        detected_texts: List of Text objects with absolute coordinates.

    Returns:
        The image with text contours removed (blacked out).
    """
    if not detected_texts:
        ### throw an error
        raise ValueError("No detected texts to process.")

    img_contours_list, _ = cv2.findContours(preprocessed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bbox_to_contours = [minmaxToContours([text.pt1.x, text.pt1.y, text.pt2.x, text.pt2.y])  for text in detected_texts]

    text_contours = filter_enclosed_contours(img_contours_list, bbox_to_contours, include_border=True)

    img_no_text = remove_contours(preprocessed_img, text_contours)

    return img_no_text 

# Helper function to get the center of a Text object's bounding box
def get_text_center(text_obj: Text) -> Point:
    """Calculates the center point of a Text object's bounding box."""
    center_x = (text_obj.pt1.x + text_obj.pt2.x) / 2.0
    center_y = (text_obj.pt1.y + text_obj.pt2.y) / 2.0
    return Point(center_x, center_y)


def link_text_to_elements(
    detected_text_list: list[Text],
    places_list: list[Place],
    transitions_list: list[Transition],
    arcs_list: list[Arc],
    config: dict
):
    """
    Associates Text objects with the closest Place, Transition, or Arc
    if the distance from the text's center to the element is within a given threshold.
    The association is done by appending the Text object to the `text` list
    of the corresponding Place, Transition, or Arc. 
    """
    distance_threshold = config.get('connection_processing', {}).get('text_linking_threshold',  25.0 )

    # Clear any previous text associations from elements
    for element_list in [places_list, transitions_list, arcs_list]:
        for element in element_list:
            element.text = [] 

    for text_obj in detected_text_list:
        text_center = get_text_center(text_obj)
        
        min_overall_distance = float('inf')
        closest_element_overall = None

        # 1. Check Places
        for place in places_list:
            dist_to_place_center = text_center.get_distance_between_points(place.center)
            distance = max(0, dist_to_place_center - place.radius) # Distance to circumference
            
            if distance < min_overall_distance:
                min_overall_distance = distance
                closest_element_overall = place

        # 2. Check Transitions
        for transition in transitions_list:
            contour_to_use = None
            if transition.original_detection_data is not None and \
               isinstance(transition.original_detection_data, np.ndarray) and \
               transition.original_detection_data.shape[0] > 0:
                contour_to_use = transition.original_detection_data
            elif transition.points and len(transition.points) > 0: # Fallback to box_points
                contour_to_use = np.array([p.get_numpy_array() for p in transition.points], dtype=np.int32).reshape((-1, 1, 2))

            if contour_to_use is None or contour_to_use.shape[0] == 0:
                continue 
                
            distance = find_closest_distance_to_contour(text_center, contour_to_use)
            if distance < min_overall_distance:
                min_overall_distance = distance
                closest_element_overall = transition
        
        # 3. Check Arcs
        for arc in arcs_list:
            arc_contour_for_dist_calc = None
            # Prioritize arc.points if available, as it represents the path
            if arc.points and len(arc.points) >= 1:
                arc_contour_for_dist_calc = np.array([p.get_numpy_array() for p in arc.points], dtype=np.int32).reshape((-1, 1, 2))
            # Fallback if arc.points is empty but start/end points are defined (simple line arc)
            elif arc.start_point and arc.end_point:
                arc_contour_for_dist_calc = np.array([
                    arc.start_point.get_numpy_array(), 
                    arc.end_point.get_numpy_array()
                ], dtype=np.int32).reshape((-1, 1, 2))
            # If arc is defined by arc.lines (more complex, potentially disjoint segments)
            # This path is less common if arc.points is expected to be canonical.
            elif arc.lines:
                current_arc_min_dist_lines = float('inf')
                for line_segment in arc.lines:
                    dist_to_segment = line_segment.distance_point_to_segment(text_center)
                    current_arc_min_dist_lines = min(current_arc_min_dist_lines, dist_to_segment)
                
                if current_arc_min_dist_lines < min_overall_distance:
                    min_overall_distance = current_arc_min_dist_lines
                    closest_element_overall = arc
                continue # Skip contour-based distance if lines were processed

            if arc_contour_for_dist_calc is not None and arc_contour_for_dist_calc.shape[0] > 0:
                distance = find_closest_distance_to_contour(text_center, arc_contour_for_dist_calc)
                if distance < min_overall_distance:
                    min_overall_distance = distance
                    closest_element_overall = arc
        
        # Associate text with the overall closest element if within threshold
        if closest_element_overall is not None and min_overall_distance <= distance_threshold:
            closest_element_overall.text.append(text_obj)
            # print(f"Associated '{text_obj.value}' (center: {text_center}) with {closest_element_overall.__class__.__name__} id={id(closest_element_overall)} (dist: {min_overall_distance:.2f})")
        # else:
            # print(f"Text '{text_obj.value}' (center: {text_center}) not associated, min_dist {min_overall_distance:.2f} > threshold {distance_threshold}")
