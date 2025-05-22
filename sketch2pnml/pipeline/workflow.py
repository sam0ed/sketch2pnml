import yaml
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Any
import os
import supervision as sv

from .data_loading import load_and_preprocess_image
from .recognize_text import detect_text, get_img_no_text, link_text_to_elements
from .recognize_node import detect_shapes, get_nodes_mask
from .commons import fill_contours, remove_contours, dilate_contour, here
from .recognize_arc import (get_hough_lines, HoughBundler, assign_proximity_nodes, 
                          get_entry_points_from_lines, find_line_paths, 
                          assign_arrowheads, get_arcs)
from .recognize_arrow import detect_arrowheads
from .models import Place, Transition, Line, Point
from skimage.morphology import skeletonize

def recognize_graph(
    image_path: str, 
    config_path: str 
) -> Dict[str, Any]:
    """
    Process an image to recognize a graph structure.
    
    Args:
        image_path: Path to the input image
        config_path: Path to the configuration file (default: 'config.yaml')
        
    Returns:
        Dictionary containing:
        - places: List of Place objects
        - transitions: List of Transition objects
        - arcs: List of Arc objects
        - visualizations: Dictionary of visualization images at different steps
    """
    # Initialize result containers
    result = {
        "places": [],
        "transitions": [],
        "arcs": [],
        "visualizations": {}
    }
    
    # Load configuration
    config = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Using empty config.")
    except Exception as e:
        print(f"Error loading or parsing {config_path}: {e}. Using empty config.")
    
    # Step 1: Load and preprocess image
    preprocessed_img, img_color_resized, img_gray_resized = load_and_preprocess_image(image_path, config)
    result["visualizations"]["original"] = Image.fromarray(img_color_resized)
    
    # Step 2: Detect and remove text
    detected_text_list = detect_text(img_color_resized, config)
    img_no_text = get_img_no_text(preprocessed_img, detected_text_list)
    result["visualizations"]["no_text"] = Image.fromarray(img_no_text)
    
    # Step 3: Detect shapes (circles and rectangles)
    circles, rectangles = detect_shapes(img_no_text, config)
    img_empty_nodes_filled = fill_contours(img_no_text, circles + rectangles)
    
    # Get isolated nodes mask
    nodes_mask = get_nodes_mask(img_empty_nodes_filled, config)
    detected_circles, detected_rectangles = detect_shapes(nodes_mask, config)
    
    # Step 4: Process node shapes
    dilated_circles = [dilate_contour(c, img_no_text.shape, config) for c in detected_circles]
    dilated_rectangles = [dilate_contour(r, img_no_text.shape, config) for r in detected_rectangles]
    img_no_shapes = remove_contours(img_empty_nodes_filled, dilated_circles + dilated_rectangles)
    
    # Visualize detected shapes
    shapes_visualization = cv2.cvtColor(img_no_shapes.copy(), cv2.COLOR_GRAY2BGR)
    for contour in detected_circles + detected_rectangles:
        cv2.drawContours(shapes_visualization, [contour], -1, (0, 255, 0), 2)
    result["visualizations"]["detected_shapes"] = Image.fromarray(shapes_visualization)
    result["visualizations"]["no_shapes"] = Image.fromarray(img_no_shapes)
    
    # Step 5: Create Place and Transition objects
    places = [Place.from_contour(circle) for circle in detected_circles]
    transitions = [Transition.from_contour(rect) for rect in detected_rectangles]
    result["places"] = places
    result["transitions"] = transitions
    
    # Visualize places and transitions
    nodes_visualization = cv2.cvtColor(img_no_shapes.copy(), cv2.COLOR_GRAY2BGR)
    for place in places:
        cv2.circle(nodes_visualization, (place.center.x, place.center.y), 
                  int(place.radius), (255, 0, 0), 2)
    for transition in transitions:
        cv2.drawContours(nodes_visualization, [transition.box_points.astype(np.int32)], 
                         -1, (0, 255, 0), 2)
    result["visualizations"]["nodes"] = Image.fromarray(nodes_visualization)
    
    # Step 6: Process lines and connections
    # Get Hough lines
    hough_lines = get_hough_lines(img_no_shapes, config)
    
    # Visualize skeletonized image with hough lines
    skeleton_img = skeletonize(img_no_shapes / 255).astype(np.uint8)*255
    hough_visualization = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
    for line in hough_lines:
        cv2.line(hough_visualization, 
                (line[0][0], line[0][1]), 
                (line[0][2], line[0][3]), 
                (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)), 2)
    result["visualizations"]["hough_lines"] = Image.fromarray(hough_visualization)
    
    # Process and merge Hough lines
    hough_bundler_min_distance = config.get('connection_processing', {}).get('hough_bundler_min_distance', 10)
    hough_bundler_min_angle = config.get('connection_processing', {}).get('hough_bundler_min_angle', 5)
    
    bundler = HoughBundler(min_distance=hough_bundler_min_distance, min_angle=hough_bundler_min_angle)
    merged_hough_lines = bundler.process_lines(hough_lines)
    lines = [Line(Point(line[0][0], line[0][1]), Point(line[0][2], line[0][3])) 
             for line in merged_hough_lines]
    
    # Visualize merged lines
    merged_lines_visualization = cv2.cvtColor(img_no_shapes.copy(), cv2.COLOR_GRAY2BGR)
    for line in lines:
        cv2.line(merged_lines_visualization, 
                (line.point1.x, line.point1.y), 
                (line.point2.x, line.point2.y), 
                (255, 0, 0), 2)
    result["visualizations"]["merged_lines"] = Image.fromarray(merged_lines_visualization)
    
    # Step 7: Assign proximity nodes and get entry points
    processed_lines, processed_places, processed_transitions = assign_proximity_nodes(
        lines, places, transitions, config
    )
    entry_points = get_entry_points_from_lines(processed_lines)

    # Visualize extended boxes used in proximity node assignment
    proximity_thres_place = config.get('connection_processing', {}).get('proximity_thres_place', 1.5)
    proximity_thres_trans_width = config.get('connection_processing', {}).get('proximity_thres_trans_width', 3)
    proximity_thres_trans_height = config.get('connection_processing', {}).get('proximity_thres_trans_height', 1.2)
    extended_boxes_visualization = cv2.cvtColor(img_no_shapes.copy(), cv2.COLOR_GRAY2BGR) # Convert to BGR for drawing
    for place in places: 
        cv2.circle(extended_boxes_visualization, (place.center.x, place.center.y), int(proximity_thres_place * place.radius), (0, 255, 0), 2) 
    for transition in transitions:
        expanded_height = transition.height * proximity_thres_trans_height
        expanded_width = transition.width * proximity_thres_trans_width
        expanded_bbox_contour = cv2.boxPoints(((float(transition.center.x), float(transition.center.y)),
                                                                    (expanded_height, expanded_width), transition.angle))
        cv2.drawContours(extended_boxes_visualization, [expanded_bbox_contour.astype(np.int32)], -1, (0, 255, 0), 2) # Draw transitions in green
    result["visualizations"]["extended_boxes"] = Image.fromarray(extended_boxes_visualization)
    
    # Visualize processed lines and entry points
    proximity_visualization = cv2.cvtColor(img_no_shapes.copy(), cv2.COLOR_GRAY2BGR)
    for line in processed_lines:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.line(proximity_visualization, 
                (line.point1.x, line.point1.y), 
                (line.point2.x, line.point2.y), 
                color, 2)
    for point in entry_points:
        cv2.circle(proximity_visualization, (point.x, point.y), 5, (0, 255, 0), -1)
    result["visualizations"]["proximity_nodes"] = Image.fromarray(proximity_visualization)
    
    # Step 8: Filter lines
    filtered_lines = []
    for line in processed_lines:
        if line.point1.proximity_node == line.point2.proximity_node != None:
            continue
        else:
            filtered_lines.append(line)
    
    # Visualize filtered lines
    filtered_lines_visualization = cv2.cvtColor(np.zeros_like(img_no_shapes), cv2.COLOR_GRAY2BGR)
    for line in filtered_lines:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.line(filtered_lines_visualization, 
                (line.point1.x, line.point1.y), 
                (line.point2.x, line.point2.y), 
                color, 2)
    result["visualizations"]["filtered_lines"] = Image.fromarray(filtered_lines_visualization)
    
    # Step 9: Find line paths
    found_paths_result = find_line_paths(
        filtered_lines,
        proximity_threshold=100.0,
        dot_product_weight=0.5,
        distance_to_line_weight=0.25,
        endpoint_distance_weight=0.25
    )
    
    # Visualize found paths
    paths_visualization = cv2.cvtColor(np.zeros_like(img_no_shapes), cv2.COLOR_GRAY2BGR)
    for path in found_paths_result:
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        for line in path["lines"]:
            cv2.line(paths_visualization, 
                    (line.point1.x, line.point1.y), 
                    (line.point2.x, line.point2.y), 
                    color, 2)
    result["visualizations"]["paths"] = Image.fromarray(paths_visualization)
    
    # Step 10: Detect arrowheads and assign to paths
    arrowhead_result = detect_arrowheads(image=img_color_resized, config=config)
    paths_with_arrows, rejected_arrows_count = assign_arrowheads(found_paths_result, arrowhead_result, config)
    
    # # Visualize paths with arrows
    arrows_visualization = img_color_resized.copy()
    detections = sv.Detections.from_inference(arrowhead_result)
    bounding_box_annotator = sv.BoxAnnotator()
    arrows_visualization = bounding_box_annotator.annotate(
        scene=arrows_visualization, detections=detections)
    result["visualizations"]["arrows_bboxes"] = Image.fromarray(arrows_visualization)
    
    # Step 11: Generate arcs
    arcs = get_arcs(paths_with_arrows)
    
    # Filter arcs to remove cycles with same source and target
    filtered_arcs = []
    for arc in arcs:
        if arc.source != arc.target and type(arc.source) != type(arc.target):
            filtered_arcs.append(arc)
    result["arcs"] = filtered_arcs
    
    # Visualize arcs
    arcs_visualization = cv2.cvtColor(np.zeros_like(img_no_shapes), cv2.COLOR_GRAY2RGB)
    for arc in filtered_arcs:
        src_color = (0, 0, 255)  # Red for source
        tgt_color = (255, 0, 0)  # Blue for target
        
        cv2.circle(arcs_visualization, (arc.start_point.x, arc.start_point.y), 5, src_color, -1)
        cv2.circle(arcs_visualization, (arc.end_point.x, arc.end_point.y), 5, tgt_color, -1)
        cv2.line(arcs_visualization, 
                (arc.start_point.x, arc.start_point.y), 
                (arc.end_point.x, arc.end_point.y), 
                (0, 255, 0), 2)
    result["visualizations"]["arcs"] = Image.fromarray(arcs_visualization)

    # link text to places and transitions
    link_text_to_elements(detected_text_list, places, transitions, filtered_arcs, config)
    # Check associations
    print("--- Associated Text ---")
    for p in places[:5]:
        if p.text: print(f"{p} has text: {[t.value for t in p.text]}")
    for t in transitions[:5]:
        if t.text: print(f"{t} has text: {[txt.value for txt in t.text]}") # changed t.text to txt.value
    for a in filtered_arcs[:5]:
        if a.text: print(f"{a} has text: {[t.value for t in a.text]}")

    # update attributes based on text
    for place in places:
        place.update_markers_from_text()
    for arc in filtered_arcs:
        arc.update_weight_from_text()

    
    return result

# Example usage:
if __name__ == "__main__":
    config_path = here("config.yaml")
    image_path = here("../data/local/mid_petri_2.png")
    result = recognize_graph(image_path, config_path)
    
    # Access the results
    places = result["places"]
    transitions = result["transitions"]
    arcs = result["arcs"]
    
    # Save or display visualizations
    output_dir = here("../data/output")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, img in result["visualizations"].items():
        img.save(f"{output_dir}/{name}.png")
        # Optional: display image
        # img.show()
    
    print(f"Recognition complete. Found {len(places)} places, {len(transitions)} transitions, and {len(arcs)} arcs.")


