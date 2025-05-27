import cv2
import numpy as np

EPS = 1e-6 


def get_circle_overlap(contour):
    """Checks if a contour is roughly circular based on the ratio of its area to its minimum enclosing circle."""
    (x, y), radius = cv2.minEnclosingCircle(contour)
    enclosing_area = np.pi * (radius ** 2) + EPS
    contour_area = cv2.contourArea(contour)
    
    return contour_area / enclosing_area 

def get_rectangle_overlap(contour):
    """Checks if a contour is roughly rectangular based on the ratio of its area to its minimum area bounding box."""
    rect = cv2.minAreaRect(contour)
    box_area = rect[1][0] * rect[1][1] + EPS
    contour_area = cv2.contourArea(contour)
        
    return contour_area / box_area 

def detect_shapes(preprocessed_img, config):

    circle_threshold = config.get('shape_detection', {}).get('fill_circle_enclosing_threshold', 0.8)
    rect_threshold = config.get('shape_detection', {}).get('fill_rect_enclosing_threshold', 0.95)
    
    contours_list, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    circles = []
    rectangles = []
    for contour in contours_list:
        circle_overlap_percentage = get_circle_overlap(contour)
        rectangle_overlap_percentage = get_rectangle_overlap(contour)

        if circle_overlap_percentage > rectangle_overlap_percentage and circle_overlap_percentage > circle_threshold:
            circles.append(contour)
            
        elif rectangle_overlap_percentage > circle_overlap_percentage and rectangle_overlap_percentage > rect_threshold:
            # print(f"Rectangle detected: rectangle_overlap_percentage: {rectangle_overlap_percentage} circle_overlap_percentage: {circle_overlap_percentage}")
            rectangles.append(contour)

    return circles, rectangles

def get_nodes_mask(img_empty_nodes_filled, config):
    """
    Isolates node structures using an iterative erosion/dilation heuristic based on contour count stability.
    """
    # Default values, will be overridden by config if available
    erosion_kernel_size = tuple(config.get('shape_detection', {}).get('erosion_kernel_size', [3, 3]))
    min_stable_length = config.get('shape_detection', {}).get('min_stable_length', 3)
    max_erosion_iterations = config.get('shape_detection', {}).get('max_erosion_iterations', 30)
    verbose = config.get('shape_detection', {}).get('verbose', True) # Control printing

    erosion_kernel = np.ones(erosion_kernel_size, np.uint8)
    contour_counts_history = []
    optimal_erosion_iterations = 0  # Default if loop doesn't run or no erosions found necessary
    optimal_condition_found = False # Flag to indicate if stability or zero contours was met

    # This image is progressively eroded to find the optimal number of iterations
    image_for_iterative_erosion = img_empty_nodes_filled.copy()

    # Loop to determine the optimal number of erosion iterations
    # If max_erosion_iterations is 0, this loop won't execute, and optimal_erosion_iterations will remain 0.
    for current_iteration in range(1, max_erosion_iterations + 1):
        # Perform one erosion step
        eroded_this_step = cv2.erode(image_for_iterative_erosion, erosion_kernel, iterations=1)
        contours, _ = cv2.findContours(eroded_this_step, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours_at_step = len(contours)
        contour_counts_history.append(num_contours_at_step)
        
        image_for_iterative_erosion = eroded_this_step # Update for the next iteration

        # Check for stability in contour count
        if len(contour_counts_history) >= min_stable_length:
            last_n_counts = contour_counts_history[-min_stable_length:]
            if all(c == last_n_counts[0] for c in last_n_counts):
                # Optimal iterations = iteration count at the start of the stable sequence
                optimal_erosion_iterations = current_iteration - min_stable_length + 1
                if verbose:
                    print(f"Stability detected: Contour count {last_n_counts[0]} stable for {min_stable_length} iterations.")
                    print(f"Optimal number of erosions determined as: {optimal_erosion_iterations}.")
                optimal_condition_found = True
                break 

        # Check if all contours have disappeared
        if num_contours_at_step == 0:
            if not optimal_condition_found: # Only set if stability wasn't the primary reason
                optimal_erosion_iterations = current_iteration # All contours gone after this many erosions
                if verbose:
                    print(f"All contours disappeared after {current_iteration} erosions.")
                    print(f"Optimal number of erosions determined as: {optimal_erosion_iterations}.")
            optimal_condition_found = True # This is a definitive condition to stop
            break
    # Loop ends

    # If the loop completed fully (max_erosion_iterations reached) without finding stability or zero contours
    if not optimal_condition_found and max_erosion_iterations > 0:
        optimal_erosion_iterations = max_erosion_iterations
        if verbose:
            print(f"Max erosions ({max_erosion_iterations}) reached without specific stability/zero-contour condition. "
                  f"Using {optimal_erosion_iterations} erosions.")

    # Obtain the node mask by applying the optimal number of erosions to the original filled image
    if optimal_erosion_iterations > 0:
        node_mask_eroded = cv2.erode(img_empty_nodes_filled, erosion_kernel, iterations=optimal_erosion_iterations)
        if verbose:
            print(f"Applied {optimal_erosion_iterations} erosions to input image to get the node mask.")
    else:
        # If no erosions are optimal, return a copy of the input to maintain consistency (always a new image object)
        node_mask_eroded = img_empty_nodes_filled.copy()
        if verbose:
            print("No erosions applied for node mask (0 optimal erosions). Using a copy of input.")

    # Dilate the eroded node mask to recover node sizes
    if optimal_erosion_iterations > 0:
        # Dilate by the same number of steps and with the same kernel
        dilated_node_mask = cv2.dilate(node_mask_eroded, erosion_kernel, iterations=optimal_erosion_iterations)
        if verbose:
            print(f"Applied {optimal_erosion_iterations} dilations to recover node sizes.")
    else:
        # If no erosions were done, no dilations are needed either.
        # node_mask_eroded is already a copy of the original (or the optimally eroded one if erosions > 0).
        dilated_node_mask = node_mask_eroded 
        if verbose:
            print("No dilations applied (0 optimal erosions).")

    if verbose: # Print history only if verbose mode is on
        print(f"Contour counts per erosion iteration: {contour_counts_history}")

    return dilated_node_mask