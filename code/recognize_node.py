import cv2
import numpy as np
import largestinteriorrectangle as lir
from constants import EPS

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
            print(f"Rectangle detected: rectangle_overlap_percentage: {rectangle_overlap_percentage} circle_overlap_percentage: {circle_overlap_percentage}")
            rectangles.append(contour)

    return circles, rectangles


def fill_contours(preprocessed_img, contours, config):
    """Fills the contours in the preprocessed image."""
    img_filled = preprocessed_img.copy()
    
    cv2.drawContours(img_filled, contours, -1, (255), thickness=cv2.FILLED)
    return img_filled   


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

def remove_contours(preprocessed_img, contours, config):
    """Removes the detected node shapes (circles and rectangles) from the image."""

    contours_mask = np.zeros_like(preprocessed_img)
    contours_mask = fill_contours(contours_mask, contours, config)
    contours_mask = cv2.bitwise_not(contours_mask)

    img_no_contours = cv2.bitwise_and(preprocessed_img, contours_mask)
    return img_no_contours



# def is_circle_enclosing(contour, threshold):
#     """Checks if a contour is roughly circular based on the ratio of its area to its minimum enclosing circle."""
#     (x, y), radius = cv2.minEnclosingCircle(contour)
#     enclosing_area = np.pi * (radius ** 2) + EPS
#     contour_area = cv2.contourArea(contour)
    
#     return contour_area / enclosing_area > threshold 

# def is_rectangle_enclosing(contour, threshold):
#     """Checks if a contour is roughly rectangular based on the ratio of its area to its minimum area bounding box."""
#     rect = cv2.minAreaRect(contour)
#     box_area = rect[1][0] * rect[1][1] + EPS
#     contour_area = cv2.contourArea(contour)
    
#     if box_area == 0:
#         return False
        
#     return contour_area / box_area > threshold


# def fill_empty_nodes(preprocessed_img, config):
#     """
#     Identifies potential "empty" nodes (thin circles or rectangles) and fills them.
#     Note: Detection logic might be refined later (see TODO).
#     """
#     # Default values, will be overridden by config if available
#     circle_threshold = config.get('shape_detection', {}).get('fill_circle_enclosing_threshold', 0.8)
#     rect_threshold = config.get('shape_detection', {}).get('fill_rect_enclosing_threshold', 0.95)
    
#     contours_list, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 


#     ### TODO: Rewrite to detect based on inscribed circle and rectangle and put these parameters in user parameters
#     # Filter for circular contours
#     contours_of_interest = []
#     for i, contour in enumerate(contours_list):
#         if is_circle_enclosing(contour, circle_threshold) or is_rectangle_enclosing(contour, rect_threshold):
#             contours_of_interest.append(contour)

#     ### add 255 pixels inside the found contours on the preprocessed_img
#     img_empty_nodes_filled = preprocessed_img.copy()
#     for contour in contours_of_interest:
#         cv2.drawContours(img_empty_nodes_filled, [contour], -1, (255), thickness=cv2.FILLED)

#     return img_empty_nodes_filled


# def detect_shapes(nodes_mask, config):
#     """
#     Detects Places (circles) and Transitions (rectangles) within the isolated node mask
#     by finding the largest inscribed circle and rectangle in each connected component (blob).
#     """
#     # Default values, will be overridden by config if available
#     circle_overlap_threshold = config.get('shape_detection', {}).get('classify_circle_overlap_threshold', 0.8)
#     rect_overlap_threshold = config.get('shape_detection', {}).get('classify_rect_overlap_threshold', 0.85) # Note: original code used 0.85 here
#     verbose = config.get('shape_detection', {}).get('verbose', True) # Control printing

#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(nodes_mask, connectivity=8)
#     blob_masks = []

#     for label in range(1, num_labels):
#         blob_mask = np.zeros_like(nodes_mask, dtype=np.uint8)
#         blob_mask[labels == label] = 255
#         blob_masks.append(blob_mask)


#     circles = []
#     rectangles = []
#     for i, blob_mask in enumerate(blob_masks):
#         ### Find inscribed circles
#         dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) # Using precise mask
#         NULL,max_val,NULL,max_indx=cv2.minMaxLoc(dist)

#         (x,y),radius = max_indx, int(max_val)

#         circle_mask = np.zeros_like(blob_mask)
#         cv2.circle(circle_mask, (x,y), radius, (255), thickness=cv2.FILLED)

#         ### find the overlap between the area of the circle and the area of the blob_mask from 0 to 1
#         overlap = cv2.bitwise_and(blob_mask, circle_mask)
#         overlap_area = cv2.countNonZero(overlap)
#         blob_area = cv2.countNonZero(blob_mask)
#         circle_overlap_percentage = (overlap_area / blob_area) if blob_area > 0 else 0
#         if verbose: print(f"Blob {i}: Overlap percentage between inscribed circle and mask: {circle_overlap_percentage:.3f}")




#         bool_blob_mask = blob_mask.astype(np.bool)
#         rectangle = lir.lir(bool_blob_mask)

#         rectangle_mask = np.zeros_like(blob_mask)
#         cv2.rectangle(rectangle_mask, lir.pt1(rectangle), lir.pt2(rectangle), (255), thickness=cv2.FILLED)

#         ### find the overlap between the area of the rectangle and the area of the blob_mask from 0 to 1
#         overlap = cv2.bitwise_and(blob_mask, rectangle_mask)
#         overlap_area = cv2.countNonZero(overlap)
#         blob_area = cv2.countNonZero(blob_mask)
#         rectangle_overlap_percentage = (overlap_area / blob_area) if blob_area > 0 else 0
#         if verbose: print(f"Blob {i}: Overlap percentage between inscribed rectangle and mask: {rectangle_overlap_percentage:.3f}")


#         # Classify based on which inscribed shape has better overlap, above its threshold
#         # TODO: Consider tie-breaking scenarios or alternative classification logic?
#         if circle_overlap_percentage > rectangle_overlap_percentage and circle_overlap_percentage > circle_overlap_threshold:
#             # Removed drawing code: cv2.circle(img, (x,y), radius, (0,255,0), 2)
#             circles.append((x,y,radius))
#             if verbose: print(f"Blob {i}: Classified as CIRCLE (Place)")

#         elif rectangle_overlap_percentage > circle_overlap_percentage and rectangle_overlap_percentage > rect_overlap_threshold:
#             # Removed drawing code: cv2.rectangle(img, lir.pt1(rectangle), lir.pt2(rectangle), (0,0,255), 2)
#             rectangles.append(rectangle)
#             if verbose: print(f"Blob {i}: Classified as RECTANGLE (Transition)")
#         else:
#             if verbose: print(f"Blob {i}: Not classified (Circle overlap {circle_overlap_percentage:.3f} <= {circle_overlap_threshold} or Rect overlap {rectangle_overlap_percentage:.3f} <= {rect_overlap_threshold} or overlaps equal)")

#     return circles, rectangles

# def remove_nodes(preprocessed_img, circles, rectangles, config):
#     """Removes the detected node shapes (circles and rectangles) from the image."""
#     # Default values, will be overridden by config if available
#     dilation_kernel_size = tuple(config.get('shape_detection', {}).get('remove_nodes_dilation_kernel_size', [3, 3]))
#     dilation_iterations = config.get('shape_detection', {}).get('remove_nodes_dilation_iterations', 3)
    
#     shapes_mask = np.zeros_like(preprocessed_img)
#     for circle in circles:
#         cv2.circle(shapes_mask, circle[:2], circle[2], (255), thickness=cv2.FILLED)

#     for rectangle in rectangles:
#         cv2.rectangle(shapes_mask, lir.pt1(rectangle), lir.pt2(rectangle), (255), thickness=cv2.FILLED)

#     ## dialate the shapes_mask by 10 pixels
#     kernel = np.ones(dilation_kernel_size, np.uint8)
#     shapes_mask = cv2.dilate(shapes_mask, kernel, iterations=dilation_iterations)
#     shapes_mask = cv2.bitwise_not(shapes_mask)

#     img_no_shapes = cv2.bitwise_and(preprocessed_img, preprocessed_img, mask=shapes_mask)
#     return img_no_shapes


### testing contours code:
    ## draw each contour with a different random color
    # for i, contour in enumerate(contours_list):
    #     cv2.drawContours(img, [contour], -1, (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)), 1)
    # Image.fromarray(img).show()

    # print("Contours list length: ", len(contours_list))
    # for i, contour in enumerate(contours_list):
    #     rect = cv2.minAreaRect(contour)
    #     box_area = rect[1][0] * rect[1][1] + EPS
    #     contour_area = cv2.contourArea(contour)
        
    #     print(f"Contour {i} is {contour_area / box_area } rectangle")

    # # draw all contours on separate images and show them

    # contour = contours_list[6]
    # contour_img = np.zeros_like(img)
    # cv2.drawContours(contour_img, [contour], -1, (255, 255, 255), 1)
    # Image.fromarray(contour_img).show()

    # contour_img = np.zeros_like(img)
    # cv2.drawContours(contour_img, contours_of_interest, -1, (255, 255, 255), 1)
    # Image.fromarray(contour_img).show()
