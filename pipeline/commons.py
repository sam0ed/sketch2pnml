import numpy as np
import cv2
from .models import Line, Point
import inspect
import os

def here(resource: str):
    """Utils that given a relative path returns the corresponding absolute path, independently from the environment

    Parameters
    ----------
    resource: str
        The relative path of the given resource

    Returns
    -------
    str
        The absolute path of the give resource
    """
    stack = inspect.stack()
    caller_frame = stack[1][0]
    caller_module = inspect.getmodule(caller_frame)
    return os.path.abspath(
        os.path.join(os.path.dirname(caller_module.__file__), resource)
    )

def minmaxToContours(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)  # Convert to contour format

def fill_contours(preprocessed_img, contours):
    """Fills the contours in the preprocessed image."""
    img_filled = preprocessed_img.copy()
    
    cv2.drawContours(img_filled, contours, -1, (255), thickness=cv2.FILLED)
    return img_filled   

def remove_contours(preprocessed_img, contours):
    """Removes the detected node shapes (circles and rectangles) from the image."""

    contours_mask = np.zeros_like(preprocessed_img)
    contours_mask = fill_contours(contours_mask, contours)
    contours_mask = cv2.bitwise_not(contours_mask)

    img_no_contours = cv2.bitwise_and(preprocessed_img, contours_mask)
    return img_no_contours

def filter_enclosed_contours(
    contours_to_filter: list[np.ndarray],
    enclosing_contours_input: list[np.ndarray],
    include_border: bool = True
) -> list[np.ndarray]:
    """
    Filters contours from contours_to_filter that are entirely enclosed by ANY contour
    in enclosing_contours_input.

    Args:
        contours_to_filter: List of contours (np.ndarray) to be filtered.
                            Each contour is typically an array of shape (N, 1, 2).
        enclosing_contours_input: List of contours (np.ndarray) that act as enclosing shapes.
                                  Each contour is typically an array of shape (M, 1, 2).
        include_border: Whether points on the border of an enclosing_contour
                        count as inside.

    Returns:
        A list of contours from contours_to_filter that are fully enclosed.
        Each contour in the returned list is a reference to an object in 
        the input contours_to_filter list.
    """
    if not contours_to_filter or not enclosing_contours_input:
        return []

    # Filter out empty enclosing contours and precompute their bounding rects
    enclosing_contours = []
    brects_enclosing = []
    for c_enclosing in enclosing_contours_input:
        # Ensure contour has points (shape[0] is the number of points)
        if c_enclosing.shape[0] > 0:
            enclosing_contours.append(c_enclosing)
            brects_enclosing.append(cv2.boundingRect(c_enclosing))

    if not enclosing_contours: # No valid enclosing contours
        return []

    result_contours = []
    # Keep track of indices of contours from contours_to_filter that have been added
    added_contour_indices = set()

    test_threshold = 0 if include_border else 1

    # Precompute bounding rects for contours_to_filter
    # (x, y, w, h)
    brects_to_filter = []
    for c in contours_to_filter:
        if c.shape[0] > 0:
            brects_to_filter.append(cv2.boundingRect(c))
        else:
            # Placeholder for empty contours, they will be skipped later
            brects_to_filter.append((0,0,0,0))


    for idx1, contour1 in enumerate(contours_to_filter):
        if idx1 in added_contour_indices:
            continue

        # An empty contour (no points) cannot be considered enclosed
        if contour1.shape[0] == 0:
            continue

        x1, y1, w1, h1 = brects_to_filter[idx1]
        is_contour1_enclosed_by_any = False

        for idx2, contour2 in enumerate(enclosing_contours):
            x2, y2, w2, h2 = brects_enclosing[idx2]

            # AABB Pruning: For contour1 to be enclosed by contour2,
            # contour1's bounding box must be within contour2's bounding box.
            if not (x1 >= x2 and \
                    y1 >= y2 and \
                    (x1 + w1) <= (x2 + w2) and \
                    (y1 + h1) <= (y2 + h2)):
                continue # Bounding box of contour1 is not contained in bounding box of contour2

            # Precise point-in-polygon test:
            # All points of contour1 must be inside (or on border of) contour2
            all_points_inside = True
            # Reshape contour1 from (N,1,2) to (N,2) for easier iteration
            points_contour1 = contour1.reshape(-1, 2)

            for pt_x, pt_y in points_contour1:
                # cv2.pointPolygonTest expects point as (float, float)
                dist = cv2.pointPolygonTest(contour2, (float(pt_x), float(pt_y)), False)
                if dist < test_threshold:
                    all_points_inside = False
                    break # This point of contour1 is outside contour2

            if all_points_inside:
                is_contour1_enclosed_by_any = True
                break # contour1 is enclosed by contour2; no need to check other enclosing_contours

        if is_contour1_enclosed_by_any:
            result_contours.append(contour1)
            added_contour_indices.add(idx1)

    return result_contours

def dilate_contour(contour, image_shape, config):
    if contour is None or len(contour) == 0:
        raise ValueError("Invalid contour provided for dilation.")
    
    dilation_kernel_size = config.get('shape_detection', {}).get('remove_nodes_dilation_kernel_size', [3, 3])
    dilation_iterations = config.get('shape_detection', {}).get('remove_nodes_dilation_iterations', 3)


    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    kernel = np.ones((dilation_kernel_size[0], dilation_kernel_size[1]), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if dilated_contours:
        largest_dilated_contour = max(dilated_contours, key=cv2.contourArea)
        return largest_dilated_contour
    else:
        raise ValueError("No contours found after dilation.")
    
def find_closest_distance_to_contour(point_obj: Point, contour_np: np.ndarray) -> float:
    """
    Calculates the shortest distance from a Point object to a contour.
    The contour is a numpy array of points.
    - If contour has 1 point, it's point-to-point distance.
    - If contour has 2 points, it's distance to the line segment defined by these points.
    - If contour has >2 points, it's distance to the boundary of the polygon defined by these points.

    Args:
        point_obj: The Point object from which to measure the distance.
        contour_np: A NumPy array representing the contour, shape (N, 1, 2) or (N, 2).
                    Coordinates are typically integers.

    Returns:
        The shortest distance as a float.
    """

    # Validate and standardize contour_np shape for point extraction
    if contour_np.ndim == 3 and contour_np.shape[1] == 1 and contour_np.shape[2] == 2:
        # Shape (N, 1, 2), reshape to (N, 2) for easier iteration
        processed_contour_points = contour_np.reshape(-1, 2)
    elif contour_np.ndim == 2 and contour_np.shape[1] == 2:
        # Shape (N, 2), use as is
        processed_contour_points = contour_np
    else:
        raise ValueError(f"Contour numpy array has an unsupported shape: {contour_np.shape}. "
                         "Expected (N, 1, 2) or (N, 2).")

    num_contour_points = processed_contour_points.shape[0]

    if num_contour_points == 0:
        return float('inf') # No points in contour, distance is infinite

    if num_contour_points == 1:
        contour_pt_coords = processed_contour_points[0]
        contour_pt_obj = Point(contour_pt_coords[0], contour_pt_coords[1])
        return point_obj.get_distance_between_points(contour_pt_obj)

    if num_contour_points == 2:
        pt_a_coords = processed_contour_points[0]
        pt_b_coords = processed_contour_points[1]
        
        # Create Point objects for the segment endpoints
        segment_pt_a = Point(pt_a_coords[0], pt_a_coords[1])
        segment_pt_b = Point(pt_b_coords[0], pt_b_coords[1])
        
        line_segment = Line(segment_pt_a, segment_pt_b)
        return line_segment.distance_point_to_segment(point_obj)
    
    # num_contour_points > 2 (Polygon case)
    else:
        # cv2.pointPolygonTest requires contour in (N, 1, 2) format and float32 type.
        # We use the original contour_np for this, as it might already be (N,1,2).
        if contour_np.ndim == 2: # Original was (N,2)
            contour_for_cv2 = contour_np.reshape((-1, 1, 2)).astype(np.float32)
        else: # Original was (N,1,2)
            contour_for_cv2 = contour_np.astype(np.float32)
            
        # The query point for pointPolygonTest needs to be a float tuple
        query_point_tuple = (float(point_obj.x), float(point_obj.y))
        
        # measureDist=True returns signed distance:
        # The absolute value is the shortest distance to any edge of the contour.
        distance = cv2.pointPolygonTest(contour_for_cv2, query_point_tuple, True)
        return abs(distance)
