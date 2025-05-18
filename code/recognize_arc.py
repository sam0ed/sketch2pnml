import math 
from skimage.morphology import skeletonize
import copy 
import numpy as np
import cv2
from models import Line, Point, Place, Transition, Arc

def get_hough_lines(img, config):
    """
    Detects lines in the image using Hough Transform.
    Returns the detected lines.
    """
    # Default values, will be overridden by config if available
    rho = config.get('connection_processing', {}).get('hough_rho', 1)
    theta = config.get('connection_processing', {}).get('hough_theta', np.pi / 180)
    threshold = config.get('connection_processing', {}).get('hough_threshold', 10)
    min_line_length = config.get('connection_processing', {}).get('min_line_length', 10)
    max_line_gap = config.get('connection_processing', {}).get('max_line_gap', 20)
    min_line_length = max(min_line_length, 1)  # Ensure it's at least 1
    

        # Skeletonize the image
    skeleton = skeletonize(img / 255).astype(np.uint8)*255
    hough_lines = cv2.HoughLinesP(skeleton, rho, np.pi/180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return hough_lines

class HoughBundler:     
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle
    
    def get_orientation(self, line):
        orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)
    
def assign_proximity_nodes(
    original_lines: list[Line], 
    original_places: list[Place], 
    original_transitions: list[Transition], 
    config: dict,
) -> tuple[list[Line], list[Place], list[Transition]]:
    """
    Assigns a 'proximity_node' attribute to points of lines if they are close
    to a place or transition. Operates on deep copies of the input objects.

    Args:
        original_lines: A list of Line objects.
        original_places: A list of Place objects.
        original_transitions: A list of Transition objects.
        proximity_threshold: A factor to expand node boundaries for proximity checks.
                           For Places, it scales the radius.
                           For Transitions, it scales the height and width.

    Returns:
        A tuple containing:
        - lines_copy: Copied lines, where points may have a 'proximity_node' attribute.
        - places_copy: Deep copies of the original places.
        - transitions_copy: Deep copies of the original transitions.
        The 'proximity_node' attributes will refer to objects within places_copy or transitions_copy.
    """
    proximity_thres_place = config.get('connection_processing', {}).get('proximity_thres_place', 1.5)
    proximity_thres_trans_width = config.get('connection_processing', {}).get('proximity_thres_trans_width', 3)
    proximity_thres_trans_height = config.get('connection_processing', {}).get('proximity_thres_trans_height', 1.2)

    # 1. Create deep copies of all input object lists
    lines_copy = copy.deepcopy(original_lines)
    places_copy = copy.deepcopy(original_places)
    transitions_copy = copy.deepcopy(original_transitions)

    all_copied_node_centers = [node.center for node in places_copy] + \
                              [node.center for node in transitions_copy]

    # 3. Iterate through copied lines and their points
    for line in lines_copy:
        for line_point in [line.point1, line.point2]:
            for node_center_copy in all_copied_node_centers:
                node_copy = node_center_copy.part_of 
                # print(f"Checking proximity for line point {line_point} to node {node_copy}")

                if isinstance(node_copy, Place):
                    # print(f"Node {node_copy} is a place")
                    distance = line_point.get_distance_between_points(node_center_copy)
                    if distance < proximity_thres_place * node_copy.radius:
                        line_point.proximity_node = node_copy
                
                elif isinstance(node_copy, Transition):
                    # print(f"Node {node_copy} is a transition")
                    expanded_height = node_copy.height * proximity_thres_trans_height
                    expanded_width = node_copy.width * proximity_thres_trans_width
                    
                    
                    expanded_bbox_contour = cv2.boxPoints(((float(node_center_copy.x), float(node_center_copy.y)),
                                                            (expanded_height, expanded_width), node_copy.angle))
                    current_line_point_coords = (float(line_point.x), float(line_point.y))
                    
                    if cv2.pointPolygonTest(expanded_bbox_contour, current_line_point_coords, False) >= 0:
                        line_point.proximity_node = node_copy

                else:
                    print(f"Node {node_copy} is not a recognized type")
                    # This case should ideally not be reached if inputs are as expected.
                    raise ValueError(f"Unknown node type encountered: {type(node_copy)}")
    
    return lines_copy, places_copy, transitions_copy

def get_entry_points_from_lines(lines_list):
    """
    Original function provided by user, slightly adapted to use a local list.
    Extracts all unique points marked as 'is_entry' from a list of lines.
    """
    entry_points_set = set()
    for line in lines_list:
        if hasattr(line.point1, "proximity_node") and line.point1.proximity_node:
            entry_points_set.add(line.point1)
        if hasattr(line.point2, "proximity_node") and line.point2.proximity_node:
            entry_points_set.add(line.point2)
    return list(entry_points_set)


def cosine_similarity(vec1_norm: np.ndarray, vec2_norm: np.ndarray) -> float:
    """Computes the cosine similarity (dot product of normalized vectors)."""
    return np.dot(vec1_norm, vec2_norm)

def find_line_paths(
    initial_lines_list: list[Line],
    proximity_threshold: float = 30.0,
    dot_product_weight: float = 0.6,
    distance_to_line_weight: float = 0.2,
    endpoint_distance_weight: float = 0.2
) -> list[dict]:
    """
    Connects lines from a pool into paths, starting from an entry point
    and ending at another entry point.

    Args:
        initial_lines_list: A list of Line objects.
        proximity_threshold: Maximum distance to search for next point.
        dot_product_weight: Weight for vector alignment score.
        distance_to_line_weight: Weight for point-to-line distance score.
        endpoint_distance_weight: Weight for endpoint-to-endpoint distance score.

    Returns:
        A list of paths. Each path is a dictionary with 'lines' (list of Line)
        and 'points' (ordered list of Point forming the path).
    """
    lines_pool = set(initial_lines_list) # Use a set for efficient removal (O(1) on average)
    all_paths_found = []
    
    # Keep track of entry points that have successfully started a path to avoid re-processing
    # or entry points that have been used as an end of a path.
    consumed_entry_points = set()

    while True:
        current_start_line = None
        current_start_entry_point = None

        # Find a new starting line with an available entry point
        # Iterate over a temporary list as lines_pool can be modified
        for line in list(lines_pool):
            potential_start_points = []
            if hasattr(line.point1, "proximity_node") and line.point1.proximity_node and line.point1 not in consumed_entry_points:
                potential_start_points.append(line.point1)
            if hasattr(line.point2, "proximity_node") and line.point2.proximity_node and line.point2 not in consumed_entry_points:
                potential_start_points.append(line.point2)
            
            if potential_start_points:
                current_start_line = line
                # Prefer point1 if both are entries and available, or just take the first one.
                current_start_entry_point = potential_start_points[0]
                break
        
        if not current_start_line:
            break # No more available entry points or lines to start a path

        current_path_lines = [current_start_line]
        current_path_points = [current_start_entry_point]
        
        lines_pool.remove(current_start_line)
        consumed_entry_points.add(current_start_entry_point) # Mark this entry point as used for path initiation

        last_line_in_path = current_start_line
        # The current tip of the path is the other point of the start_line
        current_tip_of_path = last_line_in_path.get_other_point(current_start_entry_point)
        current_path_points.append(current_tip_of_path)

        # Inner loop to extend the current path
        while True:
            # Check if the current_tip_of_path is a destination entry point
            if hasattr(current_tip_of_path, "proximity_node") and current_tip_of_path.proximity_node:
                all_paths_found.append({"lines": list(current_path_lines), "points": list(current_path_points)})
                consumed_entry_points.add(current_tip_of_path) # Mark end entry point
                break # Path successfully found, break from inner loop

            candidate_extensions = []
            # Vector of the last segment, oriented towards the current tip
            vec_last_segment_norm = last_line_in_path.get_normalized_vector(
                start_point=last_line_in_path.get_other_point(current_tip_of_path),
                end_point=current_tip_of_path
            )

            for candidate_line in list(lines_pool): # Iterate over a copy of the pool for safe removal
                for point_on_candidate in [candidate_line.point1, candidate_line.point2]:
                    # Must not connect via an intermediate entry point
                    if hasattr(point_on_candidate, "proximity_node") and point_on_candidate.proximity_node:
                        continue

                    endpoint_dist = current_tip_of_path.get_distance_between_points(point_on_candidate)

                    if endpoint_dist <= proximity_threshold:
                        # Scoring Criterion 1: Dot product of normalized vectors
                        # Vector of candidate_line, oriented away from point_on_candidate
                        vec_candidate_norm = candidate_line.get_normalized_vector(
                            start_point=point_on_candidate,
                            end_point=candidate_line.get_other_point(point_on_candidate)
                        )
                        dot_prod_score = (cosine_similarity(vec_last_segment_norm, vec_candidate_norm) + 1) / 2 # Scale to [0,1]

                        # Scoring Criterion 2: Start point of "to be merged" line is close to the infinite line
                        # formed by our last merged line.
                        dist_to_prev_line_inf = last_line_in_path.distance_point_to_infinite_line(point_on_candidate)
                        # Score: higher is better (closer to 0 distance)
                        # Avoid division by zero; add 1. Max possible distance could normalize this.
                        # For simplicity, using 1 / (1 + dist).
                        dist_line_score = 1.0 / (1.0 + dist_to_prev_line_inf) if proximity_threshold > 0 else 1.0


                        # Bonus: endpoint_distance score (closer is better)
                        endpoint_dist_score = (proximity_threshold - endpoint_dist) / proximity_threshold \
                                              if proximity_threshold > 0 else 1.0
                        
                        # Combined score
                        total_score = (dot_product_weight * dot_prod_score +
                                       distance_to_line_weight * dist_line_score +
                                       endpoint_distance_weight * endpoint_dist_score)
                        
                        candidate_extensions.append({
                            "line": candidate_line,
                            "connection_point_on_candidate": point_on_candidate,
                            "score": total_score
                        })
            
            if not candidate_extensions:
                # No suitable extension found, path terminates here (not at an entry point).
                # This path is considered "noise" or incomplete.
                break # Break from inner loop

            # Select the best candidate extension
            candidate_extensions.sort(key=lambda x: x["score"], reverse=True)
            best_extension = candidate_extensions[0]

            # Add best extension to the current path
            lines_pool.remove(best_extension["line"]) # Remove from available lines
            current_path_lines.append(best_extension["line"])
            
            last_line_in_path = best_extension["line"]
            # The connection point on the candidate becomes part of the path
            current_path_points.append(best_extension["connection_point_on_candidate"])
            # The new tip is the other end of the newly added line
            current_tip_of_path = last_line_in_path.get_other_point(best_extension["connection_point_on_candidate"])
            current_path_points.append(current_tip_of_path)
            # Continue extending this path

    return all_paths_found

def assign_arrowheads(found_paths_original: list[Line], arrowhead_result: dict, config) -> tuple[list[Line], int]:
    arrowhead_proximity_thres = config.get('connection_processing', {}).get('arrowhead_proximity_threshold', 30)

    found_paths_copy = copy.deepcopy(found_paths_original)
    path_endpoints = []
    for path in found_paths_copy:
        if path["points"]: 
            path_endpoints.extend([path["points"][0], path["points"][-1]])
        else:
            raise ValueError("Path points list is empty. Cannot assign arrowheads.")
        
    rejected_arrowhead_count = 0

    for arrowhead in arrowhead_result["predictions"]:
        arrowhead_center = Point(arrowhead["x"], arrowhead["y"])
        
        closest_point = None
        closest_distance = float("inf")
        for endpoint in path_endpoints:
            distance = arrowhead_center.get_distance_between_points(endpoint)
            if distance < closest_distance and distance < arrowhead_proximity_thres:
                closest_distance = distance
                closest_point = endpoint
                ### remove endpoint from endpoints to avoid reusing
                path_endpoints.remove(endpoint)
        
        ### check if the closest point is None and throw error
        if closest_point is None:
            print("No closest point found for the arrowhead center.")
            rejected_arrowhead_count += 1
        else: 
            closest_point.is_arrow = True

    return found_paths_copy, rejected_arrowhead_count

def get_arcs(paths):
    """
    Links the nodes of the paths based on the proximity_node attribute of the points.
    This function assumes that the paths are already processed and contain points with proximity_node.
    """

    arcs = []

    for path in paths:
        if not path["points"][0].proximity_node or not path["points"][-1].proximity_node:
            raise ValueError("Path must start and end with a proximity node.")
        if len(path["points"]) < 2:
            raise ValueError("Path must contain at least two points.")
        if len(path["lines"]) < 1:
            raise ValueError("Path must contain at least one line.")
        # Assuming a path of N points connected sequentially has N-1 lines.
        if len(path["points"]) != len(path["lines"]) * 2:
             raise ValueError("Path points and lines are inconsistent.")

        start_point = path["points"][0]
        end_point = path["points"][-1]

        # Add arc from start to end unless start is an arrow and end is not
        if not (start_point.is_arrow and not end_point.is_arrow):
            arcs.append(Arc(
                source=start_point.proximity_node,
                target=end_point.proximity_node,
                start_point=start_point,
                end_point=end_point,
                points=path["points"],
                lines=path["lines"]
            ))

        # Add arc from end to start if start is an arrow
        if start_point.is_arrow:
             arcs.append(Arc(
                source=end_point.proximity_node,
                target=start_point.proximity_node,
                start_point=end_point,
                end_point=start_point,
                points=path["points"],
                lines=path["lines"]
            ))

    return arcs
