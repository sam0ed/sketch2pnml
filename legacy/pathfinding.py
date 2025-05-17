import math
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = int(x) # Ensure integer coordinates if they represent pixels
        self.y = int(y)
        self.is_entry = False # Default, can be set dynamically later

    def get_distance_between_points(self, other_point):
        """Calculate Euclidean distance between this point and another point."""
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        """Allows Point objects to be added to sets or used as dictionary keys."""
        return hash((self.x, self.y))

class Line:
    def __init__(self, start_point, end_point, angle=None, length=None):
        """
        Initializes a Line object.
        If angle and length are not provided, they are calculated.
        """
        self.point1 = start_point
        self.point2 = end_point

        if angle is None or length is None:
            dx = self.point2.x - self.point1.x
            dy = self.point2.y - self.point1.y
            # Calculate angle in degrees
            self.angle = math.degrees(math.atan2(dy, dx)) if not (dx == 0 and dy == 0) else 0.0
            # Calculate length
            self.length = self.point1.get_distance_between_points(self.point2)
        else:
            self.angle = angle
            self.length = length

    def get_other_point(self, point: Point) -> Point:
        """Given one point of the line, returns the other point."""
        if point == self.point1:
            return self.point2
        elif point == self.point2:
            return self.point1
        else:
            # This case should ideally not be reached if logic is correct
            raise ValueError("Point is not part of this line.")

    def get_vector(self, start_point: Point = None, end_point: Point = None) -> np.ndarray:
        """
        Returns the vector of the line.
        If start_point and end_point are provided, computes vector from start to end.
        Otherwise, defaults to point1 -> point2.
        """
        if start_point and end_point:
            return np.array([end_point.x - start_point.x, end_point.y - start_point.y])
        return np.array([self.point2.x - self.point1.x, self.point2.y - self.point1.y])

    def get_normalized_vector(self, start_point: Point = None, end_point: Point = None) -> np.ndarray:
        """Returns the normalized (unit) vector of the line."""
        vec = self.get_vector(start_point, end_point)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.array([0, 0]) # Represents a zero-length line segment
        return vec / norm

    def distance_point_to_infinite_line(self, point: Point) -> float:
        """
        Calculates the perpendicular distance from a point to the infinite line
        defined by this line segment.
        """
        p1_np = np.array([self.point1.x, self.point1.y])
        p2_np = np.array([self.point2.x, self.point2.y])
        p3_np = np.array([point.x, point.y])

        if np.array_equal(p1_np, p2_np): # If the line is just a point
            return np.linalg.norm(p3_np - p1_np)
        # Formula for distance from point to line (given by two points)
        return np.abs(np.cross(p2_np - p1_np, p1_np - p3_np)) / np.linalg.norm(p2_np - p1_np)

    def __repr__(self):
        return (f"Line(start={self.point1}, end={self.point2}, "
                f"angle={self.angle:.2f}, length={self.length:.2f})")

    def __eq__(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        # A line is considered equal if its endpoints are the same, regardless of order.
        return (self.point1 == other.point1 and self.point2 == other.point2) or \
               (self.point1 == other.point2 and self.point2 == other.point1)

    def __hash__(self):
        """Allows Line objects to be added to sets. The hash is order-invariant for points."""
        return hash(tuple(sorted((hash(self.point1), hash(self.point2)))))


def get_entry_points_from_lines(lines_list):
    """
    Original function provided by user, slightly adapted to use a local list.
    Extracts all unique points marked as 'is_entry' from a list of lines.
    """
    entry_points_set = set()
    for line in lines_list:
        if hasattr(line.point1, "is_entry") and line.point1.is_entry:
            entry_points_set.add(line.point1)
        if hasattr(line.point2, "is_entry") and line.point2.is_entry:
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
            if hasattr(line.point1, "is_entry") and line.point1.is_entry and line.point1 not in consumed_entry_points:
                potential_start_points.append(line.point1)
            if hasattr(line.point2, "is_entry") and line.point2.is_entry and line.point2 not in consumed_entry_points:
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
            if hasattr(current_tip_of_path, "is_entry") and current_tip_of_path.is_entry:
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
                    if hasattr(point_on_candidate, "is_entry") and point_on_candidate.is_entry:
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

if __name__ == '__main__':
    # --- Test Data Setup ---
    # Path 1: p1 (entry) -> p2 -> p3 -> p4 (entry)
    p1 = Point(0, 0); p1.is_entry = True
    p2 = Point(10, 2)
    p3 = Point(20, -2)
    p4 = Point(30, 0); p4.is_entry = True
    line1 = Line(p1, p2)
    line2 = Line(p2, p3)
    line3 = Line(p3, p4)

    # Path 2: p5 (entry) -> p6 -> p7 (entry)
    p5 = Point(5, 15); p5.is_entry = True
    p6 = Point(15, 18)
    p7 = Point(25, 15); p7.is_entry = True
    line4 = Line(p5, p6)
    line5 = Line(p6, p7)

    # Distractor/Noise line
    p8 = Point(50, 50)
    p9 = Point(60, 55)
    line6 = Line(p8, p9) # No entry points

    # Line starting with entry but leading nowhere near another entry
    p10 = Point(70,70); p10.is_entry = True
    p11 = Point(80,70)
    line7 = Line(p10,p11)


    # Initial pool of lines for the algorithm
    # all_test_lines = [line1, line2, line3, line4, line5, line6, line7, l_jupyter_1, l_jupyter_2]
    all_test_lines = [line1, line2, line3, line4, line5, line6, line7]


    print("Initial Lines:")
    for l_idx, l_val in enumerate(all_test_lines):
        print(f"  L{l_idx}: {l_val}")
        if l_val.point1.is_entry: print(f"    -> {l_val.point1} is entry")
        if l_val.point2.is_entry: print(f"    -> {l_val.point2} is entry")

    print("\nFinding paths...")
    # Adjust proximity_threshold and weights as needed
    found_paths_result = find_line_paths(
        all_test_lines,
        proximity_threshold=15.0, # Max distance between points to consider connecting
        dot_product_weight=0.5,
        distance_to_line_weight=0.25,
        endpoint_distance_weight=0.25
    )

    print("\n--- Found Paths ---")
    if found_paths_result:
        for i, path_dict in enumerate(found_paths_result):
            print(f"Path {i + 1}:")
            print(f"  Lines ({len(path_dict['lines'])}):")
            for line_obj in path_dict['lines']:
                print(f"    {line_obj}")
            print(f"  Points sequence ({len(path_dict['points'])}):")
            points_str = " -> ".join([str(p) for p in path_dict['points']])
            print(f"    {points_str}")
            print(f"  Starts at: {path_dict['points'][0]} (Entry: {path_dict['points'][0].is_entry}), "
                  f"Ends at: {path_dict['points'][-1]} (Entry: {path_dict['points'][-1].is_entry})")
    else:
        print("No paths were found.")