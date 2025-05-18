import math
import numpy as np
import largestinteriorrectangle as lir
import cv2

# Epsilon for floating point comparisons if needed, though not used in current definitions
EPS = 1e-6 

def is_number(n):
    is_number = True
    try:
        num = float(n)
        # check for "nan" floats
        is_number = num == num   # or use `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number

class Point:
    def __init__(self, x, y):
        self.x = int(x) # Ensure integer coordinates if they represent pixels
        self.y = int(y)

        self.proximity_node = None # Placeholder for proximity node assignment
        self.is_arrow = False # Placeholder for entry point assignment

    def get_distance_between_points(self, other_point):
        """Calculate Euclidean distance between this point and another point."""
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)
    
    def is_inside_contour(self, contour):
        """Check if this point is inside a given contour using cv2.pointPolygonTest"""
        # Note: This requires cv2, which might be better placed in a different module
        point_tuple = (float(self.x), float(self.y)) # pointPolygonTest needs float tuple
        # Ensure contour is in the correct format (e.g., Nx1x2 or Nx2)
        try:
            # >= 0 means inside or on the boundary
            return cv2.pointPolygonTest(contour, point_tuple, False) >= 0 
        except Exception as e:
            print(f"Error during pointPolygonTest: {e}")
            return False
        
    def get_numpy_array(self):
        """Returns the point as a numpy array."""
        return np.array([self.x, self.y], dtype=np.int32)

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
    def __init__(self, start_point: Point, end_point: Point, angle=None, length=None):
        """
        Initializes a Line object.
        If angle and length are not provided, they are calculated.
        """
        self.point1 = start_point
        self.point2 = end_point

        # Assign self to the points for back-reference if needed later
        # self.point1.part_of = self 
        # self.point2.part_of = self

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

        numerator = np.abs(np.cross(p2_np - p1_np, p1_np - p3_np))
        denominator = np.linalg.norm(p2_np - p1_np)
        if denominator == 0:
            return np.linalg.norm(p3_np - p1_np) # Distance to the single point
        return numerator / denominator
    
    def distance_point_to_segment(self, point: Point) -> float:
        """
        Calculates the shortest distance from a query point to this line segment.
        """
        # Convert query point and segment endpoints to numpy arrays
        p_np = point.get_numpy_array().astype(float)
        a_np = self.point1.get_numpy_array().astype(float) # Segment start (self.point1)
        b_np = self.point2.get_numpy_array().astype(float) # Segment end (self.point2)

        # If the segment is essentially a point (point1 and point2 are the same)
        if self.point1 == self.point2: # Relies on Point.__eq__
            return point.get_distance_between_points(self.point1)

        # Vector from A to B (segment vector)
        vec_ab = b_np - a_np
        # Vector from A to P (point relative to segment start)
        vec_ap = p_np - a_np

        t = np.dot(vec_ap, vec_ab) / np.dot(vec_ab, vec_ab)

        if 0.0 <= t <= 1.0:
            # The projection falls on the segment AB.
            # The shortest distance is the perpendicular distance from P to the line AB.
            # This can be calculated by self.distance_point_to_infinite_line(point).
            return self.distance_point_to_infinite_line(point)
        elif t < 0.0:
            # The projection falls outside the segment, on the side of A.
            # The closest point on the segment to P is A (self.point1).
            return point.get_distance_between_points(self.point1)
        else: # t > 1.0
            return point.get_distance_between_points(self.point2)

    def __repr__(self):
        return f"Line(start={self.point1}, end={self.point2}, angle={self.angle:.2f}, length={self.length:.2f})"

    def __eq__(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        # A line is considered equal if its endpoints are the same, regardless of order.
        return (self.point1 == other.point1 and self.point2 == other.point2) or \
               (self.point1 == other.point2 and self.point2 == other.point1)

    def __hash__(self):
        """Allows Line objects to be added to sets. The hash is order-invariant for points."""
        # Hash the tuple of sorted point hashes
        return hash(tuple(sorted((hash(self.point1), hash(self.point2)))))


#####################################################################
#####################################################################
class Place:
    def __init__(
        self,
        circle: tuple[int, int, int], # (x, y, radius)
        original_detection_data=None, # Placeholder for any original detection data
    ):
        self.center = Point(circle[0], circle[1])
        self.radius = circle[2]
        self.center.part_of = self # Link back to the Place object

        self.text = [] # Placeholder for any text associated with this place
        self.original_detection_data = original_detection_data 

        self.markers = 0 # Placeholder for markers associated with this place

    @classmethod
    def from_contour(cls, contour: np.ndarray):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return cls((x, y, radius), original_detection_data= contour)
    
    def update_markers_from_text(self):
        """
        Recalculates and updates self.markers by summing numeric values
        from associated Text objects in self.text.
        Only text values that consist purely of digits after stripping whitespace
        are considered numeric.
        """
        current_sum_of_markers = 0
        for text_obj in self.text: # self.text is a list of Text objects
            value_str = text_obj.value.strip()
            if is_number(value_str):
                try:
                    num_val = float(value_str)
                    # Check for infinity, as int(inf) raises OverflowError
                    if num_val != float('inf') and num_val != float('-inf'):
                        current_sum_of_markers += int(num_val)
                    # else: print(f"Info: Skipped infinite value '{value_str}' for markers.") # Optional logging
                except ValueError:
                    pass 
        self.markers = current_sum_of_markers
        

    def __repr__(self):
        return f"Place(center={self.center}, radius={self.radius})"

class Transition:
    def __init__(
        self,
        center_coords: tuple[int, int], # (x, y)
        height: int,
        width: int,
        angle: float = 0.0, # Default angle
        original_detection_data=None, 
    ):
        self.center = Point(center_coords[0], center_coords[1])
        self.center.part_of = self

        self.height = height
        self.width = width
        self.angle = angle # Angle in degrees

        self.box_points = cv2.boxPoints(((self.center.x, self.center.y), (self.height, self.width), angle))

        self.points = [Point(int(pt[0]), int(pt[1])) for pt in self.box_points]
        for point in self.points:
            point.part_of = self

        self.text = [] 

        self.original_detection_data = original_detection_data 

    @classmethod
    def from_contour(cls, contour: np.ndarray):
        min_area_rect = cv2.minAreaRect(contour)
        return cls(min_area_rect[0], min_area_rect[1][0], min_area_rect[1][1], min_area_rect[2], original_detection_data=contour)
    
    def __repr__(self):
        return f"Transition(center={self.center}, height={self.height}, width={self.width}, angle={self.angle})"

### Potentially add an Arc class later if needed to represent the final connections
class Arc:
    def __init__(self, source, target, start_point, end_point, points=None, lines=None):
        self.source = source # Place or Transition object
        self.target = target # Place or Transition object
        self.start_point = start_point # Point object
        self.end_point = end_point # Point object
        self.points = points # Optional: Ordered list of points forming the arc geometry
        self.lines = lines   # Optional: List of Line segments forming the arc geometry

        self.text = [] # Placeholder for any text associated with this place

    def __repr__(self):
        return f"Arc(source={self.source}, target={self.target})"
    
    def __eq__(self, other):
        if not isinstance(other, Arc):
            return NotImplemented
        return (self.source == other.source and self.target == other.target)

class Text:
    """Represents a detected text element with its content and bounding box."""
    # Store geometry as absolute integer coordinates
    def __init__(self, value: str, geometry_abs: tuple[tuple[int, int], tuple[int, int]], confidence: float):
        """
        Args:
            value: The recognized text string.
            geometry_abs: Bounding box absolute coordinates ((xmin, ymin), (xmax, ymax)).
            confidence: The recognition confidence score.
        """
        self.value = value
        self.pt1 = Point(geometry_abs[0][0], geometry_abs[0][1])
        self.pt2 = Point(geometry_abs[1][0], geometry_abs[1][1])
        self.center = Point(
            (self.pt1.x + self.pt2.x) // 2,
            (self.pt1.y + self.pt2.y) // 2
        )
        self.confidence = confidence

    def __repr__(self):
        return f"Text(value='{self.value}', box=({self.pt1.x},{self.pt1.y})-({self.pt2.x},{self.pt2.y}), conf={self.confidence:.2f})"

