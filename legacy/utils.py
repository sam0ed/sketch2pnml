import largestinteriorrectangle as lir
import math
import cv2

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_distance_between_points(self, other):
        """Calculate Euclidean distance between this point and another point"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    ### TODO: test this function
    def is_inside_contour(self, contour):
        """Check if this point is inside a given contour"""
        # Convert the point to a tuple
        point_tuple = (self.x, self.y)
        # Use cv2.pointPolygonTest to check if the point is inside the contour
        return cv2.pointPolygonTest(contour, point_tuple, False) >= 0

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Place:
    def __init__(
        self,
        circle: tuple[int, int, int],
    ):
        self.center = Point(circle[0], circle[1])
        self.radius = circle[2]

    def __repr__(self):
        return f"Place(center={self.center}, radius={self.radius})"

class Transition:
    def __init__(
        self,
        rect
    ):
        self.height = lir.pt2(rect)[1] - lir.pt1(rect)[1]
        self.width = lir.pt2(rect)[0] - lir.pt1(rect)[0]
        self.center = Point(*self._get_rectangle_center(rect))

    def _get_rectangle_center(self, rectangle):
        x1, y1 = lir.pt1(rectangle)
        x2, y2 = lir.pt2(rectangle)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def __repr__(self):
        return f"Transition(center={self.center}, height={self.height}, width={self.width})"

class Line:
    def __init__(self, line):
        self.point1 = Point(line[0][0], line[0][1])
        self.point2 = Point(line[0][2], line[0][3])
        self.point1.part_of = self
        self.point2.part_of = self

        self.angle = self._get_angle()
        self.length = self._get_length()

    def _get_angle(self):
        dx = self.point2.x - self.point1.x
        dy = self.point2.y - self.point1.y
        return math.degrees(math.atan2(dy, dx))
    

    def _get_length(self):
        return self.point1.get_distance_between_points(self.point2)
    
    def __repr__(self):
        return f"Line(start={self.point1}, end={self.point2}, angle={self.angle}, length={self.length})"
    
    