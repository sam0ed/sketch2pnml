import numpy as np

def minmaxToWidthHeight(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))  # Convert to (x, y, width, height)

def minmaxToCenterWidthHeight(xyxy):
    x1, y1, x2, y2 = xyxy
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    width = int(x2 - x1)
    height = int(y2 - y1)
    return (center_x, center_y, width, height)  # Convert to (center_x, center_y, width, height)

def minmaxToContours(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)  # Convert to contour format