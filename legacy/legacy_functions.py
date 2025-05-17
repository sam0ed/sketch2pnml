def is_circle_circularity(contour, threshold): #TODO: add threshold
        # Calculate contour area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity - a perfect circle has circularity = 1
    # Formula: 4π × Area/Perimeter²
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Consider additional criteria for better circle detection
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Filter based on circularity and other optional criteria
    return circularity > 0.8 and 0.8 < aspect_ratio < 1.2 and area > 50

def is_circle_hough(contour, threshold):
    x, y, w, h = cv2.boundingRect(contour)

    # Create a blank image slightly larger than the contour
    padding = 10
    mask = np.zeros((h + 2*padding, w + 2*padding), dtype=np.uint8)

    # Shift the contour to fit in the blank image
    shifted_contour = contour.copy()
    shifted_contour[:, :, 0] = contour[:, :, 0] - x + padding
    shifted_contour[:, :, 1] = contour[:, :, 1] - y + padding

    # Draw just the contour on the blank image
    cv2.drawContours(mask, [shifted_contour], -1, 255, 1)

    # Ensure input is grayscale (HoughCircles requires single-channel)    
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Detect circles
    circles = cv2.HoughCircles(
        mask, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=5, 
        param2=25, 
        minRadius=5, 
        maxRadius=95
    )

    return circles is not None


def is_rectangle_by_angle(contour, 
                          epsilon_ratio=0.02,
                          angle_tol=10.0):
    """
    Return True if `contour` is a convex quadrilateral whose four internal
    angles are all within ±angle_tol degrees of 90°.
    
    Params:
      contour        – input contour (Nx1x2 array)
      epsilon_ratio  – approxPolyDP ε as a fraction of contour perimeter
      angle_tol      – maximum allowed deviation from 90° for each corner
    """
    # 1) Approximate to a polygon
    peri = cv2.arcLength(contour, True)
    eps  = epsilon_ratio * peri
    pts  = cv2.approxPolyDP(contour, eps, True)

    # 2) Must be exactly 4 points and convex
    if len(pts) != 4 or not cv2.isContourConvex(pts):
        return False

    pts = pts.reshape(4, 2)

    # 3) Angle at pt B between vectors BA and BC
    def corner_angle(A, B, C):
        BA = A - B
        BC = C - B
        # clamp for numerical safety
        cos = np.dot(BA, BC) / (np.linalg.norm(BA)*np.linalg.norm(BC))
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    # 4) Check all angles ≈ 90°
    for i in range(4):
        A = pts[(i - 1) % 4]
        B = pts[i]
        C = pts[(i + 1) % 4]
        ang = corner_angle(A, B, C)
        if abs(ang - 90.0) > angle_tol:
            return False

    return True


# def extract_circle_features(contours):
#     feats = []
#     for cnt in contours:
#         a = cv2.contourArea(cnt)
#         p = cv2.arcLength(cnt, True)
#         feats.append([a, p])
#     feats_raw = np.asarray(feats, float)
#     feats_log = np.log(feats_raw + np.finfo(float).eps)
#     return feats_raw, feats_log

# def extract_rect_features(contours):
#     feats = []
#     for cnt in contours:
#         rect = cv2.minAreaRect(cnt)
#         # Extract width and height from the minAreaRect result
#         # rect[1][0] is the width and rect[1][1] is the height of the rotated rectangle
#         feats.append([rect[1][0], rect[1][1]])
#     feats_raw = np.asarray(feats, float)
#     feats_log = np.log(feats_raw + np.finfo(float).eps)
#     return feats_raw, feats_log

def extract_contour_features(contours):
    feats = []
    for cnt in contours:
        feats.append([cv2.contourArea(cnt), cv2.arcLength(cnt, True)])
    feats_raw = np.asarray(feats, float)
    feats_log = np.log(feats_raw + np.finfo(float).eps)
    return feats_raw, feats_log

def cluster_contours(contours):
    feats_raw, feats_log = extract_contour_features(contours)
    
    if feats_log.shape[0] > 2:
        db = DBSCAN(eps=0.02, min_samples=3).fit(feats_log) ## TODO: could cause bugs because of incorrect clustering 
        return db.labels_

    else:
        return list(range(len(contours)))


def cluster_by_relative_diff(feats, tol=0.10):
    """
    Cluster feats so that within each cluster, every pair of points
    differs by at most `tol` in relative terms on every feature axis.
    """
    def rel_diff(u, v):
        denom = np.maximum(np.abs(u), np.abs(v))
        denom[denom == 0] = 1e-8
        return np.max(np.abs(u - v) / denom)

    # 1) pairwise distances with our custom metric
    D = pdist(feats, metric=rel_diff)
    # 2) complete‐linkage hierarchical clustering
    Z = linkage(D, method='complete')
    # 3) flat clusters at threshold = tol
    labels = fcluster(Z, t=tol, criterion='distance') - 1
    return labels


def is_contour_inside(contour1, contour2):
    """
    Checks if contour1 is completely inside contour2.
    A point on the boundary is considered inside.
    """
    num_points_c1 = contour1.shape[0]
    if num_points_c1 == 0:
        # An empty contour can be considered "inside" or an edge case.
        # Depending on requirements, could also return False or raise error.
        return True

    for i in range(num_points_c1):
        # contour1[i] is like [[x,y]] (a 1x2 array)
        # contour1[i, 0] is like [x,y] (a 1D array with 2 elements)
        coords = contour1[i, 0]
        try:
            # Ensure px and py are Python native integer scalars
            px = int(coords[0])
            py = int(coords[1])
        except (TypeError, IndexError) as e:
            # This block helps diagnose if coords[0] or coords[1] are not simple scalars
            print(f"Error converting coordinates to int: {e}")
            print(f"Problematic contour1 point index: {i}")
            print(f"Coordinates array (coords): {coords}")
            print(f"contour1 shape: {contour1.shape}, contour1 dtype: {contour1.dtype}")
            # Re-raise or handle as an error, as this point can't be tested
            raise ValueError(f"Malformed coordinate in contour1 at index {i}: {coords}") from e

        point = (px, py)
        dist = cv2.pointPolygonTest(contour2, point, False)
        if dist < 0:
            return False  # Point is outside, so contour1 is not entirely inside contour2
    return True
