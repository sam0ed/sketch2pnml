import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from doctr.models import detection_predictor, recognition_predictor, page_orientation_predictor, crop_orientation_predictor
from doctr.io import DocumentFile
from doctr.utils.geometry import detach_scores
from scipy.stats import median_abs_deviation
from sklearn.covariance import MinCovDet
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from scipy.stats import chi2

EPS = 1e-6

def preprocess(img):
    _, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return thresh_otsu

def detect_bboxes(img):
    # convert img to bytes
    _, img_encoded = cv2.imencode('.png', img)
    img_encoded = img_encoded.tobytes()
    # Helper function to convert relative coordinates to absolute pixel values
    def _to_absolute(geom, img_shape: tuple[int, int]) -> list[list[int]]:
        h, w = img_shape
        if len(geom) == 2:  # Assume straight pages = True -> [[xmin, ymin], [xmax, ymax]]
            (xmin, ymin), (xmax, ymax) = geom
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        else:  # For polygons, convert each point to absolute coordinates
            return [[int(point[0] * w), int(point[1] * h)] for point in geom]

    # Define the detection predictor
    det_predictor = detection_predictor(
        arch="db_resnet50",
        pretrained=True,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=True,
        batch_size=1,
    )  # .cuda().half()  # Uncomment this line if you have a GPU

    # Define the postprocessing parameters (optional)
    det_predictor.model.postprocessor.bin_thresh = 0.3
    det_predictor.model.postprocessor.box_thresh = 0.1

    # Load the document image
    docs = DocumentFile.from_images([img_encoded])
    results = det_predictor(docs)

    bboxes = []
    for doc, res in zip(docs, results):
        img_shape = (doc.shape[0], doc.shape[1])
        # Detach the probability scores from the results
        detached_coords, prob_scores = detach_scores([res.get("words")])

        for i, coords in enumerate(detached_coords[0]):
            coords = coords.reshape(2, 2).tolist() if coords.shape == (4,) else coords.tolist()

            # Convert relative to absolute pixel coordinates
            points = np.array(_to_absolute(coords, img_shape), dtype=np.int32).reshape((-1, 1, 2))
            bboxes.append(points)
    return  np.squeeze(np.array(bboxes), axis=2)


def contours_in_bboxes(bboxes, contours_list, include_border=True):
    inside = []
    test_threshold = 0 if include_border else 1

    # Precompute axis-aligned bounding rects for a quick AABB-reject
    contour_brects = [cv2.boundingRect(c) for c in contours_list]

    for bb in bboxes:
        # make sure our polygon is shape (4,1,2) and int32
        poly = bb.reshape(-1,1,2).astype(np.int32)

        # compute its AABB
        xs, ys = bb[:,0], bb[:,1]
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        box_inside = []
        for ctr, (cx, cy, w, h) in zip(contours_list, contour_brects):
            # fast AABB reject
            if cx < x0 or cy < y0 or (cx+w) > x1 or (cy+h) > y1:
                continue

            # flatten contour to Nx2
            pts = ctr.reshape(-1, 2)
            # test each point
            all_inside = True
            for (x_np, y_np) in pts:
                # convert to native Python floats
                x, y = float(x_np), float(y_np)
                dist = cv2.pointPolygonTest(poly, (x, y), False)
                if dist < test_threshold:
                    all_inside = False
                    break

            if all_inside:
                box_inside.append(ctr)

        inside.append(box_inside)

    return inside

def is_circle_enclosing(contour, threshold):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    enclosing_area = np.pi * (radius ** 2)
    contour_area = cv2.contourArea(contour)
    
    return contour_area / enclosing_area > threshold 

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
    

def is_rectangle_enclosing(contour, threshold):
    rect = cv2.minAreaRect(contour)
    box_area = rect[1][0] * rect[1][1]
    contour_area = cv2.contourArea(contour)
    
    if box_area == 0:
        return False
        
    return contour_area / box_area > threshold

def extract_contour_features(contours):
    feats = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        p = cv2.arcLength(cnt, True)
        feats.append([a, p])
    feats_raw = np.asarray(feats, float)
    feats_log = np.log(feats_raw + np.finfo(float).eps)
    return feats_raw, feats_log

def split_contours(contours):
    feats_raw, feats_log = extract_contour_features(contours)
    
    if feats_log.shape[0] > 2:
        db = DBSCAN(eps=0.02, min_samples=3).fit(feats_log) ## TODO: could cause bugs because of incorrect clustering 

        ## get the most common label
        labels = db.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        # return contour indices with the most common label and the rest
        return np.where(db.labels_ == most_common_label)[0].tolist(), np.where(db.labels_ != most_common_label)[0].tolist()
    else:
        return list(range(len(contours))), []
        

if __name__ == "__main__":
    # img_path = r'C:\Users\samoed\Documents\GitHub\diploma_bachelor\data\else\overlapped.png'
    # img_path = r'C:\Users\samoed\Documents\GitHub\diploma_bachelor\data\internet\hand_drawn_petri.jpg'
    # img_path = r'C:\Users\samoed\Documents\GitHub\diploma_bachelor\data\internet\petri_net_4.png'
    # img_path = r'C:\Users\samoed\Documents\GitHub\diploma_bachelor\data\local\simple_petri_1.jpg'
    img_path = r'C:\Users\samoed\Documents\GitHub\diploma_bachelor\data\upscaled\better_res\mp_2.png'
    try:
        with open(img_path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"File {img_path} not found.")
        exit(1)
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_inverted = cv2.bitwise_not(img_gray)

    preprocessed_img = preprocess(img_inverted)

    bboxes = detect_bboxes(img)
    # cv2.polylines(img, bboxes, isClosed=True, color=(255, 0, 0), thickness=2)

    contours_list, hierarchy = cv2.findContours(preprocessed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    ## display contours
    # for i, contour in enumerate(contours_list):
    #     cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    text_contours = contours_in_bboxes(bboxes, contours_list)
    # ## display all results
    # for i, box in enumerate(text_contours):
    #     for j, contour in enumerate(box):
    #         cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    # Create a blank mask with the same dimensions as preprocessed_img
    text_mask = np.ones_like(preprocessed_img)
    for text_contour in text_contours:
        cv2.drawContours(text_mask, text_contour, -1, (0), thickness=cv2.FILLED)
    img_no_text = cv2.bitwise_and(preprocessed_img, preprocessed_img, mask=text_mask)


    ### Nodes detection
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv2.dilate(img_no_text, kernel, iterations=1) ## TODO: Heuristic
    contours_list, hierarchy = cv2.findContours(img_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 

    # Filter for circular contours
    circular_contours = []
    for i, contour in enumerate(contours_list):
        if is_circle_enclosing(contour, 0.7):
            circular_contours.append(contour)

    main_indices, rest_indices = split_contours(circular_contours)
    # for i, contour in enumerate(circular_contours):
    #     if i in main_indices:
    #         cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    #     else:
    #         cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)

    # Filter for rectangular contours
    rectangular_contours = []
    for i, contour in enumerate(contours_list):
        if is_rectangle_enclosing(contour, 0.95):
            rectangular_contours.append(contour)
    
    # display rectangular contours
    for i, contour in enumerate(rectangular_contours):
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    # main_indices, rest_indices = split_contours(rectangular_contours)
    # for i, contour in enumerate(rectangular_contours):
    #     if i in main_indices:
    #         cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    #     else:
    #         cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
    
    Image.fromarray(img).show() # Show the created mask
    