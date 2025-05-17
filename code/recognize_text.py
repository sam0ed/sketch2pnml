import cv2
import numpy as np
from doctr.models import ocr_predictor
from models import Text # Import the Text class

def _geometry_to_absolute_coords(relative_geom: tuple[tuple[float, float], tuple[float, float]], 
                                 img_width: int, img_height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Converts doctr's relative coordinates to absolute integer coordinates."""
    (xmin_rel, ymin_rel), (xmax_rel, ymax_rel) = relative_geom
    
    xmin_abs = int(xmin_rel * img_width)
    ymin_abs = int(ymin_rel * img_height)
    xmax_abs = int(xmax_rel * img_width)
    ymax_abs = int(ymax_rel * img_height)
    
    return ((xmin_abs, ymin_abs), (xmax_abs, ymax_abs))

def detect_text(img_color_resized: np.ndarray, config: dict) -> list[Text]:
    """
    Detects text using doctr and returns a list of Text objects with absolute coordinates.
    (Implementation remains the same)
    """
    predictor_params = config.get('text_detection', {})
    predictor = ocr_predictor(
        det_arch='db_resnet50', 
        reco_arch='crnn_vgg16_bn', 
        pretrained=True,
    )
    predictor.det_predictor.model.postprocessor.bin_thresh = predictor_params.get('bin_thresh', 0.3)
    predictor.det_predictor.model.postprocessor.box_thresh = predictor_params.get('box_thresh', 0.1)

    out = predictor([img_color_resized])

    detected_texts: list[Text] = []
    img_height, img_width = img_color_resized.shape[:2]

    if out.pages:
        for block in out.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    abs_geom = _geometry_to_absolute_coords(word.geometry, img_width, img_height)
                    text_obj = Text(value=word.value, 
                                      geometry_abs=abs_geom, 
                                      confidence=word.confidence)
                    detected_texts.append(text_obj)
    
    return detected_texts

def contours_in_bboxes(bboxes_abs: list[tuple[tuple[int, int], tuple[int, int]]], 
                       contours_list: list[np.ndarray], 
                       include_border=True) -> list[np.ndarray]:
    """
    Finds contours from contours_list that are entirely inside any of the bounding boxes.
    Adapted from the user-provided main.ipynb snippet.

    Args:
        bboxes_abs: List of absolute bounding boxes ((xmin, ymin), (xmax, ymax)).
        contours_list: List of contours found in the image.
        include_border: Whether points on the border count as inside.

    Returns:
        A flat list of contours found fully inside any of the provided bounding boxes.
    """
    inside_contours = []
    # Keep track of added contours to avoid duplicates if a contour is in multiple boxes
    # Using a simple approach: store contour bounding boxes or a representative point hash?
    # For simplicity, let's risk duplicates for now, drawContours handles it visually.
    # A more robust method would involve contour moments or more complex comparison.
    
    test_threshold = 0 if include_border else 1

    # Precompute axis-aligned bounding rects for contours for a quick AABB-reject
    contour_brects = [cv2.boundingRect(c) for c in contours_list]

    for (xmin, ymin), (xmax, ymax) in bboxes_abs:
        # Construct the polygon format cv2.pointPolygonTest expects (4,1,2) int32
        poly = np.array([[[xmin, ymin]], [[xmax, ymin]], [[xmax, ymax]], [[xmin, ymax]]], dtype=np.int32)

        for i, (ctr, (cx, cy, w, h)) in enumerate(zip(contours_list, contour_brects)):
            # Fast AABB reject (check if contour's bbox is completely outside the text bbox)
            if (cx + w) < xmin or (cy + h) < ymin or cx > xmax or cy > ymax:
                continue 

            # Perform the more precise point-in-polygon test
            pts = ctr.reshape(-1, 2)
            all_inside = True
            for (x_np, y_np) in pts:
                dist = cv2.pointPolygonTest(poly, (float(x_np), float(y_np)), False)
                if dist < test_threshold:
                    all_inside = False
                    break
            
            if all_inside:
                inside_contours.append(ctr)
                # Potential optimization: Remove this contour from further checks? 
                # Needs careful list management if doing so.

    # TODO: Consider adding a step to remove duplicate contours if needed.
    return inside_contours

def remove_text_contours(preprocessed_img: np.ndarray, detected_texts: list[Text]) -> np.ndarray:
    """
    Removes text from the preprocessed image by finding contours within text bounding boxes
    and applying a mask using bitwise_and (similar to original notebook).

    Args:
        preprocessed_img: The thresholded image (e.g., from Otsu).
        detected_texts: List of Text objects with absolute coordinates.

    Returns:
        The image with text contours removed (blacked out).
    """
    if not detected_texts:
        return preprocessed_img.copy()

    contours_list, _ = cv2.findContours(preprocessed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bboxes_abs = [text.geometry for text in detected_texts]

    text_contours_inside = contours_in_bboxes(bboxes_abs, contours_list)

    mask = np.ones_like(preprocessed_img)
    cv2.drawContours(mask, text_contours_inside, -1, (0), thickness=cv2.FILLED)

    img_no_text = cv2.bitwise_and(preprocessed_img, preprocessed_img, mask=mask)

    return img_no_text 
