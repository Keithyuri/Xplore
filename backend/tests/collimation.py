import cv2
import numpy as np

def detect_dynamic_collimation_with_edges_stop_at_black_otsu(img, black_thresh=70, min_fraction=0.03, edge_thresh=100):
    """
    Detects collimation dynamically using Otsu's threshold for bright regions,
    stopping at black pixels so that rectangles only cover bright portions.
    Returns True/False per side and overlay image.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()
    else:
        gray = img.copy()
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape

    # Compute Otsu's threshold
    _, bright_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Adaptive bright_thresh (Otsu): {bright_thresh}")

    sides_pixel_width = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    sides_collimated = {"left": False, "right": False, "top": False, "bottom": False}

    # Scan until black pixels or until bright_fraction < min_fraction
    def scan_edge(region, axis):
        if axis == 0:  # vertical scan (columns)
            for i in range(region.shape[1]):
                col = region[:, i]
                if np.any(col < black_thresh):  # stop at black pixels
                    return i
                if np.mean(col > bright_thresh) < min_fraction:
                    return i
            return region.shape[1]
        else:  # horizontal scan (rows)
            for i in range(region.shape[0]):
                row = region[i, :]
                if np.any(row < black_thresh):
                    return i
                if np.mean(row > bright_thresh) < min_fraction:
                    return i
            return region.shape[0]

    # Left
    left_width = scan_edge(gray[:, :w//2], axis=0)
    sides_pixel_width["left"] = left_width
    if left_width > 0:
        left_edge_index = left_width - 1
        left_edge_brightness = np.mean(gray[:, left_edge_index])
        sides_collimated["left"] = left_edge_brightness >= edge_thresh
        if sides_collimated["left"]:
            cv2.rectangle(vis, (0, 0), (left_width, h), (0, 255, 0), 2)

    # Right
    right_width = scan_edge(np.fliplr(gray[:, w//2:]), axis=0)
    sides_pixel_width["right"] = right_width
    if right_width > 0:
        right_edge_index = w - right_width
        right_edge_index = min(max(right_edge_index, 0), w-1)
        right_edge_brightness = np.mean(gray[:, right_edge_index])
        sides_collimated["right"] = right_edge_brightness >= edge_thresh
        if sides_collimated["right"]:
            cv2.rectangle(vis, (w - right_width, 0), (w, h), (0, 255, 0), 2)

    # Top
    top_width = scan_edge(gray[:h//2, :], axis=1)
    sides_pixel_width["top"] = top_width
    if top_width > 0:
        top_edge_index = top_width - 1
        top_edge_brightness = np.mean(gray[top_edge_index, :])
        sides_collimated["top"] = top_edge_brightness >= edge_thresh
        if sides_collimated["top"]:
            cv2.rectangle(vis, (0, 0), (w, top_width), (0, 255, 0), 2)

    # Bottom
    bottom_width = scan_edge(np.flipud(gray[h//2:, :]), axis=1)
    sides_pixel_width["bottom"] = bottom_width
    if bottom_width > 0:
        bottom_edge_index = h - bottom_width
        bottom_edge_index = min(max(bottom_edge_index, 0), h-1)
        bottom_edge_brightness = np.mean(gray[bottom_edge_index, :])
        sides_collimated["bottom"] = bottom_edge_brightness >= edge_thresh
        if sides_collimated["bottom"]:
            cv2.rectangle(vis, (0, h - bottom_width), (w, h), (0, 255, 0), 2)

    has_collimation = any(sides_collimated.values())
    return {"has_collimation": has_collimation, "sides": sides_collimated}, vis


# Example usage
img = cv2.imread(r"C:\Users\Keith Chin\Radiography Project\backend\tests\dataset\Test\chest_test.jpg")
results, overlay = detect_dynamic_collimation_with_edges_stop_at_black_otsu(img)
print(results)

cv2.imshow("Dynamic Collimation with Otsu Threshold", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
