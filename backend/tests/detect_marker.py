import cv2
import numpy as np
import pytesseract

# -----------------------
# Set Tesseract path
# -----------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------
# OCR function for a cropped marker
# -----------------------
def recognize_marker_letter(crop):
    """
    Recognize 'L' or 'R' in a cropped marker region,
    handling inversion, rotation, and flips.
    """
    if crop is None or crop.size == 0:
        return None

    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Invert colors (white text on black background)
    crop = cv2.bitwise_not(crop)

    # Resize to improve OCR accuracy
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Try all flips and rotations
    transformations = [
        crop,
        cv2.flip(crop, 1),
        cv2.flip(crop, 0),
        cv2.flip(crop, -1)
    ]

    for img in transformations:
        for angle in [0, 90, 180, 270]:
            if angle != 0:
                M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
                rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            else:
                rotated = img

            # Threshold to clean noise
            _, thresh = cv2.threshold(rotated, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # OCR: single character, whitelist L and R
            text = pytesseract.image_to_string(
                thresh, 
                config='--psm 10 -c tessedit_char_whitelist=LR'
            ).strip().upper()

            if text in ["L", "R"]:
                return text

    return None

# -----------------------
# Marker detection function
# -----------------------
def detect_markers(xray_image, min_area=50, max_area=2000):
    """
    Detect bright blobs in the X-ray image that could be markers.
    Returns list of (x, y, w, h, letter)
    """
    gray = cv2.cvtColor(xray_image, cv2.COLOR_BGR2GRAY)

    # Threshold bright regions (adjust 200 if needed)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours of bright regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_markers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if min_area < area < max_area:
            crop = gray[y:y+h, x:x+w]
            letter = recognize_marker_letter(crop)
            if letter:
                detected_markers.append((x, y, w, h, letter))

    return detected_markers

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    xray_image = cv2.imread(r"C:\Users\Keith Chin\Radiography Project\backend\tests\dataset\Test\chest_test1.jpg")
    if xray_image is None:
        raise FileNotFoundError("X-ray image not found. Check path!")

    markers = detect_markers(xray_image)

    if not markers:
        print("No markers detected.")
    else:
        for x, y, w, h, letter in markers:
            print(f"Marker detected at ({x},{y}) - Letter: {letter}")
            # Draw rectangle + label
            cv2.rectangle(xray_image, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(xray_image, letter, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Show result
        cv2.imshow("Detected Markers", xray_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

