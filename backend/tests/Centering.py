import cv2

# ---- config ----
input_path = r"C:\Users\Keith Chin\Radiography Project\backend\Sample images\00003977_002_combined_overlay_distinct.png"       # change to your file
output_path = r"C:\Users\Keith Chin\Radiography Project\backend\Sample images\00003977_002_with_x.png"
color = (0, 255, 0)              # BGR: green
thickness = 3                    # line thickness in pixels
alpha = 0.35                     # 0=transparent, 1=opaque overlay
# ---------------

img = cv2.imread(input_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read image: {input_path}")

h, w = img.shape[:2]

# draw on an overlay so you can control transparency
overlay = img.copy()
cv2.line(overlay, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)
cv2.line(overlay, (w - 1, 0), (0, h - 1), color, thickness, cv2.LINE_AA)

# blend overlay onto the original
out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

cv2.imwrite(output_path, out)
print(f"Saved: {output_path}")