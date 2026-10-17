import cv2
import numpy as np
from pathlib import Path

MASK_DIR = Path(r"C:\Users\Keith Chin\Radiography Project\backend\tests\dataset\masks\fixed_png")

ALLOWED_VALUES = {0, 64, 192, 255}

for i in range(1, 101):
    img_path = MASK_DIR / f"chest_xray_{i}_mask.png"

    if not img_path.exists():
        print(f"❌ Missing: {img_path.name}")
        continue

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"⚠️ Could not read: {img_path.name}")
        continue

    unique_vals = set(np.unique(img))
    invalid_vals = unique_vals - ALLOWED_VALUES

    if invalid_vals:
        print(f"🚨 {img_path.name} has INVALID pixel values: {sorted(invalid_vals)}")
    else:
        print(f"✅ {img_path.name} is clean")

print("✅ Pixel value check completed.")