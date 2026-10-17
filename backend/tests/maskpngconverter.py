import cv2
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\Keith Chin\Radiography Project\backend\tests\dataset\images")
OUTPUT_DIR = INPUT_DIR / "png"

OUTPUT_DIR.mkdir(exist_ok=True)

for i in range(1, 101):
    img_path = INPUT_DIR / f"chest_xray_{i}.jpg"

    if not img_path.exists():
        print(f"❌ Missing: {img_path.name}")
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        print(f"⚠️ Could not read: {img_path.name}")
        continue

    out_path = OUTPUT_DIR / f"chest_xray_{i}.png"
    cv2.imwrite(str(out_path), img)

    print(f"✅ Converted: {out_path.name}")
