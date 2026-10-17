import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp

# ================= CONFIG =================
NEW_IMAGE_PATH = "dataset/Test/chest_test2.jpg"
OUTPUT_DIR = Path("dataset/Test/Results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "spinous_training_results/best_model.pth"
TARGET_SIZE = (512, 512)

OUTPUT_DIR.mkdir(exist_ok=True)

# ================= LOAD MODEL =================
print("🧠 Loading trained 3-class model...")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=1,
    classes=3  # Spinous | Costo | Blunted Costo
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✓ Model loaded successfully")

# ================= LOAD IMAGE =================
print(f"📂 Loading image: {NEW_IMAGE_PATH}")
img = cv2.imread(NEW_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {NEW_IMAGE_PATH}")

orig_h, orig_w = img.shape

img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
img_resized = img_resized.astype(np.float32)
img_resized = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-6)

img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ================= INFERENCE =================
print("🔍 Running inference...")
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.sigmoid(logits)[0]  # [3, 512, 512]

spinous_prob = probs[0]
costo_prob = probs[1]
blunted_prob = probs[2]

# ================= RESIZE BACK =================
def resize_back(prob):
    return F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="nearest"
    )[0, 0].cpu().numpy()

spinous_prob = resize_back(spinous_prob)
costo_prob = resize_back(costo_prob)
blunted_prob = resize_back(blunted_prob)

# ================= THRESHOLD =================
spinous_mask = (spinous_prob > 0.3).astype(np.uint8) * 255
costo_mask = (costo_prob > 0.5).astype(np.uint8) * 255

# 🫁 Gate blunting only where costo exists
blunted_mask = (
    (blunted_prob > 0.5) &
    (costo_prob > 0.5)
).astype(np.uint8) * 64  # 🔴 REQUIRED VALUE

# 🚫 Ensure no overlap (hard safety)
costo_mask[blunted_mask > 0] = 0

# ================= SAVE MASKS =================
stem = Path(NEW_IMAGE_PATH).stem

spinous_path = OUTPUT_DIR / f"{stem}_spinous.png"
costo_path = OUTPUT_DIR / f"{stem}_costo.png"
blunted_path = OUTPUT_DIR / f"{stem}_blunted.png"

cv2.imwrite(str(spinous_path), spinous_mask)
cv2.imwrite(str(costo_path), costo_mask)
cv2.imwrite(str(blunted_path), blunted_mask)

print("✅ All masks saved")

# ================= OVERLAY =================
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay = img_color.copy()

overlay[spinous_mask > 0] = (0, 255, 255)   # Yellow = spinous
overlay[costo_mask > 0] = (255, 0, 0)       # Blue = costo
overlay[blunted_mask > 0] = (0, 0, 255)     # Red = blunted

final = cv2.addWeighted(img_color, 0.7, overlay, 0.3, 0)
overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"
cv2.imwrite(str(overlay_path), final)

print(f"✅ Overlay saved: {overlay_path}")
print("\n📊 DONE — check Results folder")
