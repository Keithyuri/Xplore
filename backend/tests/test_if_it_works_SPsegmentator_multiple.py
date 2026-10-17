import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp

# ================= CONFIG =================
NEW_IMAGE_PATH = "dataset/Test/chest_test4.jpg"  # your new image
OUTPUT_DIR = Path("dataset/Test/Results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "spinous_training_results/best_model.pth"  # trained model
TARGET_SIZE = (512, 512)  # training resolution

OUTPUT_DIR.mkdir(exist_ok=True)

# ================= LOAD MODEL =================
print("🧠 Loading trained model...")
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,  # use None if you trained from scratch
    in_channels=1,
    classes=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✓ Model loaded successfully")

# ================= LOAD IMAGE =================
print(f"📂 Loading new image: {NEW_IMAGE_PATH}")
img = cv2.imread(NEW_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found: {NEW_IMAGE_PATH}")

orig_h, orig_w = img.shape
print(f"✓ Original image shape: {img.shape}")

# --- Resize to training resolution ---
img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

# --- Normalize like during training ---
img_resized = img_resized.astype(np.float32)
img_resized = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-6)

img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ================= INFERENCE =================
print("🔍 Running inference...")
with torch.no_grad():
    logits = model(img_tensor)
    prob_map_512 = torch.sigmoid(logits)[0, 0]

# --- Resize prediction back to original size ---
prob_map = F.interpolate(
    prob_map_512.unsqueeze(0).unsqueeze(0),
    size=(orig_h, orig_w),
    mode="nearest"
)[0, 0].cpu().numpy()
print("✓ Inference completed")

# ================= SAVE MASK =================
threshold = 0.3  # good for thin structures like spinous processes
pred_mask = (prob_map > threshold).astype(np.uint8) * 255
mask_path = OUTPUT_DIR / f"{Path(NEW_IMAGE_PATH).stem}_pred_mask.png"
cv2.imwrite(str(mask_path), pred_mask)
print(f"✅ Predicted mask saved: {mask_path}")

# ================= OVERLAY =================
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 20:
        cv2.drawContours(img_color, [c], -1, (0, 255, 0), 2)

overlay = img_color.copy()
overlay[pred_mask > 0] = (0, 255, 255)
final = cv2.addWeighted(img_color, 0.7, overlay, 0.3, 0)
overlay_path = OUTPUT_DIR / f"{Path(NEW_IMAGE_PATH).stem}_overlay.png"
cv2.imwrite(str(overlay_path), final)
print(f"✅ Overlay saved: {overlay_path}")

print("\n📊 ALL DONE. Check the output images in:", OUTPUT_DIR)
