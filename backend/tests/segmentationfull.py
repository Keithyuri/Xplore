import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import torchxrayvision as xrv
import torchvision
import skimage

# ================= CONFIG =================

# Folder containing the 20 images
IMAGE_DIR = Path(r"C:\Users\Keith Chin\Radiography Project\backend\Sample images\Images to be demonstrated")

# Save outputs here
SAVE_DIR = Path(r"C:\Users\Keith Chin\Radiography Project\backend\Sample images\segmentation_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Build list of image paths: chest_test_original_1.png ... chest_test_original_20.png
IMAGE_PATHS = [IMAGE_DIR / f"chest_test_original_{i}.png" for i in range(1, 21)]

# Sanity check
missing = [str(p) for p in IMAGE_PATHS if not p.exists()]
if missing:
    raise FileNotFoundError("Missing these expected images:\n" + "\n".join(missing))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (512, 512)

# ================= SEPARATE MODEL PATHS =================
SPINOUS_MODEL_PATH = r"C:\Users\Keith Chin\Radiography Project\backend\tests\separate_binary_training_results\best_spinous_model.pth"
COSTO_MODEL_PATH = r"C:\Users\Keith Chin\Radiography Project\backend\tests\separate_binary_training_results\best_costo_model.pth"
BLUNTED_MODEL_PATH = r"C:\Users\Keith Chin\Radiography Project\backend\tests\separate_binary_training_results\best_blunted_model.pth"

# ================= LOAD BINARY U-NET MODELS =================
def create_binary_unet():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1
    )
    return model

print("🧠 Loading spinous U-Net model...")
spinous_model = create_binary_unet()
spinous_model.load_state_dict(torch.load(SPINOUS_MODEL_PATH, map_location=DEVICE))
spinous_model = spinous_model.to(DEVICE)
spinous_model.eval()
print("✓ Spinous model loaded")

print("🧠 Loading costo U-Net model...")
costo_model = create_binary_unet()
costo_model.load_state_dict(torch.load(COSTO_MODEL_PATH, map_location=DEVICE))
costo_model = costo_model.to(DEVICE)
costo_model.eval()
print("✓ Costo model loaded")

print("🧠 Loading blunted U-Net model...")
blunted_model = create_binary_unet()
blunted_model.load_state_dict(torch.load(BLUNTED_MODEL_PATH, map_location=DEVICE))
blunted_model = blunted_model.to(DEVICE)
blunted_model.eval()
print("✓ Blunted model loaded")

# ================= LOAD XRV PSPNet MODEL =================
print("🫁 Loading TorchXRayVision PSPNet...")
xrv_model = xrv.baseline_models.chestx_det.PSPNet()
xrv_model.eval()

xrv_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512)
])
print("✓ XRV model loaded")

# ================= HELPER FUNCTIONS =================
def normalize_for_unet(img_gray: np.ndarray, target_size=(512, 512)) -> torch.Tensor:
    """Resize + z-score normalize grayscale image and return tensor [1,1,H,W]."""
    img_resized = cv2.resize(img_gray, target_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-6)
    return torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)

def resize_back(prob_2d_t: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
    """Resize a 2D torch tensor back to original H,W using nearest, return numpy."""
    return (
        F.interpolate(prob_2d_t[None, None], size=(orig_h, orig_w), mode="nearest")[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

def load_for_xrv(img_path: Path) -> torch.Tensor:
    """Load image for XRV PSPNet and return tensor [1,1,512,512]."""
    orig_img_np = skimage.io.imread(str(img_path))
    img_norm = xrv.datasets.normalize(orig_img_np, 255)

    if img_norm.ndim == 3:
        img_norm = img_norm.mean(2)[None, ...]
    else:
        img_norm = img_norm[None, ...]

    img_xrv = xrv_transform(img_norm)
    img_xrv_t = torch.from_numpy(img_xrv).float().unsqueeze(0)
    return img_xrv_t

def predict_binary_mask(model, img_tensor: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
    """Run one binary model and return resized probability map."""
    with torch.no_grad():
        logits = model(img_tensor)[0, 0]   # [H,W]
        probs = torch.sigmoid(logits)

    return resize_back(probs, orig_h, orig_w)

# ---- U-NET COLORS ----
# OpenCV uses BGR
unet_colors = [
    (0, 255, 255),   # Spinous - yellow in BGR
    (255, 0, 0),     # Costo - blue
    (0, 0, 255)      # Blunted - red
]

# ---- XRV COLORS ----
xrv_colors = [
    (0, 128, 255),   # Left Clavicle
    (255, 0, 128),   # Right Clavicle
    (255, 128, 0),   # Left Scapula
    (128, 0, 255),   # Right Scapula
    (0, 255, 255),   # Left Lung
    (0, 255, 255),   # Right Lung
]

# ================= THRESHOLDS =================
SPINOUS_THRESHOLD = 0.3
COSTO_THRESHOLD = 0.5
BLUNTED_THRESHOLD = 0.5

# ================= PROCESS ALL IMAGES =================
print(f"\n📦 Processing {len(IMAGE_PATHS)} images...")

for idx, img_path in enumerate(IMAGE_PATHS, start=1):
    print(f"\n--- [{idx}/{len(IMAGE_PATHS)}] {img_path.name} ---")

    # ================= LOAD IMAGE =================
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"⚠️ Could not read image (skipping): {img_path}")
        continue

    orig_h, orig_w = img_gray.shape
    stem = img_path.stem

    # Save original grayscale copy
    cv2.imwrite(str(SAVE_DIR / f"{stem}_original.png"), img_gray)

    # ================= SEPARATE U-NET INFERENCE =================
    img_tensor = normalize_for_unet(img_gray, TARGET_SIZE).to(DEVICE)

    spinous_prob = predict_binary_mask(spinous_model, img_tensor, orig_h, orig_w)
    costo_prob = predict_binary_mask(costo_model, img_tensor, orig_h, orig_w)
    blunted_prob = predict_binary_mask(blunted_model, img_tensor, orig_h, orig_w)

    # ================= THRESHOLD MASKS =================
    spinous_mask = (spinous_prob > SPINOUS_THRESHOLD).astype(np.uint8) * 255
    costo_mask = (costo_prob > COSTO_THRESHOLD).astype(np.uint8) * 255

    # Optional gating: only show blunted where costo also exists
    blunted_mask = (
        (blunted_prob > BLUNTED_THRESHOLD) & (costo_prob > COSTO_THRESHOLD)
    ).astype(np.uint8) * 255

    # Remove overlap so blunted visually overrides costo
    costo_mask[blunted_mask > 0] = 0

    # Save probability maps too (useful for debugging)
    cv2.imwrite(str(SAVE_DIR / f"{stem}_spinous_prob.png"), (spinous_prob * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite(str(SAVE_DIR / f"{stem}_costo_prob.png"), (costo_prob * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite(str(SAVE_DIR / f"{stem}_blunted_prob.png"), (blunted_prob * 255).clip(0, 255).astype(np.uint8))

    # Save binary masks
    cv2.imwrite(str(SAVE_DIR / f"{stem}_spinous.png"), spinous_mask)
    cv2.imwrite(str(SAVE_DIR / f"{stem}_costo.png"), costo_mask)
    cv2.imwrite(str(SAVE_DIR / f"{stem}_blunted.png"), blunted_mask)
    print("✅ Separate-model masks saved")

    # ================= XRV INFERENCE =================
    img_xrv = load_for_xrv(img_path)
    with torch.no_grad():
        pred = xrv_model(img_xrv)

    pred = torch.sigmoid(pred)[0].cpu().numpy()

    # ================= CREATE COMBINED OVERLAY =================
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()

    # --- apply separate U-Net overlays ---
    masks = [spinous_mask, costo_mask, blunted_mask]
    for mask, color in zip(masks, unet_colors):
        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[:, :] = color

        overlay = np.where(
            mask[..., None].astype(bool),
            cv2.addWeighted(overlay, 0.5, color_layer, 0.5, 0),
            overlay
        )

    # --- apply XRV overlays ---
    H, W = img_gray.shape
    for i in range(min(len(pred), len(xrv_colors))):
        mask_resized = cv2.resize(pred[i], (W, H))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[:, :] = xrv_colors[i]

        overlay = np.where(
            mask_binary[..., None].astype(bool),
            cv2.addWeighted(overlay, 0.5, color_layer, 0.5, 0),
            overlay
        )

    final_overlay = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(SAVE_DIR / f"{stem}_combined_overlay_distinct.png"), final_overlay)
    print("✅ Combined overlay saved")

print("\n📁 All results saved to:", SAVE_DIR)
print("🎉 Done.")