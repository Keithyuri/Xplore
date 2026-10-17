import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import random
from skimage.morphology import skeletonize

# ================= CONFIG =================
DATASET_PATH = Path("dataset")
IMAGES_PATH = DATASET_PATH / "images"
MASKS_PATH = DATASET_PATH / "masks"
OUTPUT_DIR = Path("spinous_training_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 300
LR = 1e-4
PATIENCE = 20


OUTPUT_DIR.mkdir(exist_ok=True)

# ================= DATA LOADING =================
def skeletonize_mask(mask):
    skeleton = skeletonize(mask).astype(np.uint8)
    # Dilate skeleton more aggressively (thicker mask)
    mask_dilated = cv2.dilate(skeleton, np.ones((3,3), np.uint8), iterations=2)
    return mask_dilated

def load_image_and_mask(img_path, mask_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise FileNotFoundError(f"{img_path} or {mask_path} not found")

    # Normalize image to zero mean, unit variance
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)

    # Convert mask to binary and skeletonize + dilate
    mask = (mask > 127).astype(np.uint8)
    mask = skeletonize_mask(mask)
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    return img, mask

# ================= AUGMENTATION =================
def augment(img, mask):
    if random.random() > 0.5:
        img = torch.flip(img, [-1])
        mask = torch.flip(mask, [-1])
    if random.random() > 0.5:
        img = torch.flip(img, [-2])
        mask = torch.flip(mask, [-2])
    return img, mask

# ================= Tversky LOSS =================
def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1e-6):
    probs = torch.sigmoid(logits)
    TP = (probs * target).sum()
    FP = ((1 - target) * probs).sum()
    FN = (target * (1 - probs)).sum()
    return 1 - (TP + eps) / (TP + alpha * FP + beta * FN + eps)

# ================= MODEL =================
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= DATASET SPLIT =================
image_files = sorted(IMAGES_PATH.glob("*.jpg"))
if len(image_files) == 0:
    raise ValueError("No images found")

random.seed(42)
random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")

# ================= TRAINING =================
best_val_dice = 0.0
best_val_loss = float("inf")
best_epoch = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for img_file in train_files:
        mask_file = MASKS_PATH / f"{img_file.stem}_mask.jpg"
        img, mask = load_image_and_mask(img_file, mask_file)
        img, mask = augment(img, mask)

        img = img.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)

        optimizer.zero_grad()
        logits = model(img)

        if logits.shape[2:] != mask.shape[2:]:
            mask = F.interpolate(mask, size=logits.shape[2:], mode="bilinear", align_corners=False)

        loss = tversky_loss(logits, mask) + F.binary_cross_entropy_with_logits(logits, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice = (2 * (probs * mask).sum() + 1e-6) / (probs.sum() + mask.sum() + 1e-6)
            train_dice += dice.item()

    avg_train_loss = train_loss / len(train_files)
    avg_train_dice = train_dice / len(train_files)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for img_file in val_files:
            mask_file = MASKS_PATH / f"{img_file.stem}_mask.jpg"
            img, mask = load_image_and_mask(img_file, mask_file)

            img = img.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE)

            logits = model(img)
            if logits.shape[2:] != mask.shape[2:]:
                mask = F.interpolate(mask, size=logits.shape[2:], mode="bilinear", align_corners=False)

            loss = tversky_loss(logits, mask) + F.binary_cross_entropy_with_logits(logits, mask)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            dice = (2 * (probs * mask).sum() + 1e-6) / (probs.sum() + mask.sum() + 1e-6)
            val_dice += dice.item()

    avg_val_loss = val_loss / len(val_files)
    avg_val_dice = val_dice / len(val_files)

    print(
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Train Dice: {avg_train_dice:.4f} | "
        f"Val Dice: {avg_val_dice:.4f}"
    )

    # -------- EARLY STOPPING --------
    if  avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        print("✅ Best model updated")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch+1}")
        break

# ================= SUMMARY =================
print("\n🏁 Training complete")
print(f"🏆 Best Epoch: {best_epoch}")
print(f"🏆 Best Validation Loss: {best_val_loss:.4f}")
print(f"📁 Best model saved at: {OUTPUT_DIR / 'best_model.pth'}")
