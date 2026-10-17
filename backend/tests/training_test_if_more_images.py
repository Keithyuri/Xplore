import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import random

# ================= CONFIG =================
DATASET_PATH = Path("dataset")
IMAGES_PATH = DATASET_PATH / "images"
MASKS_PATH = DATASET_PATH / "masks"
OUTPUT_DIR = Path("spinous_training_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 150
LR = 1e-4
PATIENCE = 20                  # early stopping patience
MIN_EPOCH_BEFORE_SAVE = 10     # warm-up period

OUTPUT_DIR.mkdir(exist_ok=True)

# ================= DATA LOADING =================
def load_image_and_mask(img_path, mask_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise FileNotFoundError(f"{img_path} or {mask_path} not found")

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)

    mask = (mask > 127).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0)

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

# ================= SOFT DICE =================
def soft_dice(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    return (2 * intersection + eps) / (probs.sum() + target.sum() + eps)

# ================= MODEL =================
model = smp.PSPNet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
).to(DEVICE)

bce_loss = nn.BCEWithLogitsLoss()
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

        dice = soft_dice(logits, mask)
        loss = bce_loss(logits, mask) + (1 - dice)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
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

            dice = soft_dice(logits, mask)
            loss = bce_loss(logits, mask) + (1 - dice)

            val_loss += loss.item()
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
    if epoch + 1 >= MIN_EPOCH_BEFORE_SAVE and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
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


