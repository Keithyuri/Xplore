import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import random
import time
import torchvision.transforms.functional as TF

# ================= CONFIG =================
DATASET_PATH = Path("dataset")
IMAGES_PATH = DATASET_PATH / "images" / "png"
MASKS_PATH = DATASET_PATH / "masks" / "fixed_png"
OUTPUT_DIR = Path("spinous_training_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 300
LR = 1e-4
PATIENCE = 50
SEED = 42

NUM_CLASSES = 3  # background, spinous, costophrenic

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR.mkdir(exist_ok=True)

# ================= DATA LOADING =================
def load_image_and_mask(img_path, mask_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise FileNotFoundError(f"{img_path} or {mask_path} not found")

    # Normalize image
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

    # Convert mask to 0,1,2 classes
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[mask == 255] = 1  # spinous
    new_mask[mask == 127] = 2  # costophrenic

    mask = torch.from_numpy(new_mask.astype(np.int64))  # (H, W)

    return img, mask

# ================= AUGMENTATION =================
def augment(img, mask):
    """img: (1,H,W), mask: (H,W)"""
    # ---- FLIPS ----
    if random.random() < 0.5:
        img = torch.flip(img, [-1])
        mask = torch.flip(mask, [-1])

    if random.random() < 0.5:
        img = torch.flip(img, [-2])
        mask = torch.flip(mask, [-2])

    # ---- ROTATION ----
    if random.random() < 0.7:
        angle = random.uniform(-12, 12)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask.unsqueeze(0).float(), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

    # ---- SCALE + TRANSLATION ----
    if random.random() < 0.7:
        scale = random.uniform(0.9, 1.1)
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        img = TF.affine(img, angle=0, translate=[tx, ty], scale=scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask.unsqueeze(0).float(), angle=0, translate=[tx, ty], scale=scale, shear=0, interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()

    # ---- INTENSITY (IMAGE ONLY) ----
    if random.random() < 0.5:
        contrast = random.uniform(0.9, 1.1)
        brightness = random.uniform(0.9, 1.1)
        img = img * contrast + brightness * img.mean()

    return img, mask

# ================= MODEL =================
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = smp.losses.DiceLoss(mode="multiclass")

# ================= DATASET SPLIT =================
image_files = sorted(IMAGES_PATH.glob("*.png"))
random.shuffle(image_files)
split = int(0.8 * len(image_files))
train_files = image_files[:split]
val_files = image_files[split:]

print(f"Training images: {len(train_files)}, Validation images: {len(val_files)}")

# ================= TRAINING =================
best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0
reinforce_start_epoch = 20  # only apply reinforcement after 20 epochs

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    train_dice_total = 0.0

    for img_file in train_files:
        mask_file = MASKS_PATH / f"{img_file.stem}_mask.png"

        img, mask = load_image_and_mask(img_file, mask_file)
        img, mask = augment(img, mask)

        img = img.unsqueeze(0).to(DEVICE)          # (1,1,H,W)
        mask = mask.unsqueeze(0).to(DEVICE)        # (1,H,W)

        optimizer.zero_grad()
        logits = model(img)                        # (1,NUM_CLASSES,H,W)

        # Compute loss
        loss = criterion(logits, mask)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # -------- Train Dice per class --------
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)  # (1,C,H,W)
            dice_per_class = []
            for c in range(NUM_CLASSES):
                p = probs[:,c]
                t = (mask==c).float()
                dice = (2*(p*t).sum() + 1e-6)/(p.sum() + t.sum() + 1e-6)
                dice_per_class.append(dice.item())
            train_dice_total += np.mean(dice_per_class)

    avg_train_loss = train_loss / len(train_files)
    avg_train_dice = train_dice_total / len(train_files)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_dice_total = 0.0

    with torch.no_grad():
        for img_file in val_files:
            mask_file = MASKS_PATH / f"{img_file.stem}_mask.png"
            img, mask = load_image_and_mask(img_file, mask_file)

            img = img.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE)

            logits = model(img)
            loss = criterion(logits, mask)
            val_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            dice_per_class = []
            for c in range(NUM_CLASSES):
                p = probs[:,c]
                t = (mask==c).float()
                dice = (2*(p*t).sum() + 1e-6)/(p.sum() + t.sum() + 1e-6)
                dice_per_class.append(dice.item())
            val_dice_total += np.mean(dice_per_class)

    avg_val_loss = val_loss / len(val_files)
    avg_val_dice = val_dice_total / len(val_files)

    # -------- REINFORCEMENT LEARNING IDEA --------
    # If val_dice is high, slightly increase learning rate after 20 epochs
    if epoch >= reinforce_start_epoch and avg_val_dice > best_val_dice:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 1.05  # small positive reinforcement

    # -------- EARLY STOPPING & SAVE BEST MODEL --------
    epoch_time = time.time() - epoch_start_time
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)

    print(
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        f"Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.2e} | ⏱️ {minutes}m {seconds}s"
    )

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        print("✅ Best model saved")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch+1}")
        break

# ================= SUMMARY =================
print("\n🏁 Training complete")
print(f"🏆 Best Epoch: {best_epoch}")
print(f"🏆 Best Val Dice: {best_val_dice:.4f}")
print(f"📁 Model saved to: {OUTPUT_DIR / 'best_model.pth'}")
