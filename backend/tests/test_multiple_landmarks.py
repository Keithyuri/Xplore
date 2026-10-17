import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import random
import torchvision.transforms.functional as TF
import time

# ================= CONFIG =================
DATASET_PATH = Path("dataset")
IMAGES_PATH = DATASET_PATH / "images"
MASKS_PATH = DATASET_PATH / "masks"
OUTPUT_DIR = Path("spinous_training_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 300
LR = 1e-4
PATIENCE = 100
SEED = 42

TARGET_SIZE = (512, 512)

# ---- Reinforcement settings ----
REINFORCE_AFTER = 20
LR_UP = 1.0
LR_DOWN = 1.0
MIN_LR = 1e-5
MAX_LR = 5e-4

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

    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # ---- Normalize image ----
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.from_numpy(img).unsqueeze(0)

    # ---- 3-class mask ----
    spinous = (mask == 255).astype(np.float32)
    costo = (mask == 192).astype(np.float32)
    blunted = (mask == 64).astype(np.float32)  # 🔴 NEW

    mask = np.stack([spinous, costo, blunted], axis=0)
    mask = torch.from_numpy(mask)

    return img, mask

# ================= AUGMENTATION =================
def augment(img, mask):
    if random.random() < 0.5:
        img = torch.flip(img, [-1])
        mask = torch.flip(mask, [-1])

    if random.random() < 0.5:
        img = torch.flip(img, [-2])
        mask = torch.flip(mask, [-2])

    if random.random() < 0.7:
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

    if random.random() < 0.7:
        scale = random.uniform(0.9, 1.1)
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        img = TF.affine(img, 0, [tx, ty], scale, 0,
                        interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, 0, [tx, ty], scale, 0,
                         interpolation=TF.InterpolationMode.NEAREST)

    if random.random() < 0.5:
        contrast = random.uniform(0.9, 1.1)
        brightness = random.uniform(0.9, 1.1)
        img = img * contrast + brightness * img.mean()

    return img, mask

# ================= LOSS =================
def multiclass_tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1e-6):
    probs = torch.sigmoid(logits)
    loss = 0.0

    for c in range(probs.shape[1]):
        p = probs[:, c]
        t = target[:, c]

        TP = (p * t).sum()
        FP = ((1 - t) * p).sum()
        FN = (t * (1 - p)).sum()

        loss += 1 - (TP + eps) / (TP + alpha * FP + beta * FN + eps)

    return loss / probs.shape[1]

# ================= MODEL =================
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=1,
    classes=3  # 🔴 Spinous | Costo | Blunted Costo
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= DATASET SPLIT =================
image_files = sorted(IMAGES_PATH.glob("*.jpg"))
random.shuffle(image_files)
split = int(0.8 * len(image_files))
train_files = image_files[:split]
val_files = image_files[split:]

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(val_files)}")

# ================= TRAINING =================
best_val_dice = 0.0
best_epoch = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    start = time.time()
    print(f"\n🔹 Epoch {epoch+1}/{EPOCHS}")

    # -------- TRAIN --------
    model.train()
    train_loss = 0.0
    train_dice = [0.0, 0.0, 0.0]

    for img_file in train_files:
        mask_file = MASKS_PATH / f"{img_file.stem}_mask.jpg"
        img, mask = load_image_and_mask(img_file, mask_file)
        img, mask = augment(img, mask)

        img = img.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)

        optimizer.zero_grad()
        logits = model(img)

        loss = multiclass_tversky_loss(logits, mask) + \
               F.binary_cross_entropy_with_logits(logits, mask)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            for c in range(3):
                p = probs[:, c]
                t = mask[:, c]
                dice = (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
                train_dice[c] += dice.item()

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0.0
    val_dice = [0.0, 0.0, 0.0]

    with torch.no_grad():
        for img_file in val_files:
            mask_file = MASKS_PATH / f"{img_file.stem}_mask.jpg"
            img, mask = load_image_and_mask(img_file, mask_file)

            img = img.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE)

            logits = model(img)
            loss = multiclass_tversky_loss(logits, mask) + \
                   F.binary_cross_entropy_with_logits(logits, mask)

            val_loss += loss.item()
            probs = torch.sigmoid(logits)

            for c in range(3):
                p = probs[:, c]
                t = mask[:, c]
                dice = (2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
                val_dice[c] += dice.item()

    # ---- Averages ----
    train_loss /= len(train_files)
    val_loss /= len(val_files)

    train_dice = [d / len(train_files) for d in train_dice]
    val_dice = [d / len(val_files) for d in val_dice]

    avg_val_dice = sum(val_dice) / 3
    epoch_time = time.time() - start

    print(
        f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}\n"
        f"Train Dice → Spinous {train_dice[0]:.4f} | "
        f"Costo {train_dice[1]:.4f} | Blunted {train_dice[2]:.4f}\n"
        f"Val Dice   → Spinous {val_dice[0]:.4f} | "
        f"Costo {val_dice[1]:.4f} | Blunted {val_dice[2]:.4f}\n"
        f"LR {optimizer.param_groups[0]['lr']:.2e} | ⏱ {int(epoch_time//60)}m {int(epoch_time%60)}s"
    )

    # ================= REINFORCEMENT =================
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        print("✅ Best model saved")

        if epoch + 1 >= REINFORCE_AFTER:
            for pg in optimizer.param_groups:
                pg["lr"] = min(pg["lr"] * LR_UP, MAX_LR)
            print("🟢 Reinforcement applied (LR increased)")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epochs")

        if epoch + 1 >= REINFORCE_AFTER:
            for pg in optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * LR_DOWN, MIN_LR)

        if epochs_no_improve == 20:
            for pg in optimizer.param_groups:
                pg["lr"] = LR
            print("🔄 LR reset")

    if epochs_no_improve >= PATIENCE:
        print(f"\n🛑 Early stopping at epoch {epoch+1}")
        break

# ================= SUMMARY =================
print("\n🏁 Training complete")
print(f"🏆 Best Epoch: {best_epoch}")
print(f"🏆 Best Avg Val Dice: {best_val_dice:.4f}")
print(f"📁 Saved at: {OUTPUT_DIR / 'best_model.pth'}")
