
import torch
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import random
import torchvision.transforms.functional as TF
import time
import copy
import traceback
from datetime import datetime

# ================= CONFIG =================
DATASET_PATH = Path("dataset")
IMAGES_PATH = DATASET_PATH / "images" / "png"
MASKS_PATH = DATASET_PATH / "masks" / "fixed_png"
OUTPUT_DIR = Path("separate_binary_training_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
LR = 1e-4
PATIENCE = 30
SEED = 42
TARGET_SIZE = (512, 512)

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= CLASS SETUP =================
CLASS_CONFIG = {
    "spinous": 255,
    "costo": 192,
    "blunted": 64,
}

CLASS_LOSS_WEIGHTS = {
    "spinous": 1.0,
    "costo": 1.0,
    "blunted": 3.0,
}

# Binary segmentation thresholds for each anatomical class.
CLASS_THRESHOLDS = {
    "spinous": 0.3,
    "costo": 0.5,
    "blunted": 0.5,
}

# ================= TRAINING LOG =================
LOG_FILE = OUTPUT_DIR / (
    f"training_log_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
)

def log(message=""):
    message = str(message)
    print(message, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


log("=" * 70)
log("XPLORE - SEPARATE BINARY SEGMENTATION TRAINING")
log("=" * 70)
log(f"Training started: {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"PyTorch version: {torch.__version__}")
log(f"Device: {DEVICE}")
log(f"Epochs: {EPOCHS}")
log(f"Learning rate: {LR}")
log(f"Early stopping patience: {PATIENCE}")
log(f"Random seed: {SEED}")
log(f"Target image size: {TARGET_SIZE}")
log(f"Dataset path: {DATASET_PATH.resolve()}")
log(f"Images path: {IMAGES_PATH.resolve()}")
log(f"Masks path: {MASKS_PATH.resolve()}")
log(f"Classes: {CLASS_CONFIG}")
log(f"Class loss weights: {CLASS_LOSS_WEIGHTS}")
log(f"Class thresholds: {CLASS_THRESHOLDS}")
log(f"Log file: {LOG_FILE.resolve()}")
log("=" * 70)

# ================= DATA LOADING =================
def load_image_and_binary_mask(img_path, mask_path, target_value):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = cv2.resize(
        img,
        TARGET_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    mask = cv2.resize(
        mask,
        TARGET_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-6)
    img = torch.from_numpy(img).unsqueeze(0).float()

    binary_mask = (mask == target_value).astype(np.float32)
    binary_mask = torch.from_numpy(
        binary_mask
    ).unsqueeze(0).float()

    return img, binary_mask


# ================= AUGMENTATION =================
def augment(img, mask):
    if random.random() < 0.5:
        img = torch.flip(img, [-1])
        mask = torch.flip(mask, [-1])

    if random.random() < 0.7:
        angle = random.uniform(-15, 15)

        img = TF.rotate(
            img,
            angle,
            interpolation=TF.InterpolationMode.BILINEAR
        )

        mask = TF.rotate(
            mask,
            angle,
            interpolation=TF.InterpolationMode.NEAREST
        )

    return img, mask


# ================= LOSS =================
def binary_tversky_loss(logits, target, alpha=0.3, beta=0.7):
    probs = torch.sigmoid(logits)

    TP = (probs * target).sum()
    FP = ((1 - target) * probs).sum()
    FN = (target * (1 - probs)).sum()

    tversky = (
        (TP + 1e-6)
        / (TP + alpha * FP + beta * FN + 1e-6)
    )

    return 1 - tversky


def dice_score_from_logits(logits, target, threshold=None):
    probs = torch.sigmoid(logits)

    if threshold is not None:
        probs = (probs > threshold).float()

    dice = (
        (2 * (probs * target).sum() + 1e-6)
        / (probs.sum() + target.sum() + 1e-6)
    )

    return dice.item()


# ================= MODEL =================
def create_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    )

    return model.to(DEVICE)


# ================= DATA SPLIT =================
files = sorted(IMAGES_PATH.glob("*.png"))

if len(files) == 0:
    raise RuntimeError(
        f"No .png images found in {IMAGES_PATH.resolve()}"
    )

random.shuffle(files)

split = int(0.8 * len(files))
train_files = files[:split]
val_files = files[split:]

if len(train_files) == 0 or len(val_files) == 0:
    raise RuntimeError(
        "Train/validation split produced an empty set."
    )

log("")
log("=" * 70)
log("DATASET SUMMARY")
log("=" * 70)
log(f"Total images: {len(files)}")
log(f"Training images: {len(train_files)}")
log(f"Validation images: {len(val_files)}")

# Verify image-mask pairs before training.
for img_file in files:
    mask_file = MASKS_PATH / f"{img_file.stem}_mask.png"

    if not mask_file.is_file():
        raise FileNotFoundError(
            f"Missing mask for {img_file.name}: {mask_file}"
        )

log("All image-mask pairs verified.")
log("=" * 70)


# ================= TRAINING FUNCTION =================
def train_single_class(class_name, target_value, class_weight, threshold):

    log("")
    log("=" * 70)
    log(
        f"TRAINING: {class_name.upper()} "
        f"(Mask value: {target_value}, Threshold: {threshold})"
    )
    log("=" * 70)

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_binary_dice = -1.0
    best_val_soft_dice = -1.0
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    save_path = OUTPUT_DIR / f"best_{class_name}_model.pth"

    class_start_time = time.time()

    for epoch in range(EPOCHS):
        start = time.time()

        # ================= TRAIN =================
        model.train()
        train_loss_sum = 0.0

        for img_file in train_files:
            mask_file = MASKS_PATH / f"{img_file.stem}_mask.png"

            img, mask = load_image_and_binary_mask(
                img_file,
                mask_file,
                target_value
            )

            img, mask = augment(img, mask)

            img = img.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()

            logits = model(img)

            loss = (
                binary_tversky_loss(logits, mask)
                + class_weight
                * F.binary_cross_entropy_with_logits(logits, mask)
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_files)

        # ================= VALIDATION =================
        model.eval()

        val_soft_dice_sum = 0.0
        val_binary_dice_sum = 0.0

        with torch.no_grad():
            for img_file in val_files:
                mask_file = MASKS_PATH / f"{img_file.stem}_mask.png"

                img, mask = load_image_and_binary_mask(
                    img_file,
                    mask_file,
                    target_value
                )

                img = img.unsqueeze(0).to(DEVICE)
                mask = mask.unsqueeze(0).to(DEVICE)

                logits = model(img)

                soft_dice = dice_score_from_logits(
                    logits,
                    mask,
                    threshold=None
                )

                binary_dice = dice_score_from_logits(
                    logits,
                    mask,
                    threshold=threshold
                )

                val_soft_dice_sum += soft_dice
                val_binary_dice_sum += binary_dice

        avg_val_soft_dice = val_soft_dice_sum / len(val_files)
        avg_val_binary_dice = val_binary_dice_sum / len(val_files)

        elapsed = time.time() - start

        # ================= EPOCH LOG =================
        log(
            f"[{class_name}] "
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Soft Dice: {avg_val_soft_dice:.4f} | "
            f"Val Binary Dice ({threshold}): "
            f"{avg_val_binary_dice:.4f} | "
            f"Time: {int(elapsed // 60)}m "
            f"{int(elapsed % 60)}s"
        )

        # ================= MODEL CHECKPOINT =================
        # Select model using BINARY DICE, not soft Dice.
        improved = avg_val_binary_dice > best_val_binary_dice

        if improved:
            best_val_binary_dice = avg_val_binary_dice
            best_val_soft_dice = avg_val_soft_dice
            best_epoch = epoch + 1

            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, save_path)

            epochs_no_improve = 0

            log(
                f"Saved best {class_name} model | "
                f"Epoch: {best_epoch} | "
                f"Binary Dice ({threshold}): "
                f"{best_val_binary_dice:.4f} | "
                f"Soft Dice: {best_val_soft_dice:.4f} | "
                f"Path: {save_path}"
            )

        else:
            epochs_no_improve += 1

        # ================= EARLY STOPPING =================
        if epochs_no_improve >= PATIENCE:
            log(
                f"Early stopping for {class_name} "
                f"at epoch {epoch + 1}"
            )
            break

    # ================= CLASS SUMMARY =================
    class_elapsed = time.time() - class_start_time

    log("")
    log("-" * 70)
    log(f"FINISHED TRAINING: {class_name.upper()}")
    log(f"Best epoch: {best_epoch}")
    log(
        f"Best validation binary Dice ({threshold}): "
        f"{best_val_binary_dice:.4f}"
    )
    log(
        f"Soft Dice at best checkpoint: "
        f"{best_val_soft_dice:.4f}"
    )
    log(f"Total epochs completed: {epoch + 1}")
    log(
        f"Total training time: "
        f"{int(class_elapsed // 60)}m "
        f"{int(class_elapsed % 60)}s"
    )
    log(f"Saved model: {save_path.resolve()}")
    log("-" * 70)

    return {
        "class_name": class_name,
        "best_val_binary_dice": best_val_binary_dice,
        "best_val_soft_dice": best_val_soft_dice,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "epochs_completed": epoch + 1,
        "training_seconds": class_elapsed,
        "save_path": str(save_path),
        "state_dict": best_state_dict,
    }


# ================= TRAIN ALL 3 MODELS =================
results = {}
training_start_time = time.time()

try:
    for class_name, mask_value in CLASS_CONFIG.items():

        result = train_single_class(
            class_name=class_name,
            target_value=mask_value,
            class_weight=CLASS_LOSS_WEIGHTS[class_name],
            threshold=CLASS_THRESHOLDS[class_name]
        )

        results[class_name] = result

    # ================= FINAL SUMMARY =================
    total_elapsed = time.time() - training_start_time

    log("")
    log("=" * 70)
    log("FINAL TRAINING SUMMARY")
    log("=" * 70)

    for class_name, result in results.items():
        log(
            f"{class_name.upper():10s} | "
            f"Best Epoch: {result['best_epoch']:03d} | "
            f"Threshold: {result['threshold']:.2f} | "
            f"Best Binary Dice: "
            f"{result['best_val_binary_dice']:.4f} | "
            f"Soft Dice at checkpoint: "
            f"{result['best_val_soft_dice']:.4f} | "
            f"Saved: {result['save_path']}"
        )

    log("")
    log(
        f"Total training time: "
        f"{int(total_elapsed // 3600)}h "
        f"{int(total_elapsed % 3600 // 60)}m "
        f"{int(total_elapsed % 60)}s"
    )

    log(
        f"Training completed: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    log(f"Training log saved to: {LOG_FILE.resolve()}")
    log("=" * 70)

except KeyboardInterrupt:
    log("")
    log("TRAINING INTERRUPTED BY USER")
    log(
        f"Interrupted at: "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    )
    log("Previously saved model checkpoints are retained.")
    log(f"Training log: {LOG_FILE.resolve()}")
    raise

except Exception:
    log("")
    log("TRAINING FAILED WITH AN ERROR")
    log(traceback.format_exc())
    log(f"Training log: {LOG_FILE.resolve()}")
    raise
