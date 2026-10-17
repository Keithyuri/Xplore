
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from datetime import datetime
from io import BytesIO
import base64
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torchxrayvision as xrv
import torchvision
from PIL import Image, ImageDraw

app = FastAPI(title="Radiography Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================

# This script should be placed directly inside the backend folder.
BASE_DIR = Path(__file__).resolve().parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = (512, 512)

# ================= MODEL PATHS =================

MODEL_DIR_CANDIDATES = [
    BASE_DIR / "tests" / "separate_binary_training_results",
    BASE_DIR / "separate_binary_training_results",
]

def find_model_file(filename: str) -> Path:
    for folder in MODEL_DIR_CANDIDATES:
        candidate = folder / filename

        if candidate.is_file():
            return candidate

    searched = "\n".join(
        str(p / filename) for p in MODEL_DIR_CANDIDATES
    )

    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\n{searched}"
    )


SPINOUS_MODEL_PATH = find_model_file(
    "best_spinous_model.pth"
)

COSTO_MODEL_PATH = find_model_file(
    "best_costo_model.pth"
)

BLUNTED_MODEL_PATH = find_model_file(
    "best_blunted_model.pth"
)

# ================= OUTPUT FOLDER =================

OUTPUT_DIR = BASE_DIR / "segmentation_debug_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= COLOURS =================
# BGR values, converted to RGB by the overlay function.

SPINOUS_COLOR = (255, 255, 0)
COSTO_COLOR = (255, 0, 0)
BLUNTED_COLOR = (0, 0, 255)
CLAVICLE_COLOR = (255, 0, 128)
SCAPULA_COLOR = (128, 0, 255)
LUNG_COLOR = (0, 255, 255)


# ================= LOAD MODELS =================

def create_binary_unet():
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )


def load_binary_model(path: Path):
    model = create_binary_unet()

    model.load_state_dict(
        torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model = model.to(DEVICE)
    model.eval()

    print(f"Loaded {path.name}")

    return model


print("Using device:", DEVICE)

spinous_model = load_binary_model(SPINOUS_MODEL_PATH)
costo_model = load_binary_model(COSTO_MODEL_PATH)
blunted_model = load_binary_model(BLUNTED_MODEL_PATH)

print("Loading TorchXRayVision PSPNet...")

xrv_model = xrv.baseline_models.chestx_det.PSPNet()
xrv_model.eval()

xrv_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(512),
])

print("PSPNet loaded")


# ================= PREPROCESSING =================

def normalize_for_unet(
    img_gray: np.ndarray,
    target_size=(512, 512)
) -> torch.Tensor:

    img_resized = cv2.resize(
        img_gray,
        target_size
    ).astype(np.float32)

    img_resized = (
        (img_resized - img_resized.mean())
        / (img_resized.std() + 1e-6)
    )

    return (
        torch.from_numpy(img_resized)
        .unsqueeze(0)
        .unsqueeze(0)
    )


def resize_back(
    prob_2d_t: torch.Tensor,
    orig_h: int,
    orig_w: int
) -> np.ndarray:

    return (
        F.interpolate(
            prob_2d_t[None, None],
            size=(orig_h, orig_w),
            mode="nearest"
        )[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


def load_for_xrv_from_gray(
    img_gray: np.ndarray
) -> torch.Tensor:

    img_norm = xrv.datasets.normalize(
        img_gray,
        255
    )

    if img_norm.ndim == 3:
        img_norm = img_norm.mean(2)[None, ...]
    else:
        img_norm = img_norm[None, ...]

    img_xrv = xrv_transform(img_norm)

    return (
        torch.from_numpy(img_xrv)
        .float()
        .unsqueeze(0)
    )


# ================= BASE64 ENCODING =================

def encode_pil_image_to_base64(
    pil_img: Image.Image
) -> str:

    buffer = BytesIO()

    pil_img.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def encode_mask_to_base64(mask: np.ndarray) -> str:

    return encode_pil_image_to_base64(
        Image.fromarray(mask)
    )


def encode_overlay_to_base64(
    overlay: np.ndarray
) -> str:

    return encode_pil_image_to_base64(
        Image.fromarray(overlay, mode="RGBA")
    )


# ================= TRANSPARENT OVERLAYS =================

def create_transparent_overlay_array(
    mask_color_pairs,
    height: int,
    width: int
) -> np.ndarray:

    rgba = np.zeros(
        (height, width, 4),
        dtype=np.uint8
    )

    for mask, color_bgr in mask_color_pairs:

        if mask is None:
            continue

        active = mask > 0

        b, g, r = color_bgr

        rgba[active, 0] = r
        rgba[active, 1] = g
        rgba[active, 2] = b
        rgba[active, 3] = 150

    return rgba


# ================= CENTERING LINES =================

def create_centering_overlay(
    height: int,
    width: int
) -> np.ndarray:

    rgba = np.zeros(
        (height, width, 4),
        dtype=np.uint8
    )

    overlay = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # Green diagonal lines
    color = (0, 255, 0)
    thickness = 3

    cv2.line(
        overlay,
        (0, 0),
        (width - 1, height - 1),
        color,
        thickness,
        cv2.LINE_AA
    )

    cv2.line(
        overlay,
        (width - 1, 0),
        (0, height - 1),
        color,
        thickness,
        cv2.LINE_AA
    )

    active = overlay.any(axis=2)

    rgba[active, :3] = overlay[active, ::-1]
    rgba[active, 3] = 180

    return rgba


# ================= CENTERING + SCAPULAE =================
# IMPORTANT FIX:
# Both scapulae and green centering lines are combined.

def compose_centering_and_scapulae(
    scapula_overlay: np.ndarray,
    height: int,
    width: int
) -> np.ndarray:

    # Start with pink scapulae
    combined = scapula_overlay.copy()

    # Generate green centering lines
    lines = create_centering_overlay(
        height,
        width
    )

    # Identify pixels containing green lines
    line_pixels = lines[:, :, 3] > 0

    # Draw green lines on top of pink scapulae
    combined[line_pixels] = lines[line_pixels]

    return combined


# ================= OVERLAY ON ORIGINAL X-RAY =================

def overlay_on_gray(
    original_gray: np.ndarray,
    overlay_rgba: np.ndarray
) -> Image.Image:

    base = Image.fromarray(
        original_gray
    ).convert("RGBA")

    overlay = Image.fromarray(
        overlay_rgba,
        mode="RGBA"
    )

    return Image.alpha_composite(
        base,
        overlay
    ).convert("RGB")


# ================= ALL OUTPUTS GRID =================

def create_all_outputs_grid(
    original_gray: np.ndarray,
    overlays: dict[str, np.ndarray]
) -> Image.Image:

    panel_size = (320, 320)
    header_height = 28
    margin = 12
    gap = 12

    layout = [
        ("Original", None),
        ("Rotation", "rotation_overlay"),
        ("Clavicles", "clavicles_overlay"),

        ("Spinous", "spinous_overlay"),
        ("Costo", "costo_overlay"),
        ("Blunted", "blunted_overlay"),

        ("Scapula", "scapula_overlay"),
        ("Lung Fields", "lung_fields_overlay"),
        ("Centering", "centering_overlay"),
    ]

    cell_w = panel_size[0]
    cell_h = panel_size[1] + header_height

    grid_w = margin * 2 + 3 * cell_w + 2 * gap
    grid_h = margin * 2 + 3 * cell_h + 2 * gap

    canvas = Image.new(
        "RGB",
        (grid_w, grid_h),
        "#f5f5f5"
    )

    draw = ImageDraw.Draw(canvas)

    for i, (label, overlay_key) in enumerate(layout):

        row, col = divmod(i, 3)

        x = margin + col * (cell_w + gap)
        y = margin + row * (cell_h + gap)

        # White title bar
        draw.rectangle(
            (
                x,
                y,
                x + cell_w - 1,
                y + header_height - 1
            ),
            fill="white"
        )

        draw.text(
            (x + 8, y + 7),
            label,
            fill="black"
        )

        if overlay_key is None:

            image = Image.fromarray(
                original_gray
            ).convert("RGB")

        else:

            image = overlay_on_gray(
                original_gray,
                overlays[overlay_key]
            )

        image = image.resize(
            panel_size,
            Image.Resampling.BILINEAR
        )

        canvas.paste(
            image,
            (x, y + header_height)
        )

    return canvas


# ================= SAVE OUTPUTS =================

def save_outputs(
    stem: str,
    original_gray: np.ndarray,
    outputs: dict[str, np.ndarray]
):

    sample_dir = OUTPUT_DIR / stem

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(sample_dir / "original.png"),
        original_gray
    )

    for name, array in outputs.items():

        out_path = sample_dir / f"{name}.png"

        if array.ndim == 2:

            cv2.imwrite(
                str(out_path),
                array
            )

        elif array.ndim == 3 and array.shape[2] == 4:

            Image.fromarray(
                array,
                mode="RGBA"
            ).save(out_path)

        elif array.ndim == 3 and array.shape[2] == 3:

            Image.fromarray(
                array,
                mode="RGB"
            ).save(out_path)

        else:

            raise ValueError(
                f"Unsupported output shape for "
                f"{name}: {array.shape}"
            )

    return sample_dir


# ================= BINARY U-NET INFERENCE =================

def predict_binary_mask(
    model,
    img_tensor,
    orig_h,
    orig_w
):

    with torch.no_grad():

        logits = model(img_tensor)[0, 0]
        probs = torch.sigmoid(logits)

    return resize_back(
        probs,
        orig_h,
        orig_w
    )


# ================= RUN SEGMENTATION =================

def run_segmentation(
    img_gray: np.ndarray,
    save_stem: str | None = None
):

    orig_h, orig_w = img_gray.shape

    # -------- U-Net inference --------

    img_tensor = normalize_for_unet(
        img_gray,
        TARGET_SIZE
    ).to(DEVICE)

    spinous_prob = predict_binary_mask(
        spinous_model,
        img_tensor,
        orig_h,
        orig_w
    )

    costo_prob = predict_binary_mask(
        costo_model,
        img_tensor,
        orig_h,
        orig_w
    )

    blunted_prob = predict_binary_mask(
        blunted_model,
        img_tensor,
        orig_h,
        orig_w
    )

    # -------- Binary thresholds --------

    spinous_mask = (
        (spinous_prob > 0.3).astype(np.uint8) * 255
    )

    costo_mask = (
        (costo_prob > 0.5).astype(np.uint8) * 255
    )

    blunted_mask = (
        (
            (blunted_prob > 0.5)
            & (costo_prob > 0.5)
        ).astype(np.uint8) * 255
    )

    # Blunted takes visual priority
    costo_mask[blunted_mask > 0] = 0

    # -------- TorchXRayVision inference --------

    img_xrv = load_for_xrv_from_gray(img_gray)

    with torch.no_grad():
        pred = xrv_model(img_xrv)

    pred = torch.sigmoid(pred)[0].cpu().numpy()

    masks = [
        np.zeros(
            (orig_h, orig_w),
            dtype=np.uint8
        )
        for _ in range(6)
    ]

    for i in range(min(6, len(pred))):

        masks[i] = (
            (
                cv2.resize(
                    pred[i],
                    (orig_w, orig_h)
                ) > 0.5
            ).astype(np.uint8) * 255
        )

    (
        left_clavicle_mask,
        right_clavicle_mask,
        left_scapula_mask,
        right_scapula_mask,
        left_lung_mask,
        right_lung_mask
    ) = masks

    clavicles_mask = np.maximum(
        left_clavicle_mask,
        right_clavicle_mask
    )

    # -------- Rotation overlay --------

    rotation_overlay_rgba = create_transparent_overlay_array(
        [
            (left_clavicle_mask, CLAVICLE_COLOR),
            (right_clavicle_mask, CLAVICLE_COLOR),
            (spinous_mask, SPINOUS_COLOR),
        ],
        orig_h,
        orig_w
    )

    # -------- Clavicles overlay --------

    clavicles_overlay_rgba = create_transparent_overlay_array(
        [
            (left_clavicle_mask, CLAVICLE_COLOR),
            (right_clavicle_mask, CLAVICLE_COLOR),
        ],
        orig_h,
        orig_w
    )

    # -------- Spinous overlay --------

    spinous_overlay_rgba = create_transparent_overlay_array(
        [
            (spinous_mask, SPINOUS_COLOR)
        ],
        orig_h,
        orig_w
    )

    # -------- Costophrenic overlay --------

    costo_overlay_rgba = create_transparent_overlay_array(
        [
            (costo_mask, COSTO_COLOR)
        ],
        orig_h,
        orig_w
    )

    # -------- Blunted overlay --------

    blunted_overlay_rgba = create_transparent_overlay_array(
        [
            (blunted_mask, BLUNTED_COLOR)
        ],
        orig_h,
        orig_w
    )

    # -------- Scapula overlay --------

    scapula_overlay_rgba = create_transparent_overlay_array(
        [
            (left_scapula_mask, SCAPULA_COLOR),
            (right_scapula_mask, SCAPULA_COLOR),
        ],
        orig_h,
        orig_w
    )

    # -------- Lung fields overlay --------

    lung_fields_overlay_rgba = create_transparent_overlay_array(
        [
            (left_lung_mask, LUNG_COLOR),
            (right_lung_mask, LUNG_COLOR),
        ],
        orig_h,
        orig_w
    )

    # ================= CENTERING FIX =================
    # Combine both pink scapulae and green lines.

    centering_overlay_rgba = compose_centering_and_scapulae(
        scapula_overlay_rgba,
        orig_h,
        orig_w
    )

    # -------- Collect all overlays --------

    overlay_arrays = {
        "rotation_overlay": rotation_overlay_rgba,
        "clavicles_overlay": clavicles_overlay_rgba,
        "spinous_overlay": spinous_overlay_rgba,
        "costo_overlay": costo_overlay_rgba,
        "blunted_overlay": blunted_overlay_rgba,
        "scapula_overlay": scapula_overlay_rgba,
        "lung_fields_overlay": lung_fields_overlay_rgba,
        "centering_overlay": centering_overlay_rgba,
    }

    # ================= GENERATE GRID =================

    all_output_grid = create_all_outputs_grid(
        img_gray,
        overlay_arrays
    )

    # ================= SAVE DEBUG OUTPUTS =================

    if save_stem:

        raw_masks = {
            "spinous_mask": spinous_mask,
            "costo_mask": costo_mask,
            "blunted_mask": blunted_mask,
            "left_clavicle_mask": left_clavicle_mask,
            "right_clavicle_mask": right_clavicle_mask,
            "clavicles_mask": clavicles_mask,
            "left_scapula_mask": left_scapula_mask,
            "right_scapula_mask": right_scapula_mask,
            "left_lung_mask": left_lung_mask,
            "right_lung_mask": right_lung_mask,
        }

        folder = save_outputs(
            save_stem,
            img_gray,
            {
                **raw_masks,
                **overlay_arrays
            }
        )

        # Save 3x3 panel grid
        all_output_grid.save(
            folder / "all_outputs_grid.png"
        )

        # Save centering + scapulae on X-ray
        overlay_on_gray(
            img_gray,
            centering_overlay_rgba
        ).save(
            folder / "centering_with_scapulae.png"
        )

        print(
            "Saved segmentation images to:",
            folder
        )

    # ================= RETURN API OUTPUT =================

    return {
        "spinous_mask": encode_mask_to_base64(spinous_mask),
        "costo_mask": encode_mask_to_base64(costo_mask),
        "blunted_mask": encode_mask_to_base64(blunted_mask),

        "left_clavicle_mask": encode_mask_to_base64(left_clavicle_mask),
        "right_clavicle_mask": encode_mask_to_base64(right_clavicle_mask),
        "clavicles_mask": encode_mask_to_base64(clavicles_mask),

        "left_scapula_mask": encode_mask_to_base64(left_scapula_mask),
        "right_scapula_mask": encode_mask_to_base64(right_scapula_mask),

        "left_lung_mask": encode_mask_to_base64(left_lung_mask),
        "right_lung_mask": encode_mask_to_base64(right_lung_mask),

        **{
            name: encode_overlay_to_base64(array)
            for name, array in overlay_arrays.items()
        },

        "all_output_grid": encode_pil_image_to_base64(
            all_output_grid
        ),
    }


# ================= API ROOT =================

@app.get("/")
def root():

    return {
        "message": "Radiography Segmentation API is running"
    }


# ================= SEGMENT ENDPOINT =================

@app.post("/segment")
async def segment(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        img_gray = cv2.imdecode(
            nparr,
            cv2.IMREAD_GRAYSCALE
        )

        if img_gray is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Generate unique folder for each image
        raw_stem = Path(
            file.filename or "uploaded"
        ).stem

        safe_stem = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            raw_stem
        )[:60] or "uploaded"

        save_stem = (
            f"{safe_stem}_"
            f"{datetime.now():%Y%m%d_%H%M%S_%f}"
        )

        result = run_segmentation(
            img_gray,
            save_stem=save_stem
        )

        return JSONResponse(
            content=result
        )

    except HTTPException:
        raise

    except Exception as e:

        print(
            "SEGMENT ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
