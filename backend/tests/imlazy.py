%reload_ext autoreload
%autoreload 2
import numpy as np
import skimage
import torch
import torchvision
import matplotlib.pyplot as plt
import torchxrayvision as xrv
import cv2
from PIL import Image

model = xrv.baseline_models.chestx_det.PSPNet()


# Path to the source image (keep this so we can reference the original later)
image_path = "../../../torchxrayvision-main/images/00000005_000.png"
# Load original image (uint8) for display and later resizing
orig_img_np = skimage.io.imread(image_path)
# Normalize a working copy for the model
img = xrv.datasets.normalize(orig_img_np, 255) # convert 8-bit image to [-1024, 1024] range
if img.ndim == 3:
    img = img.mean(2)[None, ...] # Make single color channel
else:
    img = img[None, ...] # Already single channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])

img = transform(img)
img = torch.from_numpy(img).float() # Ensure tensor is float32
img = img.unsqueeze(0) # Add batch dimension
with torch.no_grad():
    pred = model(img)

plt.figure(figsize = (26,5))
plt.subplot(1, len(model.targets) + 1, 1)
# Prepare image for display (convert to numpy and squeeze channel if necessary)
if isinstance(img, torch.Tensor):
    img_disp = img[0].cpu().numpy()
else:
    img_disp = np.array(img[0])
# img_disp could be (1,H,W) or (H,W,1); squeeze to (H,W)
if img_disp.ndim == 3:
    # common channel-first (1,H,W)
    if img_disp.shape[0] == 1:
        img_disp = img_disp[0]
    # channel-last (H,W,1)
    elif img_disp.shape[2] == 1:
        img_disp = img_disp[:,:,0]
plt.imshow(img_disp, cmap='gray')
for i in range(len(model.targets)):
    plt.subplot(1, len(model.targets) + 1, i+2)
    # Convert prediction to numpy if it's a torch tensor
    if isinstance(pred, torch.Tensor):
        pred_disp = pred[0, i].cpu().numpy()
    else:
        pred_disp = np.array(pred[0, i])
    plt.imshow(pred_disp)
    plt.title(model.targets[i])
    plt.axis('off')
plt.tight_layout()
pred = 1 / (1 + np.exp(-pred))  # sigmoid
pred[pred < 0.5] = 0
pred[pred > 0.5] = 1
plt.figure(figsize = (26,5))
plt.subplot(1, len(model.targets) + 1, 1) #model.targets is to check how many targets there are
# Convert and squeeze image for display
if isinstance(img, torch.Tensor):
    img_disp = img[0].cpu().numpy()
else:
    img_disp = np.array(img[0])
if img_disp.ndim == 3:
    if img_disp.shape[0] == 1:
        img_disp = img_disp[0]
    elif img_disp.shape[2] == 1:
        img_disp = img_disp[:,:,0]
plt.imshow(img_disp, cmap='gray')
for i in range(len(model.targets)):
    plt.subplot(1, len(model.targets) + 1, i+2)
    if isinstance(pred, torch.Tensor):
        pred_disp = pred[0, i].cpu().numpy()
    else:
        pred_disp = np.array(pred[0, i])
    plt.imshow(pred_disp, cmap='gray')
    plt.title(model.targets[i])
    plt.axis('off')
plt.tight_layout()


# ==========================================
# 6️⃣ Draw Contours Overlay
# ==========================================
# Convert grayscale image to RGB
img_rgb = cv2.cvtColor((img_disp * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Get predicted masks
pred_np = pred[0].cpu().numpy()

# Colors for each target (BGR for cv2)
colors = [(0,255,0), (255,0,0), (0,0,255)]  # lungs, heart, clavicles

# Draw contours
for i, target in enumerate(model.targets):
    mask = (pred_np[i] > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, colors[i % len(colors)], 2)

plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
plt.title("Segmentation Contours Overlay")
plt.axis("off")
plt.show()

# ==========================================
# 6️⃣ Draw Contours Overlay (Extended Targets)
# ==========================================

# Convert grayscale image to RGB
img_rgb = cv2.cvtColor((img_disp * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Get predicted masks
pred_np = pred[0].cpu().numpy()

# Define extended anatomical targets and colors
extended_targets = [
    "left_clavicle", "right_clavicle",
    "left_scapula", "right_scapula",
    "right_lung",
    "left_hilus_pulmonis", "right_hilus_pulmonis",
    "heart", "aorta",
    "facies_diaphragmatica",
    "mediastinum", "weasand", "spine"
]

# 13 distinct RGB colors
colors = [
    (255, 0, 0),      # left clavicle – red
    (0, 255, 0),      # right clavicle – green
    (0, 0, 255),      # left scapula – blue
    (255, 255, 0),    # right scapula – yellow
    (255, 0, 255),    # right lung – magenta
    (0, 255, 255),    # left hilus pulmonis – cyan
    (128, 0, 255),    # right hilus pulmonis – purple
    (255, 128, 0),    # heart – orange
    (128, 128, 0),    # aorta – olive
    (0, 128, 255),    # facies diaphragmatica – sky blue
    (128, 255, 128),  # mediastinum – light green
    (192, 64, 0),     # weasand – brown
    (64, 64, 64)      # spine – gray
]

# Make sure we don’t exceed model channels
num_channels = min(pred_np.shape[0], len(extended_targets))

# Draw contours for each detected mask
for i in range(num_channels):
    mask = (pred_np[i] > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, colors[i % len(colors)], 2)

# Display result
plt.figure(figsize=(10,10))
plt.imshow(img_rgb)
plt.title("Segmentation Contours Overlay (Extended Targets)")
plt.axis("off")
plt.show()

# ==========================================
# 6️⃣ Draw Contours Overlay on Original Image
# ==========================================
from PIL import Image  # ✅ make sure this is included

# 1️⃣ We already loaded the original full-size image as orig_img_np earlier
# orig_img_np is a uint8 grayscale array
# Convert to RGB for contour drawing
img_rgb = cv2.cvtColor(orig_img_np, cv2.COLOR_GRAY2RGB)

# 2️⃣ Get predicted masks from model output (C, H, W) or (1, C, H, W) depending on model
pred_np = pred[0].cpu().numpy()  # shape: (C, H, W)

# 3️⃣ Define anatomical targets and colors
extended_targets = [
    "left_clavicle", "right_clavicle",
    "left_scapula", "right_scapula",
    "right_lung",
    "left_hilus_pulmonis", "right_hilus_pulmonis",
    "heart", "aorta",
    "facies_diaphragmatica",
    "mediastinum", "weasand", "spine"
],
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (128, 128, 0), (0, 128, 255), (128, 255, 128), (192, 64, 0), (64, 64, 64),
]

# Limit channels to available predictions
num_channels = min(pred_np.shape[0], len(extended_targets))

# 4️⃣ Resize each mask back to original image size
H, W = orig_img_np.shape[:2]

for i in range(num_channels):
    # pred_np[i] is in model output resolution (e.g., 512x512). Resize to original image size (W,H)
    mask_resized = cv2.resize(pred_np[i], (W, H))  # note cv2 resize expects (width, height)
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_rgb, contours, -1, colors[i % len(colors)], 2)

# 5️⃣ Show overlay on real image
plt.figure(figsize=(10,10))
plt.imshow(img_rgb)
plt.title("Segmentation Contours Overlay (Original Image)")
plt.axis("off")
plt.show()

# 1️⃣ Convert grayscale original image to RGB
img_rgb = cv2.cvtColor(orig_img_np, cv2.COLOR_GRAY2RGB)

# 2️⃣ Get predicted masks from model output
pred_np = pred[0].cpu().numpy()  # (C, H, W)

# 3️⃣ Define extended targets and colors
extended_targets = [
    "left_clavicle", "right_clavicle",
    "left_scapula", "right_scapula",
    "right_lung",
    "left_hilus_pulmonis", "right_hilus_pulmonis",
    "heart", "aorta",
    "facies_diaphragmatica",
    "mediastinum", "weasand", "spine"
]

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (128, 128, 0), (0, 128, 255), (128, 255, 128), (192, 64, 0), (64, 64, 64)
]

num_channels = min(pred_np.shape[0], len(extended_targets))
H, W = orig_img_np.shape[:2]

# 4️⃣ Create overlay layer
overlay = img_rgb.copy()

# 5️⃣ Fill each region semi-transparently
for i in range(num_channels):
    mask_resized = cv2.resize(pred_np[i], (W, H))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    # Create a color mask for this region
    color_mask = np.zeros_like(img_rgb, dtype=np.uint8)
    color_mask[:, :] = colors[i % len(colors)]
    
    # Blend region onto overlay
    overlay = np.where(mask_binary[..., None].astype(bool),
                       cv2.addWeighted(overlay, 0.5, color_mask, 0.5, 0),
                       overlay)
    
    # Optionally draw a contour outline
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, colors[i % len(colors)], 2)

# 6️⃣ Display final overlay
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Segmentation Highlight Overlay (Original Image)")
plt.axis("off")
plt.show()
# 1️⃣ Convert grayscale original image to RGB
img_rgb = cv2.cvtColor(orig_img_np, cv2.COLOR_GRAY2RGB)

# 2️⃣ Get predicted masks from model output (C, H, W)
pred_np = pred[0].cpu().numpy()

# 3️⃣ Define extended anatomical targets and colors
extended_targets = [
    "left_clavicle", "right_clavicle",
    "left_scapula", "right_scapula",
    "right_lung",
    "left_hilus_pulmonis", "right_hilus_pulmonis",
    "heart", "aorta",
    "facies_diaphragmatica",
    "mediastinum", "weasand", "spine"
]

colors = [
    (255, 0, 0),      # left clavicle – red
    (0, 255, 0),      # right clavicle – green
    (0, 0, 255),      # left scapula – blue
    (255, 255, 0),    # right scapula – yellow
    (255, 0, 255),    # right lung – magenta
    (0, 255, 255),    # left hilus pulmonis – cyan
    (128, 0, 255),    # right hilus pulmonis – purple
    (255, 128, 0),    # heart – orange
    (128, 128, 0),    # aorta – olive
    (0, 128, 255),    # facies diaphragmatica – sky blue
    (128, 255, 128),  # mediastinum – light green
    (192, 64, 0),     # weasand – brown
    (64, 64, 64)      # spine – gray
]

# 4️⃣ Limit number of channels to what's available
num_channels = min(pred_np.shape[0], len(extended_targets))

# 5️⃣ Resize each mask back to match original image
H, W = orig_img_np.shape[:2]

for i in range(num_channels):
    mask_resized = cv2.resize(pred_np[i], (W, H))  # Resize to original image size
    mask_binary = (mask_resized > 0.5).astype(np.uint8)

    # Find and draw external contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cv2.drawContours(img_rgb, contours, -1, colors[i % len(colors)], 2)

# 6️⃣ Display result
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
#plt.title("Segmentation Contours Overlay (All Structures)")
plt.axis("off")
plt.show()
# ==========================================
#  Draw Thin Contours + Labels on Original Image
# ==========================================
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Convert grayscale original image to RGB
img_rgb = cv2.cvtColor(orig_img_np, cv2.COLOR_GRAY2RGB)

# 2️⃣ Get predicted masks from model output (C, H, W)
pred_np = pred[0].cpu().numpy()

# 3️⃣ Define extended anatomical targets and colors
extended_targets = [
    "left_clavicle", "right_clavicle",
    "left_scapula", "right_scapula", "left_lung",
    "right_lung",
    "left_hilus_pulmonis", "right_hilus_pulmonis",
    "heart", "aorta",
    "facies_diaphragmatica",
    "mediastinum", "weasand", "spine"
]

colors = [
    (255, 0, 0),      # left clavicle – red
    (0, 255, 0),      # right clavicle – green
    (0, 0, 255),      # left scapula – blue
    (255, 255, 0),    # right scapula – yellow
    (255, 0, 255),    # right lung – magenta
    (0, 255, 255),    # left hilus pulmonis – cyan
    (128, 0, 255),    # right hilus pulmonis – purple
    (255, 128, 0),    # heart – orange
    (128, 128, 0),    # aorta – olive
    (0, 128, 255),    # facies diaphragmatica – sky blue
    (128, 255, 128),  # mediastinum – light green
    (192, 64, 0),     # weasand – brown
    (64, 64, 64)      # spine – gray
]
# ✅ Display-friendly names (same order as extended_targets)
display_names = [
    "Left Clavicle", "Right Clavicle",
    "Left Scapula", "Right Scapula", "Left Lung",
    "Right Lung",
    "Left Hilum", "Right Hilum",
    "Heart", "Aorta",
    "Diaphragm",
    "Mediastinum", "Oesophagus" , "Spine"
]

# 4️⃣ Resize masks to match original image
H, W = orig_img_np.shape[:2]
num_channels = min(pred_np.shape[0], len(extended_targets))

# 5️⃣ Draw contours + labels
for i in range(num_channels):
    mask_resized = cv2.resize(pred_np[i], (W, H))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    color = colors[i % len(colors)]

    # Draw thinner contours (1 px)
    cv2.drawContours(img_rgb, contours, -1, color, 1)

    # Label each contour with text
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                img_rgb,
                display_names[i],
                (cx -20 , cy +10 ),  # offset label slightly
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,                 # font scale (smaller text)
                color,
                1,                   # text thickness
                cv2.LINE_AA
            )

# 6️⃣ Display result
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title("Segmentation Contours with Labels")
plt.axis("off")
plt.show()
