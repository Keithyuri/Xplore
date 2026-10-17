"""
Diagnostic Spine Detector - Shows what's happening at each step
This will help us figure out why nothing is being detected

Usage:
    python diagnostic_spine_detector.py path/to/your/xray.jpg
"""

import sys
import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def save_debug_images(img_gray, spine_mask, spine_binary, output_dir, filename):
    """Save intermediate results for debugging"""
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    # Raw segmentation output (before thresholding)
    axes[0, 1].imshow(spine_mask, cmap='hot')
    axes[0, 1].set_title(f'2. Raw Spine Prediction\n(max={spine_mask.max():.3f}, min={spine_mask.min():.3f})')
    axes[0, 1].axis('off')
    
    # Histogram of prediction values
    axes[0, 2].hist(spine_mask.flatten(), bins=50, color='blue', alpha=0.7)
    axes[0, 2].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
    axes[0, 2].set_title('3. Prediction Value Distribution')
    axes[0, 2].set_xlabel('Prediction Value')
    axes[0, 2].set_ylabel('Pixel Count')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Binary mask (after thresholding)
    axes[1, 0].imshow(spine_binary, cmap='gray')
    axes[1, 0].set_title(f'4. Binary Mask (threshold>0.5)\nPixels detected: {np.sum(spine_binary > 0)}')
    axes[1, 0].axis('off')
    
    # Overlay on original
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 0] = spine_binary  # Red channel
    blended = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('5. Overlay on Original')
    axes[1, 1].axis('off')
    
    # Statistics
    stats_text = f"""
    IMAGE INFO:
    • Size: {img_gray.shape}
    • Min/Max: {img_gray.min()}/{img_gray.max()}
    • Mean: {img_gray.mean():.1f}
    
    PREDICTION INFO:
    • Min value: {spine_mask.min():.4f}
    • Max value: {spine_mask.max():.4f}
    • Mean value: {spine_mask.mean():.4f}
    • Pixels > 0.5: {np.sum(spine_mask > 0.5)}
    • Pixels > 0.3: {np.sum(spine_mask > 0.3)}
    • Pixels > 0.1: {np.sum(spine_mask > 0.1)}
    
    DETECTION RESULT:
    • Spine pixels: {np.sum(spine_binary > 0)}
    • Coverage: {(np.sum(spine_binary > 0) / spine_binary.size * 100):.2f}%
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('6. Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    debug_path = output_dir / f"{filename}_DEBUG.png"
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved debug visualization: {debug_path.name}")
    
    return debug_path

def try_different_thresholds(spine_mask, img_gray, output_dir, filename):
    """Try different threshold values to see what works"""
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, thresh in enumerate(thresholds):
        binary = (spine_mask > thresh).astype(np.uint8) * 255
        pixel_count = np.sum(binary > 0)
        coverage = pixel_count / binary.size * 100
        
        # Create overlay
        overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = binary
        blended = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[idx].imshow(blended)
        axes[idx].set_title(f'Threshold={thresh}\nPixels: {pixel_count} ({coverage:.2f}%)', fontsize=10)
        axes[idx].axis('off')
    
    plt.suptitle('Different Threshold Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    thresh_path = output_dir / f"{filename}_THRESHOLDS.png"
    plt.savefig(thresh_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved threshold comparison: {thresh_path.name}")
    
    return thresh_path

def diagnose_detection(image_path, output_dir="diagnostic_results"):
    """
    Run full diagnostic on spine detection
    """
    print(f"\n{'='*70}")
    print(f"🔬 DIAGNOSTIC SPINE DETECTION")
    print(f"{'='*70}\n")
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ ERROR: File not found: {image_path}")
        return None
    
    print(f"📂 Input: {image_path.name}")
    print(f"📁 Output: {output_path}")
    
    # Load model
    print(f"\n[STEP 1] Loading TorchXRayVision model...")
    try:
        model = xrv.baseline_models.chestx_det.PSPNet()
        spine_idx = model.targets.index('Spine')
        print(f"✓ Model loaded successfully")
        print(f"✓ Available targets: {model.targets}")
        print(f"✓ Spine is at index: {spine_idx}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Read image
    print(f"\n[STEP 2] Reading and analyzing image...")
    try:
        img = io.imread(str(image_path))
        print(f"✓ Image loaded")
        print(f"  • Original shape: {img.shape}")
        print(f"  • Original dtype: {img.dtype}")
        
        # Convert to grayscale
        if len(img.shape) > 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            print(f"  • Converted to grayscale")
        else:
            img_gray = img.copy()
            print(f"  • Already grayscale")
        
        print(f"  • Grayscale shape: {img_gray.shape}")
        print(f"  • Value range: [{img_gray.min()}, {img_gray.max()}]")
        print(f"  • Mean value: {img_gray.mean():.2f}")
        
        # Save original
        orig_path = output_path / f"{image_path.stem}_original.png"
        cv2.imwrite(str(orig_path), img_gray)
        print(f"✓ Saved original: {orig_path.name}")
        
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None
    
    # Preprocess
    print(f"\n[STEP 3] Preprocessing for model...")
    try:
        img_normalized = xrv.datasets.normalize(img_gray, maxval=255, reshape=True)
        print(f"✓ Normalized image")
        print(f"  • Normalized shape: {img_normalized.shape}")
        print(f"  • Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
        print(f"✓ Converted to tensor")
        print(f"  • Tensor shape: {img_tensor.shape}")
        
    except Exception as e:
        print(f"❌ Error preprocessing: {e}")
        return None
    
    # Run model
    print(f"\n[STEP 4] Running segmentation model...")
    try:
        with torch.no_grad():
            output = model(img_tensor)
        
        print(f"✓ Model inference complete")
        print(f"  • Output shape: {output.shape}")
        print(f"  • Output contains {output.shape[1]} segmentation channels")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return None
    
    # Extract spine
    print(f"\n[STEP 5] Extracting spine segmentation...")
    spine_mask = output[0, spine_idx, :, :].numpy()
    
    print(f"✓ Extracted spine channel")
    print(f"  • Spine mask shape: {spine_mask.shape}")
    print(f"  • Value range: [{spine_mask.min():.4f}, {spine_mask.max():.4f}]")
    print(f"  • Mean value: {spine_mask.mean():.4f}")
    print(f"  • Median value: {np.median(spine_mask):.4f}")
    
    # Analyze prediction distribution
    print(f"\n[STEP 6] Analyzing prediction confidence...")
    print(f"  • Pixels with confidence > 0.1: {np.sum(spine_mask > 0.1)} ({np.sum(spine_mask > 0.1) / spine_mask.size * 100:.2f}%)")
    print(f"  • Pixels with confidence > 0.3: {np.sum(spine_mask > 0.3)} ({np.sum(spine_mask > 0.3) / spine_mask.size * 100:.2f}%)")
    print(f"  • Pixels with confidence > 0.5: {np.sum(spine_mask > 0.5)} ({np.sum(spine_mask > 0.5) / spine_mask.size * 100:.2f}%)")
    print(f"  • Pixels with confidence > 0.7: {np.sum(spine_mask > 0.7)} ({np.sum(spine_mask > 0.7) / spine_mask.size * 100:.2f}%)")
    
    # Create binary mask
    print(f"\n[STEP 7] Creating binary mask (threshold=0.5)...")
    spine_binary = (spine_mask > 0.5).astype(np.uint8) * 255
    
    # Resize to match original
    if spine_binary.shape != img_gray.shape:
        print(f"  • Resizing from {spine_binary.shape} to {img_gray.shape}...")
        spine_binary = cv2.resize(
            spine_binary,
            (img_gray.shape[1], img_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        spine_mask = cv2.resize(
            spine_mask,
            (img_gray.shape[1], img_gray.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    detected_pixels = np.sum(spine_binary > 0)
    print(f"✓ Binary mask created")
    print(f"  • Detected spine pixels: {detected_pixels}")
    print(f"  • Coverage: {detected_pixels / spine_binary.size * 100:.3f}%")
    
    # Save results
    print(f"\n[STEP 8] Saving diagnostic results...")
    
    # Save debug visualization
    debug_path = save_debug_images(img_gray, spine_mask, spine_binary, output_path, image_path.stem)
    
    # Save threshold comparison
    thresh_path = try_different_thresholds(spine_mask, img_gray, output_path, image_path.stem)
    
    # Analysis and recommendations
    print(f"\n{'='*70}")
    print(f"📊 DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    
    if detected_pixels == 0:
        print(f"\n❌ NO SPINE DETECTED")
        print(f"\nPossible reasons:")
        print(f"  1. Image is not a chest X-ray (model trained on chest X-rays)")
        print(f"  2. Image orientation is unusual")
        print(f"  3. Image quality is too low")
        print(f"  4. Spine is not visible in the image")
        print(f"\n💡 Recommendations:")
        print(f"  • Check the threshold comparison image: {thresh_path.name}")
        print(f"  • Try lowering threshold (if you see any red in threshold images)")
        print(f"  • Verify this is a PA/AP chest X-ray")
        print(f"  • Check image orientation (spine should be roughly vertical)")
    elif detected_pixels < 1000:
        print(f"\n⚠️  VERY LOW DETECTION ({detected_pixels} pixels)")
        print(f"\n💡 Recommendations:")
        print(f"  • Detection is weak - try lower threshold")
        print(f"  • Check threshold comparison to find better value")
    else:
        print(f"\n✅ SPINE DETECTED ({detected_pixels} pixels)")
        print(f"\nYou can now use the regular detector with this image!")
    
    print(f"\n📁 Check these diagnostic files:")
    print(f"  • {debug_path.name}")
    print(f"  • {thresh_path.name}")
    print(f"{'='*70}\n")
    
    return {
        'spine_mask': spine_mask,
        'spine_binary': spine_binary,
        'detected_pixels': detected_pixels,
        'max_confidence': float(spine_mask.max()),
        'output_dir': str(output_path)
    }

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n❌ Error: No image provided")
        print("\nUsage:")
        print("  python diagnostic_spine_detector.py <path_to_xray.jpg>")
        print("\nExample:")
        print("  python diagnostic_spine_detector.py chest_xray.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"\n❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    try:
        results = diagnose_detection(image_path)
        
        if results:
            print("\n✅ Diagnostic complete! Check the 'diagnostic_results' folder.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()