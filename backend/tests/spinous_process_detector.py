"""
Spinous Process Detector for Chest X-Rays
Extracts the spinous process region from spine segmentation

Usage:
    python spinous_process_detector.py path/to/your/xray.jpg
"""

import sys
import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from skimage import io
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_spinous_process(spine_mask, method='posterior'):
    """
    Extract spinous process from full spine mask
    
    Args:
        spine_mask: Binary spine segmentation mask
        method: 'posterior' (back edge) or 'centerline' (center points)
    
    Returns:
        Binary mask of spinous process region
    """
    print(f"\n⏳ Extracting spinous process using '{method}' method...")
    
    spinous_mask = np.zeros_like(spine_mask)
    
    if method == 'posterior':
        # Method 1: Extract posterior (back) edge of spine
        # In chest X-rays, spinous processes appear as the posterior-most points
        
        for y in range(spine_mask.shape[0]):
            row = spine_mask[y, :]
            if row.any():
                x_coords = np.where(row > 0)[0]
                if len(x_coords) > 0:
                    # Find the rightmost (posterior) points
                    # Take the posterior 15-25% as spinous process region
                    width = len(x_coords)
                    posterior_width = int(width * 0.20)  # 20% of spine width
                    
                    if posterior_width < 3:
                        posterior_width = min(3, width)  # At least 3 pixels
                    
                    posterior_start = x_coords[-1] - posterior_width
                    spinous_mask[y, posterior_start:x_coords[-1]+1] = 255
    
    elif method == 'centerline':
        # Method 2: Extract centerline/midline of spine
        # Spinous processes align along the center of vertebral column
        
        for y in range(spine_mask.shape[0]):
            row = spine_mask[y, :]
            if row.any():
                x_coords = np.where(row > 0)[0]
                if len(x_coords) > 0:
                    # Find center point
                    center_x = (x_coords[0] + x_coords[-1]) // 2
                    
                    # Mark a small region around center (5-7 pixels wide)
                    width = 5
                    start_x = max(0, center_x - width // 2)
                    end_x = min(spine_mask.shape[1], center_x + width // 2 + 1)
                    spinous_mask[y, start_x:end_x] = 255
    
    # Clean up small noise
    kernel = np.ones((3,3), np.uint8)
    spinous_mask = cv2.morphologyEx(spinous_mask, cv2.MORPH_CLOSE, kernel)
    spinous_mask = cv2.morphologyEx(spinous_mask, cv2.MORPH_OPEN, kernel)
    
    pixels_detected = np.sum(spinous_mask > 0)
    print(f"✓ Spinous process pixels: {pixels_detected}")
    
    return spinous_mask

def detect_individual_spinous_processes(spinous_mask, min_height=10):
    """
    Detect individual spinous processes along the spine
    
    Returns:
        List of individual spinous process regions with coordinates
    """
    print(f"\n⏳ Detecting individual spinous processes...")
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        spinous_mask.astype(np.uint8), 
        connectivity=8
    )
    
    spinous_processes = []
    
    # Skip label 0 (background)
    for i in range(1, num_labels):
        # Get stats for this component
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by minimum height (vertebrae are stacked vertically)
        if h >= min_height:
            cx, cy = centroids[i]
            spinous_processes.append({
                'id': i,
                'centroid': (int(cx), int(cy)),
                'bbox': (x, y, w, h),
                'area': area,
                'label_region': (labels == i).astype(np.uint8) * 255
            })
    
    # Sort by y-coordinate (top to bottom)
    spinous_processes.sort(key=lambda p: p['centroid'][1])
    
    print(f"✓ Found {len(spinous_processes)} individual spinous processes")
    
    return spinous_processes

def detect_spine_and_spinous(image_path, output_dir="spinous_results", method='posterior'):
    """
    Detect spine and extract spinous process from chest X-ray
    
    Args:
        image_path: Path to chest X-ray image
        output_dir: Where to save results
        method: 'posterior' or 'centerline'
    """
    print(f"\n{'='*70}")
    print(f"🔍 SPINOUS PROCESS DETECTION")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    image_path = Path(image_path)
    print(f"📂 Input: {image_path.name}")
    print(f"📁 Output: {output_path}")
    print(f"🔧 Method: {method}")
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: File not found: {image_path}")
        return None
    
    # Load model
    print(f"\n⏳ Loading model...")
    model = xrv.baseline_models.chestx_det.PSPNet()
    spine_idx = model.targets.index('Spine')
    print(f"✓ Model loaded")
    
    # Read and preprocess image
    print(f"\n⏳ Processing image...")
    img = io.imread(str(image_path))
    
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    img_normalized = xrv.datasets.normalize(img_gray, maxval=255, reshape=True)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
    
    # Run segmentation
    print(f"⏳ Running spine segmentation...")
    with torch.no_grad():
        output = model(img_tensor)
    
    # Extract spine mask
    spine_mask = output[0, spine_idx, :, :].numpy()
    spine_binary = (spine_mask > 0.5).astype(np.uint8) * 255
    
    # Resize to original image size
    if spine_binary.shape != img_gray.shape:
        spine_binary = cv2.resize(
            spine_binary,
            (img_gray.shape[1], img_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    print(f"✓ Spine segmentation complete")
    
    # Extract spinous process
    spinous_mask = extract_spinous_process(spine_binary, method=method)
    
    # Detect individual spinous processes
    individual_processes = detect_individual_spinous_processes(spinous_mask)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # 1. Save full spine mask
    spine_path = output_path / f"{image_path.stem}_full_spine.png"
    cv2.imwrite(str(spine_path), spine_binary)
    print(f"✓ Saved: {spine_path.name}")
    
    # 2. Save spinous process mask
    spinous_path = output_path / f"{image_path.stem}_spinous_process.png"
    cv2.imwrite(str(spinous_path), spinous_mask)
    print(f"✓ Saved: {spinous_path.name}")
    
    # 3. Create visualization
    print(f"\n⏳ Creating visualizations...")
    
    # Convert to color
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Visualization 1: Full spine (red) + Spinous process (yellow)
    viz1 = img_color.copy()
    
    # Add red overlay for full spine
    red_overlay = np.zeros_like(viz1)
    red_overlay[:, :, 2] = spine_binary  # Red channel
    viz1 = cv2.addWeighted(viz1, 0.8, red_overlay, 0.2, 0)
    
    # Add yellow overlay for spinous process
    yellow_overlay = np.zeros_like(viz1)
    yellow_overlay[:, :, 1] = spinous_mask  # Green channel
    yellow_overlay[:, :, 2] = spinous_mask  # Red channel (Red + Green = Yellow)
    viz1 = cv2.addWeighted(viz1, 0.7, yellow_overlay, 0.3, 0)
    
    # Add legend
    cv2.putText(viz1, 'SPINE (Red) + SPINOUS PROCESS (Yellow)', 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    viz1_path = output_path / f"{image_path.stem}_combined_viz.png"
    cv2.imwrite(str(viz1_path), viz1)
    print(f"✓ Saved: {viz1_path.name}")
    
    # Visualization 2: Spinous process only with individual markers
    viz2 = img_color.copy()
    
    # Add green overlay for spinous process
    green_overlay = np.zeros_like(viz2)
    green_overlay[:, :, 1] = spinous_mask
    viz2 = cv2.addWeighted(viz2, 0.7, green_overlay, 0.3, 0)
    
    # Mark individual spinous processes
    for idx, sp in enumerate(individual_processes):
        cx, cy = sp['centroid']
        x, y, w, h = sp['bbox']
        
        # Draw centroid
        cv2.circle(viz2, (cx, cy), 5, (0, 0, 255), -1)
        
        # Draw bounding box
        cv2.rectangle(viz2, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add label
        cv2.putText(viz2, f'SP{idx+1}', (cx+10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    cv2.putText(viz2, f'SPINOUS PROCESSES ({len(individual_processes)} detected)', 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    viz2_path = output_path / f"{image_path.stem}_spinous_labeled.png"
    cv2.imwrite(str(viz2_path), viz2)
    print(f"✓ Saved: {viz2_path.name}")
    
    # Print statistics
    print(f"\n📊 SPINOUS PROCESS STATISTICS:")
    print("-" * 70)
    print(f"  Total spinous processes detected: {len(individual_processes)}")
    print(f"  Total spinous pixels: {np.sum(spinous_mask > 0)}")
    
    if individual_processes:
        print(f"\n  Individual Processes:")
        for idx, sp in enumerate(individual_processes):
            cx, cy = sp['centroid']
            w, h = sp['bbox'][2], sp['bbox'][3]
            print(f"    SP{idx+1}: Centroid({cx}, {cy}), Size: {w}x{h}px, Area: {sp['area']}px")
    
    print("-" * 70)
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"\n📁 Results saved to: {output_path}")
    print(f"   • {spine_path.name} (full spine)")
    print(f"   • {spinous_path.name} (spinous process)")
    print(f"   • {viz1_path.name} (combined visualization)")
    print(f"   • {viz2_path.name} (labeled spinous processes)")
    print(f"{'='*70}\n")
    
    return {
        'spine_mask': spine_binary,
        'spinous_mask': spinous_mask,
        'individual_processes': individual_processes,
        'output_dir': str(output_path)
    }

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n❌ Error: No image provided")
        print("\nUsage:")
        print("  python spinous_process_detector.py <path_to_xray.jpg> [method]")
        print("\nMethods:")
        print("  posterior  - Extract posterior edge (default, recommended)")
        print("  centerline - Extract center line")
        print("\nExample:")
        print("  python spinous_process_detector.py chest_xray.jpg")
        print("  python spinous_process_detector.py chest_xray.jpg posterior")
        sys.exit(1)
    
    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'posterior'
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    # Process image
    try:
        results = detect_spine_and_spinous(image_path, method=method)
        print(f"\n✅ Success! Check the 'spinous_results' folder.")
        
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()