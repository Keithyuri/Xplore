"""
Advanced Spinous Process Extractor
Uses multiple image processing techniques to isolate spinous processes
from full spine segmentation in chest X-rays

Usage:
    python advanced_spinous_extractor.py chest_xray.jpg [method]
    
Methods:
    - skeleton: Extract centerline/skeleton (default)
    - posterior: Extract posterior edge
    - thinning: Morphological thinning
    - combined: Use multiple methods
"""

import sys
import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from skimage import io, morphology
from skimage.morphology import skeletonize, medial_axis
from scipy import ndimage
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def method_skeleton(spine_binary):
    """
    Method 1: Extract skeleton/centerline of spine
    This represents the midline where spinous processes align
    """
    print("  → Using skeleton method...")
    
    # Convert to binary
    spine_bool = spine_binary > 0
    
    # Get skeleton
    skeleton = skeletonize(spine_bool)
    
    # Dilate slightly to make it more visible
    kernel = np.ones((3, 3), np.uint8)
    skeleton_dilated = cv2.dilate(skeleton.astype(np.uint8) * 255, kernel, iterations=1)
    
    return skeleton_dilated

def method_medial_axis(spine_binary):
    """
    Method 2: Medial axis transform
    Finds the "spine" of the spine shape
    """
    print("  → Using medial axis method...")
    
    spine_bool = spine_binary > 0
    
    # Compute medial axis
    skel, distance = medial_axis(spine_bool, return_distance=True)
    
    # Use distance transform to weight the skeleton
    # Keep only points that are relatively central
    threshold = distance.max() * 0.3
    central_skel = (skel & (distance > threshold))
    
    # Dilate to make visible
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.dilate(central_skel.astype(np.uint8) * 255, kernel, iterations=1)
    
    return result

def method_posterior_edge(spine_binary, percentage=0.15):
    """
    Method 3: Extract posterior (back) edge of spine
    In frontal chest X-rays, spinous processes appear as posterior structures
    """
    print(f"  → Using posterior edge method (extracting {percentage*100}% of posterior spine)...")
    
    spinous_mask = np.zeros_like(spine_binary)
    
    for y in range(spine_binary.shape[0]):
        row = spine_binary[y, :]
        if row.any():
            x_coords = np.where(row > 0)[0]
            if len(x_coords) > 0:
                # Determine if spine is on left or right side of image
                # Usually spine is more towards the center-right in PA chest X-rays
                center_of_mass = np.mean(x_coords)
                image_center = spine_binary.shape[1] / 2
                
                if center_of_mass > image_center:
                    # Spine is on right side, posterior is rightmost
                    width = len(x_coords)
                    posterior_width = max(2, int(width * percentage))
                    posterior_start = x_coords[-1] - posterior_width
                    spinous_mask[y, posterior_start:x_coords[-1]+1] = 255
                else:
                    # Spine is on left side, posterior is leftmost
                    width = len(x_coords)
                    posterior_width = max(2, int(width * percentage))
                    spinous_mask[y, x_coords[0]:x_coords[0]+posterior_width] = 255
    
    return spinous_mask

def method_morphological_thinning(spine_binary):
    """
    Method 4: Morphological thinning to reduce spine to thin line
    """
    print("  → Using morphological thinning method...")
    
    spine_bool = spine_binary > 0
    
    # Zhang-Suen thinning
    thinned = morphology.thin(spine_bool)
    
    # Dilate slightly
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.dilate(thinned.astype(np.uint8) * 255, kernel, iterations=1)
    
    return result

def method_vertical_centerline(spine_binary):
    """
    Method 5: Extract vertical centerline by finding center of mass at each y-level
    """
    print("  → Using vertical centerline method...")
    
    spinous_mask = np.zeros_like(spine_binary)
    
    # For each horizontal slice, find the center
    for y in range(spine_binary.shape[0]):
        row = spine_binary[y, :]
        if row.any():
            x_coords = np.where(row > 0)[0]
            if len(x_coords) > 0:
                # Find center of mass for this row
                center_x = int(np.mean(x_coords))
                
                # Mark a small region around center (3-5 pixels wide)
                width = 4
                start_x = max(0, center_x - width // 2)
                end_x = min(spine_binary.shape[1], center_x + width // 2 + 1)
                spinous_mask[y, start_x:end_x] = 255
    
    return spinous_mask

def method_combined(spine_binary):
    """
    Method 6: Combine multiple methods with voting
    """
    print("  → Using combined method (voting from multiple techniques)...")
    
    # Get results from multiple methods
    skel = method_skeleton(spine_binary) > 0
    medial = method_medial_axis(spine_binary) > 0
    centerline = method_vertical_centerline(spine_binary) > 0
    
    # Vote: pixel is spinous if at least 2 methods agree
    vote = skel.astype(int) + medial.astype(int) + centerline.astype(int)
    combined = (vote >= 2).astype(np.uint8) * 255
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    return combined

def detect_individual_processes(spinous_mask, min_area=20):
    """
    Detect individual spinous processes from the extracted mask
    """
    print("\n⏳ Detecting individual spinous processes...")
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        spinous_mask.astype(np.uint8), 
        connectivity=8
    )
    
    processes = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            processes.append({
                'id': i,
                'centroid': (int(cx), int(cy)),
                'bbox': (x, y, w, h),
                'area': area,
                'y_position': int(cy)
            })
    
    # Sort by y-coordinate (top to bottom)
    processes.sort(key=lambda p: p['y_position'])
    
    print(f"✓ Found {len(processes)} potential spinous processes")
    
    return processes

def extract_spinous_process(image_path, output_dir="spinous_results", method='skeleton'):
    """
    Main function to extract spinous process from chest X-ray
    """
    print(f"\n{'='*70}")
    print(f"🔍 ADVANCED SPINOUS PROCESS EXTRACTION")
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
    print(f"🔧 Method: {method}")
    
    # Load model and segment spine
    print(f"\n[STEP 1/4] Segmenting full spine...")
    model = xrv.baseline_models.chestx_det.PSPNet()
    spine_idx = model.targets.index('Spine')
    
    img = io.imread(str(image_path))
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    img_normalized = xrv.datasets.normalize(img_gray, maxval=255, reshape=True)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
    
    spine_mask = output[0, spine_idx, :, :].numpy()
    spine_binary = (spine_mask > 0.5).astype(np.uint8) * 255
    
    if spine_binary.shape != img_gray.shape:
        spine_binary = cv2.resize(
            spine_binary,
            (img_gray.shape[1], img_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    spine_pixels = np.sum(spine_binary > 0)
    print(f"✓ Spine segmented ({spine_pixels} pixels)")
    
    if spine_pixels == 0:
        print(f"❌ No spine detected! Run diagnostic_spine_detector.py first.")
        return None
    
    # Extract spinous process using selected method
    print(f"\n[STEP 2/4] Extracting spinous process region...")
    
    methods = {
        'skeleton': method_skeleton,
        'medial': method_medial_axis,
        'posterior': method_posterior_edge,
        'thinning': method_morphological_thinning,
        'centerline': method_vertical_centerline,
        'combined': method_combined
    }
    
    if method not in methods:
        print(f"⚠️  Unknown method '{method}', using 'skeleton'")
        method = 'skeleton'
    
    spinous_mask = methods[method](spine_binary)
    spinous_pixels = np.sum(spinous_mask > 0)
    print(f"✓ Spinous process extracted ({spinous_pixels} pixels)")
    
    # Detect individual processes
    print(f"\n[STEP 3/4] Identifying individual spinous processes...")
    processes = detect_individual_processes(spinous_mask, min_area=20)
    
    # Save and visualize
    print(f"\n[STEP 4/4] Creating visualizations...")
    
    # Convert to color
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Save masks
    spine_path = output_path / f"{image_path.stem}_1_full_spine.png"
    cv2.imwrite(str(spine_path), spine_binary)
    
    spinous_path = output_path / f"{image_path.stem}_2_spinous_process.png"
    cv2.imwrite(str(spinous_path), spinous_mask)
    
    # Visualization 1: Spine (red) + Spinous (yellow)
    viz1 = img_color.copy()
    
    # Red for full spine
    red_overlay = np.zeros_like(viz1)
    red_overlay[:, :, 2] = spine_binary
    viz1 = cv2.addWeighted(viz1, 0.75, red_overlay, 0.25, 0)
    
    # Yellow for spinous process
    yellow_overlay = np.zeros_like(viz1)
    yellow_overlay[:, :, 1] = spinous_mask  # Green
    yellow_overlay[:, :, 2] = spinous_mask  # Red
    viz1 = cv2.addWeighted(viz1, 0.6, yellow_overlay, 0.4, 0)
    
    cv2.putText(viz1, f'RED=Spine | YELLOW=Spinous Process ({method})', 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    viz1_path = output_path / f"{image_path.stem}_3_overlay.png"
    cv2.imwrite(str(viz1_path), viz1)
    
    # Visualization 2: Spinous only with labels
    viz2 = img_color.copy()
    
    # Green overlay for spinous
    green_overlay = np.zeros_like(viz2)
    green_overlay[:, :, 1] = spinous_mask
    viz2 = cv2.addWeighted(viz2, 0.7, green_overlay, 0.3, 0)
    
    # Label individual processes
    for idx, sp in enumerate(processes):
        cx, cy = sp['centroid']
        x, y, w, h = sp['bbox']
        
        # Draw centroid
        cv2.circle(viz2, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(viz2, (cx, cy), 8, (255, 255, 255), 2)
        
        # Draw bounding box
        cv2.rectangle(viz2, (x, y), (x+w, y+h), (255, 165, 0), 2)
        
        # Add label
        label = f'SP{idx+1}'
        cv2.putText(viz2, label, (cx+12, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.putText(viz2, f'SPINOUS PROCESSES: {len(processes)} detected ({method} method)', 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    viz2_path = output_path / f"{image_path.stem}_4_labeled.png"
    cv2.imwrite(str(viz2_path), viz2)
    
    # Visualization 3: Compare methods (if not already combined)
    if method != 'combined':
        print("  → Creating method comparison...")
        
        fig_height = img_color.shape[0]
        fig_width = img_color.shape[1] * 3
        comparison = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        # Try 3 different methods
        test_methods = ['skeleton', 'centerline', 'posterior']
        if method in test_methods:
            test_methods = [method, 'medial', 'combined']
        
        for i, test_method in enumerate(test_methods):
            test_mask = methods[test_method](spine_binary)
            
            # Create overlay
            overlay = img_color.copy()
            green = np.zeros_like(overlay)
            green[:, :, 1] = test_mask
            overlay = cv2.addWeighted(overlay, 0.7, green, 0.3, 0)
            
            # Add label
            cv2.putText(overlay, test_method.upper(), (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Place in comparison
            comparison[:, i*img_color.shape[1]:(i+1)*img_color.shape[1], :] = overlay
        
        comp_path = output_path / f"{image_path.stem}_5_method_comparison.png"
        cv2.imwrite(str(comp_path), comparison)
        print(f"✓ Saved: {comp_path.name}")
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Full spine pixels:       {spine_pixels}")
    print(f"  Spinous process pixels:  {spinous_pixels}")
    print(f"  Reduction ratio:         {(spinous_pixels/spine_pixels*100):.1f}%")
    print(f"  Individual processes:    {len(processes)}")
    
    if processes:
        print(f"\n  Detected Spinous Processes:")
        for idx, sp in enumerate(processes):
            cx, cy = sp['centroid']
            w, h = sp['bbox'][2], sp['bbox'][3]
            print(f"    SP{idx+1}: Position({cx:4d}, {cy:4d}), Size: {w:2d}x{h:3d}px, Area: {sp['area']:4d}px")
    
    print(f"\n📁 Saved files:")
    print(f"  1. {spine_path.name}")
    print(f"  2. {spinous_path.name}")
    print(f"  3. {viz1_path.name}")
    print(f"  4. {viz2_path.name}")
    if method != 'combined':
        print(f"  5. {comp_path.name}")
    print(f"{'='*70}\n")
    
    return {
        'spine_mask': spine_binary,
        'spinous_mask': spinous_mask,
        'processes': processes,
        'method': method
    }

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n❌ Error: No image provided")
        print("\nUsage:")
        print("  python advanced_spinous_extractor.py <image.jpg> [method]")
        print("\nMethods:")
        print("  skeleton   - Extract centerline/skeleton (default, recommended)")
        print("  medial     - Medial axis transform")
        print("  centerline - Vertical centerline")
        print("  posterior  - Posterior edge extraction")
        print("  thinning   - Morphological thinning")
        print("  combined   - Vote from multiple methods")
        print("\nExamples:")
        print("  python advanced_spinous_extractor.py chest_xray.jpg")
        print("  python advanced_spinous_extractor.py chest_xray.jpg skeleton")
        print("  python advanced_spinous_extractor.py chest_xray.jpg combined")
        sys.exit(1)
    
    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'skeleton'
    
    if not Path(image_path).exists():
        print(f"\n❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    try:
        results = extract_spinous_process(image_path, method=method)
        
        if results:
            print("✅ SUCCESS! Check the 'spinous_results' folder.")
            print(f"\n💡 TIP: Try different methods to see which works best:")
            print(f"   python advanced_spinous_extractor.py {image_path} combined")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()