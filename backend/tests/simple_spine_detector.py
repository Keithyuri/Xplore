"""
Spine Detector with GUI File Picker
Just run: python spine_detector_gui.py
A window will pop up to select your X-ray image!
"""

import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from skimage import io
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
warnings.filterwarnings('ignore')

def select_image_file():
    """Open file dialog to select image"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    root.attributes('-topmost', True)  # Bring to front
    
    print("\n🖱️  Please select your chest X-ray image...")
    
    file_path = filedialog.askopenfilename(
        title="Select Chest X-Ray Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path

def detect_spine_from_xray(image_path, output_dir=None):
    """
    Detect and segment spine from chest X-ray
    """
    print("\n" + "="*70)
    print("🔍 CHEST X-RAY SPINE DETECTION")
    print("="*70 + "\n")
    
    # Convert to Path
    image_path = Path(image_path)
    
    # Set output directory next to input image if not specified
    if output_dir is None:
        output_dir = image_path.parent / "spine_detection_results"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Input image: {image_path.name}")
    print(f"✓ Output folder: {output_dir}")
    
    # Load model
    print(f"\n⏳ Loading AI model (this may take a minute on first run)...")
    try:
        model = xrv.baseline_models.chestx_det.PSPNet()
        spine_idx = model.targets.index('Spine')
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        return None
    
    # Read image
    print(f"\n⏳ Reading image...")
    try:
        img = io.imread(str(image_path))
        print(f"✓ Image loaded: {img.shape}")
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        messagebox.showerror("Error", f"Failed to read image:\n{e}")
        return None
    
    # Convert to grayscale
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    # Normalize and convert to tensor
    print(f"\n⏳ Preprocessing...")
    img_normalized = xrv.datasets.normalize(img_gray, maxval=255, reshape=True)
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)
    
    # Run segmentation
    print(f"⏳ Running spine segmentation (AI processing)...")
    with torch.no_grad():
        output = model(img_tensor)
    
    # Extract spine mask
    spine_mask = output[0, spine_idx, :, :].numpy()
    spine_binary = (spine_mask > 0.5).astype(np.uint8) * 255
    
    print(f"✓ Segmentation complete")
    print(f"✓ Spine pixels detected: {np.sum(spine_binary > 0)}")
    
    # Resize to match original
    if spine_binary.shape != img_gray.shape:
        spine_binary = cv2.resize(
            spine_binary,
            (img_gray.shape[1], img_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Check if spine was detected
    if np.sum(spine_binary) == 0:
        print("⚠️  WARNING: No spine detected in image")
        messagebox.showwarning("Warning", "No spine was detected in the image.\n\nThis could mean:\n• Image is not a chest X-ray\n• Image quality is too low\n• Spine is not visible in the image")
    
    # Save mask
    print(f"\n💾 Saving results...")
    mask_filename = f"{image_path.stem}_spine_mask.png"
    mask_path = output_dir / mask_filename
    cv2.imwrite(str(mask_path), spine_binary)
    print(f"✓ Saved: {mask_filename}")
    
    # Create visualization
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Create red overlay
    overlay = img_color.copy()
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], spine_binary)
    
    # Blend
    result = cv2.addWeighted(img_color, 0.7, overlay, 0.3, 0)
    
    # Add labels
    cv2.putText(result, 'SPINE SEGMENTATION', (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(result, f'File: {image_path.name}', (20, result.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    viz_filename = f"{image_path.stem}_spine_visualization.png"
    viz_path = output_dir / viz_filename
    cv2.imwrite(str(viz_path), result)
    print(f"✓ Saved: {viz_filename}")
    
    # Calculate statistics
    print(f"\n📊 SPINE ANALYSIS:")
    print("-" * 70)
    
    contours, _ = cv2.findContours(
        spine_binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    stats_text = []
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(main_contour)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(main_contour)
            x, y, w, h = cv2.boundingRect(main_contour)
            coverage = (area/(img.shape[0]*img.shape[1])*100)
            
            print(f"  Centroid (x, y):        ({cx}, {cy})")
            print(f"  Area:                   {area:.0f} pixels")
            print(f"  Bounding Box (w x h):   {w} x {h} pixels")
            print(f"  Coverage:               {coverage:.2f}% of image")
            
            stats_text = [
                f"Centroid: ({cx}, {cy})",
                f"Area: {area:.0f} pixels",
                f"Size: {w}x{h} pixels",
                f"Coverage: {coverage:.1f}%"
            ]
        else:
            print("  ⚠️  Could not calculate statistics")
    else:
        print("  ⚠️  No spine contours found")
    
    print("-" * 70)
    print(f"\n✅ PROCESSING COMPLETE!")
    print(f"\n📁 Results saved to:\n   {output_dir}")
    print("="*70 + "\n")
    
    # Show completion message
    message = f"✅ Spine detection complete!\n\n"
    message += f"📁 Results saved to:\n{output_dir}\n\n"
    message += f"Files created:\n"
    message += f"  • {mask_filename}\n"
    message += f"  • {viz_filename}\n"
    if stats_text:
        message += f"\n📊 Quick Stats:\n"
        for stat in stats_text:
            message += f"  • {stat}\n"
    
    messagebox.showinfo("Success", message)
    
    return {
        'mask_path': str(mask_path),
        'viz_path': str(viz_path),
        'spine_mask': spine_binary,
        'success': True
    }

def main():
    """Main function with GUI"""
    print("\n" + "="*70)
    print("🏥 CHEST X-RAY SPINE DETECTION SYSTEM")
    print("="*70)
    
    # Select image
    image_path = select_image_file()
    
    if not image_path:
        print("\n❌ No file selected. Exiting...")
        messagebox.showinfo("Cancelled", "No file selected.")
        return
    
    print(f"\n✓ Selected: {Path(image_path).name}")
    
    # Process
    try:
        results = detect_spine_from_xray(image_path)
        
        if results and results.get('success'):
            print("\n🎉 SUCCESS! Check your results folder.")
            
            # Ask if user wants to open results folder
            if messagebox.askyesno("Open Results?", "Would you like to open the results folder?"):
                import os
                import platform
                results_dir = Path(results['viz_path']).parent
                
                if platform.system() == 'Windows':
                    os.startfile(results_dir)
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{results_dir}"')
                else:  # Linux
                    os.system(f'xdg-open "{results_dir}"')
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        messagebox.showerror("Error", f"An error occurred:\n\n{e}\n\nCheck console for details.")
        
        import traceback
        print("\n📋 Full error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()