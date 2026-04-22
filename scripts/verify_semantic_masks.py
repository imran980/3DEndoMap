#!/usr/bin/env python3
"""
Verify quality of generated semantic masks for Endo-4DGS.

Runs three checks:
1. Overlay check: Overlay mask on RGB with transparency (5 random frames)
2. Temporal consistency check: Check for label flicker in 10 consecutive frames
3. Coverage check: Ensure every pixel has a valid class label

Usage:
    python scripts/verify_semantic_masks.py --dataset cutting_tissues_twice
    python scripts/verify_semantic_masks.py --dataset pulling_soft_tissues
"""

import argparse
import os
import sys
import glob
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Color map for overlay visualization (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 0),        # background - black (transparent)
    1: (0, 0, 255),      # tool - red
    2: (0, 200, 0),      # tissue - green
    3: (255, 100, 0),    # vessel - blue
}

CLASS_NAMES = {
    0: "background",
    1: "tool",
    2: "tissue",
    3: "vessel",
}


def get_dataset_paths(dataset_name):
    """Get paths for a dataset."""
    base = os.path.join(PROJECT_ROOT, "data", "endonerf", dataset_name)
    return {
        "images_dir": os.path.join(base, "images"),
        "semantic_dir": os.path.join(base, "semantic_masks"),
        "verify_dir": os.path.join(base, "semantic_masks", "verification"),
    }


def get_paired_files(images_dir, semantic_dir):
    """Get paired (image, mask) file lists sorted by name."""
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    mask_files = sorted(glob.glob(os.path.join(semantic_dir, "*.png")))
    
    # Match by basename
    mask_basenames = {os.path.basename(m) for m in mask_files}
    
    pairs = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        mask_path = os.path.join(semantic_dir, basename)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    
    return pairs


def create_overlay(image, mask, alpha=0.4):
    """Create RGBA overlay of semantic mask on RGB image."""
    overlay = image.copy()
    
    for cls_id, color in CLASS_COLORS.items():
        if cls_id == 0:
            continue  # Skip background
        region = mask == cls_id
        if region.any():
            overlay[region] = (
                (1 - alpha) * image[region] + alpha * np.array(color)
            ).astype(np.uint8)
    
    # Add legend
    y_offset = 30
    for cls_id in sorted(CLASS_COLORS.keys()):
        if cls_id == 0:
            continue
        color = CLASS_COLORS[cls_id]
        name = CLASS_NAMES[cls_id]
        pct = (mask == cls_id).sum() / mask.size * 100
        cv2.rectangle(overlay, (10, y_offset - 15), (30, y_offset), color, -1)
        cv2.putText(overlay, f"{name}: {pct:.1f}%", (35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    return overlay


def check_overlays(pairs, verify_dir, num_samples=5):
    """Check 1: Create overlay visualizations on random frames."""
    print("\n--- Check 1: Overlay Visualization ---")
    
    overlay_dir = os.path.join(verify_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Pick evenly-spaced frames
    n = len(pairs)
    indices = np.linspace(0, n - 1, num_samples, dtype=int)
    
    for idx in indices:
        img_path, mask_path = pairs[idx]
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        overlay = create_overlay(image, mask)
        
        basename = os.path.basename(img_path)
        out_path = os.path.join(overlay_dir, f"overlay_{basename}")
        cv2.imwrite(out_path, overlay)
        
        # Also save side-by-side
        # Color-code the mask for visualization
        mask_vis = np.zeros_like(image)
        for cls_id, color in CLASS_COLORS.items():
            mask_vis[mask == cls_id] = color
        
        side_by_side = np.hstack([image, overlay, mask_vis])
        out_path_sbs = os.path.join(overlay_dir, f"sbs_{basename}")
        cv2.imwrite(out_path_sbs, side_by_side)
        
        print(f"  Frame {idx:4d} ({basename}): saved overlay")
    
    print(f"  Overlays saved to: {overlay_dir}")
    return overlay_dir


def check_temporal_consistency(pairs, verify_dir, start_frame=None, num_frames=10):
    """Check 2: Verify temporal consistency across consecutive frames."""
    print("\n--- Check 2: Temporal Consistency ---")
    
    temporal_dir = os.path.join(verify_dir, "temporal")
    os.makedirs(temporal_dir, exist_ok=True)
    
    if start_frame is None:
        # Pick a start frame in the middle of the sequence
        start_frame = max(0, len(pairs) // 2 - num_frames // 2)
    
    end_frame = min(start_frame + num_frames, len(pairs))
    
    print(f"  Analyzing frames {start_frame} to {end_frame - 1}")
    
    # Load masks for the window
    masks = []
    for idx in range(start_frame, end_frame):
        _, mask_path = pairs[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    
    # Check for flicker: count per-pixel class changes
    total_changes = 0
    total_pixels = masks[0].size
    max_flicker_pct = 0
    
    for i in range(1, len(masks)):
        changes = (masks[i] != masks[i-1]).sum()
        change_pct = changes / total_pixels * 100
        total_changes += changes
        max_flicker_pct = max(max_flicker_pct, change_pct)
        print(f"  Frame {start_frame+i-1} → {start_frame+i}: {changes:6d} pixels changed ({change_pct:.2f}%)")
    
    avg_change_pct = total_changes / ((len(masks) - 1) * total_pixels) * 100
    print(f"\n  Average change per frame: {avg_change_pct:.2f}%")
    print(f"  Max change between frames: {max_flicker_pct:.2f}%")
    
    # Check for ping-pong flicker (A→B→A pattern)
    flicker_count = 0
    if len(masks) >= 3:
        for i in range(1, len(masks) - 1):
            # Pixel is A in frame i-1, B in frame i, back to A in frame i+1
            flicker = (masks[i-1] == masks[i+1]) & (masks[i] != masks[i-1])
            fc = flicker.sum()
            flicker_count += fc
            if fc > 0:
                print(f"  ⚠ Ping-pong flicker at frame {start_frame+i}: {fc} pixels")
    
    if flicker_count == 0:
        print("  ✓ No ping-pong flicker detected")
    else:
        print(f"  ⚠ Total ping-pong flicker pixels: {flicker_count}")
    
    # Save temporal visualization: stack of mask strips
    strip_height = 4  # pixels per frame in the strip
    H, W = masks[0].shape
    temporal_vis = np.zeros((len(masks) * strip_height, W, 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        mask_color = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_id, color in CLASS_COLORS.items():
            mask_color[mask == cls_id] = color
        # Take a horizontal strip from the middle
        strip = mask_color[H//2:H//2+strip_height, :]
        temporal_vis[i*strip_height:(i+1)*strip_height, :] = strip
    
    cv2.imwrite(os.path.join(temporal_dir, "temporal_strips.png"), temporal_vis)
    
    return {
        "avg_change_pct": avg_change_pct,
        "max_change_pct": max_flicker_pct,
        "flicker_count": flicker_count,
    }


def check_coverage(pairs, verify_dir, num_classes=3):
    """Check 3: Verify every pixel has a valid class label."""
    print("\n--- Check 3: Coverage Check ---")
    
    valid_classes = set(range(num_classes))
    total_frames = len(pairs)
    frames_with_issues = 0
    
    class_totals = {cls: 0 for cls in range(num_classes)}
    total_pixels = 0
    
    for idx, (img_path, mask_path) in enumerate(pairs):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_values = set(np.unique(mask))
        
        # Check for invalid class labels
        invalid = unique_values - valid_classes
        if invalid:
            print(f"  ⚠ Frame {idx}: invalid class values: {invalid}")
            frames_with_issues += 1
        
        # Accumulate class distribution
        for cls in range(num_classes):
            class_totals[cls] += (mask == cls).sum()
        total_pixels += mask.size
    
    # Report
    if frames_with_issues == 0:
        print(f"  ✓ All {total_frames} frames have valid class labels only")
    else:
        print(f"  ⚠ {frames_with_issues}/{total_frames} frames have invalid labels")
    
    print(f"\n  Overall class distribution ({total_frames} frames):")
    for cls in range(num_classes):
        pct = class_totals[cls] / total_pixels * 100
        print(f"    Class {cls} ({CLASS_NAMES.get(cls, '?')}): {pct:.1f}%")
    
    return frames_with_issues == 0


def main():
    parser = argparse.ArgumentParser(description="Verify semantic masks for Endo-4DGS")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cutting_tissues_twice", "pulling_soft_tissues", "all"])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--overlay-samples", type=int, default=5)
    parser.add_argument("--temporal-start", type=int, default=None)
    parser.add_argument("--temporal-frames", type=int, default=10)
    
    args = parser.parse_args()
    
    datasets = ["cutting_tissues_twice", "pulling_soft_tissues"] if args.dataset == "all" else [args.dataset]
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Verifying dataset: {dataset_name}")
        print(f"{'='*60}")
        
        paths = get_dataset_paths(dataset_name)
        
        if not os.path.isdir(paths["semantic_dir"]):
            print(f"ERROR: Semantic masks not found: {paths['semantic_dir']}")
            print("Run generate_semantic_masks.py first.")
            continue
        
        pairs = get_paired_files(paths["images_dir"], paths["semantic_dir"])
        print(f"Found {len(pairs)} paired (image, mask) files")
        
        if len(pairs) == 0:
            print("ERROR: No paired files found. Check filenames match.")
            continue
        
        os.makedirs(paths["verify_dir"], exist_ok=True)
        
        # Run checks
        overlay_dir = check_overlays(pairs, paths["verify_dir"], args.overlay_samples)
        temporal_results = check_temporal_consistency(
            pairs, paths["verify_dir"],
            start_frame=args.temporal_start,
            num_frames=args.temporal_frames,
        )
        coverage_ok = check_coverage(pairs, paths["verify_dir"], args.num_classes)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY for {dataset_name}")
        print(f"{'='*60}")
        print(f"  Overlay images:           {overlay_dir}")
        print(f"  Temporal avg change:      {temporal_results['avg_change_pct']:.2f}%")
        print(f"  Temporal ping-pong:       {temporal_results['flicker_count']} pixels")
        print(f"  Coverage valid:           {'✓ PASS' if coverage_ok else '✗ FAIL'}")


if __name__ == "__main__":
    main()
