#!/usr/bin/env python3
"""
Generate multi-class semantic masks for Endo-4DGS using SAM2 video mode.

Classes:
    0 = background (black borders, out-of-FOV)
    1 = surgical tool/instrument
    2 = tissue (general soft tissue surface)
    3 = vessel/anatomical landmark (optional, disabled by default)

Usage:
    python scripts/generate_semantic_masks.py --dataset cutting_tissues_twice
    python scripts/generate_semantic_masks.py --dataset pulling_soft_tissues
    python scripts/generate_semantic_masks.py --dataset all
"""

import argparse
import os
import sys
import shutil
import tempfile
import glob
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def get_dataset_config(dataset_name):
    """Get paths and naming conventions for each dataset."""
    base = os.path.join(PROJECT_ROOT, "data", "endonerf", dataset_name)
    
    if dataset_name == "cutting_tissues_twice":
        return {
            "images_dir": os.path.join(base, "images"),
            "masks_dir": os.path.join(base, "masks"),
            "semantic_dir": os.path.join(base, "semantic_masks"),
            "image_pattern": "*.png",
            # Images are named 000000.png, masks are frame-000000.mask.png
            "image_to_mask": lambda img_name: f"frame-{os.path.splitext(img_name)[0]}.mask.png",
            "image_sort_key": lambda p: int(os.path.splitext(os.path.basename(p))[0]),
        }
    elif dataset_name == "pulling_soft_tissues":
        return {
            "images_dir": os.path.join(base, "images"),
            "masks_dir": os.path.join(base, "masks"),
            "semantic_dir": os.path.join(base, "semantic_masks"),
            "image_pattern": "*.png",
            # Images are frame-000000.color.png, masks are frame-000000.mask.png
            "image_to_mask": lambda img_name: img_name.replace(".color.png", ".mask.png"),
            "image_sort_key": lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]),
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_jpeg_frames(config, tmp_dir):
    """Convert PNG frames to JPEG in a temp directory for SAM2.
    
    SAM2 video predictor requires JPEG frames named as integers (00000.jpg).
    Returns mapping from JPEG index to original image path.
    """
    images_dir = config["images_dir"]
    image_files = sorted(
        glob.glob(os.path.join(images_dir, config["image_pattern"])),
        key=config["image_sort_key"]
    )
    
    print(f"Found {len(image_files)} frames in {images_dir}")
    
    index_to_original = {}
    for idx, img_path in enumerate(tqdm(image_files, desc="Converting to JPEG")):
        img = cv2.imread(img_path)
        jpeg_path = os.path.join(tmp_dir, f"{idx:05d}.jpg")
        cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index_to_original[idx] = img_path
    
    return index_to_original


def load_existing_masks(config, index_to_original):
    """Load existing binary tool masks, aligning with frame indices."""
    masks_dir = config["masks_dir"]
    masks = {}
    
    for idx, img_path in index_to_original.items():
        img_name = os.path.basename(img_path)
        mask_name = config["image_to_mask"](img_name)
        mask_path = os.path.join(masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Binary: 255 = tool, 0 = background
            masks[idx] = (mask > 127).astype(np.uint8)
        else:
            print(f"  Warning: no mask found for frame {idx} ({mask_name})")
    
    return masks


def detect_background_region(img):
    """Detect black border/background regions in the image.
    
    Background is defined as very dark pixels (near-black borders
    that are outside the endoscope field of view).
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Very dark pixels are background (threshold at 10)
    bg_mask = gray < 10
    
    # Clean up with morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg_mask = cv2.morphologyEx(bg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    
    return bg_mask.astype(bool)


def generate_masks_with_sam2(config, checkpoint_path, model_cfg, device="cuda",
                              num_classes=3, keyframe_interval=10):
    """Generate semantic masks using SAM2 video predictor.
    
    Strategy:
    1. Convert frames to JPEG for SAM2
    2. Initialize SAM2 video predictor
    3. Use existing binary masks as mask prompts for tool (obj_id=1)
    4. Propagate through video
    5. Compose final semantic masks:
       - 0 = background (dark borders)
       - 1 = tool (SAM2 prediction)
       - 2 = tissue (everything else)
    """
    from sam2.build_sam import build_sam2_video_predictor
    
    # Create temporary JPEG directory
    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    print(f"Temp JPEG directory: {tmp_dir}")
    
    try:
        # Step 1: Prepare JPEG frames
        index_to_original = prepare_jpeg_frames(config, tmp_dir)
        num_frames = len(index_to_original)
        
        # Step 2: Load existing binary masks
        existing_masks = load_existing_masks(config, index_to_original)
        print(f"Loaded {len(existing_masks)} existing binary masks")
        
        # Step 3: Initialize SAM2 video predictor
        print(f"\nLoading SAM2 model from {checkpoint_path}...")
        predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
        
        print("Initializing video state...")
        inference_state = predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=True,  # Save GPU memory
            offload_state_to_cpu=False,
        )
        
        # Step 4: Add tool mask prompts on keyframes
        # Use existing masks on every Nth frame as conditioning
        tool_obj_id = 1  # SAM2 object ID for tool
        
        keyframes_used = []
        for idx in range(0, num_frames, keyframe_interval):
            if idx in existing_masks and existing_masks[idx].sum() > 100:
                # Only use frames where tool is actually visible (> 100 pixels)
                mask_tensor = torch.from_numpy(existing_masks[idx]).bool()
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx,
                    obj_id=tool_obj_id,
                    mask=mask_tensor,
                )
                keyframes_used.append(idx)
        
        print(f"Added tool mask prompts on {len(keyframes_used)} keyframes: {keyframes_used[:10]}...")
        
        # Step 5: Propagate through video
        print("\nPropagating through video...")
        sam2_tool_masks = {}
        
        for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
            # video_res_masks shape: (num_objects, 1, H, W) — logits
            # For tool (obj_id=1), get the mask
            for i, obj_id in enumerate(obj_ids):
                if obj_id == tool_obj_id:
                    mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                    sam2_tool_masks[frame_idx] = mask
        
        print(f"Got SAM2 predictions for {len(sam2_tool_masks)} frames")
        
        # Step 6: Compose final semantic masks
        print("\nComposing semantic masks...")
        output_dir = config["semantic_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        for idx in tqdm(range(num_frames), desc="Saving semantic masks"):
            # Load original image for background detection
            img = cv2.imread(index_to_original[idx])
            H, W = img.shape[:2]
            
            # Initialize semantic mask
            semantic = np.zeros((H, W), dtype=np.uint8)
            
            # Layer 1: Detect background (dark borders)
            bg_mask = detect_background_region(img)
            
            # Layer 2: Set tissue as default for non-background
            semantic[~bg_mask] = 2  # tissue
            
            # Layer 3: Apply SAM2 tool prediction (overrides tissue)
            if idx in sam2_tool_masks:
                tool_mask = sam2_tool_masks[idx]
                if tool_mask.shape != (H, W):
                    tool_mask = cv2.resize(tool_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                semantic[tool_mask > 0] = 1  # tool
            elif idx in existing_masks:
                # Fallback to existing mask if SAM2 didn't produce one
                semantic[existing_masks[idx] > 0] = 1
            
            # Background stays 0
            semantic[bg_mask] = 0
            
            # Save with same filename as original image
            orig_name = os.path.basename(index_to_original[idx])
            # Remove any suffix like .color.png -> use base name
            out_name = orig_name
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, semantic)
        
        print(f"\nSaved {num_frames} semantic masks to {output_dir}")
        
        # Print class distribution summary
        print("\n--- Class Distribution Summary ---")
        sample_indices = np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int)
        for idx in sample_indices:
            orig_name = os.path.basename(index_to_original[idx])
            mask_path = os.path.join(output_dir, orig_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            total = mask.size
            for cls in range(num_classes):
                pct = (mask == cls).sum() / total * 100
                print(f"  Frame {idx:4d} | class {cls}: {pct:5.1f}%")
            print()
        
        return output_dir
        
    finally:
        # Cleanup temp directory
        print(f"Cleaning up temp directory: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_masks_without_sam2(config, num_classes=3):
    """Fallback: generate semantic masks from existing binary masks without SAM2.
    
    Simple composition:
    - 0 = background (dark borders)
    - 1 = tool (from existing binary mask)
    - 2 = tissue (everything else)
    """
    images_dir = config["images_dir"]
    output_dir = config["semantic_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(
        glob.glob(os.path.join(images_dir, config["image_pattern"])),
        key=config["image_sort_key"]
    )
    
    print(f"Generating masks for {len(image_files)} frames (no SAM2)...")
    
    for img_path in tqdm(image_files, desc="Generating semantic masks"):
        img_name = os.path.basename(img_path)
        mask_name = config["image_to_mask"](img_name)
        mask_path = os.path.join(config["masks_dir"], mask_name)
        
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        
        semantic = np.zeros((H, W), dtype=np.uint8)
        
        # Background detection
        bg_mask = detect_background_region(img)
        
        # Tissue = everything not background
        semantic[~bg_mask] = 2
        
        # Tool from existing mask
        if os.path.exists(mask_path):
            tool_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            semantic[tool_mask > 127] = 1
        
        # Background
        semantic[bg_mask] = 0
        
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, semantic)
    
    print(f"Saved {len(image_files)} semantic masks to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate semantic masks for Endo-4DGS")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cutting_tissues_twice", "pulling_soft_tissues", "all"],
                        help="Dataset to process")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Number of semantic classes (3 or 4)")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "sam2.1_hiera_large.pt"),
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--model-cfg", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml",
                        help="SAM2 model config name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run SAM2 on")
    parser.add_argument("--no-sam2", action="store_true",
                        help="Skip SAM2, use only existing masks + heuristics")
    parser.add_argument("--keyframe-interval", type=int, default=10,
                        help="Interval for keyframe mask prompts (default: 10)")
    
    args = parser.parse_args()
    
    datasets = ["cutting_tissues_twice", "pulling_soft_tissues"] if args.dataset == "all" else [args.dataset]
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}\n")
        
        config = get_dataset_config(dataset_name)
        
        # Verify directories exist
        if not os.path.isdir(config["images_dir"]):
            print(f"ERROR: Images directory not found: {config['images_dir']}")
            continue
        if not os.path.isdir(config["masks_dir"]):
            print(f"ERROR: Masks directory not found: {config['masks_dir']}")
            continue
        
        if args.no_sam2:
            generate_masks_without_sam2(config, num_classes=args.num_classes)
        else:
            if not os.path.exists(args.checkpoint):
                print(f"ERROR: SAM2 checkpoint not found: {args.checkpoint}")
                print("Use --no-sam2 to skip SAM2, or provide the checkpoint path")
                continue
            
            generate_masks_with_sam2(
                config=config,
                checkpoint_path=args.checkpoint,
                model_cfg=args.model_cfg,
                device=args.device,
                num_classes=args.num_classes,
                keyframe_interval=args.keyframe_interval,
            )
    
    print("\nDone! Run verify_semantic_masks.py to check quality.")


if __name__ == "__main__":
    main()
