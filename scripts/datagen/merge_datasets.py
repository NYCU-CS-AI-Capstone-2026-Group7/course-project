#!/usr/bin/env python3
"""
LeRobot Dataset Merge Tool

This script merges two LeRobot datasets (local or on Hugging Face Hub) into a single unified dataset.
It preserves all metadata, tasks, joints, actions, observations, and video formats.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def load_dataset(path_or_repo_id):
    if os.path.isdir(path_or_repo_id):
        root_path = os.path.abspath(path_or_repo_id)
        repo_id = os.path.basename(root_path)
        print(f"[INFO] Loading local dataset from: {root_path} (repo_id={repo_id})")
        return LeRobotDataset(repo_id=repo_id, root=root_path)
    else:
        print(f"[INFO] Loading dataset from Hugging Face Hub: {path_or_repo_id}")
        return LeRobotDataset(repo_id=path_or_repo_id)

def main():
    parser = argparse.ArgumentParser(
        description="Merge two LeRobot datasets into a single unified dataset while preserving all metadata, tasks, features, and video formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # 1. Merge two Hugging Face Hub datasets into a new Hub repo ID:
  python scripts/datagen/merge_datasets.py \\
      --src1 XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay \\
      --src2 XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_2 \\
      --target XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_merged

  # 2. Merge local datasets into a local target folder:
  python scripts/datagen/merge_datasets.py \\
      --src1 ./data/dataset_part1 \\
      --src2 ./data/dataset_part2 \\
      --target ./data/dataset_merged

Note:
  - If target is a local directory path, the merged dataset will be written locally.
  - Video formats, shapes, actions, and observations must match between both source datasets.
        """
    )
    parser.add_argument(
        "--src1", 
        type=str, 
        required=True, 
        help="The Hugging Face repo ID or local path of the first source dataset."
    )
    parser.add_argument(
        "--src2", 
        type=str, 
        required=True, 
        help="The Hugging Face repo ID or local path of the second source dataset."
    )
    parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        help="The Hugging Face repo ID or local path for the target merged dataset."
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264_nvenc",
        help="Video codec for encoding. Options: 'h264_nvenc', 'hevc_nvenc', 'libsvtav1', etc. Defaults to 'h264_nvenc' for GPU acceleration."
    )
    
    args = parser.parse_args()
    
    src1 = args.src1
    src2 = args.src2
    target = args.target
    
    # Load Source 1
    try:
        ds1 = load_dataset(src1)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset 1: {e}")
        sys.exit(1)
        
    # Load Source 2
    try:
        ds2 = load_dataset(src2)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset 2: {e}")
        sys.exit(1)
        
    # Prepare Target
    if "/" in target or os.path.isabs(target) or target.startswith("."):
        target_root = os.path.abspath(target)
        target_repo_id = os.path.basename(target_root)
        print(f"[INFO] Merged dataset will be written locally to: {target_root} (repo_id={target_repo_id})")
    else:
        target_repo_id = target
        target_root = None
        print(f"[INFO] Merged dataset will be created as Hugging Face Repo: {target_repo_id}")
        
    # Verify features and configuration match
    if ds1.fps != ds2.fps:
        print(f"[WARNING] FPS mismatch: Dataset 1 has {ds1.fps} FPS, Dataset 2 has {ds2.fps} FPS. Using Dataset 1's FPS.")
        
    # Dynamically check if LeRobotDataset.create accepts vcodec parameter
    import inspect
    create_kwargs = {
        "repo_id": target_repo_id,
        "root": target_root,
        "fps": ds1.fps,
        "features": ds1.features,
        "use_videos": len(ds1.meta.video_keys) > 0,
        "tolerance_s": ds1.tolerance_s,
    }
    
    sig = inspect.signature(LeRobotDataset.create)
    if "vcodec" in sig.parameters:
        create_kwargs["vcodec"] = args.vcodec
        print(f"[INFO] Using video codec parameter: {args.vcodec}")
    else:
        print(f"[WARNING] LeRobotDataset.create does not accept 'vcodec' in this version. Fallback to default CPU encoding.")

    try:
        merged_ds = LeRobotDataset.create(**create_kwargs)
        if "vcodec" not in sig.parameters and hasattr(merged_ds, "vcodec"):
            merged_ds.vcodec = args.vcodec
            print(f"[INFO] Dynamically set merged_ds.vcodec = {args.vcodec}")
    except Exception as e:
        print(f"[ERROR] Failed to create target dataset: {e}")
        sys.exit(1)
        
    def copy_episodes(src_ds, ds_name):
        print(f"[INFO] Copying {src_ds.num_episodes} episodes from {ds_name}...")
        
        # Convert episode_index to numpy for fast index lookup
        ep_indices = np.array(src_ds.hf_dataset["episode_index"])
        
        # Exclude system metadata fields that add_frame generates automatically
        EXCLUDE_KEYS = {"index", "episode_index", "frame_index", "task_index", "timestamp"}
        
        for ep_idx in range(src_ds.num_episodes):
            frame_indices = np.where(ep_indices == ep_idx)[0]
            if len(frame_indices) == 0:
                print(f"  - Warning: Episode {ep_idx} has 0 frames in {ds_name}. Skipping.")
                continue
                
            print(f"  - Appending Episode {ep_idx + 1}/{src_ds.num_episodes} ({len(frame_indices)} frames)...")
            
            for idx in frame_indices:
                frame = src_ds[idx]
                
                # Filter dictionary keys to match features and keep metadata fields
                new_frame = {}
                for k in src_ds.features:
                    if k in frame and k not in EXCLUDE_KEYS:
                        val = frame[k]
                        # Image/video features are read as channel-first (C, H, W)
                        # but add_frame validates channel-last (H, W, C)
                        if src_ds.features[k]["dtype"] in ["image", "video"]:
                            if isinstance(val, torch.Tensor):
                                if val.ndim == 3 and val.shape[0] == 3:
                                    val = val.permute(1, 2, 0)
                            elif isinstance(val, np.ndarray):
                                if val.ndim == 3 and val.shape[0] == 3:
                                    val = val.transpose(1, 2, 0)
                        new_frame[k] = val
                
                if "task" in frame:
                    new_frame["task"] = frame["task"]
                    
                merged_ds.add_frame(new_frame)
            
            # Save the completed episode
            merged_ds.save_episode()
            
    # Append Dataset 1
    copy_episodes(ds1, "Dataset 1")
    
    # Append Dataset 2
    copy_episodes(ds2, "Dataset 2")
    
    print("[INFO] Finalizing and writing metadata for the merged dataset...")
    try:
        merged_ds.finalize()
        print(f"[SUCCESS] Datasets merged successfully into: {target}")
    except Exception as e:
        print(f"[ERROR] Finalization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
