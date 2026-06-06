#!/usr/bin/env python3
"""
LeRobot Dataset Merge Tool

This script merges two LeRobot datasets (local or on Hugging Face Hub) into a single unified dataset.
It preserves all metadata, tasks, joints, actions, observations, and video formats.

Example:
    python scripts/datagen/merge_datasets.py \
        --src1 XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay \
        --src2 XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_2 \
        --target XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_merged
"""

import argparse
import sys
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser(
        description="Merge two LeRobot datasets into a single unified dataset while preserving all metadata, tasks, features, and video formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge local or Hugging Face Hub datasets:
  python scripts/datagen/merge_datasets.py \\
      --src1 XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay \\
      --src2 XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_2 \\
      --target XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_merged

  # Specify custom local paths or HF repo IDs.
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
    
    args = parser.parse_args()
    
    src1 = args.src1
    src2 = args.src2
    target = args.target
    
    # Basic existence check if it looks like a local path
    if "/" not in src1 and not os.path.exists(src1):
        print(f"[ERROR] Source 1 '{src1}' is not a local folder and does not look like a Hugging Face repo ID.")
        sys.exit(1)
        
    print(f"[INFO] Loading Dataset 1: {src1}")
    try:
        ds1 = LeRobotDataset(src1)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset 1: {e}")
        sys.exit(1)
        
    print(f"[INFO] Loading Dataset 2: {src2}")
    try:
        ds2 = LeRobotDataset(src2)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset 2: {e}")
        sys.exit(1)
        
    print(f"[INFO] Creating target merged dataset: {target}")
    try:
        merged_ds = LeRobotDataset.create(
            repo_id=target,
            fps=ds1.fps,
            features=ds1.features,
            use_videos=ds1.use_videos,
            tolerance_s=ds1.tolerance_s,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create target dataset: {e}")
        sys.exit(1)
        
    # Append Dataset 1
    print(f"[INFO] Appending {ds1.num_episodes} episodes from Dataset 1...")
    for ep_idx in range(ds1.num_episodes):
        try:
            frames = ds1.get_episode_frames(ep_idx)
            for frame in frames:
                merged_ds.add_frame(frame)
            merged_ds.save_episode()
            print(f"  - Appended Episode {ep_idx + 1}/{ds1.num_episodes} from Dataset 1")
        except Exception as e:
            print(f"[ERROR] Failed to append episode {ep_idx} from dataset 1: {e}")
            sys.exit(1)
            
    # Append Dataset 2
    print(f"[INFO] Appending {ds2.num_episodes} episodes from Dataset 2...")
    for ep_idx in range(ds2.num_episodes):
        try:
            frames = ds2.get_episode_frames(ep_idx)
            for frame in frames:
                merged_ds.add_frame(frame)
            merged_ds.save_episode()
            print(f"  - Appended Episode {ep_idx + 1}/{ds2.num_episodes} from Dataset 2")
        except Exception as e:
            print(f"[ERROR] Failed to append episode {ep_idx} from dataset 2: {e}")
            sys.exit(1)
            
    print("[INFO] Consolidating and finalizing merged dataset...")
    try:
        merged_ds.consolidate()
        print("[SUCCESS] Datasets merged successfully into:", target)
    except Exception as e:
        print(f"[ERROR] Consolidation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
