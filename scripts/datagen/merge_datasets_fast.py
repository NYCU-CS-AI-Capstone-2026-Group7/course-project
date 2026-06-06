#!/usr/bin/env python3
"""
LeRobot Dataset Fast Merge Tool (Zero-Reencoding)

This script merges two LeRobot datasets (local or cached) into a single dataset.
Instead of decoding and re-encoding videos (which takes hours), it directly copies and renames 
the MP4 video files and adjusts the Parquet database index files using pandas/pyarrow.
This reduces the merging time from hours to seconds.
"""

import os
import sys
import json
import shutil
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def load_dataset_meta(path):
    info_path = os.path.join(path, "meta", "info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing info.json in {path}")
    with open(info_path, "r") as f:
        info = json.load(f)
    return info["total_episodes"], info["total_frames"]

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fast merge two LeRobot datasets in seconds using index shifting without video re-encoding."
    )
    parser.add_argument("--src1", required=True, help="Local path of source dataset 1")
    parser.add_argument("--src2", required=True, help="Local path of source dataset 2")
    parser.add_argument("--target", required=True, help="Local path of target merged dataset")
    args = parser.parse_args()

    src1 = os.path.abspath(args.src1)
    src2 = os.path.abspath(args.src2)
    target = os.path.abspath(args.target)

    if not os.path.exists(src1) or not os.path.exists(src2):
        print("[ERROR] One of the source directories does not exist.")
        sys.exit(1)

    print(f"[INFO] Source 1: {src1}")
    print(f"[INFO] Source 2: {src2}")
    print(f"[INFO] Target: {target}")

    # Prepare directories
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(os.path.join(target, "data"), exist_ok=True)
    os.makedirs(os.path.join(target, "meta", "episodes"), exist_ok=True)
    os.makedirs(os.path.join(target, "videos"), exist_ok=True)

    # Get dataset shapes
    n_ep1, n_fr1 = load_dataset_meta(src1)
    n_ep2, n_fr2 = load_dataset_meta(src2)
    print(f"[INFO] Dataset 1: {n_ep1} episodes, {n_fr1} frames")
    print(f"[INFO] Dataset 2: {n_ep2} episodes, {n_fr2} frames")

    # 1. Pre-compute video offsets for index shifting
    video_offset_map = {}
    video_channels1 = [os.path.basename(p) for p in glob.glob(os.path.join(src1, "videos", "*")) if os.path.isdir(p)]
    video_channels2 = [os.path.basename(p) for p in glob.glob(os.path.join(src2, "videos", "*")) if os.path.isdir(p)]
    
    for chan in video_channels1:
        chan_files = sorted(glob.glob(os.path.join(src1, "videos", chan, "chunk-*", "*.mp4")))
        video_offset_map[chan] = len(chan_files)

    # 2. Process data Parquets
    print("[INFO] Processing data parquets...")
    src1_data_files = sorted(glob.glob(os.path.join(src1, "data", "chunk-*", "*.parquet")))
    src2_data_files = sorted(glob.glob(os.path.join(src2, "data", "chunk-*", "*.parquet")))
    
    os.makedirs(os.path.join(target, "data", "chunk-000"), exist_ok=True)
    for idx, f in enumerate(src1_data_files):
        dest = os.path.join(target, "data", "chunk-000", f"file-{idx:03d}.parquet")
        shutil.copyfile(f, dest)
    d1_count = len(src1_data_files)

    for idx, f in enumerate(src2_data_files):
        dest = os.path.join(target, "data", "chunk-000", f"file-{(idx + d1_count):03d}.parquet")
        df = pd.read_parquet(f)
        df["index"] = df["index"] + n_fr1
        df["episode_index"] = df["episode_index"] + n_ep1
        
        # Shift video file_index in data tables
        for col in df.columns:
            if col.startswith("videos/") and col.endswith("/file_index"):
                chan_name = col.split("/")[1]
                offset = video_offset_map.get(chan_name, 0)
                df[col] = df[col] + offset
                
        df.to_parquet(dest, index=False)

    # 3. Process meta/episodes Parquets
    print("[INFO] Processing episodes metadata parquets...")
    src1_ep_files = sorted(glob.glob(os.path.join(src1, "meta", "episodes", "chunk-*", "*.parquet")))
    src2_ep_files = sorted(glob.glob(os.path.join(src2, "meta", "episodes", "chunk-*", "*.parquet")))
    
    os.makedirs(os.path.join(target, "meta", "episodes", "chunk-000"), exist_ok=True)
    for idx, f in enumerate(src1_ep_files):
        dest = os.path.join(target, "meta", "episodes", "chunk-000", f"file-{idx:03d}.parquet")
        shutil.copyfile(f, dest)
    e1_count = len(src1_ep_files)

    # 4. Copy and shift video files
    for chan in video_channels1:
        os.makedirs(os.path.join(target, "videos", chan, "chunk-000"), exist_ok=True)
        chan_files = sorted(glob.glob(os.path.join(src1, "videos", chan, "chunk-*", "*.mp4")))
        for idx, f in enumerate(chan_files):
            dest = os.path.join(target, "videos", chan, "chunk-000", f"file-{idx:03d}.mp4")
            shutil.copyfile(f, dest)

    for idx, f in enumerate(src2_ep_files):
        dest = os.path.join(target, "meta", "episodes", "chunk-000", f"file-{(idx + e1_count):03d}.parquet")
        df = pd.read_parquet(f)
        df["episode_index"] = df["episode_index"] + n_ep1
        df["dataset_from_index"] = df["dataset_from_index"] + n_fr1
        df["dataset_to_index"] = df["dataset_to_index"] + n_fr1
        if "meta/episodes/file_index" in df.columns:
            df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + e1_count
        if "data/file_index" in df.columns:
            df["data/file_index"] = df["data/file_index"] + d1_count

        for col in df.columns:
            if col.startswith("videos/") and col.endswith("/file_index"):
                chan_name = col.split("/")[1]
                offset = video_offset_map.get(chan_name, 0)
                df[col] = df[col] + offset

        df.to_parquet(dest, index=False)

    for chan in video_channels2:
        os.makedirs(os.path.join(target, "videos", chan, "chunk-000"), exist_ok=True)
        chan_files = sorted(glob.glob(os.path.join(src2, "videos", chan, "chunk-*", "*.mp4")))
        offset = video_offset_map.get(chan, 0)
        for idx, f in enumerate(chan_files):
            dest = os.path.join(target, "videos", chan, "chunk-000", f"file-{(idx + offset):03d}.mp4")
            shutil.copyfile(f, dest)

    # 4. Process tasks.parquet
    print("[INFO] Merging tasks.parquet...")
    tasks1_path = os.path.join(src1, "meta", "tasks.parquet")
    tasks2_path = os.path.join(src2, "meta", "tasks.parquet")
    if os.path.exists(tasks1_path) and os.path.exists(tasks2_path):
        df_t1 = pd.read_parquet(tasks1_path)
        df_t2 = pd.read_parquet(tasks2_path)
        df_t = pd.concat([df_t1, df_t2])
        df_t = df_t[~df_t.index.duplicated(keep="first")]
        df_t["task_index"] = np.arange(len(df_t))
        df_t.to_parquet(os.path.join(target, "meta", "tasks.parquet"))
    elif os.path.exists(tasks1_path):
        shutil.copyfile(tasks1_path, os.path.join(target, "meta", "tasks.parquet"))

    # 5. Process stats.json and info.json
    print("[INFO] Copying stats.json and info.json...")
    shutil.copyfile(os.path.join(src1, "meta", "stats.json"), os.path.join(target, "meta", "stats.json"))
    
    with open(os.path.join(src1, "meta", "info.json"), "r") as f:
        info = json.load(f)
    info["total_episodes"] = n_ep1 + n_ep2
    info["total_frames"] = n_fr1 + n_fr2
    if 'df_t' in locals():
        info["total_tasks"] = len(df_t)
    
    with open(os.path.join(target, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"[SUCCESS] Fast merge complete in target folder: {target}")

if __name__ == "__main__":
    main()
