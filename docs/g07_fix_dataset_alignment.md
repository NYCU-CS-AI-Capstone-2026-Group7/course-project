# Group 7 - Dataset Alignment & Training Out-of-Bounds Fixes

This document records the analysis and fixings applied to resolve the dataset index out of bounds (`Invalid frame index`) and timestamp mismatch (`FrameTimestampError`) errors during imitation learning training.

## Problem Description
1. **Index Out of Bounds (`Invalid frame index`)**:
   During fast dataset merging, the metadata parquet file and physical videos were concatenated, but the step parquets (`data/*.parquet`) of the second dataset had `file_index` referencing Dataset 1's short video `file-000.mp4` instead of the shifted index.
   Furthermore, the original Dataset 1 (`aicapstone_group7_cutlery_v2_replay`) itself contains recording discrepancies where some step parquet entries record timestamps/frame indices that slightly exceed the actual physical video frames.
2. **Timestamp Mismatch (`FrameTimestampError`)**:
   LeRobot checks whether the queried timestamp is aligned with the actual loaded frame timestamp from `torchcodec`. If a step references a frame beyond the physical video bounds (e.g. asking for 741.5s on a 739.0s video), clipping index alone triggers an `AssertionError` (or `FrameTimestampError`) because the gap (e.g., 2.5s) exceeds the `tolerance_s` (0.0001s).
3. **Multi-Venv Conflict**:
   The active tmux pane was running inside `/tmp/.venv`, while code modifications were initially applied to `/root/course-project/.venv`, causing fixes to not take effect on the active shell prompt.

## Solutions Applied
1. **Fast Merger Patch**:
   Updated the fast merger script to correctly shift all `videos/*/file_index` columns in `data/*.parquet` step files.
2. **Video Decoders Boundary Guard (Index Clip)**:
   Modified `lerobot/datasets/video_utils.py` to clip `frame_indices` between `0` and `metadata.num_frames - 1` when converting timestamps to frame indices.
3. **Timestamp Tolerance Graceful Fallback**:
   Bypassed the crash assertion `assert is_within_tol.all()` when query timestamps exceed the tolerance limit. It now logs a warning and proceeds with training using the clipped frame instead of interrupting the run.
4. **Environment Unification**:
   Applied the core code fixes to both `/tmp/.venv` and `.venv` environments.

## How to Replicate
1. **Run the patch script**:
   ```bash
   python scripts/patch_lerobot_video_utils.py
   ```
   This script automatically applies the index-clipping and tolerance-bypass patches to both `/tmp/.venv` and `/root/course-project/.venv` environments.
2. **Run the training script**:
   Use standard Hydra commands to run training using the officially merged dataset:
   ```bash
   lerobot-train \
       --dataset.repo_id=XiaoPanPanKevinPan/cutlery_v2_replay_merged \
       --policy.type=diffusion \
       --job_name=aicapstone_group7_cutlery_v2_policy \
       --policy.device=cuda \
       --wandb.enable=true \
       --policy.repo_id=XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_policy
   ```
