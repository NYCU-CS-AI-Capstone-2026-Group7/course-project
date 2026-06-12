#!/usr/bin/env python3
"""
HuggingFace Dataset Auto-Upload Daemon (Batch Commits Edition)
Provides ultra-efficient, real-time, rate-limit-aware dataset backup.
Utilizes `api.create_commit` with `CommitOperationAdd` to batch multiple file 
uploads into a single Git Commit, completely bypassing HuggingFace 429 rate limits
and avoiding branch ref-update lock conflicts during multi-instance execution.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("HF_Batch_Uploader")

# Configurable paths and targets
BASE_DIR = os.path.expanduser("~/.cache/huggingface/lerobot/XiaoPanPanKevinPan")
REPO_PREFIX = "XiaoPanPanKevinPan/g7_cutlery_v3_replay"
TARGETS = ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7"]
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_state.json")

# Metadata files that change dynamically
METADATA_NAMES = {"episodes.parquet", "tasks.parquet", "info.json", "stats.json"}

# Rate-Limit Aware Advanced Settings
BATCH_COMMIT_COOLDOWN = 180    # Wait at least 3 minutes (180s) between commits per repository
MAX_QUEUE_EPISODES = 3         # Max accumulated episodes before forcing a batch commit
WRITE_SILENCE_SECONDS = 15     # File must remain unchanged for 15s to be considered finalized
POLLING_INTERVAL = 30          # Directory polling interval in seconds

api = HfApi()

def load_state():
    """Loads the uploading state from JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state file: {e}. Starting fresh.")
    return {}

def save_state(state):
    """Saves the current uploading state to JSON file."""
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state file: {e}")

def ensure_repo_exists(repo_id):
    """Verifies repository existence or creates a private one if missing."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        logger.info(f"Repository '{repo_id}' not found. Creating a private dataset repository...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
            logger.info(f"Successfully created private dataset: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository '{repo_id}': {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to verify repository '{repo_id}': {e}")
        return False
    return True

def scan_directory(dir_path):
    """Recursively scans directory tree. Fast metadata-only scan with 0% read I/O overhead."""
    files_info = {}
    path_obj = Path(dir_path)
    if not path_obj.exists() or not path_obj.is_dir():
        return files_info
    
    for p in path_obj.rglob("*"):
        if p.is_file():
            # Exclude active temporary files, locking/lockfiles, and hidden files
            if any(part.startswith("tmp") or part.startswith(".") or part.endswith(".lock") for part in p.parts):
                continue
            try:
                stat = p.stat()
                rel_path = str(p.relative_to(path_obj))
                files_info[rel_path] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "abs_path": str(p)
                }
            except FileNotFoundError:
                # Handle file deleted mid-scan
                continue
    return files_info

def main():
    logger.info("==========================================================")
    logger.info("HuggingFace Dataset Auto-Upload Daemon (Batch Commits Edition)")
    logger.info("Designed to bypass 429 Rate-Limits & Branch Ref Lock Conflicts")
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Repo Prefix: {REPO_PREFIX}")
    logger.info(f"Monitoring Suffixes: {TARGETS}")
    logger.info(f"State Database Path: {STATE_FILE}")
    logger.info("==========================================================")
    
    state = load_state()
    
    # Initialize state dictionary structures and runtime stats
    last_commit_time = {} # Tracks wall-clock timestamp of last successful commit per suffix
    
    for suffix in TARGETS:
        if suffix not in state:
            state[suffix] = {
                "uploaded_files": {},      # Immutable files (videos/parquet chunk) -> {"size": size, "mtime": mtime}
                "uploaded_metadata": {}    # Metadata files -> {"size": size, "mtime": mtime}
            }
        last_commit_time[suffix] = 0
            
    # Verify/Create HuggingFace repos before launching loop
    logger.info("Verifying remote HuggingFace repositories...")
    active_targets = []
    for suffix in TARGETS:
        repo_name = f"{REPO_PREFIX}{suffix}"
        if ensure_repo_exists(repo_name):
            active_targets.append(suffix)
        else:
            logger.warning(f"Skipping target '{suffix}' due to validation error.")
            
    if not active_targets:
        logger.error("No active targets could be verified. Exiting daemon.")
        sys.exit(1)

    logger.info(f"Verification successful. Actively monitoring: {active_targets}")
    logger.info(f"Polling directory changes every {POLLING_INTERVAL} seconds...")

    try:
        while True:
            state_changed = False
            now = time.time()
            
            for suffix in active_targets:
                dir_name = f"aicapstone_group7_cutlery_v2_replay{suffix}"
                dir_path = os.path.join(BASE_DIR, dir_name)
                repo_name = f"{REPO_PREFIX}{suffix}"
                
                if not os.path.exists(dir_path):
                    continue
                
                # Scan local filesystem
                current_files = scan_directory(dir_path)
                repo_state = state[suffix]
                
                # Accumulators for this repository's commit operations
                operations = []
                pending_immutable_count = 0
                temp_upload_records = [] # Keeps records to apply if the batch commit succeeds
                
                for rel_path, file_meta in current_files.items():
                    file_name = os.path.basename(rel_path)
                    abs_path = file_meta["abs_path"]
                    mtime = file_meta["mtime"]
                    size = file_meta["size"]
                    
                    # 1. Stability Verification: Avoid uploading files currently being written
                    if (now - mtime) < WRITE_SILENCE_SECONDS:
                        continue
                        
                    is_metadata = file_name in METADATA_NAMES or "meta" in Path(rel_path).parts
                    
                    if is_metadata:
                        # Check if metadata file is new or modified
                        uploaded_meta = repo_state["uploaded_metadata"].get(rel_path)
                        if not uploaded_meta or uploaded_meta["mtime"] != mtime or uploaded_meta["size"] != size:
                            operations.append(CommitOperationAdd(
                                path_in_repo=rel_path,
                                path_or_fileobj=abs_path
                            ))
                            temp_upload_records.append({
                                "type": "metadata",
                                "rel_path": rel_path,
                                "size": size,
                                "mtime": mtime
                            })
                    else:
                        # Check if immutable episode data is new or modified
                        uploaded_meta = repo_state["uploaded_files"].get(rel_path)
                        if not uploaded_meta or uploaded_meta["size"] != size:
                            operations.append(CommitOperationAdd(
                                path_in_repo=rel_path,
                                path_or_fileobj=abs_path
                            ))
                            pending_immutable_count += 1
                            temp_upload_records.append({
                                "type": "immutable",
                                "rel_path": rel_path,
                                "size": size,
                                "mtime": mtime
                            })

                # Decide whether to execute the Batch Commit
                # Trigger conditions:
                # A. There are pending operations, and we have reached MAX_QUEUE_EPISODES to avoid long delay
                # B. There are pending operations, and BATCH_COMMIT_COOLDOWN has elapsed since last commit
                if operations:
                    elapsed_cooldown = now - last_commit_time[suffix]
                    should_commit = (
                        (pending_immutable_count >= MAX_QUEUE_EPISODES * 2) or # Each episode has 1 mp4 + 1 parquet
                        (elapsed_cooldown >= BATCH_COMMIT_COOLDOWN)
                    )
                    
                    if should_commit:
                        logger.info(f"[{suffix}] Triggering batch commit with {len(operations)} operations ({pending_immutable_count // 2} episodes + metadata)...")
                        try:
                            # Execute atomic batch commit
                            api.create_commit(
                                repo_id=repo_name,
                                operations=operations,
                                commit_message=f"Batch upload {pending_immutable_count // 2} episodes and update metadata",
                                repo_type="dataset"
                            )
                            
                            # Update local state ONLY on successful commit
                            for record in temp_upload_records:
                                if record["type"] == "metadata":
                                    repo_state["uploaded_metadata"][record["rel_path"]] = {
                                        "size": record["size"],
                                        "mtime": record["mtime"]
                                    }
                                else:
                                    repo_state["uploaded_files"][record["rel_path"]] = {
                                        "size": record["size"],
                                        "mtime": record["mtime"]
                                    }
                                    
                            last_commit_time[suffix] = now
                            state_changed = True
                            logger.info(f"[{suffix}] Successfully committed batch containing {len(operations)} files!")
                        except Exception as e:
                            logger.error(f"[{suffix}] Failed to execute batch commit: {e}. Will retry in next cycle.")
                    else:
                        # Throttling logging
                        logger.info(f"[{suffix}] Throttling: {len(operations)} files queued. Waiting for cooldown ({int(BATCH_COMMIT_COOLDOWN - elapsed_cooldown)}s remaining) or queue threshold ({MAX_QUEUE_EPISODES * 2 - pending_immutable_count} files remaining).")
                        
            if state_changed:
                save_state(state)
                
            time.sleep(POLLING_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Daemon gracefully stopped by user (Ctrl+C). Saving final state.")
        save_state(state)
    except Exception as e:
        logger.critical(f"Daemon crashed with critical exception: {e}")
        save_state(state)
        sys.exit(1)

if __name__ == "__main__":
    main()
