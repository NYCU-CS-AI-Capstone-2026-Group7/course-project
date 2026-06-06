#!/usr/bin/env python3
"""
Hugging Face Dataset tmp* Folder Cleaner

This script scans Hugging Face dataset repositories for any root-level folders 
starting with 'tmp' (leftovers from interrupted uploads) and deletes them.
"""

import sys
from huggingface_hub import HfApi

def clean_tmp_folders(repo_id, dry_run=True):
    api = HfApi()
    
    print(f"[INFO] Retrieving file list for dataset repo: {repo_id}...")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[ERROR] Failed to fetch repo files: {e}")
        return
        
    # 尋找根目錄下所有以 'tmp' 開頭的資料夾或檔案
    tmp_items = set()
    for f in files:
        parts = f.split("/")
        if parts[0].startswith("tmp"):
            tmp_items.add(parts[0])
            
    if not tmp_items:
        print("[INFO] No tmp* folders or files found. Clean!")
        return
        
    print(f"[FOUND] Detected {len(tmp_items)} temporary items to clean:")
    for item in sorted(tmp_items):
        print(f"  - {item}")
        
    if dry_run:
        print("[INFO] Dry-run mode: NO files were deleted. Set dry_run=False to execute deletion.")
        return
        
    # 實際執行刪除的區塊
    print(f"\n[INFO] Starting deletion of {len(tmp_items)} items from {repo_id}...")
    for item in sorted(tmp_items):
        try:
            print(f"  -> Deleting '{item}'...")
            api.delete_folder(
                repo_id=repo_id,
                path_in_repo=item,
                repo_type="dataset",
                commit_message=f"Clean up temporary upload directory: {item}"
            )
            print(f"  [SUCCESS] Deleted '{item}'")
        except Exception as e:
            print(f"  [FAILED] Could not delete '{item}': {e}")

if __name__ == "__main__":
    # 請在此放入您要清理的 Hugging Face Dataset 倉庫名稱
    TARGET_REPOS = [
        "XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay",
        "XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_2",
        "XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay_3"
    ]
    
    # 預設為 dry_run=True（僅掃描不刪除）
    # 當您確認掃描結果符合預期後，可以將 dry_run 改為 False 來進行實際刪除
    DRY_RUN = False
    
    for repo in TARGET_REPOS:
        print("\n" + "="*60)
        clean_tmp_folders(repo, dry_run=DRY_RUN)
        print("="*60)
