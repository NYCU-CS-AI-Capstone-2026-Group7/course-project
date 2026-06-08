#!/bin/bash

# 基礎路徑與 Repo 設定
BASE_DIR=~/.cache/huggingface/lerobot/XiaoPanPanKevinPan
REPO_PREFIX="XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay"

# 定義要追蹤的後綴 (包含原始的、_2、_3)
TARGETS=("" "_2" "_3")

# Parse arguments
OVERWRITE_REMOTE=false
if [ "$1" == "--overwrite" ]; then
    OVERWRITE_REMOTE=true
    echo "[INFO] Overwrite remote flag is set. The first upload will clear remote repo."
fi

# 用關聯陣列儲存最後一次的修改時間、最後上傳時間與是否為首次上傳的標記
declare -A LAST_MOD_TIMES
declare -A LAST_UPLOAD_TIMES
declare -A IS_FIRST_UPLOAD

# 初始抓取每個資料夾目前的最後修改時間
for suffix in "${TARGETS[@]}"; do
    DIR_NAME="aicapstone_group7_cutlery_v2_replay$suffix"
    DIR_PATH="$BASE_DIR/$DIR_NAME"
    KEY="${suffix:-main}"
    if [ -d "$DIR_PATH" ]; then
        # 尋找目錄下最新的檔案時間戳 (Unix timestamp)
        LAST_MOD_TIMES["$KEY"]=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
    else
        LAST_MOD_TIMES["$KEY"]=0
    fi
    LAST_UPLOAD_TIMES["$KEY"]=0
    IS_FIRST_UPLOAD["$KEY"]=true
done

echo "Starting HuggingFace Dataset Auto-Uploader..."
echo "Polling for changes every 30 seconds..."

# 建立一個全局的臨時快照基底目錄
SNAPSHOT_BASE_DIR="/tmp/hf_upload_snapshot"
mkdir -p "$SNAPSHOT_BASE_DIR"

# 用於刪除全局快照目錄的清理鉤子
trap 'rm -rf "$SNAPSHOT_BASE_DIR"' EXIT

# 進入無窮迴圈每 30 秒輪詢一次
while true; do
    for suffix in "${TARGETS[@]}"; do
        DIR_NAME="aicapstone_group7_cutlery_v2_replay$suffix"
        DIR_PATH="$BASE_DIR/$DIR_NAME"
        REPO_NAME="$REPO_PREFIX$suffix"
        SNAPSHOT_DIR="$SNAPSHOT_BASE_DIR/$DIR_NAME"
        
        if [ -d "$DIR_PATH" ]; then
            KEY="${suffix:-main}"
            CURRENT_MOD=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
            
            # 若資料夾為空，預設時間戳為 0
            if [ -z "$CURRENT_MOD" ]; then
                CURRENT_MOD=0
            fi
            
            # 若當前最新檔案的時間 大於 上次紀錄的時間
            if [ "$CURRENT_MOD" -gt "${LAST_MOD_TIMES[$KEY]:-0}" ]; then
                NOW=$(date +%s)
                ELAPSED=$((NOW - ${LAST_UPLOAD_TIMES[$KEY]:-0}))
                MIN_UPLOAD_INTERVAL=600 # 最小上傳時間間隔：10 分鐘 (秒)
                
                if [ "$ELAPSED" -lt "$MIN_UPLOAD_INTERVAL" ]; then
                    WAIT_MORE=$((MIN_UPLOAD_INTERVAL - ELAPSED))
                    echo "[$(date)] [THROTTLE] Changes detected in $DIR_NAME, but the minimum upload interval has not elapsed (last upload was ${ELAPSED}s ago, limit is ${MIN_UPLOAD_INTERVAL}s). Postponing upload (retry in ${WAIT_MORE}s)..."
                    continue
                fi
                
                echo "--------------------------------------------------------"
                echo "[$(date)] Update detected and minimum interval cleared: $DIR_NAME (last upload was ${ELAPSED}s ago)"
                
                echo "[INFO] Creating static directory snapshot using rsync to prevent read/write race condition..."
                mkdir -p "$SNAPSHOT_DIR"
                # 使用 rsync 增量複製到臨時目錄，排除上傳期間的檔案變動鎖死與殘留的 tmp* 檔案
                rsync -a --delete --exclude "tmp*" "$DIR_PATH/" "$SNAPSHOT_DIR/"
                
                # 執行上傳指令
                if [ "$OVERWRITE_REMOTE" = true ] && [ "${IS_FIRST_UPLOAD[$KEY]}" = true ]; then
                    # 第一次清空並覆蓋時，檔案通常極少，使用普通 hf upload 帶 --delete
                    echo "Running: hf upload $REPO_NAME $SNAPSHOT_DIR --repo-type dataset --delete '*'"
                    hf upload "$REPO_NAME" "$SNAPSHOT_DIR" --repo-type dataset --delete '*'
                else
                    # 增量上傳或大型上傳，使用 upload-large-folder 避免 504 逾時
                    echo "Running: hf upload-large-folder $REPO_NAME $SNAPSHOT_DIR --repo-type dataset"
                    hf upload-large-folder "$REPO_NAME" "$SNAPSHOT_DIR" --repo-type dataset
                fi
                
                UPLOAD_STATUS=$?
                
                # 刪除臨時快照以釋放空間
                rm -rf "$SNAPSHOT_DIR"
                
                # 如果上傳成功，才更新紀錄的時間戳
                if [ $UPLOAD_STATUS -eq 0 ]; then
                    LAST_MOD_TIMES["$KEY"]=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
                    LAST_UPLOAD_TIMES["$KEY"]=$(date +%s)
                    IS_FIRST_UPLOAD["$KEY"]=false
                    echo "[$(date)] Successfully uploaded $REPO_NAME"
                else
                    echo "[$(date)] Upload failed for $REPO_NAME. Will retry in next cycle."
                fi
                echo "--------------------------------------------------------"
            fi
        fi
    done
    
    sleep 30
done
