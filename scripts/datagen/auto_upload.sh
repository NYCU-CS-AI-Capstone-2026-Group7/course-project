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

# 用關聯陣列儲存最後一次的修改時間與是否為首次上傳的標記
declare -A LAST_MOD_TIMES
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
    IS_FIRST_UPLOAD["$KEY"]=true
done

echo "Starting HuggingFace Dataset Auto-Uploader..."
echo "Polling for changes every 30 seconds..."

# 進入無窮迴圈每 30 秒輪詢一次
while true; do
    for suffix in "${TARGETS[@]}"; do
        DIR_NAME="aicapstone_group7_cutlery_v2_replay$suffix"
        DIR_PATH="$BASE_DIR/$DIR_NAME"
        REPO_NAME="$REPO_PREFIX$suffix"
        
        if [ -d "$DIR_PATH" ]; then
            KEY="${suffix:-main}"
            CURRENT_MOD=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
            
            # 若資料夾為空，預設時間戳為 0
            if [ -z "$CURRENT_MOD" ]; then
                CURRENT_MOD=0
            fi
            
            # 若當前最新檔案的時間 大於 上次紀錄的時間
            if [ "$CURRENT_MOD" -gt "${LAST_MOD_TIMES[$KEY]:-0}" ]; then
                echo "--------------------------------------------------------"
                echo "[$(date)] Update detected in: $DIR_NAME"
                
                # 執行您要求的上傳指令
                if [ "$OVERWRITE_REMOTE" = true ] && [ "${IS_FIRST_UPLOAD[$KEY]}" = true ]; then
                    echo "Running: hf upload $REPO_NAME $DIR_PATH --repo-type dataset --delete '*'"
                    hf upload "$REPO_NAME" "$DIR_PATH" --repo-type dataset --delete '*'
                else
                    echo "Running: hf upload $REPO_NAME $DIR_PATH --repo-type dataset"
                    hf upload "$REPO_NAME" "$DIR_PATH" --repo-type dataset
                fi
                
                # 如果上傳成功，才更新紀錄的時間戳
                if [ $? -eq 0 ]; then
                    LAST_MOD_TIMES["$KEY"]=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
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
