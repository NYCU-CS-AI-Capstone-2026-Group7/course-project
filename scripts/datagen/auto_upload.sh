#!/bin/bash

# 基礎路徑與 Repo 設定
BASE_DIR=~/.cache/huggingface/lerobot/XiaoPanPanKevinPan
REPO_PREFIX="XiaoPanPanKevinPan/aicapstone_group7_cutlery_v2_replay"

# 定義要追蹤的後綴 (包含原始的、_2、_3)
TARGETS=("" "_2" "_3")

# 用關聯陣列儲存最後一次的修改時間
declare -A LAST_MOD_TIMES

# 初始抓取每個資料夾目前的最後修改時間
for suffix in "${TARGETS[@]}"; do
    DIR_NAME="aicapstone_group7_cutlery_v2_replay$suffix"
    DIR_PATH="$BASE_DIR/$DIR_NAME"
    if [ -d "$DIR_PATH" ]; then
        # 尋找目錄下最新的檔案時間戳 (Unix timestamp)
        LAST_MOD_TIMES["$suffix"]=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
    else
        LAST_MOD_TIMES["$suffix"]=0
    fi
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
            CURRENT_MOD=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
            
            # 若資料夾為空，預設時間戳為 0
            if [ -z "$CURRENT_MOD" ]; then
                CURRENT_MOD=0
            fi
            
            # 若當前最新檔案的時間 大於 上次紀錄的時間
            if [ "$CURRENT_MOD" -gt "${LAST_MOD_TIMES[$suffix]:-0}" ]; then
                echo "--------------------------------------------------------"
                echo "[$(date)] Update detected in: $DIR_NAME"
                echo "Running: huggingface-cli upload $REPO_NAME $DIR_PATH --repo-type dataset"
                
                # 執行您要求的上傳指令
                # 注意：如果您使用的指令真的是 `hf upload` (設定的 alias)，請將下方替換為 `hf upload`
                huggingface-cli upload "$REPO_NAME" "$DIR_PATH" --repo-type dataset
                
                # 如果上傳成功，才更新紀錄的時間戳
                if [ $? -eq 0 ]; then
                    LAST_MOD_TIMES["$suffix"]=$(find "$DIR_PATH" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
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
