# 人工數據增強使用指南 (Group 7)

本文件說明了我們為人工示範影片重建資料（`object_poses.json`）在 `feat/data-aug` 分支中實作的離線數據增強工具。

---

## 📌 TL;DR (太長不看版)
* **把少數錄影變出好幾倍**：手動錄影 (UMI) 的資料很少，用這個腳本可以對刀叉位置微幅加噪、轉一轉角度，瞬間把 10 個錄影變出 50 個不同的擺放組合。
* **餐具大亂鬥 (交叉配對)**：新增 `--mix_episodes` 參數，把 Episode A 的叉子跟 Episode B 的刀子隨機湊一對，配出全新沒錄過的擺法。
* **整張桌子搬動 (全局剛體變換)**：新增 `--global_shift` 參數，把整張桌子與餐具相對於手臂底座進行平移與旋轉，強迫模型學習相對幾何關係。
* **對 Data Gen (自動生成) 的建議**：不用對自動生成的部分做這個增強。自動生成直接調高 `generate_procedural.py` 的次數「量大管飽」就好，因為每次 reset 本來就是隨機無限生成的。這個腳本主要用來拯救實體錄影太少的問題。
* **生成出的軌跡會是歪的嗎？**：不會。因為模擬器在 reset 後會讀取增強後的餐具坐標，並在運行時即時透過 IK 狀態機規劃出對應的手臂動作，所以動作會自動跟隨坐標移動，不需要人工去扭曲軌跡。

---

## 一、 主要設計與數據增強算子

數據增強腳本 [augment_poses.py](file:///media/user/ext4Storage/癢又ㄉ/Syncing/大學/大三下/AiCapstone/course-project/scripts/datagen/augment_poses.py) 採用純幾何矩陣運算，在不需要 OpenCV 依賴的情況下，實作了嚴謹的 3D Rodrigues 剛體變換，支援以下增強模式：

1. **空間位置微調 (Translation Jittering)**：
   * 對刀叉在 ArUco 座標系下的 $x, y$ 平移坐標加上微小隨機高斯噪聲，例如 $\sigma_t \approx 1.5\text{ cm}$（$z$ 軸維持水平高度）。
2. **抓取角度抖動 (Yaw Jittering)**：
   * 提取物體旋轉向量 $r_{vec}$ 的偏角，加上高斯噪聲，並重新投影回三維 Rodrigues 向量，模擬人類擺放餐具時的角度偏差。
3. **跨 Episode 交叉配對 (Cross-Episode Mixing)**：
   * 將不同人手錄製影片中的刀子和叉子起始位置進行隨機組合，生成全新且沒發生過的「刀叉位置配對」，顯著倍增訓練場景。
4. **全局剛體坐標變換 (Global Frame Transform)**：
   * 對整個 Episode 中的所有餐具，相對於手臂底座同步疊加一個平移與繞 Z 軸旋轉。餐具間的相對距離不變，但絕對空間位置與相機背景視覺發生改變，能極大提升網路的泛化能力。

---

## 二、 核心問題解答：我們應該對 Data Gen (自動生成) 做 Augmentation 嗎？

**建議：不需要。Data Augmentation 主要針對實體人類示範資料（解決樣本稀缺性）；對於 Procedural Data Gen 的部分，直接在 Spawn Zone 中「多做幾次，量大管飽」效果最好且更乾淨。**

### 為什麼不需要對 Data Gen 的結果做後處理增強？
1. **生成資料本質上已經是 Augmentation 後的結果**：
   * 在 [generate_procedural.py](file:///media/user/ext4Storage/癢又ㄉ/Syncing/大學/大三下/AiCapstone/course-project/scripts/datagen/generate_procedural.py) 中，每一次 episode 重置時，刀叉的位置本來就是從 `Spawning Zone` 中進行完全隨機均勻撒點的。這在統計學上，就已經是「無上限的 Augmentation」。
2. **直接生成在物理上最真實**：
   * 後置數據增強（例如手動去扭曲一條現有的軌跡）可能會導致運動不符合機器人關節的動力學限制。相比之下，直接在生成時將 `--num_demos` 調大（例如從 50 加到 1000），讓 DLS IK 狀態機直接在物理引擎中即時規劃出 1000 條全新的、完全流暢的平滑軌跡，才是最乾淨、最符合物理真實性的「量大管飽」方式。
3. **影像層面增強可在訓練中自動處理**：
   * 唯一可能需要的後處理是影像（2D Image-level）的增強（如隨機亮度、色彩抖動、模糊），這部分 LeRobot 在進行 Policy 訓練時本身就已在 DataLoader 中實作了 Image Augmentation，不需要在生成數據時手動處理。

---

## 三、 執行指令與操作說明

在開始執行前，請確保已切換至本分支，並激活虛擬環境：
```bash
git checkout feat/data-aug
source .venv/bin/activate
```

### 1. 執行人類示範資料增強
讀取 Step 3 SLAM 重建出來的 JSON，並將其增強放大為 5 倍的數據量，同時開啟跨 Episode 配對與整體座標平移：
```bash
python scripts/datagen/augment_poses.py \
    --input_poses data/YYYYMMDD-taskname/demos/mapping/object_poses.json \
    --output_poses data/YYYYMMDD-taskname/demos/mapping/object_poses_augmented.json \
    --multiplier 4 \
    --mix_episodes \
    --mix_count 30 \
    --translation_std 0.02 \
    --yaw_std 10.0 \
    --global_shift
```

### 2. 利用 CPU 快速驗證增強後的 JSON 檔是否可行
在送入重負載的 Isaac Sim 渲染前，建議先用 PyBullet CPU 腳本過濾出合格路徑：
```bash
python scripts/datagen/validate_pybullet.py \
    --object_poses data/YYYYMMDD-taskname/demos/mapping/object_poses_augmented.json \
    --gui
```

### 3. 送入模擬器中實際操作手臂驗證並錄製 LeRobot 訓練集
當確認增強後的 JSON 通過碰撞測試後，即可交給模擬器，模擬器會**自動走完手臂夾取軌跡並錄製影像與動作**：
```bash
python scripts/datagen/generate.py \
    --task HCIS-CutleryArrangement-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --object_poses data/YYYYMMDD-taskname/demos/mapping/object_poses_augmented.json \
    --lerobot_dataset_repo_id ${HF_USER}/cutlery_augmented
```
