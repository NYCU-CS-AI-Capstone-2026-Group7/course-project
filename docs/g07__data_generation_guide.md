# 隨機程序化數據生成與輕量化驗證指南 (Group 7)

本文件說明了我們為餐具擺放任務（`HCIS-CutleryArrangement-SingleArm-v0`）在 `feat/data-generation` 分支中實作的隨機程序化生成方案與驗證腳本。

---

## 📌 TL;DR (太長不看版)
* **自動隨機擺餐具跟手肘位置**：不用手動調 JSON 檔，開局直接隨機把餐具與起點撒在桌上。
* **手夾餐具動得很順、像真人**：用三次樣條數學公式，把移動路徑自動「裁圓角」，讓手臂夾東西時不卡頓、極度流暢。
* **手不會自己折斷或撞爛盤子**：用了七自由度手臂的零空間公式，手臂伸出去時手肘會自動擺在最舒適的位置，避免奇異點。
* **可選擇要不要「把刀叉轉正」**：新增 `--fix_knife_yaw` 與 `--fix_fork_yaw`，一鍵決定要不要強制把抓取角度鎖死為 0。
* **不吃效能、一秒驗證有沒有撞到**：做了一個超輕量的 PyBullet 腳本，只用 CPU、不用開 Isaac Sim、不用下載幾 GB 的 3D USD 檔。
* **前置過濾人類影片**：驗證程式能直接吃 UMI 產出的 `object_poses.json`，在 CPU 端用 Franka 手臂預演抓取，一秒抓出哪些人類錄影在模擬器中會撞車或解不出 IK，直接剔除，避免污染模型！
* **與舊代碼完全分開**：全部做成外掛 (Addon) 形式，不用的話隨時可以當作沒這回事。

---

## 一、 主要設計與架構

我們架構的核心目標是**「高度解耦、外掛式 (Addon)」**：

### 1. 隨機程序化資料生成主程式 (`scripts/datagen/generate_procedural.py`)
* **防重疊隨機 Spawning**：每次 Reset 後，隨機在檯面合理範圍生成刀叉與起點，並透過距離過濾避免重疊。
* **全局變換 (Global Frame Perturbation)**：對整個餐桌與餐具坐標疊加隨機平移與 Z 軸旋轉，強迫 Policy 學習相對幾何關係。
* **新增可選的刀叉 Yaw 鎖定開關**：
  * 新增 `--fix_knife_yaw` 和 `--fix_fork_yaw` 參數，允許開發者可選地將刀子或叉子的抓取朝向角度（相對工作區）鎖死為 `0.0`，滿足助教設置的定角抓取需求。

### 2. 樣條平滑化狀態機 (`ProceduralCutleryArrangementStateMachine`)
* 檔案位置：[procedural_cutlery.py](file:///media/user/ext4Storage/癢又ㄉ/Syncing/大學/大三下/AiCapstone/course-project/packages/simulator/src/simulator/datagen/state_machine/procedural_cutlery.py)
* **Catmull-Rom 三次樣條插值**：藉由幾何控制點擬合出 $C^1$ 連續的 3D 笛卡爾平滑路徑，消除段落間的停頓。
* **零空間控制 (Null-Space IK)**：控制手臂 EE 時將剩餘的 1-DoF 投影至零空間中，將肘部推向默認舒適姿態。

### 3. CPU 輕量化快速碰撞驗證器 (`scripts/datagen/validate_pybullet.py`)
* **輕量 URDF/幾何模擬**：載入內建的 Franka URDF 手臂模型，並將餐具與盤子用簡化包圍體（Cylinders & Boxes）模擬，透過 CPU 執行碰撞攔截。

---

## 二、 舊代碼摘要與本分支改動對比

### 1. 原本代碼與文件摘要
* **`docs/synthetic_data_generation.md` (原版)**：說明如何將 UMI 重建的位姿資料，在模擬器 (Isaac Lab) 中結合 scripted 狀態機執行 pick-and-place，並使用 `LeRobotRecorderManager` 錄製生成 LeRobot 訓練資料集的 5 大步驟。
* **`scripts/datagen/generate.py` (原版)**：依賴傳入 `--object_poses` JSON 檔案，以 `load_episode_poses` 載入固定 episodes。每次 Reset 時呼叫 `_apply_episode_poses` 套用物體位置，由舊狀態機直線插值完成抓取並錄製。

### 2. 本分支 (`feat/data-generation`) 改動與新增
* **全新程序化生成主程式**：新增了平行腳本 `scripts/datagen/generate_procedural.py`，不再強綁人類的 JSON，可自由在 Isaac Sim 中全自動生成數據。
* **引入三次樣條與零空間狀態機**：新增了 `packages/simulator/src/simulator/datagen/state_machine/procedural_cutlery.py`，將生硬的分段直線插值升級為平滑的 Catmull-Rom 軌跡，並引入冗餘零空間避障與奇異點防範。
* **新增 CPU 獨立驗證腳本**：新增了 `scripts/datagen/validate_pybullet.py`，完全在 CPU 上運行，可用於隨機軌跡測試或實體 Demo JSON 前置過濾。

---

## 三、 驗證人類操作的可行性與局限性

本專案的驗證程式碼（`validate_pybullet.py`）**完全可以用來驗證人類操作影片（Step 3 產出的 JSON）**！

* **驗證原理**：
  當您使用 `--object_poses data/YYYYMMDD-taskname/demos/mapping/object_poses.json` 執行驗證時，腳本會將人類示範中偵測到的餐具「初始擺放位姿（Spawning Poses）」載入到 PyBullet 環境中，並讓虛擬 Franka 手臂執行抓取軌跡。
* **局限性說明**：
  * 由於 `object_poses.json` 中**只記錄了餐具的初始位置**，而沒有記錄人類手臂完整的「實體運動軌跡」。
  * 因此，此驗證本質上是驗證：**「在該人手示範擺放的配置下，Franka 手臂以其自身的運動學與樣條規劃，是否能順利抓起餐具且不撞擊盤子、也不超出關節極限」**。
  * 這能有效在實地訓練前，**過濾出那些因擺放位置過於極端而不可行的 Demo Episode**，防止無效軌跡污染 Diffusion Policy 訓練集。

---

## 四、 執行指令與操作說明

### 1. 執行實體錄影資料前置驗證 (UMI JSON 檔驗證)
若想對實體 Demo JSON 檔案進行驗證，並在過程中鎖定刀/叉抓取偏角為零度：
```bash
# 驗證 Step 3 輸出的 JSON 檔（以 GUI 可視化運行，且強制刀與叉抓取 yaw 為 0）
python scripts/datagen/validate_pybullet.py \
    --object_poses data/YYYYMMDD-taskname/demos/mapping/object_poses.json \
    --fix_knife_yaw \
    --fix_fork_yaw \
    --gui

# 無視窗純 CPU 快速批量檢測（輸出合格與不合格 Episode 列表）
python scripts/datagen/validate_pybullet.py \
    --object_poses data/YYYYMMDD-taskname/demos/mapping/object_poses.json
```

### 2. 執行隨機程序化路徑 CPU 測試
```bash
# 帶 GUI 視覺化隨機跑 10 次，且鎖定刀的抓取 yaw 為 0
python scripts/datagen/validate_pybullet.py --num_runs 10 --fix_knife_yaw --gui
```

### 3. 執行 Isaac Sim (USD場景) 完整模擬驗證
```bash
# 帶 GUI 模擬運行（不儲存數據，鎖定叉的抓取 yaw 為 0）
python scripts/datagen/generate_procedural.py \
    --task HCIS-CutleryArrangement-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --fix_fork_yaw
```

### 4. 執行 procedural 數據生成與錄製 (LeRobot 數據集)
```bash
python scripts/datagen/generate_procedural.py \
    --task HCIS-CutleryArrangement-SingleArm-v0 \
    --num_envs 1 \
    --device cuda \
    --enable_cameras \
    --record \
    --use_lerobot_recorder \
    --num_demos 50 \
    --fix_knife_yaw \
    --fix_fork_yaw \
    --lerobot_dataset_repo_id ${HF_USER}/cutlery_procedural
```
