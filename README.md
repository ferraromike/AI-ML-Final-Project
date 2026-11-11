# NBA Foul Detection & Prediction Pipeline (`YOLOv11-Pose` + `R(2+1)D-18` Dual-Stream Fusion)

A cross-platform Computer Vision and Spatiotemporal Deep Learning pipeline engineered to **detect and classify basketball fouls** from NBA Last Two Minute (L2M) broadcast video clips across **Windows, macOS (Intel & Apple Silicon M1/M2/M3/M4), and Linux (CUDA & CPU)**.

---

## 1. System Architecture

Our overhauled pipeline uses a **4-stage Dual-Stream Spatiotemporal Architecture** that pairs **3D Convolutional Visual Features** with **12-D Biomechanical Pose Kinematics** while preventing memory exhaustion on consumer laptops:

```text
+---------------------------------------------------------------------------------------------------+
| STAGE 1: NBA L2M Web & JSON Scraper (1_scrape_l2m.py)                                             |
|  • Scrapes official.nba.com L2M JSON/HTML reports (UTF-8 BOM-safe) & CDN broadcast MP4 streams    |
|  • Maps Officiating Reviews -> Binary Ground Truth:                                               |
|      - Class 1 (foul):    CC (Correct Call) + INC (Incorrect Non-Call)                            |
|      - Class 0 (no_foul): CNC (Correct Non-Call) + IC (Incorrect Call)                            |
|  • Outputs: data/l2m_metadata.csv (200 plays, 13 games) & data/l2m_incorrect_fouls.csv (43 plays) |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 2: Cross-Platform Two-Pass Action-Tube & Pose Extractor (2_preprocess_tubes.py)             |
|  • Pass 1 (Low-RAM Scan): Caps live play to first 15.0s (skipping multi-minute replay reviews),   |
|    decodes 12 downscaled 640x360 frames (< 8 MB RAM), closes VideoCapture, & runs YOLOv11-Pose    |
|    in micro-batches (--yolo-batch-size 4).                                                        |
|  • Extracts 12-D Biomechanical Contact Vector per frame (arm intrusion, torso IoU, gather elev.)  |
|  • Pass 2 (EMA Stabilized Crop): Smooths ROI boxes via bidirectional EMA & crops ONLY 16 frames   |
|    (1.5s window around peak contact) -> saves (3, 16, 224, 224) RGB + (16, 12) Pose .pt tensors   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+------------------------------------------------+ +------------------------------------------------+
| STREAM A: 3D Spatiotemporal Visual Backbone    | | STREAM B: Biomechanical Pose Encoder           |
|  • Input: (B, 3, 16, 224, 224) RGB Action Tube | |  • Input: (B, 16, 12) Skeletal Contact Tensor  |
|  • ConsistentVideoAugmentor (sync flip/jitter) | |  • 1D-CNN (12 -> 32 -> 64 channels, GELU)      |
|  • Pretrained R(2+1)D-18 (Kinetics-400)        | |  • Temporal Contact-Frame Attention Pooling    |
|  • Frozen: stem, layer1, layer2                | |    weights the exact frame of peak collision   |
|  • Fine-Tuned: layer3, layer4 -> 512-D Vector  | |  • Output: 64-D Kinematic Contact Embedding    |
+------------------------+-----------------------+ +-----------------------+------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 3 & 4: Late Fusion Classifier, Focal Loss & Threshold Calibration (3_model / 4_train_eval)  |
|  • Concatenates [512-D Visual || 64-D Pose] -> 576-D -> LayerNorm -> Linear(576, 128) -> Linear(1)|
|  • Trained with class-balanced FoulFocalBCEWithLogitsLoss (gamma=1.5, pos_weight ~ 3.65)          |
|  • Micro-batch gradient accumulation (batch_size=2, accum_steps=2 -> effective batch=4)           |
|  • Sweeps thresholds [0.25..0.75] per epoch to maximize Validation F1 -> best_foul_r2plus1d.pt    |
+---------------------------------------------------------------------------------------------------+
```

### 12-D Biomechanical Foul-Contact Features (`foul_features`)
For each of the 16 sampled frames, `YOLOv11-Pose` (`yolo11n-pose.pt`) extracts 17 COCO keypoints per player and computes:
1. **`[0]` Arm-to-Arm Proximity**: Minimum Euclidean distance between defender and ball-handler elbows/wrists.
2. **`[1]` Defender Arm $\rightarrow$ Shooter Upper-Body Intrusion**: Minimum distance from defender wrists/elbows to shooter head, shoulders, elbows, and wrists.
3. **`[2]` Torso/Hip Collision Proximity**: Minimum distance between shoulder/hip torso keypoints.
4. **`[3]` Pair Bounding-Box IoU**: Spatial overlap of the two primary interacting players.
5. **`[4]` Vertical Elevation Velocity**: Temporal gradient of ROI vertical center.
6. **`[5]` Defender Closing Speed**: Temporal gradient of upper-body intrusion.
7. **`[6]` Shooter Arm Elevation**: Normalized wrist height above shoulder level (identifies shooting motion/gather).
8. **`[7]` Defender Verticality Elevation**: Contesting defender's arm elevation.
9. **`[8]` Basketball Proximity**: Distance from detected sports ball (`CLS=32`) to the interacting player pair.
10. **`[9]` Paint Crowding Density**: Number of nearby players within $1.25\times$ player height.
11. **`[10]` Vertical Center Offset**: Relative jump/elevation difference between the two players.
12. **`[11]` Composite Foul-Contact Score**: Weighted biomechanical contact intensity used to anchor the 1.5s window.

---

## 2. Repository Structure

```text
AI-ML-Final-Project/
├── README.md                                  # Root architecture & cross-platform documentation
├── NBA_Foul_Classifier_Presentation.pptx      # Project slide deck (Intro, Problem, Method, Plan, Testing, Team)
├── NBA-Foul-Classifier-WIP/
│   ├── NBA_Foul_Classifier_Full_Pipeline.ipynb # Unified interactive notebook (run cell-by-cell!)
│   ├── 1_scrape_l2m.py                        # Stage 1: Cross-platform NBA L2M scraper & CSV sync
│   ├── 2_preprocess_tubes.py                  # Stage 2: Low-RAM YOLOv11-Pose ROI cropper & 12-D feature extractor
│   ├── 3_model_and_dataset.py                 # Stage 3: L2MFoulDataset & R(2+1)D-18 + BiomechanicalFoulEncoder
│   ├── 4_train_eval.py                        # Stage 4: Cross-platform training, F1 calibration & MP4 overlay
│   ├── requirements.txt                       # Unified cross-platform dependencies
│   └── data/
│       ├── l2m_metadata.csv                   # 200 labeled plays across 13 NBA games (157 no_foul, 43 foul)
│       ├── l2m_incorrect_fouls.csv            # 43 officiating review plays (INC & CC foul breakdowns)
│       ├── raw/                               # Downloaded broadcast MP4 clips ({foul,no_foul}/)
│       └── processed/                         # Pre-cropped (3, 16, 224, 224) + (16, 12) .pt tensors
└── discord-bot/                               # Team AI Engineering Assistant ("Fred")
    ├── bot.py                                 # Discord bot with automatic PROJECT_MEMORY.md logging
    └── PROJECT_MEMORY.md                      # Shared persistent engineering memory & decisions log
```

---

## 3. Cross-Platform Setup (Windows, macOS & Linux)

The codebase resolves all file paths via `Path(__file__).resolve().parent`, normalizes CSV paths to POSIX format, limits OpenMP/OpenCV thread pools to prevent FFmpeg deadlocks, and automatically selects `cuda`, `mps` (Apple Silicon), or `cpu`.

### macOS (Intel & Apple Silicon M1/M2/M3/M4) / Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r NBA-Foul-Classifier-WIP/requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r NBA-Foul-Classifier-WIP\requirements.txt
```

### Windows (Command Prompt `cmd.exe`)
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r NBA-Foul-Classifier-WIP\requirements.txt
```

---

## 4. Running the Program

### Option A: Run the Unified Interactive Notebook Cell-by-Cell (Recommended for Learning & Demos)
Open `NBA-Foul-Classifier-WIP/NBA_Foul_Classifier_Full_Pipeline.ipynb` in **VS Code**, **JupyterLab**, or **Google Colab**:
```bash
jupyter notebook NBA-Foul-Classifier-WIP/NBA_Foul_Classifier_Full_Pipeline.ipynb
```
- **Works Offline Out-of-the-Box**: Cell 5 includes `synthetic_fallback=True`, which automatically generates deterministic `(3, 16, 224, 224)` video tensors and `(16, 12)` pose tensors from `data/l2m_metadata.csv` if raw `.mp4` videos have not been downloaded yet.

### Option B: Run the Modular 4-Step CLI Pipeline (Lightweight Default Mode)
All CLI scripts can be run from either the repository root or inside `NBA-Foul-Classifier-WIP/`:

```bash
# Step 1: Scrape a small default batch (3 games, up to 12 MP4 clips)
python NBA-Foul-Classifier-WIP/1_scrape_l2m.py

# Or populate l2m_incorrect_fouls.csv offline from cached l2m_metadata.csv
python NBA-Foul-Classifier-WIP/1_scrape_l2m.py --fill-from-local

# Step 2: Preprocess a small batch of 12 clips (add --synthetic-fallback if offline without MP4s)
python NBA-Foul-Classifier-WIP/2_preprocess_tubes.py --max-clips 12

# Step 3: Verify 3D CNN + Biomechanical Pose Fusion tensor shapes & parameter counts
python NBA-Foul-Classifier-WIP/3_model_and_dataset.py

# Step 4: Train with lightweight defaults (5 epochs, batch_size=2, accum_steps=2, max_samples=20)
python NBA-Foul-Classifier-WIP/4_train_eval.py

# Step 5: Run single-clip inference & export an annotated MP4 video overlay
python NBA-Foul-Classifier-WIP/4_train_eval.py --predict NBA-Foul-Classifier-WIP/data/raw/foul/0022600186_2115_34.mp4
```

---

## 5. How to Increase the Amount of Data Scraped, Processed & Trained

By default, all scripts and notebook cells use **small, lightweight batch sizes** so the program runs smoothly on 8GB RAM laptops and CPU-only machines without downloading gigabytes of video or running out of memory.

When running on a machine with more disk space, RAM, or a dedicated GPU, use the flags below to scale up data collection, preprocessing, and training:

### Scaling Reference Table

| Pipeline Stage | Parameter / Flag | Lightweight Default (8GB Laptop / CPU) | Medium Scale (16GB RAM / Apple M-Series) | Full Production Scale (Dedicated CUDA GPU) | What It Controls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Scraping** (`1_scrape_l2m.py`) | `--max-games` | `3` | `25` | `100` | Number of NBA L2M game reports inspected |
| **1. Scraping** (`1_scrape_l2m.py`) | `--max-clips` | `12` | `200` | `1000` | Maximum MP4 broadcast clips downloaded |
| **1. Scraping** (`1_scrape_l2m.py`) | `--include-archive` | *Off* | *Off* | `--include-archive` | Also scrapes multi-season historical L2M archive |
| **2. Preprocessing** (`2_preprocess_tubes.py`) | `--max-clips` | `12` | `100` | `0` *(all clips)* | Number of clips converted into `.pt` tensors (`0` = all) |
| **2. Preprocessing** (`2_preprocess_tubes.py`) | `--max-scan-samples` | `12` | `24` | `32` | Number of frames scanned by YOLOv11-Pose in Pass 1 |
| **2. Preprocessing** (`2_preprocess_tubes.py`) | `--max-live-play-sec` | `15.0` | `20.0` | `25.0` | Active broadcast play horizon scanned per clip (seconds) |
| **2. Preprocessing** (`2_preprocess_tubes.py`) | `--yolo-batch-size` | `4` | `8` | `16` | Frames processed per YOLOv11-Pose micro-batch |
| **4. Training** (`4_train_eval.py`) | `--max-samples` | `20` | `200` | `0` *(all clips)* | Total `.pt` tensors loaded into Train/Val split (`0` = all) |
| **4. Training** (`4_train_eval.py`) | `--batch-size` | `2` | `4` | `8` or `16` | Micro-batch size per forward/backward pass |
| **4. Training** (`4_train_eval.py`) | `--accum-steps` | `2` *(eff. 4)* | `4` *(eff. 16)* | `4` *(eff. 32–64)* | Gradient accumulation steps (`batch_size * accum_steps`) |
| **4. Training** (`4_train_eval.py`) | `--epochs` | `5` | `20` | `30` | Total training epochs with Cosine Annealing LR |
| **4. Training** (`4_train_eval.py`) | `--num-workers` | `0` | `0` *(macOS/Win)* | `4` *(Linux CUDA)* | DataLoader background worker processes |

### Example: Scaling Up to Medium Dataset (200 Clips, 20 Epochs)
```bash
# 1. Scrape up to 25 games and 200 MP4 video clips
python NBA-Foul-Classifier-WIP/1_scrape_l2m.py --max-games 25 --max-clips 200

# 2. Preprocess all 200 clips (--max-clips 0) with 24 pose scan samples per clip
python NBA-Foul-Classifier-WIP/2_preprocess_tubes.py --max-clips 0 --max-scan-samples 24 --yolo-batch-size 8

# 3. Train on all processed tensors (--max-samples 0) for 20 epochs with effective batch size 16
python NBA-Foul-Classifier-WIP/4_train_eval.py --max-samples 0 --epochs 20 --batch-size 4 --accum-steps 4
```

### Example: Scaling Up in the Jupyter Notebook (`NBA_Foul_Classifier_Full_Pipeline.ipynb`)
In **Cell 1** of the notebook, simply edit the `CONFIG` dictionary before running the remaining cells:
```python
CONFIG = {
    "max_games": 50,             # Increase from 3 -> 50 games
    "max_scrape_clips": 400,     # Increase from 12 -> 400 downloaded clips
    "max_preprocess_clips": 0,   # Set to 0 to preprocess ALL rows in l2m_metadata.csv
    "max_live_play_sec": 20.0,   # Scan first 20 seconds of live action
    "max_scan_samples": 24,      # Increase Pass-1 pose scan resolution from 12 -> 24 frames
    "yolo_batch_size": 8,        # Increase YOLO frame batch size from 4 -> 8
    "batch_size": 4,             # Increase 3D CNN micro-batch size from 2 -> 4
    "accum_steps": 4,            # Effective batch size = 4 * 4 = 16
    "epochs": 20,                # Train for 20 full epochs
    "max_train_samples": 0,      # Set to 0 to train on the entire preprocessed dataset
}
```
And in **Cell 2** and **Cell 7**, set:
- `RUN_LIVE_WEB_SCRAPE = True` (to fetch new games from `official.nba.com`)
- `USE_PRETRAINED_KINETICS = True` (to download Kinetics-400 pretrained `R(2+1)D-18` weights)
