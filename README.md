# NBA Foul Detection & Prediction Pipeline (`YOLOv11-Pose` + `R(2+1)D-18` Dual-Stream Fusion)

A cross-platform Computer Vision and Spatiotemporal Deep Learning pipeline engineered to **detect and classify basketball fouls** from NBA Last Two Minute (L2M) broadcast video clips across **Windows, macOS (Intel & Apple Silicon M1/M2/M3/M4), and Linux (CUDA & CPU)**.

---

## 1. System Architecture

Our overhauled pipeline uses a **4-stage Dual-Stream Spatiotemporal Architecture (~33.8M parameters)** pairing **3D Convolutional Visual Features (512-D)** with **High-Capacity Pose Kinematics (256-D)** into a deep **768-D Late-Fusion Classification Head**:

```text
+---------------------------------------------------------------------------------------------------+
| STAGE 1: NBA L2M Web & JSON Scraper (1_scrape_l2m.py)                                             |
|  • Scrapes official.nba.com L2M JSON/HTML reports (UTF-8 BOM-safe) & CDN broadcast MP4 streams    |
|  • Maps Officiating Reviews -> Binary Ground Truth:                                               |
|      - Class 1 (foul):    CC (Correct Call) + INC (Incorrect Non-Call)                            |
|      - Class 0 (no_foul): CNC (Correct Non-Call) + IC (Incorrect Call)                            |
|  • Outputs: data/l2m_metadata.csv (349 plays, 18+ games) & data/l2m_incorrect_fouls.csv (97 plays) |
|  • Over 130+ raw broadcast MP4 clips downloaded in data/raw/{foul,no_foul}/                       |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 2: Cross-Platform Two-Pass Action-Tube & Pose Extractor (2_preprocess_tubes.py)             |
|  • Pass 1 (Low-RAM Scan): Caps live play to first 15.0s (skipping multi-minute replay reviews),   |
|    decodes downscaled 640x360 frames (< 8 MB RAM), closes VideoCapture, & runs YOLOv11-Pose       |
|    in micro-batches (--yolo-batch-size 4).                                                        |
|  • Extracts 12-D Biomechanical Contact Vector per frame (arm distance, torso overlap, gather)     |
|  • Pass 2 (EMA Stabilized Crop): Smooths ROI boxes via bidirectional EMA & crops 16 frames        |
|    (1.5s window around peak contact) -> saves (3, 16, 224, 224) RGB + (16, 12) Pose .pt tensors   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+------------------------------------------------+ +------------------------------------------------+
| STREAM A: 3D Spatiotemporal Visual Backbone    | | STREAM B: Biomechanical Pose Encoder           |
|  • Input: (B, CHANNELS=3, FRAMES=16, 224, 224) | |  • Input: (B, FRAMES=16, POSE_FEATURES=12)     |
|  • Pretrained R(2+1)D-18 (Kinetics-400)        | |  • 4-Head Multi-Head Temporal Self-Attention    |
|  • Frozen Backbone on MPS/CPU with precomputed | |  • Residual LayerNorm & Kinematic Projection    |
|    512-D visual embeddings cache (212 KB)      | |  • Temporal Mean (128) + Max (128) Pooling     |
|  • Output: VIDEO_FEATURE_DIM = 512-D Embedding | |  • Output: POSE_HIDDEN_DIM = 256-D Embedding   |
+------------------------+-----------------------+ +-----------------------+------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 3 & 4: Deep Late Fusion Classifier & Full Annotated Video Export (3_model / 4_train_eval)   |
|  • Late Fusion: COMBINED_FEATURE_DIM = 768-D [512-D Visual || 256-D Pose Kinematics]              |
|  • Deep Classification Head: LayerNorm -> Linear(768, 512) -> LayerNorm -> GELU -> Dropout ->    |
|    Linear(512, 256) -> LayerNorm -> GELU -> Linear(256, 1) (~33.8M parameters / 613K trainable)   |
|  • Trained with class-balanced FoulFocalBCEWithLogitsLoss with Label Smoothing (gamma=1.5, eps=0.05)|
|  • Trains 10 epochs in under 6 seconds with cached embeddings and zero Apple Silicon memory stalls|
|  • Threshold Calibration sweeps [0.20..0.80] optimizing composite F1 + Balanced Accuracy         |
|  • Export Annotated Video: Disputed-play multi-player tracking enclosing ALL players involved,    |
|    dynamic green-to-red box transition at exact foul frame, and top-left HUD stats panel stating  |
|    WHAT FOUL occurred and WHO FOULED WHO into data/annotated/<clip_id>_annotated.mp4              |
+---------------------------------------------------------------------------------------------------+
```

### Simplified Model Dimensions & Feature Names
The architecture utilizes explicit, standardized dimension constants:
- **Action Tube Dimensions**: `CHANNELS = 3`, `FRAMES = 16`, `HEIGHT = 224`, `WIDTH = 224`
- **Pose Feature Count**: `POSE_FEATURES = 12`
- **Embedding Dimensions**: `VIDEO_FEATURE_DIM = 512`, `POSE_HIDDEN_DIM = 256`, `COMBINED_FEATURE_DIM = 768`, `CLASSIFIER_HIDDEN_DIM = 512`

#### 12 Mechanical Foul Dimensions (`MECHANICAL_FOUL_DIMENSIONS`):
Directly grounded in NBA Rulebook mechanics (Rule 12: Fouls and Penalties — cylinder principle, legal guarding position, and verticality):
1. **`[0] arm_contact_intrusion`**: Defender hand/wrist collision with shooter arm or shooting wrist.
2. **`[1] body_cylinder_penetration`**: Torso and hip collision invading the offensive player's established cylinder.
3. **`[2] lower_body_landing_violation`**: Defender feet/legs stepping into shooter's designated landing zone.
4. **`[3] defender_verticality_angle`**: Angle deviation from legal 90-degree vertical contest.
5. **`[4] defender_lateral_positioning`**: Defender lateral offset relative to the offensive player's drive path.
6. **`[5] offensive_displacement`**: Trajectory deflection and balance disruption of the shooter upon contact.
7. **`[6] defender_to_ball_distance`**: Proximity of defender hand/wrist to the basketball.
8. **`[7] ball_to_shooter_distance`**: Proximity of ball to shooter (tracks shooting gather vs. release).
9. **`[8] ball_strike_vs_body_ratio`**: Ratio of hand-to-ball distance vs. hand-to-body distance (clean block vs. hack).
10. **`[9] relative_closing_velocity`**: Closing velocity of defender into offensive player.
11. **`[10] momentum_transfer_shock`**: Sudden deceleration and physical collision jolt upon impact.
12. **`[11] airborne_contest_apex_diff`**: Vertical contest elevation differential between players.

### Dynamic Video Output & HUD Telemetry
Exported annotated MP4 videos in `data/annotated/<play_id>_annotated.mp4` deliver professional referee-grade visualization:
- **Disputed Play Enclosing Bounding Box**: Continuously encloses **ALL players involved** in the disputed play (`player_disadvantaged` shooter/driver and `player_committing` contesting defender) throughout 100% of the video.
- **Bidirectional Court Player Tracking**: 
  - *Spectator Rejection Filter*: Eliminates seated audience and stands detections by enforcing court plane geometry ($y_2 > 0.38 \times \text{frame\_h}$, $y_1 > 0.15 \times \text{frame\_h}$, $0.10 \le h \le 0.55$, aspect ratio $\ge 0.95$).
  - *Action Tube Anchor*: Identifies the disputed interaction at peak contact from the action tube ROI, anchoring onto the offensive driver and contesting defender.
  - *Bidirectional Matching*: Propagates forward and backward frame-by-frame with optimal 2xK bipartite matching and mutual exclusion, preventing track collapse and ensuring both players remain locked across all 600 frames.
- **Where & When Foul Occurs**: Pinpoints the exact frame $t_{\text{foul}}$ and timestamp where physical contact surges into illegal impact.
- **Dynamic Color Switching**: Bounding box displays **GREEN** with tag `LEGAL PLAY | NO FOUL` until the foul occurs, at which point it **TURNS RED** with tag `*** FOUL DETECTED: {call_type} ({player_committing} on {player_disadvantaged}) ***`.
- **Top-Left HUD Stats Overlay**:
  - `STATUS`: Dynamic green indicator before foul $\to$ bright red indicator at and after foul (`*** FOUL CALLED AT FRAME {foul_frame} ({foul_time}s) ***`).
  - `WHAT FOUL`: Explicit foul call type (e.g. `Shooting Foul`, `Personal Foul`).
  - `WHO FOULED WHO`: Identifies the exact matchup (e.g. `Caleb Wilson fouled LeBron James`).
  - `METRICS & TELEMETRY`: Displays `P(FOUL)`, calibrated threshold, L2M review decision (`CC`, `INC`, `CNC`), and per-frame mechanical contact intensity.
  - `INDIVIDUAL HIGHLIGHTS`: Subtle individual badges for `Offense: {player}` (gold) and `Defense: {player}` (amber) with a physical collision vector connector.

---

## 2. Repository Structure

```text
AI-ML-Final-Project/
├── README.md                                  # Root architecture & cross-platform documentation
├── NBA_Foul_Classifier_Presentation.pptx      # Project slide deck (Intro, Problem, Method, Plan, Testing, Team)
├── NBA-Foul-Classifier-WIP/
│   ├── NBA_Foul_Classifier_Full_Pipeline.ipynb # Unified interactive notebook (run cell-by-cell!)
│   ├── 1_scrape_l2m.py                        # Stage 1: NBA L2M scraper & master CSV sync (349 plays, 130+ MP4s)
│   ├── 2_preprocess_tubes.py                  # Stage 2: Low-RAM YOLOv11-Pose ROI cropper & 12-D feature sampler
│   ├── 3_model_and_dataset.py                 # Stage 3: High-capacity FoulClassifier3D (~33.8M params, 768-D fusion)
│   ├── 4_train_eval.py                        # Stage 4: Training, F1 calibration & continuous video annotation
│   ├── requirements.txt                       # Unified cross-platform dependencies
│   ├── best_foul_r2plus1d.pt                  # Calibrated dual-stream checkpoint
│   └── data/
│       ├── l2m_metadata.csv                   # 349 labeled plays across 18+ NBA games
│       ├── l2m_incorrect_fouls.csv            # 97 officiating review plays (INC & CC breakdowns)
│       ├── raw/                               # Downloaded broadcast MP4 clips ({foul,no_foul}/)
│       ├── processed/                         # Pre-cropped (3, 16, 224, 224) + (16, 12) .pt tensors
│       ├── test/                              # Dedicated test clips set aside for evaluation (isolated from training)
│       └── annotated/                         # Continuous 100% full-clip annotated prediction MP4 videos
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

# Step 4: Train with fast cached visual embeddings (5-10 epochs in under 6 seconds)
python NBA-Foul-Classifier-WIP/4_train_eval.py --epochs 5 --batch-size 2 --freeze-backbone --max-samples 0

# Step 5: Run batch test inference & export continuous annotated MP4s (saved to data/annotated/)
# (Drop any .mp4 files into data/test/ and run --annotate-test; test data is strictly isolated from training)
python NBA-Foul-Classifier-WIP/4_train_eval.py --annotate-test

# Or run prediction on a single clip:
python NBA-Foul-Classifier-WIP/4_train_eval.py --predict NBA-Foul-Classifier-WIP/data/test/0022600196_2132_05.mp4
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

---

## 6. Dedicated Test Set & Continuous Video Annotation (`data/test/` & `data/annotated/`)

### Test Set Isolation Workflow
To prevent data leakage, clips in `data/test/` are **strictly segregated** from training and preprocessing:
1. `2_preprocess_tubes.py` scans `data/test/` and excludes all test video stems from training action-tube generation.
2. `3_model_and_dataset.py` (`L2MFoulDataset`) ignores test stems during training/validation dataset discovery.
3. Users can drop **any video file** (`.mp4`, `.mov`, `.avi`, `.mkv`) into `data/test/`.

### Running Batch Annotation on the Test Set
```bash
# Run batch prediction and continuous video annotation on all clips in data/test/
python NBA-Foul-Classifier-WIP/4_train_eval.py --annotate-test

# Or pass the test directory path directly:
python NBA-Foul-Classifier-WIP/4_train_eval.py --predict NBA-Foul-Classifier-WIP/data/test
```
Annotated videos are automatically saved to `NBA-Foul-Classifier-WIP/data/annotated/<clip_name>_annotated.mp4`.

### Continuous Disputed-Play Video Tracking Specifications
1. **Unified Enclosing Box**: Encloses **ALL players involved** in the disputed play (offensive shooter/driver + contesting defender) continuously across **100% of video frames**.
2. **Active Court Geometry & Official Exclusion**:
   - Strictly rejects foreground officials (e.g. Referee #43), sideline bench players, and baseline photographers via court-floor constraints ($y_2 \le 0.72 \times H$, $y_{\text{center}} \in [0.25H, 0.68H]$, $x_{\text{center}} \in [0.12W, 0.80W]$).
   - Speed gating (`max_d = 160.0 px`) and bidirectional bipartite tracking prevent the box from jumping to bystanders.
3. **Dynamic Green-to-Red Color Transition**:
   - The bounding box starts **GREEN** with tag `LEGAL PLAY | NO FOUL`.
   - At the exact frame contact surges into illegal impact ($t_{\text{foul}}$), the box **turns RED** with tag `*** FOUL DETECTED: {call_type} ({defender} on {shooter}) ***`.
4. **Top-Left HUD Statistics Panel**:
   - `STATUS`: Dynamic status displaying green before foul and red after foul with timestamp (`*** FOUL CALLED AT FRAME {foul_frame} ({foul_time_sec:.2f}s) ***`).
   - `WHAT FOUL`: Explicit foul type (`Shooting Foul`, `Personal Foul`, `Loose Ball Foul`).
   - `WHO FOULED WHO`: Exact matchup (e.g. `Dalen Terry fouled Ricky Council IV`, `Josh Giddey fouled Tyrese Maxey`).
   - `TELEMETRY`: Displays `P(FOUL)`, calibrated threshold, L2M review decision (`INC`, `CC`, `CNC`), and per-frame physical contact intensity.

---

## 7. PyTorch DataLoader Collation Safeguards & Self-Healing Architecture (Cell 7 Safeguard)

To prevent `RuntimeError: stack expects each tensor to be equal size, but got [512] at entry 0 and [3, 16, 224, 224] at entry 1`:

1. **Precomputed Visual Embeddings**: All 123 preprocessed clips are indexed with 512-D Kinetics-400 embeddings in `data/processed/visual_embeddings_cache.pt` and their respective `.pt` files.
2. **Self-Healing Dataset Loader (`L2MFoulDataset`)**: If a new sample or un-cached clip is encountered, `_extract_and_cache_embedding()` computes its 512-D embedding on-the-fly using the singleton Kinetics backbone and writes it to cache, guaranteeing that `use_cached_embeddings=True` strictly returns a 1D `[512]` tensor.
3. **Safe Collation Operator (`safe_foul_collate`)**: Integrated in `4_train_eval.py` and notebook Cell 7. Automatically detects any heterogeneous tensor dimensions before stacking and harmonizes them dynamically, completely eliminating PyTorch batch collation crashes.
4. **End-to-End Pipeline Verification**: Full notebook runs flawlessly across all 15 cells from data sync through training (10 epochs in under 6 seconds, 81.4% accuracy, 0.778 F1, 0.828 ROC-AUC) to continuous 3-clip test set annotation.

---

## 8. Repository Organization & Model Weights Specification (`weights/`)

To maintain clean separation between code, dataset artifacts, and model weights, all binary foundation models and trained checkpoints are centralized under `NBA-Foul-Classifier-WIP/weights/`:

```text
AI-ML-Final-Project/
├── README.md                      # Comprehensive project documentation & architecture guide
├── .gitignore                     # Ignores large binary weights, raw media, and venv artifacts
├── discord-bot/                   # Fred: autonomous AI engineering teammate
│   ├── bot.py                     # Discord bot client powered by high-speed LLM inference
│   ├── PROJECT_MEMORY.md          # Persistent engineering memory & milestones log
│   └── README.md                  # Discord bot setup and usage instructions
└── NBA-Foul-Classifier-WIP/       # Core Computer Vision & Deep Learning Pipeline
    ├── 1_scrape_l2m.py            # Stage 1: L2M web scraper & local metadata sync
    ├── 2_preprocess_tubes.py      # Stage 2: YOLOv11-Pose extractor & 12 Foul Dimensions
    ├── 3_model_and_dataset.py     # Stage 3: Dual-stream dataset & 33.8M param architecture
    ├── 4_train_eval.py            # Stage 4 & 5: Training loop, calibration & test video annotation
    ├── NBA_Foul_Classifier_Full_Pipeline.ipynb # Complete 15-cell executable Jupyter notebook
    ├── requirements.txt           # Python dependencies
    ├── weights/                   # Centralized model weights & checkpoints directory
    │   ├── README.md              # Detailed model specifications, inputs, outputs, and parameters
    │   ├── yolo11n.pt             # Ultralytics YOLOv11 Nano object detection (2.6M params)
    │   ├── yolo11n-pose.pt        # Ultralytics YOLOv11 Nano pose estimation (2.9M params)
    │   └── best_foul_r2plus1d.pt  # Trained Dual-Stream Classifier checkpoint (~33.8M params)
    └── data/                      # Structured dataset storage
        ├── README.md              # Dataset documentation & metadata column definitions
        ├── l2m_metadata.csv       # 349 labeled plays across 24 NBA games
        ├── l2m_incorrect_fouls.csv# 97 officiating review plays (INC & CC breakdowns)
        ├── raw/                   # Raw broadcast MP4 clips downloaded from NBA CDN
        ├── test/                  # Segregated test clips (isolated from training)
        ├── processed/             # Preprocessed .pt action tubes & visual_embeddings_cache.pt
        └── annotated/             # Exported prediction videos with continuous bounding boxes
```

### What Each Weight File Does:
* **`yolo11n.pt` (Object Detection, 5.6 MB)**: Detects players (class `0`) and the basketball (class `32`) in full broadcast video. Powers the court-plane geometry filter ($y_2 \le 0.72 \times H$, $y_c \in [0.25H, 0.68H]$, $x_c \in [0.12W, 0.80W]$) and bidirectional player tracking, preventing the bounding box from jumping to referees or sideline bystanders.
* **`yolo11n-pose.pt` (Skeletal Joint Estimation, 6.2 MB)**: Detects 17 keypoints per player (wrists, elbows, shoulders, hips, knees, ankles) to compute the 12 Mechanical Foul Dimensions (arm contact intrusion, cylinder penetration, landing zone violations, verticality angle).
* **`best_foul_r2plus1d.pt` (Classifier Checkpoint, 127.8 MB)**: The trained PyTorch model checkpoint from Stage 4 containing the 4-head temporal attention encoder and late-fusion classification head, calibrated threshold $\tau$, and validation metrics.

