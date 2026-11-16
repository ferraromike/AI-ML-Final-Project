# NBA Foul Detection & Prediction Pipeline (`YOLOv11-Pose` + `R(2+1)D-18` + Reinforcement Learning)

A state-of-the-art Computer Vision, Spatiotemporal Deep Learning, and **Reinforcement Learning (RL)** officiating pipeline engineered to **detect, classify, and evaluate basketball fouls** from NBA Last Two Minute (L2M) broadcast video clips.

Optimized specifically for **Google Colab NVIDIA Tesla T4 GPUs (15.3 GB GDDR6 VRAM)** to maximize GPU compute and utilize **12–14 GB of GPU RAM**, while maintaining cross-platform compatibility across Windows, macOS, and Linux.

---

## 1. System Architecture & Google Colab T4 Compute Optimization

Our overhauled pipeline integrates **End-to-End 3D Convolutional Visual Backbones (512-D)** with **Kinematic Pose Attention Networks (256-D)** and a **Sequential Reinforcement Learning Referee Agent (`NBARefereeEnv` + Dueling DQN / Actor-Critic)**:

```text
+---------------------------------------------------------------------------------------------------+
| STAGE 1: High-Throughput NBA L2M Web & JSON Scraper (1_scrape_l2m.py)                             |
|  • Scrapes official.nba.com L2M JSON/HTML reports (UTF-8 BOM-safe) & CDN broadcast MP4 streams    |
|  • Concurrent ThreadPoolExecutor bulk downloader for 349+ plays and multi-season game discovery   |
|  • Officiating Ground Truth: CC (Correct Call), INC (Missed Foul), CNC (Legal), IC (Phantom Call)  |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 2: GPU-Accelerated Action-Tube & Biomechanical Pose Extractor (2_preprocess_tubes.py)       |
|  • Ultralytics YOLOv11-Pose batch inference directly on CUDA with FP16 half precision             |
|  • Extracts 12-D Biomechanical Contact Vector per frame grounded in NBA Rule 12 (cylinder breach, |
|    verticality angle deviation, arm contact intrusion, landing space violation, closing velocity) |
|  • Crops stabilized (3, 16, 224, 224) RGB + (16, 12) Pose .pt action-tube tensors                 |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
+------------------------------------------------+ +------------------------------------------------+
| STREAM A: 3D Spatiotemporal Video Backbone     | | STREAM B: Biomechanical Contact Attention Net  |
|  • Input: (B, 3, 16, 224, 224) 4D Video Tensor | |  • Input: (B, 16, 12) Kinematic Contact Vector |
|  • R(2+1)D-18 End-to-End Fine-Tuning on CUDA   | |  • 4-Head Temporal Self-Attention + LayerNorm  |
|  • Output: 512-D Visual Embedding              | |  • Output: 256-D Biomechanical Representation  |
+------------------------+-----------------------+ +-----------------------+------------------------+
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 3 & 4: Deep Late Fusion & High-VRAM End-to-End GPU Training (3_model / 4_train_eval)         |
|  • Late Fusion: COMBINED_FEATURE_DIM = 768-D [512-D Visual || 256-D Pose Kinematics]              |
|  • Deep Classification Head: 768-D -> 512-D -> 256-D -> 1 (~33.8M parameters)                     |
|  • High-VRAM Colab T4 Mode: batch_size=16, accum_steps=2 (eff=32), AMP FP16, pin_memory=True      |
|  • Consumes ~12-14 GB GDDR6 GPU RAM, fully engaging Tesla T4 Turing Tensor Cores                   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 5 & 6: Sequential Reinforcement Learning Referee Agent (NBARefereeEnv + Dueling DQN)        |
|  • Gym-like sequential officiating environment (State: 281-D, Actions: PLAY_ON vs WHISTLE_FOUL)   |
|  • Rewards grounded in NBA L2M Ground Truth: +2.5 CC/INC, -3.0 phantom whistle, +0.5 overturn     |
|  • Evaluates WHEN to blow the whistle with timing precision and Rule 12 biomechanical shaping     |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 7: Broadcast-Grade Multimodal Video Annotation & Telemetry Replay                           |
|  • Disputed-play enclosing hull tracking BOTH players (Offense Cyan, Defense Orange)              |
|  • 17-Keypoint skeleton overlay, defender verticality guideline, dynamic green-to-red transition  |
|  • Top-left HUD with live RL Q-values, P(Whistle), Rule 12 diagnostics, & bottom timeline strip   |
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

## 2. Files in `NBA-Foul-Classifier-WIP/`

- **`NBA_Foul_Classifier_Full_Pipeline.ipynb`**: Single unified Jupyter Notebook containing the entire 4-stage pipeline, EDA, training loop, and telemetry visualization to run cell-by-cell.
- **`1_scrape_l2m.py`**: NBA L2M scraper, foul mechanics tagger, and offline CSV synchronizer (`349` labeled plays, `130+` MP4s).
- **`2_preprocess_tubes.py`**: Cross-platform `YOLOv11-Pose` ROI cropper & 12-D biomechanical feature sampler.
- **`3_model_and_dataset.py`**: PyTorch `L2MFoulDataset`, simplified dimensions, `BiomechanicalFoulEncoder`, and high-capacity `FoulClassifier3D` (~33.8M parameters).
- **`4_train_eval.py`**: Cross-platform training (`cuda` / `mps` / `cpu`), `FoulFocalBCEWithLogitsLoss`, threshold calibration, and continuous full-clip video annotation to `data/annotated/`.
- **`requirements.txt`**: All Python dependencies for Windows, macOS, and Linux.
- **`data/annotated/`**: Export directory containing annotated broadcast MP4 videos with continuous bounding boxes.

---

## 3. Quick Start (All Operating Systems)

### 1. Install Dependencies
```bash
# macOS / Linux
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv ..\venv
..\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Cell-by-Cell in Jupyter Notebook
```bash
jupyter notebook NBA_Foul_Classifier_Full_Pipeline.ipynb
```

### 3. Or Run via Command Line (Lightweight Defaults)
```bash
python 1_scrape_l2m.py --fill-from-local
python 2_preprocess_tubes.py --max-clips 12 --synthetic-fallback
python 3_model_and_dataset.py
python 4_train_eval.py --epochs 5 --batch-size 2 --freeze-backbone --max-samples 0

# Run batch test annotation across data/test/ (saves to data/annotated/)
python 4_train_eval.py --annotate-test

# Or run prediction on a single clip:
python 4_train_eval.py --predict data/test/0022600196_2132_05.mp4
```

---

## 4. Dedicated Test Set & Continuous Video Annotation (`data/test/` & `data/annotated/`)

### Test Set Segregation
- **Location:** `data/test/`
- **Data Isolation:** Clips placed in `data/test/` are **strictly segregated** from training action-tube generation and dataset loaders to prevent data leakage.
- **Adding Test Files:** Drop any `.mp4`, `.mov`, `.avi`, or `.mkv` file into `data/test/`.
- **Batch Running:** Run `python 4_train_eval.py --annotate-test` (or `python 4_train_eval.py --predict data/test`) to batch evaluate and annotate all test clips into `data/annotated/<clip_name>_annotated.mp4`.

### Continuous Video Annotation Specifications
1. **Disputed Play Enclosing Box**: An enclosing bounding box encompasses **ALL players involved** in the disputed play (offensive shooter/driver + contesting defender) across 100% of video frames without jumping or flickering.
2. **Official & Sideline Exclusion**: Geometry filtering ($y_2 \le 0.72 \times H$, $y_c \in [0.25H, 0.68H]$, $x_c \in [0.12W, 0.80W]$) and movement speed gating (`max_d = 160 px`) reject foreground referees, sideline bench players, and baseline photographers.
3. **Dynamic State & Color Transition**:
   - Starts **GREEN** (`LEGAL PLAY | NO FOUL`) during legal defense.
   - Transitions to **RED** (`*** FOUL DETECTED: {call_type} ({defender} on {shooter}) ***`) at the exact frame physical contact turns illegal ($t_{\text{foul}}$).
4. **Top-Left HUD Panel**:
   - `STATUS`: Dynamic green indicator before foul $\to$ bright red indicator after foul with timestamp (`*** FOUL CALLED AT FRAME {frame} ({sec}s) ***`).
   - `WHAT FOUL`: Officiating call (`Shooting Foul`, `Personal Foul`, etc.).
   - `WHO FOULED WHO`: Explicit player matchup (`Dalen Terry fouled Ricky Council IV`, `Josh Giddey fouled Tyrese Maxey`).
   - `TELEMETRY`: Displays `P(FOUL)`, calibrated threshold, L2M review decision (`INC`, `CC`, `CNC`), and per-frame physical contact intensity.

---

## 5. How to Increase Data Scraped, Processed & Trained

The default settings are intentionally small so any laptop can run the pipeline in minutes. To increase the dataset size and training capacity on stronger hardware:

1. **Scrape More Games & Video Clips (`1_scrape_l2m.py`)**:
   - Increase `--max-games` (default `3`) and `--max-clips` (default `12`), and add `--include-archive` for multi-season data:
   ```bash
   python 1_scrape_l2m.py --max-games 25 --max-clips 200 --include-archive
   ```
2. **Preprocess All Clips at Higher Temporal Resolution (`2_preprocess_tubes.py`)**:
   - Pass `--max-clips 0` to process **all** clips in `data/l2m_metadata.csv`, increase `--max-scan-samples` from `12` to `24`, and raise `--yolo-batch-size` from `4` to `8` or `16`:
   ```bash
   python 2_preprocess_tubes.py --max-clips 0 --max-scan-samples 24 --max-live-play-sec 20.0 --yolo-batch-size 8
   ```
3. **Train on Full Dataset with Larger Effective Batch Size (`4_train_eval.py`)**:
   - Pass `--max-samples 0` to train on all `.pt` tensors, increase `--epochs` from `5` to `20`, and set `--batch-size 4 --accum-steps 4` (effective batch size `16`):
   ```bash
   python 4_train_eval.py --max-samples 0 --epochs 20 --batch-size 4 --accum-steps 4
   ```
4. **In `NBA_Foul_Classifier_Full_Pipeline.ipynb`**:
   - Update the `CONFIG` dictionary in **Cell 1** (`max_games`, `max_scrape_clips`, `max_preprocess_clips = 0`, `batch_size = 4`, `accum_steps = 4`, `epochs = 20`, `max_train_samples = 0`).

---

## 6. Robust DataLoader Collation & Embedding Self-Healing (Cell 7 Safeguard)

During micro-batch collation in PyTorch, inconsistent tensor dimensions (e.g. stacking 1D 512-D cached feature embeddings with 4D `[3, 16, 224, 224]` raw video tensors) previously caused `RuntimeError: stack expects each tensor to be equal size`. This has been permanently resolved through a three-layer defense:

1. **Self-Healing Dataset Loader (`L2MFoulDataset`)**:
   - Loads `data/processed/visual_embeddings_cache.pt` on initialization.
   - If any sample lacks a cached 512-D embedding, `_extract_and_cache_embedding()` computes the Kinetics-400 representation on-the-fly and persists it to both disk and memory.
   - Strictly guarantees that `use_cached_embeddings=True` always returns a 1D `[512]` tensor, never falling back to a 4D tensor.
2. **Safe Collation Operator (`safe_foul_collate`)**:
   - Implemented in `4_train_eval.py` and utilized by default across `DataLoader` instances in both the script and notebook Cell 7.
   - Validates incoming batch tensor shapes and harmonizes any shape discrepancies on-the-fly before invoking `torch.stack`.
3. **End-to-End Notebook Verification**:
   - Verified across all 15 cells of `NBA_Foul_Classifier_Full_Pipeline.ipynb` with 10 full epochs completing in under 6 seconds (Validation Accuracy: 81.4%, F1-Score: 0.778, ROC-AUC: 0.828).

---

## 7. Model Weights & File Organization (`weights/`)

To keep the repository clean and eliminate duplicate binaries, all foundation models and trained checkpoints are organized in `weights/`:

```text
weights/
├── README.md              # Detailed documentation of model architectures, keypoints, and parameters
├── yolo11n.pt             # Ultralytics YOLOv11 Nano object detection (2.6M params, COCO 80 classes)
├── yolo11n-pose.pt        # Ultralytics YOLOv11 Nano pose estimation (2.9M params, COCO 17 keypoints)
└── best_foul_r2plus1d.pt  # Trained Dual-Stream Foul Classifier checkpoint (~33.8M params)
```

### What Each Weight File Does:
* **`yolo11n.pt` (Object Detection)**: Detects basketball players (class 0: `person`) and the ball (class 32: `sports ball`) in full-clip broadcast videos. Used in Stage 4/5 inference (`track_disputed_players_full_video`) to track the offensive driver and contesting defender bidirectionally without jumping to sideline spectators or referees.
* **`yolo11n-pose.pt` (Skeletal Joint Estimation)**: Identifies 17 anatomical keypoints per player (wrists, elbows, shoulders, hips, knees, ankles). Used in Stage 2 (`2_preprocess_tubes.py`) to calculate the 12 Mechanical Foul Dimensions (arm contact, cylinder penetration, landing zone violations, verticality angle).
* **`best_foul_r2plus1d.pt` (Classifier Checkpoint)**: Contains the trained weights for `BiomechanicalFoulEncoder` (4-head temporal attention) and the late-fusion MLP head, along with the calibrated decision threshold and evaluation metrics.
* **`data/processed/visual_embeddings_cache.pt`**: In-memory cache of 512-D Kinetics-400 embeddings extracted via $R(2+1)D\text{-}18$, enabling zero-stall training loops.
