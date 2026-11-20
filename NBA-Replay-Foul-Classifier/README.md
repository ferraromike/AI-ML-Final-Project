# NBA Replay Archive Binary Foul Classifier (`Foul` vs. `No Foul`)

A clean, standalone Computer Vision & Spatiotemporal Deep Learning pipeline built from scratch for official NBA Replay Center videos scraped from **[official.nba.com/replay/archive](https://official.nba.com/replay/archive/)**.

Unlike the legacy L2M pipeline, this system is built strictly for **binary classification (`FOUL` vs. `NO FOUL`)** on multi-angle, 60 fps Replay Center footage — with zero L2M dependencies, zero foul-subtype taxonomy heuristics, and no RL referee subsystem.

---

## 1. Architecture Overview

```text
+-------------------------------------------------------------------------------------------------+
| STAGE 1: NBA Replay Archive Scraper (`1_scrape_replays.py`)                                     |
|  • Queries /wp-json/api/v1/query_replay_feed across seasons (2019-20 .. 2026-27)                |
|  • Parses verbatim Replay Center rulings on each archive permalink page                         |
|  • Binary Ground Truth:                                                                         |
|      - Class 1 (foul):    Challenge of Called Foul upheld/stood OR overturned to another foul   |
|      - Class 0 (no_foul): Challenge of Called Foul explicitly overturned to NO FOUL             |
|  • Downloads balanced 1280x720 (or 320x180 SD) MP4 clips into data/raw/{foul,no_foul}/          |
+------------------------------------------------+------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
| STAGE 2: Shot-Segmented Multi-Angle Action-Tube Extractor (`2_preprocess_replays.py`)           |
|  • Scans full replay clip at constant 5 Hz (handles 60 fps and 30 fps identically)              |
|  • Detects hard camera cuts (wide broadcast, slow-mo, overhead/baseline) & isolates each shot   |
|  • Adapts contact window duration for slow-motion replay shots & normalizes velocity by dt      |
|  • Runs YOLOv11-Pose + YOLOv11 detection to extract up to K camera-angle action tubes per play  |
|  • Saves (3, 16, 224, 224) RGB + (16, 12) biomechanical pose tensors -> data/processed/         |
+------------------------------------------------+------------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
+----------------------------------------------+   +----------------------------------------------+
| STREAM A: 3D Spatiotemporal Visual Backbone  |   | STREAM B: Pose Temporal Attention Encoder    |
|  • Pretrained R(2+1)D-18 (Kinetics-400)      |   |  • Input: (B, 16, 12) biomechanical vector   |
|  • GPU: partial fine-tuning (layer3+layer4)  |   |  • Learned positional embedding + 4-head     |
|  • CPU/MPS: frozen + cached 512-D embeddings |   |    self-attention + attn & max pooling       |
|  • Output: 512-D visual embedding            |   |  • Output: 256-D kinematic embedding         |
+----------------------+-----------------------+   +-----------------------+----------------------+
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
| STAGE 3 & 4: Binary Late Fusion, Leak-Free Training & Inference (`3_model` / `4_train_eval`)    |
|  • Fusion MLP: 768-D -> 384-D -> 128-D -> 1 binary foul logit (with modality dropout)           |
|  • Game-grouped Train/Val split (`GroupShuffleSplit` on `game_id`) -> 0 cross-angle leakage     |
|  • Multi-angle play-level evaluation + threshold sweep [0.20..0.80] + Confusion Matrix          |
|  • Inference on single clips or data/test/ -> FOUL / NO FOUL verdicts + clean annotated MP4s    |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Folder Structure

```text
NBA-Replay-Foul-Classifier/
├── README.md                          # Project guide & CLI scaling reference
├── requirements.txt                   # Cross-platform Python dependencies
├── 1_scrape_replays.py                # Stage 1: Replay Archive API + ruling parser + MP4 downloader
├── 2_preprocess_replays.py            # Stage 2: Shot segmentation, 60fps/slow-mo normalization & multi-angle crops
├── 3_model_and_dataset.py             # Stage 3: ReplayFoulDataset & BinaryFoulClassifier
├── 4_train_eval.py                    # Stage 4: Grouped training, play-level calibration & binary inference
├── NBA_Replay_Foul_Classifier.ipynb   # Unified Jupyter / Google Colab T4 notebook
├── data/
│   ├── README.md                      # Dataset & metadata schema documentation
│   └── test/README.md                 # Dedicated isolated test folder instructions
└── weights/
    └── README.md                      # YOLO weights & trained binary checkpoint docs
```

---

## 3. Setup

### Local (macOS / Linux / Windows)
```bash
python3 -m venv venv
source venv/bin/activate        # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r NBA-Replay-Foul-Classifier/requirements.txt
```

### Google Colab (Recommended for GPU Training)
Open `NBA-Replay-Foul-Classifier/NBA_Replay_Foul_Classifier.ipynb` in Google Colab with a **T4 GPU** runtime. Cell 1 automatically clones the `testing` branch, installs dependencies, and configures `partial` 3D CNN fine-tuning (`layer3` + `layer4`) with FP16 AMP.

---

## 4. Running the Pipeline via CLI

### Quick Laptop Prototype (SD videos, frozen 3D backbone)
```bash
# 1. Scrape recent Coach's Challenges and download 40 balanced SD (320x180) clips
python NBA-Replay-Foul-Classifier/1_scrape_replays.py \
    --seasons 2026-27 2025-26 --max-inspect 200 --max-clips 40 --quality sd

# 2. Extract up to 2 shot-bounded camera-angle tubes per replay clip
python NBA-Replay-Foul-Classifier/2_preprocess_replays.py \
    --angles-per-clip 2 --max-clips 40

# 3. Train binary classifier with leak-free game-grouped split
python NBA-Replay-Foul-Classifier/4_train_eval.py \
    --backbone-mode frozen --epochs 15 --batch-size 8

# 4. Run binary FOUL / NO FOUL prediction on clips placed in data/test/
python NBA-Replay-Foul-Classifier/4_train_eval.py --predict-test
```

### Full GPU / Google Colab Scale (HD videos, partial 3D fine-tuning)
```bash
# 1. Scrape multi-season foul challenges (hard negatives + confirmed/other fouls)
python NBA-Replay-Foul-Classifier/1_scrape_replays.py \
    --seasons 2026-27 2025-26 2024-25 2023-24 \
    --max-inspect 1200 --max-clips 400 --quality hd --workers 8

# 2. Preprocess all downloaded clips (2 distinct angles per clip -> ~800 tubes)
python NBA-Replay-Foul-Classifier/2_preprocess_replays.py \
    --angles-per-clip 2 --max-clips 0 --yolo-batch-size 64

# 3. Fine-tune layer3+layer4 end-to-end with FP16 AMP & differential learning rates
python NBA-Replay-Foul-Classifier/4_train_eval.py \
    --backbone-mode partial --epochs 20 --batch-size 8 --accum-steps 2 --lr 2e-4

# 4. Predict on a single clip or directory
python NBA-Replay-Foul-Classifier/4_train_eval.py \
    --predict NBA-Replay-Foul-Classifier/data/test/
```

---

## 5. Useful Scraper & Inference Flags

| Stage | Flag | Behavior |
| :--- | :--- | :--- |
| **Stage 1** | `--include-flagrant-reviews` | Also add Flagrant / Clear-Path / Off-Ball called-foul reviews as Class 1 (`foul`) |
| **Stage 1** | `--include-non-foul-challenges` | Also include OOB & Goaltending challenges (clean -> `no_foul`, proximate foul -> `foul`) |
| **Stage 1** | `--quality hd \| sd` | Download `1280x720` (default) or lightweight `320x180` replay MP4s |
| **Stage 1** | `--skip-download` / `--download-missing` | Build `replay_metadata.csv` first or resume missing MP4 downloads |
| **Stage 2** | `--angles-per-clip K` | Extract up to `K` distinct camera-angle tubes per replay video (`1`..`3`, default `2`) |
| **Stage 4** | `--backbone-mode auto \| frozen \| partial \| full` | Control 3D CNN fine-tuning depth (`auto` uses `partial` on >=10 GB CUDA GPUs) |
| **Stage 4** | `--no-render` | Skip MP4 annotation rendering and output fast `predictions.json` verdicts only |
| **Stage 4** | `--clean-no-foul` | Leave predicted `NO FOUL` clips completely untouched without any overlay |
