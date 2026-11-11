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

---

## 2. Files in `NBA-Foul-Classifier-WIP/`

- **`NBA_Foul_Classifier_Full_Pipeline.ipynb`**: Single unified Jupyter Notebook containing the entire 4-stage pipeline, EDA, training loop, and telemetry visualization to run cell-by-cell.
- **`1_scrape_l2m.py`**: NBA L2M scraper, foul mechanics tagger, and offline CSV synchronizer (`--max-games 3 --max-clips 12` default).
- **`2_preprocess_tubes.py`**: Cross-platform `YOLOv11-Pose` ROI cropper & 12-D biomechanical feature sampler (`--max-clips 12 --yolo-batch-size 4 --max-scan-samples 12` default).
- **`3_model_and_dataset.py`**: PyTorch `L2MFoulDataset`, `ConsistentVideoAugmentor`, `BiomechanicalFoulEncoder`, and `FoulClassifier3D` (`R(2+1)D-18` fusion).
- **`4_train_eval.py`**: Cross-platform training (`cuda` / `mps` / `cpu`), `FoulFocalBCEWithLogitsLoss`, threshold calibration, and `--predict` video overlay export (`--batch-size 2 --accum-steps 2 --epochs 5` default).
- **`requirements.txt`**: All Python dependencies for Windows, macOS, and Linux.

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
python 4_train_eval.py --epochs 5 --batch-size 2 --accum-steps 2
```

---

## 4. How to Increase Data Scraped, Processed & Trained

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
