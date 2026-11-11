# NBA Foul Detection & Prediction Pipeline (PyTorch `R(2+1)D-18` + `YOLOv11-Pose`)

A modular 4-stage Computer Vision and Spatiotemporal Deep Learning pipeline engineered to **detect and predict basketball fouls** from NBA Last Two Minute (L2M) broadcast video clips using **Ultralytics YOLOv11-Pose** (`yolo11n-pose.pt`), **OpenCV**, and **Torchvision 3D CNNs** (`r2plus1d_18`) fused with **12-D Biomechanical Foul-Contact Features**.

---

## Repository Structure

```text
AI-ML-Final-Project/
├── 1_scrape_l2m.py          # Step 1: NBA L2M scraper, foul mechanics tagger & CSV sync
├── 2_preprocess_tubes.py    # Step 2: M1-safe YOLOv11-Pose foul-contact ROI cropper & 16-frame sampler
├── 3_model_and_dataset.py   # Step 3: PyTorch L2MFoulDataset & R(2+1)D-18 + BiomechanicalFoulEncoder
├── 4_train_eval.py          # Step 4: Apple Silicon (MPS) / CUDA training, threshold calibration & CLI inference
├── requirements.txt         # Unified dependencies (PyTorch, Torchvision, Ultralytics, OpenCV, Discord.py)
├── data/
│   ├── l2m_metadata.csv         # Master training metadata (play_id, game_id, players, call_type, label, video_path)
│   ├── l2m_incorrect_fouls.csv  # Populated foul & officiating review table (INC, IC, and confirmed fouls)
│   ├── raw/                     # Downloaded broadcast MP4 clips ({foul,no_foul}/)
│   └── processed/               # Pre-cropped (3, 16, 224, 224) + (16, 12) .pt tensors & processed_metadata.csv
└── discord-bot/             # Discord bot module (untouched)
```

---

## Setup (Apple Silicon M1/M2/M3 & CUDA)

Activate the consolidated virtual environment at the repository root:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Execution Pipeline

### Step 1: Scrape NBA L2M Reports & Populate Foul CSVs (`1_scrape_l2m.py`)
Extracts foul assessments (`Shooting Foul`, `Personal Foul`, and optional `Loose Ball` / `Offensive` fouls), maps `CC` & `INC` $\rightarrow$ `1` (`foul`) and `CNC` & `IC` $\rightarrow$ `0` (`no_foul`), downloads MP4 clips into `data/raw/{foul,no_foul}/`, and updates both `data/l2m_metadata.csv` and `data/l2m_incorrect_fouls.csv`:

```bash
./venv/bin/python3 1_scrape_l2m.py --max-games 25 --max-clips 200
```

To populate `data/l2m_incorrect_fouls.csv` immediately from already-scraped local metadata (`data/l2m_metadata.csv`):

```bash
./venv/bin/python3 1_scrape_l2m.py --fill-from-local
```

### Step 2: M1-Safe Foul-Contact Action-Tube Cropping (`2_preprocess_tubes.py`)
Caps live-play scanning to the first 20 seconds of each clip (preventing multi-minute replay review clips like `0022600186_2098_17.mp4` [12,318 frames / 238 MB] from exhausting RAM), locates peak defender-on-shooter upper-body contact via `yolo11n-pose.pt` 17-keypoint skeletons, and saves stabilized `(3, 16, 224, 224)` RGB tubes + `(16, 12)` biomechanical contact features to `data/processed/`:

```bash
./venv/bin/python3 2_preprocess_tubes.py
```

### Step 3: Verify 3D CNN + Biomechanical Foul-Contact Fusion Architecture (`3_model_and_dataset.py`)
Tests `L2MFoulDataset` and `FoulClassifier3D` (`r2plus1d_18` with `stem`, `layer1`, and `layer2` frozen + `BiomechanicalFoulEncoder` over the 12-D skeletal contact features):

```bash
./venv/bin/python3 3_model_and_dataset.py
```

### Step 4: Train, Calibrate Foul Threshold & Predict on Video Clips (`4_train_eval.py`)
Trains with `FoulFocalBCEWithLogitsLoss`, gradient accumulation (`batch_size=4`, `accum_steps=4`), and per-epoch validation threshold calibration (`best_foul_r2plus1d.pt`):

```bash
# Train & evaluate (auto-detects Apple Silicon MPS, CUDA, or CPU)
./venv/bin/python3 4_train_eval.py --epochs 20 --batch-size 4 --accum-steps 4

# Run single-clip foul prediction & generate an annotated MP4 overlay
./venv/bin/python3 4_train_eval.py --predict data/raw/foul/0022600186_2115_34.mp4
```
