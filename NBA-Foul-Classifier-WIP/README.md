# NBA AI Referee Pipeline (`NBA-Foul-Classifier-WIP/`)

Full-stack **v3.0** Computer Vision, Video Transformer Multiple Instance Learning (MIL), and Multi-Window Reinforcement Learning pipeline acting as an autonomous **NBA AI Referee** on any raw basketball broadcast footage.

---

## 1. Architecture Overview (v3.0)

1. **Stage 1 — L2M Data Scraper & Conflict Deduplicator (`1_scrape_l2m.py`)**:
   - Scrapes official NBA Last Two Minute (L2M) reports (`CC`/`INC` = Foul, `CNC`/`IC` = Legal/No Foul) with real `call_type` (`Shooting Foul` vs. `Personal Foul`) and `game_id`.
   - `deduplicate_raw_dataset()` computes content fingerprints across all raw `.mp4` files to eliminate duplicate NBA Stats video assets and resolve conflicting-label duplicates before training.
2. **Stage 2 — E-BARD Basketball Perception, 2-Team CIELAB Clustering & Swin3D-T Multi-Window MIL Bags (`2_preprocess_tubes.py`)**:
   - **E-BARD NBA Basketball Detector (`weights/ebard_basketball_yolo.pt`)**: Auto-downloaded from HuggingFace (`GabrieleGiudici/E-BARD-detection-models`, CC-BY-4.0). Natively separates **`basketball` (0)**, **`hoop` (1)**, on-court **`player` (2)**, and **`referee` (3)**—eliminating referee and crowd false detections.
   - **Secondary Long-Trousers Official Guard + Per-Clip 2-Team CIELAB K-Means Clustering**: Clusters all detected on-court players in CIELAB torso space into `Team 0` vs. `Team 1`, enforcing opposing-team pairing (`team_off != team_def`) and rejecting dark-trousers/gray-shirt official outliers.
   - **16-D Depth-Gated Rule 12 + Ball/Hoop Kinematics**: Fuses clean E-BARD `player` boxes with `yolo11n-pose.pt` 17-point skeletons and gates 2D overlap by 3D foot-plane depth alignment (`|y2_a - y2_b|`).
   - **Multi-Window Sliding Bags**: Slices contiguous wide-court live play (filtering close-up cuts and replays) into overlapping `2.0s` windows (`K <= 10` covering the full play with zero blind spots) and encodes every window with frozen Kinetics-400 **`Swin3D-T`** (`768-D`).
3. **Stage 3 & 4 — `FoulMILClassifier` & Leak-Free Game-Grouped Training (`3_model_and_dataset.py` & `4_train_eval.py`)**:
   - **Dual-Stream Window Fusion**: Projects `768-D Swin3D-T` visual features + `192-D 4-Head Temporal Pose Attention` (`16 x 16`) into `256-D` per window (~439K trainable parameters over frozen 27.8M `Swin3D-T`).
   - **Gated Attention MIL (Ilse et al., ICML 2018)** + **Instance-Level Negative Suppression**: Forces *every* window of `no_foul` clips toward `0` while focusing attention on the peak foul window of `foul` clips to learn **WHEN** fouls occur from clip-level L2M labels.
   - **Real Ground-Truth `call_type` Head**: Trained strictly on official L2M `Shooting Foul` vs. `Personal Foul` ground truth on positive fouls (zero synthetic heuristic targets).
   - **Leak-Free 5-Fold `StratifiedGroupKFold` Ensemble (`FoulMILEnsemble`) by `game_id`**: Plays from the same NBA game never cross train and validation splits; all 5 fold models are ensembled into `best_foul_r2plus1d.pt`.
   - **Global Out-of-Fold (OOF) Temperature Scaling & Honest Calibration**: Fits one global temperature $T$ via OOF validation NLL minimization and sweeps one global threshold $\tau$ across all unseen-game clips without hardcoded probability overrides.
4. **Stage 5 — Multi-Window Sequential RL Referee Agent (`NBARefereeEnv` + Double Dueling DQN)**:
   - Operates over distinct per-window `Swin3D-T` features, calibrated per-window MIL probabilities, temporal deltas, and 16-D kinematics using real L2M review rewards (`CC`, `INC`, `CNC`, `IC`).
5. **Stage 6 — Arbitrary-Length Sliding-Window Inference & Zero-Referee Annotation (`data/test/` $\to$ `data/annotated/`)**:
   - Evaluates **every** `2.0s` sliding window across raw NBA footage of **any duration** via Temporal NMS (`min_sep=3.5s`) with **zero CSV metadata access**.
   - **Foul Detected**: Highlights **WHAT foul it is**, **WHERE it occurred** (opposing-team boxes only—zero referees!—red foul zone, contact reticle `(x, y)`, court zone), and **WHEN it occurred** strictly during each foul window.
   - **No Foul Detected**: **Nothing happens** on the video (100% clean unaltered playback).

---

## 2. Files in `NBA-Foul-Classifier-WIP/`

- **`NBA_Foul_Classifier_Full_Pipeline.ipynb`**: Complete interactive 10-stage notebook (Google Colab T4 GPU, Apple Silicon MPS, Windows/Linux CPU).
- **`1_scrape_l2m.py`**: Stage 1 L2M scraper, parallel downloader, and content-hash deduplicator (`--deduplicate`).
- **`2_preprocess_tubes.py`**: Stage 2 E-BARD Basketball YOLO perception, 2-team CIELAB clustering, 16-D depth-gated kinematics, and Swin3D-T multi-window MIL bag extractor.
- **`3_model_and_dataset.py`**: Stage 3 `L2MFoulDataset`, `game_grouped_train_val_split`, `FoulMILClassifier` (alias `FoulClassifier3D`), and Multi-Window `NBARefereeEnv` / `RefereeRLAgent`.
- **`4_train_eval.py`**: Stage 4–6 game-grouped CV training, temperature calibration, zero-referee bidirectional window tracker, sliding-window any-length inference, and folder batch annotation (`--annotate-test`).
- **`requirements.txt`**: Cross-platform dependencies (`torch`, `torchvision`, `ultralytics`, `opencv-python`, `scikit-learn`, `scipy`, `matplotlib`).

---

## 3. Quick Start

### Install Dependencies
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

### Run via Interactive Notebook
```bash
jupyter notebook NBA_Foul_Classifier_Full_Pipeline.ipynb
```

### Run via Command Line
```bash
# 1. Deduplicate raw clips & metadata (or scrape more games with --max-games 60 --max-clips 250 --include-archive)
python 1_scrape_l2m.py --deduplicate

# 2. Extract v3 E-BARD + 2-Team CIELAB + Swin3D-T Multi-Window MIL bags
python 2_preprocess_tubes.py --max-clips 0

# 3. Verify FoulMILClassifier shapes and parameter summary
python 3_model_and_dataset.py

# 4. Train with leak-free 5-fold game-grouped CV, temperature scaling & Multi-Window RL Referee
python 4_train_eval.py --epochs 25 --cv-folds 5 --train-rl

# 5. Drop ANY raw NBA footage into data/test/ and run batch AI Referee annotation -> data/annotated/
python 4_train_eval.py --annotate-test

# Or evaluate a single raw video clip of any length:
python 4_train_eval.py --predict data/test/0022600196_2132_05.mp4
```

---

## 4. Easy Drop-Folder Testing (`data/test/` $\to$ `data/annotated/`)

1. Place **any raw NBA broadcast video** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) into `data/test/`.
2. Clips in `data/test/` are strictly isolated from training/preprocessing by **both filename stem AND video content hash**.
3. Run **Cell 9 & Cell 10** in `NBA_Foul_Classifier_Full_Pipeline.ipynb` (or `python 4_train_eval.py --annotate-test`).
4. Every video in `data/test/` is annotated autonomously into `data/annotated/<clip_stem>_annotated.mp4` and logged into `data/annotated/referee_decisions.json`.
