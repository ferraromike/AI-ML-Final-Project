# CSC 434 AI/ML Final Project Memory — Fred

## 1. Project Overview

- **AI Teammate:** Fred (collaborative 3rd engineering partner for Muamer and Michael/Mike).
- **Project Title:** NBA Foul Detection & Prediction Pipeline (`YOLOv11-Pose` + `R(2+1)D-18` Dual-Stream Fusion).
- **Goal:** Detect and classify NBA Shooting and Personal Fouls (`1 = foul` [CC, INC] vs. `0 = no_foul` [CNC, IC]) from NBA Last Two Minute (L2M) broadcast video clips across all operating systems (Windows, macOS Intel/Apple Silicon, Linux CUDA/CPU).

## 2. Architecture & Specifications

- **Stage 1 (`1_scrape_l2m.py`)**: Scrapes official NBA L2M JSON/HTML reports (handling UTF-8 BOM headers) and referee CDN `.mp4` streams; populates `data/l2m_metadata.csv` (200 plays across 13 games: 157 `no_foul`, 43 `foul`) and `data/l2m_incorrect_fouls.csv` (43 foul reviews).
- **Stage 2 (`2_preprocess_tubes.py`)**: Cross-platform two-pass extractor. Pass 1 scans the first `15.0s` of live broadcast action (`12` downscaled `640x360` frames, `< 8 MB` RAM) using `yolo11n-pose.pt` in micro-batches (`--yolo-batch-size 4`) after releasing `cv2.VideoCapture` to prevent FFmpeg/PyTorch deadlocks. Extracts a **12-D Biomechanical Contact Vector** per frame and crops stabilized `(3, 16, 224, 224)` RGB action tubes around peak contact.
- **Stage 3 (`3_model_and_dataset.py`)**: Dual-stream `FoulClassifier3D` combining pretrained `r2plus1d_18` (`stem`, `layer1`, `layer2` frozen; `layer3`, `layer4` fine-tuned $\rightarrow$ `512-D`) with `BiomechanicalFoulEncoder` (`Conv1d` + temporal attention pooling $\rightarrow$ `64-D`) into a `576-D` late-fusion classification head.
- **Stage 4 (`4_train_eval.py`)**: Trains with `FoulFocalBCEWithLogitsLoss` (`gamma=1.5`, class-balanced `pos_weight=3.65`), lightweight micro-batch gradient accumulation (`batch_size=2`, `accum_steps=2`), and per-epoch validation F1 threshold calibration (`[0.25..0.75]`).

## 3. Codebase Structure

- **`NBA-Foul-Classifier-WIP/`**:
  - `NBA_Foul_Classifier_Full_Pipeline.ipynb`: Unified cell-by-cell interactive notebook.
  - `1_scrape_l2m.py`, `2_preprocess_tubes.py`, `3_model_and_dataset.py`, `4_train_eval.py`, `requirements.txt`.
- **`NBA_Foul_Classifier_Presentation.pptx`**: 12-slide presentation deck & teaching guide.
- **`discord-bot/`**: Fred Discord bot (`bot.py`, `PROJECT_MEMORY.md`).

## 4. Roadmap & Milestones

- **Phase 1 (Completed)**: L2M Scraper, UTF-8 BOM fix, & 200-play ground-truth dataset curation.
- **Phase 2 (Completed)**: Two-pass low-RAM `YOLOv11-Pose` ROI cropper & 12-D biomechanical feature extraction.
- **Phase 3 (Completed)**: Dual-stream `R(2+1)D-18` + `BiomechanicalFoulEncoder` architecture & Focal Loss calibration.
- **Phase 4 (Completed)**: Cross-platform overhaul (Windows/macOS/Linux), lightweight default batch sizes, unified Jupyter Notebook, and presentation deck.

## 5. Active Decisions, Hyperparameters & Tasks Log

- Lowered default scraping & training batch sizes (`max_games=3`, `max_clips=12`, `batch_size=2`, `accum_steps=2`, `epochs=5`) so all student laptops can run the pipeline smoothly; documented full scaling flags in `README.md`. *(logged via System Overhaul, 2026-11-11)*
- Added cross-OS path resolution (`BASE_DIR`), POSIX CSV path normalization, OpenCV/OpenMP thread caps (`cv2.setNumThreads(0)`), and offline synthetic tensor fallback (`--synthetic-fallback`). *(logged via System Overhaul, 2026-11-11)*
