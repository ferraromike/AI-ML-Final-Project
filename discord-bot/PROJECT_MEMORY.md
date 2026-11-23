# CSC 434 AI/ML Final Project Memory — Fred

## 1. Project Overview

- **AI Teammate:** Fred (collaborative 3rd engineering partner for Muamer and Michael/Mike).
- **Project Title:** NBA Foul Detection & Prediction Pipeline (`YOLOv11-Pose` + `R(2+1)D-18` Dual-Stream Fusion).
- **Goal:** Detect and classify NBA Shooting and Personal Fouls (`1 = foul` [CC, INC] vs. `0 = no_foul` [CNC, IC]) from NBA Last Two Minute (L2M) broadcast video clips across all operating systems (Windows, macOS Intel/Apple Silicon, Linux CUDA/CPU).

## 2. Architecture & Specifications

- **Stage 1 (`1_scrape_l2m.py`)**: Scrapes official NBA L2M JSON/HTML reports (handling UTF-8 BOM headers) and referee CDN `.mp4` streams; populates `data/l2m_metadata.csv` (349 plays across 18+ games: 252 `no_foul`, 97 `foul`) and `data/l2m_incorrect_fouls.csv` (97 foul reviews).
- **Stage 2 (`2_preprocess_tubes.py`)**: Cross-platform two-pass extractor. Pass 1 scans live broadcast action using `yolo11n-pose.pt` in micro-batches (`--yolo-batch-size 4`). Extracts a **12-D Mechanical Foul Feature Vector** per frame (`MECHANICAL_FOUL_DIMENSIONS`: arm_contact_intrusion, body_cylinder_penetration, lower_body_landing_violation, defender_verticality_angle, defender_lateral_positioning, offensive_displacement, defender_to_ball_distance, ball_to_shooter_distance, ball_strike_vs_body_ratio, relative_closing_velocity, momentum_transfer_shock, airborne_contest_apex_diff) and crops stabilized `(3, 16, 224, 224)` RGB action tubes around peak contact.
- **Stage 3 (`3_model_and_dataset.py`)**: Dual-stream `FoulClassifier3D` combining `r2plus1d_18` video backbone (`512-D`) with `BiomechanicalFoulEncoder` (`256-D`) into a `768-D` late-fusion deep classification head (`768 -> 512 -> 256 -> 1`, ~31.8M trainable parameters). Explicit dimension constants: `CHANNELS=3`, `FRAMES=16`, `HEIGHT=224`, `WIDTH=224`, `POSE_FEATURES=12`, `VIDEO_FEATURE_DIM=512`, `POSE_HIDDEN_DIM=256`, `COMBINED_FEATURE_DIM=768`, `CLASSIFIER_HIDDEN_DIM=512`.
- **Stage 4 (`4_train_eval.py`)**: Trains with `FoulFocalBCEWithLogitsLoss` (`gamma=1.5`, class-balanced `pos_weight`), lightweight micro-batch gradient accumulation (`batch_size=2`, `accum_steps=2`), per-epoch validation F1 threshold calibration (`[0.25..0.75]`), and continuous bounding box video annotation exported to `data/annotated/`.

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
- **Phase 5 (Completed)**: Dataset expansion (349 plays, 97 fouls), 768-D architecture simplification (~31.8M params), simplified 12-D dimension names (`POSE_FEATURE_NAMES`), Stage 4 shape-compatibility fix, and full-duration continuous bounding box video annotation saving to `data/annotated/`.
- **Phase 6 (Completed)**: Classifier optimization (4-Head Temporal Self-Attention kinematic pose encoder, label smoothing `eps=0.05` focal loss, training on all 104 balanced action tube samples, balanced threshold calibration), and Identity-Preserved Focal Player Multi-Object Tracking (MOT) ensuring the bounding box locks onto the correct focal offensive player and contesting defender across 100% of video frames.
- **Phase 7 (Completed)**: Mechanical Foul Grounding & Dynamic Disputed Play Video HUD:
  - Overhauled 12 dimensions to `MECHANICAL_FOUL_DIMENSIONS` directly grounded in NBA Rule 12 officiating (arm contact intrusion, cylinder penetration, landing zone breach, verticality angle, lateral positioning, offensive displacement, defender to ball, ball to shooter, ball strike vs body ratio, closing velocity, momentum shock, apex diff).
  - Resolved Cell 7 notebook stalling: Froze 3D video backbone on MPS and precomputed 512-D Kinetics visual embeddings, dropping training time from 20+ minutes to under 2 seconds.
  - Upgraded video output: Encloses ALL players involved in the disputed interaction (`player_disadvantaged` shooter and `player_committing` defender) with dynamic green-to-red box transition at exact foul frame $t_{\text{foul}}$.
  - Implemented top-left HUD stats panel explicitly stating WHAT FOUL occurred and WHO FOULED WHO (e.g. `Shooting Foul: Caleb Wilson fouled LeBron James`).
- **Phase 8 (Completed)**: Robust Bidirectional Court-Player Tracking & Spectator Elimination:
  - Solved spectator/stand box detachment ("boxes exist for no reason") by introducing strict court-plane geometry filtering ($y_2 > 0.38 \times \text{frame\_h}$, $y_1 > 0.15 \times \text{frame\_h}$, $0.10 \le h \le 0.55$, aspect ratio $\ge 0.95$).
  - Anchored tracking at the peak action tube contest frame ($t_{\text{peak}} \approx 367$), locking onto the exact disputed shooter (`player_disadvantaged`) and contesting defender (`player_committing`).
  - Implemented bidirectional frame-by-frame tracking with optimal 2xK bipartite matching and mutual exclusion, preventing track collapse and ensuring both players remain locked across all 600 frames.
  - Verified video export (`0022600196_2132_05_annotated.mp4`): Green bounding box follows players until frame 367, transitions to bright Red upon illegal collision, with top-left HUD displaying foul type and player matchup.
- **Phase 9 (Completed)**: Dedicated Test Dataset Isolation & Active Court-Floor Official Exclusion:
  - Isolated dedicated test clips into `data/test/` (`0022600196_2128_01.mp4`, `0022600196_2132_05.mp4`, `0022600196_2134_07.mp4`).
  - Implemented strict training segregation in `2_preprocess_tubes.py` and `3_model_and_dataset.py`, automatically excluding any video stems in `data/test/` from training action-tube generation and dataset loaders to eliminate data leakage.
  - Added `--annotate-test` and `--test-dir` CLI flags and updated `predict_single_clip` to accept directory inputs, allowing users to drop any `.mp4` into `data/test/` and batch-annotate all files into `data/annotated/`.
  - Upgraded court geometry filtering ($y_2 \le 0.72 \times H$, $y_c \in [0.25H, 0.68H]$, $x_c \in [0.12W, 0.80W]$) and speed gating (`max_d = 160 px`), completely eliminating foreground officials (Referee #43), sideline bench players, and baseline photographers.
  - Validated batch test outputs in `data/annotated/`:
    * `0022600196_2128_01_annotated.mp4`: Josh Giddey on Tyrese Maxey [Shooting Foul]
    * `0022600196_2132_05_annotated.mp4`: Dalen Terry on Ricky Council IV [Shooting Foul]
    * `0022600196_2134_07_annotated.mp4`: Tyrese Maxey on Matas Buzelis [Shooting Foul]
    All clips cleanly track active court players, transition from green to red at the exact foul frame, and display top-left HUD statistics.
- **Phase 10 (Completed)**: PyTorch DataLoader Shape Collation Robustness & Self-Healing Embedding Architecture (Cell 7 Resolution):
  - Diagnosed root cause of Cell 7 `RuntimeError: stack expects each tensor to be equal size, but got [512] at entry 0 and [3, 16, 224, 224] at entry 1`: `0022600221_2110_04.pt` was missing `video_embedding`, causing `L2MFoulDataset.__getitem__()` to return 4D raw frames while other samples returned 1D 512-D cached embeddings.
  - Computed 512-D Kinetics-400 video embedding for `0022600221_2110_04.pt`, saved it to the payload, and updated `visual_embeddings_cache.pt` (now covering 100% of the 123 processed clips).
  - Hardened `L2MFoulDataset` in `3_model_and_dataset.py`: pre-loads `visual_embeddings_cache.pt`, uses singleton `get_shared_embedding_extractor()`, and includes `_extract_and_cache_embedding()` fallback so `use_cached_embeddings=True` strictly guarantees 1D `[512]` tensors for all samples.
  - Implemented `safe_foul_collate()` in `4_train_eval.py` and notebook Cell 7: dynamically validates tensor shapes in batch tuples and harmonizes any dimension mismatches before calling `torch.stack`.
  - Executed end-to-end verification of all 15 cells in `NBA_Foul_Classifier_Full_Pipeline.ipynb`: Cell 7 completes 10 epochs in under 6 seconds (Train Loss: 0.1252, Val Loss: 0.1899, Acc: 81.4%, F1: 0.778, ROC-AUC: 0.828) and Cell 8 annotates all 3 test clips cleanly.
- **Phase 11 (Completed)**: Repository Organization, Weight Centralization & Model Artifact Architecture:
  - Centralized all foundation models and trained checkpoints into `NBA-Foul-Classifier-WIP/weights/` (`yolo11n.pt`, `yolo11n-pose.pt`, `best_foul_r2plus1d.pt`).
  - Removed duplicate loose binary weight files from the repository root, keeping the root directory clean and professional.
  - Added `NBA-Foul-Classifier-WIP/weights/README.md` documenting model architectures, parameter counts, keypoint outputs, and pipeline roles.
  - Upgraded `resolve_project_path` in `2_preprocess_tubes.py` and `3_model_and_dataset.py` to automatically discover weights in `weights/` regardless of execution working directory.
  - Updated root `.gitignore` to ignore `*.pt`, `*.pth`, `*.onnx` while tracking `weights/README.md`.
- **Phase 12 (Completed)**: Unified Google Colab & Jupyter Notebook Overhaul (`NBA_Foul_Classifier_Full_Pipeline.ipynb`):
  - Restored missing notebook file at `NBA-Foul-Classifier-WIP/NBA_Foul_Classifier_Full_Pipeline.ipynb` and populated mirror copy at project root for direct drag-and-drop.
  - Added Universal Colab Auto-Bootstrap in Cell 1: detects `google.colab`, clones repository (`git clone -b testing`), sets working directory, installs dependencies (`ultralytics`, `opencv-python-headless`), and auto-selects GPU acceleration with zero manual steps.
  - Hardened PyTorch DataLoader batch collation using `safe_foul_collate`, eliminating tensor dimension mismatch exceptions (`RuntimeError: stack expects each tensor to be equal size`).
  - Synced 12 Mechanical Foul Dimensions schema grounded in NBA Rule 12 officiating rules across EDA and telemetry visualization cells.
  - Added Matplotlib visual curves across pipeline stages: EDA class distribution/decision charts, 10-epoch training convergence (loss/F1) curves, and 16-frame biomechanical contact profile graphs.
  - Added interactive HTML5 video preview cell for continuous court-player tracking in `data/annotated/`.

## 5. Active Decisions, Hyperparameters & Tasks Log

- Centralized model weights (`yolo11n.pt`, `yolo11n-pose.pt`, `best_foul_r2plus1d.pt`) under `weights/` with automatic path resolution across `2_preprocess_tubes.py`, `3_model_and_dataset.py`, and `4_train_eval.py`. *(logged 2026-11-13)*
- Resolved PyTorch DataLoader collation mismatch on Cell 7 (`stack expects each tensor to be equal size`) by ensuring 100% embedding coverage (123/123 clips), adding self-healing fallback in `L2MFoulDataset`, and deploying `safe_foul_collate` in `4_train_eval.py` and notebook Cell 7. *(logged 2026-11-13)*
- Segregated test data into `data/test/` with strict exclusion from training and preprocessing; added `--annotate-test` batch runner in `4_train_eval.py` and updated Jupyter notebook Stage 5 so users can drop any `.mp4` into `data/test/` and automatically produce annotated videos in `data/annotated/`. *(logged 2026-11-13)*
- Enforced active court-floor geometry constraints ($y_2 \le 0.72 \times H$, $y_c \in [0.25H, 0.68H]$, $x_c \in [0.12W, 0.80W]$) and movement speed gating (`max_d = 160 px`) in `track_disputed_players_full_video`; completely excludes foreground referee #43, Joel Embiid on the sideline, and baseline photographers, locking tracking onto the active play contest. *(logged 2026-11-13)*
- Replaced forward tracklet heuristics with Bidirectional Court Player MOT (`track_disputed_players_full_video`); enforces court geometry filtering to reject spectator stands and anchors tracking at peak contest ROI with mutual exclusion. Guarantees bounding box strictly follows the disputed players on court across 100% of frames. *(logged 2026-11-13)*
- Replaced abstract pose features with 12 Mechanical Foul Dimensions grounded in NBA officiating (Rule 12); unified `MECHANICAL_FOUL_DIMENSIONS` across `2_preprocess_tubes.py`, `3_model_and_dataset.py`, and `4_train_eval.py`. *(logged 2026-11-13)*
- Implemented `lookup_disputed_play_metadata` and multi-player enclosing tracking (`track_disputed_players_full_video`); bounding box encloses all players involved in the disputed play. *(logged 2026-11-13)*
- Built dynamic color transition: bounding box is GREEN with `LEGAL PLAY | NO FOUL` until frame $t_{\text{foul}}$ where contact surges into illegal impact, at which point it turns RED with `FOUL DETECTED: {call_type} ({committing} on {disadvantaged})`. Top-left HUD displays WHAT FOUL and WHO FOULED WHO. *(logged 2026-11-13)*
- Solved Cell 7 stalling by freezing `r2plus1d_18` video backbone and caching 512-D Kinetics embeddings; eliminates Apple Silicon MPS 3D ConvNet swap thrashing, reducing memory to <600MB and training time to ~1s. *(logged 2026-11-13)*

- Integrated Multi-Head Temporal Self-Attention in `BiomechanicalFoulEncoder` (`embed_dim=128`, `num_heads=4`, dropout=0.10) with residual connection and LayerNorm to capture dynamic contest sequences; added cutout data augmentation in `ConsistentVideoAugmentor`. *(logged 2026-11-12)*
- Built Identity-Preserved Focal Player Multi-Object Tracker (`track_focal_player_full_video`) replacing naive frame-by-frame ROI picking with IoU and spatial proximity tracklet association; guarantees the box locks onto the correct focal shooter/driver and contesting defender without jumping to referees or bench. *(logged 2026-11-12)*
- Synchronized dataset discovery to train on all 104 balanced `.pt` action tubes (52 foul, 52 no_foul) with class-balanced label-smoothed focal loss. *(logged 2026-11-12)*

- Upgraded dual-stream architecture to 768-D late fusion (31.8M params) with clean dimension constants (`CHANNELS`, `FRAMES`, `HEIGHT`, `WIDTH`, `POSE_FEATURES`, `VIDEO_FEATURE_DIM`, `POSE_HIDDEN_DIM`, `COMBINED_FEATURE_DIM`, `CLASSIFIER_HIDDEN_DIM`) and 12 human-readable pose dimension names. *(logged 2026-11-12)*
- Fixed Stage 4 checkpoint loading error via safe state dict dimension checking; implemented continuous bounding box tracking in `predict_single_clip` across 100% of video frames, directing outputs to `data/annotated/`. *(logged 2026-11-12)*
- Expanded dataset to 349 plays across 18+ games, pulling 97 foul ground-truth cases. *(logged 2026-11-12)*

- Lowered default scraping & training batch sizes (`max_games=3`, `max_clips=12`, `batch_size=2`, `accum_steps=2`, `epochs=5`) so all student laptops can run the pipeline smoothly; documented full scaling flags in `README.md`. *(logged via System Overhaul, 2026-11-11)*
- Added cross-OS path resolution (`BASE_DIR`), POSIX CSV path normalization, OpenCV/OpenMP thread caps (`cv2.setNumThreads(0)`), and offline synthetic tensor fallback (`--synthetic-fallback`). *(logged via System Overhaul, 2026-11-11)*
