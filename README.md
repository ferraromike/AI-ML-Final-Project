# NBA AI Referee & Foul Detection Pipeline (v3.0 — E-BARD Basketball Perception + Video Swin3D-T + Gated Attention MIL)

A cross-platform Computer Vision, Spatiotemporal Deep Learning, and Reinforcement Learning **AI Referee** pipeline engineered to detect **IF**, **WHAT**, **WHEN**, and **WHERE** basketball fouls occur on **any raw NBA game footage** across **Google Colab (NVIDIA T4 GPU), macOS (Apple Silicon & Intel), Windows, and Linux**.

---

## 1. System Architecture (v3.0 Full-Stack Overhaul)

The v3.0 architecture eliminates generic COCO referee/spectator contamination, single-crop temporal blind spots, fake call-type targets, and same-game train/val data leakage:

```text
+---------------------------------------------------------------------------------------------------+
| STAGE 1: NBA L2M Scraper, Parallel Downloader & Content-Hash Deduplicator (1_scrape_l2m.py)       |
|  • Scrapes official.nba.com L2M reports & CDN broadcast MP4 clips                                 |
|  • Ground-Truth Labels: CC/INC -> Foul (1), CNC/IC -> Legal / No Foul (0) + real `call_type`      |
|  • Content-Hash Deduplication: eliminates conflicting-label & duplicate NBA Stats video assets    |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 2: E-BARD Basketball Perception, 2-Team CIELAB Clustering & Multi-Window Bags (2_preprocess)|
|  • E-BARD NBA YOLO (`ebard_basketball_yolo.pt`): native `{basketball, hoop, player, referee}`     |
|    - Explicitly separates NBA referees (class 3) and detects ONLY active on-court players         |
|    - Secondary heather-gray shirt + full-length black trousers guard rejects occluded officials   |
|  • Unsupervised Clip-Level 2-Team CIELAB K-Means Clustering (`team_0` vs `team_1`)                |
|    - Hard opposing-team pairing (`team_off != team_def`) prevents pairing teammates or referees   |
|  • 16-D Depth-Gated NBA Rule 12 + Ball/Hoop Kinematics (`yolo11n-pose.pt` 17-keypoint fusion)     |
|  • Multi-Window Sliding Bags: slices wide-court play into overlapping 2.0s windows (`K <= 10`)    |
|    and encodes every window with frozen Kinetics-400 Video Swin Transformer (`Swin3D-T`, `768-D`)   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 3 & 4: Swin3D-T + Gated Attention MIL & Leak-Free Game-Grouped Training (3_model / 4_train) |
|  • Dual-Stream Window Encoder: `768-D Swin3D-T` + `192-D 4-Head Temporal Pose Attention` -> 256-D |
|  • Per-Window Heads: `window_foul_head` (WHEN!) + `window_type_head` (WHAT: real L2M `call_type`) |
|  • Gated Attention MIL (ICML 2018): pushes all `no_foul` windows -> 0, peaks on foul window -> 1  |
|  • ~439K trainable params over frozen 27.8M Swin3D-T (prevents overfitting on small clip counts)  |
|  • Leak-Free 5-Fold `StratifiedGroupKFold` Ensemble (`FoulMILEnsemble`) grouped by NBA `game_id`    |
|  • Global Out-of-Fold (OOF) Temperature Scaling & Threshold Sweep (zero hardcoded overrides)        |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 5: Multi-Window Sequential RL Referee Agent (`NBARefereeEnv` + Double Dueling DQN)          |
|  • Distinct per-window Swin3D-T + calibrated MIL probability + 16-D kinematic trajectory          |
|  • Trained with real L2M review rewards (`CC`, `INC`, `CNC`, `IC`) and evaluated on unseen games  |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| STAGE 6: Autonomous Sliding-Window Any-Length Inference & Zero-Referee Annotation (data/test/)    |
|  • Drop ANY raw NBA video of ANY duration into `data/test/` -> Temporal NMS across all windows    |
|  • 100% autonomous (zero CSV metadata lookups during inference)                                   |
|  • If NO foul occurs: NOTHING happens on the output video (100% clean unaltered playback)         |
|  • If 1+ fouls occur: Highlights WHAT foul it is, WHERE it occurred (opposing players only—zero   |
|    referees!), and WHEN it occurred strictly during each foul window -> `data/annotated/`         |
+---------------------------------------------------------------------------------------------------+
```

### 16 Mechanical Foul & Basketball Context Dimensions (`MECHANICAL_FOUL_DIMENSIONS`)
Grounded in NBA Rule 12 (Fouls and Penalties) plus 3D depth alignment and native E-BARD basketball/hoop tracking:
1. `[0] arm_contact_intrusion`: Depth-gated defender wrist/elbow intrusion on offensive arm/upper body.
2. `[1] body_cylinder_penetration`: Depth-gated torso/hip invasion into the offensive player's cylinder.
3. `[2] lower_body_landing_violation`: Defender feet/legs encroaching under an airborne shooter's landing space.
4. `[3] defender_verticality_angle`: Defender arm reach deviation from a legal 90° vertical contest.
5. `[4] defender_lateral_positioning`: Lateral offset relative to the offensive drive path.
6. `[5] offensive_displacement`: Frame-to-frame sudden deflection/acceleration of the offensive player's center of mass.
7. `[6] defender_to_ball_distance`: Proximity of defender wrists to the E-BARD detected basketball.
8. `[7] ball_to_shooter_distance`: Proximity of the E-BARD basketball to the offensive player's hands/torso.
9. `[8] ball_strike_vs_body_ratio`: Body collision severity relative to clean contact near the basketball.
10. `[9] relative_closing_velocity`: Temporal closing speed between defender and ball-handler.
11. `[10] momentum_transfer_shock`: Kinetic deceleration / impact shock upon contact.
12. `[11] airborne_contest_apex_diff`: Shooter upward arm elevation and vertical contest intensity.
13. `[12] hoop_proximity`: Proximity of the contest pair to the E-BARD detected hoop/rim.
14. `[13] foot_plane_depth_alignment`: Same-plane court floor alignment (`|y2_off - y2_def|`) preventing depth-overlap false positives.
15. `[14] ball_verified_on_play`: Continuous confidence that the E-BARD basketball is actively involved in the pair's play.
16. `[15] shooting_motion_score`: Shooting gather/release indicator combining wrist elevation, ball proximity, and hoop proximity.

---

## 2. Repository Structure

```text
AI-ML-Final-Project/
├── README.md                                  # Root v3.0 architecture & cross-platform setup guide
├── .gitignore                                 # Ignores binary weights, raw/processed media, venv
├── NBA_Foul_Classifier_Presentation.pptx      # Project slide deck
├── NBA-Foul-Classifier-WIP/
│   ├── NBA_Foul_Classifier_Full_Pipeline.ipynb # Unified interactive notebook (Colab T4 & local)
│   ├── 1_scrape_l2m.py                        # Stage 1: NBA L2M scraper, downloader & conflict deduplicator
│   ├── 2_preprocess_tubes.py                  # Stage 2: E-BARD perception, 2-team LAB clustering & Swin3D-T MIL bags
│   ├── 3_model_and_dataset.py                 # Stage 3: FoulMILClassifier (Swin3D-T + Gated MIL) & Multi-Window RL
│   ├── 4_train_eval.py                        # Stage 4–6: Game-grouped CV training, calibration & zero-referee annotation
│   ├── requirements.txt                       # Cross-platform Python dependencies
│   ├── weights/                               # Foundation models & trained checkpoints (auto-downloaded / gitignored)
│   │   ├── README.md                          # Detailed weight & architecture specifications
│   │   ├── ebard_basketball_yolo.pt           # E-BARD NBA detector (`basketball`, `hoop`, `player`, `referee`)
│   │   ├── yolo11n-pose.pt                    # Ultralytics YOLOv11-Pose 17-keypoint skeletal detector
│   │   ├── best_foul_r2plus1d.pt              # Trained v3 Swin3D-T + Gated MIL checkpoint + temperature/threshold
│   │   └── referee_rl_agent.pt                # Trained Multi-Window Double Dueling DQN Referee checkpoint
│   └── data/
│       ├── README.md                          # Dataset schema, MIL bag tensor format & output specs
│       ├── l2m_metadata.csv                   # Deduplicated L2M plays with real `call_type` and `game_id`
│       ├── l2m_incorrect_fouls.csv            # Officiating review breakdown (INC / IC / CC)
│       ├── raw/                               # Downloaded broadcast MP4 clips ({foul,no_foul}/)
│       ├── processed/                         # v3 Multi-Window MIL bag .pt tensors ({foul,no_foul}/)
│       ├── test/                              # Drop ANY raw NBA game footage here for autonomous testing!
│       └── annotated/                         # AI Referee annotated replays & `referee_decisions.json`
└── discord-bot/                               # Team AI Engineering Assistant ("Fred")
    ├── bot.py                                 # Discord bot with automatic PROJECT_MEMORY.md updates
    ├── PROJECT_MEMORY.md                      # Shared persistent engineering memory & milestones
    └── README.md                              # Discord bot setup guide
```

---

## 3. Cross-Platform Setup (macOS, Windows, Linux & Google Colab)

### macOS / Linux
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

---

## 4. Running the AI Referee Pipeline

### Option A: Interactive Jupyter Notebook / Google Colab (Recommended)
Open [`NBA-Foul-Classifier-WIP/NBA_Foul_Classifier_Full_Pipeline.ipynb`](NBA-Foul-Classifier-WIP/NBA_Foul_Classifier_Full_Pipeline.ipynb) in **Google Colab (T4 GPU)**, **VS Code**, or **JupyterLab**:
- **Cell 1**: Auto-bootstraps Colab (clones `testing` branch, installs packages, auto-downloads `ebard_basketball_yolo.pt`).
- **Cells 2–4**: Syncs/deduplicates L2M clips and builds v3 E-BARD + 2-Team CIELAB + Swin3D-T Multi-Window MIL bags.
- **Cells 5–8**: Runs leak-free 5-fold game-grouped CV training, temperature calibration, and Multi-Window RL Referee training/telemetry.
- **Cells 9–10**: Drop **any raw NBA game footage** into `NBA-Foul-Classifier-WIP/data/test/` and run Cells 9–10 to evaluate every video autonomously and replay annotated videos right inside the notebook.

### Option B: Command-Line Pipeline
```bash
# 1. Deduplicate metadata (or scrape additional NBA L2M games with --max-games 60 --max-clips 250 --include-archive)
python NBA-Foul-Classifier-WIP/1_scrape_l2m.py --deduplicate

# 2. Extract v3 E-BARD + 2-Team CIELAB + Swin3D-T Multi-Window MIL bags (auto-downloads ebard_basketball_yolo.pt)
python NBA-Foul-Classifier-WIP/2_preprocess_tubes.py --max-clips 0

# 3. Train FoulMILClassifier with leak-free 5-fold game-grouped CV + Temperature Calibration + Multi-Window RL
python NBA-Foul-Classifier-WIP/4_train_eval.py --epochs 25 --cv-folds 5 --train-rl

# 4. Drop ANY raw NBA videos into data/test/ and run the autonomous AI Referee
python NBA-Foul-Classifier-WIP/4_train_eval.py --annotate-test

# Or run on a single arbitrary-length raw game video:
python NBA-Foul-Classifier-WIP/4_train_eval.py --predict path/to/any_nba_game_clip.mp4
```

---

## 5. AI Referee Annotation Behavior (`data/test/` $\to$ `data/annotated/`)

- **100% Autonomous (Zero Metadata Cheating)**: Inference never reads `l2m_metadata.csv` labels, call types, or player names—it works on completely unseen games and external footage.
- **Zero-Referee / Zero-Spectator Highlighting**: Uses E-BARD native `player`/`referee`/`basketball`/`hoop` classes + clip-level 2-team CIELAB jersey clustering + long-black-trousers referee filtering so officials and spectators are never boxed as players.
- **When a Foul Occurs**: Strictly inside each detected foul window (`[-0.85s, +1.35s]` around peak impact, stopping cleanly before any close-up/replay cut):
  - **WHAT**: Predicted foul category (`Shooting Foul` / `Personal Foul` from the real ground-truth-trained `window_type_head`) + NBA Rule 12 biomechanical infraction mechanic.
  - **WHERE**: Translucent red collision zone, opposing-team `Fouled Player (Offense)` and `Committing Player (Defense)` boxes, contact impact reticle `(x, y)`, and court zone label.
  - **WHEN**: Exact timestamp (`mm:ss.xx` / seconds), frame number, and bottom timeline marker.
- **When NO Foul Occurs**: **Nothing happens** on the output video—zero boxes, zero HUD banners, and zero timeline bars (100% untouched playback).
