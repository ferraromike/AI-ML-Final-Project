# AI Referee Raw Game Footage Drop Folder (`data/test/`)

Drop **any raw NBA basketball game footage** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v` — short clips or multi-minute broadcast segments from any game) into this folder to test the v3.0 AI Referee autonomously.

## How It Works
1. **Notebook Folder Testing (Cells 9 & 10 in `NBA_Foul_Classifier_Full_Pipeline.ipynb`)**:
   - Place any raw game video file(s) into `NBA-Foul-Classifier-WIP/data/test/`.
   - Run **Cell 9** (and **Cell 10** for interactive HTML5 replay). Every video in `data/test/` is evaluated via sliding-window **Swin3D-T + Gated Attention MIL** with **E-BARD Zero-Referee Perception** and exported to `data/annotated/<video_stem>_annotated.mp4`.
2. **Command-Line Batch Testing**:
   ```bash
   python 4_train_eval.py --annotate-test
   ```
   Or evaluate any single video clip or folder directly:
   ```bash
   python 4_train_eval.py --predict data/test/<video_clip>.mp4
   ```
3. **100% Autonomous AI Referee Behavior (Zero Metadata Cheating)**:
   - Works on completely unseen NBA games and arbitrary video lengths without reading `l2m_metadata.csv`.
   - **Zero-Referee / Zero-Spectator Filtering**: Uses `ebard_basketball_yolo.pt` (`basketball`, `hoop`, `player`, `referee`) + per-clip 2-team CIELAB jersey clustering so referees and spectators are never tagged as players.
   - **When Foul(s) Occur**: Highlights **WHAT foul it is** (`Shooting Foul` / `Personal Foul` + NBA Rule 12 mechanic), **WHERE it occurred** (opposing-team boxes, translucent red zone, contact target reticle `(x, y)`, court zone), and **WHEN it occurred** (timestamp, frame index, timeline marker) strictly inside each detected foul window.
   - **When NO Foul Occurs**: **Nothing happens** on the video — it plays completely clean and unaltered from start to finish.
4. **Strict Zero-Leakage Isolation**:
   - Clips placed in `data/test/` are excluded from preprocessing and training by **both filename stem AND video content hash**.
