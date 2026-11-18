# AI Referee Raw Game Footage Drop Folder (`data/test/`)

Place any raw basketball game footage (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) in this folder to test the AI Referee.

## How It Works
1. **Notebook Folder Testing (Cell 9 & Cell 10 in `NBA_Foul_Classifier_Full_Pipeline.ipynb`)**:
   - Drop any raw game footage files into `data/test/`.
   - Run **Cell 9** (or **Cell 10**) in the notebook. Every video in `data/test/` is automatically evaluated by the AI Referee model and exported to `data/annotated/<video_stem>_annotated.mp4`.
2. **Command-Line Batch Testing**:
   ```bash
   python 4_train_eval.py --annotate-test
   ```
   Or predict on an individual clip or custom folder:
   ```bash
   python 4_train_eval.py --predict data/test/<video_clip>.mp4
   ```
3. **AI Referee Video Annotation Behavior**:
   - **When a Foul Occurs**: The annotated video highlights **WHAT foul it is** (e.g., `Shooting Foul (Illegal Arm/Wrist Contact)`, `Personal Foul (Blocking / Cylinder Breach)`), **WHERE it occurred** (red foul zone highlight, contact target reticle at `(x, y)`, and court zone label), and **WHEN it occurred** (exact timestamp and frame number).
   - **When NO Foul Occurs**: **Nothing happens** on the video — it plays completely clean and unannotated from start to finish.
4. **Zero Data Leakage**:
   - Clips placed in `data/test/` are strictly excluded from training and preprocessing splits.
