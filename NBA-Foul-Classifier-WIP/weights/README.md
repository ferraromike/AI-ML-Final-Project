# Model Weights & Checkpoints Directory (`weights/`)

This directory houses pretrained foundation models and trained classification checkpoints used across the NBA Foul Classifier pipeline.

---

## 1. Directory Contents & Model Summary

| File | Size | Architecture | Origin / Source | Primary Role in Pipeline |
| :--- | :---: | :--- | :--- | :--- |
| **`yolo11n.pt`** | ~5.6 MB | YOLOv11 Nano Detection (2.6M params) | Ultralytics / COCO (80 classes) | Court Player & Ball Detection (Class 0: `person`, Class 32: `sports ball`) in Stage 4/5 full-video inference and disputed interaction tracking |
| **`yolo11n-pose.pt`** | ~6.2 MB | YOLOv11 Nano Pose (2.9M params) | Ultralytics / COCO Pose (17 keypoints) | Skeletal joint extraction (wrists, elbows, shoulders, hips, knees, ankles) in Stage 2 action-tube extraction to compute the 12 Mechanical Foul Dimensions |
| **`best_foul_r2plus1d.pt`** | ~127.8 MB | Dual-Stream Late-Fusion (~33.8M params) | Custom Trained (Stage 4) | Calibrated foul classification checkpoint (`BiomechanicalFoulEncoder` + `classifier_head`), probability threshold, and evaluation metrics |

---

## 2. In-Depth Explanation: What Are These Files?

### A. `yolo11n.pt` (Object Detection)
* **What it is**: The lightweight ("nano") version of the state-of-the-art YOLOv11 object detection architecture released by Ultralytics.
* **Why it is here**: In Stage 4 and Stage 5 (`predict_single_clip` and `annotate_test_directory`), we need to track both players involved in the disputed contest (shooter/driver and contesting defender) across all 600 frames of full broadcast video.
* **How it works**:
  - Scans each video frame for bounding boxes with classes `0` (`person`) and `32` (`sports ball`).
  - Gated by active court geometry ($y_2 \le 0.72 \times H$, $y_c \in [0.25H, 0.68H]$, $x_c \in [0.12W, 0.80W]$) and movement speed gating (`max_d = 160 px`) to reject foreground referees (e.g. Referee #43) and sideline bystanders.
  - Matches the shooter and defender bidirectionally to form the continuous green/red enclosing bounding box.

### B. `yolo11n-pose.pt` (Skeletal Joint Estimation)
* **What it is**: The pose-estimation variant of YOLOv11 Nano, trained on the COCO Keypoints dataset.
* **Why it is here**: Traditional bounding boxes cannot measure physical contact violations such as wrist slaps, body cylinder invasion, or landing space violations. We need exact anatomical joint coordinates.
* **How it works**:
  - Detects 17 keypoints per player (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) along with confidence scores.
  - Used in `2_preprocess_tubes.py` to extract 12 Mechanical Foul Dimensions per frame:
    1. `arm_contact_intrusion` (wrist/elbow collision)
    2. `body_cylinder_penetration` (torso/hip overlap into shooter cylinder)
    3. `lower_body_landing_violation` (feet stepping into landing space)
    4. `defender_verticality_angle` (deviation from legal 90-degree vertical wall)
    5. `defender_lateral_positioning` (side-to-side displacement)
    6. `offensive_displacement` (trajectory deflection on contact)
    7. `defender_to_ball_distance` (proximity of defender hand to ball)
    8. `ball_to_shooter_distance` (gather vs release detection)
    9. `ball_strike_vs_body_ratio` (clean block vs hack ratio)
    10. `relative_closing_velocity` (speed of defender collision)
    11. `momentum_transfer_shock` (impact jolt upon collision)
    12. `airborne_contest_apex_diff` (apex elevation contest differential)

### C. `best_foul_r2plus1d.pt` (Trained Dual-Stream Classifier Checkpoint)
* **What it is**: The custom PyTorch checkpoint generated at the end of Stage 4 training (`4_train_eval.py` / notebook Cell 7).
* **Why it is here**: It stores the learned weights from training across balanced L2M action tubes.
* **What it contains**:
  - `model_state_dict`: Trained parameters for `BiomechanicalFoulEncoder` (4-head temporal self-attention across 16 frames) and `classifier_head` (3-layer MLP: 768 $\to$ 512 $\to$ 256 $\to$ 1).
  - `optimal_threshold`: Calibrated probability cutoff $\tau \in [0.20, 0.80]$ that balances F1-score and classification accuracy.
  - `metrics`: Evaluation metrics achieved on the validation split (accuracy, F1, precision, recall, ROC-AUC).
  - `epoch`: Epoch number where the best validation F1 was achieved.

---

## 3. Path Resolution & Automatic Discovery

All pipeline scripts resolve weights automatically via `resolve_project_path()`:
1. `weights/<filename>`
2. `NBA-Foul-Classifier-WIP/weights/<filename>`
3. `NBA-Foul-Classifier-WIP/<filename>`
4. Root project directory

If `yolo11n.pt` or `yolo11n-pose.pt` is missing from `weights/`, Ultralytics will download it automatically on first run into `weights/`.
