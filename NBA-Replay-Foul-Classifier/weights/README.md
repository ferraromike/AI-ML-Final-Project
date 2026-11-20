# Model Weights & Checkpoints (`weights/`)

Large `.pt` binary files are excluded by `.gitignore`. Place or auto-download the following weights here:

| File | Role |
| :--- | :--- |
| `yolo11n.pt` | Ultralytics YOLOv11 Nano object detector (players + basketball) used in Stage 2 shot/contact extraction |
| `yolo11n-pose.pt` | Ultralytics YOLOv11 Nano 17-keypoint pose estimator used to compute 12-D biomechanical contact kinematics |
| `best_replay_foul_classifier.pt` | Trained binary `BinaryFoulClassifier` checkpoint saved by `4_train_eval.py` (contains model weights, calibrated probability threshold, and play-level validation metrics) |

All pipeline scripts automatically resolve `yolo11n.pt` and `yolo11n-pose.pt` from `NBA-Replay-Foul-Classifier/weights/`, the repository root, or `NBA-Foul-Classifier-WIP/weights/`, and fall back to Ultralytics auto-download if not found locally.
