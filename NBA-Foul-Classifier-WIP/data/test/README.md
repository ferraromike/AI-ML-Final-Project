# Dedicated Test Dataset (`data/test/`)

Place any broadcast `.mp4`, `.mov`, `.avi`, or `.mkv` video clips here for inference and evaluation.

## Notes:
- Video clips in this directory are ignored by git (see `.gitignore`) to keep repository size lean.
- Clips in this folder are strictly segregated from training data to guarantee zero data leakage.
- Run batch evaluation and continuous tracking across all test clips with:
  ```bash
  python 4_train_eval.py --annotate-test
  ```
  Or predict on an individual clip:
  ```bash
  python 4_train_eval.py --predict data/test/<video_clip>.mp4
  ```
