# Dedicated Held-Out Test Clips (`data/test/`)

Drop any `.mp4`, `.mov`, `.avi`, `.mkv`, or `.webm` NBA replay clips here for isolated evaluation.

- Any video filename stem present in `data/test/` is **automatically excluded** from Stage 2 preprocessing (`2_preprocess_replays.py`) and Stage 3/4 training (`ReplayFoulDataset`) to guarantee zero data leakage.
- Run batch binary inference (`FOUL` vs. `NO FOUL`) on everything in this folder:

```bash
python NBA-Replay-Foul-Classifier/4_train_eval.py --predict-test
```

Outputs are written to `data/annotated/<clip_stem>_annotated.mp4` and `data/annotated/predictions.json`.
