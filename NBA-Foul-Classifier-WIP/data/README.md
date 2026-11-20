# NBA L2M Disputed Play Dataset & v3 Multi-Window MIL Bags (`data/`)

This directory stores official NBA Last Two Minute (L2M) metadata, raw broadcast clips, v3 Multi-Window MIL bag tensors, segregated raw test footage, and AI Referee annotated video exports.

---

## 1. Directory Structure

```text
data/
├── l2m_metadata.csv               # Deduplicated L2M plays with real `call_type` and `game_id`
├── l2m_incorrect_fouls.csv        # Officiating review breakdown (INC / IC / CC)
├── raw/                           # Raw broadcast MP4 clips downloaded from NBA CDN (gitignored)
│   ├── foul/                      # Label 1: CC (Correct Call) + INC (Incorrect Non-Call)
│   └── no_foul/                   # Label 0: CNC (Correct Non-Call) + IC (Incorrect Call)
├── processed/                     # v3 E-BARD + Swin3D-T Multi-Window MIL bag `.pt` files (gitignored)
│   ├── foul/                      # Positive MIL bags (K <= 10 windows/clip)
│   ├── no_foul/                   # Negative MIL bags (K <= 10 windows/clip)
│   └── processed_metadata.csv     # Index linking raw clips to processed `.pt` bags
├── test/                          # Drop ANY raw NBA footage here for autonomous testing!
│   └── README.md                  # Drop-folder testing instructions
└── annotated/                     # AI Referee annotated replays & `referee_decisions.json` (gitignored)
```

---

## 2. Metadata Schema & Data Hygiene (`l2m_metadata.csv`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `play_id` | `str` | Unique play identifier (e.g. `0022600196_2132_05`) |
| `game_id` | `str` | Official 10-digit NBA game code — used for **leak-free game-grouped train/val splitting** |
| `player_committing` | `str` | Official L2M committing player (used only for EDA, **never read during inference**) |
| `player_disadvantaged` | `str` | Official L2M disadvantaged player (used only for EDA, **never read during inference**) |
| `call_type` | `str` | Real ground-truth foul type (`Shooting Foul` or `Personal Foul`) used to supervise `window_type_head` |
| `review_decision` | `str` | Official NBA ruling (`CC`, `INC`, `CNC`, `IC`) used in Multi-Window RL rewards |
| `label` | `int` | Binary classification target (`1` = Foul, `0` = Legal / No Foul) |
| `video_path` | `str` | Relative POSIX path to raw broadcast MP4 |

### Automatic Deduplication & Zero-Leakage Segregation
1. **Content-Hash Deduplication (`1_scrape_l2m.py --deduplicate`)**: Multiple L2M rows occasionally point to the exact same NBA Stats `.mp4` asset. Deduplication keeps the `foul` (`1`) record when a video has conflicting labels and removes exact byte duplicates.
2. **Test Set Isolation (`data/test/`)**: `2_preprocess_tubes.py` and `3_model_and_dataset.py` exclude clips in `data/test/` by **both filename stem AND video byte hash**, preventing any duplicate raw clip in `data/raw/` from leaking into training.
3. **Game-Grouped Cross-Validation**: `3_model_and_dataset.py` splits train/validation via `StratifiedGroupKFold` grouped strictly by **`game_id`** so no two plays from the same NBA game ever appear in both train and validation.

---

## 3. v3 Multi-Window MIL Bag `.pt` Tensor Specification

Each `.pt` file in `data/processed/{foul,no_foul}/` (`extractor_v = 3`) contains:
- **`window_embeddings`**: `torch.Tensor` `(K, 768)` — Kinetics-400 **`Swin3D-T`** visual embeddings across up to `K = 10` overlapping `2.0s` windows covering the entire live-action clip with zero blind spots.
- **`window_pose_features`**: `torch.Tensor` `(K, 16, 16)` — 16-D depth-gated NBA Rule 12 + E-BARD ball/hoop kinematic sequences per window (with referees and spectators strictly excluded via E-BARD + 2-team CIELAB clustering).
- **`window_center_frames`**: `torch.Tensor` `(K,)` — Original broadcast video center frame index for each candidate window.
- **`window_activity_scores`**: `torch.Tensor` `(K,)` — Per-window on-ball contest activity summary.
- **`video`**: `torch.Tensor` `(3, 16, 224, 224)` uint8 — Stabilized RGB action tube for the peak window.
- **`label`**, **`call_type`**, **`call_type_label`** (`1` = Shooting Foul, `0` = Personal Foul), **`review_decision`**, **`game_id`**, **`fps`**, **`used_ebard`**, **`is_synthetic`**.

---

## 4. AI Referee Annotated Output (`data/annotated/`)

- **When Foul(s) Occur**: Highlights **WHAT foul it is**, **WHERE it occurred** (red foul zone, opposing-team Offense/Defense boxes with zero referees, contact reticle `(x, y)`, and court zone), and **WHEN it occurred** strictly during each foul window.
- **When NO Foul Occurs**: **Nothing happens** on the output video (100% clean unaltered playback).
- **`referee_decisions.json`**: Structured per-video JSON report listing `is_foul`, `num_fouls`, `foul_prob`, `bag_prob`, `threshold`, `what_foul`, `foul_subtype`, `foul_mechanic`, `when_frame`, `when_time_sec`, `where_zone`, and full multi-event `events` list for any-length game footage.
