# NBA L2M Disputed Play Dataset (`data/`)

This directory houses the curated Last Two Minute (L2M) dataset, scraped referee reports, raw broadcast clips, preprocessed spatiotemporal action tubes, and annotated inference videos.

---

## 1. Directory Structure

```text
data/
├── l2m_metadata.csv               # 349 labeled plays across 18+ NBA games
├── l2m_incorrect_fouls.csv        # 97 officiating review plays (INC & CC breakdowns)
├── raw/                           # Raw broadcast MP4 video clips downloaded from NBA CDN
│   ├── foul/                      # Plays reviewed as fouls (CC: Correct Call, INC: Incorrect Non-Call)
│   └── no_foul/                   # Plays reviewed as legal (CNC: Correct Non-Call, IC: Incorrect Call)
├── test/                          # Dedicated test clips set aside for evaluation (isolated from training)
│   ├── 0022600196_2128_01.mp4     # Giddey on Maxey (Shooting Foul)
│   ├── 0022600196_2132_05.mp4     # Terry on Council IV (Shooting Foul)
│   └── 0022600196_2134_07.mp4     # Legal rim contest / driving defense
├── processed/                     # Preprocessed action tubes and cached feature embeddings
│   ├── foul/                      # 61 preprocessed foul .pt tensors
│   ├── no_foul/                   # 66 preprocessed no_foul .pt tensors
│   ├── processed_metadata.csv     # Master link between raw MP4s and .pt action tube files
│   └── visual_embeddings_cache.pt # 123 cached 512-D Kinetics-400 visual embeddings (214 KB)
└── annotated/                     # Exported prediction videos with continuous bounding boxes
    ├── 0022600196_2128_01_annotated.mp4
    ├── 0022600196_2132_05_annotated.mp4
    └── 0022600196_2134_07_annotated.mp4
```

---

## 2. Metadata Schema (`l2m_metadata.csv` & `processed_metadata.csv`)

Each record contains the ground truth officiating review and disputed player identities:

| Column | Type | Description |
| :--- | :--- | :--- |
| `play_id` | `str` | Unique play identifier (e.g. `0022600196_2132_05`) |
| `game_id` | `str` | Official NBA 10-digit game code (e.g. `0022600196`) |
| `player_committing` | `str` | Defender involved in the contest (e.g. `Dalen Terry`, `Josh Giddey`) |
| `player_disadvantaged`| `str` | Offensive shooter or ball handler (e.g. `Ricky Council IV`, `Tyrese Maxey`) |
| `call_type` | `str` | Officiating call (e.g. `Shooting Foul`, `Personal Foul`) |
| `review_decision` | `str` | L2M ruling (`CC`: Correct Call, `INC`: Incorrect Non-Call, `CNC`: Correct Non-Call, `IC`: Incorrect Call) |
| `label` | `int` | Binary classification target (`1`: Foul, `0`: Legal / No Foul) |
| `video_path` | `str` | Relative path to raw MP4 broadcast video |
| `tensor_path` | `str` | Relative path to preprocessed `.pt` action tube tensor |

---

## 3. Dedicated Test Set (`data/test/`)

The `data/test/` directory is **strictly segregated** from training and preprocessing:
1. **Zero Data Leakage**: `2_preprocess_tubes.py` and `3_model_and_dataset.py` explicitly scan `data/test/` and exclude all matching video stems from training action-tube generation and dataset loaders.
2. **User Workflow**: Drop any video file (`.mp4`, `.mov`, `.avi`, `.mkv`) into `data/test/`.
3. **Batch Running & Annotation**:
   ```bash
   python 4_train_eval.py --annotate-test
   ```
   Or run directly on the folder:
   ```bash
   python 4_train_eval.py --predict data/test
   ```
   All test clips are automatically evaluated, continuously tracked across 100% of frames, and exported to `data/annotated/<clip_name>_annotated.mp4`.

---

## 4. Action Tube `.pt` Tensor Specification

Each `.pt` file in `data/processed/{foul,no_foul}/` contains:
- **`video`**: `torch.Tensor` of shape `(3, 16, 224, 224)` representing 16 spatial frames sampled over the 1.5-second contact window.
- **`foul_features`**: `torch.Tensor` of shape `(16, 12)` capturing the 12 Mechanical Foul Dimensions across time.
- **`video_embedding`**: Precomputed 512-D Kinetics-400 visual embedding (`torch.Size([512])`) allowing instant, zero-stall model training.
- **`label`**: Ground truth integer (`1` or `0`).
- **`crop_boxes`**: `List[Tuple[int, int, int, int]]` ROI crop coordinates in original frame space.
- **`sampled_indices`**: `List[int]` exact frame numbers in the original broadcast video.
- **`fps`**: Frame rate of the broadcast video (typically 29.97 or 30.0 fps).

### Robust Collation & Self-Healing Cache (Cell 7 Safeguard)
All 123 processed samples are indexed in `data/processed/visual_embeddings_cache.pt`. `L2MFoulDataset` and `safe_foul_collate()` enforce strict 100% homogeneous batch collation:
- If a sample lacks a cached 512-D embedding, `_extract_and_cache_embedding()` computes it on-the-fly and persists it to both the `.pt` file and in-memory cache.
- `safe_foul_collate()` dynamically validates batch shapes before stacking, completely eliminating shape-mismatch crashes.

---

## 5. Annotated Video Output (`data/annotated/`)

Annotated videos exported into `data/annotated/<play_id>_annotated.mp4` act as an **AI Referee**:
1. **When a Foul Occurs (`is_foul == True`)**:
   - **WHAT Foul It Is**: Displays the predicted foul category & NBA Rule 12 infraction subtype (e.g., `Shooting Foul (Illegal Arm/Wrist Contact)`, `Shooting Foul (Landing Space Violation)`, `Personal Foul (Blocking / Cylinder Breach)`).
   - **WHERE It Occurred**: Highlights the foul zone with a translucent red region and bold red box, individual Offense/Defense boxes, and a **Foul Contact Target Reticle** at the exact contact coordinates `(x, y)` with court zone classification (`Left Paint / Rim Area`, `Right Paint / Rim Area`, etc.).
   - **WHEN It Occurred**: Displays the exact foul timestamp and frame (`WHEN: 04.74s (Frame 142)`) on the foul call header, top-left AI Referee Whistle Banner, and bottom timeline marker during the foul window.
2. **When NO Foul Occurs (`is_foul == False`)**:
   - **Nothing Happens**: When no foul occurs, nothing happens on the video (zero boxes, zero HUD banners, 100% clean unaltered game footage).
3. **Structured Telemetry (`data/annotated/referee_decisions.json`)**:
   - Persists each evaluated clip's `is_foul`, `foul_prob`, `what_foul`, `foul_subtype`, `foul_mechanic`, `when_frame`, `when_time_sec`, `when_str`, `where_zone`, `where_point`, and `where_box` for instant in-notebook inspection.
