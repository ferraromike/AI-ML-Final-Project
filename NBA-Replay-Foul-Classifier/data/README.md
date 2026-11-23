# NBA Replay Archive Dataset (`data/`)

This directory holds metadata, downloaded official NBA Replay Center videos organized by challenge/review type, preprocessed multi-angle 16-D court/pose action tubes, held-out test footage, and annotated challenge ruling videos.

## Directory Layout

```text
data/
├── replay_metadata.csv            # Master catalog scraped from official.nba.com/replay/archive/
├── raw/                           # Downloaded Replay Center MP4 clips classified by challenge type
│   ├── foul/                      # Coach's Challenge of Called Foul (stands, overturned to no foul, overturned to other foul)
│   ├── oob/                       # Coach's Challenge of Out of Bounds (stands, overturned, proximate foul)
│   ├── goaltending/               # Coach's Challenge of Goaltending / Basket Interference (stands, overturned)
│   └── flagrant/                  # Review of Called Foul / Flagrant 1 & 2 / Clear-Path (flagrant confirmed, common foul)
├── processed/                     # Preprocessed multi-angle .pt tensors classified by challenge type
│   ├── foul/
│   ├── oob/
│   ├── goaltending/
│   ├── flagrant/
│   ├── processed_metadata.csv     # Index linking each angle tube (.pt) to play_id, challenge_type & call_label
│   └── visual_embeddings_cache.pt # Cached 512-D R(2+1)D-18 embeddings for fast CPU/MPS training
├── test/                          # Held-out video clips strictly excluded from training/preprocessing
└── annotated/                     # Challenge ruling & specific call prediction videos + predictions.json
```

## Challenge & Call Ground-Truth Schema (`replay_metadata.csv`)

| Column | Description |
| :--- | :--- |
| `play_id` | Unique replay identifier (`{game_id}_{opsreporting_video_id}`) |
| `game_id` | 10-digit NBA game ID (used for leak-free `GroupShuffleSplit`) |
| `season` | NBA season (`2026-27` .. `2019-20`) |
| `trigger` | Official Replay Center trigger slug (`coachs-challenge`, `challenge-oob`, `challenge-goaltending-bi`, `review-of-called-foul`) |
| `challenge_type` | Folder category & what is being challenged: `foul` (`0`), `oob` (`1`), `goaltending` (`2`), or `flagrant` (`3`) |
| `outcome_category` | Specific 10-class outcome (`foul_stands`, `foul_overturned_no_foul`, `foul_overturned_other_foul`, `oob_stands`, `oob_overturned`, `oob_proximate_foul`, `goaltending_bi_stands`, `goaltending_bi_overturned`, `flagrant_confirmed`, `flagrant_downgraded_common_foul`) |
| `call_label` | Integer class target `0..9` corresponding to `outcome_category` |
| `call_resolution` | `CALL STANDS` or `CALL OVERTURNED` |
| `correct_call` | Official human-readable referee call to be made on the floor |
| `ruling_text` | Verbatim NBA Replay Center ruling text scraped from the archive permalink |
| `label` | Binary foul target: `1` = **Foul on Play**, `0` = **No Foul on Play** |
| `video_path` | Relative POSIX path to local MP4 (`data/raw/{foul,oob,goaltending,flagrant}/{play_id}.mp4`) |
| `mp4_url` | Direct `videos.nba.com` MP4 stream URL |
| `permalink` | Source `official.nba.com/archive/...` URL |
