# NBA Replay Archive Dataset (`data/`)

This directory holds metadata, downloaded official NBA Replay Center videos, preprocessed multi-angle action tubes, held-out test footage, and annotated prediction videos.

## Directory Layout

```text
data/
├── replay_metadata.csv            # Master catalog scraped from official.nba.com/replay/archive/
├── raw/                           # Downloaded Replay Center MP4 clips (ignored by git)
│   ├── foul/                      # Label 1: Confirmed/upheld fouls & overturned-to-other-foul plays
│   └── no_foul/                   # Label 0: Foul challenges overturned to no foul
├── processed/                     # Preprocessed multi-angle .pt tensors (ignored by git)
│   ├── foul/
│   ├── no_foul/
│   ├── processed_metadata.csv     # Index linking each angle tube (.pt) to play_id and game_id
│   └── visual_embeddings_cache.pt # Optional cached 512-D R(2+1)D-18 embeddings for fast CPU/MPS mode
├── test/                          # Held-out video clips strictly excluded from training/preprocessing
└── annotated/                     # Binary FOUL / NO FOUL prediction videos & predictions.json
```

## Binary Ground-Truth Mapping (`replay_metadata.csv`)

| Column | Description |
| :--- | :--- |
| `play_id` | Unique replay identifier (`{game_id}_{opsreporting_video_id}`) |
| `game_id` | 10-digit NBA game ID (used for leak-free `GroupShuffleSplit`) |
| `season` | NBA season (e.g. `2026-27`, `2025-26`) |
| `game_date` | Date of the game |
| `trigger` | Official Replay Center trigger slug (`coachs-challenge`, `review-of-called-foul`, etc.) |
| `title` | Archive title (e.g. `Challenge of Called Foul: Jazz @ Clippers`) |
| `outcome_category` | Parsed outcome (`foul_upheld`, `overturned_to_other_foul`, `overturned_to_no_foul`, `flagrant_review_foul`, etc.) |
| `ruling_text` | Verbatim NBA Replay Center ruling text scraped from the archive permalink |
| `label` | Binary target: `1` = **Foul**, `0` = **No Foul** |
| `video_path` | Relative POSIX path to local MP4 (`data/raw/{foul,no_foul}/{play_id}.mp4`) |
| `mp4_url` | Direct `videos.nba.com` MP4 stream URL |
| `permalink` | Source `official.nba.com/archive/...` URL |
