# NBA Coach's Challenge & Replay Archive Classifier (`NBA-Replay-Foul-Classifier`)

A standalone Computer Vision & Context-Conditioned Multi-Head Spatiotemporal Deep Learning pipeline built for official NBA Replay Center videos scraped from **[official.nba.com/replay/archive](https://official.nba.com/replay/archive/)**.

The system knows **what call is being challenged** (Called Foul, Out of Bounds, or Goaltending / Basket Interference — either from archive metadata, CLI context, filename trigger, or its own learned `challenge_type_head`) and determines **the exact Replay Center ruling (`CALL STANDS` vs. `CALL OVERTURNED`) and the specific correct call to be made on the floor**.

---

## 1. Architecture Overview

```text
+-------------------------------------------------------------------------------------------------+
| STAGE 1: Full NBA Replay Archive Scraper (`1_scrape_replays.py`)                                |
|  • Queries /wp-json/api/v1/query_replay_feed across all seasons (2019-20 .. 2026-27) by default |
|  • Scrapes ALL Coach's Challenge types (Called Foul, Out of Bounds, Goaltending/BI) with no cap |
|  • Parses verbatim Replay Center rulings into:                                                  |
|      - Challenge Type (`0=challenge_foul`, `1=challenge_oob`, `2=challenge_goaltending_bi`)     |
|      - Specific Correct Call (`0..7`: foul_stands, foul_overturned_no_foul,                     |
|        foul_overturned_other_foul, oob_stands, oob_overturned, oob_proximate_foul,              |
|        goaltending_bi_stands, goaltending_bi_overturned)                                        |
|      - Binary Foul Target (`1 = foul on play`, `0 = no foul on play`)                           |
+------------------------------------------------+------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
| STAGE 2: Shot-Segmented Multi-Angle 16-D Court & Pose Extractor (`2_preprocess_replays.py`)     |
|  • Scans full replay clip at constant 5 Hz (handles 60 fps, 30 fps, and truncated MP4s safely)  |
|  • Detects hard camera cuts & filters out low-saliency dead-ball intro/outro angles             |
|  • Extracts up to K shot-bounded (3, 16, 224, 224) RGB tubes + (16, 16) biomechanical & court   |
|    kinematics (arm intrusion, cylinder invasion, landing zone, verticality, ball proximity,     |
|    collision shock, rim proximity, sideline/baseline boundary proximity, deflection, charge)    |
+------------------------------------------------+------------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
+----------------------------------------------+   +----------------------------------------------+
| STREAM A: 3D Spatiotemporal Visual Backbone  |   | STREAM B: 16-D Pose & Court Attention        |
|  • Pretrained R(2+1)D-18 (Kinetics-400)      |   |  • Input: (B, 16, 16) kinematic/court vector |
|  • GPU: partial fine-tuning (layer3+layer4)  |   |  • 4-head temporal self-attention + max pool |
|  • CPU/MPS: frozen + cached 512-D embeddings |   |    + explicit temporal summary statistics    |
|  • Output: 512-D visual embedding            |   |  • Output: 256-D kinematic/court embedding   |
+----------------------+-----------------------+   +-----------------------+----------------------+
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
| STAGE 3 & 4: Context-Conditioned Multi-Head Classifier (`3_model` / `4_train_eval`)             |
|  • Head 1 (`challenge_type_head`): Identifies what is being challenged (Foul / OOB / GT-BI)     |
|  • Context Embedding (`challenge_context_embed`): Conditions decision heads on challenge type   |
|  • Head 2 (`call_head`): Predicts the 8 specific call outcomes valid for the challenged call    |
|  • Head 3 (`resolution_head`): Predicts CALL STANDS (0) vs. CALL OVERTURNED (1)                 |
|  • Head 4 (`foul_head`): Predicts calibrated binary foul probability + biomechanical prior      |
|  • Saliency-weighted multi-angle pooling + broadcast HUD annotated MP4s & predictions.json      |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Supported Challenge Types & Specific Call Taxonomy

| `call_label` | `outcome_category` | `challenge_type` (Folder) | Resolution | Specific Correct Call to Be Made | `is_foul` |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **0** | `foul_stands` | `foul` | `CALL STANDS` | `CALLED FOUL STANDS (Personal / Shooting Foul Upheld)` | `True` |
| **1** | `foul_overturned_no_foul` | `foul` | `CALL OVERTURNED` | `NO FOUL (Overturned — Legal Guarding / Marginal Contact)` | `False` |
| **2** | `foul_overturned_other_foul` | `foul` | `CALL OVERTURNED` | `FOUL OVERTURNED TO OFFENSIVE / LOOSE-BALL FOUL` | `True` |
| **3** | `oob_stands` | `oob` | `CALL STANDS` | `OUT OF BOUNDS STANDS (Original Possession Upheld)` | `False` |
| **4** | `oob_overturned` | `oob` | `CALL OVERTURNED` | `OUT OF BOUNDS OVERTURNED (Possession Reversed)` | `False` |
| **5** | `oob_proximate_foul` | `oob` | `CALL OVERTURNED` | `OUT OF BOUNDS OVERTURNED — PROXIMATE FOUL ASSESSED` | `True` |
| **6** | `goaltending_bi_stands` | `goaltending` | `CALL STANDS` | `GOALTENDING / BASKET INTERFERENCE STANDS` | `False` |
| **7** | `goaltending_bi_overturned` | `goaltending` | `CALL OVERTURNED` | `NO GOALTENDING / BI (Overturned — Legal Play / Clean Block)` | `False` |
| **8** | `flagrant_confirmed` | `flagrant` | `CALL STANDS` | `FLAGRANT FOUL (PENALTY 1 / 2) CONFIRMED` | `True` |
| **9** | `flagrant_downgraded_common_foul` | `flagrant` | `CALL OVERTURNED` | `COMMON PERSONAL FOUL (Did Not Meet Flagrant Criteria)` | `True` |

---

## 3. Running the Pipeline via CLI

### Scrape All Coach's Challenge Clips Across All Seasons (Default)
```bash
# Scrapes all seasons (2019-20 .. 2026-27) and all challenge types (Foul, OOB, Goaltending/BI) with no limits
python NBA-Replay-Foul-Classifier/1_scrape_replays.py

# Optional: scrape a specific subset or SD resolution for rapid laptop testing
python NBA-Replay-Foul-Classifier/1_scrape_replays.py \
    --seasons 2026-27 2025-26 --max-inspect 200 --max-clips 80 --quality sd
```

### Preprocess, Train & Predict
```bash
# 2. Extract shot-bounded camera-angle tubes & 16-D court/pose features
python NBA-Replay-Foul-Classifier/2_preprocess_replays.py --angles-per-clip 2

# 3. Train context-conditioned multi-head challenge replay classifier
python NBA-Replay-Foul-Classifier/4_train_eval.py --backbone-mode auto --epochs 18

# 4. Run inference on all held-out clips in data/test/ (auto-detects what is being challenged)
python NBA-Replay-Foul-Classifier/4_train_eval.py --predict-test

# 5. Or run inference on a single clip with an explicit challenge type ('auto', 'foul', 'oob', 'goaltending_bi')
python NBA-Replay-Foul-Classifier/4_train_eval.py \
    --predict NBA-Replay-Foul-Classifier/data/test/celticsfoul.mp4 --challenge-type foul
```
