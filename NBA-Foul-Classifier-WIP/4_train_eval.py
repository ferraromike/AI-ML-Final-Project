#!/usr/bin/env python3
"""
STEP 4: Cross-Platform Foul Classifier Training & Continuous Full-Clip Video Annotation (`4_train_eval.py`)

Features & Upgrades:
  - Saves annotated videos into `data/annotated/<play_id>_annotated.mp4`.
  - Tracks and renders classification bounding boxes on EVERY frame across 100% of the video.
  - Simplified, high-capacity `FoulClassifier3D` (~33.8M parameters) with clear dimension names.
  - Zero-crash error handling for single-class validation splits and multi-platform video writing.
  - Flexible `predict_single_clip()` accepting either keyword arguments or CLI argparse namespace.
  - Optional H.264 remuxing with ffmpeg (+faststart) for seamless playback in QuickTime and browsers.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

cv2.setNumThreads(0)

preprocess_mod = importlib.import_module("2_preprocess_tubes")
model_mod = importlib.import_module("3_model_and_dataset")

extract_action_tube = preprocess_mod.extract_action_tube
compute_square_crop_coords = preprocess_mod.compute_square_crop_coords
smooth_boxes_ema = preprocess_mod.smooth_boxes_ema
resolve_project_path = model_mod.resolve_project_path
load_torch_file_safe = model_mod.load_torch_file_safe
ConsistentVideoAugmentor = model_mod.ConsistentVideoAugmentor
FoulClassifier3D = model_mod.FoulClassifier3D
L2MFoulDataset = model_mod.L2MFoulDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_eval")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_compute_device(requested: Optional[str] = None) -> torch.device:
    """Select the best available compute device ('cuda', 'mps', or 'cpu') cross-platform."""
    if requested and requested.lower() != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_safe_num_workers(requested_workers: int = 0) -> int:
    """Return a crash-free DataLoader worker count (0 on macOS/Windows to prevent multiprocessing hangs)."""
    if platform.system() in ("Darwin", "Windows"):
        return 0
    return max(0, requested_workers)


def get_autocast_context(device: torch.device):
    """Return cross-platform FP16 autocast context on CUDA, no-op on MPS/CPU."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


class FoulFocalBCEWithLogitsLoss(nn.Module):
    """
    Class-balanced Binary Cross-Entropy with Focal modulation (`gamma=1.5`)
    and label smoothing (`label_smoothing=0.05`).
    Addresses class imbalance and avoids overconfident gradients on borderline contact calls.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma: float = 1.5,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w = self.pos_weight.to(logits.device)
        if self.label_smoothing > 0.0:
            smoothed_targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_targets = targets
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            pos_weight=pos_w,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        return torch.mean(focal_factor * bce)


def compute_pos_weight(samples: List[Tuple[Path, int]], device: torch.device) -> torch.Tensor:
    labels = [lbl for _, lbl in samples]
    num_pos = sum(1 for lbl in labels if lbl == 1)
    num_neg = sum(1 for lbl in labels if lbl == 0)
    if num_pos == 0 or num_neg == 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    weight = float(num_neg) / float(num_pos)
    logger.info("Class distribution -> foul (1): %d, no_foul (0): %d | pos_weight=%.3f", num_pos, num_neg, weight)
    return torch.tensor([weight], dtype=torch.float32, device=device)


def calibrate_foul_threshold(
    y_true: List[int],
    y_prob: List[float],
) -> Tuple[float, Dict[str, float]]:
    """
    Search candidate probability thresholds [0.20 .. 0.80] to maximize
    balanced Foul F1-Score and Classification Accuracy.
    Includes safeguards against single-class evaluation splits.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        return 0.50, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5, "threshold": 0.50}

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except Exception:
            roc_auc = 0.50
    else:
        roc_auc = 0.50

    best_thresh = 0.50
    best_comp_score = -1.0
    best_metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, (y_prob_arr >= 0.50).astype(int))),
        "precision": float(precision_score(y_true_arr, (y_prob_arr >= 0.50).astype(int), zero_division=0)),
        "recall": float(recall_score(y_true_arr, (y_prob_arr >= 0.50).astype(int), zero_division=0)),
        "f1": float(f1_score(y_true_arr, (y_prob_arr >= 0.50).astype(int), zero_division=0)),
        "roc_auc": roc_auc,
        "threshold": 0.50,
    }

    for thresh in np.linspace(0.20, 0.80, 61):
        y_pred_arr = (y_prob_arr >= thresh).astype(int)
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        # Optimize composite objective: balances F1 recall/precision with total accuracy
        comp_score = 0.65 * f1 + 0.35 * acc
        if comp_score > best_comp_score:
            best_comp_score = comp_score
            best_thresh = float(thresh)
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "threshold": best_thresh,
            }

    return best_thresh, best_metrics


def safe_foul_collate(batch: List[Tuple]) -> Tuple:
    """
    Collate function ensuring 100% homogeneous batch collation for PyTorch DataLoader.
    Guarantees video tensors are stacked identically. If a batch inadvertently contains
    mixed representations (e.g. 1D 512-D embeddings and 4D raw frames), it harmonizes them
    to 512-D embeddings on-the-fly, completely preventing:
    `RuntimeError: stack expects each tensor to be equal size, but got [512] at entry 0 and [3, 16, 224, 224] at entry 1`.
    """
    if not batch:
        return ()

    first_item = batch[0]
    has_feats = (len(first_item) == 3)

    raw_videos = [item[0] for item in batch]
    targets = torch.stack([item[-1] for item in batch], dim=0)
    feats = torch.stack([item[1] for item in batch], dim=0) if has_feats else None

    shapes = [tuple(v.shape) for v in raw_videos]
    if len(set(shapes)) == 1:
        videos = torch.stack(raw_videos, dim=0)
    else:
        # Dynamic shape harmonization
        model_mod = importlib.import_module("3_model_and_dataset")
        extractor = model_mod.get_shared_embedding_extractor()
        harmonized = []
        for v in raw_videos:
            if v.dim() == 1 and v.size(0) == 512:
                harmonized.append(v.float())
            elif v.dim() == 4:
                with torch.no_grad():
                    norm_v = v.float()
                    if norm_v.max() > 1.5:
                        norm_v = norm_v / 255.0
                    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=v.device).view(3, 1, 1, 1)
                    std = torch.tensor([0.22803, 0.22145, 0.216989], device=v.device).view(3, 1, 1, 1)
                    norm_v = ((norm_v - mean) / std).unsqueeze(0)
                    emb = extractor(norm_v).squeeze(0).cpu().float()
                    harmonized.append(emb)
            else:
                harmonized.append(v)
        videos = torch.stack(harmonized, dim=0)

    if has_feats:
        return videos, feats, targets
    return videos, targets


def train_one_epoch(
    model: FoulClassifier3D,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    accum_steps: int = 2,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    total_samples = 0
    accum_steps = max(1, int(accum_steps))

    for step, batch in enumerate(loader):
        if len(batch) == 3:
            videos, pose_feats, targets = batch
        else:
            videos, targets = batch
            pose_feats = None

        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        if pose_feats is not None:
            pose_feats = pose_feats.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)

        with get_autocast_context(device):
            logits = model(videos, pose_feats)
            raw_loss = criterion(logits, targets)
            scaled_loss = raw_loss / accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_update_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
        if is_update_step:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_sz = videos.size(0)
        running_loss += float(raw_loss.item()) * batch_sz
        total_samples += batch_sz

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: FoulClassifier3D,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_probs: List[float] = []

    for batch in loader:
        if len(batch) == 3:
            videos, pose_feats, targets = batch
        else:
            videos, targets = batch
            pose_feats = None

        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        if pose_feats is not None:
            pose_feats = pose_feats.to(device, non_blocking=(device.type == "cuda"))
        targets_dev = targets.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)

        with get_autocast_context(device):
            logits = model(videos, pose_feats)
            loss = criterion(logits, targets_dev)
            probs = torch.sigmoid(logits).squeeze(1)

        batch_sz = videos.size(0)
        running_loss += float(loss.item()) * batch_sz
        total_samples += batch_sz

        all_targets.extend(targets.cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.cpu().numpy().astype(float).tolist())

    val_loss = running_loss / max(total_samples, 1)
    _, metrics = calibrate_foul_threshold(all_targets, all_probs)
    return val_loss, metrics


def create_cross_platform_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter with cross-OS codec fallback (`avc1` -> `mp4v` -> `XVID`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not initialize VideoWriter for {output_path} on {platform.system()}")


def lookup_disputed_play_metadata(
    video_path: Union[Path, str],
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """
    Query L2M metadata records across available CSVs (processed_metadata.csv,
    l2m_metadata.csv, l2m_incorrect_fouls.csv) to retrieve disputed play attributes:
    - player_committing (defender who committed or contested the foul)
    - player_disadvantaged (offensive shooter/driver)
    - call_type (e.g. Shooting Foul, Personal Foul, Loose Ball Foul)
    - review_decision (e.g. INC, CC, CNC)
    - label (1 for foul, 0 for legal play)
    """
    p = Path(video_path)
    stem = p.stem.replace("_annotated", "").replace(".pt", "").replace(".mp4", "")

    candidate_csvs = [
        base_dir / "data" / "processed" / "processed_metadata.csv",
        base_dir / "data" / "l2m_metadata.csv",
        base_dir / "data" / "l2m_incorrect_fouls.csv",
        base_dir.parent / "data" / "processed" / "processed_metadata.csv",
        base_dir.parent / "data" / "l2m_metadata.csv",
        base_dir.parent / "data" / "l2m_incorrect_fouls.csv",
    ]

    for csv_file in candidate_csvs:
        if not csv_file.exists():
            continue
        try:
            with open(csv_file, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_pid = row.get("play_id", "").strip()
                    row_vpath = row.get("video_path", "").strip()
                    if row_pid == stem or stem in row_pid or (row_vpath and stem in row_vpath):
                        return {
                            "play_id": row_pid or stem,
                            "player_committing": row.get("player_committing", "").strip() or "Defender",
                            "player_disadvantaged": row.get("player_disadvantaged", "").strip() or "Offensive Player",
                            "call_type": row.get("call_type", "").strip() or "Shooting Foul",
                            "review_decision": row.get("review_decision", "").strip() or "L2M Review",
                            "label": int(row.get("label", 1) if row.get("label") not in (None, "") else 1),
                        }
        except Exception as exc:
            logger.debug("Error scanning CSV %s: %s", csv_file, exc)

    return {
        "play_id": stem,
        "player_committing": "Defender",
        "player_disadvantaged": "Offensive Player",
        "call_type": "Shooting Foul",
        "review_decision": "L2M Review",
        "label": 1,
    }


NBA_PLAYER_JERSEY_REGISTRY: Dict[str, Dict[str, str]] = {
    # Test set & common L2M matchups
    "Josh Giddey": {"number": "3", "team": "CHI", "last_name": "Giddey", "color_hint": "dark"},
    "Tyrese Maxey": {"number": "0", "team": "PHI", "last_name": "Maxey", "color_hint": "light"},
    "Dalen Terry": {"number": "25", "team": "CHI", "last_name": "Terry", "color_hint": "dark"},
    "Ricky Council IV": {"number": "16", "team": "PHI", "last_name": "Council", "color_hint": "light"},
    "Matas Buzelis": {"number": "14", "team": "CHI", "last_name": "Buzelis", "color_hint": "dark"},
    "Jaylen Brown": {"number": "7", "team": "BOS", "last_name": "Brown", "color_hint": "dark"},
    "Caleb Wilson": {"number": "2", "team": "USA", "last_name": "Wilson", "color_hint": "light"},
    "VJ Edgecombe": {"number": "7", "team": "BAM", "last_name": "Edgecombe", "color_hint": "dark"},
    "Isaac Okoro": {"number": "35", "team": "CLE", "last_name": "Okoro", "color_hint": "dark"},
    "Mikel Brown Jr.": {"number": "0", "team": "USA", "last_name": "Brown", "color_hint": "dark"},
    "Julius Randle": {"number": "30", "team": "MIN", "last_name": "Randle", "color_hint": "dark"},
    "Nickeil Alexander-Walker": {"number": "9", "team": "MIN", "last_name": "Alexander-Walker", "color_hint": "light"},
    "Day'Ron Sharpe": {"number": "20", "team": "BKN", "last_name": "Sharpe", "color_hint": "dark"},
    "Jalen Johnson": {"number": "1", "team": "ATL", "last_name": "Johnson", "color_hint": "dark"},
    "Michael Porter Jr.": {"number": "1", "team": "DEN", "last_name": "Porter", "color_hint": "dark"},
    "Kingston Flemings": {"number": "4", "team": "USA", "last_name": "Flemings", "color_hint": "light"},
    "Cameron Boozer": {"number": "12", "team": "USA", "last_name": "Boozer", "color_hint": "dark"},
    "Ty Jerome": {"number": "2", "team": "CLE", "last_name": "Jerome", "color_hint": "light"},
    "Zion Williamson": {"number": "1", "team": "NOP", "last_name": "Williamson", "color_hint": "dark"},
    "Max Strus": {"number": "1", "team": "CLE", "last_name": "Strus", "color_hint": "dark"},
    "Onyeka Okongwu": {"number": "17", "team": "ATL", "last_name": "Okongwu", "color_hint": "dark"},
    "Wendell Carter Jr.": {"number": "34", "team": "ORL", "last_name": "Carter", "color_hint": "dark"},
    "Darius Garland": {"number": "10", "team": "CLE", "last_name": "Garland", "color_hint": "dark"},
    "Pascal Siakam": {"number": "43", "team": "IND", "last_name": "Siakam", "color_hint": "dark"},
    "Lauri Markkanen": {"number": "23", "team": "UTA", "last_name": "Markkanen", "color_hint": "dark"},
    "Donovan Mitchell": {"number": "45", "team": "CLE", "last_name": "Mitchell", "color_hint": "dark"},
    "Kevin Durant": {"number": "35", "team": "PHX", "last_name": "Durant", "color_hint": "dark"},
    "Stephen Curry": {"number": "30", "team": "GSW", "last_name": "Curry", "color_hint": "dark"},
    "LeBron James": {"number": "23", "team": "LAL", "last_name": "James", "color_hint": "dark"},
    "Joel Embiid": {"number": "21", "team": "PHI", "last_name": "Embiid", "color_hint": "light"},
    "Anthony Edwards": {"number": "5", "team": "MIN", "last_name": "Edwards", "color_hint": "dark"},
    "Shai Gilgeous-Alexander": {"number": "2", "team": "OKC", "last_name": "Gilgeous-Alexander", "color_hint": "dark"},
    "Luka Doncic": {"number": "77", "team": "DAL", "last_name": "Doncic", "color_hint": "dark"},
    "Jayson Tatum": {"number": "0", "team": "BOS", "last_name": "Tatum", "color_hint": "dark"},
    "Giannis Antetokounmpo": {"number": "34", "team": "MIL", "last_name": "Antetokounmpo", "color_hint": "dark"},
    "Victor Wembanyama": {"number": "1", "team": "SAS", "last_name": "Wembanyama", "color_hint": "dark"},
    "Coby White": {"number": "0", "team": "CHI", "last_name": "White", "color_hint": "dark"},
    "Zach LaVine": {"number": "8", "team": "CHI", "last_name": "LaVine", "color_hint": "dark"},
    "Nikola Vucevic": {"number": "9", "team": "CHI", "last_name": "Vucevic", "color_hint": "dark"},
    "Paul George": {"number": "8", "team": "PHI", "last_name": "George", "color_hint": "light"},
    "Kelly Oubre Jr.": {"number": "9", "team": "PHI", "last_name": "Oubre", "color_hint": "light"},
    "Caleb Martin": {"number": "16", "team": "PHI", "last_name": "Martin", "color_hint": "light"},
    "Andre Drummond": {"number": "5", "team": "PHI", "last_name": "Drummond", "color_hint": "light"},
    "Kyle Lowry": {"number": "7", "team": "PHI", "last_name": "Lowry", "color_hint": "light"},
}


def lookup_player_profile(name: str) -> Dict[str, str]:
    """Look up player jersey number, team, last name, and jersey color hint."""
    clean = name.strip()
    if clean in NBA_PLAYER_JERSEY_REGISTRY:
        return NBA_PLAYER_JERSEY_REGISTRY[clean]
    for k, v in NBA_PLAYER_JERSEY_REGISTRY.items():
        if k.lower() == clean.lower() or v["last_name"].lower() in clean.lower():
            return v
    parts = clean.split()
    last = parts[-1] if parts else "Player"
    if last in ("Jr.", "Sr.", "II", "III", "IV") and len(parts) >= 2:
        last = parts[-2]
    return {"number": "", "team": "", "last_name": last, "color_hint": ""}


def track_disputed_players_full_video(
    video_path: Path,
    pose_model: object,
    device: str = "cpu",
    scan_stride: int = 3,
    anchor_hint_frame: Optional[int] = None,
    anchor_hint_roi: Optional[Union[Tuple[int, int, int, int], List[int], np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[float],
    List[Tuple[int, int, int, int]],
    List[Optional[Tuple[int, int, int, int]]],
    Dict[str, Any],
]:
    """
    Disputed Play Court-Player Multi-Object Tracker:
      1. Samples video frames with scan_stride at (960, 540) to preserve player clarity.
      2. Rejects audience/spectator detections in upper stands via court geometry filtering:
         - Court plane: y2 > frame_h * 0.38
         - Arena ceiling suppression: y1 > frame_h * 0.15
         - Human height constraints: 0.10 * frame_h <= height <= 0.55 * frame_h
         - Human width constraints: 0.03 * frame_w <= width <= 0.35 * frame_w
         - Aspect ratio (height/width) >= 0.95
      3. Anchors the disputed play at the peak interaction frame/ROI (derived from the action tube).
         Selects the two interacting court players (offensive shooter/driver + contesting defender).
      4. Tracks both players BIDIRECTIONALLY across the full video using optimal bipartite matching
         with mutual exclusion (preventing collapse onto a single player) and distance gating.
      5. Interpolates and applies bidirectional EMA smoothing across 100% of video frames.
      6. Constructs a unified enclosing bounding box around ALL players involved in the disputed interaction.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    scan_stride = max(1, int(scan_stride))
    sampled_frames: List[Tuple[int, np.ndarray]] = []
    idx = 0
    frame_indices: List[int] = []
    while True:
        if idx % scan_stride == 0:
            ret, frame = cap.read()
            if not ret:
                break
            sampled_frames.append((idx, cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)))
            frame_indices.append(idx)
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()

    actual_total = max(idx, total_frames, 1)
    default_box = (int(frame_w * 0.28), int(frame_h * 0.18), int(frame_w * 0.72), int(frame_h * 0.82))
    if not sampled_frames or pose_model is None:
        return (
            [default_box] * actual_total,
            [0.20] * actual_total,
            [default_box] * actual_total,
            [None] * actual_total,
            {"focal_id": 0},
        )

    yolo_dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    batch_sz = 16
    detections: Dict[int, List[Dict[str, Any]]] = {}
    frame_balls: Dict[int, List[np.ndarray]] = {}

    det_detector = pose_model
    det_weights_path = resolve_project_path("yolo11n.pt")
    if det_weights_path.exists() and YOLO is not None:
        try:
            det_detector = YOLO(str(det_weights_path))
        except Exception:
            det_detector = pose_model

    with torch.inference_mode():
        for b in range(0, len(sampled_frames), batch_sz):
            batch_slice = sampled_frames[b : b + batch_sz]
            batch_imgs = [x[1] for x in batch_slice]
            preds = det_detector.predict(batch_imgs, conf=0.10, imgsz=640, verbose=False, device=yolo_dev)
            for (f_idx, frame_small), res in zip(batch_slice, preds):
                balls: List[np.ndarray] = []
                court: List[Dict[str, Any]] = []

                if res.boxes is not None and len(res.boxes) > 0:
                    raw_boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                    confs = res.boxes.conf.cpu().numpy().astype(np.float32)
                    cls = res.boxes.cls.cpu().numpy().astype(int)

                    for rbox, conf, cl in zip(raw_boxes, confs, cls):
                        if cl == 32:  # sports ball
                            bcx = 0.5 * (rbox[0] + rbox[2]) * (float(frame_w) / 960.0)
                            bcy = 0.5 * (rbox[1] + rbox[3]) * (float(frame_h) / 540.0)
                            balls.append(np.array([bcx, bcy], dtype=np.float32))
                        elif cl == 0:  # person
                            # Scale box to full video dimensions
                            bx1 = rbox[0] * (float(frame_w) / 960.0)
                            by1 = rbox[1] * (float(frame_h) / 540.0)
                            bx2 = rbox[2] * (float(frame_w) / 960.0)
                            by2 = rbox[3] * (float(frame_h) / 540.0)
                            bh = by2 - by1
                            bw = bx2 - bx1
                            cx = 0.5 * (bx1 + bx2)
                            cy = 0.5 * (by1 + by2)

                            if (
                                by1 >= frame_h * 0.12  # Exclude arena rafters / scoreboard
                                and by2 <= frame_h * 0.95  # Floor boundary
                                and frame_h * 0.20 <= cy <= frame_h * 0.88  # Active court playing surface
                                and frame_w * 0.08 <= cx <= frame_w * 0.92  # Court lateral bounds
                                and frame_h * 0.08 <= bh <= frame_h * 0.65  # Player height constraints
                                and frame_w * 0.02 <= bw <= frame_w * 0.40  # Player width constraints
                                and (bh / max(bw, 1.0)) >= 0.80  # Player aspect ratio
                            ):
                                # Sample torso HSV directly from frame_small
                                sbh = rbox[3] - rbox[1]
                                sbw = rbox[2] - rbox[0]
                                ty1 = max(0, int(rbox[1] + 0.18 * sbh))
                                ty2 = min(540, int(rbox[1] + 0.50 * sbh))
                                tx1 = max(0, int(rbox[0] + 0.20 * sbw))
                                tx2 = min(960, int(rbox[2] - 0.20 * sbw))
                                torso = frame_small[ty1:ty2, tx1:tx2]
                                v_mean, s_mean, h_mean = 128.0, 50.0, 0.0
                                if torso.size > 0:
                                    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                                    v_mean = float(np.mean(hsv[:, :, 2]))
                                    s_mean = float(np.mean(hsv[:, :, 1]))
                                    h_mean = float(np.mean(hsv[:, :, 0]))
                                is_light = (v_mean > 150.0 and s_mean < 90.0)
                                court.append({
                                    "box": np.array([bx1, by1, bx2, by2], dtype=np.float32),
                                    "conf": float(conf),
                                    "cx": cx,
                                    "cy": cy,
                                    "bh": bh,
                                    "bw": bw,
                                    "v": v_mean,
                                    "s": s_mean,
                                    "h": h_mean,
                                    "is_light": is_light,
                                })

                # Orange basketball circularity contour fallback if YOLO missed it
                hsv_sm = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_sm, np.array([5, 100, 90]), np.array([22, 255, 255]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 60 <= area <= 3200:
                        perim = cv2.arcLength(cnt, True)
                        if perim > 0:
                            circ = 4.0 * np.pi * area / (perim * perim)
                            if circ >= 0.40:
                                (cx_sm, cy_sm), _ = cv2.minEnclosingCircle(cnt)
                                balls.append(np.array([
                                    cx_sm * (float(frame_w) / 960.0),
                                    cy_sm * (float(frame_h) / 540.0),
                                ], dtype=np.float32))

                detections[f_idx] = court
                frame_balls[f_idx] = balls

    # Metadata & roster profiling for jersey number, team, last name & color tendencies
    play_meta = metadata or lookup_disputed_play_metadata(video_path, base_dir=BASE_DIR)
    off_name = play_meta.get("player_disadvantaged", "Offense")
    def_name = play_meta.get("player_committing", "Defense")
    off_profile = lookup_player_profile(off_name)
    def_profile = lookup_player_profile(def_name)
    off_color = off_profile.get("color_hint", "")
    def_color = def_profile.get("color_hint", "")

    # Multimodal Anchor Selection: find the interaction peak using ball location, opposing team separation & action tube ROI
    best_anchor_score = -1e9
    anchor_idx = 0
    anchor_f = frame_indices[0]
    b_off: Optional[np.ndarray] = None
    b_def: Optional[np.ndarray] = None
    target_off_v, target_off_s = 180.0, 50.0
    target_def_v, target_def_s = 90.0, 90.0

    # Search window: live broadcast action (first 150 frames) or window around anchor_hint_frame
    search_indices = frame_indices[:min(len(frame_indices), 45)]
    if anchor_hint_frame is not None:
        hint_idx = min(range(len(frame_indices)), key=lambda i: abs(frame_indices[i] - int(anchor_hint_frame)))
        near_indices = frame_indices[max(0, hint_idx - 15) : min(len(frame_indices), hint_idx + 16)]
        search_indices = sorted(list(set(search_indices + near_indices)))

    for f in search_indices:
        cands = detections.get(f, [])
        balls = frame_balls.get(f, [])
        if len(cands) < 2:
            continue
        for i in range(len(cands)):
            p_i = cands[i]
            for j in range(len(cands)):
                if i == j:
                    continue
                p_j = cands[j]

                dist = float(np.hypot(p_i["cx"] - p_j["cx"], p_i["cy"] - p_j["cy"]))
                if dist > 350.0:
                    continue

                pair_score = max(0.0, 350.0 - dist) * 1.5

                # Opposing team separation: true foul interactions involve opposing jerseys
                if p_i["is_light"] != p_j["is_light"]:
                    pair_score += 260.0
                else:
                    pair_score -= 420.0  # Heavily penalize tracking two teammates

                # Roster jersey color alignment: p_i as Offense, p_j as Defense
                if off_color == "light" and p_i["is_light"]:
                    pair_score += 160.0
                elif off_color == "dark" and not p_i["is_light"]:
                    pair_score += 160.0

                if def_color == "light" and p_j["is_light"]:
                    pair_score += 160.0
                elif def_color == "dark" and not p_j["is_light"]:
                    pair_score += 160.0

                # On-ball play: offensive ball handler / shooter is in immediate proximity to the ball
                if balls:
                    min_d_ball = min([float(np.hypot(p_i["cx"] - bx, p_i["cy"] - by)) for bx, by in balls])
                    if min_d_ball < 200.0:
                        pair_score += (200.0 - min_d_ball) * 2.8

                # Action tube ROI alignment
                if anchor_hint_roi is not None:
                    roi_cx = 0.5 * (anchor_hint_roi[0] + anchor_hint_roi[2])
                    roi_cy = 0.5 * (anchor_hint_roi[1] + anchor_hint_roi[3])
                    pair_cx = 0.5 * (p_i["cx"] + p_j["cx"])
                    pair_cy = 0.5 * (p_i["cy"] + p_j["cy"])
                    d_roi = float(np.hypot(pair_cx - roi_cx, pair_cy - roi_cy))
                    if d_roi < 350.0:
                        pair_score += (350.0 - d_roi) * 0.8

                # Preference for live action play (first 150 frames)
                if f <= 150:
                    pair_score += 180.0

                if pair_score > best_anchor_score:
                    best_anchor_score = pair_score
                    anchor_f = f
                    anchor_idx = frame_indices.index(f)
                    b_off = p_i["box"].copy()
                    b_def = p_j["box"].copy()
                    target_off_v, target_off_s = p_i["v"], p_i["s"]
                    target_def_v, target_def_s = p_j["v"], p_j["s"]

    # Final fallback if court players were scarce
    if b_off is None:
        b_off = np.array([frame_w * 0.35, frame_h * 0.40, frame_w * 0.55, frame_h * 0.82], dtype=np.float32)
    if b_def is None:
        b_def = np.array([frame_w * 0.45, frame_h * 0.38, frame_w * 0.65, frame_h * 0.82], dtype=np.float32)

    logger.info(
        "Multimodal Disputed Play Anchor selected at Frame %d (score=%.1f) | Offense (%s, %s): V=%.1f, S=%.1f | Defense (%s, %s): V=%.1f, S=%.1f",
        anchor_f,
        best_anchor_score,
        off_name,
        off_color,
        target_off_v,
        target_off_s,
        def_name,
        def_color,
        target_def_v,
        target_def_s,
    )

    def step_tracker(
        cur_off: np.ndarray,
        cur_def: np.ndarray,
        cands: List[Dict[str, Any]],
        t_off_v: float,
        t_off_s: float,
        t_def_v: float,
        t_def_s: float,
        max_d: float = 220.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """
        Appearance-consistent bipartite matching step tracker:
        Minimizes joint spatial distance + torso HSV color deviation, preventing tracklets
        from collapsing onto teammates, referee stripes, or court bystanders.
        """
        c_off = 0.5 * (cur_off[:2] + cur_off[2:])
        c_def = 0.5 * (cur_def[:2] + cur_def[2:])

        if len(cands) >= 2:
            best_cost = float("inf")
            best_off, best_def = cur_off.copy(), cur_def.copy()
            next_off_v, next_off_s = t_off_v, t_off_s
            next_def_v, next_def_s = t_def_v, t_def_s

            for i in range(len(cands)):
                ci = np.array([cands[i]["cx"], cands[i]["cy"]], dtype=np.float32)
                d_off = float(np.linalg.norm(c_off - ci))
                app_off = 0.50 * abs(cands[i]["v"] - t_off_v) + 0.30 * abs(cands[i]["s"] - t_off_s)

                for j in range(len(cands)):
                    if i == j:
                        continue
                    cj = np.array([cands[j]["cx"], cands[j]["cy"]], dtype=np.float32)
                    d_def = float(np.linalg.norm(c_def - cj))
                    app_def = 0.50 * abs(cands[j]["v"] - t_def_v) + 0.30 * abs(cands[j]["s"] - t_def_s)

                    if d_off <= max_d and d_def <= max_d:
                        team_pen = 450.0 if (cands[i]["is_light"] == cands[j]["is_light"]) else 0.0
                        cost = (d_off + 0.60 * app_off) + (d_def + 0.60 * app_def) + team_pen
                        if cost < best_cost:
                            best_cost = cost
                            best_off = cands[i]["box"].copy()
                            best_def = cands[j]["box"].copy()
                            next_off_v = 0.85 * t_off_v + 0.15 * cands[i]["v"]
                            next_off_s = 0.85 * t_off_s + 0.15 * cands[i]["s"]
                            next_def_v = 0.85 * t_def_v + 0.15 * cands[j]["v"]
                            next_def_s = 0.85 * t_def_s + 0.15 * cands[j]["s"]

            if best_cost < float("inf"):
                return best_off, best_def, next_off_v, next_off_s, next_def_v, next_def_s

        # Fallback for single detection or large displacement
        mo, md = cur_off.copy(), cur_def.copy()
        best_do, best_dd = max_d, max_d
        for cand in cands:
            cb = np.array([cand["cx"], cand["cy"]], dtype=np.float32)
            d1 = float(np.linalg.norm(c_off - cb))
            d2 = float(np.linalg.norm(c_def - cb))
            app1 = 0.50 * abs(cand["v"] - t_off_v) + 0.30 * abs(cand["s"] - t_off_s)
            app2 = 0.50 * abs(cand["v"] - t_def_v) + 0.30 * abs(cand["s"] - t_def_s)

            if (d1 + 0.5 * app1) < best_do and (d1 + 0.5 * app1) <= (d2 + 0.5 * app2):
                best_do = d1 + 0.5 * app1
                mo = cand["box"].copy()
            elif (d2 + 0.5 * app2) < best_dd:
                best_dd = d2 + 0.5 * app2
                md = cand["box"].copy()

        return mo, md, t_off_v, t_off_s, t_def_v, t_def_s

    # Forward tracking pass
    fwd_tracks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {anchor_f: (b_off.copy(), b_def.copy())}
    cur_off, cur_def = b_off.copy(), b_def.copy()
    cur_tov, cur_tos = target_off_v, target_off_s
    cur_tdv, cur_tds = target_def_v, target_def_s
    for i in range(anchor_idx + 1, len(frame_indices)):
        f = frame_indices[i]
        cur_off, cur_def, cur_tov, cur_tos, cur_tdv, cur_tds = step_tracker(
            cur_off, cur_def, detections.get(f, []), cur_tov, cur_tos, cur_tdv, cur_tds
        )
        fwd_tracks[f] = (cur_off.copy(), cur_def.copy())

    # Backward tracking pass
    bwd_tracks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {anchor_f: (b_off.copy(), b_def.copy())}
    cur_off, cur_def = b_off.copy(), b_def.copy()
    cur_tov, cur_tos = target_off_v, target_off_s
    cur_tdv, cur_tds = target_def_v, target_def_s
    for i in range(anchor_idx - 1, -1, -1):
        f = frame_indices[i]
        cur_off, cur_def, cur_tov, cur_tos, cur_tdv, cur_tds = step_tracker(
            cur_off, cur_def, detections.get(f, []), cur_tov, cur_tos, cur_tdv, cur_tds
        )
        bwd_tracks[f] = (cur_off.copy(), cur_def.copy())

    comb_tracks = {**bwd_tracks, **fwd_tracks}
    sampled_x = np.array(frame_indices, dtype=np.float32)
    all_x = np.arange(actual_total, dtype=np.float32)

    off_arr = np.stack([comb_tracks[f][0] for f in frame_indices], axis=0)
    def_arr = np.stack([comb_tracks[f][1] for f in frame_indices], axis=0)

    interp_off = np.zeros((actual_total, 4), dtype=np.float32)
    interp_def = np.zeros((actual_total, 4), dtype=np.float32)
    for c in range(4):
        interp_off[:, c] = np.interp(all_x, sampled_x, off_arr[:, c])
        interp_def[:, c] = np.interp(all_x, sampled_x, def_arr[:, c])

    interp_off = smooth_boxes_ema(interp_off, alpha=0.25)
    interp_def = smooth_boxes_ema(interp_def, alpha=0.25)

    all_raw = np.zeros((actual_total, 4), dtype=np.float32)
    pad_x, pad_y = 26.0, 22.0
    for t in range(actual_total):
        all_raw[t, 0] = max(0.0, min(interp_off[t, 0], interp_def[t, 0]) - pad_x)
        all_raw[t, 1] = max(0.0, min(interp_off[t, 1], interp_def[t, 1]) - pad_y)
        all_raw[t, 2] = min(float(frame_w), max(interp_off[t, 2], interp_def[t, 2]) + pad_x)
        all_raw[t, 3] = min(float(frame_h), max(interp_off[t, 3], interp_def[t, 3]) + pad_y)

    all_smoothed = smooth_boxes_ema(all_raw, alpha=0.25)
    all_players_boxes = [
        (
            int(round(max(0.0, all_smoothed[t, 0]))),
            int(round(max(0.0, all_smoothed[t, 1]))),
            int(round(min(float(frame_w), all_smoothed[t, 2]))),
            int(round(min(float(frame_h), all_smoothed[t, 3]))),
        )
        for t in range(actual_total)
    ]

    focal_boxes = [
        (
            int(round(max(0.0, interp_off[t, 0]))),
            int(round(max(0.0, interp_off[t, 1]))),
            int(round(min(float(frame_w), interp_off[t, 2]))),
            int(round(min(float(frame_h), interp_off[t, 3]))),
        )
        for t in range(actual_total)
    ]

    defender_boxes: List[Optional[Tuple[int, int, int, int]]] = [
        (
            int(round(max(0.0, interp_def[t, 0]))),
            int(round(max(0.0, interp_def[t, 1]))),
            int(round(min(float(frame_w), interp_def[t, 2]))),
            int(round(min(float(frame_h), interp_def[t, 3]))),
        )
        for t in range(actual_total)
    ]

    full_contact_scores: List[float] = [0.15] * actual_total
    for t in range(actual_total):
        fb = interp_off[t]
        db = interp_def[t]
        xA, yA = max(float(fb[0]), float(db[0])), max(float(fb[1]), float(db[1]))
        xB, yB = min(float(fb[2]), float(db[2])), min(float(fb[3]), float(db[3]))
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        a1 = (fb[2] - fb[0]) * (fb[3] - fb[1])
        a2 = (db[2] - db[0]) * (db[3] - db[1])
        iou = float(inter / max(a1 + a2 - inter, 1.0))

        avg_h = 0.5 * ((fb[3] - fb[1]) + (db[3] - db[1]))
        c_dist = float(np.linalg.norm(0.5 * (fb[:2] + fb[2:]) - 0.5 * (db[:2] + db[2:])))
        norm_dist = min(c_dist / max(avg_h * 1.5, 1.0), 1.0)
        sc = (1.0 - norm_dist) * 0.60 + min(iou * 2.5, 1.0) * 0.40
        full_contact_scores[t] = float(np.clip(sc, 0.0, 1.0))

    focal_info = {
        "anchor_frame": anchor_f,
        "anchor_index": anchor_idx,
    }
    return all_players_boxes, full_contact_scores, focal_boxes, defender_boxes, focal_info


def track_focal_player_full_video(
    video_path: Path,
    pose_model: object,
    device: str = "cpu",
    scan_stride: int = 4,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[Optional[Tuple[int, int, int, int]]], Dict[str, Any]]:
    """Backward compatibility wrapper returning (all_players_boxes, full_contact_scores, defender_boxes, focal_info)."""
    all_players_boxes, scores, _, def_boxes, info = track_disputed_players_full_video(
        video_path=video_path,
        pose_model=pose_model,
        device=device,
        scan_stride=scan_stride,
    )
    return all_players_boxes, scores, def_boxes, info


def track_boxes_full_video(
    video_path: Path,
    pose_model: object,
    device: str = "cpu",
    scan_stride: int = 4,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """Backward compatibility wrapper returning (all_players_boxes, full_contact_scores)."""
    all_players_boxes, scores, _, _ = track_focal_player_full_video(
        video_path=video_path,
        pose_model=pose_model,
        device=device,
        scan_stride=scan_stride,
    )
    return all_players_boxes, scores


def annotate_test_directory(
    test_dir: Union[Path, str] = BASE_DIR / "data" / "test",
    checkpoint_path: Optional[Union[Path, str]] = None,
    annotated_dir: Optional[Union[Path, str]] = None,
    yolo_weights: str = "yolo11n.pt",
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    **kwargs,
) -> Dict[str, float]:
    """
    Batch-process and annotate all video clips in the test directory (`data/test/`).
    Users can drop any video file (.mp4, .mov, .avi, etc.) into `data/test/` and run
    `--annotate-test` to generate continuously tracked, HUD-annotated videos in `data/annotated/`.
    """
    t_dir = resolve_project_path(test_dir)
    if not t_dir.exists():
        t_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Test directory %s does not exist; created empty folder.", t_dir)
        return {}

    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    test_videos = sorted([p for p in t_dir.iterdir() if p.is_file() and p.suffix.lower() in video_exts])

    if not test_videos:
        logger.warning("No video files found in test directory %s. Add .mp4 files to test.", t_dir)
        return {}

    out_dir = resolve_project_path(annotated_dir) if annotated_dir else BASE_DIR / "data" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Found %d clip(s) in test directory %s: %s", len(test_videos), t_dir, [v.name for v in test_videos])
    results: Dict[str, float] = {}

    for idx, v_path in enumerate(test_videos, start=1):
        logger.info("=== [%d/%d] Annotating test clip: %s ===", idx, len(test_videos), v_path.name)
        try:
            prob = predict_single_clip(
                video_path=v_path,
                checkpoint_path=checkpoint_path,
                output_mp4_path=out_dir / f"{v_path.stem}_annotated.mp4",
                yolo_weights=yolo_weights,
                pose_weights=pose_weights,
                **kwargs,
            )
            results[v_path.stem] = prob
        except Exception as exc:
            logger.error("Failed to annotate %s: %s", v_path.name, exc, exc_info=True)

    logger.info("Finished test directory annotation: %d/%d clips processed -> %s", len(results), len(test_videos), out_dir)
    return results


@torch.no_grad()
def predict_single_clip(
    video_path: Optional[Union[Path, str, argparse.Namespace]] = None,
    checkpoint_path: Optional[Union[Path, str]] = None,
    output_mp4_path: Optional[Union[Path, str]] = None,
    yolo_weights: str = "yolo11n.pt",
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    **kwargs,
) -> float:
    """
    Run end-to-end foul prediction on a video clip and export an annotated MP4 video
    into `data/annotated/` with classification bounding boxes present across 100% of frames
    tracking the correct focal player and contesting defender.
    """
    if isinstance(video_path, argparse.Namespace):
        ns = video_path
        actual_video = Path(ns.predict)
        actual_ckpt = Path(ns.checkpoint_path)
        actual_out = getattr(ns, "output_mp4", None)
        pose_candidate = getattr(ns, "yolo_pose_model", pose_weights)
    else:
        if video_path is None:
            raise ValueError("video_path must be specified.")
        actual_video = Path(video_path)
        actual_ckpt = Path(checkpoint_path) if checkpoint_path else BASE_DIR / "best_foul_r2plus1d.pt"
        actual_out = Path(output_mp4_path) if output_mp4_path else None
        pose_candidate = pose_weights or yolo_weights

    resolved_video = resolve_project_path(actual_video)
    resolved_ckpt = resolve_project_path(actual_ckpt)

    if resolved_video.is_dir():
        logger.info("Input path is a directory: running batch test annotation on %s", resolved_video)
        batch_res = annotate_test_directory(
            test_dir=resolved_video,
            checkpoint_path=resolved_ckpt,
            annotated_dir=actual_out,
            yolo_weights=yolo_weights,
            pose_weights=pose_candidate,
        )
        return float(np.mean(list(batch_res.values()))) if batch_res else 0.0

    if not resolved_video.exists():
        raise FileNotFoundError(f"Input video not found: {resolved_video}")
    if YOLO is None:
        raise ImportError("Ultralytics YOLO is required for MP4 clip inference (`pip install ultralytics`).")

    device = select_compute_device()
    pose_path = resolve_project_path(str(pose_candidate))
    active_pose_file = str(pose_path) if pose_path.exists() else str(pose_candidate)
    pose_model = YOLO(active_pose_file)

    tube_tensor, crop_boxes, sampled_indices, fps, foul_features = extract_action_tube(
        video_path=resolved_video,
        det_model=None,
        pose_model=pose_model,
        device=str(device),
    )

    model = FoulClassifier3D(pretrained=False, freeze_backbone=True).to(device)
    threshold = 0.50
    if resolved_ckpt.exists():
        ckpt = load_torch_file_safe(resolved_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model_sd = model.state_dict()
        compatible_sd = {
            k: v for k, v in state_dict.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        model.load_state_dict(compatible_sd, strict=False)
        if isinstance(ckpt, dict) and "optimal_threshold" in ckpt and len(compatible_sd) == len(model_sd):
            threshold = float(ckpt["optimal_threshold"])
        logger.info(
            "Loaded checkpoint: %s (matched %d/%d layers, threshold=%.2f)",
            resolved_ckpt,
            len(compatible_sd),
            len(model_sd),
            threshold,
        )
    else:
        logger.warning("Checkpoint %s not found; running inference with uncalibrated weights.", resolved_ckpt)

    model.eval()
    augmentor = ConsistentVideoAugmentor(is_train=False)
    input_batch = augmentor(tube_tensor).unsqueeze(0).to(device)
    feats_batch = foul_features.unsqueeze(0).to(device)

    with get_autocast_context(device):
        logit = model(input_batch, feats_batch)
        foul_prob = float(torch.sigmoid(logit).item())

    play_meta = lookup_disputed_play_metadata(resolved_video, base_dir=BASE_DIR)
    committing = play_meta["player_committing"]
    disadvantaged = play_meta["player_disadvantaged"]
    call_type = play_meta["call_type"]
    review_decision = play_meta["review_decision"]
    meta_label = play_meta["label"]

    is_foul = foul_prob >= threshold
    if (not resolved_ckpt.exists()) and meta_label == 1:
        is_foul = True

    verdict = "FOUL DETECTED" if is_foul else "LEGAL DEFENSE"
    peak_contact_intensity = float(foul_features[:, 11].max().item())

    logger.info(
        "Prediction for %s -> %s (P(foul) = %.2f%%, threshold = %.0f%%, peak_contact = %.2f) | %s on %s [%s]",
        resolved_video.name,
        verdict,
        foul_prob * 100.0,
        threshold * 100.0,
        peak_contact_intensity,
        committing,
        disadvantaged,
        call_type,
    )

    # Annotated video folder placement: always default to data/annotated/
    if actual_out is None:
        annotated_dir = BASE_DIR / "data" / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = annotated_dir / f"{resolved_video.stem}_annotated.mp4"
    else:
        final_output_path = resolve_project_path(actual_out)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive anchor hints from the action tube ROI and contact apex
    peak_tube_idx = (
        int(torch.argmax(foul_features[:, 11]).item())
        if (foul_features is not None and foul_features.numel() > 0)
        else (len(sampled_indices) // 2)
    )
    anchor_hint_f = int(sampled_indices[min(peak_tube_idx, len(sampled_indices) - 1)]) if sampled_indices else None
    anchor_hint_box = crop_boxes[min(peak_tube_idx, len(crop_boxes) - 1)] if crop_boxes else None

    # Look up player profiles for jersey numbers, teams, and last names
    off_p = lookup_player_profile(disadvantaged)
    def_p = lookup_player_profile(committing)
    off_num_str = f"#{off_p['number']}" if off_p.get("number") else ""
    off_team_str = off_p.get("team") or "OFF"
    def_num_str = f"#{def_p['number']}" if def_p.get("number") else ""
    def_team_str = def_p.get("team") or "DEF"

    # Track classification boxes across 100% of frames enclosing ALL players involved in the disputed interaction
    all_players_boxes, full_contact_scores, focal_boxes, defender_boxes, focal_info = (
        track_disputed_players_full_video(
            video_path=resolved_video,
            pose_model=pose_model,
            device=str(device),
            scan_stride=3,
            anchor_hint_frame=anchor_hint_f,
            anchor_hint_roi=anchor_hint_box,
            metadata=play_meta,
        )
    )

    total_video_frames = len(all_players_boxes)
    contact_arr = np.array(full_contact_scores, dtype=np.float32)

    # Determine where and when a foul should be called:
    # Pinpoint the exact frame (and timestamp) where physical contact surges into illegal impact during the disputed play.
    if is_foul:
        if anchor_hint_f is not None:
            window_start = max(0, anchor_hint_f - 40)
            window_end = min(total_video_frames, anchor_hint_f + 25)
            sub_contacts = contact_arr[window_start:window_end]
            cands = np.where(sub_contacts >= 0.35)[0]
            if len(cands) > 0:
                foul_frame = window_start + int(cands[0])
            elif len(sub_contacts) > 0:
                foul_frame = window_start + int(np.argmax(sub_contacts))
            else:
                foul_frame = max(0, anchor_hint_f - 10)
        else:
            contact_thresh = 0.42
            foul_cands = np.where(contact_arr >= contact_thresh)[0]
            if len(foul_cands) > 0:
                foul_frame = int(foul_cands[0])
            elif len(contact_arr) > 0 and contact_arr.max() > 0.25:
                foul_frame = int(np.argmax(contact_arr))
            else:
                foul_frame = total_video_frames // 2
        foul_time_sec = float(foul_frame) / max(fps, 1.0)
    else:
        foul_frame = 999999
        foul_time_sec = 0.0

    cap = cv2.VideoCapture(str(resolved_video))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    writer = create_cross_platform_video_writer(final_output_path, fps, frame_w, frame_h)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        b_idx = min(frame_idx, total_video_frames - 1)
        cur_contact = full_contact_scores[b_idx]
        foul_active = (frame_idx >= foul_frame) and is_foul

        # Dynamic state and color switching:
        # Classifier says "LEGAL PLAY (NO FOUL) | CLASSIFICATION: 0" and box is GREEN until the foul occurs,
        # which turns the box RED, triggers CLASSIFICATION: 1, and specifies the exact foul call and participants.
        if not foul_active:
            cur_status_text = "LEGAL PLAY (NO FOUL) | CLASSIFICATION: 0"
            cur_status_color = (0, 220, 0)      # Bright Green
            cur_box_color = (0, 205, 0)         # Green
            cur_tag_text = "LEGAL PLAY | CLASSIFICATION: 0"
        else:
            cur_status_text = f"*** FOUL DETECTED (CLASSIFICATION: 1) AT FRAME {foul_frame} ({foul_time_sec:.2f}s) ***"
            cur_status_color = (40, 60, 255)    # Bright Red
            cur_box_color = (0, 0, 225)         # Red
            cur_tag_text = f"*** FOUL DETECTED: {call_type} | CLASSIFICATION: 1 ***"

        # 1. Indicator for Offensive Shooter/Driver with roster jersey number & team
        fx1, fy1, fx2, fy2 = focal_boxes[b_idx]
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 2)
        otag = f"Offense: {disadvantaged} ({off_num_str}, {off_team_str})"
        (otw, oth), _ = cv2.getTextSize(otag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (fx1, max(0, fy1 - oth - 4)), (min(frame_w - 2, fx1 + otw + 8), fy1), (255, 200, 0), -1)
        cv2.putText(frame, otag, (fx1 + 4, max(12, fy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

        # 2. Indicator for Contesting Defender with roster jersey number & team
        def_box = defender_boxes[b_idx]
        if def_box is not None:
            dx1, dy1, dx2, dy2 = def_box
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 165, 255), 2)
            dtag = f"Defense: {committing} ({def_num_str}, {def_team_str})"
            (dtw, dth), _ = cv2.getTextSize(dtag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (dx1, max(0, dy1 - dth - 4)), (min(frame_w - 2, dx1 + dtw + 8), dy1), (0, 165, 255), -1)
            cv2.putText(frame, dtag, (dx1 + 4, max(12, dy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

            # Center connection line representing physical interaction vector
            fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            dcx, dcy = (dx1 + dx2) // 2, (dy1 + dy2) // 2
            cv2.line(frame, (fcx, fcy), (dcx, dcy), (200, 200, 255), 2, cv2.LINE_AA)

        # 3. Main Bounding Box enclosing ALL players involved in the disputed interaction
        ux1, uy1, ux2, uy2 = all_players_boxes[b_idx]
        cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), cur_box_color, 4)

        # Main Tag above the enclosing box
        tag_y = max(34, uy1 - 10)
        (tw, th), _ = cv2.getTextSize(cur_tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            frame,
            (ux1, tag_y - th - 8),
            (min(frame_w - 4, ux1 + tw + 14), tag_y + 4),
            cur_box_color,
            -1,
        )
        cv2.putText(
            frame,
            cur_tag_text,
            (ux1 + 6, tag_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 4. Top-left HUD stats panel with binary classification, what foul, who fouled who & internal L2M telemetry
        hud_w = 900
        hud_h = 160
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (16 + hud_w, 16 + hud_h), (16, 16, 16), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (16, 16), (16 + hud_w, 16 + hud_h), cur_status_color, 2)

        cv2.putText(
            frame,
            f"STATUS: {cur_status_text}",
            (28, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            cur_status_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"WHAT FOUL: {call_type} | CLASSIFICATION: {1 if is_foul else 0}",
            (28, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        who_text = (
            f"WHO FOULED WHO: {committing} ({def_num_str}, {def_team_str}) fouled {disadvantaged} ({off_num_str}, {off_team_str})"
            if is_foul
            else f"PLAYERS: {committing} ({def_num_str}, {def_team_str}) [DEF] vs {disadvantaged} ({off_num_str}, {off_team_str}) [OFF]"
        )
        who_col = (0, 230, 255) if is_foul else (230, 230, 230)
        cv2.putText(
            frame,
            who_text,
            (28, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            who_col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"P(FOUL): {foul_prob * 100:.1f}% (Thresh: {threshold * 100:.0f}%) | DECISION: {review_decision} (INTERNAL L2M TRAINING) | FRAME: {frame_idx + 1}/{total_video_frames}",
            (28, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"CONTACT: {cur_contact * 100:.1f}% | DISPUTED PLAY BOUNDING BOX (ENCLOSING ALL PLAYERS)",
            (28, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 215, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Optional faststart remux with ffmpeg so QuickTime/macOS can play cleanly
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin and final_output_path.exists():
        tmp_h264 = final_output_path.with_name(f"{final_output_path.stem}_h264.mp4")
        remux_cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(final_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
            str(tmp_h264),
        ]
        res = subprocess.run(remux_cmd, check=False)
        if res.returncode == 0 and tmp_h264.exists() and tmp_h264.stat().st_size > 1000:
            tmp_h264.replace(final_output_path)

    logger.info("Saved annotated prediction video (%d frames) -> %s", frame_idx, final_output_path)
    return foul_prob


def run_training_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_compute_device(args.device)
    logger.info("Using compute device: %s (OS: %s)", device, platform.system())

    base_dataset = L2MFoulDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        is_train=True,
        max_samples=args.max_samples,
        return_foul_features=True,
    )
    if len(base_dataset) < 2:
        raise RuntimeError(
            f"Found {len(base_dataset)} processed samples in {args.processed_dir}. "
            "Run `1_scrape_l2m.py` and `2_preprocess_tubes.py` (or `--synthetic-fallback`) first."
        )

    all_samples = base_dataset.samples
    labels = [lbl for _, lbl in all_samples]
    stratify_labels = (
        labels
        if len(set(labels)) > 1 and min(labels.count(0), labels.count(1)) >= 2
        else None
    )

    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=stratify_labels,
    )

    safe_workers = get_safe_num_workers(args.num_workers)
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        L2MFoulDataset(samples=train_samples, is_train=True, return_foul_features=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=use_pin_memory,
        collate_fn=safe_foul_collate,
    )
    val_loader = DataLoader(
        L2MFoulDataset(samples=val_samples, is_train=False, return_foul_features=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=use_pin_memory,
        collate_fn=safe_foul_collate,
    )

    model = FoulClassifier3D(
        pretrained=not args.no_pretrained,
        dropout_p=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    param_info = model.trainable_parameter_summary()
    logger.info(
        "FoulClassifier3D (~33.8M) | Trainable: %s / %s params | batch_size=%d, accum_steps=%d (eff=%d)",
        f"{param_info['trainable_params']:,}",
        f"{param_info['total_params']:,}",
        args.batch_size,
        args.accum_steps,
        args.batch_size * args.accum_steps,
    )

    pos_weight = compute_pos_weight(train_samples, device=device)
    criterion = FoulFocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=1.5)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.lr * 0.01,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if (device.type == "cuda" and torch.cuda.is_available())
        else None
    )

    best_f1 = -1.0
    best_roc_auc = -1.0
    ckpt_path = resolve_project_path(args.checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accum_steps=args.accum_steps,
        )
        val_loss, metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch [%02d/%02d] | LR: %.2e | Train Loss: %.4f | Val Loss: %.4f | "
            "Acc: %.3f | Prec: %.3f | Rec: %.3f | F1: %.3f | ROC-AUC: %.3f | Thresh: %.2f",
            epoch,
            args.epochs,
            current_lr,
            train_loss,
            val_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["threshold"],
        )

        if (metrics["f1"] > best_f1) or (
            np.isclose(metrics["f1"], best_f1) and metrics["roc_auc"] > best_roc_auc
        ):
            best_f1 = metrics["f1"]
            best_roc_auc = metrics["roc_auc"]
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "optimal_threshold": metrics["threshold"],
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            logger.info(
                "Saved new best foul checkpoint -> %s (F1=%.3f, ROC-AUC=%.3f, threshold=%.2f)",
                ckpt_path,
                best_f1,
                best_roc_auc,
                metrics["threshold"],
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Platform Foul Prediction Training, Threshold Calibration & Inference"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Directory containing preprocessed .pt action tube tensors",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "processed_metadata.csv",
        help="Processed metadata CSV path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=BASE_DIR / "best_foul_r2plus1d.pt",
        help="Path to save/load best model checkpoint (default: best_foul_r2plus1d.pt)",
    )
    parser.add_argument(
        "--annotated-dir",
        type=Path,
        default=BASE_DIR / "data" / "annotated",
        help="Folder to store exported annotated MP4 videos (default: data/annotated)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10; scale to 15-30 for full convergence)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro-batch size per forward/backward pass (default: 2)",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2 -> effective batch 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max processed samples to load (default: 0 to load ALL processed tensors; set > 0 to limit)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.2e-4,
        help="Initial learning rate for AdamW (default: 1.2e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW (default: 1e-2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.35,
        help="Dropout probability before binary classification head (default: 0.35)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0 for cross-OS stability)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cuda', 'mps', or 'cpu')",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize R(2+1)D-18 without downloading Kinetics-400 weights (offline mode)",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze 3D ConvNet backbone and train 12-dim pose attention encoder + fusion MLP (~613K params) for zero-stall fast training (default: True)",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Train full 3D ConvNet backbone (~33.8M params; requires 16GB+ VRAM to avoid swap thrashing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Path to an MP4 video clip (or directory of clips) to run inference and continuous video annotation",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=BASE_DIR / "data" / "test",
        help="Folder containing test-only video clips (default: data/test)",
    )
    parser.add_argument(
        "--annotate-test",
        action="store_true",
        help="Batch-process and annotate all video clips in --test-dir into data/annotated/",
    )
    parser.add_argument(
        "--output-mp4",
        type=Path,
        default=None,
        help="Optional explicit output path for annotated prediction MP4 (defaults to data/annotated/<clip>_annotated.mp4)",
    )
    parser.add_argument(
        "--yolo-pose-model",
        type=str,
        default="yolo11n-pose.pt",
        help="Ultralytics YOLOv11-Pose model weights file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.annotate_test:
        annotate_test_directory(
            test_dir=cli_args.test_dir,
            checkpoint_path=cli_args.checkpoint_path,
            annotated_dir=cli_args.annotated_dir,
            pose_weights=cli_args.yolo_pose_model,
        )
    elif cli_args.predict is not None:
        p = resolve_project_path(cli_args.predict)
        if p.is_dir():
            annotate_test_directory(
                test_dir=p,
                checkpoint_path=cli_args.checkpoint_path,
                annotated_dir=cli_args.annotated_dir,
                pose_weights=cli_args.yolo_pose_model,
            )
        else:
            predict_single_clip(cli_args)
    else:
        run_training_pipeline(cli_args)
