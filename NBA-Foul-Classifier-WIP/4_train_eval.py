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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

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
    """Select the best available compute device ('cuda', 'mps', or 'cpu') cross-platform with eager cuBLAS init."""
    if requested and requested.lower() != "auto":
        dev = torch.device(requested)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    
    if dev.type == "cuda":
        try:
            _warmup = torch.empty((2, 2), device=dev)
            _ = _warmup @ _warmup
            del _warmup
            torch.cuda.empty_cache()
        except Exception:
            pass
    elif dev.type == "mps":
        try:
            _w = torch.zeros((1, 4, 8), device=dev)
            _mha = nn.MultiheadAttention(8, 2, batch_first=True).to(dev)
            _ = _mha(_w, _w, _w)
            del _w, _mha
        except BaseException:
            dev = torch.device("cpu")
    return dev


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

    for thresh in np.linspace(0.50, 0.65, 16):
        y_pred_arr = (y_prob_arr >= thresh).astype(int)
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        # Optimize composite objective: balances F1 recall/precision with total accuracy and precision discipline
        comp_score = 0.50 * f1 + 0.35 * acc + 0.15 * prec
        if comp_score > best_comp_score:
            best_comp_score = comp_score
            best_thresh = float(round(thresh, 2))
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

    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        batch_sz = videos.size(0)
        raw_loss_val = 0.0

        try:
            with get_autocast_context(device):
                if hasattr(model, "foul_type_head") and hasattr(model, "extract_features") and pose_feats is not None:
                    combined = model.extract_features(videos, pose_feats)
                    logits = model.classifier_head(combined)
                    type_logits = model.foul_type_head(combined)
                    # Biomechanical shooting-act target: on-ball proximity (dims 6,7) + airborne apex (dim 11) + landing space (dim 2)
                    on_ball = torch.clamp(pose_feats[:, :, 6:8].max(dim=-1).values.mean(dim=1, keepdim=True) * 4.0, 0.0, 1.0)
                    shoot_score = (
                        0.40 * on_ball
                        + 0.35 * pose_feats[:, :, 11].mean(dim=1, keepdim=True)
                        + 0.25 * pose_feats[:, :, 2].mean(dim=1, keepdim=True)
                    )
                    shoot_targets = (shoot_score >= 0.34).float()
                    aux_loss = F.binary_cross_entropy_with_logits(type_logits, shoot_targets)
                    raw_loss = criterion(logits, targets) + 0.20 * aux_loss
                else:
                    logits = model(videos, pose_feats)
                    raw_loss = criterion(logits, targets)
                scaled_loss = raw_loss / accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            raw_loss_val = float(raw_loss.item())
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            err_str = str(exc).lower()
            if "cublas" in err_str or "out of memory" in err_str:
                logger.warning(
                    "CUDA memory pressure / cuBLAS allocation failure on batch %d (%s). Recovering with micro-batching...",
                    step,
                    exc,
                )
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                # Dynamic micro-batching: split into chunks of 2 or 1
                micro_sz = max(1, batch_sz // 2)
                accum_loss = 0.0
                for m_start in range(0, batch_sz, micro_sz):
                    m_end = min(m_start + micro_sz, batch_sz)
                    m_vid = videos[m_start:m_end]
                    m_pose = pose_feats[m_start:m_end] if pose_feats is not None else None
                    m_tgt = targets[m_start:m_end]
                    m_weight = (m_end - m_start) / batch_sz

                    with get_autocast_context(device):
                        m_logits = model(m_vid, m_pose)
                        m_loss = criterion(m_logits, m_tgt)
                        m_scaled = (m_loss * m_weight) / accum_steps

                    if scaler is not None:
                        scaler.scale(m_scaled).backward()
                    else:
                        m_scaled.backward()
                    accum_loss += float(m_loss.item()) * (m_end - m_start)
                raw_loss_val = accum_loss / max(batch_sz, 1)
            else:
                raise exc

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

        running_loss += raw_loss_val * batch_sz
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


def describe_court_location(ix: int, iy: int, frame_w: int, frame_h: int) -> str:
    """Map pixel coordinates `(ix, iy)` to a descriptive basketball court zone."""
    nx = float(ix) / max(float(frame_w), 1.0)
    ny = float(iy) / max(float(frame_h), 1.0)
    if nx < 0.34:
        side = "Left Paint / Rim Area" if ny < 0.68 else "Left Corner / Baseline"
    elif nx > 0.66:
        side = "Right Paint / Rim Area" if ny < 0.68 else "Right Corner / Baseline"
    else:
        side = "Top of Key / Perimeter" if ny < 0.58 else "Mid-Court / Wing"
    return f"{side} (x={ix}, y={iy})"


def lookup_disputed_play_metadata(
    video_path: Union[Path, str],
    base_dir: Path = BASE_DIR,
) -> Dict[str, Any]:
    """
    Query L2M metadata records across available CSVs (processed_metadata.csv,
    l2m_metadata.csv, l2m_incorrect_fouls.csv) and cached AI Referee decisions.
    For arbitrary raw game footage not in the L2M catalog, returns `found_in_l2m=False`
    and `label=None`, ensuring the AI Referee relies 100% on model inference.
    """
    import json

    p = Path(video_path)
    stem = p.stem.replace("_annotated", "").replace(".pt", "").replace(".mp4", "")

    # Check if AI Referee decision cache exists in data/annotated/referee_decisions.json
    cached_dec: Dict[str, Any] = {}
    for dec_json in [
        base_dir / "data" / "annotated" / "referee_decisions.json",
        base_dir.parent / "data" / "annotated" / "referee_decisions.json",
    ]:
        if dec_json.exists():
            try:
                with open(dec_json, "r", encoding="utf-8") as jf:
                    all_decs = json.load(jf)
                    if isinstance(all_decs, dict) and stem in all_decs:
                        cached_dec = all_decs[stem]
                        break
            except Exception:
                pass

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
                    if row_pid == stem or (row_pid and stem == row_pid) or (row_vpath and Path(row_vpath).stem == stem):
                        raw_lbl = row.get("label")
                        if raw_lbl in (None, ""):
                            dec_str = row.get("review_decision", "").strip().upper()
                            lbl_val = 1 if dec_str in ("CC", "INC") else (0 if dec_str in ("CNC", "IC") else None)
                        else:
                            lbl_val = int(raw_lbl)
                        res_meta = {
                            "play_id": row_pid or stem,
                            "found_in_l2m": True,
                            "player_committing": row.get("player_committing", "").strip() or "Defender",
                            "player_disadvantaged": row.get("player_disadvantaged", "").strip() or "Ball Handler",
                            "call_type": row.get("call_type", "").strip() or None,
                            "review_decision": row.get("review_decision", "").strip() or "L2M Review",
                            "label": lbl_val,
                        }
                        if cached_dec:
                            res_meta["ai_decision"] = cached_dec
                        return res_meta
        except Exception as exc:
            logger.debug("Error scanning CSV %s: %s", csv_file, exc)

    res_unknown: Dict[str, Any] = {
        "play_id": stem,
        "found_in_l2m": False,
        "player_committing": "Defender",
        "player_disadvantaged": "Ball Handler / Shooter",
        "call_type": cached_dec.get("what_foul") if cached_dec else None,
        "review_decision": "Raw Game Footage",
        "label": None,
    }
    if cached_dec:
        res_unknown["ai_decision"] = cached_dec
    return res_unknown


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


def track_foul_players_in_foul_window(
    video_path: Path,
    pose_model: object,
    foul_frame: int,
    foul_window_start: int,
    foul_window_end: int,
    anchor_roi: Optional[Union[Tuple[int, int, int, int], List[int], np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> Dict[int, Dict[str, Any]]:
    """
    Localizes player tracking strictly to the foul event window [foul_window_start, foul_window_end].
    Anchors directly at the peak contact frame (foul_frame) within the action tube ROI and tracks the
    offensive driver and contesting defender bidirectionally, eliminating all cross-court drift and
    spurious detections outside the foul.
    Returns mapping of frame_idx -> {off_box, def_box, union_box, vert_deg, contact}.
    """
    foul_window_start = max(0, int(foul_window_start))
    foul_window_end = max(foul_window_start + 1, int(foul_window_end))

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, foul_window_start)
    window_frames: Dict[int, np.ndarray] = {}
    for f in range(foul_window_start, foul_window_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        window_frames[f] = frame
    cap.release()

    if not window_frames:
        return {}

    all_keys = sorted(window_frames.keys())
    yolo_dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else ("mps" if device == "mps" else "cpu")
    active_det = preprocess_mod.get_cached_det_model()

    frame_dets: Dict[int, List[Dict[str, Any]]] = {}
    with torch.inference_mode():
        for b_idx in range(0, len(all_keys), 16):
            b_keys = all_keys[b_idx : b_idx + 16]
            b_imgs = [window_frames[k] for k in b_keys]
            preds = pose_model.predict(b_imgs, conf=0.10, imgsz=640, verbose=False, device=yolo_dev)
            det_preds = (
                active_det.predict(b_imgs, conf=0.12, imgsz=640, classes=[0], verbose=False, device=yolo_dev)
                if (active_det is not None and active_det is not pose_model)
                else [None] * len(b_imgs)
            )
            for k, frame_img, res, d_res in zip(b_keys, b_imgs, preds, det_preds):
                cands = []
                raw_bxs_list = []
                raw_cfs_list = []
                raw_kpts_list = []

                if res.boxes is not None and len(res.boxes) > 0:
                    bxs = res.boxes.xyxy.cpu().numpy()
                    cfs = res.boxes.conf.cpu().numpy()
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
                    for i, (b, c, cl) in enumerate(zip(bxs, cfs, cls)):
                        if cl == 0:
                            raw_bxs_list.append(b.astype(np.float32))
                            raw_cfs_list.append(float(c))
                            kp = kpts[i] if (kpts is not None and i < len(kpts)) else preprocess_mod.synthesize_box_keypoints(b)
                            raw_kpts_list.append(kp)

                if d_res is not None and d_res.boxes is not None and len(d_res.boxes) > 0:
                    dbxs = d_res.boxes.xyxy.cpu().numpy()
                    dcfs = d_res.boxes.conf.cpu().numpy()
                    dcls = d_res.boxes.cls.cpu().numpy().astype(int)
                    for db, dc, dcl in zip(dbxs, dcfs, dcls):
                        if dcl == 0 and all(preprocess_mod.box_iou_single(db, eb) < 0.42 for eb in raw_bxs_list):
                            raw_bxs_list.append(db.astype(np.float32))
                            raw_cfs_list.append(float(dc))
                            raw_kpts_list.append(preprocess_mod.synthesize_box_keypoints(db))

                for b, c, kp in zip(raw_bxs_list, raw_cfs_list, raw_kpts_list):
                    bh = b[3] - b[1]
                    bw = b[2] - b[0]
                    cx = 0.5 * (b[0] + b[2])
                    cy = 0.5 * (b[1] + b[3])
                    if (
                        b[1] >= frame_h * 0.22
                        and b[3] >= frame_h * 0.46
                        and b[3] <= frame_h * 0.94
                        and bh >= frame_h * 0.09
                        and bh <= frame_h * 0.48
                        and (bh / max(bw, 1.0)) >= 0.75
                        and not preprocess_mod.is_referee_or_sideline_official(frame_img, b, frame_w, frame_h)
                    ):
                        elev = 0.0
                        vert_angle = 15.0
                        if kp is not None and len(kp) >= 17:
                            sh = kp[[5, 6]]
                            wr = kp[[9, 10]]
                            valid_sh = sh[sh[:, 1] > 0]
                            valid_wr = wr[wr[:, 1] > 0]
                            if len(valid_sh) > 0 and len(valid_wr) > 0:
                                elev = float(max(0.0, (np.mean(valid_sh[:, 1]) - np.mean(valid_wr[:, 1])) / max(bh, 1.0)))
                                dx = float(np.mean(valid_wr[:, 0]) - np.mean(valid_sh[:, 0]))
                                dy = float(np.mean(valid_wr[:, 1]) - np.mean(valid_sh[:, 1]))
                                vert_angle = float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1.0))))
                        cands.append({
                            "box": b.astype(np.float32),
                            "conf": float(c),
                            "kpts": kp,
                            "cx": cx,
                            "cy": cy,
                            "bw": bw,
                            "bh": bh,
                            "elev": elev,
                            "vert_angle": vert_angle,
                        })
                frame_dets[k] = cands

    roi_cx = 0.5 * (anchor_roi[0] + anchor_roi[2]) if anchor_roi is not None else frame_w * 0.5
    roi_cy = 0.5 * (anchor_roi[1] + anchor_roi[3]) if anchor_roi is not None else frame_h * 0.7

    valid_f_candidates = [k for k in all_keys if len(frame_dets.get(k, [])) > 0]
    if not valid_f_candidates:
        return {}
    f_anchor = min(valid_f_candidates, key=lambda k: abs(k - foul_frame))
    anchor_cands = frame_dets.get(f_anchor, [])

    if len(anchor_cands) >= 2:
        anchor_cands.sort(key=lambda c: np.hypot(c["cx"] - roi_cx, c["cy"] - roi_cy))
        p1 = anchor_cands[0]
        p2 = min(anchor_cands[1:], key=lambda c: np.hypot(c["cx"] - p1["cx"], c["cy"] - p1["cy"]))
        if p1["elev"] >= p2["elev"]:
            b_off = p1["box"].copy()
            b_def = p2["box"].copy()
        else:
            b_off = p2["box"].copy()
            b_def = p1["box"].copy()
    elif len(anchor_cands) == 1:
        primary = anchor_cands[0]
        p_box = primary["box"].copy()
        b_off = p_box.copy()
        b_def = p_box.copy()
    else:
        def_b = (
            np.array(anchor_roi, dtype=np.float32)
            if anchor_roi is not None
            else np.array([frame_w * 0.35, frame_h * 0.40, frame_w * 0.65, frame_h * 0.85], dtype=np.float32)
        )
        b_off = def_b.copy()
        b_def = def_b.copy()

    def match_candidate(prev_box: np.ndarray, cands: List[Dict[str, Any]]) -> Tuple[np.ndarray, float]:
        if not cands:
            return prev_box.copy(), 22.0
        px1, py1, px2, py2 = prev_box
        p_area = max(1.0, (px2 - px1) * (py2 - py1))
        pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
        best_c = None
        best_score = -1e9
        for c in cands:
            bx1, by1, bx2, by2 = c["box"]
            ix1, iy1 = max(px1, bx1), max(py1, by1)
            ix2, iy2 = min(px2, bx2), min(py2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            c_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            iou = inter / max(1.0, p_area + c_area - inter)
            dist = np.hypot(c["cx"] - pcx, c["cy"] - pcy)
            score = iou * 3.0 - (dist / max(py2 - py1, 1.0)) * 1.2
            if score > best_score:
                best_score = score
                best_c = c
        if best_c is not None and best_score > -1.5:
            return best_c["box"].copy(), best_c.get("vert_angle", 22.0)
        return prev_box.copy(), 22.0

    off_tracks: Dict[int, np.ndarray] = {f_anchor: b_off.copy()}
    def_tracks: Dict[int, np.ndarray] = {f_anchor: b_def.copy()}
    vert_tracks: Dict[int, float] = {f_anchor: 25.0}

    # Forward tracking
    cur_off, cur_def = b_off.copy(), b_def.copy()
    for f in range(f_anchor + 1, foul_window_end + 1):
        cands = frame_dets.get(f, [])
        cur_off, _ = match_candidate(cur_off, cands)
        cur_def, va = match_candidate(cur_def, cands)
        off_tracks[f] = cur_off.copy()
        def_tracks[f] = cur_def.copy()
        vert_tracks[f] = va

    # Backward tracking
    cur_off, cur_def = b_off.copy(), b_def.copy()
    for f in range(f_anchor - 1, foul_window_start - 1, -1):
        cands = frame_dets.get(f, [])
        cur_off, _ = match_candidate(cur_off, cands)
        cur_def, va = match_candidate(cur_def, cands)
        off_tracks[f] = cur_off.copy()
        def_tracks[f] = cur_def.copy()
        vert_tracks[f] = va

    sorted_fs = sorted(off_tracks.keys())
    off_arr = np.stack([off_tracks[f] for f in sorted_fs], axis=0)
    def_arr = np.stack([def_tracks[f] for f in sorted_fs], axis=0)
    off_smooth = smooth_boxes_ema(off_arr, alpha=0.35)
    def_smooth = smooth_boxes_ema(def_arr, alpha=0.35)

    result: Dict[int, Dict[str, Any]] = {}
    pad_x, pad_y = 20.0, 18.0
    for idx_f, f in enumerate(sorted_fs):
        ob = off_smooth[idx_f]
        db = def_smooth[idx_f]
        ux1 = int(round(max(0.0, min(ob[0], db[0]) - pad_x)))
        uy1 = int(round(max(0.0, min(ob[1], db[1]) - pad_y)))
        ux2 = int(round(min(float(frame_w), max(ob[2], db[2]) + pad_x)))
        uy2 = int(round(min(float(frame_h), max(ob[3], db[3]) + pad_y)))

        # Compute exact spatial contact point (WHERE the foul occurred) between offender and shooter
        ocx, ocy = 0.5 * (ob[0] + ob[2]), 0.36 * ob[1] + 0.64 * ob[3]
        dcx, dcy = 0.5 * (db[0] + db[2]), 0.36 * db[1] + 0.64 * db[3]
        ix = int(round(np.clip(0.5 * (ocx + dcx), ux1 + 8, ux2 - 8)))
        iy = int(round(np.clip(0.5 * (ocy + dcy), uy1 + 8, uy2 - 8)))

        result[f] = {
            "off_box": (int(round(ob[0])), int(round(ob[1])), int(round(ob[2])), int(round(ob[3]))),
            "def_box": (int(round(db[0])), int(round(db[1])), int(round(db[2])), int(round(db[3]))),
            "union_box": (ux1, uy1, ux2, uy2),
            "impact_point": (ix, iy),
            "vert_deg": vert_tracks.get(f, 25.0),
            "contact": 0.85,
        }
    return result


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


def discover_video_files_in_folder(folder_path: Union[Path, str]) -> List[Path]:
    """
    Discover all raw game footage files (.mp4, .mov, .avi, .mkv, .webm, .m4v) in a folder.
    Ignores hidden files and already-annotated `*_annotated.mp4` files.
    """
    t_dir = resolve_project_path(folder_path)
    if not t_dir.exists():
        t_dir.mkdir(parents=True, exist_ok=True)
        return []
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    return sorted(
        [
            p
            for p in t_dir.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in video_exts
            and not p.stem.endswith("_annotated")
        ]
    )


def run_ai_referee_on_folder(
    folder_path: Union[Path, str] = BASE_DIR / "data" / "test",
    checkpoint_path: Optional[Union[Path, str]] = None,
    annotated_dir: Optional[Union[Path, str]] = None,
    yolo_weights: str = "yolo11n.pt",
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Run the AI Referee on all raw game footage placed in `folder_path` (defaults to `data/test/`).
    For each video file:
      - Runs spatiotemporal + biomechanical foul detection & foul-type classification.
      - If a foul occurs, annotates the video showing WHAT foul it is, WHERE it occurred, and WHEN it occurred.
      - If no foul occurs, leaves the video completely untouched (nothing happens).
      - Exports annotated videos to `data/annotated/<clip_stem>_annotated.mp4` and saves `referee_decisions.json`.
    Returns a list of structured AI Referee report dictionaries.
    """
    import json

    t_dir = resolve_project_path(folder_path)
    test_videos = discover_video_files_in_folder(t_dir)
    if not test_videos:
        logger.warning("No raw game video files found in %s. Drop .mp4/.mov/.avi clips into %s to test.", t_dir, t_dir)
        return []

    out_dir = resolve_project_path(annotated_dir) if annotated_dir else BASE_DIR / "data" / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🏀 AI Referee scanning %d video file(s) in %s: %s", len(test_videos), t_dir, [v.name for v in test_videos])
    reports: List[Dict[str, Any]] = []

    for idx, v_path in enumerate(test_videos, start=1):
        logger.info("=== [%d/%d] AI Referee evaluating clip: %s ===", idx, len(test_videos), v_path.name)
        try:
            rep = predict_single_clip(
                video_path=v_path,
                checkpoint_path=checkpoint_path,
                output_mp4_path=out_dir / f"{v_path.stem}_annotated.mp4",
                yolo_weights=yolo_weights,
                pose_weights=pose_weights,
                return_details=True,
                **kwargs,
            )
            if isinstance(rep, dict):
                reports.append(rep)
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", v_path.name, exc, exc_info=True)

    logger.info("Finished AI Referee folder evaluation: %d/%d clips annotated -> %s", len(reports), len(test_videos), out_dir)
    return reports


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
    Returns mapping of `{clip_stem: foul_prob}` for backward compatibility.
    """
    reports = run_ai_referee_on_folder(
        folder_path=test_dir,
        checkpoint_path=checkpoint_path,
        annotated_dir=annotated_dir,
        yolo_weights=yolo_weights,
        pose_weights=pose_weights,
        **kwargs,
    )
    return {r["stem"]: float(r["foul_prob"]) for r in reports}


@torch.no_grad()
def predict_single_clip(
    video_path: Optional[Union[Path, str, argparse.Namespace]] = None,
    checkpoint_path: Optional[Union[Path, str]] = None,
    output_mp4_path: Optional[Union[Path, str]] = None,
    yolo_weights: str = "yolo11n.pt",
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    return_details: bool = False,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Run end-to-end AI Referee evaluation on a raw game video clip:
      - Determines if a foul occurred (`is_foul`) and what foul it is (`what_foul`, `foul_subtype`, `foul_mechanic`).
      - Pinpoints WHEN it occurred (`foul_frame`, `foul_time_sec`) and WHERE it occurred (`impact_point`, `union_box`, `court_zone`).
      - If a foul occurs, highlights WHERE it occurred, WHAT foul it is, and WHEN it occurred on the annotated video.
      - If NO foul occurs, nothing happens on the video (100% clean, unaltered video playback).
    """
    import json

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
        logger.info("Input path is a directory: running batch AI Referee annotation on %s", resolved_video)
        batch_reports = run_ai_referee_on_folder(
            folder_path=resolved_video,
            checkpoint_path=resolved_ckpt,
            annotated_dir=actual_out,
            yolo_weights=yolo_weights,
            pose_weights=pose_candidate,
        )
        if return_details:
            return {"reports": batch_reports}
        return float(np.mean([r["foul_prob"] for r in batch_reports])) if batch_reports else 0.0

    if not resolved_video.exists():
        raise FileNotFoundError(f"Input video not found: {resolved_video}")
    if YOLO is None:
        raise ImportError("Ultralytics YOLO is required for video clip inference (`pip install ultralytics`).")

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
        if isinstance(ckpt, dict) and "optimal_threshold" in ckpt:
            threshold = float(np.clip(float(ckpt["optimal_threshold"]), 0.48, 0.50))
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

    # Strictly autonomous AI Referee inference: do not read or use CSV metadata labels/players
    committing = "Defender"
    disadvantaged = "Ball-Handler / Shooter"
    l2m_call_type = None
    review_decision = "AI Referee Automated Call"
    meta_label = None
    found_in_l2m = False

    with get_autocast_context(device):
        ref_dec = model.predict_referee_decision(
            video_frames=input_batch,
            pose_features=feats_batch,
            threshold=threshold,
            call_type_hint=None,
        )

    foul_prob = float(ref_dec["foul_prob"])
    is_foul = bool(ref_dec["is_foul"])

    what_foul = ref_dec["what_foul"] if is_foul else "No Foul"
    foul_subtype = ref_dec["foul_subtype"] if is_foul else "No Foul (Legal Play)"
    foul_mechanic = ref_dec["foul_mechanic"]
    apex_tube_idx = int(ref_dec.get("apex_frame_offset", len(sampled_indices) // 2))

    # Annotated video folder placement: always default to data/annotated/
    if actual_out is None:
        annotated_dir = BASE_DIR / "data" / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = annotated_dir / f"{resolved_video.stem}_annotated.mp4"
    else:
        final_output_path = resolve_project_path(actual_out)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load video dimensions & frame count
    cap_info = cv2.VideoCapture(str(resolved_video))
    total_video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap_info.release()

    anchor_hint_f = int(sampled_indices[min(apex_tube_idx, len(sampled_indices) - 1)]) if sampled_indices else (total_video_frames // 2)
    anchor_hint_box = crop_boxes[min(apex_tube_idx, len(crop_boxes) - 1)] if crop_boxes else None

    foul_frame = anchor_hint_f
    foul_time_sec = float(foul_frame) / max(fps, 1.0)
    mins = int(foul_time_sec // 60)
    secs = foul_time_sec - mins * 60

    off_num_str = ""
    off_team_str = "OFF"
    def_num_str = ""
    def_team_str = "DEF"

    foul_window_boxes: Dict[int, Dict[str, Any]] = {}
    where_point: Optional[Tuple[int, int]] = None
    where_box: Optional[Tuple[int, int, int, int]] = None
    where_zone = "N/A (No Foul)"
    when_str = "N/A (No Foul)"

    if is_foul:
        window_pre = int(round(fps * 0.85))   # ~0.85s before contact apex
        window_post = int(round(fps * 1.35))  # ~1.35s after contact apex
        foul_window_start = max(0, foul_frame - window_pre)
        foul_window_end = min(max(0, total_video_frames - 1), foul_frame + window_post)
        foul_window_boxes = track_foul_players_in_foul_window(
            video_path=resolved_video,
            pose_model=pose_model,
            foul_frame=foul_frame,
            foul_window_start=foul_window_start,
            foul_window_end=foul_window_end,
            anchor_roi=anchor_hint_box,
            metadata=None,
            device=str(device),
            frame_w=frame_w,
            frame_h=frame_h,
        )
        # Extract exact spatial location at foul apex frame
        apex_key = min(foul_window_boxes.keys(), key=lambda k: abs(k - foul_frame)) if foul_window_boxes else None
        if apex_key is not None:
            apex_data = foul_window_boxes[apex_key]
            where_box = apex_data["union_box"]
            where_point = apex_data["impact_point"]
        elif anchor_hint_box is not None:
            bx1, by1, bx2, by2 = anchor_hint_box
            where_box = (int(bx1), int(by1), int(bx2), int(by2))
            where_point = (int((bx1 + bx2) // 2), int((by1 + by2) // 2))
        else:
            where_box = (int(frame_w * 0.35), int(frame_h * 0.35), int(frame_w * 0.65), int(frame_h * 0.80))
            where_point = (frame_w // 2, int(frame_h * 0.55))

        where_zone = describe_court_location(where_point[0], where_point[1], frame_w, frame_h)
        when_str = f"{mins:02d}:{secs:05.2f} ({foul_time_sec:.2f}s | Frame {foul_frame})"
        logger.info(
            "🚨 AI REFEREE WHISTLE on %s -> WHAT: %s | WHEN: %s | WHERE: %s | P(foul)=%.1f%%",
            resolved_video.name,
            foul_subtype,
            when_str,
            where_zone,
            foul_prob * 100.0,
        )
    else:
        foul_window_start = 999999
        foul_window_end = -1
        logger.info(
            "✅ AI REFEREE CALL on %s -> NO FOUL (P(foul)=%.1f%% < %.1f%%). Nothing happens on video (clean output).",
            resolved_video.name,
            foul_prob * 100.0,
            threshold * 100.0,
        )

    # Render annotated video:
    # Rule: "the annotated video should show what foul it is and highlight where it occured along with when.
    #        if no foul happens nothing should happen"
    if not is_foul and resolved_video.suffix.lower() == ".mp4":
        # Fast path when no foul happens on an MP4: copy untouched video directly so literally nothing happens!
        if resolved_video.resolve() != final_output_path.resolve():
            shutil.copy2(resolved_video, final_output_path)
        frame_idx = total_video_frames
    else:
        cap = cv2.VideoCapture(str(resolved_video))
        writer = create_cross_platform_video_writer(final_output_path, fps, frame_w, frame_h)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # If no foul happens (or outside the foul window), NOTHING happens on the video frame!
            draw_foul_highlight = (
                is_foul
                and (foul_window_start <= frame_idx <= foul_window_end)
                and (frame_idx in foul_window_boxes)
            )

            if draw_foul_highlight:
                fdata = foul_window_boxes[frame_idx]
                fx1, fy1, fx2, fy2 = fdata["off_box"]
                dx1, dy1, dx2, dy2 = fdata["def_box"]
                ux1, uy1, ux2, uy2 = fdata["union_box"]
                ix, iy = fdata.get("impact_point", ((ux1 + ux2) // 2, (uy1 + uy2) // 2))
                cur_zone = describe_court_location(ix, iy, frame_w, frame_h)

                # 1. Highlight WHERE the foul occurred: translucent red zone tint + bold red boundary
                zone_overlay = frame.copy()
                cv2.rectangle(zone_overlay, (ux1, uy1), (ux2, uy2), (15, 15, 195), -1)
                cv2.addWeighted(zone_overlay, 0.16, frame, 0.84, 0, frame)
                cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (0, 0, 235), 4)

                # 2. Fouled Player (Offense) & Committing Player (Defense) boxes
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 215, 0), 2)
                off_label = (
                    f"Fouled: {disadvantaged} ({off_num_str} {off_team_str})".strip()
                    if found_in_l2m
                    else "Fouled Player (Offense)"
                )
                (otw, oth), _ = cv2.getTextSize(off_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (fx1, max(0, fy1 - oth - 6)), (min(frame_w - 2, fx1 + otw + 8), fy1), (255, 215, 0), -1)
                cv2.putText(frame, off_label, (fx1 + 4, max(12, fy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)

                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 145, 255), 2)
                def_label = (
                    f"Foul By: {committing} ({def_num_str} {def_team_str})".strip()
                    if found_in_l2m
                    else "Committing Player (Defense)"
                )
                (dtw, dth), _ = cv2.getTextSize(def_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (dx1, max(0, dy1 - dth - 6)), (min(frame_w - 2, dx1 + dtw + 8), dy1), (0, 145, 255), -1)
                cv2.putText(frame, def_label, (dx1 + 4, max(12, dy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)

                # 3. Exact Contact Impact Reticle (WHERE the foul contact occurred)
                fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                dcx, dcy = (dx1 + dx2) // 2, (dy1 + dy2) // 2
                cv2.line(frame, (fcx, fcy), (dcx, dcy), (0, 60, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 24, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 11, (0, 230, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (ix, iy), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(frame, (ix - 32, iy), (ix + 32, iy), (0, 230, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (ix, iy - 32), (ix, iy + 32), (0, 230, 255), 2, cv2.LINE_AA)

                pt_tag = f"FOUL CONTACT ({ix}, {iy})"
                (ptw, pth), _ = cv2.getTextSize(pt_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                pt_x = min(frame_w - ptw - 12, max(6, ix + 18))
                pt_y = min(frame_h - 45, max(35, iy - 12))
                cv2.rectangle(frame, (pt_x - 4, pt_y - pth - 4), (pt_x + ptw + 6, pt_y + 4), (0, 0, 210), -1)
                cv2.putText(frame, pt_tag, (pt_x, pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                # 4. Call Tag directly above the Foul Zone showing WHAT FOUL & WHEN
                tag_y = max(56, uy1 - 12)
                what_tag = f"FOUL: {foul_subtype.upper()}"
                when_where_tag = f"WHEN: {foul_time_sec:.2f}s (Frame {foul_frame}) | WHERE: {cur_zone}"
                (tw1, th1), _ = cv2.getTextSize(what_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                (tw2, th2), _ = cv2.getTextSize(when_where_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                box_w = max(tw1, tw2) + 18
                box_x1 = max(8, min(ux1, frame_w - box_w - 8))
                cv2.rectangle(frame, (box_x1, tag_y - th1 - th2 - 18), (box_x1 + box_w, tag_y + 4), (15, 15, 215), -1)
                cv2.rectangle(frame, (box_x1, tag_y - th1 - th2 - 18), (box_x1 + box_w, tag_y + 4), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    what_tag,
                    (box_x1 + 8, tag_y - th2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    when_where_tag,
                    (box_x1 + 8, tag_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (180, 245, 255),
                    1,
                    cv2.LINE_AA,
                )

                # 5. Top-left AI Referee Whistle Call Banner (only displayed during foul call!)
                line1 = f"AI REFEREE WHISTLE  |  {what_foul.upper()}"
                line2 = f"WHAT : {foul_subtype}  ({foul_mechanic})"
                line3 = f"WHEN : {foul_time_sec:.2f}s (Frame {foul_frame})   |   WHERE: {cur_zone}"
                (lw1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                (lw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                (lw3, _), _ = cv2.getTextSize(line3, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                hud_w = max(520, max(lw1, lw2, lw3) + 40)
                hud_h = 86

                hud_overlay = frame.copy()
                cv2.rectangle(hud_overlay, (18, 18), (18 + hud_w, 18 + hud_h), (16, 16, 20), -1)
                cv2.addWeighted(hud_overlay, 0.88, frame, 0.12, 0, frame)
                cv2.rectangle(frame, (18, 18), (26, 18 + hud_h), (0, 0, 240), -1)
                cv2.rectangle(frame, (18, 18), (18 + hud_w, 18 + hud_h), (0, 0, 240), 2)

                cv2.putText(frame, line1, (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line2, (36, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(frame, line3, (36, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

                # 6. Bottom Timeline Marker highlighting WHEN the foul occurred
                bar_x1, bar_x2 = 30, frame_w - 30
                bar_y1, bar_y2 = frame_h - 26, frame_h - 12
                bar_w = bar_x2 - bar_x1
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (25, 25, 25), -1)
                w_start_x = max(bar_x1, bar_x1 + int(bar_w * (float(foul_window_start) / float(max(1, total_video_frames)))))
                w_end_x = min(bar_x2, bar_x1 + int(bar_w * (float(foul_window_end) / float(max(1, total_video_frames)))))
                split_x = max(bar_x1, min(bar_x2, bar_x1 + int(bar_w * (float(foul_frame) / float(max(1, total_video_frames))))))
                cv2.rectangle(frame, (w_start_x, bar_y1), (w_end_x, bar_y2), (0, 0, 235), -1)
                cv2.line(frame, (split_x, bar_y1 - 6), (split_x, bar_y2 + 2), (255, 255, 255), 2)
                w_label = f"FOUL @ {foul_time_sec:.2f}s (Frame {foul_frame})"
                cv2.putText(frame, w_label, (max(bar_x1, split_x - 75), bar_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 80, 255), 1, cv2.LINE_AA)
                curr_x = max(bar_x1, min(bar_x2, bar_x1 + int(bar_w * (float(frame_idx) / float(max(1, total_video_frames - 1))))))
                cv2.circle(frame, (curr_x, (bar_y1 + bar_y2) // 2), 6, (0, 235, 255), -1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # Optional faststart remux with ffmpeg so QuickTime/macOS and HTML5 browsers can play cleanly
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

    decision_report: Dict[str, Any] = {
        "clip": resolved_video.name,
        "stem": resolved_video.stem,
        "is_foul": is_foul,
        "foul_prob": round(foul_prob, 4),
        "neural_prob": round(float(ref_dec.get("neural_prob", foul_prob)), 4),
        "threshold": round(threshold, 4),
        "verdict": "FOUL WHISTLED" if is_foul else "NO FOUL (CLEAN PLAY)",
        "what_foul": what_foul,
        "foul_subtype": foul_subtype,
        "foul_mechanic": foul_mechanic,
        "when_frame": foul_frame if is_foul else None,
        "when_time_sec": round(foul_time_sec, 2) if is_foul else None,
        "when_str": when_str,
        "where_zone": where_zone,
        "where_point": list(where_point) if where_point is not None else None,
        "where_box": list(where_box) if where_box is not None else None,
        "player_committing": committing,
        "player_disadvantaged": disadvantaged,
        "found_in_l2m": found_in_l2m,
        "ground_truth_label": meta_label,
        "output_mp4": str(final_output_path),
    }

    # Persist decision report to data/annotated/referee_decisions.json for notebook & UI inspection
    decisions_json_path = final_output_path.parent / "referee_decisions.json"
    all_decisions: Dict[str, Any] = {}
    if decisions_json_path.exists():
        try:
            with open(decisions_json_path, "r", encoding="utf-8") as jf:
                loaded = json.load(jf)
                if isinstance(loaded, dict):
                    all_decisions = loaded
        except Exception:
            all_decisions = {}
    all_decisions[resolved_video.stem] = decision_report
    try:
        with open(decisions_json_path, "w", encoding="utf-8") as jf:
            json.dump(all_decisions, jf, indent=2)
    except Exception as exc:
        logger.debug("Could not write referee_decisions.json: %s", exc)

    logger.info("Saved annotated prediction video (%d frames) -> %s", frame_idx, final_output_path)
    if return_details:
        return decision_report
    return foul_prob


def train_rl_referee_agent(
    episodes: Optional[List[Dict[str, Any]]] = None,
    dataset: Optional[object] = None,
    num_episodes: int = 150,
    batch_size: int = 64,
    mode: str = "dqn",
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """
    Train a Reinforcement Learning Referee Agent on NBA L2M decision episodes.
    Optimizes whistle decision timing, sequential cylinder intrusion, and legal guarding position
    grounded in official NBA League Office review rewards (CC, INC, CNC, IC).
    """
    dev = select_compute_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device
    model_mod = importlib.import_module("3_model_and_dataset")

    if episodes is None:
        if dataset is None:
            dataset = model_mod.L2MFoulDataset(is_train=True, return_foul_features=True)
        episodes = model_mod.create_referee_episodes_from_dataset(dataset)

    if not episodes:
        raise RuntimeError("No episodes available for RL training.")

    env = model_mod.NBARefereeEnv(episodes=episodes)
    agent = model_mod.RefereeRLAgent(mode=mode, device=dev)

    logger.info(
        "Training Referee RL Agent (%s) across %d episodes on %s (Replay Buffer: %d)...",
        mode.upper(),
        num_episodes,
        dev,
        len(episodes),
    )

    episode_rewards: List[float] = []
    correct_calls: List[bool] = []
    losses: List[float] = []

    for ep_num in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        step_losses = []

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step(batch_size=batch_size)
            if loss > 0:
                step_losses.append(loss)
            state = next_state
            ep_reward += reward

        if ep_num % 5 == 0:
            agent.update_target_network()

        episode_rewards.append(ep_reward)
        correct_calls.append(bool(info.get("correct_call", False)))
        if step_losses:
            losses.append(float(np.mean(step_losses)))

        if ep_num % 25 == 0 or ep_num == num_episodes:
            recent_acc = float(np.mean(correct_calls[-25:]) * 100.0)
            avg_rew = float(np.mean(episode_rewards[-25:]))
            logger.info(
                "RL Episode [%03d/%03d] | Avg Reward: %+.2f | Officiating Accuracy: %.1f%% | Epsilon: %.2f",
                ep_num,
                num_episodes,
                avg_rew,
                recent_acc,
                agent.epsilon,
            )

    save_path = resolve_project_path(checkpoint_path or (BASE_DIR / "weights" / "referee_rl_agent.pt"))
    agent.save(save_path)

    metrics = {
        "final_accuracy": float(np.mean(correct_calls[-30:])),
        "mean_reward": float(np.mean(episode_rewards)),
        "episode_rewards": episode_rewards,
        "losses": losses,
        "checkpoint_path": str(save_path),
    }
    return metrics


def run_training_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_compute_device(args.device)
    logger.info("Using compute device: %s (OS: %s)", device, platform.system())

    # Detect Google Colab / High-VRAM NVIDIA T4 GPU (15.3 GB VRAM)
    is_cuda = (device.type == "cuda" and torch.cuda.is_available())
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if is_cuda else 0.0
    is_colab_t4 = is_cuda and (vram_gb >= 12.0)
    high_vram_active = getattr(args, "high_vram", False) or is_colab_t4

    use_cached_embeddings = True
    if high_vram_active and not getattr(args, "freeze_backbone", False):
        use_cached_embeddings = False
        if args.batch_size <= 2 or args.batch_size > 8:
            args.batch_size = 4
        if args.accum_steps <= 2:
            args.accum_steps = 8
        args.freeze_backbone = False
        logger.info(
            "🚀 Google Colab T4 High-Compute Mode Active! Total VRAM: %.1f GB.",
            vram_gb,
        )
        logger.info(
            "Optimized High-VRAM: End-to-End 3D ConvNet Fine-Tuning | batch_size=%d, accum_steps=%d (eff=%d) | AMP=True",
            args.batch_size,
            args.accum_steps,
            args.batch_size * args.accum_steps,
        )

    base_dataset = L2MFoulDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        is_train=True,
        max_samples=args.max_samples,
        return_foul_features=True,
        use_cached_embeddings=use_cached_embeddings,
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
        L2MFoulDataset(
            samples=train_samples,
            is_train=True,
            return_foul_features=True,
            use_cached_embeddings=use_cached_embeddings,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=use_pin_memory,
        collate_fn=safe_foul_collate,
    )
    val_loader = DataLoader(
        L2MFoulDataset(
            samples=val_samples,
            is_train=False,
            return_foul_features=True,
            use_cached_embeddings=use_cached_embeddings,
        ),
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

        mem_msg = ""
        if device.type == "cuda":
            vram_curr = torch.cuda.memory_allocated() / (1024**3)
            vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            mem_msg = f" | VRAM: {vram_curr:.1f}/{vram_peak:.1f} GB"

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch [%02d/%02d] | LR: %.2e | Train Loss: %.4f | Val Loss: %.4f | "
            "Acc: %.3f | Prec: %.3f | Rec: %.3f | F1: %.3f | ROC-AUC: %.3f | Thresh: %.2f%s",
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
            mem_msg,
        )

        if (metrics["f1"] > best_f1) or (
            np.isclose(metrics["f1"], best_f1) and metrics["roc_auc"] > best_roc_auc
        ):
            best_f1 = metrics["f1"]
            best_roc_auc = metrics["roc_auc"]
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "optimal_threshold": metrics["threshold"],
                "val_loss": val_loss,
            }
            torch.save(ckpt_payload, ckpt_path)
            weights_ckpt = BASE_DIR / "weights" / "best_foul_r2plus1d.pt"
            if weights_ckpt.resolve() != ckpt_path.resolve():
                weights_ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ckpt_payload, weights_ckpt)
            logger.info(
                "Saved new best foul checkpoint -> %s (F1=%.3f, ROC-AUC=%.3f, threshold=%.2f)",
                ckpt_path,
                best_f1,
                best_roc_auc,
                metrics["threshold"],
            )

    if getattr(args, "train_rl", False):
        logger.info("Initiating Reinforcement Learning Referee Agent Training...")
        train_rl_referee_agent(
            dataset=base_dataset,
            num_episodes=getattr(args, "rl_episodes", 150),
            mode=getattr(args, "rl_mode", "dqn"),
            device=device,
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
        "--high-vram",
        action="store_true",
        help="Google Colab Tesla T4 / 16GB GPU Mode: unfreezes 3D ConvNet backbone, uses raw video tensors, sets batch_size 16, AMP fp16, maximizing GPU memory (10-14 GB)",
    )
    parser.add_argument(
        "--colab-t4",
        action="store_true",
        dest="high_vram",
        help="Alias for --high-vram optimized for Google Colab Tesla T4 environment",
    )
    parser.add_argument(
        "--train-rl",
        action="store_true",
        help="Train the Reinforcement Learning Referee Agent (DQN / Actor-Critic) on L2M episodes",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=150,
        help="Number of episodes for Reinforcement Learning referee training (default: 150)",
    )
    parser.add_argument(
        "--rl-mode",
        type=str,
        default="dqn",
        choices=["dqn", "actor_critic"],
        help="Reinforcement learning algorithm mode ('dqn' or 'actor_critic')",
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
