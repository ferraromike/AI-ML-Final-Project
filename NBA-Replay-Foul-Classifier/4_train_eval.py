#!/usr/bin/env python3
"""
STAGE 4: Multi-Head Challenge Replay Training, Calibration & Inference (`4_train_eval.py`)

Trains, evaluates, and runs inference for `ChallengeReplayClassifier` across all NBA Coach's
Challenge types (`challenge_foul`, `challenge_oob`, `challenge_goaltending_bi`) and all 8
specific call resolutions (`foul_stands`, `foul_overturned_no_foul`, `foul_overturned_other_foul`,
`oob_stands`, `oob_overturned`, `oob_proximate_foul`, `goaltending_bi_stands`, `goaltending_bi_overturned`):
  - Leak-Free Grouped Splitting: Splits Train/Val by `game_id` via `GroupShuffleSplit` and excludes
    all held-out `data/test/` clips (matched by stem, CDN media slug, or visual signature).
  - Multi-Task Training: Jointly optimizes Binary Foul Focal BCE, 8-Class Specific Call Cross-Entropy,
    3-Class Challenge-Type Identification Cross-Entropy, and Challenge Resolution (`STANDS` vs. `OVERTURNED`).
  - Saliency-Weighted Play-Level Evaluation: Aggregates multi-angle predictions per `play_id` weighted
    by camera-shot action saliency, calibrates the decision threshold, and logs both Binary Foul metrics
    and Multi-Class Challenge/Call Accuracy.
  - Context-Aware Inference & Video Export: Knows what is being challenged (from CLI `--challenge-type`,
    archive metadata/CDN slug, filename trigger, or `challenge_type_head` neural prediction) and outputs
    the exact ruling & correct call to `data/annotated/predictions.json` and annotated MP4s.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import argparse
import contextlib
import csv
import gc
import importlib
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

cv2.setNumThreads(0)

scrape_mod = importlib.import_module("1_scrape_replays")
preprocess_mod = importlib.import_module("2_preprocess_replays")
model_mod = importlib.import_module("3_model_and_dataset")

CALL_CLASSES = scrape_mod.CALL_CLASSES
CALL_CLASS_SPECS = scrape_mod.CALL_CLASS_SPECS
CHALLENGE_TYPES = scrape_mod.CHALLENGE_TYPES
CHALLENGE_TYPE_DISPLAY = scrape_mod.CHALLENGE_TYPE_DISPLAY
CHALLENGE_TYPE_TO_ID = scrape_mod.CHALLENGE_TYPE_TO_ID
ID_TO_CALL_CLASS = scrape_mod.ID_TO_CALL_CLASS
ID_TO_CHALLENGE_TYPE = scrape_mod.ID_TO_CHALLENGE_TYPE
extract_cdn_media_slug = scrape_mod.extract_cdn_media_slug
upgrade_metadata_row = scrape_mod.upgrade_metadata_row

ChallengeReplayClassifier = model_mod.ChallengeReplayClassifier
BinaryFoulClassifier = model_mod.BinaryFoulClassifier
ConsistentVideoAugmentor = model_mod.ConsistentVideoAugmentor
ReplayFoulDataset = model_mod.ReplayFoulDataset
load_torch_file_safe = model_mod.load_torch_file_safe
resolve_project_path = model_mod.resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("replay_train_eval")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_compute_device(requested: Optional[str] = None) -> torch.device:
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
            _w = torch.empty((2, 2), device=dev)
            _ = _w @ _w
            del _w
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


def get_safe_num_workers(requested: int = 0) -> int:
    if platform.system() in ("Darwin", "Windows"):
        return 0
    return max(0, int(requested))


def get_autocast_context(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


class FocalBCEWithLogitsLoss(nn.Module):
    """Class-balanced Binary Focal Cross-Entropy with mild label smoothing."""

    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma: float = 1.5,
        label_smoothing: float = 0.04,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w = self.pos_weight.to(logits.device)
        smoothed = (
            targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            if self.label_smoothing > 0.0
            else targets
        )
        bce = F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pos_w, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - p_t).pow(self.gamma)
        return torch.mean(focal * bce)


class MultiTaskChallengeLoss(nn.Module):
    """
    Joint Multi-Task Loss for `ChallengeReplayClassifier`:
      1. `foul_loss`: Class-balanced Focal BCE for binary foul presence (`0` vs `1`)
      2. `call_loss`: Class-weighted Cross-Entropy over the 8 specific call resolutions (`0..7`)
      3. `ctype_loss`: Class-weighted Cross-Entropy over the 3 challenge types (`0..2`)
      4. `res_loss`: Binary Cross-Entropy for challenge resolution (`0=STANDS` vs `1=OVERTURNED`)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        call_weights: torch.Tensor,
        ctype_weights: torch.Tensor,
        gamma: float = 1.5,
        label_smoothing: float = 0.04,
    ) -> None:
        super().__init__()
        self.foul_criterion = FocalBCEWithLogitsLoss(
            pos_weight=pos_weight,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
        self.register_buffer("call_weights", call_weights)
        self.register_buffer("ctype_weights", ctype_weights)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        t_foul: torch.Tensor,
        t_ctype: Optional[torch.Tensor] = None,
        t_call: Optional[torch.Tensor] = None,
        t_res: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.foul_criterion(outputs, t_foul)

        foul_logits = outputs["foul_logits"]
        loss_foul = self.foul_criterion(foul_logits, t_foul)
        total_loss = loss_foul

        if t_call is not None and "call_logits" in outputs:
            cw = self.call_weights.to(foul_logits.device)
            loss_call = F.cross_entropy(
                outputs["call_logits"],
                t_call.to(foul_logits.device).long(),
                weight=cw,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + 0.85 * loss_call

        if t_ctype is not None and "challenge_type_logits" in outputs:
            ctw = self.ctype_weights.to(foul_logits.device)
            loss_ctype = F.cross_entropy(
                outputs["challenge_type_logits"],
                t_ctype.to(foul_logits.device).long(),
                weight=ctw,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + 0.45 * loss_ctype

        if t_res is not None and "resolution_logits" in outputs:
            res_t = t_res.to(foul_logits.device).view_as(outputs["resolution_logits"])
            loss_res = F.binary_cross_entropy_with_logits(outputs["resolution_logits"], res_t)
            total_loss = total_loss + 0.35 * loss_res

        return total_loss


def compute_pos_weight(records: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    num_pos = sum(1 for r in records if int(r["label"]) == 1)
    num_neg = sum(1 for r in records if int(r["label"]) == 0)
    if num_pos == 0 or num_neg == 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    w = float(np.clip(float(num_neg) / float(num_pos), 0.5, 2.5))
    logger.info("Train class balance -> foul (1): %d | no_foul (0): %d | pos_weight=%.3f", num_pos, num_neg, w)
    return torch.tensor([w], dtype=torch.float32, device=device)


def compute_multiclass_weights(
    records: List[Dict[str, Any]],
    key: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(int(r.get(key, 0)) for r in records)
    total = max(1, len(records))
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = float(np.clip((total / (num_classes * float(counts[c]))) ** 0.25, 0.75, 1.35))
        else:
            weights[c] = 1.0
    weights = weights / max(float(np.mean(weights)), 1e-6)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def split_records_grouped(
    records: List[Dict[str, Any]],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Leak-free Train/Val split grouped by `game_id` (falling back to `play_id` if too few games).
    Selects a deterministic split where both Train and Val have balanced classes and challenge types.
    """
    if len(records) < 4:
        mid = max(1, len(records) - 1)
        return records[:mid], records[mid:]

    labels = np.array([int(r["label"]) for r in records])
    ctypes = np.array([int(r.get("challenge_type_id", 0)) for r in records])
    calls = np.array([int(r.get("call_label", 0)) for r in records])
    game_groups = np.array([str(r["game_id"]) for r in records])
    play_groups = np.array([str(r["play_id"]) for r in records])

    total_pos_rate = float(np.mean(labels))
    uniq_ctypes = set(ctypes.tolist())
    uniq_calls = set(calls.tolist())

    for group_name, groups in (("game_id", game_groups), ("play_id", play_groups)):
        uniq = np.unique(groups)
        if len(uniq) < 2:
            continue
        best_pair: Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = None
        best_split_quality = -1e9

        for offset in range(40):
            gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed + offset)
            tr_idx, va_idx = next(gss.split(records, labels, groups=groups))
            tr_lbls = set(labels[tr_idx].tolist())
            va_lbls = set(labels[va_idx].tolist())
            if len(set(labels.tolist())) > 1 and (len(tr_lbls) < 2 or len(va_lbls) < 2):
                continue

            va_pos_rate = float(np.mean(labels[va_idx]))
            tr_ct_cov = len(set(ctypes[tr_idx].tolist())) / max(1, len(uniq_ctypes))
            va_ct_cov = len(set(ctypes[va_idx].tolist())) / max(1, len(uniq_ctypes))
            tr_call_cov = len(set(calls[tr_idx].tolist())) / max(1, len(uniq_calls))
            va_call_cov = len(set(calls[va_idx].tolist())) / max(1, len(uniq_calls))

            quality = (
                2.0 * tr_ct_cov
                + 1.5 * va_ct_cov
                + 1.5 * tr_call_cov
                + 1.0 * va_call_cov
                - 3.0 * abs(va_pos_rate - total_pos_rate)
            )
            if quality > best_split_quality:
                best_split_quality = quality
                train_recs = [records[i] for i in tr_idx]
                val_recs = [records[i] for i in va_idx]
                best_pair = (train_recs, val_recs)

        if best_pair is not None:
            train_recs, val_recs = best_pair
            tr_plays = {r["play_id"] for r in train_recs}
            va_plays = {r["play_id"] for r in val_recs}
            assert tr_plays.isdisjoint(va_plays), "Data leakage detected across train/val play_ids!"
            logger.info(
                "Grouped split (%s) -> Train: %d tubes (%d plays) | Val: %d tubes (%d plays) | 0 play overlap",
                group_name,
                len(train_recs),
                len(tr_plays),
                len(val_recs),
                len(va_plays),
            )
            return train_recs, val_recs

    strat = labels if (len(set(labels.tolist())) > 1 and min(int(np.sum(labels == 0)), int(np.sum(labels == 1))) >= 2) else None
    tr_recs, va_recs = train_test_split(records, test_size=val_split, random_state=seed, stratify=strat)
    return list(tr_recs), list(va_recs)


def safe_replay_collate(batch: List[Tuple]) -> Tuple:
    """Collate dataset tuples safely for both training and metadata-enriched validation."""
    if not batch:
        return ()
    n_fields = len(batch[0])
    raw_videos = [item[0] for item in batch]
    poses = torch.stack([item[1] for item in batch], dim=0)
    t_foul = torch.stack([item[2] for item in batch], dim=0)

    shapes = {tuple(v.shape) for v in raw_videos}
    if len(shapes) == 1:
        videos = torch.stack(raw_videos, dim=0)
    else:
        extractor = model_mod.get_shared_visual_extractor()
        ext_dev = next(extractor.parameters()).device
        harmonized = []
        mean = torch.tensor(model_mod.KINETICS400_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(model_mod.KINETICS400_STD).view(3, 1, 1, 1)
        for v in raw_videos:
            if v.dim() == 1 and v.size(0) == model_mod.VIDEO_FEATURE_DIM:
                harmonized.append(v.float())
            elif v.dim() == 4:
                with torch.no_grad():
                    nv = v.float()
                    if nv.max() > 1.5:
                        nv = nv / 255.0
                    nv = ((nv - mean) / std).unsqueeze(0).to(ext_dev)
                    harmonized.append(extractor(nv).squeeze(0).cpu().float())
            else:
                harmonized.append(v)
        videos = torch.stack(harmonized, dim=0)

    if n_fields == 3:
        return videos, poses, t_foul
    if n_fields == 5:
        play_ids = [item[3] for item in batch]
        sample_ids = [item[4] for item in batch]
        return videos, poses, t_foul, play_ids, sample_ids

    t_ctype = torch.stack([item[3] for item in batch], dim=0)
    t_call = torch.stack([item[4] for item in batch], dim=0)
    t_res = torch.stack([item[5] for item in batch], dim=0)

    if n_fields >= 9:
        play_ids = [item[6] for item in batch]
        sample_ids = [item[7] for item in batch]
        saliencies = [float(item[8]) for item in batch]
        return videos, poses, t_foul, t_ctype, t_call, t_res, play_ids, sample_ids, saliencies
    return videos, poses, t_foul, t_ctype, t_call, t_res


def calibrate_binary_threshold(
    y_true: List[int],
    y_prob: List[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    Sweep probability thresholds across `[0.35 .. 0.65]` with a mild centering prior around `0.50`
    so small validation splits remain well-calibrated and never drift to extreme cutoffs.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        return 0.50, {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "threshold": 0.50,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }

    roc_auc = 0.5
    if len(np.unique(yt)) > 1:
        try:
            roc_auc = float(roc_auc_score(yt, yp))
        except Exception:
            roc_auc = 0.5

    best_thresh = 0.50
    best_obj = -1.0
    best_metrics: Dict[str, Any] = {}

    for thresh in np.linspace(0.35, 0.65, 31):
        pred = (yp >= thresh).astype(int)
        f1 = float(f1_score(yt, pred, zero_division=0))
        acc = float(accuracy_score(yt, pred))
        bal_acc = float(balanced_accuracy_score(yt, pred)) if len(np.unique(yt)) > 1 else acc
        prec = float(precision_score(yt, pred, zero_division=0))
        rec = float(recall_score(yt, pred, zero_division=0))
        # Mild regularization toward 0.50 prevents tiny val sets from picking extreme thresholds
        center_pen = 0.05 * abs(float(thresh) - 0.50)
        obj = 0.50 * f1 + 0.50 * bal_acc - center_pen

        if obj > best_obj or (np.isclose(obj, best_obj) and abs(thresh - 0.50) < abs(best_thresh - 0.50)):
            best_obj = obj
            best_thresh = float(round(float(thresh), 3))
            cm = confusion_matrix(yt, pred, labels=[0, 1])
            best_metrics = {
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "threshold": best_thresh,
                "confusion_matrix": {
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                },
            }

    return best_thresh, best_metrics


def train_one_epoch(
    model: ChallengeReplayClassifier,
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
        videos, poses, t_foul = batch[0], batch[1], batch[2]
        has_multitask = len(batch) >= 6
        t_ctype = batch[3].to(device) if has_multitask else None
        t_call = batch[4].to(device) if has_multitask else None
        t_res = batch[5].to(device) if has_multitask else None

        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        poses = poses.to(device, non_blocking=(device.type == "cuda"))
        targets = t_foul.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)
        bsz = videos.size(0)

        try:
            with get_autocast_context(device):
                if isinstance(criterion, MultiTaskChallengeLoss) and has_multitask:
                    outputs = model(videos, poses, challenge_type_ids=t_ctype, return_dict=True)
                    raw_loss = criterion(outputs, targets, t_ctype=t_ctype, t_call=t_call, t_res=t_res)
                else:
                    logits = model(videos, poses)
                    raw_loss = criterion(logits, targets)
                scaled_loss = raw_loss / accum_steps

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss_val = float(raw_loss.item())
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            err_msg = str(exc).lower()
            if "out of memory" not in err_msg and "cublas" not in err_msg:
                raise
            logger.warning("CUDA OOM on batch %d; falling back to micro-batching.", step)
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            micro = max(1, bsz // 2)
            accum_val = 0.0
            for m0 in range(0, bsz, micro):
                m1 = min(m0 + micro, bsz)
                w = float(m1 - m0) / float(bsz)
                with get_autocast_context(device):
                    if isinstance(criterion, MultiTaskChallengeLoss) and has_multitask:
                        m_out = model(
                            videos[m0:m1],
                            poses[m0:m1],
                            challenge_type_ids=t_ctype[m0:m1] if t_ctype is not None else None,
                            return_dict=True,
                        )
                        m_loss = criterion(
                            m_out,
                            targets[m0:m1],
                            t_ctype=t_ctype[m0:m1] if t_ctype is not None else None,
                            t_call=t_call[m0:m1] if t_call is not None else None,
                            t_res=t_res[m0:m1] if t_res is not None else None,
                        )
                    else:
                        m_logits = model(videos[m0:m1], poses[m0:m1])
                        m_loss = criterion(m_logits, targets[m0:m1])
                    m_scaled = (m_loss * w) / accum_steps
                if scaler is not None:
                    scaler.scale(m_scaled).backward()
                else:
                    m_scaled.backward()
                accum_val += float(m_loss.item()) * (m1 - m0)
            loss_val = accum_val / max(bsz, 1)

        if ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader)):
            trainable = [p for p in model.parameters() if p.requires_grad]
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss_val * bsz
        total_samples += bsz

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_play_level(
    model: ChallengeReplayClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate model on validation split and aggregate multi-angle predictions per `play_id`
    using saliency-weighted angle pooling across both Binary Foul and 8-Class Specific Call heads.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    play_videos: Dict[str, List[torch.Tensor]] = defaultdict(list)
    play_poses: Dict[str, List[torch.Tensor]] = defaultdict(list)
    play_sals: Dict[str, List[float]] = defaultdict(list)
    play_foul_labels: Dict[str, int] = {}
    play_ctype_labels: Dict[str, int] = {}
    play_call_labels: Dict[str, int] = {}
    play_res_labels: Dict[str, int] = {}

    for batch in loader:
        videos, poses, t_foul = batch[0], batch[1], batch[2]
        has_multitask = len(batch) >= 9
        if has_multitask:
            t_ctype = batch[3].to(device)
            t_call = batch[4].to(device)
            t_res = batch[5].to(device)
            play_ids = batch[6]
            saliencies = batch[8]
        else:
            t_ctype = torch.zeros(videos.size(0), dtype=torch.long, device=device)
            t_call = t_foul.long().to(device)
            t_res = torch.zeros(videos.size(0), dtype=torch.float32, device=device)
            play_ids = batch[3] if len(batch) >= 4 else [f"sample_{total_samples + i}" for i in range(videos.size(0))]
            saliencies = [0.75] * videos.size(0)

        videos_dev = videos.to(device, non_blocking=(device.type == "cuda"))
        poses_dev = poses.to(device, non_blocking=(device.type == "cuda"))
        targets_dev = t_foul.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)

        with get_autocast_context(device):
            if isinstance(criterion, MultiTaskChallengeLoss) and has_multitask:
                outputs = model(videos_dev, poses_dev, challenge_type_ids=t_ctype, return_dict=True)
                loss = criterion(outputs, targets_dev, t_ctype=t_ctype, t_call=t_call, t_res=t_res)
            else:
                logits = model(videos_dev, poses_dev)
                loss = criterion(logits, targets_dev)

        bsz = videos.size(0)
        running_loss += float(loss.item()) * bsz
        total_samples += bsz

        for i in range(bsz):
            pid = str(play_ids[i])
            play_videos[pid].append(videos[i].detach().cpu())
            play_poses[pid].append(poses[i].detach().cpu())
            play_sals[pid].append(float(saliencies[i]))
            play_foul_labels[pid] = int(t_foul[i].item())
            play_ctype_labels[pid] = int(t_ctype[i].item())
            play_call_labels[pid] = int(t_call[i].item())
            play_res_labels[pid] = int(t_res[i].item())

    val_loss = running_loss / max(total_samples, 1)
    y_true: List[int] = []
    y_prob: List[float] = []
    raw_play_cache: List[Tuple[str, torch.Tensor, torch.Tensor, List[float], Dict[str, Any]]] = []

    for pid in sorted(play_videos.keys()):
        v_stack = torch.stack(play_videos[pid], dim=0)
        p_stack = torch.stack(play_poses[pid], dim=0)
        s_list = play_sals[pid]
        true_ctype_id = play_ctype_labels[pid]
        pred_uncal = model.predict_challenge(
            video_input=v_stack,
            pose_features=p_stack,
            challenge_type=true_ctype_id,
            angle_saliencies=s_list,
            threshold=0.50,
        )
        y_true.append(play_foul_labels[pid])
        y_prob.append(float(pred_uncal["foul_prob"]))
        raw_play_cache.append((pid, v_stack, p_stack, s_list, pred_uncal))

    best_thresh, metrics = calibrate_binary_threshold(y_true, y_prob)

    # Re-evaluate specific call & challenge resolution using the calibrated threshold
    call_true_list: List[int] = []
    call_pred_list: List[int] = []
    ctype_true_list: List[int] = []
    ctype_pred_list: List[int] = []
    res_true_list: List[int] = []
    res_pred_list: List[int] = []
    play_predictions: List[Dict[str, Any]] = []

    for pid, v_stack, p_stack, s_list, _ in raw_play_cache:
        true_foul = play_foul_labels[pid]
        true_ctype = play_ctype_labels[pid]
        true_call = play_call_labels[pid]
        true_res = play_res_labels[pid]

        # 1. Evaluate autonomous challenge-type prediction head (without giving ground-truth challenge_type)
        pred_auto = model.predict_challenge(
            video_input=v_stack,
            pose_features=p_stack,
            challenge_type=None,
            angle_saliencies=s_list,
            threshold=best_thresh,
        )
        # 2. Evaluate specific call resolution knowing what is being challenged
        pred_cond = model.predict_challenge(
            video_input=v_stack,
            pose_features=p_stack,
            challenge_type=true_ctype,
            angle_saliencies=s_list,
            threshold=best_thresh,
        )

        pred_call = int(pred_cond["call_label"])
        pred_ctype = int(pred_auto["challenge_type_id"])
        pred_res = 1 if pred_cond["call_resolution"] == "CALL OVERTURNED" else 0

        call_true_list.append(true_call)
        call_pred_list.append(pred_call)
        ctype_true_list.append(true_ctype)
        ctype_pred_list.append(pred_ctype)
        res_true_list.append(true_res)
        res_pred_list.append(pred_res)

        play_predictions.append({
            "play_id": pid,
            "true_label": true_foul,
            "true_challenge_type": ID_TO_CHALLENGE_TYPE.get(true_ctype, "foul"),
            "pred_challenge_type": pred_auto["challenge_type"],
            "true_call": ID_TO_CALL_CLASS.get(true_call, "foul_stands"),
            "pred_call": pred_cond["outcome_category"],
            "call_resolution": pred_cond["call_resolution"],
            "correct_call": pred_cond["correct_call"],
            "verdict": pred_cond["verdict"],
            "foul_prob": pred_cond["foul_prob"],
            "call_confidence": pred_cond["call_confidence"],
            "angle_probs": pred_cond["angle_probs"],
            "angle_saliencies": pred_cond["angle_saliencies"],
        })

    if call_true_list:
        metrics["call_accuracy"] = round(float(accuracy_score(call_true_list, call_pred_list)), 4)
        metrics["challenge_type_accuracy"] = round(float(accuracy_score(ctype_true_list, ctype_pred_list)), 4)
        metrics["resolution_accuracy"] = round(float(accuracy_score(res_true_list, res_pred_list)), 4)
    else:
        metrics["call_accuracy"] = 0.0
        metrics["challenge_type_accuracy"] = 0.0
        metrics["resolution_accuracy"] = 0.0

    metrics["num_val_plays"] = len(y_true)
    metrics["num_val_tubes"] = total_samples
    return val_loss, metrics, play_predictions


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("avc1", "mp4v", "XVID"):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open VideoWriter for {output_path}")


def render_binary_prediction_video(
    input_video: Path,
    output_video: Path,
    is_foul: bool,
    foul_prob: float,
    threshold: float,
    tubes: List[Dict[str, Any]],
    angle_probs: List[float],
    clean_no_foul: bool = False,
    challenge_pred: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export an official NBA Replay Center broadcast-style annotated video displaying:
      - Line 1: What is being challenged (`CHALLENGE: CHALLENGE OF CALLED FOUL / OUT OF BOUNDS / GOALTENDING`)
      - Line 2: Official Ruling & Verdict (`VERDICT: FOUL (CALL STANDS) / NO FOUL (CALL OVERTURNED) / ...`)
      - Line 3: Specific Correct Call to be made on the floor + angles evaluated
      - Action ROI bounding box strictly inside the winning camera angle's shot boundaries
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if not is_foul and clean_no_foul and input_video.suffix.lower() == ".mp4":
        if input_video.resolve() != output_video.resolve():
            shutil.copy2(input_video, output_video)
        return

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ctype_disp = (
        str(challenge_pred.get("challenge_type_display"))
        if challenge_pred and challenge_pred.get("challenge_type_display")
        else "CHALLENGE OF CALLED FOUL"
    )
    verdict_short = (
        str(challenge_pred.get("verdict"))
        if challenge_pred and challenge_pred.get("verdict")
        else ("FOUL (CALL STANDS)" if is_foul else "NO FOUL (CALL OVERTURNED)")
    )
    correct_call_str = (
        str(challenge_pred.get("correct_call"))
        if challenge_pred and challenge_pred.get("correct_call")
        else ("CALLED FOUL STANDS" if is_foul else "NO FOUL — LEGAL CONTACT")
    )
    conf_pct = (
        float(challenge_pred.get("call_confidence", foul_prob if is_foul else (1.0 - foul_prob))) * 100.0
        if challenge_pred
        else (foul_prob if is_foul else (1.0 - foul_prob)) * 100.0
    )

    highlight_boxes: Dict[int, Tuple[int, int, int, int]] = {}
    win_start, win_end = -1, -1
    if tubes:
        sals = [float(t.get("saliency", 0.5)) for t in tubes]
        best_a = int(np.argmax(sals)) if sals else 0
        best_tube = tubes[min(best_a, len(tubes) - 1)]
        s_idxs = best_tube["sampled_indices"]
        c_boxes = best_tube["crop_boxes"]
        if s_idxs and c_boxes:
            win_start = max(int(best_tube["shot_start_frame"]), int(min(s_idxs)))
            win_end = min(int(best_tube["shot_end_frame"]), int(max(s_idxs)))
            xs = np.array(s_idxs, dtype=np.float32)
            box_arr = np.array(c_boxes, dtype=np.float32)
            for f in range(win_start, win_end + 1):
                bx = [int(round(float(np.interp(f, xs, box_arr[:, c])))) for c in range(4)]
                highlight_boxes[f] = (bx[0], bx[1], bx[2], bx[3])

    writer = create_video_writer(output_video, fps, width, height)
    # Color coding: Red for Foul calls, Amber/Blue for OOB/Goaltending, Green for Clean/No-Foul overturns
    if is_foul:
        accent = (30, 30, 235)
    elif challenge_pred and challenge_pred.get("challenge_type") in ("oob", "goaltending", "challenge_oob", "challenge_goaltending_bi"):
        accent = (235, 155, 30)
    else:
        accent = (45, 195, 70)

    line1_text = f"CHALLENGED: {ctype_disp}"
    line2_text = f"RULING: {verdict_short}  ({conf_pct:.1f}%)"
    line3_text = f"Call: {correct_call_str[:58]} | P(foul)={foul_prob * 100.0:.1f}%"

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        banner_w = min(width - 32, 620)
        banner_h = 92
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (16 + banner_w, 16 + banner_h), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.rectangle(frame, (16, 16), (25, 16 + banner_h), accent, -1)
        cv2.rectangle(frame, (16, 16), (16 + banner_w, 16 + banner_h), accent, 2)
        cv2.putText(frame, line1_text, (34, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 205, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line2_text, (34, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line3_text, (34, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 215, 225), 1, cv2.LINE_AA)

        if f_idx in highlight_boxes:
            x1, y1, x2, y2 = highlight_boxes[f_idx]
            tint = frame.copy()
            cv2.rectangle(tint, (x1, y1), (x2, y2), accent, -1)
            cv2.addWeighted(tint, 0.10, frame, 0.90, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 3)
            tag = f"{verdict_short} ({conf_pct:.1f}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            ty = max(th + 10, y1 - 6)
            cv2.rectangle(frame, (x1, ty - th - 6), (min(width - 2, x1 + tw + 10), ty + 4), accent, -1)
            cv2.putText(frame, tag, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        if total_frames > 1:
            bx1, bx2 = 24, width - 24
            by1, by2 = height - 18, height - 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 34), -1)
            if win_end > win_start >= 0:
                wsx = bx1 + int((bx2 - bx1) * (win_start / float(max(total_frames, f_idx + 1))))
                wex = bx1 + int((bx2 - bx1) * (win_end / float(max(total_frames, f_idx + 1))))
                cv2.rectangle(frame, (wsx, by1), (max(wsx + 2, wex), by2), accent, -1)
            cx = bx1 + int((bx2 - bx1) * (f_idx / float(max(total_frames - 1, f_idx + 1))))
            cv2.circle(frame, (cx, (by1 + by2) // 2), 5, (255, 255, 255), -1, cv2.LINE_AA)

        writer.write(frame)
        f_idx += 1

    cap.release()
    writer.release()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin and output_video.exists():
        tmp_h264 = output_video.with_name(f"{output_video.stem}_h264.mp4")
        res = subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", str(output_video),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-loglevel", "error", str(tmp_h264),
            ],
            check=False,
        )
        if res.returncode == 0 and tmp_h264.exists() and tmp_h264.stat().st_size > 1000:
            tmp_h264.replace(output_video)


def discover_video_files(folder_path: Union[Path, str]) -> List[Path]:
    t_dir = resolve_project_path(folder_path)
    if not t_dir.exists():
        t_dir.mkdir(parents=True, exist_ok=True)
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    return sorted(
        p for p in t_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts and not p.stem.endswith("_annotated")
    )


def resolve_clip_challenge_context(
    video_path: Path,
    explicit_challenge_type: Optional[str] = None,
    metadata_csv: Path = BASE_DIR / "data" / "replay_metadata.csv",
) -> Tuple[Optional[str], str]:
    """
    Determine what call is being challenged (`challenge_type`) for an input replay clip:
      1. Explicit CLI/argument (`--challenge-type foul|oob|goaltending_bi`) -> `source="cli"`
      2. Archive metadata lookup (`play_id`, CDN media slug `trigger16_...`, or visual frame match
         to identify ONLY what is being challenged, never the outcome!) -> `source="metadata"`
      3. Filename trigger/keyword heuristic (`trigger16`/`foul`, `trigger17`/`oob`, `trigger18`/`goaltend`)
         -> `source="filename_trigger"`
      4. `None` -> Autonomous neural prediction via `model.challenge_type_head` (`source="model_predicted"`)
    """
    if explicit_challenge_type and explicit_challenge_type.strip().lower() not in ("auto", "none", ""):
        raw = explicit_challenge_type.strip().lower()
        if raw in scrape_mod.CHALLENGE_TYPE_ALIASES:
            return scrape_mod.CHALLENGE_TYPE_ALIASES[raw], "cli"

    # If the clip sits inside one of data/raw/{foul,oob,goaltending,flagrant}/, use its parent folder name
    parent_low = video_path.parent.name.strip().lower()
    if parent_low in scrape_mod.CHALLENGE_TYPES:
        return parent_low, "folder"

    stem = video_path.stem
    stem_low = stem.lower()

    # Check metadata catalog for challenge_type context (by play_id, CDN slug, or visual signature)
    meta_p = resolve_project_path(metadata_csv)
    if meta_p.exists():
        try:
            with open(meta_p, "r", newline="", encoding="utf-8") as f:
                rows = [upgrade_metadata_row(dict(r)) for r in csv.DictReader(f)]
            pid_to_row = {str(r.get("play_id") or "").strip(): r for r in rows}
            for pid, r in pid_to_row.items():
                if pid and (stem == pid or stem.startswith(pid)):
                    return str(r["challenge_type"]), "metadata"
            slug = extract_cdn_media_slug(stem_low)
            if slug:
                for r in rows:
                    r_slug = extract_cdn_media_slug(str(r.get("mp4_url") or ""))
                    if r_slug and r_slug == slug:
                        return str(r["challenge_type"]), "metadata"
            _, stem_map = preprocess_mod.discover_held_out_test_ids(video_path.parent, rows)
            if stem in stem_map and stem_map[stem] in pid_to_row:
                return str(pid_to_row[stem_map[stem]]["challenge_type"]), "metadata"
        except Exception:
            pass

    # Check NBA Replay CDN trigger prefix or filename keyword
    if "trigger3" in stem_low or "trigger8" in stem_low or "flagrant" in stem_low:
        return "flagrant", "filename_trigger"
    if "trigger18" in stem_low or "trigger13" in stem_low or "goaltend" in stem_low or "interference" in stem_low:
        return "goaltending", "filename_trigger"
    if "trigger17" in stem_low or "oob" in stem_low or "out_of_bounds" in stem_low or "out-of-bounds" in stem_low:
        return "oob", "filename_trigger"
    if "trigger16" in stem_low or "foul" in stem_low:
        return "foul", "filename_trigger"

    return None, "model_predicted"


def load_trained_classifier(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[ChallengeReplayClassifier, float]:
    model = ChallengeReplayClassifier(pretrained=False, backbone_mode="frozen").to(device)
    threshold = 0.50
    if checkpoint_path.exists():
        ckpt = load_torch_file_safe(checkpoint_path, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model_sd = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(matched, strict=False)
        if isinstance(ckpt, dict) and "optimal_threshold" in ckpt:
            threshold = float(ckpt["optimal_threshold"])
        logger.info("Loaded checkpoint %s (%d/%d layers, threshold=%.3f)", checkpoint_path, len(matched), len(model_sd), threshold)
    else:
        logger.warning("Checkpoint %s not found; running with uncalibrated weights.", checkpoint_path)
    model.eval()
    return model, threshold


@torch.no_grad()
def predict_clip(
    video_path: Union[Path, str],
    checkpoint_path: Union[Path, str] = BASE_DIR / "weights" / "best_replay_foul_classifier.pt",
    output_mp4_path: Optional[Union[Path, str]] = None,
    challenge_type: Optional[str] = "auto",
    yolo_weights: str = "yolo11n.pt",
    pose_weights: str = "yolo11n-pose.pt",
    angles_per_clip: int = 2,
    render_video: bool = True,
    clean_no_foul: bool = False,
    device_name: str = "auto",
    loaded_model: Optional[ChallengeReplayClassifier] = None,
    loaded_threshold: Optional[float] = None,
    loaded_pose_yolo: Optional[Any] = None,
    loaded_det_yolo: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run multi-angle Coach's Challenge inference on a single NBA replay clip:
      1. Identifies what is being challenged (`challenge_type`: `challenge_foul`, `challenge_oob`,
         or `challenge_goaltending_bi`) from CLI/metadata/trigger or `challenge_type_head`.
      2. Determines the exact challenge resolution (`CALL STANDS` vs. `CALL OVERTURNED`) and
         the specific correct call to be made (`outcome_category` / `correct_call`).
    """
    vpath = resolve_project_path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {vpath}")
    if YOLO is None and loaded_pose_yolo is None:
        raise ImportError("Ultralytics YOLO is required (`pip install ultralytics`).")

    device = select_compute_device(device_name)
    ckpt_p = resolve_project_path(checkpoint_path)

    if loaded_model is not None:
        model = loaded_model
        threshold = float(loaded_threshold if loaded_threshold is not None else 0.50)
    else:
        model, threshold = load_trained_classifier(ckpt_p, device)

    pose_yolo = loaded_pose_yolo
    det_yolo = loaded_det_yolo
    if pose_yolo is None:
        p_w = resolve_project_path(pose_weights)
        pose_yolo = YOLO(str(p_w) if p_w.exists() else pose_weights)
    if det_yolo is None:
        d_w = resolve_project_path(yolo_weights)
        det_yolo = YOLO(str(d_w) if d_w.exists() else yolo_weights)

    resolved_ctype, ctype_source = resolve_clip_challenge_context(
        video_path=vpath,
        explicit_challenge_type=challenge_type,
    )

    tubes = preprocess_mod.extract_replay_action_tubes(
        video_path=vpath,
        pose_model=pose_yolo,
        det_model=det_yolo,
        angles_per_clip=angles_per_clip,
        device=str(device),
    )

    augmentor = ConsistentVideoAugmentor(is_train=False)
    vid_batch = torch.stack([augmentor(t["video"]) for t in tubes], dim=0).to(device)
    pose_batch = torch.stack([t["pose_features"] for t in tubes], dim=0).to(device)
    angle_sals = [float(t.get("saliency", 0.5)) for t in tubes]

    with get_autocast_context(device):
        pred = model.predict_challenge(
            video_input=vid_batch,
            pose_features=pose_batch,
            challenge_type=resolved_ctype,
            angle_saliencies=angle_sals,
            threshold=threshold,
        )

    out_dir = BASE_DIR / "data" / "annotated"
    final_mp4 = resolve_project_path(output_mp4_path) if output_mp4_path else (out_dir / f"{vpath.stem}_annotated.mp4")

    if render_video:
        render_binary_prediction_video(
            input_video=vpath,
            output_video=final_mp4,
            is_foul=pred["is_foul"],
            foul_prob=pred["foul_prob"],
            threshold=threshold,
            tubes=tubes,
            angle_probs=pred["angle_probs"],
            clean_no_foul=clean_no_foul,
            challenge_pred=pred,
        )

    report: Dict[str, Any] = {
        "clip": vpath.name,
        "stem": vpath.stem,
        "challenge_type": pred["challenge_type"],
        "challenge_type_display": pred["challenge_type_display"],
        "challenge_type_source": ctype_source,
        "challenge_type_probs": pred["challenge_type_probs"],
        "call_resolution": pred["call_resolution"],
        "outcome_category": pred["outcome_category"],
        "call_label": pred["call_label"],
        "correct_call": pred["correct_call"],
        "verdict": pred["verdict"],
        "binary_verdict": pred["binary_verdict"],
        "is_foul": pred["is_foul"],
        "foul_prob": round(float(pred["foul_prob"]), 4),
        "call_confidence": round(float(pred["call_confidence"]), 4),
        "call_probs": pred["call_probs"],
        "threshold": round(float(threshold), 4),
        "angles_evaluated": len(tubes),
        "angle_probs": [round(float(p), 4) for p in pred["angle_probs"]],
        "angle_saliencies": [round(float(s), 4) for s in pred["angle_saliencies"]],
        "output_mp4": str(final_mp4) if render_video else None,
    }

    pred_json = final_mp4.parent / "predictions.json"
    pred_json.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if pred_json.exists():
        try:
            loaded = json.loads(pred_json.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    existing[vpath.stem] = report
    pred_json.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    logger.info(
        "[%s] %s (%s) -> %s | Correct Call: %s | P(foul)=%.1f%% (conf=%.1f%%, angles=%d)",
        vpath.name,
        report["challenge_type_display"],
        report["challenge_type_source"],
        report["verdict"],
        report["correct_call"],
        report["foul_prob"] * 100.0,
        report["call_confidence"] * 100.0,
        len(tubes),
    )
    return report


def predict_folder(
    folder_path: Union[Path, str] = BASE_DIR / "data" / "test",
    checkpoint_path: Union[Path, str] = BASE_DIR / "weights" / "best_replay_foul_classifier.pt",
    annotated_dir: Union[Path, str] = BASE_DIR / "data" / "annotated",
    challenge_type: Optional[str] = "auto",
    yolo_weights: str = "yolo11n.pt",
    pose_weights: str = "yolo11n-pose.pt",
    angles_per_clip: int = 2,
    render_video: bool = True,
    clean_no_foul: bool = False,
    device_name: str = "auto",
) -> List[Dict[str, Any]]:
    videos = discover_video_files(folder_path)
    if not videos:
        logger.warning("No video clips found in %s.", resolve_project_path(folder_path))
        return []

    device = select_compute_device(device_name)
    ckpt_p = resolve_project_path(checkpoint_path)
    out_dir = resolve_project_path(annotated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, threshold = load_trained_classifier(ckpt_p, device)
    p_w = resolve_project_path(pose_weights)
    d_w = resolve_project_path(yolo_weights)
    pose_yolo = YOLO(str(p_w) if p_w.exists() else pose_weights) if YOLO is not None else None
    det_yolo = YOLO(str(d_w) if d_w.exists() else yolo_weights) if YOLO is not None else None

    reports: List[Dict[str, Any]] = []
    for idx, vp in enumerate(videos, start=1):
        logger.info("=== [%d/%d] Evaluating Challenge Replay %s ===", idx, len(videos), vp.name)
        rep = predict_clip(
            video_path=vp,
            checkpoint_path=ckpt_p,
            output_mp4_path=out_dir / f"{vp.stem}_annotated.mp4",
            challenge_type=challenge_type,
            angles_per_clip=angles_per_clip,
            render_video=render_video,
            clean_no_foul=clean_no_foul,
            device_name=str(device),
            loaded_model=model,
            loaded_threshold=threshold,
            loaded_pose_yolo=pose_yolo,
            loaded_det_yolo=det_yolo,
        )
        reports.append(rep)
    return reports


def run_training_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    seed_everything(args.seed)
    device = select_compute_device(args.device)
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if is_cuda else 0.0
    logger.info("Compute device: %s (VRAM=%.1f GB | OS=%s)", device, vram_gb, platform.system())

    backbone_mode = args.backbone_mode
    if backbone_mode == "auto":
        backbone_mode = "partial" if (is_cuda and vram_gb >= 10.0) else "frozen"
    use_cached = (backbone_mode == "frozen") and (not args.no_cache)

    if is_cuda and vram_gb >= 12.0 and backbone_mode != "frozen" and args.batch_size < 8:
        args.batch_size = 8

    base_ds = ReplayFoulDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        test_dir=args.test_dir,
        is_train=True,
        max_samples=args.max_samples,
        use_cached_embeddings=use_cached,
    )
    if len(base_ds) < 2:
        raise RuntimeError(
            f"Found only {len(base_ds)} processed samples in {args.processed_dir}. "
            "Run `1_scrape_replays.py` and `2_preprocess_replays.py` first."
        )

    train_recs, val_recs = split_records_grouped(base_ds.records, val_split=args.val_split, seed=args.seed)

    safe_workers = get_safe_num_workers(args.num_workers)
    train_ds = ReplayFoulDataset(
        records=train_recs,
        processed_dir=args.processed_dir,
        test_dir=args.test_dir,
        is_train=True,
        use_cached_embeddings=use_cached,
        return_meta=False,
    )
    val_ds = ReplayFoulDataset(
        records=val_recs,
        processed_dir=args.processed_dir,
        test_dir=args.test_dir,
        is_train=False,
        use_cached_embeddings=use_cached,
        return_meta=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=is_cuda,
        collate_fn=safe_replay_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=is_cuda,
        collate_fn=safe_replay_collate,
    )

    model = ChallengeReplayClassifier(
        pretrained=not args.no_pretrained,
        backbone_mode=backbone_mode,
        dropout_p=args.dropout,
        modality_dropout=args.modality_dropout,
    ).to(device)

    pinfo = model.trainable_parameter_summary()
    logger.info(
        "ChallengeReplayClassifier (backbone_mode=%s, cached_emb=%s) | Trainable: %s / %s | batch=%d x accum=%d",
        backbone_mode,
        use_cached,
        f"{pinfo['trainable_params']:,}",
        f"{pinfo['total_params']:,}",
        args.batch_size,
        args.accum_steps,
    )

    pos_weight = compute_pos_weight(train_recs, device=device)
    call_weights = compute_multiclass_weights(train_recs, key="call_label", num_classes=len(CALL_CLASSES), device=device)
    ctype_weights = compute_multiclass_weights(train_recs, key="challenge_type_id", num_classes=len(CHALLENGE_TYPES), device=device)

    criterion = MultiTaskChallengeLoss(
        pos_weight=pos_weight,
        call_weights=call_weights,
        ctype_weights=ctype_weights,
        gamma=1.5,
        label_smoothing=0.04,
    )

    optimizer = AdamW(
        model.parameter_groups(base_lr=args.lr, backbone_lr_scale=0.1),
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if is_cuda else None

    ckpt_path = resolve_project_path(args.checkpoint_path)
    best_score = -1.0
    best_metrics: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            accum_steps=args.accum_steps,
        )
        va_loss, metrics, val_preds = evaluate_play_level(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        cm = metrics["confusion_matrix"]
        cur_lr = optimizer.param_groups[-1]["lr"]
        logger.info(
            "Epoch [%02d/%02d] | LR %.2e | TrLoss %.4f | VaLoss %.4f | "
            "FoulAcc %.3f | BalAcc %.3f | F1 %.3f | CallAcc %.3f | CTypeAcc %.3f | Thresh %.2f | CM[TN=%d FP=%d FN=%d TP=%d]",
            epoch,
            args.epochs,
            cur_lr,
            tr_loss,
            va_loss,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics.get("call_accuracy", 0.0),
            metrics.get("challenge_type_accuracy", 0.0),
            metrics["threshold"],
            cm["tn"],
            cm["fp"],
            cm["fn"],
            cm["tp"],
        )

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **metrics})
        comp_score = (
            0.35 * metrics["f1"]
            + 0.25 * metrics["balanced_accuracy"]
            + 0.20 * metrics.get("call_accuracy", 0.0)
            + 0.10 * metrics.get("challenge_type_accuracy", 0.0)
            + 0.10 * metrics["roc_auc"]
        )
        if comp_score > best_score:
            best_score = comp_score
            best_metrics = dict(metrics)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_mode": backbone_mode,
                    "model_state_dict": model.state_dict(),
                    "optimal_threshold": metrics["threshold"],
                    "metrics": metrics,
                    "val_loss": va_loss,
                    "val_play_predictions": val_preds,
                    "call_classes": list(CALL_CLASSES),
                    "challenge_types": list(CHALLENGE_TYPES),
                },
                ckpt_path,
            )
            logger.info(
                "Saved new best checkpoint -> %s (F1=%.3f, BalAcc=%.3f, CallAcc=%.3f, CTypeAcc=%.3f, thresh=%.2f)",
                ckpt_path,
                metrics["f1"],
                metrics["balanced_accuracy"],
                metrics.get("call_accuracy", 0.0),
                metrics.get("challenge_type_accuracy", 0.0),
                metrics["threshold"],
            )

    if use_cached:
        train_ds.save_embedding_cache()
        val_ds.save_embedding_cache()

    return {"best_metrics": best_metrics, "history": history, "checkpoint_path": str(ckpt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: Multi-Head NBA Challenge Replay Training, Calibration & Inference"
    )
    parser.add_argument("--processed-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--metadata-csv", type=Path, default=BASE_DIR / "data" / "processed" / "processed_metadata.csv")
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "data" / "test")
    parser.add_argument("--annotated-dir", type=Path, default=BASE_DIR / "data" / "annotated")
    parser.add_argument("--checkpoint-path", type=Path, default=BASE_DIR / "weights" / "best_replay_foul_classifier.pt")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--modality-dropout", type=float, default=0.08)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--backbone-mode",
        choices=("auto", "frozen", "partial", "full"),
        default="auto",
        help="3D CNN backbone fine-tuning mode ('auto': partial on >=10GB CUDA GPU, frozen otherwise)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable cached 512-D embeddings even in frozen mode")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip downloading Kinetics-400 weights (offline mode)")
    parser.add_argument("--seed", type=int, default=42)

    # Inference CLI options
    parser.add_argument("--predict", type=Path, default=None, help="Predict Challenge Ruling & Correct Call on a video or folder")
    parser.add_argument("--predict-test", action="store_true", help="Run batch Challenge inference on all clips in --test-dir")
    parser.add_argument(
        "--challenge-type",
        type=str,
        default="auto",
        help="What is being challenged: 'auto' (default), 'foul', 'oob', or 'goaltending_bi'",
    )
    parser.add_argument("--output-mp4", type=Path, default=None)
    parser.add_argument("--angles-per-clip", type=int, default=2)
    parser.add_argument("--no-render", action="store_true", help="Skip MP4 annotation export (fast verdict JSON only)")
    parser.add_argument("--clean-no-foul", action="store_true", help="Leave NO FOUL videos untouched with zero overlay")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n.pt")
    parser.add_argument("--pose-weights", type=str, default="yolo11n-pose.pt")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.predict_test:
        predict_folder(
            folder_path=cli_args.test_dir,
            checkpoint_path=cli_args.checkpoint_path,
            annotated_dir=cli_args.annotated_dir,
            challenge_type=cli_args.challenge_type,
            yolo_weights=cli_args.yolo_weights,
            pose_weights=cli_args.pose_weights,
            angles_per_clip=cli_args.angles_per_clip,
            render_video=not cli_args.no_render,
            clean_no_foul=cli_args.clean_no_foul,
            device_name=cli_args.device,
        )
    elif cli_args.predict is not None:
        target = resolve_project_path(cli_args.predict)
        if target.is_dir():
            predict_folder(
                folder_path=target,
                checkpoint_path=cli_args.checkpoint_path,
                annotated_dir=cli_args.annotated_dir,
                challenge_type=cli_args.challenge_type,
                yolo_weights=cli_args.yolo_weights,
                pose_weights=cli_args.pose_weights,
                angles_per_clip=cli_args.angles_per_clip,
                render_video=not cli_args.no_render,
                clean_no_foul=cli_args.clean_no_foul,
                device_name=cli_args.device,
            )
        else:
            predict_clip(
                video_path=target,
                checkpoint_path=cli_args.checkpoint_path,
                output_mp4_path=cli_args.output_mp4,
                challenge_type=cli_args.challenge_type,
                yolo_weights=cli_args.yolo_weights,
                pose_weights=cli_args.pose_weights,
                angles_per_clip=cli_args.angles_per_clip,
                render_video=not cli_args.no_render,
                clean_no_foul=cli_args.clean_no_foul,
                device_name=cli_args.device,
            )
    else:
        run_training_pipeline(cli_args)
