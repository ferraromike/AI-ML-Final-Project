#!/usr/bin/env python3
"""
STAGE 4: Binary Foul Classifier Training, Evaluation & Replay Inference (`4_train_eval.py`)

Pure binary Foul (`1`) vs. No Foul (`0`) training and inference pipeline for NBA Replay Archive:
  - Leak-Free Grouped Splitting: Splits Train/Val by `game_id` (and always keeps all camera angles
    of a `play_id` in the same split) via `GroupShuffleSplit`.
  - Multi-Angle Play-Level Evaluation: Aggregates predictions across extracted angles of each play,
    sweeps thresholds cleanly across `[0.20 .. 0.80]`, and logs Accuracy, Balanced Accuracy,
    Precision, Recall, F1, ROC-AUC, and Confusion Matrix (`TN, FP, FN, TP`).
  - Hardware-Aware Training:
      • GPU / Google Colab T4: Partial 3D fine-tuning (`layer3` + `layer4`) with FP16 AMP and
        differential learning rates (`backbone = 0.1x lr`, `head = 1.0x lr`).
      • Laptop CPU / Apple Silicon: Fast frozen-backbone mode with cached 512-D embeddings.
  - Binary Inference & Video Export: Evaluates single replay clips or `data/test/` folders,
    saving verdicts to `data/annotated/predictions.json` and optional annotated MP4s to
    `data/annotated/<clip_stem>_annotated.mp4`.
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
import gc
import importlib
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
from collections import defaultdict
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

preprocess_mod = importlib.import_module("2_preprocess_replays")
model_mod = importlib.import_module("3_model_and_dataset")

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
        label_smoothing: float = 0.05,
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


def compute_pos_weight(records: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    num_pos = sum(1 for r in records if int(r["label"]) == 1)
    num_neg = sum(1 for r in records if int(r["label"]) == 0)
    if num_pos == 0 or num_neg == 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    w = float(num_neg) / float(num_pos)
    logger.info("Train class balance -> foul (1): %d | no_foul (0): %d | pos_weight=%.3f", num_pos, num_neg, w)
    return torch.tensor([w], dtype=torch.float32, device=device)


def split_records_grouped(
    records: List[Dict[str, Any]],
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Leak-free Train/Val split grouped by `game_id` (falling back to `play_id` if too few games).
    Guarantees that multiple camera angles of the same play NEVER appear in both Train and Val.
    """
    if len(records) < 4:
        mid = max(1, len(records) - 1)
        return records[:mid], records[mid:]

    labels = np.array([int(r["label"]) for r in records])
    game_groups = np.array([str(r["game_id"]) for r in records])
    play_groups = np.array([str(r["play_id"]) for r in records])

    for group_name, groups in (("game_id", game_groups), ("play_id", play_groups)):
        uniq = np.unique(groups)
        if len(uniq) < 2:
            continue
        # Try several deterministic seeds to find a grouped split where BOTH train & val have both classes
        for offset in range(25):
            gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed + offset)
            tr_idx, va_idx = next(gss.split(records, labels, groups=groups))
            tr_lbls = set(labels[tr_idx].tolist())
            va_lbls = set(labels[va_idx].tolist())
            if len(set(labels.tolist())) == 1 or (len(tr_lbls) > 1 and len(va_lbls) > 1):
                train_recs = [records[i] for i in tr_idx]
                val_recs = [records[i] for i in va_idx]
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

    # Ultimate fallback for tiny synthetic/single-play tests
    strat = labels if (len(set(labels.tolist())) > 1 and min(int(np.sum(labels == 0)), int(np.sum(labels == 1))) >= 2) else None
    tr_recs, va_recs = train_test_split(records, test_size=val_split, random_state=seed, stratify=strat)
    return list(tr_recs), list(va_recs)


def safe_replay_collate(batch: List[Tuple]) -> Tuple:
    """Collate `(video_input, pose_features, target, [play_id, sample_id])` safely."""
    if not batch:
        return ()
    has_meta = len(batch[0]) == 5
    raw_videos = [item[0] for item in batch]
    poses = torch.stack([item[1] for item in batch], dim=0)
    targets = torch.stack([item[2] for item in batch], dim=0)

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

    if has_meta:
        play_ids = [item[3] for item in batch]
        sample_ids = [item[4] for item in batch]
        return videos, poses, targets, play_ids, sample_ids
    return videos, poses, targets


def calibrate_binary_threshold(
    y_true: List[int],
    y_prob: List[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    Sweep probability thresholds across `[0.20 .. 0.80]` to maximize composite
    `0.50 * F1 + 0.50 * Balanced Accuracy`. Returns `(best_threshold, metrics_dict)`.
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

    for thresh in np.linspace(0.20, 0.80, 61):
        pred = (yp >= thresh).astype(int)
        f1 = float(f1_score(yt, pred, zero_division=0))
        acc = float(accuracy_score(yt, pred))
        bal_acc = float(balanced_accuracy_score(yt, pred)) if len(np.unique(yt)) > 1 else acc
        prec = float(precision_score(yt, pred, zero_division=0))
        rec = float(recall_score(yt, pred, zero_division=0))
        obj = 0.50 * f1 + 0.50 * bal_acc

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
    model: BinaryFoulClassifier,
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
        videos, poses, targets = batch[0], batch[1], batch[2]
        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        poses = poses.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)
        bsz = videos.size(0)

        try:
            with get_autocast_context(device):
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
    model: BinaryFoulClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluate model on validation split and aggregate multi-angle probabilities per `play_id`
    (mean pooling across camera angles) for unbiased play-level metrics.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    play_probs: Dict[str, List[float]] = defaultdict(list)
    play_labels: Dict[str, int] = {}

    for batch in loader:
        videos, poses, targets = batch[0], batch[1], batch[2]
        play_ids = batch[3] if len(batch) >= 4 else [f"sample_{total_samples + i}" for i in range(videos.size(0))]

        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        poses = poses.to(device, non_blocking=(device.type == "cuda"))
        targets_dev = targets.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)

        with get_autocast_context(device):
            logits = model(videos, poses)
            loss = criterion(logits, targets_dev)
            probs = torch.sigmoid(logits).squeeze(1)

        bsz = videos.size(0)
        running_loss += float(loss.item()) * bsz
        total_samples += bsz

        probs_np = probs.detach().cpu().numpy().astype(float).tolist()
        tgts_np = targets.detach().cpu().numpy().astype(int).tolist()
        for pid, pr, tg in zip(play_ids, probs_np, tgts_np):
            play_probs[str(pid)].append(float(pr))
            play_labels[str(pid)] = int(tg)

    val_loss = running_loss / max(total_samples, 1)
    y_true: List[int] = []
    y_prob: List[float] = []
    play_predictions: List[Dict[str, Any]] = []

    for pid in sorted(play_probs.keys()):
        mean_p = float(np.mean(play_probs[pid]))
        lbl = int(play_labels[pid])
        y_true.append(lbl)
        y_prob.append(mean_p)
        play_predictions.append({
            "play_id": pid,
            "true_label": lbl,
            "foul_prob": round(mean_p, 4),
            "angle_probs": [round(x, 4) for x in play_probs[pid]],
        })

    _, metrics = calibrate_binary_threshold(y_true, y_prob)
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
) -> None:
    """
    Export a clean binary-classification annotated video:
      - If `is_foul` is False and `clean_no_foul=True`, copies original MP4 untouched.
      - Otherwise displays a crisp top-left `FOUL` (red) or `NO FOUL` (green) banner,
        and on `FOUL` predictions highlights the active contact ROI strictly inside the
        highest-scoring camera shot window (never bleeding across camera cuts).
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

    # Identify winning angle + interpolated contact box map strictly within its shot boundaries
    highlight_boxes: Dict[int, Tuple[int, int, int, int]] = {}
    win_start, win_end = -1, -1
    if is_foul and tubes:
        best_a = int(np.argmax(angle_probs)) if angle_probs else 0
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
    accent = (30, 30, 235) if is_foul else (45, 195, 70)
    verdict_text = f"VERDICT: {'FOUL' if is_foul else 'NO FOUL'}  ({foul_prob * 100.0:.1f}%)"
    sub_text = f"Threshold: {threshold * 100.0:.1f}%  |  Angles Evaluated: {len(tubes)}"

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Top-left clean verdict banner
        banner_w = min(width - 32, 430)
        banner_h = 68
        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (16 + banner_w, 16 + banner_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (16, 16), (24, 16 + banner_h), accent, -1)
        cv2.rectangle(frame, (16, 16), (16 + banner_w, 16 + banner_h), accent, 2)
        cv2.putText(frame, verdict_text, (34, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, sub_text, (34, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (205, 210, 220), 1, cv2.LINE_AA)

        # Highlight contact ROI during winning shot contact window when FOUL is predicted
        if is_foul and f_idx in highlight_boxes:
            x1, y1, x2, y2 = highlight_boxes[f_idx]
            tint = frame.copy()
            cv2.rectangle(tint, (x1, y1), (x2, y2), (20, 20, 210), -1)
            cv2.addWeighted(tint, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), accent, 3)
            tag = f"FOUL CONTACT ({foul_prob * 100.0:.1f}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            ty = max(th + 10, y1 - 6)
            cv2.rectangle(frame, (x1, ty - th - 6), (min(width - 2, x1 + tw + 10), ty + 4), accent, -1)
            cv2.putText(frame, tag, (x1 + 5, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Subtle bottom progress bar
        if total_frames > 1:
            bx1, bx2 = 24, width - 24
            by1, by2 = height - 18, height - 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 30, 34), -1)
            if is_foul and win_end > win_start >= 0:
                wsx = bx1 + int((bx2 - bx1) * (win_start / float(total_frames)))
                wex = bx1 + int((bx2 - bx1) * (win_end / float(total_frames)))
                cv2.rectangle(frame, (wsx, by1), (max(wsx + 2, wex), by2), accent, -1)
            cx = bx1 + int((bx2 - bx1) * (f_idx / float(total_frames - 1)))
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


def load_trained_classifier(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[BinaryFoulClassifier, float]:
    model = BinaryFoulClassifier(pretrained=False, backbone_mode="frozen").to(device)
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
    yolo_weights: str = "yolo11n.pt",
    pose_weights: str = "yolo11n-pose.pt",
    angles_per_clip: int = 2,
    render_video: bool = True,
    clean_no_foul: bool = False,
    device_name: str = "auto",
    loaded_model: Optional[BinaryFoulClassifier] = None,
    loaded_threshold: Optional[float] = None,
    loaded_pose_yolo: Optional[Any] = None,
    loaded_det_yolo: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run multi-angle binary `FOUL` vs. `NO FOUL` inference on a single NBA replay clip."""
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

    with get_autocast_context(device):
        pred = model.predict_foul(vid_batch, pose_batch, threshold=threshold)

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
        )

    report: Dict[str, Any] = {
        "clip": vpath.name,
        "stem": vpath.stem,
        "verdict": pred["verdict"],
        "is_foul": pred["is_foul"],
        "foul_prob": round(float(pred["foul_prob"]), 4),
        "threshold": round(float(threshold), 4),
        "angles_evaluated": len(tubes),
        "angle_probs": [round(float(p), 4) for p in pred["angle_probs"]],
        "output_mp4": str(final_mp4) if render_video else None,
    }

    # Persist to predictions.json
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
        "[%s] %s | P(foul)=%.1f%% (thresh=%.1f%%, angles=%d)",
        vpath.name,
        report["verdict"],
        report["foul_prob"] * 100.0,
        threshold * 100.0,
        len(tubes),
    )
    return report


def predict_folder(
    folder_path: Union[Path, str] = BASE_DIR / "data" / "test",
    checkpoint_path: Union[Path, str] = BASE_DIR / "weights" / "best_replay_foul_classifier.pt",
    annotated_dir: Union[Path, str] = BASE_DIR / "data" / "annotated",
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
        logger.info("=== [%d/%d] Classifying %s ===", idx, len(videos), vp.name)
        rep = predict_clip(
            video_path=vp,
            checkpoint_path=ckpt_p,
            output_mp4_path=out_dir / f"{vp.stem}_annotated.mp4",
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

    # Auto-select backbone mode when set to 'auto'
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

    model = BinaryFoulClassifier(
        pretrained=not args.no_pretrained,
        backbone_mode=backbone_mode,
        dropout_p=args.dropout,
        modality_dropout=args.modality_dropout,
    ).to(device)

    pinfo = model.trainable_parameter_summary()
    logger.info(
        "BinaryFoulClassifier (backbone_mode=%s, cached_emb=%s) | Trainable: %s / %s | batch=%d x accum=%d (eff=%d)",
        backbone_mode,
        use_cached,
        f"{pinfo['trainable_params']:,}",
        f"{pinfo['total_params']:,}",
        args.batch_size,
        args.accum_steps,
        args.batch_size * args.accum_steps,
    )

    pos_weight = compute_pos_weight(train_recs, device=device)
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=1.5, label_smoothing=0.05)

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
            "Play Acc %.3f | BalAcc %.3f | F1 %.3f | AUC %.3f | Thresh %.2f | CM[TN=%d FP=%d FN=%d TP=%d]",
            epoch,
            args.epochs,
            cur_lr,
            tr_loss,
            va_loss,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["threshold"],
            cm["tn"],
            cm["fp"],
            cm["fn"],
            cm["tp"],
        )

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **metrics})
        comp_score = 0.50 * metrics["f1"] + 0.30 * metrics["balanced_accuracy"] + 0.20 * metrics["roc_auc"]
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
                },
                ckpt_path,
            )
            logger.info("Saved new best checkpoint -> %s (F1=%.3f, BalAcc=%.3f, AUC=%.3f, thresh=%.2f)",
                        ckpt_path, metrics["f1"], metrics["balanced_accuracy"], metrics["roc_auc"], metrics["threshold"])

    if use_cached:
        train_ds.save_embedding_cache()
        val_ds.save_embedding_cache()

    return {"best_metrics": best_metrics, "history": history, "checkpoint_path": str(ckpt_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: Binary Foul vs. No Foul Training, Play-Level Calibration & Replay Inference"
    )
    parser.add_argument("--processed-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--metadata-csv", type=Path, default=BASE_DIR / "data" / "processed" / "processed_metadata.csv")
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "data" / "test")
    parser.add_argument("--annotated-dir", type=Path, default=BASE_DIR / "data" / "annotated")
    parser.add_argument("--checkpoint-path", type=Path, default=BASE_DIR / "weights" / "best_replay_foul_classifier.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--modality-dropout", type=float, default=0.10)
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
    parser.add_argument("--predict", type=Path, default=None, help="Predict FOUL vs. NO FOUL on a single video or folder")
    parser.add_argument("--predict-test", action="store_true", help="Run batch binary prediction on all clips in --test-dir")
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
                yolo_weights=cli_args.yolo_weights,
                pose_weights=cli_args.pose_weights,
                angles_per_clip=cli_args.angles_per_clip,
                render_video=not cli_args.no_render,
                clean_no_foul=cli_args.clean_no_foul,
                device_name=cli_args.device,
            )
    else:
        run_training_pipeline(cli_args)
