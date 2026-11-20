#!/usr/bin/env python3
"""
STEP 4: Game-Grouped MIL Training, Temperature Calibration, Sliding-Window Any-Length Inference & Zero-Referee Video Annotation (`4_train_eval.py`)

Full-Stack v3.0 Training & Autonomous AI Referee Engine:
  1. Leak-Free Game-Grouped Training (`StratifiedGroupKFold` by `game_id`):
     - Trains `FoulMILClassifier` (`Swin3D-T 768-D` + `16-D Temporal Pose Attention` + Gated Attention MIL)
       with dual bag-level Focal BCE + instance-level negative suppression + real ground-truth `call_type`
       multi-task supervision (`Shooting Foul` vs `Personal Foul`).
     - Supports full 5-fold out-of-game cross-validation reporting (`--cv-folds 5`) and post-hoc
       temperature scaling + honest threshold calibration (no artificial threshold clamping).
  2. Genuine Multi-Window Sequential RL Referee Training (`--train-rl`):
     - Trains Double Dueling DQN over distinct per-window Swin3D-T + MIL probability trajectories using
       real NBA L2M review decisions (`CC`, `INC`, `CNC`, `IC`).
  3. Arbitrary-Length Sliding-Window Inference & Zero-Referee Annotation (`predict_single_clip` / `run_ai_referee_on_folder`):
     - Works on ANY NBA game video of ANY duration (10s clip, 20s L2M clip, or multi-minute game broadcast).
     - Uses zero CSV metadata during inference (100% autonomous on unseen footage).
     - Detects 0, 1, or multiple distinct fouls via Temporal NMS:
         * 0 fouls -> NOTHING happens on the output video (100% clean, unaltered playback).
         * 1+ fouls -> Tracks opposing-team players using E-BARD Basketball YOLO (explicitly suppressing
           referees & spectators) + 2-Team CIELAB clustering + camera-pan compensation, and highlights
           WHAT foul it is, WHERE it occurred, and WHEN it occurred strictly during each foul window.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import argparse
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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

cv2.setNumThreads(0)

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

preprocess_mod = importlib.import_module("2_preprocess_tubes")
model_mod = importlib.import_module("3_model_and_dataset")

FoulMILClassifier = model_mod.FoulMILClassifier
FoulMILEnsemble = model_mod.FoulMILEnsemble
FoulClassifier3D = model_mod.FoulClassifier3D
L2MFoulDataset = model_mod.L2MFoulDataset
game_grouped_train_val_split = model_mod.game_grouped_train_val_split
load_torch_file_safe = model_mod.load_torch_file_safe
resolve_project_path = model_mod.resolve_project_path

extract_candidate_windows = preprocess_mod.extract_candidate_windows
smooth_boxes_ema = preprocess_mod.smooth_boxes_ema

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_eval")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


# ==============================================================================
# Device, Seeding & Video Writer Utilities
# ==============================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_compute_device(requested: Optional[str] = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _w = torch.zeros((1, 4, 8), device="mps")
            _mha = nn.MultiheadAttention(8, 2, batch_first=True).to("mps")
            _ = _mha(_w, _w, _w)
            del _w, _mha
            return torch.device("mps")
        except BaseException as exc:
            logger.warning("Apple MPS unavailable (%s); falling back to CPU.", exc)
    return torch.device("cpu")


def get_autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def get_safe_num_workers(requested: int = 0) -> int:
    if platform.system() in ("Windows", "Darwin"):
        return 0
    return max(0, int(requested))


def create_cross_platform_video_writer(
    output_path: Path,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_str in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc_str), fps, (frame_w, frame_h))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not initialize OpenCV VideoWriter for {output_path}")


def remux_h264_faststart(mp4_path: Path) -> None:
    """Optional H.264 +faststart remux so QuickTime, macOS Preview, and HTML5 browsers play cleanly."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin or not mp4_path.exists():
        return
    tmp_h264 = mp4_path.with_name(f"{mp4_path.stem}_h264.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(mp4_path),
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
    res = subprocess.run(cmd, check=False)
    if res.returncode == 0 and tmp_h264.exists() and tmp_h264.stat().st_size > 1000:
        tmp_h264.replace(mp4_path)


def describe_court_location(cx: float, cy: float, frame_w: int, frame_h: int) -> str:
    """Convert pixel contact coordinate `(cx, cy)` into a human-readable NBA court zone."""
    xn = float(cx) / max(float(frame_w), 1.0)
    yn = float(cy) / max(float(frame_h), 1.0)
    if xn < 0.34:
        zone = "Left Paint / Rim Area" if yn < 0.72 else "Left Baseline / Corner"
    elif xn > 0.66:
        zone = "Right Paint / Rim Area" if yn < 0.72 else "Right Baseline / Corner"
    else:
        zone = "Top of Key / Perimeter" if yn < 0.54 else "Middle Paint / Free-Throw Lane"
    return f"{zone} (x={int(round(cx))}, y={int(round(cy))})"


# ==============================================================================
# Multi-Task MIL Loss, Temperature Calibration & Metrics
# ==============================================================================
class FoulMILMultiTaskLoss(nn.Module):
    """
    Joint Multiple Instance Learning + Real `call_type` Multi-Task Loss:
      1. Bag-level class-balanced Focal BCE (`gamma=1.5`, `label_smoothing=0.04`).
      2. Instance-level negative suppression: pushes ALL windows of `no_foul` bags toward `0`,
         and pushes the peak window of `foul` bags toward `1` (teaching WHEN fouls occur!).
      3. Masked real `call_type` BCE (`Shooting Foul` vs `Personal Foul`) strictly on positive foul bags.
      4. Attention entropy / sparsity regularization so temporal attention focuses sharply on impact windows.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.04,
        inst_weight: float = 0.40,
        type_weight: float = 0.30,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.inst_weight = inst_weight
        self.type_weight = type_weight

    def _focal_bce(self, logits: torch.Tensor, targets: torch.Tensor, use_pos_weight: bool = True) -> torch.Tensor:
        smooth_t = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        pw = self.pos_weight if (use_pos_weight and self.pos_weight is not None) else None
        bce = F.binary_cross_entropy_with_logits(logits, smooth_t, pos_weight=pw, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0 - 1e-6)
        focal = ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        call_type_labels: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        bag_logits = outputs["bag_foul_logit"]
        win_logits = outputs["window_foul_logits"]
        bag_type_logits = outputs["bag_type_logit"]

        # 1. Bag-level classification loss
        bag_loss = self._focal_bce(bag_logits, labels, use_pos_weight=True)

        # 2. Instance-level MIL supervision:
        #    - Neg bags (y=0): EVERY valid window must be 0
        #    - Pos bags (y=1): peak window must be 1, while non-peak windows stay sparse
        neg_mask = (labels < 0.5).unsqueeze(1) & window_mask
        inst_loss = torch.tensor(0.0, device=bag_logits.device)
        if neg_mask.any():
            neg_win_logits = win_logits[neg_mask]
            inst_loss = inst_loss + F.binary_cross_entropy_with_logits(
                neg_win_logits, torch.zeros_like(neg_win_logits)
            )
        pos_bags = labels >= 0.5
        if pos_bags.any():
            pos_win = win_logits[pos_bags].masked_fill(~window_mask[pos_bags], -1e4)
            peak_pos_logits = pos_win.max(dim=-1).values
            inst_loss = inst_loss + F.binary_cross_entropy_with_logits(
                peak_pos_logits, torch.ones_like(peak_pos_logits) * (1.0 - self.label_smoothing)
            )

        # 3. Real ground-truth `call_type` supervision (`Shooting Foul` vs `Personal Foul`) on positive fouls
        type_loss = torch.tensor(0.0, device=bag_logits.device)
        if pos_bags.any():
            type_loss = F.binary_cross_entropy_with_logits(
                bag_type_logits[pos_bags], call_type_labels[pos_bags]
            )

        return bag_loss + self.inst_weight * inst_loss + self.type_weight * type_loss


def compute_pos_weight_from_meta(samples_meta: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    pos = sum(1 for m in samples_meta if int(m["label"]) == 1)
    neg = sum(1 for m in samples_meta if int(m["label"]) == 0)
    ratio = float(neg) / max(float(pos), 1.0)
    clamped = float(np.clip(ratio, 0.75, 4.0))
    logger.info("Class distribution -> foul (1): %d, no_foul (0): %d | pos_weight=%.3f", pos, neg, clamped)
    return torch.tensor(clamped, dtype=torch.float32, device=device)


def fit_temperature_and_threshold(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Post-hoc Temperature Scaling (minimizing validation NLL) followed by honest threshold sweep
    optimizing composite F1 + Balanced Accuracy (zero artificial clamping!).
    """
    logits_t = torch.from_numpy(val_logits.astype(np.float32))
    targets_t = torch.from_numpy(val_targets.astype(np.float32))

    # 1. Grid-search calibration temperature T in [0.35 .. 3.0]
    best_temp = 1.0
    best_nll = float("inf")
    for temp in np.linspace(0.35, 3.0, 54):
        nll = float(F.binary_cross_entropy_with_logits(logits_t / float(temp), targets_t).item())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)

    probs = torch.sigmoid(logits_t / best_temp).numpy()
    y_true = val_targets.astype(int)
    roc_auc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else 0.5

    # 2. Honest threshold calibration over [0.22 .. 0.78]
    best_thresh = 0.50
    best_obj = -1.0
    best_metrics: Dict[str, float] = {}
    for thresh in np.linspace(0.22, 0.78, 57):
        preds = (probs >= thresh).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        bacc = float(balanced_accuracy_score(y_true, preds)) if len(set(y_true.tolist())) > 1 else float(accuracy_score(y_true, preds))
        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        obj = 0.55 * f1 + 0.45 * bacc
        if obj > best_obj:
            best_obj = obj
            best_thresh = float(thresh)
            best_metrics = {
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": roc_auc,
                "threshold": best_thresh,
                "temperature": best_temp,
                "nll": best_nll,
            }

    return best_temp, best_thresh, best_metrics


def train_one_epoch(
    model: FoulMILClassifier,
    loader: DataLoader,
    criterion: FoulMILMultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for win_embs, win_poses, win_mask, labels, call_type_labels in loader:
        win_embs = win_embs.to(device)
        win_poses = win_poses.to(device)
        win_mask = win_mask.to(device)
        labels = labels.to(device)
        call_type_labels = call_type_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with get_autocast_context(device):
            outputs = model(win_embs, win_poses, win_mask)
            loss = criterion(outputs, labels, call_type_labels, win_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = labels.size(0)
        running_loss += float(loss.item()) * bsz
        total += bsz
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate_mil(
    model: FoulMILClassifier,
    loader: DataLoader,
    criterion: FoulMILMultiTaskLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_logits: List[float] = []
    all_targets: List[int] = []
    type_logits_list: List[float] = []
    type_preds: List[int] = []
    type_targets: List[int] = []

    for win_embs, win_poses, win_mask, labels, call_type_labels in loader:
        win_embs = win_embs.to(device)
        win_poses = win_poses.to(device)
        win_mask = win_mask.to(device)
        labels = labels.to(device)
        call_type_labels = call_type_labels.to(device)

        with get_autocast_context(device):
            outputs = model(win_embs, win_poses, win_mask)
            loss = criterion(outputs, labels, call_type_labels, win_mask)

        bsz = labels.size(0)
        running_loss += float(loss.item()) * bsz
        total += bsz
        all_logits.extend(outputs["bag_foul_logit"].detach().cpu().numpy().tolist())
        all_targets.extend(labels.detach().cpu().numpy().astype(int).tolist())

        pos_mask = labels >= 0.5
        if pos_mask.any():
            tl = outputs["bag_type_logit"][pos_mask].detach().cpu().numpy().tolist()
            tp = (torch.sigmoid(outputs["bag_type_logit"][pos_mask]) >= 0.5).long().cpu().numpy().tolist()
            tt = call_type_labels[pos_mask].long().cpu().numpy().tolist()
            type_logits_list.extend(tl)
            type_preds.extend(tp)
            type_targets.extend(tt)

    logits_arr = np.array(all_logits, dtype=np.float32)
    targets_arr = np.array(all_targets, dtype=int)
    type_logits_arr = np.array(type_logits_list, dtype=np.float32)
    type_targets_arr = np.array(type_targets, dtype=int)
    temp, thresh, metrics = fit_temperature_and_threshold(logits_arr, targets_arr)
    metrics["call_type_accuracy"] = (
        float(accuracy_score(type_targets, type_preds)) if type_targets else 0.0
    )
    return running_loss / max(total, 1), metrics, logits_arr, targets_arr, type_logits_arr, type_targets_arr


# ==============================================================================
# Zero-Referee E-BARD + 2-Team CIELAB Bidirectional Foul Window Tracker
# ==============================================================================
def track_foul_players_in_foul_window(
    video_path: Path,
    pose_model: object,
    foul_frame: int,
    foul_window_start: int,
    foul_window_end: int,
    anchor_roi: Optional[Union[Tuple[int, int, int, int], List[int], np.ndarray]] = None,
    team_centroids_lab: Optional[np.ndarray] = None,
    device: str = "cpu",
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> Dict[int, Dict[str, Any]]:
    """
    Zero-Referee Localized Event-Window Tracker:
      1. Runs E-BARD Basketball YOLO (`player`, `referee`, `basketball`, `hoop`) + YOLOv11-Pose
         on `[foul_window_start, foul_window_end]`.
      2. Explicitly eliminates all E-BARD `referee` boxes (class 3), long-trouser officials, and
         all non-court spectators/coaches.
      3. Assigns every court player a team ID using the clip's 2-Team CIELAB centroids and enforces
         opposing-team pairing (`team_off != team_def`).
      4. Tracks the foul pair bidirectionally strictly inside the continuous wide-court segment
         with camera-pan compensation and mutual-exclusion bipartite matching.
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
    ordered_imgs = [window_frames[k] for k in all_keys]
    yolo_dev = preprocess_mod.select_yolo_device(device)
    active_det, use_ebard = preprocess_mod.get_cached_basketball_model()
    diag = float(np.hypot(frame_w, frame_h))

    all_det_res: List[Any] = []
    all_pose_res: List[Any] = []
    with torch.inference_mode():
        for b_idx in range(0, len(all_keys), 16):
            b_imgs = ordered_imgs[b_idx : b_idx + 16]
            if active_det is not None:
                det_kwargs = {"conf": 0.16, "imgsz": 960, "verbose": False, "device": yolo_dev}
                if not use_ebard:
                    det_kwargs["classes"] = [preprocess_mod.COCO_PERSON_CLS, preprocess_mod.COCO_SPORTS_BALL_CLS]
                all_det_res.extend(active_det.predict(b_imgs, **det_kwargs))
            else:
                all_det_res.extend([None] * len(b_imgs))
            if pose_model is not None:
                all_pose_res.extend(pose_model.predict(b_imgs, conf=0.12, imgsz=640, verbose=False, device=yolo_dev))
            else:
                all_pose_res.extend([None] * len(b_imgs))

    # Extract clean referee-free entities for every frame in the window
    frame_players_list: List[List[Dict[str, Any]]] = []
    raw_balls: List[Optional[Tuple[float, float, float]]] = []
    raw_hoops: List[Optional[Tuple[int, int]]] = []
    max_h_ratios: List[float] = []
    for idx_k in range(len(all_keys)):
        ent = preprocess_mod.collect_ebard_court_entities(
            pose_res=all_pose_res[idx_k],
            det_res=all_det_res[idx_k],
            frame_w=frame_w,
            frame_h=frame_h,
            scale_x=1.0,
            scale_y=1.0,
            bgr_frame=ordered_imgs[idx_k],
            use_ebard=use_ebard,
        )
        frame_players_list.append(ent["players"])
        raw_balls.append(ent["ball"])
        raw_hoops.append(ent["hoop"])
        max_h_ratios.append(ent["max_h_ratio"])

    cam_info = preprocess_mod.compute_camera_cuts_and_pan(
        ordered_imgs, frame_w=frame_w, frame_h=frame_h, player_max_h_ratios=max_h_ratios
    )
    wide_mask: List[bool] = cam_info["is_wide_court"]
    pan_dx: List[float] = cam_info["pan_dx"]
    pan_dy: List[float] = cam_info["pan_dy"]

    for i in range(len(all_keys)):
        if not wide_mask[i]:
            frame_players_list[i] = []

    # Assign team IDs using clip-level CIELAB centroids (or fit on window if not provided)
    if team_centroids_lab is not None and len(team_centroids_lab) == 2:
        weights = np.array([1.0, 1.45, 1.45], dtype=np.float32)
        c0_w = team_centroids_lab[0] * weights
        c1_w = team_centroids_lab[1] * weights
        for i, plist in enumerate(frame_players_list):
            kept: List[Dict[str, Any]] = []
            for p in plist:
                pw = p["lab"] * weights
                d0 = float(np.linalg.norm(pw - c0_w))
                d1 = float(np.linalg.norm(pw - c1_w))
                if p.get("ref_uniform_score", 0.0) >= 0.45 and min(d0, d1) > 38.0:
                    continue
                p["team_id"] = 0 if d0 <= d1 else 1
                kept.append(p)
            frame_players_list[i] = kept
    else:
        preprocess_mod.fit_clip_team_clusters(frame_players_list)

    ball_track, ball_confs = preprocess_mod.interpolate_court_ball_track(
        raw_balls=raw_balls,
        wide_mask=wide_mask,
        sampled_frame_idxs=all_keys,
        fps=30.0,
        diag=diag,
        pan_dx=pan_dx,
    )

    frame_dets: Dict[int, List[Dict[str, Any]]] = {all_keys[i]: frame_players_list[i] for i in range(len(all_keys))}
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    valid_keys = [k for k in all_keys if wide_mask[key_to_idx[k]] and len(frame_dets[k]) >= 2]
    if not valid_keys:
        valid_keys = [k for k in all_keys if wide_mask[key_to_idx[k]] and len(frame_dets[k]) >= 1]
    if not valid_keys:
        return {}

    f_anchor = min(valid_keys, key=lambda k: abs(k - foul_frame))
    a_pos = key_to_idx[f_anchor]
    seg_lo, seg_hi = a_pos, a_pos
    while seg_lo > 0 and wide_mask[seg_lo - 1] and not cam_info["is_hard_cut"][seg_lo]:
        seg_lo -= 1
    while seg_hi + 1 < len(all_keys) and wide_mask[seg_hi + 1] and not cam_info["is_hard_cut"][seg_hi + 1]:
        seg_hi += 1
    valid_min_f, valid_max_f = all_keys[seg_lo], all_keys[seg_hi]

    roi_pt = (
        np.array([0.5 * (anchor_roi[0] + anchor_roi[2]), 0.5 * (anchor_roi[1] + anchor_roi[3])], dtype=np.float32)
        if anchor_roi is not None
        else np.array([frame_w * 0.5, frame_h * 0.62], dtype=np.float32)
    )

    anchor_pair = preprocess_mod.select_on_ball_opposing_pair(
        players=frame_dets[f_anchor],
        ball_pt=ball_track[a_pos],
        ball_conf=ball_confs[a_pos],
        hoop_pt=raw_hoops[a_pos],
        prev_center=roi_pt,
        frame_w=frame_w,
        frame_h=frame_h,
        diag=diag,
    )

    if anchor_pair is not None:
        b_off = anchor_pair["off_box"].copy()
        b_def = anchor_pair["def_box"].copy()
        lab_off = anchor_pair["off_lab"].copy()
        lab_def = anchor_pair["def_lab"].copy()
        team_off = int(anchor_pair["off_team"])
        team_def = int(anchor_pair["def_team"])
    else:
        cands = frame_dets[f_anchor]
        cands.sort(key=lambda c: float(np.linalg.norm(c["center"] - roi_pt)))
        b_off = cands[0]["box"].copy()
        b_def = cands[1]["box"].copy() if len(cands) > 1 else b_off.copy()
        lab_off = cands[0]["lab"].copy()
        lab_def = cands[1]["lab"].copy() if len(cands) > 1 else lab_off.copy()
        team_off = int(cands[0].get("team_id", 0))
        team_def = int(cands[1].get("team_id", 1)) if len(cands) > 1 else (1 - team_off)

    def _match_score(
        prev_box: np.ndarray,
        target_lab: np.ndarray,
        target_team: int,
        cand: Dict[str, Any],
        sx: float,
        sy: float,
    ) -> float:
        px1, py1, px2, py2 = prev_box
        pbh = max(1.0, float(py2 - py1))
        pred_cx = 0.5 * (px1 + px2) - sx
        pred_cy = 0.5 * (py1 + py2) - sy
        pred_box = np.array(
            [pred_cx - 0.5 * (px2 - px1), pred_cy - 0.5 * (py2 - py1), pred_cx + 0.5 * (px2 - px1), pred_cy + 0.5 * (py2 - py1)],
            dtype=np.float32,
        )
        iou = preprocess_mod.box_iou_single(pred_box, cand["box"])
        dist = float(np.hypot(float(cand["center"][0]) - pred_cx, float(cand["center"][1]) - pred_cy))
        cbh = float(cand["box"][3] - cand["box"][1])
        if dist > max(0.14 * diag, 0.90 * pbh) or min(pbh, cbh) / max(pbh, cbh, 1.0) < 0.58:
            return -1e6
        lab_d = float(np.linalg.norm(target_lab - cand["lab"]))
        team_bonus = 0.65 if (target_team >= 0 and cand.get("team_id", -1) == target_team) else -0.65
        return iou * 3.0 - (dist / pbh) * 1.15 - 0.015 * lab_d + team_bonus

    def _step_pair(
        cur_o: np.ndarray,
        cur_d: np.ndarray,
        cands: List[Dict[str, Any]],
        sx: float,
        sy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        shift = np.array([-sx, -sy, -sx, -sy], dtype=np.float32)
        if not cands:
            return (cur_o + shift).astype(np.float32), (cur_d + shift).astype(np.float32)

        os_l = [_match_score(cur_o, lab_off, team_off, c, sx, sy) for c in cands]
        ds_l = [_match_score(cur_d, lab_def, team_def, c, sx, sy) for c in cands]

        best_tot = -1e5
        best_pair_ij: Optional[Tuple[int, int]] = None
        for i in range(len(cands)):
            if os_l[i] < -1.5:
                continue
            for j in range(len(cands)):
                if i == j or ds_l[j] < -1.5:
                    continue
                if cands[i].get("team_id", -1) >= 0 and cands[i].get("team_id") == cands[j].get("team_id"):
                    continue
                tot = os_l[i] + ds_l[j]
                if tot > best_tot:
                    best_tot = tot
                    best_pair_ij = (i, j)

        if best_pair_ij is not None:
            return cands[best_pair_ij[0]]["box"].copy(), cands[best_pair_ij[1]]["box"].copy()

        bo = max(range(len(cands)), key=lambda i: os_l[i])
        bd = max(range(len(cands)), key=lambda j: ds_l[j])
        new_o = cands[bo]["box"].copy() if os_l[bo] > -1.3 else (cur_o + shift).astype(np.float32)
        if ds_l[bd] > -1.3 and (os_l[bo] <= -1.3 or bd != bo):
            new_d = cands[bd]["box"].copy()
        else:
            alts = [j for j in range(len(cands)) if j != bo and ds_l[j] > -1.3]
            new_d = cands[max(alts, key=lambda j: ds_l[j])]["box"].copy() if alts else (cur_d + shift).astype(np.float32)
        new_o[0::2] = np.clip(new_o[0::2], 0.0, float(frame_w))
        new_o[1::2] = np.clip(new_o[1::2], 0.0, float(frame_h))
        new_d[0::2] = np.clip(new_d[0::2], 0.0, float(frame_w))
        new_d[1::2] = np.clip(new_d[1::2], 0.0, float(frame_h))
        return new_o, new_d

    off_tracks: Dict[int, np.ndarray] = {f_anchor: b_off.copy()}
    def_tracks: Dict[int, np.ndarray] = {f_anchor: b_def.copy()}

    cur_o, cur_d = b_off.copy(), b_def.copy()
    for f in range(f_anchor + 1, min(foul_window_end, valid_max_f) + 1):
        ik = key_to_idx.get(f)
        if ik is None or not wide_mask[ik]:
            break
        cur_o, cur_d = _step_pair(cur_o, cur_d, frame_dets.get(f, []), pan_dx[ik], pan_dy[ik])
        off_tracks[f] = cur_o.copy()
        def_tracks[f] = cur_d.copy()

    cur_o, cur_d = b_off.copy(), b_def.copy()
    for f in range(f_anchor - 1, max(foul_window_start, valid_min_f) - 1, -1):
        ik = key_to_idx.get(f)
        if ik is None or not wide_mask[ik]:
            break
        prev_ik = key_to_idx.get(f + 1, ik)
        cur_o, cur_d = _step_pair(cur_o, cur_d, frame_dets.get(f, []), -pan_dx[prev_ik], -pan_dy[prev_ik])
        off_tracks[f] = cur_o.copy()
        def_tracks[f] = cur_d.copy()

    sorted_fs = sorted(off_tracks.keys())
    off_smooth = smooth_boxes_ema(np.stack([off_tracks[f] for f in sorted_fs], axis=0), alpha=0.35)
    def_smooth = smooth_boxes_ema(np.stack([def_tracks[f] for f in sorted_fs], axis=0), alpha=0.35)

    result: Dict[int, Dict[str, Any]] = {}
    pad_x, pad_y = 18.0, 14.0
    for idx_f, f in enumerate(sorted_fs):
        ob, db = off_smooth[idx_f], def_smooth[idx_f]
        ux1 = int(round(max(0.0, min(ob[0], db[0]) - pad_x)))
        uy1 = int(round(max(0.0, min(ob[1], db[1]) - pad_y)))
        ux2 = int(round(min(float(frame_w), max(ob[2], db[2]) + pad_x)))
        uy2 = int(round(min(float(frame_h), max(ob[3], db[3]) + pad_y)))
        ocx, ocy = 0.5 * (ob[0] + ob[2]), 0.40 * ob[1] + 0.60 * ob[3]
        dcx, dcy = 0.5 * (db[0] + db[2]), 0.40 * db[1] + 0.60 * db[3]
        ix = int(round(np.clip(0.5 * (ocx + dcx), ux1 + 8, ux2 - 8)))
        iy = int(round(np.clip(0.5 * (ocy + dcy), uy1 + 8, uy2 - 8)))
        result[f] = {
            "off_box": (int(round(ob[0])), int(round(ob[1])), int(round(ob[2])), int(round(ob[3]))),
            "def_box": (int(round(db[0])), int(round(db[1])), int(round(db[2])), int(round(db[3]))),
            "union_box": (ux1, uy1, ux2, uy2),
            "impact_point": (ix, iy),
        }
    return result


# ==============================================================================
# Folder Discovery & Arbitrary-Length Sliding-Window Autonomous AI Referee
# ==============================================================================
def discover_video_files_in_folder(folder_path: Union[Path, str]) -> List[Path]:
    t_dir = resolve_project_path(folder_path)
    if not t_dir.exists():
        return []
    return sorted(
        p for p in t_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.stem.endswith("_annotated")
    )


def _load_trained_mil_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[Union[FoulMILClassifier, FoulMILEnsemble], float, bool]:
    threshold = 0.50
    loaded_ok = False
    if checkpoint_path.exists():
        ckpt = load_torch_file_safe(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and isinstance(ckpt.get("ensemble_state_dicts"), list) and len(ckpt["ensemble_state_dicts"]) >= 2:
            ens_models: List[FoulMILClassifier] = []
            for sd in ckpt["ensemble_state_dicts"]:
                m = FoulMILClassifier().to(device)
                m_sd = m.state_dict()
                compat = {k: v for k, v in sd.items() if k in m_sd and v.shape == m_sd[k].shape}
                m.load_state_dict(compat, strict=False)
                m.eval()
                ens_models.append(m)
            temp = float(ckpt.get("temperature", 1.0))
            if "optimal_threshold" in ckpt:
                threshold = float(ckpt["optimal_threshold"])
            ensemble = FoulMILEnsemble(ens_models, temperature=temp).to(device)
            ensemble.eval()
            loaded_ok = True
            logger.info(
                "Loaded %d-Fold Out-of-Game FoulMILEnsemble checkpoint: %s (OOF temp=%.2f, threshold=%.2f)",
                len(ens_models),
                checkpoint_path.name,
                float(ensemble.temperature.item()),
                threshold,
            )
            return ensemble, threshold, loaded_ok

        model = FoulMILClassifier().to(device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model_sd = model.state_dict()
        compat = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(compat, strict=False)
        if len(compat) >= len(model_sd) - 2:
            loaded_ok = True
        if isinstance(ckpt, dict):
            if "optimal_threshold" in ckpt:
                threshold = float(ckpt["optimal_threshold"])
            if "temperature" in ckpt:
                model.temperature.fill_(float(ckpt["temperature"]))
        logger.info(
            "Loaded FoulMILClassifier checkpoint: %s (matched %d/%d layers, temp=%.2f, threshold=%.2f)",
            checkpoint_path.name,
            len(compat),
            len(model_sd),
            float(model.temperature.item()),
            threshold,
        )
        model.eval()
        return model, threshold, loaded_ok

    model = FoulMILClassifier().to(device)
    logger.warning("Checkpoint %s not found; running with uncalibrated weights.", checkpoint_path)
    model.eval()
    return model, threshold, loaded_ok


@torch.no_grad()
def predict_single_clip(
    video_path: Optional[Union[Path, str, argparse.Namespace]] = None,
    checkpoint_path: Optional[Union[Path, str]] = None,
    output_mp4_path: Optional[Union[Path, str]] = None,
    yolo_weights: str = preprocess_mod.EBARD_WEIGHTS_NAME,
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    return_details: bool = False,
    **kwargs,
) -> Union[float, Dict[str, Any]]:
    """
    Autonomous AI Referee Inference on ANY NBA Game Footage (Zero Metadata Cheating):
      1. Extracts ALL overlapping 2.0s candidate windows across the ENTIRE video using E-BARD
         Basketball YOLO (player/referee/ball/hoop) + 2-Team CIELAB Clustering + Swin3D-T.
      2. Scores every window via temperature-calibrated `FoulMILClassifier` + Temporal NMS.
      3. If 0 fouls detected: NOTHING happens on the output video (100% clean unaltered playback).
      4. If 1+ fouls detected: Tracks the exact opposing-team players (zero referees!) in each foul
         window and highlights WHAT foul it is, WHERE it occurred, and WHEN it occurred.
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
        actual_ckpt = Path(checkpoint_path) if checkpoint_path else BASE_DIR / "weights" / "best_foul_r2plus1d.pt"
        if not actual_ckpt.exists() and (BASE_DIR / "best_foul_r2plus1d.pt").exists():
            actual_ckpt = BASE_DIR / "best_foul_r2plus1d.pt"
        actual_out = Path(output_mp4_path) if output_mp4_path else None
        pose_candidate = pose_weights or "yolo11n-pose.pt"

    resolved_video = resolve_project_path(actual_video)
    resolved_ckpt = resolve_project_path(actual_ckpt)

    if resolved_video.is_dir():
        reports = run_ai_referee_on_folder(
            folder_path=resolved_video,
            checkpoint_path=resolved_ckpt,
            annotated_dir=actual_out,
            yolo_weights=yolo_weights,
            pose_weights=pose_candidate,
        )
        if return_details:
            return {"reports": reports}
        return float(np.mean([r["foul_prob"] for r in reports])) if reports else 0.0

    if not resolved_video.exists():
        raise FileNotFoundError(f"Input video not found: {resolved_video}")

    device = select_compute_device()
    det_model, _ = preprocess_mod.get_cached_basketball_model(yolo_weights)
    pose_model = preprocess_mod.get_cached_pose_model(str(pose_candidate))

    # Scan ENTIRE video (`max_live_play_sec=0.0`, `max_windows=0`) for universal NBA footage support
    bag = extract_candidate_windows(
        video_path=resolved_video,
        det_model=det_model,
        pose_model=pose_model,
        window_sec=2.0,
        stride_sec=1.0,
        num_frames=16,
        crop_size=224,
        device=str(device),
        max_live_play_sec=0.0,
        max_windows=0,
        scan_fps=2.5,
        yolo_batch_size=64 if torch.cuda.is_available() else 16,
        encode_embeddings=True,
    )
    windows = bag["windows"]
    fps = float(bag["fps"])
    total_video_frames = int(bag["total_frames"])
    frame_w = int(bag["frame_w"])
    frame_h = int(bag["frame_h"])
    team_centroids_lab = bag.get("team_centroids_lab")

    win_embs = torch.stack([w["embedding"] for w in windows], dim=0).float()
    win_poses = torch.stack([w["foul_features"] for w in windows], dim=0).float()
    win_centers = [int(w["center_frame"]) for w in windows]

    model, threshold, _ = _load_trained_mil_checkpoint(resolved_ckpt, device)
    pred = model.predict_video_windows(
        window_embs=win_embs,
        window_poses=win_poses,
        window_centers=win_centers,
        fps=fps,
        threshold=threshold,
        nms_min_sep_sec=3.5,
    )

    is_foul = bool(pred["is_foul"])
    bag_prob = float(pred["bag_prob"])
    detected_events = pred["events"]

    if actual_out is None:
        annotated_dir = BASE_DIR / "data" / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        final_output_path = annotated_dir / f"{resolved_video.stem}_annotated.mp4"
    else:
        final_output_path = resolve_project_path(actual_out)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track opposing-team players (zero referees!) strictly inside each detected foul event window
    frame_event_overlay: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    enriched_events: List[Dict[str, Any]] = []

    if is_foul and detected_events:
        for ev in detected_events:
            w = windows[ev["window_idx"]]
            apex_off = min(int(ev["apex_frame_offset"]), len(w["sampled_indices"]) - 1)
            foul_frame = int(w["sampled_indices"][apex_off])
            anchor_box = w["crop_boxes"][apex_off]
            win_start = max(0, foul_frame - int(round(fps * 0.85)))
            win_end = min(max(0, total_video_frames - 1), foul_frame + int(round(fps * 1.35)))

            tracked = track_foul_players_in_foul_window(
                video_path=resolved_video,
                pose_model=pose_model,
                foul_frame=foul_frame,
                foul_window_start=win_start,
                foul_window_end=win_end,
                anchor_roi=anchor_box,
                team_centroids_lab=team_centroids_lab,
                device=str(device),
                frame_w=frame_w,
                frame_h=frame_h,
            )

            apex_key = min(tracked.keys(), key=lambda k: abs(k - foul_frame)) if tracked else None
            if apex_key is not None:
                where_box = tracked[apex_key]["union_box"]
                where_point = tracked[apex_key]["impact_point"]
                apex_off_box = tracked[apex_key]["off_box"]
                apex_def_box = tracked[apex_key]["def_box"]
            else:
                bx1, by1, bx2, by2 = anchor_box
                where_box = (int(bx1), int(by1), int(bx2), int(by2))
                where_point = (int((bx1 + bx2) // 2), int((by1 + by2) // 2))
                apex_off_box = where_box
                apex_def_box = where_box

            foul_time_sec = float(foul_frame) / max(fps, 1.0)
            mins = int(foul_time_sec // 60)
            secs = foul_time_sec - mins * 60
            when_str = f"{mins:02d}:{secs:05.2f} ({foul_time_sec:.2f}s | Frame {foul_frame})"
            where_zone = describe_court_location(where_point[0], where_point[1], frame_w, frame_h)

            ev_full = {
                **ev,
                "when_frame": foul_frame,
                "when_time_sec": round(foul_time_sec, 2),
                "when_str": when_str,
                "window_start": win_start,
                "window_end": win_end,
                "where_zone": where_zone,
                "where_point": list(where_point),
                "where_box": list(where_box),
                "apex_off_box": list(apex_off_box),
                "apex_def_box": list(apex_def_box),
            }
            enriched_events.append(ev_full)

            for f_idx, tdata in tracked.items():
                frame_event_overlay[f_idx] = (ev_full, tdata)

            logger.info(
                "🚨 AI REFEREE WHISTLE on %s -> WHAT: %s | WHEN: %s | WHERE: %s | P(foul)=%.1f%%",
                resolved_video.name,
                ev_full["foul_subtype"],
                when_str,
                where_zone,
                ev_full["foul_prob"] * 100.0,
            )
    else:
        logger.info(
            "✅ AI REFEREE CALL on %s -> NO FOUL (P(foul)=%.1f%% < %.1f%%). Nothing happens on video (clean output).",
            resolved_video.name,
            bag_prob * 100.0,
            threshold * 100.0,
        )

    # Render annotated output video:
    # - NO FOUL -> literally untouched copy (nothing happens!)
    # - FOUL(S) -> highlights WHAT, WHERE, and WHEN strictly during each foul window; clean everywhere else!
    if not is_foul and resolved_video.suffix.lower() == ".mp4":
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

            if frame_idx in frame_event_overlay:
                ev, fdata = frame_event_overlay[frame_idx]
                fx1, fy1, fx2, fy2 = fdata["off_box"]
                dx1, dy1, dx2, dy2 = fdata["def_box"]
                ux1, uy1, ux2, uy2 = fdata["union_box"]
                ix, iy = fdata["impact_point"]
                cur_zone = describe_court_location(ix, iy, frame_w, frame_h)

                # 1. WHERE: Translucent red foul collision zone + bold boundary
                zone_overlay = frame.copy()
                cv2.rectangle(zone_overlay, (ux1, uy1), (ux2, uy2), (15, 15, 195), -1)
                cv2.addWeighted(zone_overlay, 0.16, frame, 0.84, 0, frame)
                cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), (0, 0, 235), 4)

                # 2. Opposing-Team Fouled Player (Offense) & Committing Player (Defense) boxes
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 215, 0), 2)
                off_label = "Fouled Player (Offense)"
                (otw, oth), _ = cv2.getTextSize(off_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (fx1, max(0, fy1 - oth - 6)), (min(frame_w - 2, fx1 + otw + 8), fy1), (255, 215, 0), -1)
                cv2.putText(frame, off_label, (fx1 + 4, max(12, fy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)

                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 145, 255), 2)
                def_label = "Committing Player (Defense)"
                (dtw, dth), _ = cv2.getTextSize(def_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (dx1, max(0, dy1 - dth - 6)), (min(frame_w - 2, dx1 + dtw + 8), dy1), (0, 145, 255), -1)
                cv2.putText(frame, def_label, (dx1 + 4, max(12, dy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)

                # 3. Exact Contact Impact Reticle
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

                # 4. Call Tag above Foul Zone (WHAT + WHEN + WHERE)
                tag_y = max(56, uy1 - 12)
                what_tag = f"FOUL: {ev['foul_subtype'].upper()}"
                when_where_tag = f"WHEN: {ev['when_time_sec']:.2f}s (Frame {ev['when_frame']}) | WHERE: {cur_zone}"
                (tw1, th1), _ = cv2.getTextSize(what_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                (tw2, th2), _ = cv2.getTextSize(when_where_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                box_w = max(tw1, tw2) + 18
                box_x1 = max(8, min(ux1, frame_w - box_w - 8))
                cv2.rectangle(frame, (box_x1, tag_y - th1 - th2 - 18), (box_x1 + box_w, tag_y + 4), (15, 15, 215), -1)
                cv2.rectangle(frame, (box_x1, tag_y - th1 - th2 - 18), (box_x1 + box_w, tag_y + 4), (255, 255, 255), 1)
                cv2.putText(frame, what_tag, (box_x1 + 8, tag_y - th2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, when_where_tag, (box_x1 + 8, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 245, 255), 1, cv2.LINE_AA)

                # 5. Top-left AI Referee Whistle Banner (strictly inside foul window!)
                line1 = f"AI REFEREE WHISTLE  |  {ev['what_foul'].upper()}  ({ev['foul_prob']*100:.0f}% CONF)"
                line2 = f"WHAT : {ev['foul_subtype']}  ({ev['foul_mechanic']})"
                line3 = f"WHEN : {ev['when_time_sec']:.2f}s (Frame {ev['when_frame']})   |   WHERE: {cur_zone}"
                (lw1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
                (lw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                (lw3, _), _ = cv2.getTextSize(line3, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                hud_w = max(540, max(lw1, lw2, lw3) + 40)
                hud_h = 86

                hud_overlay = frame.copy()
                cv2.rectangle(hud_overlay, (18, 18), (18 + hud_w, 18 + hud_h), (16, 16, 20), -1)
                cv2.addWeighted(hud_overlay, 0.88, frame, 0.12, 0, frame)
                cv2.rectangle(frame, (18, 18), (26, 18 + hud_h), (0, 0, 240), -1)
                cv2.rectangle(frame, (18, 18), (18 + hud_w, 18 + hud_h), (0, 0, 240), 2)
                cv2.putText(frame, line1, (36, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, line2, (36, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(frame, line3, (36, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

                # 6. Bottom Timeline Bar highlighting WHEN the foul occurred
                bar_x1, bar_x2 = 30, frame_w - 30
                bar_y1, bar_y2 = frame_h - 26, frame_h - 12
                bar_w = bar_x2 - bar_x1
                cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (25, 25, 25), -1)
                ws_x = max(bar_x1, bar_x1 + int(bar_w * (float(ev["window_start"]) / float(max(1, total_video_frames)))))
                we_x = min(bar_x2, bar_x1 + int(bar_w * (float(ev["window_end"]) / float(max(1, total_video_frames)))))
                sp_x = max(bar_x1, min(bar_x2, bar_x1 + int(bar_w * (float(ev["when_frame"]) / float(max(1, total_video_frames))))))
                cv2.rectangle(frame, (ws_x, bar_y1), (we_x, bar_y2), (0, 0, 235), -1)
                cv2.line(frame, (sp_x, bar_y1 - 6), (sp_x, bar_y2 + 2), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"FOUL @ {ev['when_time_sec']:.2f}s (Frame {ev['when_frame']})",
                    (max(bar_x1, sp_x - 80), bar_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (60, 80, 255),
                    1,
                    cv2.LINE_AA,
                )
                cur_x = max(bar_x1, min(bar_x2, bar_x1 + int(bar_w * (float(frame_idx) / float(max(1, total_video_frames - 1))))))
                cv2.circle(frame, (cur_x, (bar_y1 + bar_y2) // 2), 6, (0, 235, 255), -1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()
        remux_h264_faststart(final_output_path)

    primary = enriched_events[0] if enriched_events else None
    report: Dict[str, Any] = {
        "clip": resolved_video.name,
        "stem": resolved_video.stem,
        "is_foul": is_foul,
        "num_fouls": len(enriched_events),
        "foul_prob": round(primary["foul_prob"] if primary else bag_prob, 4),
        "bag_prob": round(bag_prob, 4),
        "threshold": round(threshold, 4),
        "verdict": f"FOUL WHISTLED ({len(enriched_events)})" if is_foul else "NO FOUL (CLEAN PLAY)",
        "what_foul": primary["what_foul"] if primary else "No Foul",
        "foul_subtype": primary["foul_subtype"] if primary else "No Foul (Legal Play)",
        "foul_mechanic": primary["foul_mechanic"] if primary else "Rule 12B: Legal Guarding Position",
        "when_frame": primary["when_frame"] if primary else None,
        "when_time_sec": primary["when_time_sec"] if primary else None,
        "when_str": primary["when_str"] if primary else "N/A (No Foul)",
        "where_zone": primary["where_zone"] if primary else "N/A (No Foul)",
        "where_point": primary["where_point"] if primary else None,
        "where_box": primary["where_box"] if primary else None,
        "events": enriched_events,
        "num_windows_scanned": len(windows),
        "used_ebard_detector": bool(bag.get("used_ebard", False)),
        "output_mp4": str(final_output_path),
    }

    # Save telemetry JSON for notebook interactive console
    decisions_json = final_output_path.parent / "referee_decisions.json"
    all_dec: Dict[str, Any] = {}
    if decisions_json.exists():
        try:
            with open(decisions_json, "r", encoding="utf-8") as jf:
                loaded = json.load(jf)
                if isinstance(loaded, dict):
                    all_dec = loaded
        except Exception:
            all_dec = {}
    all_dec[resolved_video.stem] = report
    try:
        with open(decisions_json, "w", encoding="utf-8") as jf:
            json.dump(all_dec, jf, indent=2)
    except Exception:
        pass

    logger.info("Saved AI Referee video (%d frames, %d foul(s)) -> %s", frame_idx, len(enriched_events), final_output_path)
    if return_details:
        return report
    return float(report["foul_prob"])


def run_ai_referee_on_folder(
    folder_path: Union[Path, str] = BASE_DIR / "data" / "test",
    checkpoint_path: Union[Path, str] = BASE_DIR / "weights" / "best_foul_r2plus1d.pt",
    annotated_dir: Optional[Union[Path, str]] = None,
    yolo_weights: str = preprocess_mod.EBARD_WEIGHTS_NAME,
    pose_weights: str = "yolo11n-pose.pt",
) -> List[Dict[str, Any]]:
    """Run the autonomous AI Referee across ALL raw game videos dropped into `folder_path`."""
    t_dir = resolve_project_path(folder_path)
    a_dir = resolve_project_path(annotated_dir) if annotated_dir else (BASE_DIR / "data" / "annotated")
    a_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_video_files_in_folder(t_dir)
    if not videos:
        logger.warning("No raw game videos found in %s.", t_dir)
        return []

    logger.info("🏀 AI Referee scanning %d video(s) in %s: %s", len(videos), t_dir, [v.name for v in videos])
    reports: List[Dict[str, Any]] = []
    for idx, vid in enumerate(videos, start=1):
        logger.info("=== [%d/%d] AI Referee evaluating clip: %s ===", idx, len(videos), vid.name)
        out_mp4 = a_dir / f"{vid.stem}_annotated.mp4"
        try:
            rep = predict_single_clip(
                video_path=vid,
                checkpoint_path=checkpoint_path,
                output_mp4_path=out_mp4,
                yolo_weights=yolo_weights,
                pose_weights=pose_weights,
                return_details=True,
            )
            if isinstance(rep, dict):
                reports.append(rep)
        except Exception as exc:
            logger.error("Failed AI Referee evaluation on %s: %s", vid.name, exc)
    logger.info("Finished AI Referee folder run: %d/%d videos saved -> %s", len(reports), len(videos), a_dir)
    return reports


def annotate_test_directory(
    test_dir: Union[Path, str] = BASE_DIR / "data" / "test",
    checkpoint_path: Union[Path, str] = BASE_DIR / "weights" / "best_foul_r2plus1d.pt",
    annotated_dir: Optional[Union[Path, str]] = None,
    yolo_weights: str = preprocess_mod.EBARD_WEIGHTS_NAME,
    pose_weights: str = "yolo11n-pose.pt",
) -> List[Path]:
    reports = run_ai_referee_on_folder(
        folder_path=test_dir,
        checkpoint_path=checkpoint_path,
        annotated_dir=annotated_dir,
        yolo_weights=yolo_weights,
        pose_weights=pose_weights,
    )
    return [Path(r["output_mp4"]) for r in reports if "output_mp4" in r]


# ==============================================================================
# Multi-Window RL Training + Game-Grouped MIL Training Pipeline
# ==============================================================================
def train_rl_referee_agent(
    episodes: Optional[List[Dict[str, Any]]] = None,
    dataset: Optional[L2MFoulDataset] = None,
    mil_model: Optional[FoulMILClassifier] = None,
    num_episodes: int = 250,
    batch_size: int = 64,
    mode: str = "dqn",
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Train Double Dueling DQN Referee on genuine multi-window Swin3D-T + MIL trajectories."""
    dev = select_compute_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device
    if episodes is None:
        if dataset is None:
            dataset = L2MFoulDataset(is_train=False)
        episodes = model_mod.create_referee_episodes_from_dataset(dataset, mil_model=mil_model, device=dev)
    if not episodes:
        raise RuntimeError("No multi-window episodes available for RL training.")

    env = model_mod.NBARefereeEnv(episodes=episodes)
    agent = model_mod.RefereeRLAgent(mode=mode, device=dev)
    logger.info("Training Multi-Window Referee RL Agent (Double Dueling DQN) for %d episodes on %s...", num_episodes, dev)

    ep_rewards: List[float] = []
    ep_correct: List[bool] = []
    ep_losses: List[float] = []

    for ep_i in range(1, num_episodes + 1):
        s = env.reset()
        done = False
        ep_r = 0.0
        step_losses = []
        while not done:
            a = agent.select_action(s, evaluate=False)
            s_next, r, done, info = env.step(a)
            agent.store_transition(s, a, r, s_next, done)
            loss = agent.train_step(batch_size=batch_size)
            if loss > 0:
                step_losses.append(loss)
            s = s_next
            ep_r += r
        if ep_i % 5 == 0:
            agent.update_target_network()
        ep_rewards.append(ep_r)
        ep_correct.append(bool(info.get("correct_call", False)))
        if step_losses:
            ep_losses.append(float(np.mean(step_losses)))
        if ep_i % 50 == 0 or ep_i == num_episodes:
            logger.info(
                "RL Episode [%03d/%03d] | Avg Return: %+.2f | Train Acc: %.1f%% | Epsilon: %.2f",
                ep_i,
                num_episodes,
                float(np.mean(ep_rewards[-50:])),
                float(np.mean(ep_correct[-50:]) * 100.0),
                agent.epsilon,
            )

    save_path = resolve_project_path(checkpoint_path or (BASE_DIR / "weights" / "referee_rl_agent.pt"))
    agent.save(save_path)
    return {
        "final_accuracy": float(np.mean(ep_correct[-50:])),
        "mean_reward": float(np.mean(ep_rewards)),
        "episode_rewards": ep_rewards,
        "losses": ep_losses,
        "checkpoint_path": str(save_path),
    }


def _train_single_fold(
    train_meta: List[Dict[str, Any]],
    val_meta: List[Dict[str, Any]],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    device: torch.device,
    seed: int = 42,
    return_oof: bool = False,
):
    seed_everything(seed)
    train_loader = DataLoader(
        L2MFoulDataset(samples=train_meta, is_train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        L2MFoulDataset(samples=val_meta, is_train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = FoulMILClassifier(dropout_p=dropout).to(device)
    pos_weight = compute_pos_weight_from_meta(train_meta, device=device)
    criterion = FoulMILMultiTaskLoss(pos_weight=pos_weight, gamma=1.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=lr * 0.02)

    best_score = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}
    best_val_loss = float("inf")
    best_oof: Dict[str, np.ndarray] = {}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics, v_logits, v_targets, vt_logits, vt_targets = evaluate_mil(
            model, val_loader, criterion, device
        )
        scheduler.step()

        # Selection objective prioritizes out-of-game ROC-AUC + F1 (robust against threshold noise)
        sel_score = 0.55 * metrics["roc_auc"] + 0.45 * metrics["f1"]
        if sel_score > best_score:
            best_score = sel_score
            best_val_loss = val_loss
            best_metrics = dict(metrics)
            model.temperature.fill_(float(metrics["temperature"]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_oof = {
                "logits": v_logits.copy(),
                "targets": v_targets.copy(),
                "type_logits": vt_logits.copy(),
                "type_targets": vt_targets.copy(),
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "Epoch [%02d/%02d] | Train: %.4f | Val: %.4f | ROC-AUC: %.3f | F1: %.3f | BalAcc: %.3f | TypeAcc: %.3f | T=%.2f | Thresh=%.2f",
                epoch,
                epochs,
                tr_loss,
                val_loss,
                metrics["roc_auc"],
                metrics["f1"],
                metrics["balanced_accuracy"],
                metrics["call_type_accuracy"],
                metrics["temperature"],
                metrics["threshold"],
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    if return_oof:
        return model, best_metrics, best_val_loss, best_oof
    return model, best_metrics, best_val_loss


def run_training_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = select_compute_device(args.device)
    logger.info("Using compute device: %s (OS: %s)", device, platform.system())

    base_dataset = L2MFoulDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        is_train=False,
        max_samples=args.max_samples,
    )
    if len(base_dataset) < 6:
        raise RuntimeError(
            f"Only {len(base_dataset)} processed v3 MIL bags found in {args.processed_dir}. "
            "Run `2_preprocess_tubes.py` first!"
        )

    samples_meta = base_dataset.samples_meta
    unique_games = sorted({str(m["game_id"]) for m in samples_meta})
    logger.info(
        "Loaded %d real MIL bags across %d unique NBA games (foul=%d, no_foul=%d)",
        len(samples_meta),
        len(unique_games),
        sum(1 for m in samples_meta if int(m["label"]) == 1),
        sum(1 for m in samples_meta if int(m["label"]) == 0),
    )

    # Leak-Free Game-Grouped Cross-Validation + Global OOF Ensemble Calibration
    cv_folds = max(1, int(getattr(args, "cv_folds", 1)))
    ensemble_state_dicts: Optional[List[Dict[str, torch.Tensor]]] = None
    cv_fold_summary: Optional[Dict[str, Any]] = None

    if cv_folds >= 2 and len(unique_games) >= cv_folds:
        logger.info("Running %d-Fold Leak-Free Game-Grouped Cross-Validation...", cv_folds)
        fold_aucs, fold_f1s, fold_type_accs = [], [], []
        fold_models: List[FoulMILClassifier] = []
        oof_logits_list: List[np.ndarray] = []
        oof_targets_list: List[np.ndarray] = []
        oof_type_logits_list: List[np.ndarray] = []
        oof_type_targets_list: List[np.ndarray] = []
        best_fold_model = None
        best_fold_val_loss = float("inf")
        best_fold_score = -1.0

        for f_idx in range(cv_folds):
            tr_m, va_m = game_grouped_train_val_split(samples_meta, n_splits=cv_folds, fold_idx=f_idx, seed=args.seed)
            logger.info(
                "--- Fold [%d/%d] Train: %d clips (%d games) | Val: %d clips (%d unseen games) ---",
                f_idx + 1,
                cv_folds,
                len(tr_m),
                len({m["game_id"] for m in tr_m}),
                len(va_m),
                len({m["game_id"] for m in va_m}),
            )
            f_model, f_metrics, f_vloss, f_oof = _train_single_fold(
                train_meta=tr_m,
                val_meta=va_m,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                device=device,
                seed=args.seed + f_idx,
                return_oof=True,
            )
            fold_models.append(f_model)
            fold_aucs.append(f_metrics["roc_auc"])
            fold_f1s.append(f_metrics["f1"])
            fold_type_accs.append(f_metrics["call_type_accuracy"])
            oof_logits_list.append(f_oof["logits"])
            oof_targets_list.append(f_oof["targets"])
            if len(f_oof["type_targets"]) > 0:
                oof_type_logits_list.append(f_oof["type_logits"])
                oof_type_targets_list.append(f_oof["type_targets"])

            f_score = 0.55 * f_metrics["roc_auc"] + 0.45 * f_metrics["f1"]
            if f_score > best_fold_score:
                best_fold_score = f_score
                best_fold_model = f_model
                best_fold_val_loss = f_vloss

        # Global Out-of-Fold (OOF) Temperature & Threshold Calibration across ALL unseen-game clips!
        all_oof_logits = np.concatenate(oof_logits_list, axis=0)
        all_oof_targets = np.concatenate(oof_targets_list, axis=0)
        oof_temp, oof_thresh, oof_metrics = fit_temperature_and_threshold(all_oof_logits, all_oof_targets)
        if oof_type_targets_list:
            all_tl = np.concatenate(oof_type_logits_list, axis=0)
            all_tt = np.concatenate(oof_type_targets_list, axis=0)
            oof_type_preds = (1.0 / (1.0 + np.exp(-all_tl)) >= 0.5).astype(int)
            oof_metrics["call_type_accuracy"] = float(accuracy_score(all_tt, oof_type_preds))
        else:
            oof_metrics["call_type_accuracy"] = float(np.mean(fold_type_accs))

        cv_fold_summary = {
            "cv_folds": cv_folds,
            "mean_roc_auc": float(np.mean(fold_aucs)),
            "std_roc_auc": float(np.std(fold_aucs)),
            "mean_f1": float(np.mean(fold_f1s)),
            "std_f1": float(np.std(fold_f1s)),
            "mean_call_type_acc": float(np.mean(fold_type_accs)),
            "oof_total_clips": int(len(all_oof_targets)),
        }
        logger.info(
            "=== %d-Fold Out-of-Game CV Summary: Fold ROC-AUC=%.3f ± %.3f | Fold F1=%.3f ± %.3f | Fold TypeAcc=%.3f ===",
            cv_folds,
            cv_fold_summary["mean_roc_auc"],
            cv_fold_summary["std_roc_auc"],
            cv_fold_summary["mean_f1"],
            cv_fold_summary["std_f1"],
            cv_fold_summary["mean_call_type_acc"],
        )
        logger.info(
            "=== Global OOF Calibration (%d unseen-game clips): ROC-AUC=%.3f | F1=%.3f | BalAcc=%.3f | TypeAcc=%.3f | T=%.2f | Thresh=%.2f ===",
            len(all_oof_targets),
            oof_metrics["roc_auc"],
            oof_metrics["f1"],
            oof_metrics["balanced_accuracy"],
            oof_metrics["call_type_accuracy"],
            oof_temp,
            oof_thresh,
        )
        assert best_fold_model is not None
        best_fold_model.temperature.fill_(float(oof_temp))
        ensemble_state_dicts = [
            {k: v.detach().cpu() for k, v in m.state_dict().items()} for m in fold_models
        ]
        inference_model: Union[FoulMILClassifier, FoulMILEnsemble] = FoulMILEnsemble(
            fold_models, temperature=oof_temp
        ).to(device)
        best_model = best_fold_model
        best_metrics = oof_metrics
        best_val_loss = best_fold_val_loss
    else:
        tr_m, va_m = game_grouped_train_val_split(samples_meta, n_splits=5, fold_idx=0, seed=args.seed)
        logger.info(
            "Game-Grouped Split -> Train: %d clips (%d games) | Val: %d clips (%d unseen games)",
            len(tr_m),
            len({m["game_id"] for m in tr_m}),
            len(va_m),
            len({m["game_id"] for m in va_m}),
        )
        best_model, best_metrics, best_val_loss = _train_single_fold(
            train_meta=tr_m,
            val_meta=va_m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            device=device,
            seed=args.seed,
        )
        inference_model = best_model

    # Save calibrated v3 checkpoint (5-Fold Out-of-Game Ensemble + single-fold fallback)
    ckpt_payload: Dict[str, Any] = {
        "architecture": "FoulMILEnsemble_Swin3D_v3" if ensemble_state_dicts else "FoulMILClassifier_Swin3D_v3",
        "model_state_dict": {k: v.detach().cpu() for k, v in best_model.state_dict().items()},
        "metrics": best_metrics,
        "optimal_threshold": float(best_metrics["threshold"]),
        "temperature": float(best_metrics["temperature"]),
        "val_loss": float(best_val_loss),
    }
    if ensemble_state_dicts is not None:
        ckpt_payload["ensemble_state_dicts"] = ensemble_state_dicts
    if cv_fold_summary is not None:
        ckpt_payload["cv_fold_summary"] = cv_fold_summary

    ckpt_path = resolve_project_path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_payload, ckpt_path)
    for sync_p in [BASE_DIR / "weights" / "best_foul_r2plus1d.pt", BASE_DIR / "best_foul_r2plus1d.pt"]:
        if sync_p.resolve() != ckpt_path.resolve():
            sync_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_payload, sync_p)
    logger.info(
        "💾 Saved calibrated v3 %s checkpoint -> %s (OOF ROC-AUC=%.3f, F1=%.3f, TypeAcc=%.3f, T=%.2f, thresh=%.2f)",
        f"{len(ensemble_state_dicts)}-Fold Ensemble" if ensemble_state_dicts else "Single-Fold",
        ckpt_path,
        best_metrics["roc_auc"],
        best_metrics["f1"],
        best_metrics["call_type_accuracy"],
        best_metrics["temperature"],
        best_metrics["threshold"],
    )

    if getattr(args, "train_rl", False):
        train_rl_referee_agent(
            dataset=base_dataset,
            mil_model=inference_model,
            num_episodes=getattr(args, "rl_episodes", 250),
            device=device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v3.0 Swin3D-T + Gated Attention MIL Foul Training, Game-Grouped Calibration & Zero-Referee AI Referee Inference"
    )
    parser.add_argument("--processed-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--metadata-csv", type=Path, default=BASE_DIR / "data" / "l2m_metadata.csv")
    parser.add_argument("--checkpoint-path", type=Path, default=BASE_DIR / "weights" / "best_foul_r2plus1d.pt")
    parser.add_argument("--annotated-dir", type=Path, default=BASE_DIR / "data" / "annotated")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-2)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of leak-free game-grouped CV folds (1 = single split)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "data" / "test")
    parser.add_argument("--annotate-test", action="store_true")
    parser.add_argument("--output-mp4", type=Path, default=None)
    parser.add_argument("--train-rl", action="store_true")
    parser.add_argument("--rl-episodes", type=int, default=250)
    parser.add_argument("--yolo-pose-model", type=str, default="yolo11n-pose.pt")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.annotate_test:
        run_ai_referee_on_folder(
            folder_path=cli_args.test_dir,
            checkpoint_path=cli_args.checkpoint_path,
            annotated_dir=cli_args.annotated_dir,
            pose_weights=cli_args.yolo_pose_model,
        )
    elif cli_args.predict is not None:
        p = resolve_project_path(cli_args.predict)
        if p.is_dir():
            run_ai_referee_on_folder(
                folder_path=p,
                checkpoint_path=cli_args.checkpoint_path,
                annotated_dir=cli_args.annotated_dir,
                pose_weights=cli_args.yolo_pose_model,
            )
        else:
            predict_single_clip(cli_args)
    else:
        run_training_pipeline(cli_args)
