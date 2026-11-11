#!/usr/bin/env python3
"""
STEP 4: Cross-Platform Foul Training, Threshold Calibration & Single-Clip Inference (`4_train_eval.py`)

Engineered to run on ALL machines (Windows, Linux, Intel Mac, Apple Silicon M1/M2/M3/M4, CUDA GPUs, or CPU laptops):
  - Auto-selects compute device (`cuda`, `mps`, or `cpu`) and configures safe DataLoader
    worker settings (`num_workers=0` on Windows and macOS to prevent multiprocessing spawn hangs).
  - Lightweight Default Batch Sizes:
    * `--batch-size 2` with `--accum-steps 2` (effective batch size 4, cutting peak RAM/VRAM in half).
    * `--epochs 5` and `--max-samples 20` by default so training completes in minutes on any machine.
  - Uses `FoulFocalBCEWithLogitsLoss` (`gamma=1.5`) and calibrates `optimal_threshold` on validation
    data each epoch to maximize Foul F1-Score (`best_foul_r2plus1d.pt`).
  - CLI inference (`--predict path/to/clip.mp4`) includes cross-OS video codec fallback (`mp4v`, `avc1`, `XVID`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
import contextlib
import importlib
import logging
import platform
import random
from typing import Dict, List, Optional, Tuple

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
    """Select the best available compute device ('cuda', 'mps', or 'cpu') across all OSes."""
    if requested and requested.lower() != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_safe_num_workers(requested_workers: int) -> int:
    """
    Return a crash-free DataLoader `num_workers` count across operating systems.
    On Windows and macOS (`Darwin`), Python's `spawn` multiprocessing start method
    duplicates PyTorch memory and can hang with 3D video tensors, so `0` is safest.
    """
    if platform.system() in ("Darwin", "Windows"):
        return 0
    return max(0, requested_workers)


def get_autocast_context(device: torch.device):
    """Return a cross-platform FP16 autocast context on CUDA and a no-op context on MPS/CPU."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


class FoulFocalBCEWithLogitsLoss(nn.Module):
    """
    Class-balanced Binary Cross-Entropy with Focal modulation (`gamma=1.5`)
    specifically designed for NBA L2M foul prediction where `no_foul` (CNC/IC)
    outnumbers `foul` (CC/INC) by ~3.5x and subtle arm/wrist fouls are hard examples.
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 1.5) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
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
    logger.info(
        "Class distribution -> foul (1): %d, no_foul (0): %d | pos_weight=%.3f",
        num_pos,
        num_neg,
        weight,
    )
    return torch.tensor([weight], dtype=torch.float32, device=device)


def calibrate_foul_threshold(
    y_true: List[int],
    y_prob: List[float],
) -> Tuple[float, Dict[str, float]]:
    """
    Search candidate probability thresholds `[0.25 .. 0.75]` on the validation set
    to find the threshold that maximizes Foul F1-Score.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if len(np.unique(y_true_arr)) > 1 else 0.5

    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    for thresh in np.linspace(0.25, 0.75, 51):
        y_pred_arr = (y_prob_arr >= thresh).astype(int)
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_metrics = {
                "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
                "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
                "f1": f1,
                "roc_auc": roc_auc,
                "threshold": best_thresh,
            }

    return best_thresh, best_metrics


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
        videos, foul_feats, targets = batch
        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        foul_feats = foul_feats.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)

        with get_autocast_context(device):
            logits = model(videos, foul_feats)
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
        videos, foul_feats, targets = batch
        videos = videos.to(device, non_blocking=(device.type == "cuda"))
        foul_feats = foul_feats.to(device, non_blocking=(device.type == "cuda"))
        targets_dev = targets.to(device, non_blocking=(device.type == "cuda")).unsqueeze(1)

        with get_autocast_context(device):
            logits = model(videos, foul_feats)
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
    stratify_labels = labels if len(set(labels)) > 1 and min(labels.count(0), labels.count(1)) >= 2 else None

    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=stratify_labels,
    )

    train_dataset = L2MFoulDataset(
        samples=train_samples,
        is_train=True,
        return_foul_features=True,
    )
    val_dataset = L2MFoulDataset(
        samples=val_samples,
        is_train=False,
        return_foul_features=True,
    )

    safe_workers = get_safe_num_workers(args.num_workers)
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=use_pin_memory,
    )

    model = FoulClassifier3D(
        pretrained=not args.no_pretrained,
        dropout_p=args.dropout,
        freeze_early=True,
    ).to(device)
    param_info = model.trainable_parameter_summary()
    logger.info(
        "Initialized R(2+1)D-18 + Biomechanical Foul-Pose Classifier | Trainable: %s / %s params | "
        "batch_size=%d, accum_steps=%d (effective batch=%d)",
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


def create_cross_platform_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter with cross-OS codec fallback (`mp4v` -> `avc1` -> `XVID`)."""
    for codec in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not initialize VideoWriter for {output_path} on {platform.system()}")


@torch.no_grad()
def predict_single_clip(
    video_path: Path,
    checkpoint_path: Path,
    output_mp4_path: Optional[Path] = None,
    yolo_weights: str = "yolo11n.pt",
    pose_weights: Optional[str] = "yolo11n-pose.pt",
) -> float:
    video_path = resolve_project_path(video_path)
    checkpoint_path = resolve_project_path(checkpoint_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Input clip not found: {video_path}")
    if YOLO is None:
        raise ImportError("Ultralytics is required for raw MP4 clip inference (`pip install ultralytics`).")

    device = select_compute_device()
    pose_candidate = resolve_project_path(pose_weights) if pose_weights else None
    active_pose = (
        str(pose_candidate)
        if (pose_candidate and pose_candidate.exists())
        else (pose_weights or yolo_weights)
    )
    pose_model = YOLO(active_pose)

    tube_tensor, crop_boxes, sampled_indices, fps, foul_features = extract_action_tube(
        video_path=video_path,
        det_model=None,
        pose_model=pose_model,
        device=str(device),
    )

    model = FoulClassifier3D(pretrained=False, freeze_early=False).to(device)
    threshold = 0.50
    if checkpoint_path.exists():
        ckpt = load_torch_file_safe(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        if isinstance(ckpt, dict) and "optimal_threshold" in ckpt:
            threshold = float(ckpt["optimal_threshold"])
        logger.info("Loaded trained checkpoint: %s (threshold=%.2f)", checkpoint_path, threshold)
    else:
        logger.warning(
            "Checkpoint %s not found; running inference with uncalibrated weights.",
            checkpoint_path,
        )

    model.eval()
    augmentor = ConsistentVideoAugmentor(is_train=False)
    input_batch = augmentor(tube_tensor).unsqueeze(0).to(device)
    feats_batch = foul_features.unsqueeze(0).to(device)

    with get_autocast_context(device):
        logit = model(input_batch, feats_batch)
        foul_prob = float(torch.sigmoid(logit).item())

    peak_contact_intensity = float(foul_features[:, 11].max().item())
    verdict = "FOUL DETECTED" if foul_prob >= threshold else "LEGAL CONTACT / NO FOUL"
    logger.info(
        "Prediction for %s -> %s (P(foul) = %.2f%%, threshold = %.0f%%, peak_contact = %.2f)",
        video_path.name,
        verdict,
        foul_prob * 100.0,
        threshold * 100.0,
        peak_contact_intensity,
    )

    if output_mp4_path is None:
        output_mp4_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")
    else:
        output_mp4_path = resolve_project_path(output_mp4_path)

    sample_box_map = dict(zip(sampled_indices, crop_boxes))
    min_sampled, max_sampled = min(sampled_indices), max(sampled_indices)

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = create_cross_platform_video_writer(output_mp4_path, fps, width, height)

    active_box = crop_boxes[0]
    frame_idx = 0
    banner_color = (0, 0, 220) if foul_prob >= threshold else (0, 180, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in sample_box_map:
            active_box = sample_box_map[frame_idx]

        if min_sampled <= frame_idx <= max_sampled:
            x1, y1, x2, y2 = active_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), banner_color, 3)
            cv2.putText(
                frame,
                "Foul Contact Zone (224x224)",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                banner_color,
                2,
                cv2.LINE_AA,
            )

        overlay = frame.copy()
        cv2.rectangle(overlay, (16, 16), (560, 122), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.putText(
            frame,
            f"Verdict: {verdict}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            banner_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Foul Probability: {foul_prob * 100:.1f}% (Thresh: {threshold * 100:.0f}%)",
            (30, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Peak Biomechanical Contact Score: {peak_contact_intensity * 100:.1f}%",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (180, 220, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Saved annotated prediction video -> %s", output_mp4_path)
    return foul_prob


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
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5; increase to 20-30 for full convergence)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro-batch size per step (default: 2 for low RAM/VRAM; increase to 4-8 on GPUs)",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2 -> effective batch size 4; increase to 4-8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max processed samples to load per run (default: 20; set 0 to train on all processed tensors)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW (default: 1e-4)",
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
        default=0.4,
        help="Dropout probability before binary classification head (default: 0.4)",
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
        help="DataLoader worker processes (default: 0 for cross-OS stability on Windows/macOS)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Path to a single MP4 video clip to run end-to-end YOLO + R(2+1)D-18 inference",
    )
    parser.add_argument(
        "--output-mp4",
        type=Path,
        default=None,
        help="Optional output path for the annotated prediction MP4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.predict is not None:
        predict_single_clip(
            video_path=cli_args.predict,
            checkpoint_path=cli_args.checkpoint_path,
            output_mp4_path=cli_args.output_mp4,
        )
    else:
        run_training_pipeline(cli_args)
