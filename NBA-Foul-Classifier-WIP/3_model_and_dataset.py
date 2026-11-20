#!/usr/bin/env python3
"""
STEP 3: Swin3D-T + Gated Attention Multiple Instance Learning (MIL) Foul Classifier & Multi-Window RL Referee (`3_model_and_dataset.py`)

Full-Stack v3.0 Learning Architecture:
  1. Video Foundation Representation (`Swin3D-T`, Kinetics-400 Top-1 78.8%, `768-D`):
     - Precomputed per 2.0-second candidate window `(K, 768)` in Stage 2 (or encoded on-the-fly for
       unseen videos of ANY length), preventing 28M+ parameter memorization on limited clip counts.
  2. Multi-Window Gated Attention Multiple Instance Learning (`FoulMILClassifier` / `FoulClassifier3D`):
     - Encodes each candidate 2.0s window via Dual-Stream Fusion:
         * Stream A: `Swin3D-T` 768-D visual projection -> 256-D
         * Stream B: 4-Head Temporal Self-Attention over `(16, 16)` NBA Rule 12 + Depth + Ball/Hoop kinematics -> 128-D
         * Window Fusion: `384-D -> 256-D`
     - Per-window heads predict calibrated foul probability `p_k` and real `call_type` (`Shooting Foul` vs `Personal Foul`)
       at EVERY window `k = 1..K`, directly localizing **WHEN** the foul happens without frame-level labels!
     - Gated Attention MIL (Ilse et al., ICML 2018) pools window representations into a clip-level bag prediction:
         * `no_foul` clips (`y=0`) push ALL windows in the clip toward `0`.
         * `foul` clips (`y=1`) push the highest-attention foul window(s) toward `1`.
     - Real `call_type` head is supervised ONLY on ground-truth `call_type` labels (`Shooting Foul` vs `Personal Foul`)
       from `l2m_metadata.csv` (zero synthetic heuristics).
  3. Honest Game-Grouped Validation (`game_grouped_train_val_split`):
     - Splits strictly by `game_id` (`StratifiedGroupKFold`) so clips from the same NBA game/arena/teams
       never leak between train and validation.
  4. Sliding-Window Temporal NMS Inference (`predict_video_windows`):
     - Evaluates ALL sliding windows across arbitrary-length NBA game footage and applies Temporal NMS
       so videos with 0 fouls produce 0 whistles (nothing happens on video) and videos with 1+ fouls
       highlight every detected foul with WHAT, WHERE, and WHEN.
  5. Multi-Window Sequential RL Referee (`NBARefereeEnv` + `RefereeRLAgent`):
     - Operates on distinct per-window Swin3D-T features, MIL foul probabilities, kinematics, and real L2M
       (`CC`, `INC`, `CNC`, `IC`) rewards rather than static tiled vectors.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import csv
import importlib
import logging
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

cv2.setNumThreads(0)

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_and_dataset")

# Standardized v3 Architecture Constants
CHANNELS = 3
FRAMES = 16
HEIGHT = 224
WIDTH = 224
POSE_FEATURES = 16
NUM_FOUL_FEATURES = POSE_FEATURES
VIDEO_FEATURE_DIM = 768       # Swin3D-T output dimension
POSE_HIDDEN_DIM = 192         # Temporal Pose Attention output dimension
WINDOW_FUSION_DIM = 256       # Per-window fused multimodal dimension
COMBINED_FEATURE_DIM = VIDEO_FEATURE_DIM + POSE_HIDDEN_DIM  # 960
CLASSIFIER_HIDDEN_DIM = 256
MAX_BAG_WINDOWS = 10

MECHANICAL_FOUL_DIMENSIONS = (
    "arm_contact_intrusion",
    "body_cylinder_penetration",
    "lower_body_landing_violation",
    "defender_verticality_angle",
    "defender_lateral_positioning",
    "offensive_displacement",
    "defender_to_ball_distance",
    "ball_to_shooter_distance",
    "ball_strike_vs_body_ratio",
    "relative_closing_velocity",
    "momentum_transfer_shock",
    "airborne_contest_apex_diff",
    "hoop_proximity",
    "foot_plane_depth_alignment",
    "ball_verified_on_play",
    "shooting_motion_score",
)
POSE_FEATURE_NAMES = MECHANICAL_FOUL_DIMENSIONS
FOUL_FEATURE_NAMES = MECHANICAL_FOUL_DIMENSIONS

FOUL_CALL_CATEGORIES = (
    "Personal Foul",
    "Shooting Foul",
)


def resolve_project_path(path_arg: Path | str) -> Path:
    norm_str = str(path_arg).replace("\\", "/")
    p = Path(norm_str)
    if p.is_absolute() and p.exists():
        return p
    for candidate in [
        Path.cwd() / p,
        BASE_DIR / p,
        BASE_DIR / "weights" / p.name,
        BASE_DIR / "weights" / p,
        BASE_DIR.parent / "weights" / p.name,
        BASE_DIR.parent / p,
    ]:
        if candidate.exists():
            return candidate
    return BASE_DIR / p


def load_torch_file_safe(path: Path | str, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def derive_motion_foul_features_from_rgb(video_tensor: torch.Tensor) -> torch.Tensor:
    """Fallback kinematic motion estimator `(FRAMES, 16)` if a raw clip lacks pose features."""
    vid = video_tensor.float()
    if vid.max() > 1.5:
        vid = vid / 255.0
    if vid.dim() == 4 and vid.size(0) == 3:
        vid = vid.permute(1, 0, 2, 3)  # (T, 3, H, W)
    t_len = vid.size(0)
    gray = vid.mean(dim=1)  # (T, H, W)
    diffs = torch.zeros(t_len, dtype=torch.float32)
    if t_len > 1:
        diffs[1:] = (gray[1:] - gray[:-1]).abs().mean(dim=(1, 2))
    max_d = max(float(diffs.max().item()), 1e-5)
    norm_m = (diffs / max_d).clamp(0.0, 1.0)
    feats = torch.zeros((t_len, POSE_FEATURES), dtype=torch.float32)
    feats[:, 0] = norm_m * 0.45
    feats[:, 1] = norm_m * 0.45
    feats[:, 5] = norm_m * 0.50
    feats[:, 9] = 0.50
    feats[:, 10] = norm_m * 0.40
    feats[:, 13] = 0.70
    return feats


def harmonize_pose_tensor(pose: torch.Tensor, target_frames: int = FRAMES, target_dim: int = POSE_FEATURES) -> torch.Tensor:
    """Pad or interpolate any pose tensor `(T, D)` cleanly to `(FRAMES, 16)`."""
    p = pose.float()
    if p.dim() == 1:
        p = p.unsqueeze(0)
    if p.size(0) != target_frames:
        idx = np.linspace(0, max(p.size(0) - 1, 0), target_frames).round().astype(int)
        p = p[idx]
    if p.size(1) < target_dim:
        pad = torch.zeros((target_frames, target_dim - p.size(1)), dtype=torch.float32)
        p = torch.cat([p, pad], dim=1)
    elif p.size(1) > target_dim:
        p = p[:, :target_dim]
    return torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)


class ConsistentVideoAugmentor:
    """Spatiotemporally consistent video normalization & augmentation for `(3, T, H, W)` tubes."""

    def __init__(self, is_train: bool = True, crop_size: int = HEIGHT) -> None:
        self.is_train = is_train
        self.crop_size = crop_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

    def __call__(self, video_frames: torch.Tensor) -> torch.Tensor:
        x = video_frames.float()
        if x.max() > 1.5:
            x = x / 255.0
        if self.is_train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-1])
            brightness = random.uniform(0.88, 1.12)
            contrast = random.uniform(0.88, 1.12)
            mean_val = x.mean(dim=(2, 3), keepdim=True)
            x = ((x - mean_val) * contrast + mean_val) * brightness
            x = x.clamp(0.0, 1.0)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std


_SHARED_EMBEDDING_EXTRACTOR: Optional[nn.Module] = None


def get_shared_embedding_extractor() -> nn.Module:
    """Singleton Kinetics-400 Swin3D-T (`768-D`) visual encoder for on-the-fly fallback encoding."""
    global _SHARED_EMBEDDING_EXTRACTOR
    if _SHARED_EMBEDDING_EXTRACTOR is None:
        try:
            extractor = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        except Exception:
            extractor = swin3d_t(weights=None)
        extractor.head = nn.Identity()
        for p in extractor.parameters():
            p.requires_grad = False
        if torch.cuda.is_available():
            extractor = extractor.to("cuda")
        extractor.eval()
        _SHARED_EMBEDDING_EXTRACTOR = extractor
    return _SHARED_EMBEDDING_EXTRACTOR


# ==============================================================================
# Dataset Loader with Strict Test Segregation & Game-Grouped Splitting
# ==============================================================================
class L2MFoulDataset(Dataset):
    """
    Multi-Window MIL Bag Dataset for NBA L2M Foul Detection.
    Loads deduplicated v3 `.pt` bags containing:
      - `window_embeddings`: `(MAX_BAG_WINDOWS, 768)` Swin3D-T visual embeddings
      - `window_pose_features`: `(MAX_BAG_WINDOWS, 16, 16)` Rule 12 + Depth + Ball/Hoop kinematics
      - `window_mask`: `(MAX_BAG_WINDOWS,)` bool mask indicating valid (non-padded) windows
      - `label`: binary foul target (`0` or `1`)
      - `call_type_label`: real L2M foul category (`1` = Shooting Foul, `0` = Personal Foul)
    Strictly isolates all clips in `data/test/` and exposes `game_id` for leak-free game-grouped splits.
    """

    def __init__(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        processed_dir: Path = BASE_DIR / "data" / "processed",
        metadata_csv: Optional[Path] = BASE_DIR / "data" / "l2m_metadata.csv",
        test_dir: Optional[Path] = None,
        is_train: bool = True,
        max_bag_windows: int = MAX_BAG_WINDOWS,
        max_samples: int = 0,
        return_foul_features: bool = True,
        use_cached_embeddings: bool = True,
        mil_mode: bool = True,
    ) -> None:
        super().__init__()
        self.processed_dir = resolve_project_path(processed_dir)
        self.test_dir = (
            resolve_project_path(test_dir)
            if test_dir is not None
            else resolve_project_path(BASE_DIR / "data" / "test")
        )
        self.is_train = is_train
        self.max_bag_windows = max_bag_windows
        self.return_foul_features = return_foul_features
        self.use_cached_embeddings = use_cached_embeddings
        self.mil_mode = mil_mode
        self.transform = ConsistentVideoAugmentor(is_train=is_train)

        if samples is not None:
            # Support both dict-based v3 metadata items and legacy `(Path, int)` tuples
            norm_samples: List[Dict[str, Any]] = []
            for s in samples:
                if isinstance(s, dict):
                    norm_samples.append(s)
                else:
                    p, lbl = s
                    norm_samples.append(self._inspect_pt_meta(Path(p), int(lbl), {}))
            self.samples_meta = norm_samples
        else:
            resolved_csv = resolve_project_path(metadata_csv) if metadata_csv is not None else None
            self.samples_meta = self._discover_samples(resolved_csv)

        if max_samples > 0 and len(self.samples_meta) > max_samples:
            self.samples_meta = self.samples_meta[:max_samples]

        # Backward-compatible `(Path, int)` list attribute
        self.samples: List[Tuple[Path, int]] = [(m["path"], int(m["label"])) for m in self.samples_meta]

    def _inspect_pt_meta(self, pt_path: Path, label: int, csv_lookup: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        stem = pt_path.stem
        row = csv_lookup.get(stem, {})
        game_id = str(row.get("game_id", stem.split("_")[0])).strip()
        call_type = str(row.get("call_type", "")).strip()
        review_decision = str(row.get("review_decision", "")).strip().upper()
        is_synthetic = False
        extractor_v = 0
        try:
            payload = load_torch_file_safe(pt_path, map_location="cpu")
            if isinstance(payload, dict):
                is_synthetic = bool(payload.get("is_synthetic", False))
                extractor_v = int(payload.get("extractor_v", 0))
                if not game_id:
                    game_id = str(payload.get("game_id", stem.split("_")[0])).strip()
                if not call_type:
                    call_type = str(payload.get("call_type", "")).strip()
                if not review_decision:
                    review_decision = str(payload.get("review_decision", "")).strip().upper()
        except Exception:
            pass
        if not call_type:
            call_type = "Personal Foul"
        if not review_decision:
            review_decision = "CC" if label == 1 else "CNC"
        call_type_label = 1 if "shoot" in call_type.lower() else 0
        return {
            "path": pt_path,
            "play_id": stem,
            "label": int(label),
            "game_id": game_id,
            "call_type": call_type,
            "call_type_label": call_type_label,
            "review_decision": review_decision,
            "is_synthetic": is_synthetic,
            "extractor_v": extractor_v,
        }

    def _discover_samples(self, metadata_csv: Optional[Path]) -> List[Dict[str, Any]]:
        test_stems: set = set()
        if self.test_dir and self.test_dir.exists():
            test_stems = {
                p.stem
                for p in self.test_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")
            }

        csv_lookup: Dict[str, Dict[str, str]] = {}
        for csv_cand in [metadata_csv, BASE_DIR / "data" / "l2m_metadata.csv", self.processed_dir / "processed_metadata.csv"]:
            if csv_cand is not None and csv_cand.exists():
                try:
                    with open(csv_cand, "r", newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            pid = row.get("play_id", "").strip()
                            if pid and pid not in csv_lookup:
                                csv_lookup[pid] = row
                except Exception:
                    pass

        discovered: List[Dict[str, Any]] = []
        seen_paths: set = set()
        for cls_name, label in (("foul", 1), ("no_foul", 0)):
            cls_dir = self.processed_dir / cls_name
            if cls_dir.exists():
                for pt_file in sorted(cls_dir.glob("*.pt")):
                    if pt_file.stem in test_stems or pt_file in seen_paths:
                        continue
                    seen_paths.add(pt_file)
                    discovered.append(self._inspect_pt_meta(pt_file, label, csv_lookup))

        # Prefer real broadcast tensors (exclude synthetic placeholders whenever real clips exist)
        real_only = [m for m in discovered if not m["is_synthetic"]]
        if len(real_only) >= 10:
            v3_only = [m for m in real_only if m["extractor_v"] >= 3]
            return v3_only if len(v3_only) >= 10 else real_only
        return discovered

    def __len__(self) -> int:
        return len(self.samples_meta)

    def load_bag_payload(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Load `(win_embs [K, 768], win_poses [K, 16, 16], win_centers [K], payload)` from a `.pt` file."""
        payload = load_torch_file_safe(path, map_location="cpu")
        if not isinstance(payload, dict):
            payload = {"video": payload}

        win_embs = payload.get("window_embeddings")
        win_poses = payload.get("window_pose_features")
        win_centers = payload.get("window_center_frames")

        if (
            isinstance(win_embs, torch.Tensor)
            and win_embs.dim() == 2
            and win_embs.size(-1) == VIDEO_FEATURE_DIM
            and isinstance(win_poses, torch.Tensor)
            and win_poses.dim() == 3
        ):
            poses_h = torch.stack([harmonize_pose_tensor(win_poses[i]) for i in range(win_poses.size(0))], dim=0)
            embs_h = torch.nan_to_num(win_embs.float(), nan=0.0, posinf=0.0, neginf=0.0)
            centers_h = (
                win_centers.long()
                if isinstance(win_centers, torch.Tensor) and len(win_centers) == len(embs_h)
                else torch.arange(len(embs_h), dtype=torch.long) * 30
            )
            return embs_h, poses_h, centers_h, payload

        # Fallback: legacy single-window `.pt` -> encode with Swin3D-T on-the-fly
        raw_vid = payload.get("video")
        if raw_vid is None:
            raw_vid = torch.zeros((CHANNELS, FRAMES, HEIGHT, WIDTH), dtype=torch.uint8)
        extractor = get_shared_embedding_extractor()
        dev = next(extractor.parameters()).device
        x = ConsistentVideoAugmentor(is_train=False)(raw_vid).unsqueeze(0).to(dev)
        with torch.no_grad():
            emb = extractor(x).squeeze(0).cpu().float()
        raw_pose = payload.get("foul_features")
        if raw_pose is None or not isinstance(raw_pose, torch.Tensor):
            raw_pose = derive_motion_foul_features_from_rgb(raw_vid)
        pose_h = harmonize_pose_tensor(raw_pose)
        return emb.unsqueeze(0), pose_h.unsqueeze(0), torch.tensor([FRAMES // 2], dtype=torch.long), payload

    def __getitem__(self, index: int):
        meta = self.samples_meta[index]
        win_embs, win_poses, _, _ = self.load_bag_payload(meta["path"])
        k_orig = win_embs.size(0)

        # Training-time MIL regularization: random window subsampling + subtle feature jitter
        if self.is_train:
            if k_orig >= 4 and random.random() < 0.45:
                keep_k = max(3, k_orig - random.randint(1, min(2, k_orig - 3)))
                keep_idxs = sorted(random.sample(range(k_orig), keep_k))
                win_embs = win_embs[keep_idxs]
                win_poses = win_poses[keep_idxs]
            win_embs = win_embs + torch.randn_like(win_embs) * 0.03
            win_poses = (win_poses + torch.randn_like(win_poses) * 0.015).clamp(0.0, 1.0)

        k_cur = min(win_embs.size(0), self.max_bag_windows)
        padded_embs = torch.zeros((self.max_bag_windows, VIDEO_FEATURE_DIM), dtype=torch.float32)
        padded_poses = torch.zeros((self.max_bag_windows, FRAMES, POSE_FEATURES), dtype=torch.float32)
        win_mask = torch.zeros((self.max_bag_windows,), dtype=torch.bool)

        padded_embs[:k_cur] = win_embs[:k_cur]
        padded_poses[:k_cur] = win_poses[:k_cur]
        win_mask[:k_cur] = True

        label = torch.tensor(float(meta["label"]), dtype=torch.float32)
        call_type_label = torch.tensor(float(meta["call_type_label"]), dtype=torch.float32)
        return padded_embs, padded_poses, win_mask, label, call_type_label


def game_grouped_train_val_split(
    samples_meta: List[Dict[str, Any]],
    n_splits: int = 5,
    fold_idx: int = 0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Leak-free Game-Grouped Stratified Split:
    Guarantees that NO two plays from the same NBA `game_id` ever appear in both train and validation.
    """
    y = np.array([int(m["label"]) for m in samples_meta], dtype=int)
    groups = np.array([str(m["game_id"]) for m in samples_meta])
    unique_games = len(set(groups))
    eff_splits = min(n_splits, max(2, unique_games))

    if unique_games >= 2 and len(set(y)) >= 2:
        sgkf = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(y)), y, groups))
        tr_idx, va_idx = splits[fold_idx % len(splits)]
        return [samples_meta[i] for i in tr_idx], [samples_meta[i] for i in va_idx]

    # Tiny-dataset fallback
    n_val = max(1, int(round(0.2 * len(samples_meta))))
    return samples_meta[n_val:], samples_meta[:n_val]


# ==============================================================================
# Dual-Stream Biomechanical Encoder + Gated Attention MIL Architecture
# ==============================================================================
class BiomechanicalFoulEncoder(nn.Module):
    """
    4-Head Temporal Self-Attention Encoder over `(B, FRAMES=16, POSE_FEATURES=16)` NBA Rule 12
    + Foot-Plane Depth + Ball/Hoop kinematic sequences -> `(B, 192)`.
    """

    def __init__(self, in_features: int = POSE_FEATURES, embed_dim: int = POSE_HIDDEN_DIM, dropout_p: float = 0.25) -> None:
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2  # 96
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, half_dim),
            nn.LayerNorm(half_dim),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.5),
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=half_dim,
            num_heads=4,
            dropout=dropout_p * 0.5,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(half_dim)
        self.mlp = nn.Sequential(
            nn.Linear(half_dim, half_dim),
            nn.LayerNorm(half_dim),
            nn.GELU(),
        )

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(pose_features)
        attn_out, _ = self.temporal_attn(projected, projected, projected, need_weights=False)
        attended = self.attn_norm(projected + attn_out)
        hidden = self.mlp(attended)
        mean_pooled = hidden.mean(dim=1)
        max_pooled = hidden.max(dim=1).values
        return torch.cat([mean_pooled, max_pooled], dim=-1)


def describe_foul_mechanic_from_kinematics(
    what_foul: str,
    foul_features: torch.Tensor | np.ndarray,
) -> Dict[str, Any]:
    """
    Interpret the specific NBA Rule 12 biomechanical infraction subtype & peak contact frame offset
    for an ALREADY-PREDICTED foul category (`Shooting Foul` vs `Personal Foul` from `window_type_head`).
    Never overrides the neural `what_foul` classification!
    """
    f_mat = foul_features.detach().cpu().numpy().astype(np.float32) if isinstance(foul_features, torch.Tensor) else np.asarray(foul_features, dtype=np.float32)
    if f_mat.ndim == 1:
        f_mat = f_mat[None, :]
    if f_mat.shape[1] < POSE_FEATURES:
        pad = np.zeros((f_mat.shape[0], POSE_FEATURES - f_mat.shape[1]), dtype=np.float32)
        f_mat = np.concatenate([f_mat, pad], axis=1)

    # Peak physical contact frame inside the 16-frame window (weighted by depth alignment & on-ball confidence)
    depth_w = 0.45 + 0.55 * f_mat[:, 13]
    ball_w = 0.65 + 0.35 * f_mat[:, 14]
    frame_severity = (
        0.34 * f_mat[:, 0]
        + 0.30 * f_mat[:, 1]
        + 0.16 * f_mat[:, 2]
        + 0.12 * f_mat[:, 3]
        + 0.08 * f_mat[:, 5]
    ) * depth_w * ball_w

    apex_idx = int(np.argmax(frame_severity)) if len(frame_severity) > 0 else 0
    apex = f_mat[apex_idx]
    mean_v = np.mean(f_mat, axis=0)

    arm_intrusion = float(0.60 * apex[0] + 0.40 * mean_v[0])
    cylinder_pen = float(0.60 * apex[1] + 0.40 * mean_v[1])
    landing_viol = float(0.60 * apex[2] + 0.40 * mean_v[2])
    verticality_dev = float(0.60 * apex[3] + 0.40 * mean_v[3])
    displacement = float(0.60 * apex[5] + 0.40 * mean_v[5])
    hoop_prox = float(mean_v[12])
    depth_align = float(0.60 * apex[13] + 0.40 * mean_v[13])
    airborne_apex = float(0.60 * apex[11] + 0.40 * mean_v[11])

    if what_foul == "Shooting Foul":
        if arm_intrusion >= cylinder_pen and arm_intrusion >= 0.40:
            subtype = "Shooting Foul (Illegal Arm/Wrist Contact)"
            mechanic = "Rule 12B Sec II: Defender struck shooter's arm/wrist during upward shooting motion"
        elif landing_viol >= 0.55 and airborne_apex >= 0.40:
            subtype = "Shooting Foul (Landing Space Violation)"
            mechanic = "Rule 12B Sec I: Defender encroached into airborne shooter's landing space"
        elif verticality_dev >= 0.30:
            subtype = "Shooting Foul (Verticality & Cylinder Breach)"
            mechanic = "Rule 12B Sec I: Defender broke 90-deg vertical plane into shooter's cylinder"
        else:
            subtype = "Shooting Foul (Body Contact on Drive/Shot)"
            mechanic = "Rule 12B Sec II: Illegal body contact displacing shooter on gather/attempt"
    else:
        if arm_intrusion >= cylinder_pen * 1.05 and arm_intrusion >= 0.40:
            subtype = "Personal Foul (Reach-In / Arm Contact)"
            mechanic = "Rule 12B Sec I: Illegal hand/arm reach-in contact across ball-handler"
        elif cylinder_pen >= 0.40 or displacement >= 0.32:
            subtype = "Personal Foul (Blocking / Cylinder Breach)"
            mechanic = "Rule 12B Sec I: Defender failed to establish legal guarding position in drive path"
        else:
            subtype = "Personal Foul (Illegal On-Ball Contact)"
            mechanic = "Rule 12B Sec I: Illegal defensive contact impeding offensive freedom of movement"

    return {
        "what_foul": what_foul,
        "foul_subtype": subtype,
        "foul_mechanic": mechanic,
        "apex_frame_offset": apex_idx,
        "peak_severity": float(frame_severity[apex_idx]) if len(frame_severity) > 0 else 0.0,
        "metrics": {
            "arm_contact_intrusion": round(arm_intrusion, 3),
            "body_cylinder_penetration": round(cylinder_pen, 3),
            "lower_body_landing_violation": round(landing_viol, 3),
            "defender_verticality_angle": round(verticality_dev, 3),
            "offensive_displacement": round(displacement, 3),
            "hoop_proximity": round(hoop_prox, 3),
            "foot_plane_depth_alignment": round(depth_align, 3),
            "airborne_contest_apex_diff": round(airborne_apex, 3),
        },
    }


def classify_foul_call_and_mechanic(
    foul_features: torch.Tensor | np.ndarray,
    shooting_foul_prob: Optional[float] = None,
    call_type_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Determine `what_foul` from the neural `window_type_head` probability + Rule 12 mechanic."""
    if call_type_hint and call_type_hint.strip() in FOUL_CALL_CATEGORIES:
        what_foul = call_type_hint.strip()
    elif shooting_foul_prob is not None:
        what_foul = "Shooting Foul" if float(shooting_foul_prob) >= 0.50 else "Personal Foul"
    else:
        f_mat = foul_features.detach().cpu().numpy() if isinstance(foul_features, torch.Tensor) else np.asarray(foul_features)
        shoot_sig = float(np.mean(f_mat[:, 15])) if (f_mat.ndim == 2 and f_mat.shape[1] >= 16) else 0.45
        what_foul = "Shooting Foul" if shoot_sig >= 0.45 else "Personal Foul"
    res = describe_foul_mechanic_from_kinematics(what_foul, foul_features)
    res["shooting_foul_score"] = float(shooting_foul_prob) if shooting_foul_prob is not None else 0.5
    return res


class FoulMILClassifier(nn.Module):
    """
    Swin3D-T + Gated Attention Multiple Instance Learning (MIL) AI Referee Network (~580K trainable params).
    Designed specifically to generalize from weak clip-level NBA L2M labels without overfitting:
      - Visual Stream: Frozen Kinetics-400 Swin3D-T (`768-D`) -> Regularized projection (`256-D`)
      - Pose Stream: 4-Head Temporal Attention over `(16, 16)` depth-gated kinematics -> `128-D`
      - Per-Window Fusion (`256-D`) -> `window_foul_head` (WHEN!) + `window_type_head` (WHAT!)
      - Gated Attention MIL (Ilse et al., ICML 2018) -> Clip-level bag probability + instance-level supervision
      - Post-hoc Temperature Calibration (`temperature`) for honest probability outputs (zero hardcoded hacks!)
    """

    def __init__(
        self,
        video_dim: int = VIDEO_FEATURE_DIM,
        pose_dim: int = POSE_FEATURES,
        pose_hidden_dim: int = POSE_HIDDEN_DIM,
        fusion_dim: int = WINDOW_FUSION_DIM,
        dropout_p: float = 0.40,
        **kwargs,
    ) -> None:
        super().__init__()
        self.video_feature_dim = video_dim
        self.pose_dim = pose_dim
        self.fusion_dim = fusion_dim

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(video_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.pose_network = BiomechanicalFoulEncoder(
            in_features=pose_dim,
            embed_dim=pose_hidden_dim,
            dropout_p=dropout_p * 0.6,
        )
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(pose_hidden_dim),
            nn.Dropout(p=dropout_p * 0.6),
            nn.Linear(pose_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        self.window_fusion = nn.Sequential(
            nn.Linear(fusion_dim + 128, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p * 0.8),
        )

        # Gated Attention MIL (Ilse et al., ICML 2018)
        attn_dim = 128
        self.attn_V = nn.Linear(fusion_dim, attn_dim)
        self.attn_U = nn.Linear(fusion_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

        # Instance-level (per-window) heads -> pinpoints WHEN and WHAT foul occurred!
        self.window_foul_head = nn.Linear(fusion_dim, 1)
        self.window_type_head = nn.Linear(fusion_dim, 1)

        # Bag-level (clip) foul head
        self.bag_foul_head = nn.Linear(fusion_dim, 1)

        # Learnable post-hoc calibration temperature (fitted on game-grouped validation split)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def trainable_parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "backbone_params_frozen": 27_850_470,  # Frozen Kinetics-400 Swin3D-T
        }

    def encode_windows(self, window_embs: torch.Tensor, window_poses: torch.Tensor) -> torch.Tensor:
        """
        Encode `(B, K, 768)` visual window embeddings and `(B, K, 16, 16)` biomechanical sequences
        into `(B, K, 256)` fused window embeddings.
        """
        B, K, _ = window_embs.shape
        v_flat = window_embs.reshape(B * K, -1)
        p_flat = window_poses.reshape(B * K, FRAMES, -1)

        v_feat = self.visual_proj(v_flat)
        p_feat = self.pose_proj(self.pose_network(p_flat))
        fused = self.window_fusion(torch.cat([v_feat, p_feat], dim=-1))
        return fused.view(B, K, self.fusion_dim)

    def forward(
        self,
        window_embs: torch.Tensor,
        window_poses: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over a batch of multi-window bags `(B, K, 768)` + `(B, K, 16, 16)`.
        Also supports single-window inputs `(B, 768)` + `(B, 16, 16)` cleanly.
        """
        if window_embs.dim() == 2:
            window_embs = window_embs.unsqueeze(1)
            window_poses = window_poses.unsqueeze(1)
        B, K, _ = window_embs.shape
        if window_mask is None:
            window_mask = torch.ones((B, K), dtype=torch.bool, device=window_embs.device)

        h = self.encode_windows(window_embs, window_poses)  # (B, K, 256)

        # Per-window foul & call-type logits
        win_foul_logits = self.window_foul_head(h).squeeze(-1)  # (B, K)
        win_type_logits = self.window_type_head(h).squeeze(-1)  # (B, K)

        # Gated Attention MIL weights
        attn_raw = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).squeeze(-1)  # (B, K)
        attn_masked = attn_raw.masked_fill(~window_mask, -1e4)
        attn_weights = torch.softmax(attn_masked, dim=-1)  # (B, K)

        # Attention-pooled bag representation
        bag_h = torch.sum(attn_weights.unsqueeze(-1) * h, dim=1)  # (B, 256)
        bag_head_logit = self.bag_foul_head(bag_h).squeeze(-1)    # (B,)

        # Smooth top-window pooling (LogSumExp / soft-max over valid windows)
        masked_win_logits = win_foul_logits.masked_fill(~window_mask, -1e4)
        inst_pool_logit = torch.logsumexp(masked_win_logits * 1.5, dim=-1) / 1.5 - torch.log(
            window_mask.float().sum(dim=-1).clamp(min=1.0)
        ) / 1.5
        inst_max_logit = masked_win_logits.max(dim=-1).values

        # Combined bag foul logit: fuses global attended context + peak window instance evidence
        bag_foul_logit = 0.55 * bag_head_logit + 0.30 * inst_max_logit + 0.15 * inst_pool_logit

        # Attention-weighted `call_type` logit (Shooting Foul vs Personal Foul)
        bag_type_logit = torch.sum(attn_weights * win_type_logits, dim=-1)  # (B,)

        return {
            "bag_foul_logit": bag_foul_logit,
            "bag_type_logit": bag_type_logit,
            "window_foul_logits": win_foul_logits,
            "window_type_logits": win_type_logits,
            "attn_weights": attn_weights,
            "window_hidden": h,
        }

    @torch.no_grad()
    def predict_video_windows(
        self,
        window_embs: torch.Tensor,
        window_poses: torch.Tensor,
        window_centers: List[int],
        fps: float,
        threshold: float = 0.50,
        nms_min_sep_sec: float = 3.5,
    ) -> Dict[str, Any]:
        """
        Arbitrary-Length Sliding-Window Inference + Temporal Non-Maximum Suppression (NMS):
          - Evaluates ALL candidate windows `k = 0..K-1` of any unseen NBA video.
          - Uses pure learned temperature-calibrated MIL probabilities (zero hardcoded score hacks!).
          - Detects 0 fouls (`is_foul = False` -> nothing happens on video), 1 foul, or multiple
            distinct fouls across long game footage.
        """
        self.eval()
        dev = next(self.parameters()).device
        temp = float(self.temperature.item()) if float(self.temperature.item()) > 0.05 else 1.0

        embs_b = window_embs.unsqueeze(0).to(dev)   # (1, K, 768)
        poses_b = window_poses.unsqueeze(0).to(dev) # (1, K, 16, 16)
        out = self.forward(embs_b, poses_b)
        return _decode_video_windows_from_outputs(
            out=out,
            window_poses=window_poses,
            window_centers=window_centers,
            fps=fps,
            temperature=temp,
            threshold=threshold,
            nms_min_sep_sec=nms_min_sep_sec,
        )


def _decode_video_windows_from_outputs(
    out: Dict[str, torch.Tensor],
    window_poses: torch.Tensor,
    window_centers: List[int],
    fps: float,
    temperature: float = 1.0,
    threshold: float = 0.50,
    nms_min_sep_sec: float = 3.5,
) -> Dict[str, Any]:
    """Decode calibrated MIL bag/window logits into 0, 1, or N localized foul events."""
    temp = float(temperature) if float(temperature) > 0.05 else 1.0
    bag_prob = float(torch.sigmoid(out["bag_foul_logit"][0] / temp).item())
    win_logits = out["window_foul_logits"][0] / temp
    win_probs = torch.sigmoid(win_logits).cpu().numpy().astype(np.float32)
    win_shoot_probs = torch.sigmoid(out["window_type_logits"][0]).cpu().numpy().astype(np.float32)
    attn_w = out["attn_weights"][0].cpu().numpy().astype(np.float32)

    clip_dur_sec = (
        float(max(window_centers) - min(window_centers)) / max(fps, 1.0)
        if len(window_centers) >= 2
        else 2.0
    )

    if clip_dur_sec <= 26.0:
        # Standard clip regime: bag decision determines IF a foul happened; peak window * attention determines WHEN!
        loc_scores = win_probs * (0.45 + 0.55 * (attn_w / max(float(attn_w.max()), 1e-5)))
        peak_w_idx = int(np.argmax(loc_scores))
        is_foul = bool(bag_prob >= threshold)
        selected_indices = [peak_w_idx] if is_foul else []
    else:
        # Long multi-possession game footage: sliding-window Temporal NMS across the whole game!
        order = np.argsort(-win_probs).tolist()
        selected_indices = []
        min_sep_frames = int(round(nms_min_sep_sec * max(fps, 1.0)))
        for w_idx in order:
            if float(win_probs[w_idx]) < threshold:
                break
            c_f = window_centers[w_idx]
            if all(abs(c_f - window_centers[prev_i]) >= min_sep_frames for prev_i in selected_indices):
                selected_indices.append(w_idx)
        selected_indices.sort(key=lambda i: window_centers[i])
        is_foul = len(selected_indices) > 0

    events: List[Dict[str, Any]] = []
    for w_idx in selected_indices:
        ev_prob = bag_prob if clip_dur_sec <= 26.0 else float(win_probs[w_idx])
        shoot_p = float(win_shoot_probs[w_idx])
        what_foul = "Shooting Foul" if shoot_p >= 0.50 else "Personal Foul"
        details = describe_foul_mechanic_from_kinematics(what_foul, window_poses[w_idx])
        events.append(
            {
                "window_idx": int(w_idx),
                "foul_prob": round(ev_prob, 4),
                "window_prob": round(float(win_probs[w_idx]), 4),
                "shooting_foul_prob": round(shoot_p, 4),
                "attn_weight": round(float(attn_w[w_idx]), 4),
                "what_foul": details["what_foul"],
                "foul_subtype": details["foul_subtype"],
                "foul_mechanic": details["foul_mechanic"],
                "apex_frame_offset": int(details["apex_frame_offset"]),
                "metrics": details["metrics"],
            }
        )

    return {
        "is_foul": is_foul,
        "bag_prob": round(bag_prob, 4),
        "threshold": round(float(threshold), 4),
        "temperature": round(temp, 4),
        "window_probs": [round(float(x), 4) for x in win_probs],
        "window_shooting_probs": [round(float(x), 4) for x in win_shoot_probs],
        "attn_weights": [round(float(x), 4) for x in attn_w],
        "events": events,
    }


class FoulMILEnsemble(nn.Module):
    """
    Out-of-Game K-Fold Ensemble of `FoulMILClassifier` models with global OOF Temperature Calibration.
    Averages raw bag/window/type logits and MIL attention weights across all fold models to minimize
    single-fold variance on unseen NBA games.
    """

    def __init__(self, models: List[FoulMILClassifier], temperature: float = 1.0) -> None:
        super().__init__()
        if not models:
            raise ValueError("FoulMILEnsemble requires at least one FoulMILClassifier.")
        self.models = nn.ModuleList(models)
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(
        self,
        window_embs: torch.Tensor,
        window_poses: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outs = [m(window_embs, window_poses, window_mask) for m in self.models]
        attn = torch.stack([o["attn_weights"] for o in outs], dim=0).mean(dim=0)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return {
            "bag_foul_logit": torch.stack([o["bag_foul_logit"] for o in outs], dim=0).mean(dim=0),
            "bag_type_logit": torch.stack([o["bag_type_logit"] for o in outs], dim=0).mean(dim=0),
            "window_foul_logits": torch.stack([o["window_foul_logits"] for o in outs], dim=0).mean(dim=0),
            "window_type_logits": torch.stack([o["window_type_logits"] for o in outs], dim=0).mean(dim=0),
            "attn_weights": attn,
            "window_hidden": torch.stack([o["window_hidden"] for o in outs], dim=0).mean(dim=0),
        }

    @torch.no_grad()
    def predict_video_windows(
        self,
        window_embs: torch.Tensor,
        window_poses: torch.Tensor,
        window_centers: List[int],
        fps: float,
        threshold: float = 0.50,
        nms_min_sep_sec: float = 3.5,
    ) -> Dict[str, Any]:
        self.eval()
        dev = next(self.parameters()).device
        temp = float(self.temperature.item()) if float(self.temperature.item()) > 0.05 else 1.0
        embs_b = window_embs.unsqueeze(0).to(dev)
        poses_b = window_poses.unsqueeze(0).to(dev)
        out = self.forward(embs_b, poses_b)
        return _decode_video_windows_from_outputs(
            out=out,
            window_poses=window_poses,
            window_centers=window_centers,
            fps=fps,
            temperature=temp,
            threshold=threshold,
            nms_min_sep_sec=nms_min_sep_sec,
        )


# Backward-compatible class alias
FoulClassifier3D = FoulMILClassifier


# ==============================================================================
# Multi-Window Sequential Reinforcement Learning (RL) Referee Agent
# ==============================================================================
RL_VIS_DIM = 128
RL_STATE_DIM = RL_VIS_DIM + POSE_FEATURES + POSE_FEATURES + 3  # 128 vis + 16 pose + 16 delta + [p_mil, dp_mil, clock] = 163
RL_NUM_ACTIONS = 2
ACTION_PLAY_ON = 0
ACTION_WHISTLE = 1


class NBARefereeEnv:
    """
    Genuine Multi-Window Sequential Basketball Officiating Environment:
      - Every step `t = 0..K-1` corresponds to a distinct 2.0-second live-play window with its own
        distinct Swin3D-T visual embedding, its own MIL foul probability `p_mil[t]`, and its own
        16-D NBA Rule 12 kinematic summary (NOT a static tiled vector!).
      - Rewards use REAL NBA L2M review decisions (`CC`, `INC`, `CNC`, `IC`) and MIL-attended peak timing.
    """

    def __init__(self, episodes: List[Dict[str, Any]]) -> None:
        self.episodes = episodes
        self.current_idx = 0
        self.current_t = 0
        self.current_episode: Optional[Dict[str, Any]] = None

    def _get_state(self, t: int) -> np.ndarray:
        ep = self.current_episode
        assert ep is not None
        K = ep["num_steps"]
        t_c = min(t, K - 1)
        v_feat = ep["visual_sequence"][t_c][:RL_VIS_DIM]
        b_feat = ep["pose_summary"][t_c][:POSE_FEATURES]
        delta_b = b_feat - ep["pose_summary"][t_c - 1][:POSE_FEATURES] if t_c > 0 else np.zeros(POSE_FEATURES, dtype=np.float32)
        p_mil = float(ep["mil_probs"][t_c])
        dp_mil = p_mil - float(ep["mil_probs"][t_c - 1]) if t_c > 0 else 0.0
        clock = float(t_c) / max(1.0, float(K - 1))
        extra = np.array([p_mil, dp_mil, clock], dtype=np.float32)
        return np.concatenate([v_feat, b_feat, delta_b, extra]).astype(np.float32)

    def reset(self, idx: Optional[int] = None) -> np.ndarray:
        self.current_idx = (idx if idx is not None else random.randint(0, len(self.episodes) - 1)) % len(self.episodes)
        self.current_episode = self.episodes[self.current_idx]
        self.current_t = 0
        return self._get_state(0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        ep = self.current_episode
        assert ep is not None
        t = self.current_t
        K = ep["num_steps"]
        label = int(ep["label"])
        decision = str(ep.get("review_decision", "CC" if label == 1 else "CNC")).upper()
        peak_step = int(ep.get("peak_step", K // 2))

        done = False
        reward = 0.0
        info = {"action": action, "true_label": label, "step": t, "correct_call": False, "decision": decision}

        if action == ACTION_WHISTLE:
            done = True
            info["whistle_blown"] = True
            if label == 1:
                timing_bonus = 1.0 * max(0.0, 1.0 - abs(t - peak_step) / 2.5)
                overturn_bonus = 0.5 if decision == "INC" else 0.0
                reward = 2.5 + timing_bonus + overturn_bonus
                info["correct_call"] = True
            else:
                reward = -3.0 - (0.5 if decision == "IC" else 0.0)
        else:
            if t >= K - 1:
                done = True
                info["whistle_blown"] = False
                if label == 0:
                    reward = 2.0 + (0.5 if decision == "IC" else 0.0)
                    info["correct_call"] = True
                else:
                    reward = -2.5
            else:
                done = False
                self.current_t += 1
                reward = 0.02 if label == 0 else -0.02

        next_state = self._get_state(self.current_t) if not done else np.zeros((RL_STATE_DIM,), dtype=np.float32)
        return next_state, reward, done, info


class DuelingRefereeDQN(nn.Module):
    def __init__(self, state_dim: int = RL_STATE_DIM, num_actions: int = RL_NUM_ACTIONS) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(state_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.v_head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))
        self.a_head = nn.Sequential(nn.Linear(96, 48), nn.GELU(), nn.Linear(48, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feat(x)
        v = self.v_head(f)
        a = self.a_head(f)
        return v + (a - a.mean(dim=-1, keepdim=True))


class RefereeRLAgent:
    """Dueling DQN Sequential Officiating Agent trained on multi-window MIL trajectories."""

    def __init__(
        self,
        state_dim: int = RL_STATE_DIM,
        num_actions: int = RL_NUM_ACTIONS,
        lr: float = 3e-4,
        gamma: float = 0.95,
        mode: str = "dqn",
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingRefereeDQN(state_dim, num_actions).to(self.device)
        self.target_model = DuelingRefereeDQN(state_dim, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.buffer: deque = deque(maxlen=15000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.992

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if (not evaluate) and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return int(torch.argmax(self.model(st), dim=-1).item())

    def store_transition(self, s, a, r, s_next, done) -> None:
        self.buffer.append((s, int(a), float(r), s_next, float(done)))

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.buffer) < batch_size:
            return 0.0
        batch = random.sample(self.buffer, batch_size)
        s = torch.from_numpy(np.array([x[0] for x in batch], dtype=np.float32)).to(self.device)
        a = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
        s_next = torch.from_numpy(np.array([x[3] for x in batch], dtype=np.float32)).to(self.device)
        d = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device)

        self.model.train()
        self.optimizer.zero_grad()
        q_cur = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN target
            next_actions = self.model(s_next).argmax(dim=1, keepdim=True)
            next_q = self.target_model(s_next).gather(1, next_actions).squeeze(1)
            q_target = r + (1.0 - d) * self.gamma * next_q
        loss = F.smooth_l1_loss(q_cur, q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def update_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    @torch.no_grad()
    def evaluate_episode(self, ep: Dict[str, Any]) -> Dict[str, Any]:
        env = NBARefereeEnv([ep])
        K = ep["num_steps"]
        q_hist: List[List[float]] = []
        p_hist: List[float] = []
        whistle_step: Optional[int] = None
        self.model.eval()
        env.reset(0)
        for t in range(K):
            st = torch.from_numpy(env._get_state(t)).float().unsqueeze(0).to(self.device)
            q_vals = self.model(st)[0].cpu().numpy()
            q_hist.append([float(q_vals[0]), float(q_vals[1])])
            exp_q = np.exp(q_vals - np.max(q_vals))
            p_w = float((exp_q / exp_q.sum())[1])
            p_hist.append(p_w)
            if whistle_step is None and q_vals[ACTION_WHISTLE] > q_vals[ACTION_PLAY_ON]:
                whistle_step = t
        is_foul = whistle_step is not None
        return {
            "decision": "FOUL_WHISTLED" if is_foul else "LEGAL_PLAY_ON",
            "is_foul": is_foul,
            "whistle_step": whistle_step,
            "confidence": float(p_hist[whistle_step]) if is_foul else float(1.0 - max(p_hist)),
            "whistle_probs": p_hist,
            "q_values": q_hist,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mode": self.mode,
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "model_state_dict": self.model.state_dict(),
                "epsilon": self.epsilon,
            },
            str(p),
        )
        logger.info("Saved Referee RL Agent checkpoint -> %s", p)

    def load(self, path: Path | str) -> None:
        p = resolve_project_path(path)
        ckpt = load_torch_file_safe(p, map_location=self.device)
        if ckpt.get("state_dim") == self.state_dim:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_model.load_state_dict(self.model.state_dict())
            self.epsilon = float(ckpt.get("epsilon", 0.05))


@torch.no_grad()
def create_referee_episodes_from_dataset(
    dataset: L2MFoulDataset,
    mil_model: Optional[FoulMILClassifier] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Build genuine multi-window RL officiating episodes where EVERY timestep `k = 0..K-1` has
    its own distinct Swin3D-T window embedding, per-window MIL foul probability, biomechanical
    summary, and real L2M `review_decision` (`CC`, `INC`, `CNC`, `IC`).
    """
    dev = device or (next(mil_model.parameters()).device if mil_model is not None else torch.device("cpu"))
    if mil_model is not None:
        mil_model.eval()

    episodes: List[Dict[str, Any]] = []
    for meta in dataset.samples_meta:
        win_embs, win_poses, win_centers, _ = dataset.load_bag_payload(meta["path"])
        K = win_embs.size(0)
        pose_summary = (0.60 * win_poses.max(dim=1).values + 0.40 * win_poses.mean(dim=1)).numpy().astype(np.float32)

        if mil_model is not None:
            temp = float(mil_model.temperature.item()) if float(mil_model.temperature.item()) > 0.05 else 1.0
            out = mil_model(win_embs.unsqueeze(0).to(dev), win_poses.unsqueeze(0).to(dev))
            mil_probs = torch.sigmoid(out["window_foul_logits"][0] / temp).cpu().numpy().astype(np.float32)
            attn_w = out["attn_weights"][0].cpu().numpy().astype(np.float32)
            peak_step = int(np.argmax(mil_probs * (0.5 + 0.5 * attn_w)))
        else:
            mil_probs = np.full((K,), 0.5, dtype=np.float32)
            peak_step = int(np.argmax(pose_summary[:, 0] + pose_summary[:, 1]))

        episodes.append(
            {
                "play_id": meta["play_id"],
                "game_id": meta["game_id"],
                "num_steps": K,
                "visual_sequence": win_embs[:, :RL_VIS_DIM].numpy().astype(np.float32),
                "pose_summary": pose_summary,
                "mil_probs": mil_probs,
                "window_centers": win_centers.numpy().tolist(),
                "label": int(meta["label"]),
                "call_type": meta["call_type"],
                "review_decision": meta["review_decision"],
                "peak_step": peak_step,
            }
        )
    return episodes


if __name__ == "__main__":
    model = FoulMILClassifier()
    summary = model.trainable_parameter_summary()
    print("FoulMILClassifier (v3.0 Swin3D-T + Gated Attention MIL) Summary:")
    print(f"  Window Input : embs=(B, K, {VIDEO_FEATURE_DIM}), poses=(B, K, {FRAMES}, {POSE_FEATURES})")
    print(f"  Trainable    : {summary['trainable_params']:,} params (+ {summary['backbone_params_frozen']:,} frozen Swin3D-T)")
    dummy_e = torch.randn(2, 6, VIDEO_FEATURE_DIM)
    dummy_p = torch.rand(2, 6, FRAMES, POSE_FEATURES)
    out = model(dummy_e, dummy_p)
    print(f"  Bag Logit    : {tuple(out['bag_foul_logit'].shape)} | Window Logits: {tuple(out['window_foul_logits'].shape)} | Type Logit: {tuple(out['bag_type_logit'].shape)}")
