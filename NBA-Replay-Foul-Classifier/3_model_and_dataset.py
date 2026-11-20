#!/usr/bin/env python3
"""
STAGE 3: Binary Dual-Stream Foul Classifier & Replay Dataset (`3_model_and_dataset.py`)

Pure binary classification (`1 = FOUL`, `0 = NO FOUL`) engineered for NBA Replay Archive clips:
  1. `ReplayFoulDataset`:
       - Loads shot-bounded multi-angle `.pt` action tubes `(3, 16, 224, 224)` + `(16, 12)` pose kinematics.
       - Tracks `play_id` and `game_id` per sample for leak-free `GroupShuffleSplit` in Stage 4.
       - Strictly excludes any video stems present in `data/test/`.
       - Supports raw 4D video training (for GPU fine-tuning) and self-healing cached 512-D embeddings
         (for rapid laptop CPU/MPS training).
  2. `BinaryFoulClassifier`:
       - Stream A: Pretrained `R(2+1)D-18` spatiotemporal video backbone -> `512-D` visual embedding
         (supports `backbone_mode`: `'frozen'`, `'partial'` [layer3+layer4], or `'full'`).
       - Stream B: `PoseTemporalEncoder` with learned positional encoding, 4-head temporal self-attention,
         and attention + max pooling -> `256-D` kinematic embedding.
       - Fusion Head: `768-D -> 384-D -> 128-D -> 1` binary logit with modality dropout regularization.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("replay_model")

# Standardized tensor dimensions
CHANNELS = 3
FRAMES = 16
HEIGHT = 224
WIDTH = 224
POSE_FEATURES = 12

VIDEO_FEATURE_DIM = 512
POSE_HIDDEN_DIM = 256
COMBINED_FEATURE_DIM = VIDEO_FEATURE_DIM + POSE_HIDDEN_DIM  # 768
CLASSIFIER_HIDDEN_DIM = 384

KINETICS400_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS400_STD = (0.22803, 0.22145, 0.216989)

POSE_FEATURE_NAMES: Tuple[str, ...] = (
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
)


def resolve_project_path(path_arg: Path | str) -> Path:
    norm_str = str(path_arg).replace("\\", "/")
    p = Path(norm_str)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == BASE_DIR.name:
        return (BASE_DIR.parent / p).resolve()
    for cand in (
        Path.cwd() / p,
        BASE_DIR / p,
        BASE_DIR / "weights" / p.name,
        BASE_DIR / "weights" / p,
        BASE_DIR.parent / p.name,
        BASE_DIR.parent / p,
        BASE_DIR.parent / "NBA-Foul-Classifier-WIP" / "weights" / p.name,
    ):
        if cand.exists() or cand.parent.exists():
            return cand.resolve()
    return (BASE_DIR / p).resolve()


def load_torch_file_safe(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def derive_motion_features_from_rgb(video_frames: torch.Tensor) -> torch.Tensor:
    """Fallback `(T, 12)` temporal motion descriptor if a `.pt` file lacks YOLO pose features."""
    v = video_frames.float()
    if v.max() > 1.5:
        v = v / 255.0
    _, num_frames, fh, fw = v.shape
    crop = v[:, :, int(fh * 0.15) : int(fh * 0.80), int(fw * 0.18) : int(fw * 0.82)]
    diffs = torch.zeros(num_frames, dtype=torch.float32)
    if num_frames > 1:
        diffs[1:] = torch.mean(torch.abs(crop[:, 1:] - crop[:, :-1]), dim=(0, 2, 3)) * 6.0
        diffs[0] = diffs[1]
    spread = torch.std(crop, dim=(0, 2, 3))
    bright = torch.mean(crop, dim=(0, 2, 3))

    feats = torch.zeros((num_frames, POSE_FEATURES), dtype=torch.float32)
    feats[:, 0] = torch.clamp(diffs, 0.0, 1.0)
    feats[:, 1] = torch.clamp(diffs * 1.15, 0.0, 1.0)
    feats[:, 2] = torch.clamp(spread * 2.0, 0.0, 1.0)
    feats[:, 3] = torch.clamp(diffs * spread * 3.0, 0.0, 1.0)
    feats[:, 4] = torch.clamp(0.5 + torch.gradient(bright)[0] * 4.0, 0.0, 1.0)
    feats[:, 5] = torch.clamp(0.5 + torch.gradient(diffs)[0] * 2.0, 0.0, 1.0)
    feats[:, 6] = torch.clamp(bright, 0.0, 1.0)
    feats[:, 11] = torch.clamp(0.6 * diffs + 0.4 * spread * 2.0, 0.0, 1.0)
    return feats


class ConsistentVideoAugmentor:
    """Temporally consistent vectorized augmentations for `(3, T, H, W)` video tubes."""

    def __init__(
        self,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
        cutout_prob: float = 0.20,
        mean: Tuple[float, float, float] = KINETICS400_MEAN,
        std: Tuple[float, float, float] = KINETICS400_STD,
    ) -> None:
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.cutout_prob = cutout_prob
        self.mean = torch.tensor(mean, dtype=torch.float32).view(CHANNELS, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(CHANNELS, 1, 1, 1)

    def __call__(self, video_frames: torch.Tensor) -> torch.Tensor:
        if video_frames.dtype == torch.uint8:
            v = video_frames.float() / 255.0
        else:
            v = video_frames.float()
            if v.max() > 1.5:
                v = v / 255.0

        if self.is_train:
            if random.random() < self.hflip_prob:
                v = torch.flip(v, dims=[-1])
            b_off = random.uniform(-self.brightness, self.brightness)
            c_fac = random.uniform(max(0.1, 1.0 - self.contrast), 1.0 + self.contrast)
            v = ((v - 0.5) * c_fac + 0.5 + b_off).clamp(0.0, 1.0)
            if random.random() < self.cutout_prob:
                _, _, fh, fw = v.shape
                ch = random.randint(int(fh * 0.10), int(fh * 0.24))
                cw = random.randint(int(fw * 0.10), int(fw * 0.24))
                cy = random.randint(0, max(0, fh - ch))
                cx = random.randint(0, max(0, fw - cw))
                v[:, :, cy : cy + ch, cx : cx + cw] = 0.0

        mean = self.mean.to(v.device)
        std = self.std.to(v.device)
        return (v - mean) / std


_SHARED_VISUAL_EXTRACTOR: Optional[nn.Module] = None


def get_shared_visual_extractor() -> nn.Module:
    """Singleton pretrained Kinetics-400 `R(2+1)D-18` visual extractor for 512-D embedding caching."""
    global _SHARED_VISUAL_EXTRACTOR
    if _SHARED_VISUAL_EXTRACTOR is None:
        try:
            ext = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        except Exception:
            ext = r2plus1d_18(weights=None)
        ext.fc = nn.Identity()
        if torch.cuda.is_available():
            ext = ext.to("cuda")
        ext.eval()
        _SHARED_VISUAL_EXTRACTOR = ext
    return _SHARED_VISUAL_EXTRACTOR


class ReplayFoulDataset(Dataset):
    """
    Dataset loader for preprocessed NBA Replay Archive multi-angle `.pt` action tubes.
    Tracks `play_id` and `game_id` for leak-free grouped Train/Val splitting and play-level
    multi-angle evaluation.
    """

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        processed_dir: Path = BASE_DIR / "data" / "processed",
        metadata_csv: Optional[Path] = BASE_DIR / "data" / "processed" / "processed_metadata.csv",
        test_dir: Optional[Path] = BASE_DIR / "data" / "test",
        is_train: bool = True,
        num_frames: int = FRAMES,
        crop_size: int = HEIGHT,
        max_samples: int = 0,
        use_cached_embeddings: bool = False,
        pose_noise_std: float = 0.02,
        return_meta: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.processed_dir = resolve_project_path(processed_dir)
        self.test_dir = resolve_project_path(test_dir) if test_dir is not None else resolve_project_path(BASE_DIR / "data" / "test")
        self.is_train = is_train
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.use_cached_embeddings = use_cached_embeddings
        self.pose_noise_std = pose_noise_std if is_train else 0.0
        self.return_meta = return_meta
        self.transform = transform or ConsistentVideoAugmentor(is_train=is_train)

        self._cache_path = self.processed_dir / "visual_embeddings_cache.pt"
        self._emb_cache: Dict[str, torch.Tensor] = {}
        if self.use_cached_embeddings and self._cache_path.exists():
            try:
                loaded = load_torch_file_safe(self._cache_path, map_location="cpu")
                if isinstance(loaded, dict):
                    self._emb_cache = loaded
            except Exception as exc:
                logger.warning("Could not load embedding cache %s: %s", self._cache_path, exc)

        if records is not None:
            self.records = list(records)
        else:
            resolved_csv = resolve_project_path(metadata_csv) if metadata_csv is not None else None
            self.records = self._discover_records(resolved_csv)

        if max_samples > 0 and len(self.records) > max_samples:
            self.records = self.records[:max_samples]

        # Convenience attributes
        self.samples: List[Tuple[Path, int]] = [(r["path"], int(r["label"])) for r in self.records]
        self.play_ids: List[str] = [str(r["play_id"]) for r in self.records]
        self.game_ids: List[str] = [str(r["game_id"]) for r in self.records]

    def _get_test_stems(self) -> Set[str]:
        if not self.test_dir or not self.test_dir.exists():
            return set()
        return {
            p.stem
            for p in self.test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")
        }

    def _discover_records(self, metadata_csv: Optional[Path]) -> List[Dict[str, Any]]:
        test_stems = self._get_test_stems()
        discovered: List[Dict[str, Any]] = []
        seen_paths: Set[Path] = set()

        if metadata_csv is not None and metadata_csv.exists():
            with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    raw_lbl = str(row.get("label", "")).strip()
                    if raw_lbl == "":
                        continue
                    label = int(raw_lbl)
                    play_id = str(row.get("play_id", "")).strip()
                    sample_id = str(row.get("sample_id", "")).strip() or play_id
                    if play_id in test_stems or sample_id in test_stems:
                        continue
                    game_id = str(row.get("game_id", "")).strip() or play_id.split("_")[0]
                    t_path_str = str(row.get("tensor_path", "")).strip()
                    cand = resolve_project_path(t_path_str) if t_path_str else (
                        self.processed_dir / ("foul" if label == 1 else "no_foul") / f"{sample_id}.pt"
                    )
                    if cand.exists() and cand not in seen_paths:
                        seen_paths.add(cand)
                        discovered.append({
                            "path": cand,
                            "label": label,
                            "sample_id": sample_id,
                            "play_id": play_id or cand.stem.split("_a")[0],
                            "game_id": game_id,
                            "outcome_category": str(row.get("outcome_category", "")),
                            "angle_idx": int(row.get("angle_idx", 0) or 0),
                        })

        # Also scan processed/{foul,no_foul} for any `.pt` files not yet in CSV
        for cls_name, label in (("foul", 1), ("no_foul", 0)):
            cls_dir = self.processed_dir / cls_name
            if not cls_dir.exists():
                continue
            for pt_file in sorted(cls_dir.glob("*.pt")):
                if pt_file in seen_paths:
                    continue
                stem = pt_file.stem
                base_play = stem.rsplit("_a", 1)[0] if "_a" in stem else stem
                if stem in test_stems or base_play in test_stems:
                    continue
                seen_paths.add(pt_file)
                discovered.append({
                    "path": pt_file,
                    "label": label,
                    "sample_id": stem,
                    "play_id": base_play,
                    "game_id": base_play.split("_")[0],
                    "outcome_category": "",
                    "angle_idx": 0,
                })

        # Exclude synthetic placeholder tensors whenever real processed clips exist
        real_records: List[Dict[str, Any]] = []
        for rec in discovered:
            try:
                payload = load_torch_file_safe(rec["path"], map_location="cpu")
                if isinstance(payload, dict) and payload.get("is_synthetic", False):
                    continue
            except Exception:
                continue
            real_records.append(rec)

        return real_records if real_records else discovered

    def save_embedding_cache(self) -> None:
        if self._emb_cache:
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._emb_cache, self._cache_path)
            except Exception as exc:
                logger.debug("Could not persist embedding cache: %s", exc)

    def __len__(self) -> int:
        return len(self.records)

    def _load_item(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        payload = load_torch_file_safe(path, map_location="cpu")
        if isinstance(payload, dict):
            video = payload["video"]
            pose = payload.get("pose_features")
            if pose is None:
                pose = payload.get("foul_features")
            emb = payload.get("video_embedding")
        else:
            video, pose, emb = payload, None, None

        if video.shape != (CHANNELS, self.num_frames, self.crop_size, self.crop_size):
            video = TF.resize(
                video.permute(1, 0, 2, 3),
                [self.crop_size, self.crop_size],
                antialias=True,
            ).permute(1, 0, 2, 3)

        if pose is None or not isinstance(pose, torch.Tensor) or pose.shape != (self.num_frames, POSE_FEATURES):
            pose = derive_motion_features_from_rgb(video)
        else:
            pose = pose.float()

        if emb is not None:
            emb = emb.float()
        elif path.stem in self._emb_cache:
            emb = self._emb_cache[path.stem].float()

        return video, pose, emb

    def _extract_and_cache_embedding(self, raw_video: torch.Tensor, path: Path) -> torch.Tensor:
        if path.stem in self._emb_cache:
            return self._emb_cache[path.stem].float()

        norm_v = raw_video.float() / 255.0 if raw_video.dtype == torch.uint8 else raw_video.float()
        if norm_v.max() > 1.5:
            norm_v = norm_v / 255.0
        mean = torch.tensor(KINETICS400_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(KINETICS400_STD).view(3, 1, 1, 1)
        norm_v = ((norm_v - mean) / std).unsqueeze(0)

        extractor = get_shared_visual_extractor()
        dev = next(extractor.parameters()).device
        with torch.no_grad():
            emb = extractor(norm_v.to(dev)).squeeze(0).cpu().float()

        self._emb_cache[path.stem] = emb
        try:
            payload = load_torch_file_safe(path, map_location="cpu")
            if isinstance(payload, dict):
                payload["video_embedding"] = emb
                torch.save(payload, path)
        except Exception:
            pass
        return emb

    def __getitem__(self, index: int):
        rec = self.records[index]
        path: Path = rec["path"]
        label = float(rec["label"])

        raw_video, pose_feats, video_emb = self._load_item(path)
        if self.use_cached_embeddings:
            if video_emb is None:
                video_emb = self._extract_and_cache_embedding(raw_video, path)
            video_input = video_emb
        else:
            video_input = self.transform(raw_video)

        if self.pose_noise_std > 0.0:
            pose_feats = (pose_feats + torch.randn_like(pose_feats) * self.pose_noise_std).clamp(0.0, 1.0)

        target = torch.tensor(label, dtype=torch.float32)
        if self.return_meta:
            return video_input, pose_feats, target, rec["play_id"], rec["sample_id"]
        return video_input, pose_feats, target


class PoseTemporalEncoder(nn.Module):
    """
    Temporal Transformer-style biomechanical contact encoder:
      Input : `(B, FRAMES=16, POSE_FEATURES=12)`
      Output: `(B, POSE_HIDDEN_DIM=256)` via learned attention pooling + temporal max pooling.
    """

    def __init__(
        self,
        in_features: int = POSE_FEATURES,
        embed_dim: int = POSE_HIDDEN_DIM,
        num_frames: int = FRAMES,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        half_dim = embed_dim // 2  # 128
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, half_dim),
            nn.LayerNorm(half_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, half_dim) * 0.02)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=half_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(half_dim)
        self.ffn = nn.Sequential(
            nn.Linear(half_dim, half_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(half_dim * 2, half_dim),
        )
        self.norm2 = nn.LayerNorm(half_dim)
        self.pool_weights = nn.Linear(half_dim, 1)

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        T = pose_features.size(1)
        x = self.input_proj(pose_features)
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T, :]
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        # Learned temporal attention pooling + temporal max pooling
        weights = torch.softmax(self.pool_weights(x), dim=1)  # (B, T, 1)
        attn_pooled = (x * weights).sum(dim=1)                # (B, 128)
        max_pooled = x.max(dim=1).values                      # (B, 128)
        return torch.cat([attn_pooled, max_pooled], dim=-1)   # (B, 256)


class BinaryFoulClassifier(nn.Module):
    """
    Pure Binary Foul vs. No Foul Dual-Stream Classifier:
      - `video_backbone`: Kinetics-400 `R(2+1)D-18` -> `512-D`
      - `pose_network`: `PoseTemporalEncoder` -> `256-D`
      - `fusion_head`: `768-D -> 384-D -> 128-D -> 1` single binary foul logit.
    """

    def __init__(
        self,
        pretrained: bool = True,
        backbone_mode: str = "partial",
        dropout_p: float = 0.35,
        modality_dropout: float = 0.10,
        pose_hidden_dim: int = POSE_HIDDEN_DIM,
        classifier_hidden_dim: int = CLASSIFIER_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        try:
            self.video_backbone = r2plus1d_18(weights=weights)
        except Exception as exc:
            logger.warning("Pretrained Kinetics-400 weights unavailable (%s); using random init.", exc)
            self.video_backbone = r2plus1d_18(weights=None)

        self.video_feature_dim = self.video_backbone.fc.in_features  # 512
        self.video_backbone.fc = nn.Identity()

        self.pose_hidden_dim = pose_hidden_dim
        self.pose_network = PoseTemporalEncoder(
            in_features=POSE_FEATURES,
            embed_dim=pose_hidden_dim,
            num_frames=FRAMES,
        )
        self.combined_feature_dim = self.video_feature_dim + self.pose_hidden_dim  # 768
        self.modality_dropout = float(modality_dropout)

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(self.combined_feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.combined_feature_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p * 0.6),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 3),
            nn.LayerNorm(classifier_hidden_dim // 3),
            nn.GELU(),
            nn.Linear(classifier_hidden_dim // 3, 1),
        )

        self.backbone_mode = backbone_mode.lower()
        self.set_backbone_mode(self.backbone_mode)

    def set_backbone_mode(self, mode: str) -> None:
        """
        Configure trainable layers of the 3D CNN backbone:
          - `'frozen'` : Freeze entire `r2plus1d_18` (fast CPU/MPS mode with cached 512-D embeddings)
          - `'partial'`: Freeze `stem`, `layer1`, `layer2`; fine-tune `layer3`, `layer4` (recommended GPU mode)
          - `'full'`   : Fine-tune all layers end-to-end
        """
        mode = mode.lower()
        self.backbone_mode = mode
        if mode == "frozen":
            for p in self.video_backbone.parameters():
                p.requires_grad = False
        elif mode == "partial":
            for mod in (self.video_backbone.stem, self.video_backbone.layer1, self.video_backbone.layer2):
                for p in mod.parameters():
                    p.requires_grad = False
            for mod in (self.video_backbone.layer3, self.video_backbone.layer4):
                for p in mod.parameters():
                    p.requires_grad = True
        elif mode == "full":
            for p in self.video_backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unsupported backbone_mode='{mode}'. Choose from 'frozen', 'partial', 'full'.")

        for mod in (self.pose_network, self.fusion_head):
            for p in mod.parameters():
                p.requires_grad = True

    def parameter_groups(self, base_lr: float, backbone_lr_scale: float = 0.1) -> List[Dict[str, Any]]:
        """Differential learning rates: smaller LR for pretrained 3D backbone, full LR for pose + fusion head."""
        backbone_params = [p for p in self.video_backbone.parameters() if p.requires_grad]
        head_params = [
            p
            for mod in (self.pose_network, self.fusion_head)
            for p in mod.parameters()
            if p.requires_grad
        ]
        groups: List[Dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_scale})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr})
        return groups

    def trainable_parameter_summary(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }

    def forward(
        self,
        video_input: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_input.dim() == 2 and video_input.size(-1) == self.video_feature_dim:
            vis_emb = video_input
        elif self.backbone_mode == "frozen":
            with torch.no_grad():
                vis_emb = self.video_backbone(video_input)
        else:
            vis_emb = self.video_backbone(video_input)

        if pose_features is None:
            batch_feats = [
                derive_motion_features_from_rgb(video_input[b].detach().cpu())
                for b in range(video_input.size(0))
            ]
            pose_features = torch.stack(batch_feats, dim=0).to(video_input.device)
        else:
            pose_features = pose_features.to(video_input.device)

        pose_emb = self.pose_network(pose_features)

        # Mild modality dropout during training prevents either branch from collapsing
        if self.training and self.modality_dropout > 0.0:
            r = random.random()
            if r < self.modality_dropout * 0.5:
                pose_emb = torch.zeros_like(pose_emb)
            elif r < self.modality_dropout:
                vis_emb = torch.zeros_like(vis_emb)

        combined = torch.cat([vis_emb, pose_emb], dim=-1)
        return self.fusion_head(combined)

    @torch.no_grad()
    def predict_foul(
        self,
        video_input: torch.Tensor,
        pose_features: torch.Tensor,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Pure binary foul prediction (`FOUL` vs. `NO FOUL`) with no heuristic overrides."""
        self.eval()
        logits = self.forward(video_input, pose_features)
        probs = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
        mean_prob = float(np.mean(probs))
        is_foul = bool(mean_prob >= threshold)
        return {
            "is_foul": is_foul,
            "verdict": "FOUL" if is_foul else "NO FOUL",
            "foul_prob": mean_prob,
            "angle_probs": [float(p) for p in np.atleast_1d(probs)],
            "threshold": float(threshold),
        }


# Clean alias
FoulClassifier3D = BinaryFoulClassifier


if __name__ == "__main__":
    model = BinaryFoulClassifier(pretrained=False, backbone_mode="partial")
    summary = model.trainable_parameter_summary()
    dummy_v = torch.randn(2, CHANNELS, FRAMES, HEIGHT, WIDTH)
    dummy_p = torch.rand(2, FRAMES, POSE_FEATURES)
    logits = model(dummy_v, dummy_p)
    print("BinaryFoulClassifier (NBA Replay Archive — Foul vs. No Foul):")
    print(f"  Video Input : (B, {CHANNELS}, {FRAMES}, {HEIGHT}, {WIDTH}) -> {VIDEO_FEATURE_DIM}-D")
    print(f"  Pose Input  : (B, {FRAMES}, {POSE_FEATURES}) -> {POSE_HIDDEN_DIM}-D")
    print(f"  Fusion Head : {COMBINED_FEATURE_DIM}-D -> {CLASSIFIER_HIDDEN_DIM}-D -> {CLASSIFIER_HIDDEN_DIM // 3}-D -> 1")
    print(f"  Output Shape: {tuple(logits.shape)}")
    print(
        f"  Parameters  : Total={summary['total_params']:,} | "
        f"Trainable={summary['trainable_params']:,} | Frozen={summary['frozen_params']:,}"
    )
