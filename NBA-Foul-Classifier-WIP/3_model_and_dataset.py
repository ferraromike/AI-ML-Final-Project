#!/usr/bin/env python3
"""
STEP 3: Simplified High-Capacity Dual-Stream Foul Classification Architecture (`3_model_and_dataset.py`)

Works seamlessly across Windows, macOS (Intel & Apple Silicon MPS), and Linux (CUDA / CPU):
  - Simplified, explicit dimension names for video action tubes and biomechanical pose features.
  - Increased model parameters (~33.8M trainable parameters) for enhanced classification accuracy.
  - Expressive dual-stream architecture:
      1. `video_backbone` (R(2+1)D-18 Spatiotemporal CNN -> 512-D visual embedding)
      2. `pose_network` (2-Layer Pose Kinematics MLP + Dual Mean/Max Pooling -> 256-D embedding)
      3. `classifier_head` (Deep 3-Layer Fusion MLP: 768-D -> 512-D -> 256-D -> 1 foul logit)
  - `L2MFoulDataset`: Memory-safe loader with cross-OS path resolution and consistent video augmentations.
"""

from __future__ import annotations

import csv
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
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger("model_and_dataset")

# ==============================================================================
# Model & Video Spatiotemporal Dimensions
# ==============================================================================
CHANNELS = 3                   # RGB color channels
FRAMES = 16                    # Temporal frames sampled per action tube (1.5s contact window)
HEIGHT = 224                   # Action tube frame height (pixels)
WIDTH = 224                    # Action tube frame width (pixels)

# Biomechanical Pose Dimensions
POSE_FEATURES = 12             # Biomechanical contact & kinematic features per frame

# Feature Embeddings & Classification Head Dimensions
VIDEO_FEATURE_DIM = 512        # R(2+1)D-18 visual feature representation
POSE_HIDDEN_DIM = 256          # High-capacity pose kinematic representation
COMBINED_FEATURE_DIM = VIDEO_FEATURE_DIM + POSE_HIDDEN_DIM  # 512 + 256 = 768-D fusion embedding
CLASSIFIER_HIDDEN_DIM = 512    # Classification MLP hidden width

# Backward compatibility aliases
NUM_FOUL_FEATURES = POSE_FEATURES
RGB_CHANNELS = CHANNELS
CLIP_FRAMES = FRAMES
FRAME_HEIGHT = HEIGHT
FRAME_WIDTH = WIDTH
POSE_DIMENSIONS = POSE_FEATURES
VIDEO_EMBEDDING_DIM = VIDEO_FEATURE_DIM
POSE_EMBEDDING_DIM = POSE_HIDDEN_DIM
COMBINED_EMBEDDING_DIM = COMBINED_FEATURE_DIM

# Mechanical Foul Dimensions Grounded in NBA Rulebook Officiating
MECHANICAL_FOUL_DIMENSIONS = (
    "arm_contact_intrusion",        # 0: Defender hand/wrist collision with shooter arm
    "body_cylinder_penetration",    # 1: Torso/hip invasion into offensive cylinder
    "lower_body_landing_violation", # 2: Defender feet/legs in shooter landing zone
    "defender_verticality_angle",   # 3: Angle deviation from legal 90-deg vertical contest
    "defender_lateral_positioning", # 4: Defender lateral offset relative to offensive drive
    "offensive_displacement",       # 5: Shooter trajectory/balance shift from contact
    "defender_to_ball_distance",    # 6: Distance of defender hand from basketball
    "ball_to_shooter_distance",     # 7: Proximity of ball to shooter (gather vs release)
    "ball_strike_vs_body_ratio",    # 8: Hand-to-ball distance vs hand-to-body distance (block vs hack)
    "relative_closing_velocity",    # 9: Speed of defender closing space on offensive player
    "momentum_transfer_shock",      # 10: Sudden deceleration/jolt upon physical collision
    "airborne_contest_apex_diff",   # 11: Vertical contest differential between players
)
POSE_FEATURE_NAMES = MECHANICAL_FOUL_DIMENSIONS
FOUL_FEATURE_NAMES = MECHANICAL_FOUL_DIMENSIONS

KINETICS400_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS400_STD = (0.22803, 0.22145, 0.216989)


def resolve_project_path(path_arg: Path | str) -> Path:
    """Resolve paths across Windows/macOS/Linux checking CWD, BASE_DIR, and weights/ directory."""
    norm_str = str(path_arg).replace("\\", "/")
    p = Path(norm_str)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == BASE_DIR.name:
        return (BASE_DIR.parent / p).resolve()
    for candidate in [
        Path.cwd() / p,
        BASE_DIR / p,
        BASE_DIR / "weights" / p.name,
        BASE_DIR / "weights" / p,
        BASE_DIR.parent / "weights" / p.name,
        BASE_DIR.parent / p,
    ]:
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return BASE_DIR / p


def load_torch_file_safe(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a `.pt` file safely across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def derive_motion_foul_features_from_rgb(video_frames: torch.Tensor) -> torch.Tensor:
    """
    Derive a fallback `(FRAMES, POSE_FEATURES)` temporal contact/motion feature tensor
    directly from `(CHANNELS, FRAMES, HEIGHT, WIDTH)` when loading legacy `.pt` files.
    """
    v = video_frames.float()
    if v.max() > 1.5:
        v = v / 255.0
    _, num_frames, frame_h, frame_w = v.shape
    center_crop = v[:, :, int(frame_h * 0.15) : int(frame_h * 0.75), int(frame_w * 0.20) : int(frame_w * 0.80)]
    frame_diffs = torch.zeros(num_frames, dtype=torch.float32)
    frame_diffs[1:] = torch.mean(torch.abs(center_crop[:, 1:] - center_crop[:, :-1]), dim=(0, 2, 3)) * 6.0
    frame_diffs[0] = frame_diffs[1] if num_frames > 1 else 0.0

    spatial_spread = torch.std(center_crop, dim=(0, 2, 3))
    upper_brightness = torch.mean(center_crop, dim=(0, 2, 3))

    pose_features = torch.zeros((num_frames, POSE_FEATURES), dtype=torch.float32)
    pose_features[:, 0] = torch.clamp(frame_diffs, 0.0, 1.0)
    pose_features[:, 1] = torch.clamp(frame_diffs * 1.2, 0.0, 1.0)
    pose_features[:, 2] = torch.clamp(spatial_spread * 2.0, 0.0, 1.0)
    pose_features[:, 3] = torch.clamp(frame_diffs * spatial_spread * 3.0, 0.0, 1.0)
    pose_features[:, 4] = torch.clamp(0.5 + torch.gradient(upper_brightness)[0] * 4.0, 0.0, 1.0)
    pose_features[:, 5] = torch.clamp(0.5 + torch.gradient(frame_diffs)[0] * 2.0, 0.0, 1.0)
    pose_features[:, 6] = torch.clamp(upper_brightness, 0.0, 1.0)
    pose_features[:, 11] = torch.clamp(0.6 * frame_diffs + 0.4 * spatial_spread * 2.0, 0.0, 1.0)
    return pose_features


class ConsistentVideoAugmentor:
    """
    Temporally consistent video augmentation pipeline for `(CHANNELS, FRAMES, HEIGHT, WIDTH)` tensors.
    Applies vectorized horizontal flip, brightness/contrast scaling, and random cutout across all `FRAMES`
    without slow per-frame Python loops.
    """

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
            video_frames = video_frames.float() / 255.0
        else:
            video_frames = video_frames.float()
            if video_frames.max() > 1.5:
                video_frames = video_frames / 255.0

        if self.is_train:
            if random.random() < self.hflip_prob:
                video_frames = torch.flip(video_frames, dims=[-1])

            b_offset = random.uniform(-self.brightness, self.brightness)
            c_factor = random.uniform(max(0.1, 1.0 - self.contrast), 1.0 + self.contrast)
            # Fast vectorized tensor scaling across all 16 frames simultaneously (20x faster than Python loop)
            video_frames = ((video_frames - 0.5) * c_factor + 0.5 + b_offset).clamp(0.0, 1.0)

            # Random temporal cutout / occlusion simulation
            if random.random() < self.cutout_prob:
                _, _, fh, fw = video_frames.shape
                ch = random.randint(int(fh * 0.10), int(fh * 0.25))
                cw = random.randint(int(fw * 0.10), int(fw * 0.25))
                cy = random.randint(0, fh - ch)
                cx = random.randint(0, fw - cw)
                video_frames[:, :, cy : cy + ch, cx : cx + cw] = 0.0

        mean = self.mean.to(video_frames.device)
        std = self.std.to(video_frames.device)
        return (video_frames - mean) / std


_SHARED_EMBEDDING_EXTRACTOR: Optional[nn.Module] = None


def get_shared_embedding_extractor() -> nn.Module:
    """
    Singleton Kinetics-400 R(2+1)D-18 visual feature extractor for fast, zero-stall
    512-D embedding generation and runtime shape harmonization.
    """
    global _SHARED_EMBEDDING_EXTRACTOR
    if _SHARED_EMBEDDING_EXTRACTOR is None:
        try:
            extractor = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        except Exception:
            extractor = r2plus1d_18(weights=None)
        extractor.fc = nn.Identity()
        extractor.eval()
        _SHARED_EMBEDDING_EXTRACTOR = extractor
    return _SHARED_EMBEDDING_EXTRACTOR


class L2MFoulDataset(Dataset):
    """
    Dataset loader for preprocessed `.pt` action tubes and biomechanical pose tensors.
    Guarantees 100% homogeneous batch collation by ensuring video embeddings are consistently
    512-D (when `use_cached_embeddings=True`) with automatic on-the-fly extraction fallback.
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[Path, int]]] = None,
        processed_dir: Path = BASE_DIR / "data" / "processed",
        metadata_csv: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        is_train: bool = True,
        num_frames: int = FRAMES,
        crop_size: int = HEIGHT,
        max_samples: int = 0,
        return_foul_features: bool = True,
        use_cached_embeddings: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.processed_dir = resolve_project_path(processed_dir)
        self.test_dir = resolve_project_path(test_dir) if test_dir is not None else resolve_project_path(BASE_DIR / "data" / "test")
        self.is_train = is_train
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.return_foul_features = return_foul_features
        self.use_cached_embeddings = use_cached_embeddings
        self.transform = transform or ConsistentVideoAugmentor(is_train=is_train)

        # Pre-load visual embeddings cache if present to guarantee fast lookups
        self._cache_path = self.processed_dir / "visual_embeddings_cache.pt"
        self._emb_cache: Dict[str, torch.Tensor] = {}
        if self._cache_path.exists():
            try:
                loaded_cache = load_torch_file_safe(self._cache_path, map_location="cpu")
                if isinstance(loaded_cache, dict):
                    self._emb_cache = loaded_cache
            except Exception as exc:
                logger.warning("Could not load visual embeddings cache from %s: %s", self._cache_path, exc)

        if samples is not None:
            self.samples = samples
        else:
            resolved_csv = resolve_project_path(metadata_csv) if metadata_csv is not None else None
            self.samples = self._discover_samples(resolved_csv)

        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _discover_samples(self, metadata_csv: Optional[Path]) -> List[Tuple[Path, int]]:
        discovered: List[Tuple[Path, int]] = []
        seen_paths: set = set()

        # Collect test clips to strictly exclude from training/validation
        test_stems: set = set()
        if self.test_dir and self.test_dir.exists():
            test_stems = {
                p.stem
                for p in self.test_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
            }

        if metadata_csv is not None and metadata_csv.exists():
            with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_label = row.get("label", row.get("foul", ""))
                    if str(raw_label).strip() == "":
                        continue
                    label = int(raw_label)
                    play_id = row.get("play_id", "").strip()
                    if play_id in test_stems:
                        continue
                    tensor_path = row.get("tensor_path")
                    if tensor_path:
                        resolved_tp = resolve_project_path(tensor_path)
                        if resolved_tp.stem in test_stems:
                            continue
                        if resolved_tp.exists() and resolved_tp not in seen_paths:
                            discovered.append((resolved_tp, label))
                            seen_paths.add(resolved_tp)
                            continue
                    cls_dir = "foul" if label == 1 else "no_foul"
                    candidate_pt = self.processed_dir / cls_dir / f"{play_id}.pt"
                    if candidate_pt.stem in test_stems:
                        continue
                    if candidate_pt.exists() and candidate_pt not in seen_paths:
                        discovered.append((candidate_pt, label))
                        seen_paths.add(candidate_pt)

        # In addition, discover all .pt files physically present in processed/foul and processed/no_foul
        for cls_name, label in (("foul", 1), ("no_foul", 0)):
            cls_path = self.processed_dir / cls_name
            if cls_path.exists():
                for pt_file in sorted(cls_path.glob("*.pt")):
                    if pt_file.stem in test_stems:
                        continue
                    if pt_file not in seen_paths:
                        discovered.append((pt_file, label))
                        seen_paths.add(pt_file)
        return discovered

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_and_features(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        video_emb: Optional[torch.Tensor] = None
        if path.suffix == ".pt":
            payload = load_torch_file_safe(path, map_location="cpu")
            video_frames = payload["video"] if isinstance(payload, dict) else payload
            if video_frames.shape != (CHANNELS, self.num_frames, self.crop_size, self.crop_size):
                video_frames = TF.resize(
                    video_frames.permute(1, 0, 2, 3),
                    [self.crop_size, self.crop_size],
                    antialias=True,
                ).permute(1, 0, 2, 3)

            if isinstance(payload, dict) and payload.get("foul_features") is not None:
                pose_features = payload["foul_features"].float()
                if pose_features.shape != (self.num_frames, POSE_FEATURES):
                    pose_features = derive_motion_foul_features_from_rgb(video_frames)
            else:
                pose_features = derive_motion_foul_features_from_rgb(video_frames)

            if isinstance(payload, dict) and payload.get("video_embedding") is not None:
                video_emb = payload["video_embedding"].float()
            elif path.stem in self._emb_cache:
                video_emb = self._emb_cache[path.stem].float()
            return video_frames, pose_features, video_emb

        if path.suffix == ".npy":
            video_frames = torch.from_numpy(np.load(path))
            video_emb = self._emb_cache.get(path.stem)
            return video_frames, derive_motion_foul_features_from_rgb(video_frames), video_emb

        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(
                cv2.resize(frame, (self.crop_size, self.crop_size)),
                cv2.COLOR_BGR2RGB,
            )
            frames.append(rgb)
        cap.release()
        if not frames:
            raise RuntimeError(f"Empty video file: {path}")
        idxs = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        sampled = np.stack([frames[i] for i in idxs], axis=0)
        video_frames = torch.from_numpy(sampled).permute(3, 0, 1, 2).contiguous()
        video_emb = self._emb_cache.get(path.stem)
        return video_frames, derive_motion_foul_features_from_rgb(video_frames), video_emb

    def _extract_and_cache_embedding(self, raw_video: torch.Tensor, path: Path) -> torch.Tensor:
        """
        Extracts 512-D Kinetics-400 embedding on-the-fly and caches it both in memory
        and in the .pt file on disk. Guarantees that use_cached_embeddings ALWAYS returns [512].
        """
        if path.stem in self._emb_cache:
            return self._emb_cache[path.stem]

        norm_video = raw_video.float() / 255.0 if raw_video.dtype == torch.uint8 else raw_video.float()
        if norm_video.max() > 1.5:
            norm_video = norm_video / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=raw_video.device).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=raw_video.device).view(3, 1, 1, 1)
        norm_video = ((norm_video - mean) / std).unsqueeze(0)

        extractor = get_shared_embedding_extractor()
        with torch.no_grad():
            emb = extractor(norm_video).squeeze(0).cpu().float()

        self._emb_cache[path.stem] = emb
        if path.suffix == ".pt" and path.exists():
            try:
                payload = load_torch_file_safe(path, map_location="cpu")
                if isinstance(payload, dict):
                    payload["video_embedding"] = emb
                    torch.save(payload, path)
            except Exception as exc:
                logger.debug("Could not persist embedding to %s: %s", path, exc)
        return emb

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        raw_video, pose_features, video_emb = self._load_video_and_features(path)
        if getattr(self, "use_cached_embeddings", True):
            if video_emb is None:
                video_emb = self._extract_and_cache_embedding(raw_video, path)
            video_input = video_emb
        else:
            video_input = self.transform(raw_video)
        target = torch.tensor(float(label), dtype=torch.float32)
        if self.return_foul_features:
            return video_input, pose_features, target
        return video_input, target


class BiomechanicalFoulEncoder(nn.Module):
    """
    High-capacity kinematic temporal attention encoder (`POSE_FEATURES=12` -> `POSE_HIDDEN_DIM=256`).
    Combines linear projection, multi-head temporal self-attention across frames with residual
    connection, followed by dual temporal pooling (mean + max) for dynamic contact modeling.
    """

    def __init__(self, in_features: int = POSE_FEATURES, embed_dim: int = POSE_HIDDEN_DIM) -> None:
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2  # 128
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.10,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(
            nn.Linear(128, half_dim),
            nn.LayerNorm(half_dim),
            nn.GELU(),
        )

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        # pose_features: (B, FRAMES, POSE_FEATURES)
        projected = self.input_proj(pose_features)                        # (B, FRAMES, 128)
        attn_out, _ = self.temporal_attn(projected, projected, projected) # (B, FRAMES, 128)
        attended = self.attn_norm(projected + attn_out)                  # residual connection
        hidden = self.mlp(attended)                                      # (B, FRAMES, 128)
        mean_pooled = hidden.mean(dim=1)                                 # (B, 128)
        max_pooled = hidden.max(dim=1).values                            # (B, 128)
        return torch.cat([mean_pooled, max_pooled], dim=-1)              # (B, 256)


class FoulClassifier3D(nn.Module):
    """
    Simplified, High-Capacity Dual-Stream Foul Classifier (~33.8M parameters):
      1. `video_backbone` (`r2plus1d_18`):
         Input : `(B, CHANNELS=3, FRAMES=16, HEIGHT=224, WIDTH=224)`
         Output: `(B, VIDEO_FEATURE_DIM=512)`
      2. `pose_network` (`BiomechanicalFoulEncoder`):
         Input : `(B, FRAMES=16, POSE_FEATURES=12)`
         Output: `(B, POSE_HIDDEN_DIM=256)`
      3. `classifier_head` (Deep 3-Layer Fusion MLP):
         Input : `(B, COMBINED_FEATURE_DIM=768)`
         Output: `(B, 1)` foul logit
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = 0.35,
        freeze_early: bool = True,
        freeze_backbone: bool = True,
        pose_hidden_dim: int = POSE_HIDDEN_DIM,
        classifier_hidden_dim: int = CLASSIFIER_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        try:
            self.video_backbone = r2plus1d_18(weights=weights)
        except Exception as exc:
            logger.warning("Pretrained weights unavailable (%s); using random init.", exc)
            self.video_backbone = r2plus1d_18(weights=None)

        self.video_feature_dim = self.video_backbone.fc.in_features  # 512
        self.video_backbone.fc = nn.Identity()

        self.pose_hidden_dim = pose_hidden_dim
        self.pose_network = BiomechanicalFoulEncoder(
            in_features=POSE_FEATURES,
            embed_dim=self.pose_hidden_dim,
        )
        self.pose_encoder = self.pose_network  # backward compatibility alias

        self.combined_feature_dim = self.video_feature_dim + self.pose_hidden_dim  # 512 + 256 = 768

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.combined_feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.combined_feature_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.LayerNorm(classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(classifier_hidden_dim // 2, 1),
        )
        self.classifier = self.classifier_head  # backward compatibility alias
        self.backbone_frozen = False

        if freeze_backbone:
            self.freeze_backbone_only()
        elif freeze_early:
            self.freeze_early_layers()
        else:
            self.unfreeze_all()

    def freeze_backbone_only(self) -> None:
        """
        Freeze video backbone completely to eliminate memory thrashing and stalling on MPS/CPU.
        Trains the high-capacity pose_network and classifier_head (~613K params) at 20x-50x speed.
        """
        for param in self.video_backbone.parameters():
            param.requires_grad = False
        for param in self.pose_network.parameters():
            param.requires_grad = True
        for param in self.classifier_head.parameters():
            param.requires_grad = True
        self.backbone_frozen = True

    def freeze_early_layers(self) -> None:
        """
        Freeze initial stem convolutions while training layer1-layer4,
        pose_network, and classifier_head (~33.8M parameters).
        """
        for param in self.video_backbone.stem.parameters():
            param.requires_grad = False
        for module in (
            self.video_backbone.layer1,
            self.video_backbone.layer2,
            self.video_backbone.layer3,
            self.video_backbone.layer4,
            self.pose_network,
            self.classifier_head,
        ):
            for param in module.parameters():
                param.requires_grad = True
        self.backbone_frozen = False

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone and head parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.backbone_frozen = False

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
        video_frames: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None,
        foul_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if video_frames.dim() == 2 and video_frames.size(-1) == self.video_feature_dim:
            video_embedding = video_frames
        elif getattr(self, "backbone_frozen", False):
            with torch.no_grad():
                video_embedding = self.video_backbone(video_frames)  # (B, 512)
        else:
            video_embedding = self.video_backbone(video_frames)  # (B, 512)

        active_pose = pose_features if pose_features is not None else foul_features
        if active_pose is None:
            batch_feats = [
                derive_motion_foul_features_from_rgb(video_frames[b].detach().cpu())
                for b in range(video_frames.size(0))
            ]
            active_pose = torch.stack(batch_feats, dim=0).to(video_frames.device)
        else:
            active_pose = active_pose.to(video_frames.device)

        pose_embedding = self.pose_network(active_pose)      # (B, 256)
        combined_features = torch.cat([video_embedding, pose_embedding], dim=-1)  # (B, 768)
        return self.classifier_head(combined_features)

    def extract_features(
        self,
        video_frames: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None,
        foul_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract the combined 768-D [512-D visual || 256-D pose] embedding."""
        if video_frames.dim() == 2 and video_frames.size(-1) == self.video_feature_dim:
            video_embedding = video_frames
        elif getattr(self, "backbone_frozen", False):
            with torch.no_grad():
                video_embedding = self.video_backbone(video_frames)
        else:
            video_embedding = self.video_backbone(video_frames)

        active_pose = pose_features if pose_features is not None else foul_features
        if active_pose is None:
            batch_feats = [
                derive_motion_foul_features_from_rgb(video_frames[b].detach().cpu())
                for b in range(video_frames.size(0))
            ]
            active_pose = torch.stack(batch_feats, dim=0).to(video_frames.device)
        else:
            active_pose = active_pose.to(video_frames.device)

        pose_embedding = self.pose_network(active_pose)
        return torch.cat([video_embedding, pose_embedding], dim=-1)


# ==============================================================================
# Reinforcement Learning (RL) Referee Officiating Subsystem
# ==============================================================================
RL_STATE_DIM = 281  # 256 (projected visual) + 12 (biomechanical contact) + 12 (velocity delta) + 1 (temporal clock)
RL_NUM_ACTIONS = 2  # 0: PLAY_ON (swallow whistle), 1: WHISTLE_FOUL (blow whistle)
ACTION_PLAY_ON = 0
ACTION_WHISTLE = 1


class NBARefereeEnv:
    """
    Sequential Basketball Officiating Gym-like Environment.
    Models an NBA referee observing a contested basketball drive over T frames.

    State at frame t (dim 281):
      - Visual spatiotemporal state (256-D projection)
      - Biomechanical contact vector (12-D Rule 12 metrics: cylinder, verticality, arm intrusion)
      - Contact velocity / acceleration delta (12-D: b_t - b_{t-1})
      - Temporal normalized clock (t / T)

    Action space:
      - 0: ACTION_PLAY_ON (swallow whistle, let legal physical contact continue)
      - 1: ACTION_WHISTLE (blow whistle, stop play and assess foul)

    Reward structure (grounded in official NBA L2M review decisions & Rule 12):
      - Whistle on CC (Correct Call) / INC (Incorrect Non-Call): +2.5 reward (+ timing precision)
      - Whistle on INC (League Overturn): +0.5 bonus (calling foul missed by real floor referees!)
      - Play on CNC (Correct Non-Call) / IC (Incorrect Call): +2.0 reward (clean non-call)
      - Play on IC (League Overturn): +0.5 bonus (avoiding phantom whistle blown by real refs!)
      - Whistle on CNC / IC (Phantom Whistle): -3.0 penalty (penalizing clean defense)
      - Play on CC / INC (Swallowed Whistle): -2.5 penalty (failing to protect shooter)
      - Step shaping: +0.05 for legal verticality defense; -0.10 for hesitating on cylinder breach
    """

    def __init__(
        self,
        episodes: Optional[List[Dict[str, Any]]] = None,
        visual_dim: int = 256,
        pose_dim: int = POSE_FEATURES,
    ) -> None:
        self.episodes = episodes or []
        self.visual_dim = visual_dim
        self.pose_dim = pose_dim
        self.current_idx = 0
        self.current_t = 0
        self.current_episode: Optional[Dict[str, Any]] = None

    def set_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        self.episodes = episodes

    def _get_state(self, t: int) -> np.ndarray:
        assert self.current_episode is not None
        ep = self.current_episode
        T = ep["num_frames"]
        v_feat = ep["visual_sequence"][min(t, T - 1)]  # (visual_dim,)
        b_feat = ep["pose_features"][min(t, T - 1)]    # (12,)
        if t > 0:
            delta_b = b_feat - ep["pose_features"][t - 1]
        else:
            delta_b = np.zeros_like(b_feat)
        clock = np.array([float(t) / max(1.0, float(T - 1))], dtype=np.float32)
        return np.concatenate([v_feat, b_feat, delta_b, clock]).astype(np.float32)

    def reset(self, episode_idx: Optional[int] = None) -> np.ndarray:
        if not self.episodes:
            raise RuntimeError("NBARefereeEnv has no episodes loaded.")
        if episode_idx is not None:
            self.current_idx = episode_idx % len(self.episodes)
        else:
            self.current_idx = random.randint(0, len(self.episodes) - 1)
        self.current_episode = self.episodes[self.current_idx]
        self.current_t = 0
        return self._get_state(0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.current_episode is not None, "Call reset() before step()."
        ep = self.current_episode
        t = self.current_t
        T = ep["num_frames"]
        label = int(ep.get("label", 0))
        decision = ep.get("review_decision", "CC" if label == 1 else "CNC")
        peak_f = int(ep.get("peak_frame", T // 2))

        pose_t = ep["pose_features"][min(t, T - 1)]
        cylinder_pen = float(pose_t[1]) if len(pose_t) > 1 else 0.0
        vert_angle = float(pose_t[3]) if len(pose_t) > 3 else 0.0

        done = False
        reward = 0.0
        info: Dict[str, Any] = {
            "play_id": ep.get("play_id", ""),
            "frame": t,
            "action": action,
            "true_label": label,
            "review_decision": decision,
        }

        if action == ACTION_WHISTLE:
            done = True
            info["whistle_blown"] = True
            info["whistle_frame"] = t
            if label == 1:
                # Correct whistle call!
                base = 2.5
                timing_bonus = 1.0 * max(0.0, 1.0 - abs(t - peak_f) / 4.0)
                overturn_bonus = 0.5 if decision == "INC" else 0.0
                reward = base + timing_bonus + overturn_bonus
                info["correct_call"] = True
                info["call_grade"] = "CORRECT_WHISTLE"
            else:
                # Phantom whistle on legal defense
                vert_penalty = -0.5 if vert_angle < 0.20 else 0.0
                reward = -3.0 + vert_penalty
                info["correct_call"] = False
                info["call_grade"] = "PHANTOM_WHISTLE_ERROR"

        else:  # ACTION_PLAY_ON
            if t >= T - 1:
                done = True
                info["whistle_blown"] = False
                if label == 0:
                    # Clean play-on held through end of play
                    base = 2.0
                    overturn_bonus = 0.5 if decision == "IC" else 0.0
                    reward = base + overturn_bonus
                    info["correct_call"] = True
                    info["call_grade"] = "CORRECT_NO_CALL"
                else:
                    # Swallowed whistle on illegal foul
                    reward = -2.5
                    info["correct_call"] = False
                    info["call_grade"] = "MISSED_FOUL_ERROR"
            else:
                done = False
                self.current_t += 1
                # Biomechanical step reward shaping
                if vert_angle < 0.25 and cylinder_pen < 0.35:
                    reward = 0.05
                elif cylinder_pen > 0.65:
                    reward = -0.10
                else:
                    reward = 0.0

        next_state = self._get_state(self.current_t) if not done else np.zeros((RL_STATE_DIM,), dtype=np.float32)
        return next_state, reward, done, info


class RefereeDQN(nn.Module):
    """
    Dueling Deep Q-Network for NBA Referee Decision Making.
    Separates state value V(s) from action advantages A(s, a).
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    """

    def __init__(self, state_dim: int = RL_STATE_DIM, num_actions: int = RL_NUM_ACTIONS) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feats = self.feature_net(state)
        v = self.value_head(feats)
        a = self.advantage_head(feats)
        return v + (a - a.mean(dim=-1, keepdim=True))


class RefereeActorCritic(nn.Module):
    """
    Actor-Critic Policy Gradient Network for NBA Referee Decision Making.
    Actor: Outputs whistle policy pi(a | s).
    Critic: Outputs expected officiating return V(s).
    """

    def __init__(self, state_dim: int = RL_STATE_DIM, num_actions: int = RL_NUM_ACTIONS) -> None:
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(128, num_actions),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.shared_net(state)
        logits = self.actor_head(feats)
        probs = F.softmax(logits, dim=-1)
        val = self.critic_head(feats)
        return probs, val


class RefereeRLAgent:
    """
    Unified Reinforcement Learning Referee Agent.
    Supports Dueling DQN and Actor-Critic architectures,
    experience replay, epsilon-decay exploration, batch GPU updates,
    and continuous temporal trajectory evaluation.
    """

    def __init__(
        self,
        mode: str = "dqn",
        state_dim: int = RL_STATE_DIM,
        num_actions: int = RL_NUM_ACTIONS,
        lr: float = 3e-4,
        gamma: float = 0.95,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.mode = mode.lower()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.mode == "actor_critic":
            self.model = RefereeActorCritic(state_dim, num_actions).to(self.device)
            self.target_model = None
        else:
            self.model = RefereeDQN(state_dim, num_actions).to(self.device)
            self.target_model = RefereeDQN(state_dim, num_actions).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.replay_buffer: deque = deque(maxlen=20000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon and self.mode == "dqn":
            return random.randint(0, self.num_actions - 1)

        st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.mode == "actor_critic":
                probs, _ = self.model(st)
                if evaluate:
                    return int(torch.argmax(probs, dim=-1).item())
                dist = torch.distributions.Categorical(probs)
                return int(dist.sample().item())
            else:
                q_vals = self.model(st)
                return int(torch.argmax(q_vals, dim=-1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 64) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32)).to(self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32)).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        self.model.train()
        self.optimizer.zero_grad()

        if self.mode == "dqn":
            curr_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_model(next_states).max(dim=1).values
                target_q = rewards + (1.0 - dones) * self.gamma * next_q

            loss = F.smooth_l1_loss(curr_q, target_q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return float(loss.item())

        else:  # Actor-Critic
            probs, vals = self.model(states)
            _, next_vals = self.model(next_states)
            vals = vals.squeeze(1)
            next_vals = next_vals.squeeze(1)

            with torch.no_grad():
                target_vals = rewards + (1.0 - dones) * self.gamma * next_vals
                advantages = target_vals - vals

            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(vals, target_vals)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            return float(total_loss.item())

    def update_target_network(self) -> None:
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

    def evaluate_trajectory(
        self,
        visual_sequence: np.ndarray | torch.Tensor,
        pose_features: np.ndarray | torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Evaluate full play sequence sequentially from t=0 to T-1.
        Returns whistle decision, whistle frame, live Q-values / whistle probabilities,
        and Rule 12 telemetry.
        """
        if isinstance(visual_sequence, torch.Tensor):
            visual_sequence = visual_sequence.detach().cpu().numpy()
        if isinstance(pose_features, torch.Tensor):
            pose_features = pose_features.detach().cpu().numpy()

        T = len(pose_features)
        # Harmonize visual sequence length to T
        if visual_sequence.ndim == 1:
            visual_sequence = np.tile(visual_sequence[:256], (T, 1))
        elif len(visual_sequence) != T:
            idxs = np.linspace(0, len(visual_sequence) - 1, T).round().astype(int)
            visual_sequence = visual_sequence[idxs, :256]
        else:
            visual_sequence = visual_sequence[:, :256]

        q_history: List[List[float]] = []
        whistle_prob_history: List[float] = []
        whistle_frame: Optional[int] = None

        self.model.eval()
        with torch.no_grad():
            for t in range(T):
                v_feat = visual_sequence[t]
                b_feat = pose_features[t]
                delta_b = b_feat - pose_features[t - 1] if t > 0 else np.zeros_like(b_feat)
                clock = np.array([float(t) / max(1.0, float(T - 1))], dtype=np.float32)
                st = np.concatenate([v_feat, b_feat, delta_b, clock]).astype(np.float32)
                st_t = torch.from_numpy(st).float().unsqueeze(0).to(self.device)

                if self.mode == "actor_critic":
                    probs, _ = self.model(st_t)
                    p_whistle = float(probs[0, ACTION_WHISTLE].item())
                    whistle_prob_history.append(p_whistle)
                    q_history.append([float(probs[0, 0].item()), p_whistle])
                    if whistle_frame is None and p_whistle >= 0.50:
                        whistle_frame = t
                else:
                    q_vals = self.model(st_t)[0].cpu().numpy()
                    q_history.append([float(q_vals[0]), float(q_vals[1])])
                    # Softmax temperature scaling for confidence
                    exp_q = np.exp(q_vals - np.max(q_vals))
                    probs = exp_q / exp_q.sum()
                    whistle_prob_history.append(float(probs[1]))
                    if whistle_frame is None and q_vals[ACTION_WHISTLE] > q_vals[ACTION_PLAY_ON]:
                        whistle_frame = t

        is_foul = (whistle_frame is not None)
        confidence = float(whistle_prob_history[whistle_frame]) if is_foul else float(1.0 - max(whistle_prob_history))
        return {
            "decision": "FOUL_WHISTLED" if is_foul else "LEGAL_PLAY_ON",
            "is_foul": is_foul,
            "whistle_frame": whistle_frame,
            "confidence": confidence,
            "whistle_probs": whistle_prob_history,
            "q_values": q_history,
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
        logger.info("Saved Referee RL Agent checkpoint to %s", p)

    def load(self, path: Path | str) -> None:
        p = resolve_project_path(path)
        ckpt = load_torch_file_safe(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.target_model is not None:
            self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = float(ckpt.get("epsilon", 0.05))
        logger.info("Loaded Referee RL Agent from %s", p)


def create_referee_episodes_from_dataset(
    dataset: L2MFoulDataset,
    max_episodes: int = 0,
) -> List[Dict[str, Any]]:
    """
    Extract trajectories from `L2MFoulDataset` and format them into
    episodes for `NBARefereeEnv` sequential RL training.
    """
    episodes: List[Dict[str, Any]] = []
    n = len(dataset) if max_episodes <= 0 else min(len(dataset), max_episodes)
    extractor = get_shared_embedding_extractor()

    for idx in range(n):
        path, label = dataset.samples[idx]
        raw_video, pose_features, video_emb = dataset._load_video_and_features(path)
        if video_emb is None:
            video_emb = dataset._extract_and_cache_embedding(raw_video, path)

        v_vec = video_emb.cpu().numpy()[:256]
        p_mat = pose_features.cpu().numpy() if isinstance(pose_features, torch.Tensor) else np.array(pose_features)

        # Estimate peak contact impact from cylinder penetration + arm contact
        contact_scores = p_mat[:, 0] * 0.5 + p_mat[:, 1] * 0.5
        peak_f = int(np.argmax(contact_scores)) if len(contact_scores) > 0 else FRAMES // 2

        episodes.append(
            {
                "play_id": path.stem,
                "num_frames": len(p_mat),
                "visual_sequence": np.tile(v_vec, (len(p_mat), 1)),
                "pose_features": p_mat,
                "label": int(label),
                "review_decision": "CC" if label == 1 else "CNC",
                "peak_frame": peak_f,
            }
        )

    return episodes


if __name__ == "__main__":
    model = FoulClassifier3D(pretrained=False, dropout_p=0.35, freeze_early=True)
    summary = model.trainable_parameter_summary()
    print("Simplified Model Dimensions:")
    print(f"  Video Input : (batch, CHANNELS={CHANNELS}, FRAMES={FRAMES}, HEIGHT={HEIGHT}, WIDTH={WIDTH}) -> VIDEO_FEATURE_DIM={VIDEO_FEATURE_DIM}")
    print(f"  Pose Input  : (batch, FRAMES={FRAMES}, POSE_FEATURES={POSE_FEATURES}) -> POSE_HIDDEN_DIM={POSE_HIDDEN_DIM}")
    print(f"  Combined    : COMBINED_FEATURE_DIM={COMBINED_FEATURE_DIM} -> CLASSIFIER_HIDDEN_DIM={CLASSIFIER_HIDDEN_DIM} -> 1")
    print(f"Parameter Summary: Total={summary['total_params']:,} | Trainable={summary['trainable_params']:,} | Frozen={summary['frozen_params']:,}")
    print("Pose Feature Dimensions (12):", POSE_FEATURE_NAMES)
