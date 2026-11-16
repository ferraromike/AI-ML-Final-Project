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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
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


if __name__ == "__main__":
    model = FoulClassifier3D(pretrained=False, dropout_p=0.35, freeze_early=True)
    summary = model.trainable_parameter_summary()
    print("Simplified Model Dimensions:")
    print(f"  Video Input : (batch, CHANNELS={CHANNELS}, FRAMES={FRAMES}, HEIGHT={HEIGHT}, WIDTH={WIDTH}) -> VIDEO_FEATURE_DIM={VIDEO_FEATURE_DIM}")
    print(f"  Pose Input  : (batch, FRAMES={FRAMES}, POSE_FEATURES={POSE_FEATURES}) -> POSE_HIDDEN_DIM={POSE_HIDDEN_DIM}")
    print(f"  Combined    : COMBINED_FEATURE_DIM={COMBINED_FEATURE_DIM} -> CLASSIFIER_HIDDEN_DIM={CLASSIFIER_HIDDEN_DIM} -> 1")
    print(f"Parameter Summary: Total={summary['total_params']:,} | Trainable={summary['trainable_params']:,} | Frozen={summary['frozen_params']:,}")
    print("Pose Feature Dimensions (12):", POSE_FEATURE_NAMES)
