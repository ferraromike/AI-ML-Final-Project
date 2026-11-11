#!/usr/bin/env python3
"""
STEP 3: PyTorch 3D CNN + Biomechanical Foul-Contact Architecture (`3_model_and_dataset.py`)

Implements:
  1. `L2MFoulDataset`: Loads pre-cropped 16-frame 224x224 action tubes `(3, 16, 224, 224)`
     along with 12-D per-frame biomechanical foul-contact features `(16, 12)` (synthesizing
     motion-energy contact proxies automatically for legacy `.pt` files), applying temporally
     consistent video-level augmentations (horizontal flip and color jitter across all 16 frames).
  2. `FoulClassifier3D`: Wraps `torchvision.models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)`
     with `freeze_early_layers()` (`stem`, `layer1`, `layer2` frozen; `layer3`, `layer4` fine-tuned)
     and fuses the 512-D 3D CNN representation with a Temporal Foul-Contact Kinematic Encoder
     (`Conv1d` + temporal attention pooling over the 16-frame pose-contact features) to predict
     binary foul probability.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

KINETICS400_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS400_STD = (0.22803, 0.22145, 0.216989)
NUM_FOUL_FEATURES = 12


def derive_motion_foul_features_from_rgb(video_u8_or_float: torch.Tensor) -> torch.Tensor:
    """
    Derive a fallback `(T, 12)` temporal contact/motion feature tensor directly from
    a `(3, T, H, W)` action tube when loading legacy `.pt` files that were saved before
    explicit pose-feature extraction.
    """
    v = video_u8_or_float.float()
    if v.max() > 1.5:
        v = v / 255.0
    _, t_len, h, w = v.shape
    # Focus on central upper-body interaction zone
    center_crop = v[:, :, int(h * 0.15) : int(h * 0.75), int(w * 0.20) : int(w * 0.80)]
    diffs = torch.zeros(t_len, dtype=torch.float32)
    diffs[1:] = torch.mean(torch.abs(center_crop[:, 1:] - center_crop[:, :-1]), dim=(0, 2, 3)) * 6.0
    diffs[0] = diffs[1] if t_len > 1 else 0.0

    spatial_std = torch.std(center_crop, dim=(0, 2, 3))
    upper_energy = torch.mean(center_crop, dim=(0, 2, 3))

    feats = torch.zeros((t_len, NUM_FOUL_FEATURES), dtype=torch.float32)
    feats[:, 0] = torch.clamp(diffs, 0.0, 1.0)
    feats[:, 1] = torch.clamp(diffs * 1.2, 0.0, 1.0)
    feats[:, 2] = torch.clamp(spatial_std * 2.0, 0.0, 1.0)
    feats[:, 3] = torch.clamp(diffs * spatial_std * 3.0, 0.0, 1.0)
    feats[:, 4] = torch.clamp(0.5 + torch.gradient(upper_energy)[0] * 4.0, 0.0, 1.0)
    feats[:, 5] = torch.clamp(0.5 + torch.gradient(diffs)[0] * 2.0, 0.0, 1.0)
    feats[:, 6] = torch.clamp(upper_energy, 0.0, 1.0)
    feats[:, 11] = torch.clamp(0.6 * diffs + 0.4 * spatial_std * 2.0, 0.0, 1.0)
    return feats


class ConsistentVideoAugmentor:
    """
    Temporally consistent video augmentation pipeline for `(C, T, H, W)` tensors.
    Applies the exact same random horizontal flip and photometric color jitter
    across all `T = 16` frames so temporal motion gradients across contact frames
    remain physically coherent.
    """

    def __init__(
        self,
        is_train: bool = True,
        hflip_prob: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        mean: Tuple[float, float, float] = KINETICS400_MEAN,
        std: Tuple[float, float, float] = KINETICS400_STD,
    ) -> None:
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        else:
            video = video.float()
            if video.max() > 1.5:
                video = video / 255.0

        if self.is_train:
            if random.random() < self.hflip_prob:
                video = torch.flip(video, dims=[-1])

            b_factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
            c_factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            s_factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
            h_factor = random.uniform(-self.hue, self.hue)

            frames = video.permute(1, 0, 2, 3)
            augmented_frames = []
            for frame in frames:
                frame = TF.adjust_brightness(frame, b_factor)
                frame = TF.adjust_contrast(frame, c_factor)
                frame = TF.adjust_saturation(frame, s_factor)
                frame = TF.adjust_hue(frame, h_factor)
                augmented_frames.append(frame)
            video = torch.stack(augmented_frames, dim=1).clamp(0.0, 1.0)

        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        video = (video - mean) / std
        return video


class L2MFoulDataset(Dataset):
    """
    Custom PyTorch Dataset for NBA Foul Detection action tubes.
    Loads pre-saved `.pt` tensors from `data/processed/` containing:
      - `video`: `(3, 16, 224, 224)` RGB action tube centered on the foul interaction
      - `foul_features`: `(16, 12)` per-frame biomechanical foul-contact feature sequence
      - `label`: `1.0` (`foul`: CC / INC) or `0.0` (`no_foul`: CNC / IC)
    """

    def __init__(
        self,
        samples: Optional[List[Tuple[Path, int]]] = None,
        processed_dir: Path = Path("data/processed"),
        metadata_csv: Optional[Path] = None,
        is_train: bool = True,
        num_frames: int = 16,
        crop_size: int = 224,
        return_foul_features: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.is_train = is_train
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.return_foul_features = return_foul_features
        self.transform = transform or ConsistentVideoAugmentor(is_train=is_train)

        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._discover_samples(metadata_csv)

    def _discover_samples(self, metadata_csv: Optional[Path]) -> List[Tuple[Path, int]]:
        discovered: List[Tuple[Path, int]] = []
        if metadata_csv is not None and metadata_csv.exists():
            with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = int(row["label"])
                    tensor_path = row.get("tensor_path")
                    if tensor_path and Path(tensor_path).exists():
                        discovered.append((Path(tensor_path), label))
                        continue
                    play_id = row["play_id"]
                    cls_dir = "foul" if label == 1 else "no_foul"
                    candidate_pt = self.processed_dir / cls_dir / f"{play_id}.pt"
                    if candidate_pt.exists():
                        discovered.append((candidate_pt, label))
            if discovered:
                return discovered

        for cls_name, label in (("foul", 1), ("no_foul", 0)):
            cls_path = self.processed_dir / cls_name
            if cls_path.exists():
                for pt_file in sorted(cls_path.glob("*.pt")):
                    discovered.append((pt_file, label))
        return discovered

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video_and_features(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        if path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=False)
            video = payload["video"] if isinstance(payload, dict) else payload
            if video.shape != (3, self.num_frames, self.crop_size, self.crop_size):
                video = TF.resize(
                    video.permute(1, 0, 2, 3),
                    [self.crop_size, self.crop_size],
                    antialias=True,
                ).permute(1, 0, 2, 3)

            if isinstance(payload, dict) and "foul_features" in payload and payload["foul_features"] is not None:
                foul_feats = payload["foul_features"].float()
                if foul_feats.shape != (self.num_frames, NUM_FOUL_FEATURES):
                    foul_feats = derive_motion_foul_features_from_rgb(video)
            else:
                foul_feats = derive_motion_foul_features_from_rgb(video)
            return video, foul_feats

        if path.suffix == ".npy":
            video = torch.from_numpy(np.load(path))
            return video, derive_motion_foul_features_from_rgb(video)

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
        video = torch.from_numpy(sampled).permute(3, 0, 1, 2).contiguous()
        return video, derive_motion_foul_features_from_rgb(video)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        raw_video, foul_feats = self._load_video_and_features(path)
        video = self.transform(raw_video)
        target = torch.tensor(float(label), dtype=torch.float32)
        if self.return_foul_features:
            return video, foul_feats, target
        return video, target


class BiomechanicalFoulEncoder(nn.Module):
    """
    Temporal 1D-CNN + Contact-Frame Attention encoder over the `(B, T=16, F=12)`
    biomechanical foul-contact sequence (wrist-to-arm proximity, torso collision IoU,
    shooter upward gather elevation, and defender closing speed).
    """

    def __init__(self, in_features: int = NUM_FOUL_FEATURES, embed_dim: int = 64) -> None:
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.attn_Readout = nn.Linear(embed_dim, 1)

    def forward(self, foul_features: torch.Tensor) -> torch.Tensor:
        # foul_features: (B, T, F) -> (B, F, T)
        x = foul_features.permute(0, 2, 1).contiguous()
        h = self.conv_net(x).permute(0, 2, 1)  # (B, T, embed_dim)
        attn_weights = torch.softmax(self.attn_Readout(h), dim=1)  # Peak contact frame weighting
        pooled = torch.sum(h * attn_weights, dim=1)  # (B, embed_dim)
        return pooled


class FoulClassifier3D(nn.Module):
    """
    3D Spatiotemporal ResNet (`R(2+1)D-18`) + Biomechanical Foul-Contact Fusion Model.
    Wraps `torchvision.models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)`
    with `stem`, `layer1`, and `layer2` frozen so `layer3`, `layer4`, the biomechanical
    foul encoder, and the binary foul classification head (`nn.Linear(..., 1)`) are fine-tuned.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = 0.4,
        freeze_early: bool = True,
        pose_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)

        in_features = self.backbone.fc.in_features  # 512
        # Replace backbone.fc with Identity so we can fuse 512-D visual features + 64-D foul kinematics
        self.backbone.fc = nn.Identity()

        self.pose_encoder = BiomechanicalFoulEncoder(
            in_features=NUM_FOUL_FEATURES,
            embed_dim=pose_embed_dim,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features + pose_embed_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features + pose_embed_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(128, 1),
        )

        if freeze_early:
            self.freeze_early_layers()

    def freeze_early_layers(self) -> None:
        """
        Freeze `stem`, `layer1`, and `layer2` so only `layer3`, `layer4`, `pose_encoder`,
        and `classifier` are fine-tuned for foul prediction.
        """
        for module in (self.backbone.stem, self.backbone.layer1, self.backbone.layer2):
            for param in module.parameters():
                param.requires_grad = False

        for module in (self.backbone.layer3, self.backbone.layer4, self.pose_encoder, self.classifier):
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone layers for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

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
        x: torch.Tensor,
        foul_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        visual_feat = self.backbone(x)  # (B, 512)
        if foul_features is None:
            # Synthesize motion-energy contact features on the fly if only video tensor was passed
            batch_feats = [derive_motion_foul_features_from_rgb(x[b].detach().cpu()) for b in range(x.size(0))]
            foul_features = torch.stack(batch_feats, dim=0).to(x.device)
        pose_feat = self.pose_encoder(foul_features.to(x.device))  # (B, 64)
        fused = torch.cat([visual_feat, pose_feat], dim=-1)  # (B, 576)
        return self.classifier(fused)


if __name__ == "__main__":
    model = FoulClassifier3D(pretrained=False, dropout_p=0.4, freeze_early=True)
    summary = model.trainable_parameter_summary()
    dummy_clip = torch.randint(0, 256, (3, 16, 224, 224), dtype=torch.uint8)
    dummy_feats = torch.rand(1, 16, NUM_FOUL_FEATURES)
    augmentor = ConsistentVideoAugmentor(is_train=True)
    aug_tensor = augmentor(dummy_clip).unsqueeze(0)
    with torch.no_grad():
        out_logit = model(aug_tensor, dummy_feats)
    print(f"Input shape : {tuple(aug_tensor.shape)} + foul_features {tuple(dummy_feats.shape)}")
    print(f"Output shape: {tuple(out_logit.shape)}")
    print(f"Parameters  : {summary}")
