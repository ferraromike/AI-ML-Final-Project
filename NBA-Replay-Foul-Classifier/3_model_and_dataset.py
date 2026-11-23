#!/usr/bin/env python3
"""
STAGE 3: Context-Conditioned Multi-Head Challenge Replay Classifier & Dataset (`3_model_and_dataset.py`)

Engineered for official NBA Replay Center videos across all Coach's Challenge types:
  1. `ReplayFoulDataset` (aliased as `ReplayChallengeDataset`):
       - Loads shot-bounded multi-angle `.pt` action tubes `(3, 16, 224, 224)` + `(16, 16)` biomechanical
         and court-context features (auto-upgrading legacy `(16, 12)` tensors on the fly).
       - Tracks `play_id`, `game_id`, `challenge_type_id` (`0..2`), `call_label` (`0..7`),
         `resolution_label` (`0=STANDS, 1=OVERTURNED`), binary `label` (`0=NO FOUL, 1=FOUL`),
         and per-angle `saliency`.
       - Strictly excludes any play matching `data/test/` (by stem, CDN media slug, or visual signature).
  2. `ChallengeReplayClassifier` (aliased as `BinaryFoulClassifier` & `FoulClassifier3D`):
       - Stream A: Pretrained `R(2+1)D-18` spatiotemporal 3D CNN -> `512-D` visual embedding.
       - Stream B: `PoseTemporalEncoder` with 4-head temporal self-attention + kinematic summary
         statistics over the 16 biomechanical/court/rim/boundary channels -> `256-D` embedding.
       - Head 1 (`challenge_type_head`): Identifies what call is being challenged
         (`0=challenge_foul`, `1=challenge_oob`, `2=challenge_goaltending_bi`).
       - Context Conditioning (`challenge_context_embed`): Conditions decision heads on the known
         or inferred challenge type so the AI knows what is being challenged.
       - Head 2 (`call_head`): Predicts the 8 specific call outcomes (`foul_stands`,
         `foul_overturned_no_foul`, `foul_overturned_other_foul`, `oob_stands`, `oob_overturned`,
         `oob_proximate_foul`, `goaltending_bi_stands`, `goaltending_bi_overturned`).
       - Head 3 (`resolution_head`): Predicts `CALL STANDS` (`0`) vs. `CALL OVERTURNED` (`1`).
       - Head 4 (`foul_head`): Predicts calibrated binary foul probability (`P(foul)`).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import importlib
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

scrape_mod = importlib.import_module("1_scrape_replays")
preprocess_mod = importlib.import_module("2_preprocess_replays")

CALL_CLASSES = scrape_mod.CALL_CLASSES
CALL_CLASS_SPECS = scrape_mod.CALL_CLASS_SPECS
CALL_CLASS_TO_ID = scrape_mod.CALL_CLASS_TO_ID
ID_TO_CALL_CLASS = scrape_mod.ID_TO_CALL_CLASS
CHALLENGE_TYPES = scrape_mod.CHALLENGE_TYPES
CHALLENGE_TYPE_DISPLAY = scrape_mod.CHALLENGE_TYPE_DISPLAY
CHALLENGE_TYPE_TO_ID = scrape_mod.CHALLENGE_TYPE_TO_ID
ID_TO_CHALLENGE_TYPE = scrape_mod.ID_TO_CHALLENGE_TYPE
CHALLENGE_TYPE_VALID_CALL_IDS = scrape_mod.CHALLENGE_TYPE_VALID_CALL_IDS
upgrade_metadata_row = scrape_mod.upgrade_metadata_row
upgrade_pose_features_16d = preprocess_mod.upgrade_pose_features_16d
discover_held_out_test_ids = preprocess_mod.discover_held_out_test_ids

logger = logging.getLogger("replay_model")

# Standardized tensor dimensions
CHANNELS = 3
FRAMES = 16
HEIGHT = 224
WIDTH = 224
POSE_FEATURES = preprocess_mod.POSE_FEATURES  # 16
NUM_CHALLENGE_TYPES = len(CHALLENGE_TYPES)     # 3
NUM_CALL_CLASSES = len(CALL_CLASSES)           # 8

VIDEO_FEATURE_DIM = 512
POSE_HIDDEN_DIM = 256
COMBINED_FEATURE_DIM = VIDEO_FEATURE_DIM + POSE_HIDDEN_DIM  # 768
CLASSIFIER_HIDDEN_DIM = 384
CONTEXT_EMBED_DIM = 64

KINETICS400_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS400_STD = (0.22803, 0.22145, 0.216989)

POSE_FEATURE_NAMES: Tuple[str, ...] = preprocess_mod.POSE_FEATURE_NAMES


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
    """Fallback `(T, 16)` temporal motion & court descriptor if a `.pt` file lacks YOLO pose features."""
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

    feats12 = torch.zeros((num_frames, 12), dtype=torch.float32)
    feats12[:, 0] = torch.clamp(diffs, 0.0, 1.0)
    feats12[:, 1] = torch.clamp(diffs * 1.15, 0.0, 1.0)
    feats12[:, 2] = torch.clamp(spread * 2.0, 0.0, 1.0)
    feats12[:, 3] = torch.clamp(diffs * spread * 3.0, 0.0, 1.0)
    feats12[:, 4] = torch.clamp(0.5 + torch.gradient(bright)[0] * 4.0, 0.0, 1.0)
    feats12[:, 5] = torch.clamp(0.5 + torch.gradient(diffs)[0] * 2.0, 0.0, 1.0)
    feats12[:, 6] = torch.clamp(bright, 0.0, 1.0)
    feats12[:, 11] = torch.clamp(0.6 * diffs + 0.4 * spread * 2.0, 0.0, 1.0)
    return upgrade_pose_features_16d(feats12, video_frames)


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
    Tracks `play_id`, `game_id`, `challenge_type_id`, `call_label`, `resolution_label`,
    and `saliency` for leak-free grouped Train/Val splitting and multi-task challenge evaluation.
    """

    def __init__(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        processed_dir: Path = BASE_DIR / "data" / "processed",
        metadata_csv: Optional[Path] = BASE_DIR / "data" / "processed" / "processed_metadata.csv",
        master_metadata_csv: Optional[Path] = BASE_DIR / "data" / "replay_metadata.csv",
        test_dir: Optional[Path] = BASE_DIR / "data" / "test",
        is_train: bool = True,
        num_frames: int = FRAMES,
        crop_size: int = HEIGHT,
        max_samples: int = 0,
        use_cached_embeddings: bool = False,
        pose_noise_std: float = 0.015,
        return_meta: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.processed_dir = resolve_project_path(processed_dir)
        self.master_metadata_csv = (
            resolve_project_path(master_metadata_csv)
            if master_metadata_csv is not None
            else resolve_project_path(BASE_DIR / "data" / "replay_metadata.csv")
        )
        self.test_dir = (
            resolve_project_path(test_dir)
            if test_dir is not None
            else resolve_project_path(BASE_DIR / "data" / "test")
        )
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

        self.samples: List[Tuple[Path, int]] = [(r["path"], int(r["label"])) for r in self.records]
        self.play_ids: List[str] = [str(r["play_id"]) for r in self.records]
        self.game_ids: List[str] = [str(r["game_id"]) for r in self.records]

    def _load_master_lookup(self) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        lookup: Dict[str, Dict[str, Any]] = {}
        if self.master_metadata_csv and self.master_metadata_csv.exists():
            with open(self.master_metadata_csv, "r", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    ur = upgrade_metadata_row(dict(r))
                    pid = str(ur.get("play_id") or "").strip()
                    if pid:
                        rows.append(ur)
                        lookup[pid] = ur
        return rows, lookup

    def _discover_records(self, metadata_csv: Optional[Path]) -> List[Dict[str, Any]]:
        master_rows, master_lookup = self._load_master_lookup()
        excluded_test_ids, _ = discover_held_out_test_ids(self.test_dir, master_rows)
        discovered: List[Dict[str, Any]] = []
        seen_paths: Set[Path] = set()

        def _enrich_record(
            cand_path: Path,
            play_id: str,
            sample_id: str,
            game_id: str,
            raw_row: Dict[str, Any],
        ) -> Dict[str, Any]:
            m_row = master_lookup.get(play_id, {})
            merged = {**m_row, **raw_row}
            upgraded = upgrade_metadata_row(merged)
            outcome_cat = str(upgraded["outcome_category"])
            spec = CALL_CLASS_SPECS[outcome_cat]
            res_lbl = 1 if spec["call_resolution"] == "CALL OVERTURNED" else 0
            sal = float(raw_row.get("saliency") or 0.70)
            return {
                "path": cand_path,
                "label": int(spec["is_foul"]),
                "sample_id": sample_id,
                "play_id": play_id,
                "game_id": game_id,
                "challenge_type": str(spec["challenge_type"]),
                "challenge_type_id": int(spec["challenge_type_id"]),
                "outcome_category": outcome_cat,
                "call_label": int(spec["call_label"]),
                "call_resolution": str(spec["call_resolution"]),
                "resolution_label": res_lbl,
                "angle_idx": int(raw_row.get("angle_idx", 0) or 0),
                "saliency": sal,
            }

        if metadata_csv is not None and metadata_csv.exists():
            with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    raw_lbl = str(row.get("label", "")).strip()
                    if raw_lbl == "":
                        continue
                    play_id = str(row.get("play_id", "")).strip()
                    sample_id = str(row.get("sample_id", "")).strip() or play_id
                    if play_id in excluded_test_ids or sample_id in excluded_test_ids:
                        continue
                    game_id = str(row.get("game_id", "")).strip() or play_id.split("_")[0]
                    t_path_str = str(row.get("tensor_path", "")).strip()
                    ctype_folder = scrape_mod.normalize_challenge_type(
                        row.get("challenge_type") or master_lookup.get(play_id, {}).get("challenge_type")
                    )
                    cand = resolve_project_path(t_path_str) if t_path_str else (
                        self.processed_dir / ctype_folder / f"{sample_id}.pt"
                    )
                    if not cand.exists():
                        for fallback_folder in (*CHALLENGE_TYPES, "no_foul"):
                            alt = self.processed_dir / fallback_folder / f"{sample_id}.pt"
                            if alt.exists():
                                cand = alt
                                break
                    if cand.exists() and cand not in seen_paths:
                        seen_paths.add(cand)
                        discovered.append(_enrich_record(cand, play_id, sample_id, game_id, dict(row)))

        for folder_name in (*CHALLENGE_TYPES, "no_foul"):
            cls_dir = self.processed_dir / folder_name
            if not cls_dir.exists():
                continue
            for pt_file in sorted(cls_dir.glob("*.pt")):
                if pt_file in seen_paths:
                    continue
                stem = pt_file.stem
                base_play = stem.rsplit("_a", 1)[0] if "_a" in stem else stem
                if stem in excluded_test_ids or base_play in excluded_test_ids:
                    continue
                seen_paths.add(pt_file)
                default_ctype = "foul" if folder_name == "no_foul" else folder_name
                discovered.append(
                    _enrich_record(
                        pt_file,
                        base_play,
                        stem,
                        base_play.split("_")[0],
                        {"challenge_type": default_ctype, "angle_idx": 0},
                    )
                )

        real_records: List[Dict[str, Any]] = []
        for rec in discovered:
            try:
                payload = load_torch_file_safe(rec["path"], map_location="cpu")
                if isinstance(payload, dict):
                    if payload.get("is_synthetic", False):
                        continue
                    if "saliency" in payload:
                        rec["saliency"] = float(payload["saliency"])
            except Exception:
                continue
            real_records.append(rec)

        active = real_records if real_records else discovered
        # Per-play saliency gate: drop dead-ball secondary angles whose saliency is <58% of the play's best angle
        by_play: Dict[str, float] = {}
        for r in active:
            pid = str(r["play_id"])
            by_play[pid] = max(by_play.get(pid, 0.0), float(r.get("saliency", 0.5)))

        filtered: List[Dict[str, Any]] = [
            r for r in active
            if float(r.get("saliency", 0.5)) >= 0.58 * by_play.get(str(r["play_id"]), 0.0)
        ]
        return filtered if filtered else active

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

        if pose is None or not isinstance(pose, torch.Tensor):
            pose = derive_motion_features_from_rgb(video)
        else:
            pose = upgrade_pose_features_16d(pose.float(), video)

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
        foul_label = float(rec["label"])
        ctype_id = int(rec.get("challenge_type_id", 0))
        call_label = int(rec.get("call_label", 0))
        res_label = float(rec.get("resolution_label", 0))
        saliency = float(rec.get("saliency", 0.70))

        raw_video, pose_feats, video_emb = self._load_item(path)
        if self.use_cached_embeddings:
            if video_emb is None:
                video_emb = self._extract_and_cache_embedding(raw_video, path)
            video_input = video_emb
            if self.is_train:
                # Subtle embedding-space regularization in frozen/cached mode
                video_input = video_input + torch.randn_like(video_input) * 0.015
        else:
            video_input = self.transform(raw_video)

        if self.pose_noise_std > 0.0:
            pose_feats = (pose_feats + torch.randn_like(pose_feats) * self.pose_noise_std).clamp(0.0, 1.0)

        t_foul = torch.tensor(foul_label, dtype=torch.float32)
        t_ctype = torch.tensor(ctype_id, dtype=torch.long)
        t_call = torch.tensor(call_label, dtype=torch.long)
        t_res = torch.tensor(res_label, dtype=torch.float32)

        if self.return_meta:
            return (
                video_input,
                pose_feats,
                t_foul,
                t_ctype,
                t_call,
                t_res,
                str(rec["play_id"]),
                str(rec["sample_id"]),
                saliency,
            )
        return video_input, pose_feats, t_foul, t_ctype, t_call, t_res


class PoseTemporalEncoder(nn.Module):
    """
    Temporal Transformer + Kinematic Summary Encoder for 16-D biomechanical & court-context features:
      Input : `(B, FRAMES=16, POSE_FEATURES=16)` (auto-upgrades `(B, 16, 12)` if passed)
      Output: `(B, POSE_HIDDEN_DIM=256)` combining:
                - 4-head Temporal Self-Attention pooled representation (`128-D`)
                - Temporal Max-Pooled apex representation (`64-D`)
                - Explicit Temporal Summary Statistics (`mean`, `max`, `std`, `apex_delta` -> `64-D`)
    """

    def __init__(
        self,
        in_features: int = POSE_FEATURES,
        embed_dim: int = POSE_HIDDEN_DIM,
        num_frames: int = FRAMES,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        half_dim = embed_dim // 2   # 128
        quarter_dim = embed_dim // 4  # 64

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
        self.max_proj = nn.Sequential(
            nn.Linear(half_dim, quarter_dim),
            nn.LayerNorm(quarter_dim),
            nn.GELU(),
        )
        # Explicit summary encoder over [mean, max, std, apex_center_mean] (4 * 16 = 64 -> 64)
        self.stats_proj = nn.Sequential(
            nn.Linear(in_features * 4, quarter_dim),
            nn.LayerNorm(quarter_dim),
            nn.GELU(),
        )

    def _ensure_16d(self, pose_features: torch.Tensor) -> torch.Tensor:
        if pose_features.size(-1) == self.in_features:
            return pose_features
        upgraded = [upgrade_pose_features_16d(pose_features[b]) for b in range(pose_features.size(0))]
        return torch.stack(upgraded, dim=0).to(pose_features.device)

    def forward(self, pose_features: torch.Tensor) -> torch.Tensor:
        p = self._ensure_16d(pose_features)
        T = p.size(1)
        x = self.input_proj(p)
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T, :]
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        weights = torch.softmax(self.pool_weights(x), dim=1)  # (B, T, 1)
        attn_pooled = (x * weights).sum(dim=1)                # (B, 128)
        max_pooled = self.max_proj(x.max(dim=1).values)       # (B, 64)

        p_mean = p.mean(dim=1)
        p_max = p.max(dim=1).values
        p_std = p.std(dim=1, unbiased=False)
        mid_lo = max(0, T // 4)
        mid_hi = max(mid_lo + 1, (3 * T) // 4)
        p_apex = p[:, mid_lo:mid_hi, :].mean(dim=1)
        stats_vec = self.stats_proj(torch.cat([p_mean, p_max, p_std, p_apex], dim=-1))  # (B, 64)

        return torch.cat([attn_pooled, max_pooled, stats_vec], dim=-1)  # (B, 256)


class ChallengeReplayClassifier(nn.Module):
    """
    Context-Conditioned Multi-Head NBA Challenge Replay Classifier:
      - Stream A (`video_backbone`): Pretrained `R(2+1)D-18` -> `512-D` spatiotemporal visual embedding
      - Stream B (`pose_network`): `PoseTemporalEncoder` -> `256-D` biomechanical + court-context embedding
      - `challenge_type_head`: Predicts what is being challenged (`0=challenge_foul`, `1=challenge_oob`, `2=challenge_goaltending_bi`)
      - `challenge_context_embed`: Embeds the known or predicted challenge type (`64-D`) to condition all ruling heads
      - `call_head`: Predicts the 8 specific call outcomes (`CALL_CLASSES` `0..7`)
      - `resolution_head`: Predicts whether the call on the floor `STANDS (0)` or is `OVERTURNED (1)`
      - `foul_head`: Predicts calibrated binary `FOUL (1)` vs. `NO FOUL (0)` logit on the play
    """

    def __init__(
        self,
        pretrained: bool = True,
        backbone_mode: str = "partial",
        dropout_p: float = 0.30,
        modality_dropout: float = 0.08,
        context_drop_prob: float = 0.20,
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
        self.context_drop_prob = float(context_drop_prob)

        # Shared play trunk (768 -> 384)
        self.shared_trunk = nn.Sequential(
            nn.LayerNorm(self.combined_feature_dim),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.combined_feature_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
        )

        # Head 1: What is being challenged? (3 classes: foul, oob, goaltending_bi)
        self.challenge_type_head = nn.Sequential(
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(classifier_hidden_dim + POSE_FEATURES * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, NUM_CHALLENGE_TYPES),
        )

        # Context embedding for what is being challenged (conditions all decision heads)
        self.challenge_context_embed = nn.Embedding(NUM_CHALLENGE_TYPES, CONTEXT_EMBED_DIM)
        cond_dim = classifier_hidden_dim + CONTEXT_EMBED_DIM + POSE_FEATURES * 2  # 384 + 64 + 32 = 480

        # Head 2: Specific Correct Call Head (8 classes across Foul, OOB, and Goaltending/BI)
        self.call_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Dropout(p=dropout_p * 0.6),
            nn.Linear(cond_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, NUM_CALL_CLASSES),
        )

        # Head 3: Challenge Resolution Head (0 = CALL STANDS, 1 = CALL OVERTURNED)
        self.resolution_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(cond_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Head 4: Binary Foul vs. No-Foul Head (also aliased via fusion_head compatibility)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Dropout(p=dropout_p * 0.6),
            nn.Linear(cond_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Learnable weight on biomechanical contact-vs-clean-ball prior
        self.kinematic_prior_scale = nn.Parameter(torch.tensor(1.15, dtype=torch.float32))

        self.backbone_mode = backbone_mode.lower()
        self.set_backbone_mode(self.backbone_mode)

    def set_backbone_mode(self, mode: str) -> None:
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

        for mod in (
            self.pose_network,
            self.shared_trunk,
            self.challenge_type_head,
            self.challenge_context_embed,
            self.call_head,
            self.resolution_head,
            self.fusion_head,
        ):
            for p in mod.parameters():
                p.requires_grad = True
        self.kinematic_prior_scale.requires_grad = True

    def parameter_groups(self, base_lr: float, backbone_lr_scale: float = 0.1) -> List[Dict[str, Any]]:
        backbone_params = [p for p in self.video_backbone.parameters() if p.requires_grad]
        head_params = [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("video_backbone.")
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

    def _compute_kinematic_priors(
        self,
        p16: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute interpretable biomechanical & court-geometry log-odds priors from `p16` `(B, T, 16)`:
          - `foul_prior` `(B, 1)`:
              Positive when arm intrusion (0), cylinder invasion (1), collision shock (10),
              and ball_vs_body (8) are high while clean defender-on-ball proximity (6) is low.
              Negative when defender makes clean play on the ball (6) with low collision shock (10)
              or play is a non-contact OOB/goaltending boundary play.
          - `ctype_prior` `(B, 3)`:
              Court-geometry cues for `[challenge_foul, challenge_oob, challenge_goaltending_bi]`.
          - `call_prior` `(B, 8)`:
              Fine-grained kinematic cues for the 8 specific call classes.
        """
        p_mean = p16.mean(dim=1)          # (B, 16)
        p_max = p16.max(dim=1).values     # (B, 16)

        arm_c = 0.55 * p_mean[:, 0] + 0.45 * p_max[:, 0]
        cyl_c = 0.55 * p_mean[:, 1] + 0.45 * p_max[:, 1]
        land_c = 0.50 * p_mean[:, 2] + 0.50 * p_max[:, 2]
        vert_dev = p_mean[:, 3]
        def_ball = 0.60 * p_mean[:, 6] + 0.40 * p_max[:, 6]
        body_vs_ball = p_mean[:, 8]
        shock = 0.65 * p_mean[:, 10] + 0.35 * p_max[:, 10]
        apex = p_mean[:, 11]
        rim_prox = 0.50 * p_mean[:, 12] + 0.50 * p_max[:, 12]
        bound_prox = 0.55 * p_mean[:, 13] + 0.45 * p_max[:, 13]
        deflect = 0.50 * p_mean[:, 14] + 0.50 * p_max[:, 14]
        charge_idx = 0.55 * p_mean[:, 15] + 0.45 * p_max[:, 15]

        # Clean ball contest vs. illegal contact shock score (gated by actual player-pair contact)
        clean_ball_play = p_mean[:, 6]
        vertical_shock = p_mean[:, 10]
        contact_sum = p_mean[:, 0] + p_mean[:, 1]
        contact_gate = torch.clamp((contact_sum - 0.45) / 0.55, 0.0, 1.0)
        raw_illegal = (
            0.75 * contact_sum
            + 1.50 * vertical_shock
            - 3.60 * clean_ball_play
            + 0.25 * land_c * (1.0 - clean_ball_play)
        )
        illegal_contact = contact_gate * raw_illegal - (1.0 - contact_gate) * 0.65
        foul_prior = illegal_contact.unsqueeze(-1)  # (B, 1)

        # Challenge type prior: [foul, oob, goaltending, flagrant]
        foul_cue = 1.10 * (arm_c + cyl_c) + 0.35
        oob_cue = 1.15 * bound_prox * (1.0 - 0.60 * cyl_c) + 0.40 * deflect - 0.30 * rim_prox
        gt_cue = 1.20 * rim_prox * (1.0 - 0.50 * cyl_c) + 0.40 * apex - 0.35 * bound_prox
        flagrant_cue = 0.95 * (arm_c + vertical_shock) + 0.45 * body_vs_ball - 0.30 * clean_ball_play
        ctype_prior = torch.stack([foul_cue, oob_cue, gt_cue, flagrant_cue], dim=-1)

        # 10-class call outcome kinematic prior
        p_foul_stands = illegal_contact + 0.30 * vert_dev
        p_foul_no_foul = -illegal_contact + 0.85 * def_ball
        p_foul_other = 0.80 * illegal_contact + 0.75 * charge_idx - 0.35 * vert_dev
        p_oob_stands = oob_cue - 0.30 * deflect
        p_oob_over = oob_cue + 0.45 * deflect
        p_oob_foul = 0.65 * oob_cue + 0.85 * illegal_contact
        p_gt_stands = gt_cue + 0.30 * rim_prox
        p_gt_over = gt_cue + 0.35 * def_ball - 0.25 * cyl_c
        p_flag_conf = flagrant_cue + 0.65 * vertical_shock + 0.35 * arm_c
        p_flag_comm = flagrant_cue - 0.35 * vertical_shock + 0.25 * cyl_c
        call_prior = torch.stack(
            [
                p_foul_stands,
                p_foul_no_foul,
                p_foul_other,
                p_oob_stands,
                p_oob_over,
                p_oob_foul,
                p_gt_stands,
                p_gt_over,
                p_flag_conf,
                p_flag_comm,
            ],
            dim=-1,
        )
        return foul_prior, ctype_prior, call_prior

    def forward(
        self,
        video_input: torch.Tensor,
        pose_features: Optional[torch.Tensor] = None,
        challenge_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if video_input.dim() == 2 and video_input.size(-1) == self.video_feature_dim:
            vis_emb = video_input
        elif self.backbone_mode == "frozen":
            with torch.no_grad():
                vis_emb = self.video_backbone(video_input)
        else:
            vis_emb = self.video_backbone(video_input)

        if pose_features is None:
            if video_input.dim() == 5:
                batch_feats = [
                    derive_motion_features_from_rgb(video_input[b].detach().cpu())
                    for b in range(video_input.size(0))
                ]
                p16 = torch.stack(batch_feats, dim=0).to(video_input.device)
            else:
                p16 = torch.zeros((video_input.size(0), FRAMES, POSE_FEATURES), dtype=torch.float32, device=video_input.device)
        else:
            p16 = self.pose_network._ensure_16d(pose_features.to(video_input.device))

        pose_emb = self.pose_network(p16)

        if self.training and self.modality_dropout > 0.0:
            r = random.random()
            if r < self.modality_dropout * 0.5:
                pose_emb = torch.zeros_like(pose_emb)
            elif r < self.modality_dropout:
                vis_emb = torch.zeros_like(vis_emb)

        combined = torch.cat([vis_emb, pose_emb], dim=-1)
        h_play = self.shared_trunk(combined)  # (B, 384)

        p_mean = p16.mean(dim=1)
        p_max = p16.max(dim=1).values
        pose_summary = torch.cat([p_mean, p_max], dim=-1)  # (B, 32)
        foul_prior, ctype_prior, call_prior = self._compute_kinematic_priors(p16)

        # Head 1: Predict what is being challenged
        challenge_type_logits = self.challenge_type_head(torch.cat([h_play, pose_summary], dim=-1)) + 0.35 * ctype_prior
        challenge_type_probs = torch.softmax(challenge_type_logits, dim=-1)

        # Build Challenge-Context Embedding (explicit context when given, soft expectation when unknown/dropped)
        soft_ctx = challenge_type_probs @ self.challenge_context_embed.weight  # (B, 64)
        if challenge_type_ids is not None:
            ct_ids = challenge_type_ids.to(video_input.device).long()
            valid_mask = (ct_ids >= 0) & (ct_ids < NUM_CHALLENGE_TYPES)
            clamped_ids = ct_ids.clamp(0, NUM_CHALLENGE_TYPES - 1)
            hard_ctx = self.challenge_context_embed(clamped_ids)
            if self.training and self.context_drop_prob > 0.0:
                drop_mask = (torch.rand(ct_ids.size(0), 1, device=video_input.device) < self.context_drop_prob)
                use_hard = valid_mask.unsqueeze(-1) & (~drop_mask)
            else:
                use_hard = valid_mask.unsqueeze(-1)
            ctx_emb = torch.where(use_hard, hard_ctx, soft_ctx)
        else:
            ctx_emb = soft_ctx

        h_cond = torch.cat([h_play, ctx_emb, pose_summary], dim=-1)  # (B, 480)

        prior_w = torch.clamp(self.kinematic_prior_scale, 0.2, 2.5)
        foul_logits = self.fusion_head(h_cond) + prior_w * foul_prior
        resolution_logits = self.resolution_head(h_cond)
        call_logits = self.call_head(h_cond) + 0.45 * prior_w * call_prior

        if not return_dict:
            return foul_logits

        return {
            "foul_logits": foul_logits,
            "challenge_type_logits": challenge_type_logits,
            "call_logits": call_logits,
            "resolution_logits": resolution_logits,
        }

    @torch.no_grad()
    def predict_challenge(
        self,
        video_input: torch.Tensor,
        pose_features: torch.Tensor,
        challenge_type: Optional[Union[str, int]] = None,
        angle_saliencies: Optional[List[float]] = None,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Full multi-angle Coach's Challenge inference:
          1. Identifies what is being challenged (`challenge_type`: either provided via metadata/CLI
             or predicted by `challenge_type_head`).
          2. Pools multi-angle predictions using saliency-weighted attention so primary contact/boundary/rim
             shots are never washed out by dead-ball camera angles.
          3. Determines the exact Challenge Resolution (`CALL STANDS` vs. `CALL OVERTURNED`) and
             Specific Correct Call (`CALL_CLASSES` `0..7`) consistent with the challenged call.
        """
        self.eval()
        dev = next(self.parameters()).device
        vid_in = video_input.to(dev)
        pose_in = self.pose_network._ensure_16d(pose_features.to(dev))
        B = vid_in.size(0)

        explicit_ctype_id: Optional[int] = None
        if isinstance(challenge_type, int) and 0 <= challenge_type < NUM_CHALLENGE_TYPES:
            explicit_ctype_id = int(challenge_type)
        elif isinstance(challenge_type, str):
            norm_ct = challenge_type.strip().lower()
            if norm_ct not in ("auto", "none", ""):
                mapped = scrape_mod.CHALLENGE_TYPE_ALIASES.get(norm_ct)
                if mapped in CHALLENGE_TYPE_TO_ID:
                    explicit_ctype_id = CHALLENGE_TYPE_TO_ID[mapped]

        ct_tensor = (
            torch.full((B,), explicit_ctype_id, dtype=torch.long, device=dev)
            if explicit_ctype_id is not None
            else None
        )
        out = self.forward(vid_in, pose_in, challenge_type_ids=ct_tensor, return_dict=True)
        assert isinstance(out, dict)

        foul_probs_angle = torch.sigmoid(out["foul_logits"]).squeeze(-1).detach().cpu().numpy()
        foul_probs_angle = np.atleast_1d(foul_probs_angle).astype(float)
        res_probs_angle = torch.sigmoid(out["resolution_logits"]).squeeze(-1).detach().cpu().numpy()
        res_probs_angle = np.atleast_1d(res_probs_angle).astype(float)
        ctype_logits_angle = out["challenge_type_logits"].detach().cpu().numpy()
        call_logits_angle = out["call_logits"].detach().cpu().numpy()

        # Saliency + actual tube-activity weighted multi-angle aggregation
        if angle_saliencies is not None and len(angle_saliencies) == B:
            sal_arr = np.asarray(angle_saliencies, dtype=float)
        else:
            sal_arr = np.ones(B, dtype=float)
        tube_activity = pose_in[:, :, [0, 1, 12, 13]].mean(dim=(1, 2)).detach().cpu().numpy().astype(float)
        eff_sal = sal_arr * (0.35 + 0.65 * tube_activity)
        sal_logits = (eff_sal - np.max(eff_sal)) * 5.0
        angle_weights = np.exp(sal_logits)
        angle_weights = angle_weights / max(float(np.sum(angle_weights)), 1e-8)

        best_sal_idx = int(np.argmax(eff_sal))
        weighted_foul_prob = float(np.sum(angle_weights * foul_probs_angle))
        agg_foul_prob = float(0.65 * weighted_foul_prob + 0.35 * float(foul_probs_angle[best_sal_idx]))
        agg_overturned_prob = float(np.sum(angle_weights * res_probs_angle))

        agg_ctype_logits = np.sum(angle_weights[:, None] * ctype_logits_angle, axis=0)
        exp_ct = np.exp(agg_ctype_logits - np.max(agg_ctype_logits))
        ctype_probs = exp_ct / max(float(np.sum(exp_ct)), 1e-8)

        active_ctype_id = explicit_ctype_id if explicit_ctype_id is not None else int(np.argmax(ctype_probs))
        active_ctype = ID_TO_CHALLENGE_TYPE[active_ctype_id]
        valid_call_ids = CHALLENGE_TYPE_VALID_CALL_IDS[active_ctype_id]

        # Align 8-class call logits with calibrated foul probability & resolution head within valid_call_ids
        agg_call_logits = np.sum(angle_weights[:, None] * call_logits_angle, axis=0).copy()
        eps = 1e-5
        cal_foul_log_odds = float(np.log(max(agg_foul_prob, eps) / max(1.0 - agg_foul_prob, eps))) - float(
            np.log(max(threshold, eps) / max(1.0 - threshold, eps))
        )
        overturn_log_odds = float(np.log(max(agg_overturned_prob, eps) / max(1.0 - agg_overturned_prob, eps)))

        if active_ctype_id == 0:  # foul: {0: foul_stands, 1: foul_overturned_no_foul, 2: foul_overturned_other_foul}
            agg_call_logits[0] += 0.90 * cal_foul_log_odds - 0.35 * overturn_log_odds
            agg_call_logits[1] += -1.15 * cal_foul_log_odds + 0.45 * overturn_log_odds
            agg_call_logits[2] += 0.70 * cal_foul_log_odds + 0.35 * overturn_log_odds
        elif active_ctype_id == 1:  # oob: {3: oob_stands, 4: oob_overturned, 5: oob_proximate_foul}
            agg_call_logits[3] += -0.60 * overturn_log_odds - 0.30 * max(0.0, cal_foul_log_odds)
            agg_call_logits[4] += 0.60 * overturn_log_odds - 0.25 * max(0.0, cal_foul_log_odds)
            agg_call_logits[5] += 0.50 * overturn_log_odds + 0.85 * cal_foul_log_odds
        elif active_ctype_id == 2:  # goaltending: {6: goaltending_bi_stands, 7: goaltending_bi_overturned}
            agg_call_logits[6] += -0.75 * overturn_log_odds
            agg_call_logits[7] += 0.75 * overturn_log_odds
        elif active_ctype_id == 3:  # flagrant: {8: flagrant_confirmed, 9: flagrant_downgraded_common_foul}
            agg_call_logits[8] += -0.65 * overturn_log_odds + 0.40 * cal_foul_log_odds
            agg_call_logits[9] += 0.65 * overturn_log_odds

        # Mask to valid call outcomes for the active challenge type
        valid_logits = np.array([agg_call_logits[idx] for idx in valid_call_ids], dtype=float)
        exp_valid = np.exp(valid_logits - np.max(valid_logits))
        valid_probs = exp_valid / max(float(np.sum(exp_valid)), 1e-8)

        # Ensure strict consistency between calibrated foul threshold and foul_challenge outcome
        if active_ctype_id == 0:
            is_foul_decision = bool(agg_foul_prob >= threshold)
            if is_foul_decision:
                # Choose between 0 (foul_stands) and 2 (foul_overturned_other_foul)
                chosen_call_id = 0 if agg_call_logits[0] >= agg_call_logits[2] else 2
            else:
                chosen_call_id = 1  # foul_overturned_no_foul
        else:
            best_local_idx = int(np.argmax(valid_probs))
            chosen_call_id = int(valid_call_ids[best_local_idx])

        chosen_outcome = ID_TO_CALL_CLASS[chosen_call_id]
        spec = CALL_CLASS_SPECS[chosen_outcome]
        is_foul = bool(spec["is_foul"] == 1)

        call_probs_dict = {
            ID_TO_CALL_CLASS[cid]: round(float(p), 4)
            for cid, p in zip(valid_call_ids, valid_probs)
        }
        chosen_conf = float(call_probs_dict.get(chosen_outcome, float(np.max(valid_probs))))
        if active_ctype_id == 0:
            chosen_conf = max(chosen_conf, agg_foul_prob if is_foul else (1.0 - agg_foul_prob))

        return {
            "challenge_type": active_ctype,
            "challenge_type_id": active_ctype_id,
            "challenge_type_display": CHALLENGE_TYPE_DISPLAY[active_ctype],
            "challenge_type_probs": {
                CHALLENGE_TYPES[i]: round(float(ctype_probs[i]), 4)
                for i in range(NUM_CHALLENGE_TYPES)
            },
            "call_label": chosen_call_id,
            "outcome_category": chosen_outcome,
            "call_resolution": spec["call_resolution"],
            "correct_call": spec["correct_call"],
            "verdict": spec["short_verdict"],
            "binary_verdict": "FOUL" if is_foul else "NO FOUL",
            "is_foul": is_foul,
            "foul_prob": round(agg_foul_prob, 4),
            "call_confidence": round(chosen_conf, 4),
            "overturned_prob": round(agg_overturned_prob, 4),
            "call_probs": call_probs_dict,
            "angle_probs": [round(float(p), 4) for p in foul_probs_angle],
            "angle_saliencies": [round(float(s), 4) for s in sal_arr],
            "angle_weights": [round(float(w), 4) for w in angle_weights],
            "threshold": round(float(threshold), 4),
        }

    @torch.no_grad()
    def predict_foul(
        self,
        video_input: torch.Tensor,
        pose_features: torch.Tensor,
        threshold: float = 0.50,
        challenge_type: Optional[Union[str, int]] = None,
        angle_saliencies: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper returning both binary foul fields and full challenge call resolution."""
        return self.predict_challenge(
            video_input=video_input,
            pose_features=pose_features,
            challenge_type=challenge_type,
            angle_saliencies=angle_saliencies,
            threshold=threshold,
        )


# Backward-compatible aliases
ReplayChallengeDataset = ReplayFoulDataset
BinaryFoulClassifier = ChallengeReplayClassifier
FoulClassifier3D = ChallengeReplayClassifier


if __name__ == "__main__":
    model = ChallengeReplayClassifier(pretrained=False, backbone_mode="partial")
    summary = model.trainable_parameter_summary()
    dummy_v = torch.randn(2, CHANNELS, FRAMES, HEIGHT, WIDTH)
    dummy_p = torch.rand(2, FRAMES, POSE_FEATURES)
    dummy_ct = torch.tensor([0, 1], dtype=torch.long)
    out = model(dummy_v, dummy_p, challenge_type_ids=dummy_ct, return_dict=True)
    print("ChallengeReplayClassifier (NBA Replay Archive — Multi-Head Challenge & Call Classifier):")
    print(f"  Video Input         : (B, {CHANNELS}, {FRAMES}, {HEIGHT}, {WIDTH}) -> {VIDEO_FEATURE_DIM}-D")
    print(f"  Pose/Court Input    : (B, {FRAMES}, {POSE_FEATURES}) -> {POSE_HIDDEN_DIM}-D")
    print(f"  Challenge Type Head : {tuple(out['challenge_type_logits'].shape)} ({NUM_CHALLENGE_TYPES} types)")
    print(f"  Specific Call Head  : {tuple(out['call_logits'].shape)} ({NUM_CALL_CLASSES} calls)")
    print(f"  Resolution Head     : {tuple(out['resolution_logits'].shape)} (STANDS vs. OVERTURNED)")
    print(f"  Binary Foul Head    : {tuple(out['foul_logits'].shape)} (FOUL vs. NO FOUL)")
    print(
        f"  Parameters          : Total={summary['total_params']:,} | "
        f"Trainable={summary['trainable_params']:,} | Frozen={summary['frozen_params']:,}"
    )
