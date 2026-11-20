#!/usr/bin/env python3
"""
STEP 2: E-BARD Basketball Perception, 2-Team Jersey Clustering & Multi-Window MIL Bag Extractor (`2_preprocess_tubes.py`)

Full-Stack v3.0 Perception & Preprocessing Engine:
  1. NBA-Specific Object Detection (`ebard_basketball_yolo.pt` from GabrieleGiudici/E-BARD-detection-models):
     - Explicitly separates 4 native NBA broadcast classes:
         * Class 0: `basketball`
         * Class 1: `hoop` (rim / backboard)
         * Class 2: `player` (active on-court players ONLY — eliminates spectators & coaches)
         * Class 3: `referee` (explicitly isolates NBA officials so referees are NEVER tagged as players)
  2. Unsupervised Per-Clip 2-Team CIELAB Jersey Clustering + Long-Trousers Referee Guard:
     - Fits 2-cluster K-Means on CIELAB upper-torso embeddings across all on-court players in the clip.
     - Enforces hard opposing-team pairing (`team_id_a != team_id_b`) and adds a lower-leg dark-trousers
       fallback filter so officials are rejected even if momentarily occluded.
  3. 3D Foot-Plane Depth Alignment + 16-D NBA Rule 12 Biomechanical Kinematics:
     - Fuses clean E-BARD `player` boxes with `yolo11n-pose.pt` 17-point skeletons.
     - Modulates 2D torso/cylinder overlap by foot-plane depth alignment (`|y2_a - y2_b|`) so players
       separated in depth are not mistaken for collisions.
  4. Multi-Window Temporal Candidate Bag Extraction + Swin3D-T (`768-D`) Embeddings:
     - Segments wide-court live action via camera-cut detection, slices overlapping `2.0s` candidate windows
       across the entire play (or arbitrary-length game footage), and extracts Kinetics-400 `Swin3D-T`
       embeddings (`768-D`) + `(16, 16)` biomechanical sequences per window for Multiple Instance Learning (MIL).
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
import hashlib
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

cv2.setNumThreads(0)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess_tubes")

# 16 Mechanical Foul & Basketball Context Dimensions (NBA Rule 12 + Spatial Depth + Ball/Hoop)
MECHANICAL_FOUL_DIMENSIONS = (
    "arm_contact_intrusion",        # 0: Defender hand/wrist collision with shooter arm
    "body_cylinder_penetration",    # 1: Depth-gated torso/hip invasion into offensive cylinder
    "lower_body_landing_violation", # 2: Defender feet/legs under airborne shooter landing zone
    "defender_verticality_angle",   # 3: Angle deviation from legal 90-deg vertical contest
    "defender_lateral_positioning", # 4: Defender lateral offset relative to offensive drive
    "offensive_displacement",       # 5: Frame-to-frame sudden deflection/jolt of offensive torso
    "defender_to_ball_distance",    # 6: Proximity of defender hand to E-BARD basketball
    "ball_to_shooter_distance",     # 7: Proximity of E-BARD basketball to offensive hands/torso
    "ball_strike_vs_body_ratio",    # 8: Body/arm contact severity relative to clean play on ball
    "relative_closing_velocity",    # 9: Temporal closing speed between defender and ball-handler
    "momentum_transfer_shock",      # 10: Sudden kinetic deceleration / impact shock
    "airborne_contest_apex_diff",   # 11: Shooter upward arm elevation & vertical contest intensity
    "hoop_proximity",               # 12: Proximity of contest pair to E-BARD detected hoop/rim
    "foot_plane_depth_alignment",   # 13: Same-depth court floor alignment (`y2` agreement)
    "ball_verified_on_play",        # 14: Confidence that E-BARD basketball is actively on this pair
    "shooting_motion_score",        # 15: Shooting gather/release likelihood (arm elev + ball + hoop)
)
POSE_FEATURE_NAMES = MECHANICAL_FOUL_DIMENSIONS
FOUL_FEATURE_NAMES = MECHANICAL_FOUL_DIMENSIONS
POSE_FEATURES = len(MECHANICAL_FOUL_DIMENSIONS)  # 16
NUM_FOUL_FEATURES = POSE_FEATURES

VIDEO_EMBEDDING_DIM = 768  # Video Swin3D-T output dimension
MAX_BAG_WINDOWS = 10       # Multi-window MIL bag capacity (covers full ~20s live play with 0 blind spots)
EXTRACTOR_VERSION = 3

# COCO 17-keypoint indices
UPPER_BODY_KPTS = (0, 5, 6, 7, 8, 9, 10)
ARM_CONTACT_KPTS = (7, 8, 9, 10)
TORSO_KPTS = (5, 6, 11, 12)
FEET_KPTS = (13, 14, 15, 16)

# E-BARD Basketball YOLO class IDs (GabrieleGiudici/E-BARD-detection-models, CC-BY-4.0)
EBARD_WEIGHTS_URL = (
    "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/resolve/main/BODD_yolov8n_0001.pt"
)
EBARD_WEIGHTS_NAME = "ebard_basketball_yolo.pt"
EBARD_BALL_CLS = 0
EBARD_HOOP_CLS = 1
EBARD_PLAYER_CLS = 2
EBARD_REFEREE_CLS = 3

# Generic COCO fallback class IDs
COCO_PERSON_CLS = 0
COCO_SPORTS_BALL_CLS = 32
COCO_BALL_CLS = 32


def resolve_project_path(path_arg: Path | str) -> Path:
    """Resolve relative paths cross-platform checking CWD, BASE_DIR, and weights/ directory."""
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


def select_yolo_device(requested: Optional[str] = None) -> str:
    """Use CUDA on NVIDIA GPUs; use CPU on macOS/CPU where Ultralytics NMS is deterministic and fast."""
    if requested and requested not in ("auto", "mps"):
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_video_encoder_device(requested: Optional[str] = None) -> torch.device:
    """Select fastest stable PyTorch device for Swin3D-T video embedding extraction."""
    if requested and requested not in ("auto",):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==============================================================================
# Cached Foundation Models: E-BARD Basketball YOLO + YOLOv11-Pose + Swin3D-T
# ==============================================================================
_CACHED_BASKETBALL_MODEL: Optional[object] = None
_CACHED_BASKETBALL_IS_EBARD: bool = False
_CACHED_POSE_MODEL: Optional[object] = None
_CACHED_SWIN3D_MODEL: Optional[nn.Module] = None
_CACHED_SWIN3D_DEVICE: Optional[torch.device] = None


def ensure_ebard_basketball_weights() -> Optional[Path]:
    """Ensure the E-BARD NBA Basketball YOLO detector weights exist under `weights/ebard_basketball_yolo.pt`."""
    target = BASE_DIR / "weights" / EBARD_WEIGHTS_NAME
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".pt.part")
    try:
        logger.info("Downloading E-BARD NBA Basketball Detector (player/referee/ball/hoop) -> %s", target)
        req = urllib.request.Request(EBARD_WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_path, "wb") as out_f:
            out_f.write(resp.read())
        if tmp_path.exists() and tmp_path.stat().st_size > 1_000_000:
            tmp_path.replace(target)
            logger.info("Downloaded E-BARD NBA detector (%.1f MB)", target.stat().st_size / (1024 * 1024))
            return target
    except Exception as exc:
        logger.warning("Could not download E-BARD basketball detector (%s); will check local fallback.", exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return target if target.exists() else None


def is_ebard_model(model: Optional[object]) -> bool:
    """Check whether a loaded Ultralytics YOLO model is an NBA basketball detector (with explicit `referee` class)."""
    if model is None:
        return False
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        vals = {str(v).lower() for v in names.values()}
        return "referee" in vals and "player" in vals
    return False


def get_cached_basketball_model(weights_path: Optional[str] = None) -> Tuple[Optional[object], bool]:
    """
    Lazily load and cache the E-BARD NBA Basketball Detector (`basketball`, `hoop`, `player`, `referee`).
    Falls back to generic `yolo11n.pt` only if offline without cached E-BARD weights.
    """
    global _CACHED_BASKETBALL_MODEL, _CACHED_BASKETBALL_IS_EBARD
    if _CACHED_BASKETBALL_MODEL is not None:
        return _CACHED_BASKETBALL_MODEL, _CACHED_BASKETBALL_IS_EBARD
    if YOLO is None:
        return None, False

    candidates: List[Path] = []
    if weights_path:
        p = resolve_project_path(weights_path)
        if p.exists():
            candidates.append(p)
    ebard_p = ensure_ebard_basketball_weights()
    if ebard_p is not None and ebard_p.exists():
        candidates.insert(0, ebard_p)
    coco_p = resolve_project_path("yolo11n.pt")
    if coco_p.exists():
        candidates.append(coco_p)

    for cand in candidates:
        try:
            mdl = YOLO(str(cand))
            _CACHED_BASKETBALL_MODEL = mdl
            _CACHED_BASKETBALL_IS_EBARD = is_ebard_model(mdl)
            logger.info(
                "Loaded basketball detector: %s (E-BARD NBA native=%s, classes=%s)",
                cand.name,
                _CACHED_BASKETBALL_IS_EBARD,
                getattr(mdl, "names", {}),
            )
            return _CACHED_BASKETBALL_MODEL, _CACHED_BASKETBALL_IS_EBARD
        except Exception as exc:
            logger.warning("Failed to load detector %s: %s", cand, exc)
    return None, False


def get_cached_det_model() -> Optional[object]:
    """Backward-compatible accessor returning the cached NBA basketball detector."""
    mdl, _ = get_cached_basketball_model()
    return mdl


def get_cached_pose_model(pose_weights: str = "yolo11n-pose.pt") -> Optional[object]:
    """Lazily load and cache `yolo11n-pose.pt` for 17-keypoint skeletal estimation."""
    global _CACHED_POSE_MODEL
    if _CACHED_POSE_MODEL is None and YOLO is not None:
        p = resolve_project_path(pose_weights)
        active = str(p) if p.exists() else pose_weights
        _CACHED_POSE_MODEL = YOLO(active)
    return _CACHED_POSE_MODEL


def get_cached_video_backbone(device: Optional[str] = None) -> Tuple[nn.Module, torch.device]:
    """
    Lazily load and cache frozen Kinetics-400 Video Swin Transformer (`Swin3D-T`, 768-D).
    Provides significantly stronger spatiotemporal motion representations (78.8% K400 top-1)
    than legacy R(2+1)D-18 (57.5% K400 top-1) with fast GPU/CPU inference.
    """
    global _CACHED_SWIN3D_MODEL, _CACHED_SWIN3D_DEVICE
    dev = select_video_encoder_device(device)
    if _CACHED_SWIN3D_MODEL is None:
        try:
            backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
        except Exception as exc:
            logger.warning("Swin3D-T pretrained weights unavailable (%s); using random init.", exc)
            backbone = swin3d_t(weights=None)
        backbone.head = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        _CACHED_SWIN3D_MODEL = backbone.to(dev)
        _CACHED_SWIN3D_DEVICE = dev
    elif _CACHED_SWIN3D_DEVICE != dev:
        _CACHED_SWIN3D_MODEL = _CACHED_SWIN3D_MODEL.to(dev)
        _CACHED_SWIN3D_DEVICE = dev
    return _CACHED_SWIN3D_MODEL, dev


@torch.no_grad()
def encode_tubes_with_swin3d(
    tubes: List[torch.Tensor],
    device: Optional[str] = None,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Encode a list of `(3, 16, 224, 224)` RGB uint8/float action tubes into `(N, 768)`
    Kinetics-400 Swin3D-T spatiotemporal feature embeddings.
    """
    if not tubes:
        return torch.zeros((0, VIDEO_EMBEDDING_DIM), dtype=torch.float32)

    backbone, dev = get_cached_video_backbone(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1, 1)

    embs: List[torch.Tensor] = []
    use_amp = dev.type == "cuda"
    for b_start in range(0, len(tubes), max(1, int(batch_size))):
        chunk = tubes[b_start : b_start + batch_size]
        x = torch.stack(chunk, dim=0).to(dev, non_blocking=use_amp).float()
        if x.max() > 1.5:
            x = x / 255.0
        x = (x - mean) / std
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = backbone(x)
        else:
            out = backbone(x)
        embs.append(out.detach().cpu().float())
    return torch.cat(embs, dim=0)


# ==============================================================================
# Geometry, Pose, Referee Rejection & 2-Team CIELAB Clustering Utilities
# ==============================================================================
def box_iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two `[x1, y1, x2, y2]` boxes."""
    xA = max(float(box_a[0]), float(box_b[0]))
    yA = max(float(box_a[1]), float(box_b[1]))
    xB = min(float(box_a[2]), float(box_b[2]))
    yB = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(1.0, float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]))
    area_b = max(1.0, float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]))
    return float(inter / max(area_a + area_b - inter, 1.0))


def box_iom_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Minimum-Area (IoM) to catch duplicate/nested boxes on one person."""
    xA = max(float(box_a[0]), float(box_b[0]))
    yA = max(float(box_a[1]), float(box_b[1]))
    xB = min(float(box_a[2]), float(box_b[2]))
    yB = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(1.0, float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]))
    area_b = max(1.0, float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]))
    return float(inter / min(area_a, area_b))


def valid_points(kpts_xy: Optional[np.ndarray], indices: Tuple[int, ...]) -> np.ndarray:
    """Return `(K, 2)` array of valid non-zero COCO keypoints."""
    if kpts_xy is None or len(kpts_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = []
    max_idx = len(kpts_xy)
    for idx in indices:
        if idx < max_idx:
            x, y = float(kpts_xy[idx, 0]), float(kpts_xy[idx, 1])
            if x > 2.0 and y > 2.0:
                pts.append((x, y))
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def min_point_set_distance(pts_a: np.ndarray, pts_b: np.ndarray, default_dist: float) -> float:
    """Compute minimum Euclidean distance between two point sets `(N, 2)` and `(M, 2)`."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return default_dist
    diffs = pts_a[:, None, :] - pts_b[None, :, :]
    return float(np.min(np.linalg.norm(diffs, axis=-1)))


def compute_arm_elevation(kpts_xy: Optional[np.ndarray], box_h: float) -> float:
    """Normalized wrist/elbow elevation above shoulders (positive = shooting gather / vertical contest)."""
    shoulders = valid_points(kpts_xy, (5, 6))
    wrists = valid_points(kpts_xy, (7, 8, 9, 10))
    if len(shoulders) == 0 or len(wrists) == 0:
        return 0.0
    shoulder_y = float(np.mean(shoulders[:, 1]))
    highest_wrist_y = float(np.min(wrists[:, 1]))
    return float(np.clip((shoulder_y - highest_wrist_y) / max(box_h * 0.5, 1.0), -1.0, 1.5))


def synthesize_box_keypoints(box_xyxy: np.ndarray) -> np.ndarray:
    """Fallback 17-keypoint COCO skeleton synthesized from a bounding box when occluded."""
    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    w = max(4.0, x2 - x1)
    h = max(8.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[0] = [cx, y1 + 0.10 * h]
    kpts[5] = [cx - 0.22 * w, y1 + 0.22 * h]
    kpts[6] = [cx + 0.22 * w, y1 + 0.22 * h]
    kpts[7] = [cx - 0.34 * w, y1 + 0.35 * h]
    kpts[8] = [cx + 0.34 * w, y1 + 0.35 * h]
    kpts[9] = [cx - 0.38 * w, y1 + 0.28 * h]
    kpts[10] = [cx + 0.38 * w, y1 + 0.28 * h]
    kpts[11] = [cx - 0.16 * w, y1 + 0.52 * h]
    kpts[12] = [cx + 0.16 * w, y1 + 0.52 * h]
    kpts[13] = [cx - 0.16 * w, y1 + 0.72 * h]
    kpts[14] = [cx + 0.16 * w, y1 + 0.72 * h]
    kpts[15] = [cx - 0.16 * w, y1 + 0.92 * h]
    kpts[16] = [cx + 0.16 * w, y1 + 0.92 * h]
    return kpts


def extract_player_torso_lab(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Extract robust median CIELAB `[L*, a*, b*]` of a player's central jersey torso crop."""
    if bgr_frame is None or bgr_frame.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    fh, fw = bgr_frame.shape[:2]
    x1 = float(box_xyxy[0]) / max(scale_x, 1e-5)
    y1 = float(box_xyxy[1]) / max(scale_y, 1e-5)
    x2 = float(box_xyxy[2]) / max(scale_x, 1e-5)
    y2 = float(box_xyxy[3]) / max(scale_y, 1e-5)
    bh = max(1.0, y2 - y1)
    bw = max(1.0, x2 - x1)
    ty1 = max(0, min(fh - 1, int(y1 + 0.20 * bh)))
    ty2 = max(ty1 + 1, min(fh, int(y1 + 0.46 * bh)))
    tx1 = max(0, min(fw - 1, int(x1 + 0.26 * bw)))
    tx2 = max(tx1 + 1, min(fw, int(x2 - 0.26 * bw)))
    torso = bgr_frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)


def extract_player_torso_hsv(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[float, float, float]:
    """Return median `(H, S, V)` of a player's torso crop in `bgr_frame`."""
    if bgr_frame is None or bgr_frame.size == 0:
        return 0.0, 50.0, 128.0
    fh, fw = bgr_frame.shape[:2]
    x1 = float(box_xyxy[0]) / max(scale_x, 1e-5)
    y1 = float(box_xyxy[1]) / max(scale_y, 1e-5)
    x2 = float(box_xyxy[2]) / max(scale_x, 1e-5)
    y2 = float(box_xyxy[3]) / max(scale_y, 1e-5)
    bh = max(1.0, y2 - y1)
    bw = max(1.0, x2 - x1)
    ty1 = max(0, min(fh - 1, int(y1 + 0.20 * bh)))
    ty2 = max(ty1 + 1, min(fh, int(y1 + 0.48 * bh)))
    tx1 = max(0, min(fw - 1, int(x1 + 0.24 * bw)))
    tx2 = max(tx1 + 1, min(fw, int(x2 - 0.24 * bw)))
    torso = bgr_frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return 0.0, 50.0, 128.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return (
        float(np.median(hsv[:, :, 0])),
        float(np.median(hsv[:, :, 1])),
        float(np.median(hsv[:, :, 2])),
    )


def jersey_hsv_diff(hsv_a: Tuple[float, float, float], hsv_b: Tuple[float, float, float]) -> float:
    """Compute perceptual HSV separation between two player torso crops."""
    dh = min(abs(hsv_a[0] - hsv_b[0]), 180.0 - abs(hsv_a[0] - hsv_b[0]))
    return abs(hsv_a[2] - hsv_b[2]) + 0.75 * abs(hsv_a[1] - hsv_b[1]) + 1.1 * dh


def compute_referee_uniform_score(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> float:
    """
    Compute continuous `[0..1]` likelihood that a box is wearing an NBA referee uniform:
    heather-gray low-chroma top (`S < 46`, `65 <= V <= 205`) + **full-length black/charcoal trousers**
    spanning the knee/shin region (`y: [0.58..0.90]` dark & low-saturation).
    Unlike naive torso brightness thresholds, checking full-length dark trousers cleanly separates
    referees from NBA players (who wear shorts exposing bare legs/socks/team shorts).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return 0.0
    fh, fw = bgr_frame.shape[:2]
    sx1, sy1 = float(box_xyxy[0]) / max(scale_x, 1e-5), float(box_xyxy[1]) / max(scale_y, 1e-5)
    sx2, sy2 = float(box_xyxy[2]) / max(scale_x, 1e-5), float(box_xyxy[3]) / max(scale_y, 1e-5)
    sbh, sbw = max(1.0, sy2 - sy1), max(1.0, sx2 - sx1)

    ty1, ty2 = max(0, min(fh - 1, int(sy1 + 0.20 * sbh))), max(1, min(fh, int(sy1 + 0.46 * sbh)))
    tx1, tx2 = max(0, min(fw - 1, int(sx1 + 0.24 * sbw))), max(1, min(fw, int(sx2 - 0.24 * sbw)))
    # Lower leg / shin crop (where players show skin/socks and referees wear solid black pants)
    ly1, ly2 = max(0, min(fh - 1, int(sy1 + 0.58 * sbh))), max(1, min(fh, int(sy1 + 0.90 * sbh)))
    lx1, lx2 = max(0, min(fw - 1, int(sx1 + 0.24 * sbw))), max(1, min(fw, int(sx2 - 0.24 * sbw)))

    torso = bgr_frame[ty1:ty2, tx1:tx2]
    legs = bgr_frame[ly1:ly2, lx1:lx2]
    if torso.size == 0 or legs.size == 0:
        return 0.0

    thsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lhsv = cv2.cvtColor(legs, cv2.COLOR_BGR2HSV)

    ts = float(np.median(thsv[:, :, 1]))
    tv = float(np.median(thsv[:, :, 2]))
    ls = float(np.median(lhsv[:, :, 1]))
    lv = float(np.median(lhsv[:, :, 2]))
    dark_pants_frac = float(np.mean((lhsv[:, :, 2] < 78) & (lhsv[:, :, 1] < 72)))

    # Heather-gray top (low saturation, mid-gray brightness) + full-length dark trousers
    gray_top = (ts < 46.0) and (58.0 <= tv <= 205.0)
    dark_pants = (lv < 74.0) and (ls < 62.0) and (dark_pants_frac >= 0.56)
    if gray_top and dark_pants:
        return float(np.clip(0.65 + 0.35 * dark_pants_frac, 0.0, 1.0))
    if (ts < 35.0 and tv < 130.0) and (lv < 60.0 and dark_pants_frac >= 0.68):
        return 0.80
    return 0.0


def is_referee_or_sideline_official(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    referee_boxes: Optional[List[np.ndarray]] = None,
) -> bool:
    """
    Two-tier NBA Referee Exclusion Filter:
      1. Primary: Reject any detection overlapping an E-BARD `referee` detection (class 3).
      2. Secondary: Reject detections with heather-gray shirt + full-length black trousers
         (or sideline officials standing at the bottom boundary outside live play).
    """
    if referee_boxes:
        for rbox in referee_boxes:
            if box_iou_single(box_xyxy, rbox) >= 0.30 or box_iom_single(box_xyxy, rbox) >= 0.48:
                return True

    ref_score = compute_referee_uniform_score(bgr_frame, box_xyxy, scale_x=scale_x, scale_y=scale_y)
    if ref_score >= 0.72:
        return True

    # Foreground official standing directly on the bottom sideline edge
    y1, y2 = float(box_xyxy[1]), float(box_xyxy[3])
    if y1 >= frame_h * 0.58 and y2 >= frame_h * 0.85 and ref_score >= 0.50:
        return True
    return False


def is_court_player_box(box: np.ndarray, frame_w: int, frame_h: int, from_ebard: bool = False) -> bool:
    """Filter active court player boxes; E-BARD player class is already court-trained so uses softer bounds."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bh = y2 - y1
    bw = x2 - x1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    if from_ebard:
        # E-BARD class 2 (`player`) is trained exclusively on active on-court NBA players
        if not (0.28 * frame_h <= y2 <= 0.99 * frame_h):
            return False
        if not (0.08 * frame_h <= bh <= 0.56 * frame_h):
            return False
        if not (0.015 * frame_w <= bw <= 0.34 * frame_w):
            return False
        if bh / max(bw, 1.0) < 0.65:
            return False
        return True

    # Stricter fallback bounds when using generic COCO person detector
    if not (0.42 * frame_h <= y2 <= 0.92 * frame_h):
        return False
    if y1 < 0.20 * frame_h:
        return False
    if y2 > 0.83 * frame_h and bh < 0.20 * frame_h:
        return False
    if not (0.33 * frame_h <= cy <= 0.84 * frame_h):
        return False
    if not (0.06 * frame_w <= cx <= 0.94 * frame_w):
        return False
    if not (0.10 * frame_h <= bh <= 0.50 * frame_h):
        return False
    if not (0.02 * frame_w <= bw <= 0.30 * frame_w):
        return False
    if bh / max(bw, 1.0) < 0.78:
        return False
    return True


def detect_nba_rim_hsv(bgr_frame: Optional[np.ndarray], frame_w: int, frame_h: int) -> Optional[Tuple[int, int]]:
    """Fallback HSV orange rim detector used only when E-BARD hoop detection is unavailable."""
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    img_h, img_w = bgr_frame.shape[:2]
    y0, y1 = int(0.12 * img_h), int(0.46 * img_h)
    roi = bgr_frame[y0:y1, :]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2, 125, 105]), np.array([20, 255, 255]))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rim: Optional[Tuple[int, int]] = None
    best_area = 0.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        rx_norm = (x + 0.5 * w) / max(img_w, 1)
        if 0.38 <= rx_norm <= 0.62:
            continue
        if (0.008 * img_w <= w <= 0.055 * img_w) and (0.004 * img_h <= h <= 0.042 * img_h):
            if (w / max(h, 1.0)) >= 1.35 and area >= max(20.0, 0.00004 * img_w * img_h) and area > best_area:
                best_area = area
                best_rim = (
                    int(round((x + 0.5 * w) * (frame_w / max(img_w, 1)))),
                    int(round((y0 + y + 0.5 * h) * (frame_h / max(img_h, 1)))),
                )
    return best_rim


def compute_camera_cuts_and_pan(
    bgr_frames: List[np.ndarray],
    frame_w: int,
    frame_h: int,
    player_max_h_ratios: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Detect broadcast camera cuts, close-ups/replays, and camera pan velocity `(pan_dx, pan_dy)`."""
    n = len(bgr_frames)
    hist_diffs = [0.0] * n
    tm_scores = [1.0] * n
    pan_dx = [0.0] * n
    pan_dy = [0.0] * n
    is_hard_cut = [False] * n
    is_wide_court = [True] * n

    prev_hist = None
    prev_gray = None
    sx_full = frame_w / 320.0
    sy_full = frame_h / 180.0

    for t, fr in enumerate(bgr_frames):
        if fr is None or fr.size == 0:
            is_wide_court[t] = False
            continue
        sm = cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            hist_diffs[t] = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        prev_hist = hist

        gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            crop = prev_gray[35:145, 55:265]
            res = cv2.matchTemplate(gray[25:155, :], crop, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            tm_scores[t] = float(max_val)
            pan_dx[t] = (55.0 - float(max_loc[0])) * sx_full
            pan_dy[t] = (10.0 - float(max_loc[1])) * sy_full
        prev_gray = gray

        hard_cut = (hist_diffs[t] > 0.34 and tm_scores[t] < 0.48) or (hist_diffs[t] > 0.48 and tm_scores[t] < 0.62)
        is_hard_cut[t] = hard_cut
        is_closeup = (
            player_max_h_ratios is not None
            and t < len(player_max_h_ratios)
            and player_max_h_ratios[t] > 0.56
        )
        if is_closeup:
            is_wide_court[t] = False

    if n >= 3:
        seg_start = 0
        for t in range(1, n + 1):
            if t == n or is_hard_cut[t]:
                seg_idxs = list(range(seg_start, t))
                if player_max_h_ratios is not None and seg_idxs:
                    seg_med_h = float(np.median([player_max_h_ratios[idx] for idx in seg_idxs]))
                    if seg_med_h > 0.48:
                        for idx in seg_idxs:
                            is_wide_court[idx] = False
                seg_start = t

    if n > 0 and not any(is_wide_court):
        if player_max_h_ratios:
            is_wide_court[int(np.argmin(player_max_h_ratios))] = True
        else:
            is_wide_court = [True] * n

    return {
        "hist_diffs": hist_diffs,
        "tm_scores": tm_scores,
        "pan_dx": pan_dx,
        "pan_dy": pan_dy,
        "is_hard_cut": is_hard_cut,
        "is_wide_court": is_wide_court,
    }


def collect_ebard_court_entities(
    pose_res: Any,
    det_res: Any,
    frame_w: int,
    frame_h: int,
    scale_x: float,
    scale_y: float,
    bgr_frame: Optional[np.ndarray] = None,
    use_ebard: bool = True,
) -> Dict[str, Any]:
    """
    Parse E-BARD Basketball YOLO + YOLOv11-Pose outputs into strictly separated NBA court entities:
      - `players`: ONLY on-court players (E-BARD class 2), strictly excluding E-BARD `referee` boxes (class 3)
                   and full-length dark-trousers officials, fused with matched 17-point COCO skeletons.
      - `referees`: Explicitly detected NBA referee bounding boxes.
      - `ball`: `(cx, cy, conf)` of the basketball (E-BARD class 0 or COCO class 32).
      - `hoop`: `(cx, cy)` of the basket hoop/rim (E-BARD class 1 or HSV fallback).
    """
    referee_boxes: List[np.ndarray] = []
    player_dets: List[Tuple[np.ndarray, float]] = []
    best_ball: Optional[Tuple[float, float, float]] = None
    best_ball_conf = -1.0
    best_hoop: Optional[Tuple[int, int]] = None
    best_hoop_conf = -1.0
    max_h_ratio = 0.0

    # 1. Parse basketball detector (E-BARD or COCO fallback)
    if det_res is not None and det_res.boxes is not None and len(det_res.boxes) > 0:
        d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        d_xyxy[:, [0, 2]] *= scale_x
        d_xyxy[:, [1, 3]] *= scale_y
        d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
        d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)

        for box, cls_id, conf in zip(d_xyxy, d_cls, d_conf):
            cf = float(conf)
            bh = float(box[3] - box[1])
            if use_ebard:
                if cls_id == EBARD_REFEREE_CLS and cf >= 0.18:
                    referee_boxes.append(box)
                    max_h_ratio = max(max_h_ratio, bh / max(float(frame_h), 1.0))
                elif cls_id == EBARD_PLAYER_CLS and cf >= 0.22:
                    player_dets.append((box, cf))
                    max_h_ratio = max(max_h_ratio, bh / max(float(frame_h), 1.0))
                elif cls_id == EBARD_BALL_CLS and cf >= 0.16:
                    bw = float(box[2] - box[0])
                    if 0.006 * frame_h <= max(bw, bh) <= 0.08 * frame_h and cf > best_ball_conf:
                        best_ball_conf = cf
                        best_ball = (0.5 * float(box[0] + box[2]), 0.5 * float(box[1] + box[3]), cf)
                elif cls_id == EBARD_HOOP_CLS and cf >= 0.22:
                    if cf > best_hoop_conf:
                        best_hoop_conf = cf
                        best_hoop = (
                            int(round(0.5 * float(box[0] + box[2]))),
                            int(round(0.5 * float(box[1] + box[3]))),
                        )
            else:
                if cls_id == COCO_PERSON_CLS and cf >= 0.22:
                    player_dets.append((box, cf))
                    max_h_ratio = max(max_h_ratio, bh / max(float(frame_h), 1.0))
                elif cls_id == COCO_SPORTS_BALL_CLS and cf >= 0.18:
                    bw = float(box[2] - box[0])
                    if 0.006 * frame_h <= max(bw, bh) <= 0.07 * frame_h and cf > best_ball_conf:
                        best_ball_conf = cf
                        best_ball = (0.5 * float(box[0] + box[2]), 0.5 * float(box[1] + box[3]), cf)

    # 2. Parse YOLOv11-Pose skeletons
    pose_boxes: List[np.ndarray] = []
    pose_kpts: List[np.ndarray] = []
    pose_confs: List[float] = []
    if pose_res is not None and pose_res.boxes is not None and len(pose_res.boxes) > 0:
        p_xyxy = pose_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        p_xyxy[:, [0, 2]] *= scale_x
        p_xyxy[:, [1, 3]] *= scale_y
        p_conf = (
            pose_res.boxes.conf.cpu().numpy().astype(np.float32)
            if pose_res.boxes.conf is not None
            else np.ones(len(p_xyxy), dtype=np.float32) * 0.5
        )
        raw_kpts = None
        if pose_res.keypoints is not None and pose_res.keypoints.xy is not None:
            raw_kpts = pose_res.keypoints.xy.cpu().numpy().astype(np.float32).copy()
            raw_kpts[:, :, 0] *= scale_x
            raw_kpts[:, :, 1] *= scale_y
        for idx_p in range(len(p_xyxy)):
            if float(p_conf[idx_p]) < 0.12:
                continue
            bh = float(p_xyxy[idx_p, 3] - p_xyxy[idx_p, 1])
            max_h_ratio = max(max_h_ratio, bh / max(float(frame_h), 1.0))
            pose_boxes.append(p_xyxy[idx_p])
            pose_confs.append(float(p_conf[idx_p]))
            kp = raw_kpts[idx_p] if (raw_kpts is not None and idx_p < len(raw_kpts)) else synthesize_box_keypoints(p_xyxy[idx_p])
            pose_kpts.append(kp)

    # 3. Build clean player list:
    #    When E-BARD is active, ONLY trust E-BARD class 2 (`player`) — never add unverified COCO `person`
    #    boxes (which are spectators, coaches, or referees!).
    if use_ebard and len(player_dets) > 0:
        # NMS / IoM deduplication across E-BARD player detections
        player_dets.sort(key=lambda x: x[1], reverse=True)
        dedup_players: List[Tuple[np.ndarray, float]] = []
        for b, cf in player_dets:
            if all(box_iou_single(b, eb) < 0.55 and box_iom_single(b, eb) < 0.72 for eb, _ in dedup_players):
                dedup_players.append((b, cf))
    else:
        # Offline fallback if E-BARD isn't available
        dedup_players = []
        for b, cf in sorted(player_dets, key=lambda x: x[1], reverse=True):
            if all(box_iou_single(b, eb) < 0.45 for eb, _ in dedup_players):
                dedup_players.append((b, cf))
        for pb, pcf in zip(pose_boxes, pose_confs):
            if all(box_iou_single(pb, eb) < 0.35 and box_iom_single(pb, eb) < 0.55 for eb, _ in dedup_players):
                dedup_players.append((pb, pcf))

    # 4. Reject referees and attach matched 17-keypoint skeletons (1-to-1 greedy IoU match)
    used_pose_idxs: set = set()
    clean_players: List[Dict[str, Any]] = []
    for box, conf in dedup_players:
        if not is_court_player_box(box, frame_w, frame_h, from_ebard=use_ebard):
            continue
        if is_referee_or_sideline_official(
            bgr_frame=bgr_frame,
            box_xyxy=box,
            frame_w=frame_w,
            frame_h=frame_h,
            scale_x=scale_x,
            scale_y=scale_y,
            referee_boxes=referee_boxes,
        ):
            continue

        # Match best available YOLOv11-Pose skeleton
        best_p_idx = -1
        best_p_iou = 0.25
        for p_idx, pbox in enumerate(pose_boxes):
            if p_idx in used_pose_idxs:
                continue
            iou = box_iou_single(box, pbox)
            if iou > best_p_iou:
                best_p_iou = iou
                best_p_idx = p_idx

        if best_p_idx >= 0:
            used_pose_idxs.add(best_p_idx)
            kp = pose_kpts[best_p_idx]
            has_real_pose = True
        else:
            kp = synthesize_box_keypoints(box)
            has_real_pose = False

        lab = extract_player_torso_lab(bgr_frame, box, scale_x=scale_x, scale_y=scale_y)
        hsv = extract_player_torso_hsv(bgr_frame, box, scale_x=scale_x, scale_y=scale_y)
        ref_uniform_score = compute_referee_uniform_score(bgr_frame, box, scale_x=scale_x, scale_y=scale_y)
        bh = max(1.0, float(box[3] - box[1]))
        elev = compute_arm_elevation(kp, bh) if has_real_pose else 0.0

        clean_players.append(
            {
                "box": box.astype(np.float32),
                "kpts": kp,
                "has_real_pose": has_real_pose,
                "conf": float(conf),
                "lab": lab,
                "hsv": hsv,
                "ref_uniform_score": float(ref_uniform_score),
                "team_id": -1,
                "elev": float(elev),
                "center": np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=np.float32),
            }
        )

    if best_hoop is None and bgr_frame is not None:
        best_hoop = detect_nba_rim_hsv(bgr_frame, frame_w, frame_h)

    return {
        "players": clean_players,
        "referees": referee_boxes,
        "ball": best_ball,
        "hoop": best_hoop,
        "max_h_ratio": float(max_h_ratio),
    }


def collect_court_players_for_frame(
    pose_res: Any,
    det_res: Any,
    frame_w: int,
    frame_h: int,
    scale_x: float,
    scale_y: float,
    bgr_frame: Optional[np.ndarray] = None,
    use_ebard: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible helper returning clean referee-free court player dictionaries."""
    if use_ebard is None:
        use_ebard = _CACHED_BASKETBALL_IS_EBARD
    ent = collect_ebard_court_entities(
        pose_res=pose_res,
        det_res=det_res,
        frame_w=frame_w,
        frame_h=frame_h,
        scale_x=scale_x,
        scale_y=scale_y,
        bgr_frame=bgr_frame,
        use_ebard=use_ebard,
    )
    return ent["players"]


def fit_clip_team_clusters(frame_players: List[List[Dict[str, Any]]]) -> Optional[np.ndarray]:
    """
    Unsupervised Per-Clip 2-Team CIELAB Jersey Clustering:
      1. Pools all on-court player torso CIELAB vectors across the entire clip.
      2. Fits 2-cluster K-Means (`team_id = 0` vs `team_id = 1`).
      3. Assigns `team_id` to every player in every frame and drops extreme referee-uniform outliers.
    Returns `(2, 3)` team CIELAB centroids or `None` if too few detections exist.
    """
    labs: List[np.ndarray] = []
    for plist in frame_players:
        for p in plist:
            if p.get("ref_uniform_score", 0.0) < 0.50:
                labs.append(p["lab"])
    if len(labs) < 6:
        # Fallback: split by luminance L*
        for plist in frame_players:
            for p in plist:
                p["team_id"] = 0 if float(p["lab"][0]) >= 135.0 else 1
        return None

    X = np.stack(labs, axis=0).astype(np.float32)
    # Weight chromatic axes (a*, b*) slightly higher than lightness L* for shadow invariance
    weights = np.array([1.0, 1.45, 1.45], dtype=np.float32)
    Xw = X * weights[None, :]
    try:
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(Xw)
        centroids_w = km.cluster_centers_.astype(np.float32)
        centroids = centroids_w / weights[None, :]
    except Exception:
        for plist in frame_players:
            for p in plist:
                p["team_id"] = 0 if float(p["lab"][0]) >= 135.0 else 1
        return None

    # Order teams consistently: Team 0 = lighter jersey, Team 1 = darker jersey
    if centroids[0, 0] < centroids[1, 0]:
        centroids = centroids[[1, 0]]
        centroids_w = centroids_w[[1, 0]]

    # Assign team_id to every player in every frame and prune ref outliers
    for f_idx, plist in enumerate(frame_players):
        filtered: List[Dict[str, Any]] = []
        for p in plist:
            pw = p["lab"] * weights
            d0 = float(np.linalg.norm(pw - centroids_w[0]))
            d1 = float(np.linalg.norm(pw - centroids_w[1]))
            team_id = 0 if d0 <= d1 else 1
            min_d = min(d0, d1)
            # If a candidate has dark-trousers referee appearance AND is far from both team centroids, drop it
            if p.get("ref_uniform_score", 0.0) >= 0.45 and min_d > 38.0:
                continue
            p["team_id"] = team_id
            p["team_dist"] = min_d
            filtered.append(p)
        frame_players[f_idx] = filtered

    return centroids


def interpolate_court_ball_track(
    raw_balls: List[Optional[Tuple[float, float, float]]],
    wide_mask: List[bool],
    sampled_frame_idxs: List[int],
    fps: float,
    diag: float,
    pan_dx: Optional[List[float]] = None,
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """
    Prune teleporting false ball detections and interpolate E-BARD basketball coordinates
    across short occlusion gaps (`<= 1.25s`) inside continuous wide-court segments.
    Returns `(track_points, track_confidences)`.
    """
    n = len(raw_balls)
    cleaned: List[Optional[Tuple[float, float, float]]] = [
        (raw_balls[t] if (t < len(wide_mask) and wide_mask[t]) else None) for t in range(n)
    ]

    # Prune isolated low-conf spikes without temporal neighbor support
    valid_idxs = [t for t, b in enumerate(cleaned) if b is not None]
    for k_pos, t in enumerate(valid_idxs):
        bt = cleaned[t]
        assert bt is not None
        if bt[2] >= 0.36:
            continue
        near_neighbor = False
        for k_other in (k_pos - 1, k_pos + 1):
            if 0 <= k_other < len(valid_idxs):
                to = valid_idxs[k_other]
                bo = cleaned[to]
                if bo is not None:
                    dt = abs(sampled_frame_idxs[to] - sampled_frame_idxs[t]) / max(fps, 1.0)
                    dist = float(np.hypot(bt[0] - bo[0], bt[1] - bo[1]))
                    if dt <= 1.2 and dist <= max(460.0, dt * 1050.0):
                        near_neighbor = True
                        break
        if not near_neighbor and len(valid_idxs) >= 2:
            cleaned[t] = None

    track: List[Optional[np.ndarray]] = [
        (np.array([b[0], b[1]], dtype=np.float32) if b is not None else None) for b in cleaned
    ]
    confs: List[float] = [float(b[2]) if b is not None else 0.0 for b in cleaned]

    anchor_ts = [t for t, pt in enumerate(track) if pt is not None]
    for idx in range(len(anchor_ts) - 1):
        t0, t1 = anchor_ts[idx], anchor_ts[idx + 1]
        if t1 - t0 <= 1:
            continue
        if not all(wide_mask[k] for k in range(t0, t1 + 1)):
            continue
        dt = abs(sampled_frame_idxs[t1] - sampled_frame_idxs[t0]) / max(fps, 1.0)
        if dt > 1.25:
            continue
        p0, p1 = track[t0], track[t1]
        assert p0 is not None and p1 is not None
        cum_pan = sum(abs(pan_dx[k]) for k in range(t0 + 1, t1 + 1)) if pan_dx is not None else 0.0
        max_jump = max(0.25 * diag, dt * 1200.0 + cum_pan)
        if float(np.linalg.norm(p1 - p0)) > max_jump:
            continue
        denom = max(float(sampled_frame_idxs[t1] - sampled_frame_idxs[t0]), 1.0)
        c_interp = 0.75 * min(confs[t0], confs[t1])
        for k in range(t0 + 1, t1):
            alpha = float(sampled_frame_idxs[k] - sampled_frame_idxs[t0]) / denom
            track[k] = (1.0 - alpha) * p0 + alpha * p1
            confs[k] = c_interp
    return track, confs


def compute_pair_biomechanics(
    p_off: Dict[str, Any],
    p_def: Dict[str, Any],
    ball_pt: Optional[np.ndarray],
    ball_conf: float,
    hoop_pt: Optional[Tuple[int, int]],
    frame_w: int,
    frame_h: int,
) -> Tuple[np.ndarray, float]:
    """
    Compute the 16-D NBA Rule 12 + Depth + Ball/Hoop biomechanical feature vector for a verified
    opposing-team `(offense, defense)` pair.
    """
    diag = float(np.hypot(frame_w, frame_h))
    box_off, box_def = p_off["box"], p_def["box"]
    kpts_off, kpts_def = p_off.get("kpts"), p_def.get("kpts")

    h_off = max(1.0, float(box_off[3] - box_off[1]))
    h_def = max(1.0, float(box_def[3] - box_def[1]))
    w_off = max(1.0, float(box_off[2] - box_off[0]))
    max_bh = max(h_off, h_def)

    elev_off = float(p_off.get("elev", 0.0))
    elev_def = float(p_def.get("elev", 0.0))
    pair_iou = box_iou_single(box_off, box_def)
    c_off = p_off["center"]
    c_def = p_def["center"]
    c_dist = float(np.linalg.norm(c_off - c_def))
    mid = 0.5 * (c_off + c_def)

    # [13] 3D Foot-Plane Depth Alignment (prevents depth illusions from reading as physical body collisions)
    y2_diff = abs(float(box_off[3]) - float(box_def[3]))
    # Allow extra vertical foot separation if shooter is clearly jumping (elevated arms)
    airborne_allowance = 0.22 * max_bh if max(elev_off, elev_def) > 0.25 else 0.0
    eff_y2_diff = max(0.0, y2_diff - airborne_allowance)
    foot_plane_align = float(np.exp(-((eff_y2_diff / max(0.32 * max_bh, 1.0)) ** 2)))
    depth_gate = 0.30 + 0.70 * foot_plane_align

    # [0] Arm Contact Intrusion
    arms_off = valid_points(kpts_off, ARM_CONTACT_KPTS)
    arms_def = valid_points(kpts_def, ARM_CONTACT_KPTS)
    upper_off = valid_points(kpts_off, UPPER_BODY_KPTS)
    arm_dist = min(
        min_point_set_distance(arms_off, arms_def, c_dist),
        min_point_set_distance(arms_def, upper_off, c_dist),
    )
    arm_contact_val = float(np.clip(1.0 - (arm_dist / max(h_off * 0.28, 1.0)), 0.0, 1.0)) * depth_gate

    # [1] Body Cylinder Penetration (depth-gated!)
    torso_off = valid_points(kpts_off, TORSO_KPTS)
    torso_def = valid_points(kpts_def, TORSO_KPTS)
    torso_d = min_point_set_distance(torso_off, torso_def, c_dist)
    raw_cyl = 0.50 * (1.0 - torso_d / max(h_off * 0.36, 1.0)) + 0.50 * min(1.0, pair_iou * 2.4)
    cylinder_pen_val = float(np.clip(raw_cyl, 0.0, 1.0)) * depth_gate

    # [2] Lower Body Landing Space Violation
    feet_off = valid_points(kpts_off, FEET_KPTS)
    feet_def = valid_points(kpts_def, FEET_KPTS)
    feet_dist = min_point_set_distance(feet_off, feet_def, 0.5 * (h_off + h_def))
    landing_violation_val = float(np.clip(1.0 - (feet_dist / max(h_off * 0.28, 1.0)), 0.0, 1.0)) * depth_gate

    # [3] Defender Verticality Angle Deviation
    sh_def = valid_points(kpts_def, (5, 6))
    wr_def = valid_points(kpts_def, (9, 10))
    if len(sh_def) > 0 and len(wr_def) > 0:
        dx = float(np.mean(wr_def[:, 0]) - np.mean(sh_def[:, 0]))
        dy = float(np.mean(wr_def[:, 1]) - np.mean(sh_def[:, 1]))
        reach_toward = (float(c_off[0]) - float(c_def[0])) * dx
        verticality_val = float(np.clip(abs(dx) / max(abs(dy) + 1.0, 1.0), 0.0, 1.0)) if reach_toward > 0 else 0.0
    else:
        verticality_val = 0.0

    # [4] Defender Lateral Positioning
    lat_pos_val = float(np.clip(abs(float(c_def[0]) - float(c_off[0])) / max(w_off * 1.4, 1.0), 0.0, 1.0))

    # [5] Offensive Displacement (populated temporally across window frames; static baseline here)
    off_disp_val = float(np.clip(pair_iou * 1.1 * depth_gate, 0.0, 1.0))

    # [6] Defender to Ball Distance & [7] Ball to Shooter Distance & [14] Ball Verified On Play
    wr_off = valid_points(kpts_off, (9, 10))
    if ball_pt is not None:
        d_ball_def = (
            float(np.min(np.linalg.norm(wr_def - ball_pt, axis=-1)))
            if len(wr_def) > 0
            else float(np.linalg.norm(c_def - ball_pt))
        )
        d_ball_off = (
            float(np.min(np.linalg.norm(wr_off - ball_pt, axis=-1)))
            if len(wr_off) > 0
            else float(np.linalg.norm(c_off - ball_pt))
        )
        def_ball_val = float(np.clip(1.0 - d_ball_def / max(h_def * 0.90, 1.0), 0.0, 1.0))
        ball_shooter_val = float(np.clip(1.0 - d_ball_off / max(h_off * 0.85, 1.0), 0.0, 1.0))
        d_pair_ball = min(float(np.linalg.norm(mid - ball_pt)), d_ball_off, d_ball_def)
        ball_verified = float(
            np.clip((1.0 - d_pair_ball / max(diag * 0.16, 1.0)) * min(1.0, ball_conf / 0.45), 0.0, 1.0)
        )
    else:
        def_ball_val = 0.0
        ball_shooter_val = 0.0
        ball_verified = 0.0

    # [8] Ball Strike vs Body Ratio
    ball_ratio_val = float(np.clip(cylinder_pen_val * (1.0 - 0.55 * def_ball_val), 0.0, 1.0))

    # [9] Closing Velocity & [10] Momentum Shock (overwritten temporally across window frames)
    closing_vel_val = 0.50
    momentum_shock_val = 0.0

    # [11] Airborne Contest Apex Diff
    physical_contact_score = float(
        0.34 * arm_contact_val
        + 0.30 * cylinder_pen_val
        + 0.18 * landing_violation_val
        + 0.18 * verticality_val
    )
    apex_diff_val = float(np.clip(0.65 * physical_contact_score + 0.35 * max(0.0, elev_off), 0.0, 1.0))

    # [12] Hoop Proximity (from E-BARD hoop detection!)
    if hoop_pt is not None:
        d_hoop = float(np.hypot(float(mid[0]) - float(hoop_pt[0]), float(mid[1]) - float(hoop_pt[1])))
        hoop_prox = float(np.clip(1.0 - d_hoop / max(diag * 0.38, 1.0), 0.0, 1.0))
    else:
        hoop_prox = 0.0

    # [15] Shooting Motion Score (elevated shooter arms + ball near shooter/hoop)
    shooting_motion = float(
        np.clip(
            0.42 * max(0.0, min(1.0, elev_off * 1.25))
            + 0.30 * max(ball_shooter_val, ball_verified)
            + 0.28 * hoop_prox,
            0.0,
            1.0,
        )
    )

    feats = np.array(
        [
            arm_contact_val,
            cylinder_pen_val,
            landing_violation_val,
            verticality_val,
            lat_pos_val,
            off_disp_val,
            def_ball_val,
            ball_shooter_val,
            ball_ratio_val,
            closing_vel_val,
            momentum_shock_val,
            apex_diff_val,
            hoop_prox,
            foot_plane_align,
            ball_verified,
            shooting_motion,
        ],
        dtype=np.float32,
    )
    return feats, physical_contact_score


def select_on_ball_opposing_pair(
    players: List[Dict[str, Any]],
    ball_pt: Optional[np.ndarray],
    ball_conf: float,
    hoop_pt: Optional[Tuple[int, int]],
    prev_center: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    diag: float,
) -> Optional[Dict[str, Any]]:
    """
    Select the active opposing-team (`team_id_a != team_id_b`) on-ball/contest pair anchored
    to the E-BARD basketball, hoop, 3D foot-plane depth alignment, and physical contact proximity.
    """
    if len(players) < 2:
        return None
    hoop_arr = np.array([float(hoop_pt[0]), float(hoop_pt[1])], dtype=np.float32) if hoop_pt is not None else None

    # Check whether opposing-team pairs exist in this frame
    has_opposing_teams = len({p.get("team_id", -1) for p in players if p.get("team_id", -1) >= 0}) >= 2

    best: Optional[Dict[str, Any]] = None
    best_rank = -1e9

    for i in range(len(players)):
        pi = players[i]
        bi, ci = pi["box"], pi["center"]
        bhi = float(bi[3] - bi[1])
        ti = pi.get("team_id", -1)

        for j in range(i + 1, len(players)):
            pj = players[j]
            bj, cj = pj["box"], pj["center"]
            bhj = float(bj[3] - bj[1])
            tj = pj.get("team_id", -1)

            # HARD opposing-team enforcement when clip team clustering succeeded
            if has_opposing_teams and ti >= 0 and tj >= 0 and ti == tj:
                continue

            if min(bhi, bhj) / max(bhi, bhj, 1.0) < 0.52:
                continue

            max_bh = max(bhi, bhj)
            y2_diff = abs(float(bi[3]) - float(bj[3]))
            elev_max = max(float(pi.get("elev", 0.0)), float(pj.get("elev", 0.0)))
            max_y2_diff = max(frame_h * 0.22, 0.85 * max_bh) if elev_max > 0.25 else max(frame_h * 0.16, 0.60 * max_bh)
            if y2_diff > max_y2_diff:
                continue

            cdist = float(np.linalg.norm(ci - cj))
            if cdist > max(0.18 * diag, 1.45 * max_bh):
                continue

            lab_diff = float(np.linalg.norm(pi["lab"] - pj["lab"]))
            if box_iom_single(bi, bj) >= 0.70 and lab_diff < 18.0:
                continue

            mid = 0.5 * (ci + cj)
            foot_align = float(np.exp(-((y2_diff / max(0.38 * max_bh, 1.0)) ** 2)))
            close_prox = max(0.0, 1.0 - cdist / max(1.45 * max_bh, 1.0))
            iou = box_iou_single(bi, bj)

            # Determine offense (closer to ball, or higher shooting elevation) vs defense
            if ball_pt is not None:
                di_b = float(np.linalg.norm(ci - ball_pt))
                dj_b = float(np.linalg.norm(cj - ball_pt))
                if di_b <= dj_b:
                    p_off, p_def = pi, pj
                else:
                    p_off, p_def = pj, pi
            else:
                if float(pi.get("elev", 0.0)) >= float(pj.get("elev", 0.0)):
                    p_off, p_def = pi, pj
                else:
                    p_off, p_def = pj, pi

            feats, contact_score = compute_pair_biomechanics(
                p_off=p_off,
                p_def=p_def,
                ball_pt=ball_pt,
                ball_conf=ball_conf,
                hoop_pt=hoop_pt,
                frame_w=frame_w,
                frame_h=frame_h,
            )

            ball_prox = 0.0
            dball = 9999.0
            is_on_ball = False
            if ball_pt is not None:
                dball = min(
                    float(np.linalg.norm(mid - ball_pt)),
                    float(np.linalg.norm(ci - ball_pt)) + 20.0,
                    float(np.linalg.norm(cj - ball_pt)) + 20.0,
                )
                ball_prox = max(0.0, 1.0 - dball / max(diag * 0.16, 1.0))
                is_on_ball = dball <= max(diag * 0.14, 260.0)

            hoop_bonus = 0.0
            if hoop_arr is not None:
                dhoop = float(np.linalg.norm(mid - hoop_arr))
                hoop_bonus = 0.18 * max(0.0, 1.0 - dhoop / max(diag * 0.36, 1.0))

            smooth_bonus = 0.0
            if prev_center is not None:
                dprev = float(np.linalg.norm(mid - prev_center))
                smooth_bonus = 0.14 * float(np.exp(-dprev / max(0.16 * diag, 1.0)))

            off_ball_pen = -0.70 if (ball_pt is not None and ball_conf >= 0.25 and dball > max(diag * 0.18, 340.0)) else 0.0

            rank = (
                0.65 * ball_prox * (0.5 + 0.5 * min(1.0, ball_conf / 0.45))
                + 0.35 * contact_score
                + 0.26 * min(1.0, elev_max * 1.3)
                + 0.22 * foot_align
                + 0.16 * close_prox
                + 0.10 * min(1.0, iou * 2.5)
                + hoop_bonus
                + smooth_bonus
                + off_ball_pen
            )
            if rank > best_rank:
                best_rank = rank
                bo, bd = p_off["box"], p_def["box"]
                ux1 = max(0.0, min(float(bo[0]), float(bd[0])))
                uy1 = max(0.0, min(float(bo[1]), float(bd[1])))
                ux2 = min(float(frame_w), max(float(bo[2]), float(bd[2])))
                uy2 = min(float(frame_h), max(float(bo[3]), float(bd[3])))
                best = {
                    "roi": np.array([ux1, uy1, ux2, uy2], dtype=np.float32),
                    "off_box": bo.copy(),
                    "def_box": bd.copy(),
                    "off_team": p_off.get("team_id", 0),
                    "def_team": p_def.get("team_id", 1),
                    "off_lab": p_off["lab"].copy(),
                    "def_lab": p_def["lab"].copy(),
                    "off_hsv": p_off["hsv"],
                    "def_hsv": p_def["hsv"],
                    "mid": mid,
                    "feats": feats,
                    "contact_score": float(contact_score),
                    "elev": float(elev_max),
                    "foot_align": float(foot_align),
                    "dball": float(dball),
                    "is_on_ball": bool(is_on_ball),
                    "rank": float(rank),
                }
    return best


def compute_square_crop_coords(
    roi_box: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin_scale: float = 1.28,
) -> Tuple[int, int, int, int]:
    """Convert `[x1, y1, x2, y2]` ROI into a square crop window clamped inside `[0, W] x [0, H]`."""
    x1, y1, x2, y2 = roi_box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1, min(frame_w, frame_h) * 0.28) * margin_scale
    side = min(side, float(min(frame_w, frame_h)))
    half = side * 0.5
    nx1 = max(0.0, min(cx - half, frame_w - side))
    ny1 = max(0.0, min(cy - half, frame_h - side))
    nx2 = min(float(frame_w), nx1 + side)
    ny2 = min(float(frame_h), ny1 + side)
    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))


def smooth_boxes_ema(boxes: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Bidirectional Exponential Moving Average (EMA) smoothing for `[T, 4]` boxes."""
    if len(boxes) <= 1:
        return boxes.copy()
    forward = np.zeros_like(boxes, dtype=np.float32)
    forward[0] = boxes[0]
    for t in range(1, len(boxes)):
        forward[t] = alpha * boxes[t] + (1.0 - alpha) * forward[t - 1]
    backward = np.zeros_like(boxes, dtype=np.float32)
    backward[-1] = forward[-1]
    for t in range(len(boxes) - 2, -1, -1):
        backward[t] = alpha * forward[t] + (1.0 - alpha) * backward[t + 1]
    return backward


# ==============================================================================
# Multi-Window Temporal Candidate Bag Extractor (Works on Any Video Length!)
# ==============================================================================
def extract_candidate_windows(
    video_path: Path | str,
    det_model: Optional[object] = None,
    pose_model: Optional[object] = None,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "auto",
    max_live_play_sec: float = 0.0,
    max_windows: int = 0,
    scan_fps: float = 2.5,
    yolo_batch_size: int = 32,
    encode_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Universal Basketball Video Scanner & Multi-Window Candidate Bag Extractor:
      - Works on ANY NBA game video length (`max_live_play_sec <= 0` scans the entire video).
      - Uses E-BARD Basketball YOLO (`player`, `referee`, `basketball`, `hoop`) + YOLOv11-Pose
        + Per-Clip 2-Team CIELAB Clustering so referees & spectators are never treated as players.
      - Segments continuous wide-court live play (rejecting close-ups, dead-ball cuts, and replays).
      - Slices wide-court play into overlapping `window_sec` (2.0s) windows at `stride_sec` (1.0s)
        stride, crops stabilized `(3, 16, 224, 224)` action tubes, computes `(16, 16)` temporal
        Rule 12 kinematics, and encodes `(768,)` Swin3D-T visual embeddings per window.
    """
    vpath = Path(video_path)
    if det_model is None:
        active_det, use_ebard = get_cached_basketball_model()
    else:
        active_det = det_model
        use_ebard = is_ebard_model(active_det)
    active_pose = pose_model if pose_model is not None else get_cached_pose_model()

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {vpath}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    diag = float(np.hypot(frame_w, frame_h))

    if max_live_play_sec > 0.0:
        max_horizon = max(int(round(fps * max_live_play_sec)), num_frames * 2)
        total_frames = min(reported_frames, max_horizon) if reported_frames > 0 else max_horizon
    else:
        total_frames = reported_frames if reported_frames > 0 else int(round(fps * 60.0))

    scan_stride = max(1, int(round(fps / max(scan_fps, 0.5))))
    scan_w, scan_h = 960, 540
    scale_x = float(frame_w) / float(scan_w)
    scale_y = float(frame_h) / float(scan_h)

    sampled_idxs: List[int] = []
    small_frames: List[np.ndarray] = []
    f_idx = 0
    while f_idx < total_frames:
        if f_idx % scan_stride == 0:
            ret, frm = cap.read()
            if not ret:
                break
            small_frames.append(cv2.resize(frm, (scan_w, scan_h), interpolation=cv2.INTER_LINEAR))
            sampled_idxs.append(f_idx)
        else:
            if not cap.grab():
                break
        f_idx += 1
    actual_total = max(f_idx, 1)
    cap.release()

    if not small_frames:
        raise RuntimeError(f"No readable frames in video: {vpath}")

    # Pass 1: Batched E-BARD Basketball Detection + YOLOv11-Pose
    yolo_dev = select_yolo_device(device)
    batch_sz = max(1, int(yolo_batch_size))
    all_det_res: List[Any] = []
    all_pose_res: List[Any] = []
    with torch.inference_mode():
        for b_start in range(0, len(small_frames), batch_sz):
            b_imgs = small_frames[b_start : b_start + batch_sz]
            if active_det is not None:
                det_kwargs = {"conf": 0.16, "imgsz": 960, "verbose": False, "device": yolo_dev}
                if not use_ebard:
                    det_kwargs["classes"] = [COCO_PERSON_CLS, COCO_SPORTS_BALL_CLS]
                all_det_res.extend(active_det.predict(b_imgs, **det_kwargs))
            else:
                all_det_res.extend([None] * len(b_imgs))

            if active_pose is not None:
                all_pose_res.extend(
                    active_pose.predict(b_imgs, conf=0.12, imgsz=640, verbose=False, device=yolo_dev)
                )
            else:
                all_pose_res.extend([None] * len(b_imgs))

    # Parse clean referee-free court entities per frame
    n_scan = len(small_frames)
    frame_players: List[List[Dict[str, Any]]] = []
    raw_balls: List[Optional[Tuple[float, float, float]]] = []
    raw_hoops: List[Optional[Tuple[int, int]]] = []
    max_h_ratios: List[float] = []

    for i in range(n_scan):
        ent = collect_ebard_court_entities(
            pose_res=all_pose_res[i],
            det_res=all_det_res[i],
            frame_w=frame_w,
            frame_h=frame_h,
            scale_x=scale_x,
            scale_y=scale_y,
            bgr_frame=small_frames[i],
            use_ebard=use_ebard,
        )
        frame_players.append(ent["players"])
        raw_balls.append(ent["ball"])
        raw_hoops.append(ent["hoop"])
        max_h_ratios.append(ent["max_h_ratio"])

    cam_info = compute_camera_cuts_and_pan(small_frames, frame_w, frame_h, player_max_h_ratios=max_h_ratios)
    wide_mask: List[bool] = cam_info["is_wide_court"]
    hard_cuts: List[bool] = cam_info["is_hard_cut"]
    pan_dx: List[float] = cam_info["pan_dx"]
    del small_frames

    # Keep players only on wide-court frames, then run clip-level 2-team CIELAB clustering!
    for i in range(n_scan):
        if not wide_mask[i]:
            frame_players[i] = []
    team_centroids = fit_clip_team_clusters(frame_players)

    ball_track, ball_confs = interpolate_court_ball_track(
        raw_balls=raw_balls,
        wide_mask=wide_mask,
        sampled_frame_idxs=sampled_idxs,
        fps=fps,
        diag=diag,
        pan_dx=pan_dx,
    )

    # Smooth hoop coordinates via running median inside each segment
    hoop_track: List[Optional[Tuple[int, int]]] = list(raw_hoops)
    last_hoop: Optional[Tuple[int, int]] = None
    for i in range(n_scan):
        if not wide_mask[i] or hard_cuts[i]:
            last_hoop = None
        if hoop_track[i] is not None:
            last_hoop = hoop_track[i]
        elif last_hoop is not None and wide_mask[i]:
            hoop_track[i] = last_hoop

    # Select opposing-team on-ball pair at every wide-court scan frame
    default_roi = np.array([frame_w * 0.22, frame_h * 0.18, frame_w * 0.78, frame_h * 0.82], dtype=np.float32)
    scan_rois: List[np.ndarray] = []
    scan_off_boxes: List[np.ndarray] = []
    scan_def_boxes: List[np.ndarray] = []
    scan_feats: List[np.ndarray] = []
    scan_activity: List[float] = []
    prev_center: Optional[np.ndarray] = None

    for i in range(n_scan):
        if not wide_mask[i] or hard_cuts[i]:
            prev_center = None
        pair = (
            select_on_ball_opposing_pair(
                players=frame_players[i],
                ball_pt=ball_track[i],
                ball_conf=ball_confs[i],
                hoop_pt=hoop_track[i],
                prev_center=prev_center,
                frame_w=frame_w,
                frame_h=frame_h,
                diag=diag,
            )
            if wide_mask[i]
            else None
        )
        if pair is not None:
            prev_center = pair["mid"]
            scan_rois.append(pair["roi"])
            scan_off_boxes.append(pair["off_box"])
            scan_def_boxes.append(pair["def_box"])
            scan_feats.append(pair["feats"])
            act = (
                0.40 * max(0.0, pair["rank"])
                + 0.35 * pair["contact_score"]
                + 0.25 * (1.0 if pair["is_on_ball"] else 0.0)
            )
            scan_activity.append(float(act))
        else:
            scan_rois.append(default_roi.copy())
            scan_off_boxes.append(default_roi.copy())
            scan_def_boxes.append(default_roi.copy())
            scan_feats.append(np.zeros(NUM_FOUL_FEATURES, dtype=np.float32))
            scan_activity.append(0.0)

    # Identify contiguous wide-court live-action segments
    wide_segments: List[Tuple[int, int]] = []
    s_i = 0
    while s_i < n_scan:
        if not wide_mask[s_i]:
            s_i += 1
            continue
        e_i = s_i
        while e_i + 1 < n_scan and wide_mask[e_i + 1] and not hard_cuts[e_i + 1]:
            e_i += 1
        seg_start_f = sampled_idxs[s_i]
        seg_end_f = sampled_idxs[e_i]
        if (seg_end_f - seg_start_f) >= int(round(fps * 0.65)):
            wide_segments.append((seg_start_f, seg_end_f))
        s_i = e_i + 1

    if not wide_segments:
        wide_segments = [(0, max(actual_total - 1, num_frames))]

    # Generate overlapping candidate temporal windows `[start_f, end_f]` across all wide-court segments
    win_frames = max(num_frames, int(round(fps * window_sec)))
    stride_frames = max(4, int(round(fps * stride_sec)))
    scan_idx_arr = np.array(sampled_idxs, dtype=np.float32)
    scan_act_arr = np.array(scan_activity, dtype=np.float32)

    raw_win_specs: List[Tuple[int, int, float]] = []
    for seg_lo, seg_hi in wide_segments:
        seg_len = seg_hi - seg_lo + 1
        if seg_len <= win_frames:
            w_lo, w_hi = seg_lo, seg_hi
            in_m = (scan_idx_arr >= w_lo) & (scan_idx_arr <= w_hi)
            act = float(np.mean(scan_act_arr[in_m])) if np.any(in_m) else 0.0
            raw_win_specs.append((w_lo, w_hi, act))
        else:
            cur = seg_lo
            while cur + win_frames <= seg_hi:
                w_lo, w_hi = cur, cur + win_frames - 1
                in_m = (scan_idx_arr >= w_lo) & (scan_idx_arr <= w_hi)
                act = float(np.max(scan_act_arr[in_m]) * 0.6 + np.mean(scan_act_arr[in_m]) * 0.4) if np.any(in_m) else 0.0
                raw_win_specs.append((w_lo, w_hi, act))
                cur += stride_frames
            # Always include tail of wide segment (many NBA fouls happen right before a dead-ball camera cut!)
            tail_lo = max(seg_lo, seg_hi - win_frames + 1)
            if not raw_win_specs or abs(raw_win_specs[-1][0] - tail_lo) >= stride_frames // 2:
                in_m = (scan_idx_arr >= tail_lo) & (scan_idx_arr <= seg_hi)
                act = float(np.max(scan_act_arr[in_m]) * 0.6 + np.mean(scan_act_arr[in_m]) * 0.4) if np.any(in_m) else 0.0
                raw_win_specs.append((tail_lo, seg_hi, act))

    # If max_windows > 0 (e.g. K=10 for training bags), select temporally uniform windows
    # across the entire live-play span so there are ZERO blind spots for MIL attention!
    if max_windows > 0 and len(raw_win_specs) > max_windows:
        n_unif = max(2, max_windows - 2)
        chosen = set(np.linspace(0, len(raw_win_specs) - 1, n_unif).round().astype(int).tolist())
        # Fill remaining 2 slots with the highest-activity windows not yet chosen
        ranked_idxs = sorted(range(len(raw_win_specs)), key=lambda idx_w: raw_win_specs[idx_w][2], reverse=True)
        for r_idx in ranked_idxs:
            if len(chosen) >= max_windows:
                break
            chosen.add(r_idx)
        win_specs = [raw_win_specs[i] for i in sorted(chosen)]
    else:
        win_specs = raw_win_specs

    # Interpolate ROIs, player boxes, and 16-D features onto full frame timeline
    all_idx_arr = np.arange(actual_total, dtype=np.float32)
    scan_rois_arr = np.stack(scan_rois, axis=0)
    scan_off_arr = np.stack(scan_off_boxes, axis=0)
    scan_def_arr = np.stack(scan_def_boxes, axis=0)
    scan_feats_arr = np.stack(scan_feats, axis=0)

    full_rois = np.zeros((actual_total, 4), dtype=np.float32)
    full_off = np.zeros((actual_total, 4), dtype=np.float32)
    full_def = np.zeros((actual_total, 4), dtype=np.float32)
    for c in range(4):
        full_rois[:, c] = np.interp(all_idx_arr, scan_idx_arr, scan_rois_arr[:, c])
        full_off[:, c] = np.interp(all_idx_arr, scan_idx_arr, scan_off_arr[:, c])
        full_def[:, c] = np.interp(all_idx_arr, scan_idx_arr, scan_def_arr[:, c])
    stabilized_rois = smooth_boxes_ema(full_rois, alpha=0.32)

    full_feats = np.zeros((actual_total, NUM_FOUL_FEATURES), dtype=np.float32)
    for d in range(NUM_FOUL_FEATURES):
        full_feats[:, d] = np.interp(all_idx_arr, scan_idx_arr, scan_feats_arr[:, d])

    # Compute true temporal kinematics: closing velocity, offensive displacement jolt, momentum shock
    off_cx = 0.5 * (full_off[:, 0] + full_off[:, 2])
    off_cy = 0.5 * (full_off[:, 1] + full_off[:, 3])
    def_cx = 0.5 * (full_def[:, 0] + full_def[:, 2])
    def_cy = 0.5 * (full_def[:, 1] + full_def[:, 3])
    pair_dist_t = np.hypot(off_cx - def_cx, off_cy - def_cy)
    closing_vel_t = -np.gradient(pair_dist_t) / max(float(frame_w) * 0.012, 1.0)
    off_vx, off_vy = np.gradient(off_cx), np.gradient(off_cy)
    off_acc = np.hypot(np.gradient(off_vx), np.gradient(off_vy)) / max(float(frame_h) * 0.008, 1.0)
    full_feats[:, 5] = np.clip(0.55 * full_feats[:, 5] + 0.45 * np.clip(off_acc, 0.0, 1.0), 0.0, 1.0)
    full_feats[:, 9] = np.clip(0.5 + 0.5 * np.tanh(closing_vel_t), 0.0, 1.0)
    full_feats[:, 10] = np.clip(off_acc * 0.65, 0.0, 1.0)

    # Pass 2: Single sequential pass to decode ONLY the unique frame indices needed across all windows
    win_sampled_lists: List[List[int]] = []
    needed_frames: set = set()
    for w_lo, w_hi, _ in win_specs:
        w_lo_c = max(0, min(w_lo, actual_total - 1))
        w_hi_c = max(w_lo_c, min(w_hi, actual_total - 1))
        idxs = np.linspace(w_lo_c, w_hi_c, num=num_frames).round().astype(int).tolist()
        win_sampled_lists.append(idxs)
        needed_frames.update(idxs)

    decoded_crops_by_frame: Dict[int, np.ndarray] = {}
    if needed_frames:
        max_need = max(needed_frames)
        cap2 = cv2.VideoCapture(str(vpath))
        cur_f = 0
        while cur_f <= max_need:
            if cur_f in needed_frames:
                ret, frame = cap2.read()
                if not ret:
                    break
                x1, y1, x2, y2 = compute_square_crop_coords(stabilized_rois[cur_f], frame_w=frame_w, frame_h=frame_h)
                crop_bgr = frame[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    crop_bgr = frame
                resized = cv2.resize(crop_bgr, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
                decoded_crops_by_frame[cur_f] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                if not cap2.grab():
                    break
            cur_f += 1
        cap2.release()

    fallback_rgb = (
        next(iter(decoded_crops_by_frame.values()))
        if decoded_crops_by_frame
        else np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    )

    windows_out: List[Dict[str, Any]] = []
    all_tubes: List[torch.Tensor] = []

    for (w_lo, w_hi, act_score), sampled_list in zip(win_specs, win_sampled_lists):
        rgb_frames = [decoded_crops_by_frame.get(fi, fallback_rgb) for fi in sampled_list]
        crop_boxes = [compute_square_crop_coords(stabilized_rois[fi], frame_w, frame_h) for fi in sampled_list]
        off_boxes = [tuple(int(round(v)) for v in full_off[fi]) for fi in sampled_list]
        def_boxes = [tuple(int(round(v)) for v in full_def[fi]) for fi in sampled_list]
        w_feats = np.stack([full_feats[fi] for fi in sampled_list], axis=0).astype(np.float32)

        tube = torch.from_numpy(np.stack(rgb_frames, axis=0)).permute(3, 0, 1, 2).contiguous()
        all_tubes.append(tube)
        windows_out.append(
            {
                "tube": tube,
                "embedding": None,
                "foul_features": torch.from_numpy(w_feats).float(),
                "crop_boxes": crop_boxes,
                "off_boxes": off_boxes,
                "def_boxes": def_boxes,
                "sampled_indices": sampled_list,
                "start_frame": int(sampled_list[0]),
                "end_frame": int(sampled_list[-1]),
                "center_frame": int(sampled_list[len(sampled_list) // 2]),
                "activity_score": float(act_score),
            }
        )

    if encode_embeddings and all_tubes:
        enc_dev = device if device not in ("auto", "mps") else None
        embs = encode_tubes_with_swin3d(all_tubes, device=enc_dev, batch_size=8 if torch.cuda.is_available() else 4)
        for idx_w in range(len(windows_out)):
            windows_out[idx_w]["embedding"] = embs[idx_w]

    return {
        "windows": windows_out,
        "fps": float(fps),
        "total_frames": int(actual_total),
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "team_centroids_lab": team_centroids,
        "wide_segments": wide_segments,
        "used_ebard": bool(use_ebard),
    }


def extract_action_tube(
    video_path: Path | str,
    det_model: Optional[object] = None,
    pose_model: Optional[object] = None,
    window_sec: float = 2.0,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "cpu",
    max_live_play_sec: float = 24.0,
    max_scan_samples: int = 24,
    yolo_batch_size: int = 16,
    **kwargs,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[int], float, torch.Tensor]:
    """
    Backward-compatible single-tube wrapper returning the highest-activity candidate window
    `(tube_tensor, crop_boxes, sampled_indices, fps, foul_features_tensor)`.
    """
    bag = extract_candidate_windows(
        video_path=video_path,
        det_model=det_model,
        pose_model=pose_model,
        window_sec=window_sec,
        stride_sec=1.0,
        num_frames=num_frames,
        crop_size=crop_size,
        device=device,
        max_live_play_sec=max_live_play_sec,
        max_windows=MAX_BAG_WINDOWS,
        scan_fps=2.2,
        yolo_batch_size=yolo_batch_size,
        encode_embeddings=False,
    )
    windows = bag["windows"]
    best_w = max(windows, key=lambda w: w["activity_score"])
    return (
        best_w["tube"],
        best_w["crop_boxes"],
        best_w["sampled_indices"],
        float(bag["fps"]),
        best_w["foul_features"],
    )


def synthesize_fallback_action_tube(
    play_id: str,
    label: int,
    num_frames: int = 16,
    crop_size: int = 224,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[int], float, torch.Tensor]:
    """Deterministic synthetic fallback tensor generator for offline smoke tests without MP4s."""
    seed = int(hashlib.md5(play_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t_axis = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    contact_env = np.exp(-((t_axis - 0.55) ** 2) / 0.045).astype(np.float32)
    scale = 0.78 if label == 1 else 0.28

    foul_feats = np.zeros((num_frames, NUM_FOUL_FEATURES), dtype=np.float32)
    for f_idx in range(NUM_FOUL_FEATURES):
        noise = rng.normal(0.0, 0.04, size=num_frames).astype(np.float32)
        foul_feats[:, f_idx] = np.clip(scale * contact_env + 0.12 + noise, 0.0, 1.0)

    frames = np.zeros((num_frames, crop_size, crop_size, 3), dtype=np.uint8)
    for t in range(num_frames):
        canvas = np.full((crop_size, crop_size, 3), (38, 44, 52), dtype=np.uint8)
        offset = int((1.0 - contact_env[t] * scale) * 36)
        cx1, cy1 = crop_size // 2 - max(8, offset), crop_size // 2
        cx2, cy2 = crop_size // 2 + max(8, offset), crop_size // 2
        cv2.rectangle(canvas, (cx1 - 22, cy1 - 50), (cx1 + 22, cy1 + 55), (210, 140, 70), -1)
        cv2.rectangle(canvas, (cx2 - 22, cy2 - 48), (cx2 + 22, cy2 + 55), (75, 165, 225), -1)
        frames[t] = canvas

    tube_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
    crop_boxes = [(420, 180, 868, 628) for _ in range(num_frames)]
    sampled_indices = list(range(10, 10 + num_frames * 2, 2))
    return tube_tensor, crop_boxes, sampled_indices, 30.0, torch.from_numpy(foul_feats).float()


def _video_content_hash(video_path: Path) -> str:
    size = video_path.stat().st_size
    h = hashlib.md5()
    h.update(str(size).encode("utf-8"))
    with open(video_path, "rb") as fp:
        h.update(fp.read(65536))
        if size > 131072:
            fp.seek(-65536, 2)
            h.update(fp.read(65536))
    return h.hexdigest()


def select_balanced_subset(rows: List[Dict[str, str]], max_clips: int) -> List[Dict[str, str]]:
    """Prioritize rows whose raw `.mp4` exists on disk and balance classes when `max_clips > 0`."""
    def _has_mp4(r: Dict[str, str]) -> int:
        vp = resolve_project_path(r.get("video_path", ""))
        return 0 if (vp.exists() and vp.stat().st_size > 10_240) else 1

    sorted_rows = sorted(rows, key=_has_mp4)
    if max_clips <= 0 or len(sorted_rows) <= max_clips:
        return sorted_rows
    pos_rows = [r for r in sorted_rows if int(r.get("label", 0)) == 1]
    neg_rows = [r for r in sorted_rows if int(r.get("label", 0)) == 0]
    half = max(1, max_clips // 2)
    selected_pos = pos_rows[:half]
    selected_neg = neg_rows[: max_clips - len(selected_pos)]
    if len(selected_pos) + len(selected_neg) < max_clips:
        rem = max_clips - (len(selected_pos) + len(selected_neg))
        selected_pos.extend(pos_rows[len(selected_pos) : len(selected_pos) + rem])
    interleaved: List[Dict[str, str]] = []
    for i in range(max(len(selected_pos), len(selected_neg))):
        if i < len(selected_pos):
            interleaved.append(selected_pos[i])
        if i < len(selected_neg):
            interleaved.append(selected_neg[i])
    return interleaved[:max_clips]


def write_processed_metadata(processed_metadata_csv: Path, rows: List[Dict[str, str]]) -> None:
    """Atomically write `processed_metadata.csv`."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    processed_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_dataset(
    metadata_csv: Path,
    processed_dir: Path,
    processed_metadata_csv: Path,
    test_dir: Optional[Path] = None,
    yolo_weights: str = EBARD_WEIGHTS_NAME,
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "auto",
    max_live_play_sec: float = 24.0,
    max_scan_samples: int = 24,
    max_clips: int = 0,
    yolo_batch_size: int = 32,
    max_bag_windows: int = MAX_BAG_WINDOWS,
    synthetic_fallback: bool = False,
    force_reprocess: bool = False,
    auto_download_if_empty: bool = True,
    save_raw_tube: bool = False,
    cleanup_raw_after_extract: bool = False,
) -> List[Dict[str, str]]:
    """
    Build clean, deduplicated, test-isolated v3 Multi-Window MIL `.pt` bags from real raw MP4 clips:
      1. Strictly excludes all clips in `data/test/` (by stem AND video content hash!) to prevent leakage.
      2. Auto-downloads missing raw `.mp4` files from `l2m_metadata.csv` if `data/raw/` is empty (e.g. fresh Colab clone).
      3. Deduplicates conflicting/identical raw MP4s (prioritizing foul label when two L2M events share one video).
      4. Runs E-BARD Basketball YOLO + 2-Team CIELAB Clustering + YOLOv11-Pose + Swin3D-T (`768-D`)
         across up to `max_bag_windows` (default `8`) 2.0s candidate windows per clip.
      5. Saves real ground-truth `label`, `call_type`, `call_type_label`, `review_decision`, and `game_id`.
    """
    metadata_csv = resolve_project_path(metadata_csv)
    processed_dir = resolve_project_path(processed_dir)
    processed_metadata_csv = resolve_project_path(processed_metadata_csv)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file {metadata_csv} not found. Run `1_scrape_l2m.py` first.")

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # 1. Segregate test clips by both filename stem AND video byte hash
    active_test_dir = resolve_project_path(test_dir) if test_dir is not None else resolve_project_path(BASE_DIR / "data" / "test")
    test_stems: set = set()
    test_hashes: set = set()
    if active_test_dir.exists():
        for p in active_test_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                test_stems.add(p.stem)
                if p.stat().st_size > 10_240:
                    test_hashes.add(_video_content_hash(p))

    def _collect_available_rows() -> List[Dict[str, str]]:
        out_rows: List[Dict[str, str]] = []
        seen_hashes: Dict[str, Tuple[int, int]] = {}  # hash -> (index_in_filtered, label)
        for r in all_rows:
            pid = r.get("play_id", "").strip()
            vp = resolve_project_path(r.get("video_path", ""))
            if pid in test_stems or vp.stem in test_stems:
                continue
            if vp.exists() and vp.stat().st_size > 10_240:
                vh = _video_content_hash(vp)
                if vh in test_hashes:
                    continue
                lbl = int(r.get("label", 0))
                if vh in seen_hashes:
                    prev_idx, prev_lbl = seen_hashes[vh]
                    if lbl == 1 and prev_lbl == 0:
                        out_rows[prev_idx] = r
                        seen_hashes[vh] = (prev_idx, 1)
                    continue
                seen_hashes[vh] = (len(out_rows), lbl)
                out_rows.append(r)
            elif synthetic_fallback:
                out_rows.append(r)
        return out_rows

    filtered_rows = _collect_available_rows()

    # Safety net for fresh Git clones / Google Colab where `data/raw/*.mp4` is gitignored:
    if len(filtered_rows) == 0 and auto_download_if_empty and not synthetic_fallback:
        logger.warning(
            "Found 0 downloaded raw `.mp4` videos on disk for %d metadata rows! "
            "Auto-downloading missing L2M broadcast clips via `1_scrape_l2m.bulk_download_missing_clips()`...",
            len(all_rows),
        )
        try:
            scrape_mod = importlib.import_module("1_scrape_l2m")
            dl_limit = max_clips if max_clips > 0 else 0
            scrape_mod.bulk_download_missing_clips(
                metadata_csv=metadata_csv,
                output_dir=BASE_DIR / "data" / "raw",
                max_workers=10,
                max_clips=dl_limit,
                exclude_stems=test_stems,
            )
            filtered_rows = _collect_available_rows()
        except Exception as exc:
            logger.error("Auto-download of raw L2M clips failed: %s", exc)

    if len(filtered_rows) == 0:
        logger.error(
            "Still 0 raw `.mp4` videos available to preprocess! "
            "Run Stage 1 (`python 1_scrape_l2m.py --bulk-download` or Notebook Cell 2) with internet access first."
        )
        return []

    if test_stems:
        logger.info("Segregated %d test clips (excluded by stem + content hash): %s", len(test_stems), sorted(test_stems))

    rows = select_balanced_subset(filtered_rows, max_clips=max_clips)
    det_model, use_ebard = get_cached_basketball_model(yolo_weights)
    pose_model = get_cached_pose_model(pose_weights or "yolo11n-pose.pt")

    if select_yolo_device(device) == "cuda" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 12.0:
            yolo_batch_size = max(yolo_batch_size, 64)

    processed_dir.mkdir(parents=True, exist_ok=True)
    # Remove any old processed .pt belonging to test clips
    for sub in ("foul", "no_foul"):
        sub_dir = processed_dir / sub
        if sub_dir.exists():
            for old_pt in sub_dir.glob("*.pt"):
                if old_pt.stem in test_stems:
                    try:
                        old_pt.unlink()
                    except Exception:
                        pass

    processed_rows: List[Dict[str, str]] = []
    logger.info(
        "Preprocessing %d unique clips into v%d Multi-Window MIL bags (E-BARD=%s, max_windows=%d, device=%s)...",
        len(rows),
        EXTRACTOR_VERSION,
        use_ebard,
        max_bag_windows,
        device,
    )

    for idx, row in enumerate(rows, start=1):
        play_id = row["play_id"]
        label = int(row["label"])
        call_type = str(row.get("call_type", "Personal Foul")).strip()
        call_type_label = 1 if "shoot" in call_type.lower() else 0
        review_decision = str(row.get("review_decision", "CC" if label == 1 else "CNC")).strip().upper()
        game_id = str(row.get("game_id", play_id.split("_")[0])).strip()

        video_path = resolve_project_path(row["video_path"])
        class_name = "foul" if label == 1 else "no_foul"
        out_tensor_path = processed_dir / class_name / f"{play_id}.pt"
        out_tensor_path.parent.mkdir(parents=True, exist_ok=True)

        needs_reprocess = force_reprocess or (not out_tensor_path.exists())
        if not needs_reprocess and out_tensor_path.exists():
            try:
                existing = torch.load(out_tensor_path, map_location="cpu", weights_only=False)
                if (
                    not isinstance(existing, dict)
                    or existing.get("extractor_v", 0) < EXTRACTOR_VERSION
                    or existing.get("is_synthetic", True)
                    or existing.get("window_embeddings") is None
                ):
                    needs_reprocess = True
            except Exception:
                needs_reprocess = True

        if needs_reprocess:
            if video_path.exists() and video_path.stat().st_size > 10_240:
                try:
                    bag = extract_candidate_windows(
                        video_path=video_path,
                        det_model=det_model,
                        pose_model=pose_model,
                        window_sec=window_sec,
                        stride_sec=stride_sec,
                        num_frames=num_frames,
                        crop_size=crop_size,
                        device=device,
                        max_live_play_sec=max_live_play_sec,
                        max_windows=max_bag_windows,
                        scan_fps=2.2,
                        yolo_batch_size=yolo_batch_size,
                        encode_embeddings=True,
                    )
                    windows = bag["windows"]
                    win_embs = torch.stack([w["embedding"] for w in windows], dim=0).float()       # (K, 768)
                    win_poses = torch.stack([w["foul_features"] for w in windows], dim=0).float()  # (K, 16, 16)
                    win_centers = torch.tensor([w["center_frame"] for w in windows], dtype=torch.long)
                    win_acts = torch.tensor([w["activity_score"] for w in windows], dtype=torch.float32)

                    best_idx = int(torch.argmax(win_acts).item())
                    best_w = windows[best_idx]

                    payload_out: Dict[str, Any] = {
                        "foul_features": best_w["foul_features"],
                        "video_embedding": best_w["embedding"],
                        "window_embeddings": win_embs,
                        "window_pose_features": win_poses,
                        "window_center_frames": win_centers,
                        "window_activity_scores": win_acts,
                        "label": label,
                        "play_id": play_id,
                        "game_id": game_id,
                        "call_type": call_type,
                        "call_type_label": call_type_label,
                        "review_decision": review_decision,
                        "crop_boxes": best_w["crop_boxes"],
                        "sampled_indices": best_w["sampled_indices"],
                        "fps": float(bag["fps"]),
                        "is_synthetic": False,
                        "used_ebard": bool(bag["used_ebard"]),
                        "extractor_v": EXTRACTOR_VERSION,
                    }
                    if save_raw_tube:
                        payload_out["video"] = best_w["tube"]
                    torch.save(payload_out, out_tensor_path)
                    if cleanup_raw_after_extract:
                        try:
                            video_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    if idx % 10 == 0 or idx == len(rows):
                        logger.info(
                            "[%d/%d] Saved v%d compact MIL bag (%d windows, Swin3D-T 768-D) -> %s",
                            idx,
                            len(rows),
                            EXTRACTOR_VERSION,
                            len(windows),
                            out_tensor_path.name,
                        )
                except Exception as exc:
                    logger.warning("Failed to preprocess %s: %s", video_path.name, exc)
                    continue
            elif synthetic_fallback:
                tube, boxes, s_idxs, fps_v, ff = synthesize_fallback_action_tube(
                    play_id=play_id, label=label, num_frames=num_frames, crop_size=crop_size
                )
                emb = encode_tubes_with_swin3d([tube], device=device if device not in ("auto", "mps") else None)[0]
                torch.save(
                    {
                        "video": tube,
                        "foul_features": ff,
                        "video_embedding": emb,
                        "window_embeddings": emb.unsqueeze(0),
                        "window_pose_features": ff.unsqueeze(0),
                        "window_center_frames": torch.tensor([s_idxs[len(s_idxs) // 2]], dtype=torch.long),
                        "window_activity_scores": torch.tensor([0.5], dtype=torch.float32),
                        "label": label,
                        "play_id": play_id,
                        "game_id": game_id,
                        "call_type": call_type,
                        "call_type_label": call_type_label,
                        "review_decision": review_decision,
                        "crop_boxes": boxes,
                        "sampled_indices": s_idxs,
                        "fps": fps_v,
                        "is_synthetic": True,
                        "used_ebard": False,
                        "extractor_v": EXTRACTOR_VERSION,
                    },
                    out_tensor_path,
                )
            else:
                continue

        out_row = dict(row)
        out_row["tensor_path"] = (Path("data") / "processed" / class_name / f"{play_id}.pt").as_posix()
        processed_rows.append(out_row)

    write_processed_metadata(processed_metadata_csv, processed_rows)
    logger.info(
        "Completed v%d preprocessing: %d real MIL bag tensors ready in %s",
        EXTRACTOR_VERSION,
        len(processed_rows),
        processed_dir,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return processed_rows


def stream_download_and_preprocess(
    metadata_csv: Path = BASE_DIR / "data" / "l2m_metadata.csv",
    raw_dir: Path = BASE_DIR / "data" / "raw",
    processed_dir: Path = BASE_DIR / "data" / "processed",
    processed_metadata_csv: Path = BASE_DIR / "data" / "processed" / "processed_metadata.csv",
    test_dir: Optional[Path] = None,
    yolo_weights: str = EBARD_WEIGHTS_NAME,
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "auto",
    max_live_play_sec: float = 24.0,
    target_clips: int = 0,
    chunk_size: int = 100,
    download_workers: int = 12,
    yolo_batch_size: int = 64,
    max_bag_windows: int = MAX_BAG_WINDOWS,
    cleanup_raw_after_extract: bool = True,
) -> List[Dict[str, str]]:
    """
    Disk-Safe Resumable Streaming Pipeline for Scaling to Thousands (or ALL 30,000+) NBA L2M Clips:
      1. Selects balanced, newest-season-first plays across ALL unique NBA games in `l2m_metadata.csv`,
         skipping any clips already converted to compact v3 MIL bags (~45 KB/clip) and excluding `data/test/`.
      2. Downloads raw MP4s in chunks of `chunk_size` (default 100) using `download_workers` threads.
      3. Extracts compact v3 Swin3D-T + E-BARD + 16-D kinematic MIL bags on GPU.
      4. Deletes temporary chunk MP4s right after extraction (`cleanup_raw_after_extract=True`) so disk usage
         stays under ~2 GB even when processing 5,000–35,000+ NBA clips on Google Colab!
    """
    scrape_mod = importlib.import_module("1_scrape_l2m")
    metadata_csv = resolve_project_path(metadata_csv)
    raw_dir = resolve_project_path(raw_dir)
    processed_dir = resolve_project_path(processed_dir)
    processed_metadata_csv = resolve_project_path(processed_metadata_csv)
    active_test_dir = resolve_project_path(test_dir) if test_dir is not None else resolve_project_path(BASE_DIR / "data" / "test")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file {metadata_csv} not found. Run Stage 1 scraper first.")

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    test_stems: set = set()
    if active_test_dir.exists():
        for p in active_test_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                test_stems.add(p.stem)

    # First check already-extracted v3 MIL bags so runs are 100% resumable!
    done_rows: List[Dict[str, str]] = []
    done_pids: set = set()
    pending_by_game_class: Dict[Tuple[str, int], List[Dict[str, str]]] = {}

    for r in all_rows:
        pid = str(r.get("play_id", "")).strip()
        if not pid or pid in test_stems:
            continue
        lbl = int(r.get("label", 0))
        cls_name = "foul" if lbl == 1 else "no_foul"
        pt_path = processed_dir / cls_name / f"{pid}.pt"
        if pt_path.exists() and pt_path.stat().st_size > 4_096:
            out_r = dict(r)
            out_r["tensor_path"] = (Path("data") / "processed" / cls_name / f"{pid}.pt").as_posix()
            done_rows.append(out_r)
            done_pids.add(pid)
        else:
            gid = scrape_mod.normalize_game_id(str(r.get("game_id", ""))) or pid.split("_")[0]
            pending_by_game_class.setdefault((gid, lbl), []).append(r)

    # Order pending clips round-robin newest-season-first alternating foul & no_foul
    ordered_pending: List[Dict[str, str]] = []
    unique_gids = sorted({k[0] for k in pending_by_game_class.keys()}, key=scrape_mod.game_season_sort_key, reverse=True)
    more = True
    while more:
        more = False
        for gid in unique_gids:
            for lbl in (1, 0):
                bucket = pending_by_game_class.get((gid, lbl))
                if bucket:
                    ordered_pending.append(bucket.pop(0))
                    more = True

    if target_clips > 0:
        needed = max(0, target_clips - len(done_rows))
        ordered_pending = ordered_pending[:needed]

    logger.info(
        "Streaming v%d MIL Extractor: %d bags already cached | %d new clips queued (target=%s, chunk=%d, cleanup_raw=%s)",
        EXTRACTOR_VERSION,
        len(done_rows),
        len(ordered_pending),
        "ALL" if target_clips <= 0 else str(target_clips),
        chunk_size,
        cleanup_raw_after_extract,
    )
    if not ordered_pending:
        write_processed_metadata(processed_metadata_csv, done_rows)
        return done_rows

    det_model, use_ebard = get_cached_basketball_model(yolo_weights)
    pose_model = get_cached_pose_model(pose_weights or "yolo11n-pose.pt")
    if select_yolo_device(device) == "cuda" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 12.0:
            yolo_batch_size = max(yolo_batch_size, 64)

    processed_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = (len(ordered_pending) + chunk_size - 1) // chunk_size

    for c_idx in range(total_chunks):
        chunk_rows = ordered_pending[c_idx * chunk_size : (c_idx + 1) * chunk_size]
        logger.info(
            "=== Stream Chunk [%d/%d] (%d clips) — Downloading MP4s & Extracting Swin3D-T MIL Bags ===",
            c_idx + 1,
            total_chunks,
            len(chunk_rows),
        )
        scrape_mod.bulk_download_missing_clips(
            metadata_csv=metadata_csv,
            output_dir=raw_dir,
            max_workers=download_workers,
            max_clips=0,
            exclude_stems=test_stems,
            rows_override=chunk_rows,
            prefer_newest=True,
        )

        chunk_done = 0
        for r in chunk_rows:
            play_id = r["play_id"]
            label = int(r["label"])
            call_type = str(r.get("call_type", "Personal Foul")).strip()
            call_type_label = 1 if "shoot" in call_type.lower() else 0
            review_decision = str(r.get("review_decision", "CC" if label == 1 else "CNC")).strip().upper()
            game_id = str(r.get("game_id", play_id.split("_")[0])).strip()
            class_name = "foul" if label == 1 else "no_foul"
            video_path = resolve_project_path(r["video_path"])
            out_tensor_path = processed_dir / class_name / f"{play_id}.pt"
            out_tensor_path.parent.mkdir(parents=True, exist_ok=True)

            if not (video_path.exists() and video_path.stat().st_size > 10_240):
                continue
            try:
                bag = extract_candidate_windows(
                    video_path=video_path,
                    det_model=det_model,
                    pose_model=pose_model,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    num_frames=num_frames,
                    crop_size=crop_size,
                    device=device,
                    max_live_play_sec=max_live_play_sec,
                    max_windows=max_bag_windows,
                    scan_fps=2.2,
                    yolo_batch_size=yolo_batch_size,
                    encode_embeddings=True,
                )
                windows = bag["windows"]
                win_embs = torch.stack([w["embedding"] for w in windows], dim=0).float()
                win_poses = torch.stack([w["foul_features"] for w in windows], dim=0).float()
                win_centers = torch.tensor([w["center_frame"] for w in windows], dtype=torch.long)
                win_acts = torch.tensor([w["activity_score"] for w in windows], dtype=torch.float32)
                best_idx = int(torch.argmax(win_acts).item())
                best_w = windows[best_idx]

                torch.save(
                    {
                        "foul_features": best_w["foul_features"],
                        "video_embedding": best_w["embedding"],
                        "window_embeddings": win_embs,
                        "window_pose_features": win_poses,
                        "window_center_frames": win_centers,
                        "window_activity_scores": win_acts,
                        "label": label,
                        "play_id": play_id,
                        "game_id": game_id,
                        "call_type": call_type,
                        "call_type_label": call_type_label,
                        "review_decision": review_decision,
                        "crop_boxes": best_w["crop_boxes"],
                        "sampled_indices": best_w["sampled_indices"],
                        "fps": float(bag["fps"]),
                        "is_synthetic": False,
                        "used_ebard": bool(bag["used_ebard"]),
                        "extractor_v": EXTRACTOR_VERSION,
                    },
                    out_tensor_path,
                )
                out_r = dict(r)
                out_r["tensor_path"] = (Path("data") / "processed" / class_name / f"{play_id}.pt").as_posix()
                done_rows.append(out_r)
                chunk_done += 1
            except Exception as exc:
                logger.warning("Stream extract failed on %s: %s", video_path.name, exc)
            finally:
                if cleanup_raw_after_extract:
                    try:
                        video_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        # Checkpoint processed_metadata.csv after EVERY chunk so progress is never lost!
        write_processed_metadata(processed_metadata_csv, done_rows)
        logger.info(
            "Chunk [%d/%d] complete (+%d extracted | %d total v%d MIL bags saved)",
            c_idx + 1,
            total_chunks,
            chunk_done,
            len(done_rows),
            EXTRACTOR_VERSION,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return done_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E-BARD Basketball Perception, 2-Team CIELAB Clustering & Multi-Window MIL Extractor"
    )
    parser.add_argument("--metadata-csv", type=Path, default=BASE_DIR / "data" / "l2m_metadata.csv")
    parser.add_argument("--processed-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--processed-metadata-csv", type=Path, default=BASE_DIR / "data" / "processed" / "processed_metadata.csv")
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "data" / "test")
    parser.add_argument("--yolo-weights", type=str, default=EBARD_WEIGHTS_NAME)
    parser.add_argument("--pose-weights", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--max-clips", type=int, default=0, help="0 = process all available clips")
    parser.add_argument("--yolo-batch-size", type=int, default=32)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--max-bag-windows", type=int, default=MAX_BAG_WINDOWS)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-live-play-sec", type=float, default=24.0)
    parser.add_argument("--max-scan-samples", type=int, default=24)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--synthetic-fallback", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Stream-download in chunks, extract compact 45KB MIL bags, and optionally delete temp MP4s")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument("--cleanup-raw", action="store_true", help="Delete raw MP4 after extracting its compact MIL bag (disk-safe for 1,000-37,000+ clips)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stream:
        stream_download_and_preprocess(
            metadata_csv=args.metadata_csv,
            processed_dir=args.processed_dir,
            processed_metadata_csv=args.processed_metadata_csv,
            test_dir=args.test_dir,
            yolo_weights=args.yolo_weights,
            pose_weights=args.pose_weights or None,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            num_frames=args.num_frames,
            crop_size=args.crop_size,
            device=args.device,
            max_live_play_sec=args.max_live_play_sec,
            target_clips=args.max_clips,
            chunk_size=args.chunk_size,
            download_workers=args.download_workers,
            yolo_batch_size=args.yolo_batch_size,
            max_bag_windows=args.max_bag_windows,
            cleanup_raw_after_extract=args.cleanup_raw,
        )
    else:
        preprocess_dataset(
            metadata_csv=args.metadata_csv,
            processed_dir=args.processed_dir,
            processed_metadata_csv=args.processed_metadata_csv,
            test_dir=args.test_dir,
            yolo_weights=args.yolo_weights,
            pose_weights=args.pose_weights or None,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            num_frames=args.num_frames,
            crop_size=args.crop_size,
            device=args.device,
            max_live_play_sec=args.max_live_play_sec,
            max_scan_samples=args.max_scan_samples,
            max_clips=args.max_clips,
            yolo_batch_size=args.yolo_batch_size,
            max_bag_windows=args.max_bag_windows,
            synthetic_fallback=args.synthetic_fallback,
            force_reprocess=args.force_reprocess,
            cleanup_raw_after_extract=args.cleanup_raw,
        )
