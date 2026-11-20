#!/usr/bin/env python3
"""
STAGE 2: Shot-Segmented Multi-Angle Replay Action-Tube & Pose Extractor (`2_preprocess_replays.py`)

Engineered specifically for NBA Official Replay Archive clips (`official.nba.com/replay/archive/`):
  - Multi-Angle Shot Segmentation: Official Replay Center MP4s (~25-45s, 60 fps) stitch together
    multiple camera angles (wide broadcast, alternate slow-motion, overhead/baseline). We detect
    hard camera cuts first and NEVER interpolate ROIs, ball tracks, or action tubes across cuts.
  - Angle-Aware Geometry: Accepts both wide side-court and zoomed baseline/overhead replay angles
    (supporting centered rims and larger player heights up to 0.80 * frame_h) while rejecting
    referees, seated courtside spectators, and face-only dead-ball close-ups.
  - FPS-Normalized & Slow-Motion-Aware Kinematics: Scans at constant physical rate (5 Hz) so
    60 fps and 30 fps clips behave identically; adapts window length for slow-motion replay shots
    and normalizes velocity/shock features by elapsed wall-clock `dt`.
  - Multi-Angle Extraction (`--angles-per-clip 2`): Extracts up to K clean camera-angle action
    tubes per replay clip, saving `(3, 16, 224, 224)` uint8 RGB + `(16, 12)` pose kinematics
    into `data/processed/{foul,no_foul}/` and indexing `play_id` + `game_id` in
    `data/processed/processed_metadata.csv` for leak-free grouped splitting in Stage 4.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

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
logger = logging.getLogger("preprocess_replays")

POSE_FEATURE_NAMES: Tuple[str, ...] = (
    "arm_contact_intrusion",        # 0: Defender hand/wrist proximity to offensive arm/wrist
    "body_cylinder_penetration",    # 1: Torso/hip overlap invading offensive cylinder
    "lower_body_landing_violation", # 2: Defender feet/legs encroaching landing zone
    "defender_verticality_angle",   # 3: Arm reach deviation from legal vertical contest
    "defender_lateral_positioning", # 4: Lateral offset between defender & offensive player
    "offensive_displacement",       # 5: Collision overlap / displacement proxy
    "defender_to_ball_distance",    # 6: Defender wrist proximity to basketball
    "ball_to_shooter_distance",     # 7: Offensive wrist proximity to basketball
    "ball_strike_vs_body_ratio",    # 8: Body contact relative to clean play on ball
    "relative_closing_velocity",    # 9: Time-normalized closing velocity (1/sec)
    "momentum_transfer_shock",      # 10: Time-normalized collision jolt / acceleration
    "airborne_contest_apex_diff",   # 11: Composite contact + arm elevation contest apex
)
POSE_FEATURES = len(POSE_FEATURE_NAMES)  # 12

# COCO 17-keypoint indices
UPPER_BODY_KPTS = (0, 5, 6, 7, 8, 9, 10)
ARM_CONTACT_KPTS = (7, 8, 9, 10)
TORSO_KPTS = (5, 6, 11, 12)
LOWER_BODY_KPTS = (13, 14, 15, 16)
COCO_PERSON_CLS = 0
COCO_SPORTS_BALL_CLS = 32

PROCESSED_CSV_COLUMNS: List[str] = [
    "sample_id",
    "play_id",
    "game_id",
    "season",
    "trigger",
    "outcome_category",
    "angle_idx",
    "shot_start_frame",
    "shot_end_frame",
    "peak_frame",
    "fps",
    "est_speed_ratio",
    "label",
    "video_path",
    "tensor_path",
]


def resolve_project_path(path_arg: Path | str) -> Path:
    """Resolve paths cross-platform checking CWD, project dir, weights/, and repo root."""
    norm_str = str(path_arg).replace("\\", "/")
    p = Path(norm_str)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == BASE_DIR.name:
        return (BASE_DIR.parent / p).resolve()
    candidates = [
        Path.cwd() / p,
        BASE_DIR / p,
        BASE_DIR / "weights" / p.name,
        BASE_DIR / "weights" / p,
        BASE_DIR.parent / p.name,
        BASE_DIR.parent / p,
        BASE_DIR.parent / "NBA-Foul-Classifier-WIP" / "weights" / p.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (BASE_DIR / p).resolve()


def select_yolo_device(requested: Optional[str] = None) -> str:
    """Select reliable Ultralytics YOLO device (`cuda` on GPU, `cpu` on macOS/CPU for deterministic pose NMS)."""
    if requested and requested.lower() not in ("auto", "mps"):
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xA = max(float(box_a[0]), float(box_b[0]))
    yA = max(float(box_a[1]), float(box_b[1]))
    xB = min(float(box_a[2]), float(box_b[2]))
    yB = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(1.0, float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]))
    area_b = max(1.0, float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]))
    return float(inter / max(area_a + area_b - inter, 1.0))


def box_iom(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Minimum-Area to eliminate duplicate boxes on the same player."""
    xA = max(float(box_a[0]), float(box_b[0]))
    yA = max(float(box_a[1]), float(box_b[1]))
    xB = min(float(box_a[2]), float(box_b[2]))
    yB = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area_a = max(1.0, float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]))
    area_b = max(1.0, float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]))
    return float(inter / min(area_a, area_b))


def valid_points(kpts_xy: np.ndarray, indices: Tuple[int, ...]) -> np.ndarray:
    pts = []
    max_idx = len(kpts_xy)
    for idx in indices:
        if idx < max_idx:
            x, y = float(kpts_xy[idx, 0]), float(kpts_xy[idx, 1])
            if x > 2.0 and y > 2.0:
                pts.append((x, y))
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def min_point_set_distance(pts_a: np.ndarray, pts_b: np.ndarray, default_dist: float) -> float:
    if len(pts_a) == 0 or len(pts_b) == 0:
        return default_dist
    diffs = pts_a[:, None, :] - pts_b[None, :, :]
    return float(np.min(np.linalg.norm(diffs, axis=-1)))


def synthesize_box_keypoints(box_xyxy: np.ndarray) -> np.ndarray:
    """Synthesize a fallback 17-point COCO skeleton from a bounding box when pose NMS misses an occluded player."""
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


def compute_arm_elevation(kpts_xy: np.ndarray, box_h: float) -> float:
    shoulders = valid_points(kpts_xy, (5, 6))
    wrists = valid_points(kpts_xy, (7, 8, 9, 10))
    if len(shoulders) == 0 or len(wrists) == 0:
        return 0.0
    shoulder_y = float(np.mean(shoulders[:, 1]))
    highest_wrist_y = float(np.min(wrists[:, 1]))
    return float(np.clip((shoulder_y - highest_wrist_y) / max(box_h * 0.5, 1.0), -1.0, 1.5))


def extract_player_torso_hsv(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[float, float, float]:
    if bgr_frame is None or bgr_frame.size == 0:
        return 0.0, 50.0, 128.0
    fh, fw = bgr_frame.shape[:2]
    x1 = float(box_xyxy[0]) / max(scale_x, 1e-5)
    y1 = float(box_xyxy[1]) / max(scale_y, 1e-5)
    x2 = float(box_xyxy[2]) / max(scale_x, 1e-5)
    y2 = float(box_xyxy[3]) / max(scale_y, 1e-5)
    bh, bw = max(1.0, y2 - y1), max(1.0, x2 - x1)
    ty1 = max(0, min(fh - 1, int(y1 + 0.18 * bh)))
    ty2 = max(ty1 + 1, min(fh, int(y1 + 0.52 * bh)))
    tx1 = max(0, min(fw - 1, int(x1 + 0.22 * bw)))
    tx2 = max(tx1 + 1, min(fw, int(x2 - 0.22 * bw)))
    torso = bgr_frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return 0.0, 50.0, 128.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0])), float(np.mean(hsv[:, :, 1])), float(np.mean(hsv[:, :, 2]))


def jersey_hsv_diff(hsv_a: Tuple[float, float, float], hsv_b: Tuple[float, float, float]) -> float:
    dh = min(abs(hsv_a[0] - hsv_b[0]), 180.0 - abs(hsv_a[0] - hsv_b[0]))
    return abs(hsv_a[2] - hsv_b[2]) + 0.75 * abs(hsv_a[1] - hsv_b[1]) + 1.1 * dh


def is_referee_box(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    frame_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> bool:
    """Filter out NBA officials (gray/black low-saturation shirt + black trousers)."""
    if bgr_frame is None or bgr_frame.size == 0:
        return False
    fh, fw = bgr_frame.shape[:2]
    sx1 = float(box_xyxy[0]) / max(scale_x, 1e-5)
    sy1 = float(box_xyxy[1]) / max(scale_y, 1e-5)
    sx2 = float(box_xyxy[2]) / max(scale_x, 1e-5)
    sy2 = float(box_xyxy[3]) / max(scale_y, 1e-5)
    sbh, sbw = max(1.0, sy2 - sy1), max(1.0, sx2 - sx1)

    ty1, ty2 = max(0, min(fh - 1, int(sy1 + 0.18 * sbh))), max(1, min(fh, int(sy1 + 0.48 * sbh)))
    tx1, tx2 = max(0, min(fw - 1, int(sx1 + 0.22 * sbw))), max(1, min(fw, int(sx2 - 0.22 * sbw)))
    ly1, ly2 = max(0, min(fh - 1, int(sy1 + 0.52 * sbh))), max(1, min(fh, int(sy1 + 0.86 * sbh)))
    lx1, lx2 = max(0, min(fw - 1, int(sx1 + 0.24 * sbw))), max(1, min(fw, int(sx2 - 0.24 * sbw)))

    torso = bgr_frame[ty1:ty2, tx1:tx2]
    lower = bgr_frame[ly1:ly2, lx1:lx2]
    if torso.size == 0 or lower.size == 0:
        return False
    thsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lhsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    ts, tv = float(np.mean(thsv[:, :, 1])), float(np.mean(thsv[:, :, 2]))
    lv = float(np.mean(lhsv[:, :, 2]))

    if tv < 90.0 and ts < 75.0 and lv < 86.0:
        return True
    if float(box_xyxy[1]) >= frame_h * 0.60 and float(box_xyxy[3]) >= frame_h * 0.85 and ts < 70.0 and tv < 135.0:
        return True
    return False


def is_replay_court_player_box(box: np.ndarray, frame_w: int, frame_h: int) -> bool:
    """
    Angle-aware court player filter that works for BOTH wide broadcast views and
    closer baseline / overhead replay camera angles while rejecting upper stands
    and seated courtside photographers.
    """
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bh = y2 - y1
    bw = x2 - x1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    if bh < 0.08 * frame_h or bh > 0.82 * frame_h:
        return False
    if bw < 0.018 * frame_w or bw > 0.52 * frame_w:
        return False
    if bh / max(bw, 1.0) < 0.68:
        return False
    # Reject upper stands / scoreboard people (small boxes near top)
    if y2 < 0.34 * frame_h and bh < 0.22 * frame_h:
        return False
    # Reject tiny seated spectators along the very bottom frame edge
    if y2 > 0.86 * frame_h and bh < 0.18 * frame_h:
        return False
    if not (0.03 * frame_w <= cx <= 0.97 * frame_w):
        return False
    if not (0.14 * frame_h <= cy <= 0.92 * frame_h):
        return False
    return True


def detect_rim_hsv(bgr_frame: np.ndarray, frame_w: int, frame_h: int) -> Optional[Tuple[int, int]]:
    """
    Detect the orange basketball rim across BOTH side-court broadcast angles and
    centered overhead/baseline replay angles (`rx_norm` anywhere in `[0.08, 0.92]`).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    img_h, img_w = bgr_frame.shape[:2]
    y0, y1 = int(0.08 * img_h), int(0.48 * img_h)
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
        if not (0.08 <= rx_norm <= 0.92):
            continue
        if (0.008 * img_w <= w <= 0.085 * img_w) and (0.004 * img_h <= h <= 0.055 * img_h):
            if (w / max(h, 1.0)) >= 1.25 and area >= max(18.0, 0.00004 * img_w * img_h) and area > best_area:
                best_area = area
                best_rim = (
                    int(round((x + 0.5 * w) * (frame_w / max(img_w, 1)))),
                    int(round((y0 + y + 0.5 * h) * (frame_h / max(img_h, 1)))),
                )
    return best_rim


def extract_verified_ball(
    det_res: Any,
    scale_x: float,
    scale_y: float,
    frame_w: int,
    frame_h: int,
    bgr_frame: Optional[np.ndarray] = None,
    rim_pt: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[float, float, float]]:
    """Extract orange-verified basketball center `(cx, cy, conf)` from YOLO detections."""
    if det_res is None or det_res.boxes is None or len(det_res.boxes) == 0:
        return None
    d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
    d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
    d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)

    img_h, img_w = bgr_frame.shape[:2] if bgr_frame is not None else (frame_h, frame_w)
    inv_sx = float(img_w) / max(float(frame_w), 1.0)
    inv_sy = float(img_h) / max(float(frame_h), 1.0)

    best_ball: Optional[Tuple[float, float, float]] = None
    best_rank = -1.0
    for box, cls_id, conf in zip(d_xyxy, d_cls, d_conf):
        if cls_id != COCO_SPORTS_BALL_CLS or float(conf) < 0.15:
            continue
        bx1, by1 = float(box[0]) * scale_x, float(box[1]) * scale_y
        bx2, by2 = float(box[2]) * scale_x, float(box[3]) * scale_y
        bw, bh = bx2 - bx1, by2 - by1
        if not (0.008 * frame_h <= bw <= 0.11 * frame_h and 0.008 * frame_h <= bh <= 0.11 * frame_h):
            continue
        if max(bw, bh) / max(min(bw, bh), 1.0) > 1.75:
            continue
        cx, cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        if not (0.04 * frame_h <= cy <= 0.90 * frame_h):
            continue
        if rim_pt is not None and float(conf) < 0.30:
            if np.hypot(cx - float(rim_pt[0]), cy - float(rim_pt[1])) < 30.0:
                continue

        orange_frac = 0.25
        if bgr_frame is not None and bgr_frame.size > 0:
            ix1 = max(0, int(bx1 * inv_sx))
            iy1 = max(0, int(by1 * inv_sy))
            ix2 = min(img_w, max(ix1 + 1, int(bx2 * inv_sx)))
            iy2 = min(img_h, max(iy1 + 1, int(by2 * inv_sy)))
            patch = bgr_frame[iy1:iy2, ix1:ix2]
            if patch.size > 0:
                phsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                orange_frac = float(
                    np.mean((phsv[:, :, 0] >= 3) & (phsv[:, :, 0] <= 24) & (phsv[:, :, 1] >= 60))
                )
        if orange_frac < 0.12 or (float(conf) < 0.28 and orange_frac < 0.22):
            continue
        rank = float(conf) + 0.22 * orange_frac
        if rank > best_rank:
            best_rank = rank
            best_ball = (cx, cy, float(conf))
    return best_ball


def segment_replay_shots(
    small_bgr_frames: List[np.ndarray],
    sampled_indices: List[int],
    fps: float,
    min_shot_sec: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Detect hard camera cuts across a multi-angle NBA Replay Center clip and partition
    scanned frames into contiguous shot segments. Also estimates each shot's playback
    speed ratio (`est_speed_ratio` in `[0.35, 1.0]`) to adapt slow-motion windows.
    """
    n = len(small_bgr_frames)
    if n == 0:
        return []
    if n == 1:
        return [{
            "scan_lo": 0,
            "scan_hi": 0,
            "start_frame": sampled_indices[0],
            "end_frame": sampled_indices[0],
            "duration_sec": 0.0,
            "est_speed_ratio": 1.0,
        }]

    hists: List[np.ndarray] = []
    grays: List[np.ndarray] = []
    for fr in small_bgr_frames:
        sm = cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        hists.append(h)
        grays.append(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY))

    cut_positions: List[int] = [0]
    step_motions: List[float] = [0.0] * n
    for t in range(1, n):
        hist_diff = float(cv2.compareHist(hists[t - 1], hists[t], cv2.HISTCMP_BHATTACHARYYA))
        crop = grays[t - 1][16:74, 24:136]
        tm = float(cv2.minMaxLoc(cv2.matchTemplate(grays[t][10:80, :], crop, cv2.TM_CCOEFF_NORMED))[1])
        mad = float(np.mean(np.abs(grays[t].astype(np.float32) - grays[t - 1].astype(np.float32))))
        dt = max(1e-3, float(sampled_indices[t] - sampled_indices[t - 1]) / max(fps, 1.0))
        step_motions[t] = mad / max(dt * 5.0, 0.25)

        is_cut = (
            (hist_diff > 0.30 and tm < 0.56)
            or (hist_diff > 0.44 and tm < 0.70)
            or (mad > 28.0 and tm < 0.48)
        )
        if is_cut:
            cut_positions.append(t)
    cut_positions.append(n)

    raw_shots: List[Dict[str, Any]] = []
    for k in range(len(cut_positions) - 1):
        lo = cut_positions[k]
        hi = cut_positions[k + 1] - 1
        if hi < lo:
            continue
        start_f = int(sampled_indices[lo])
        end_f = int(sampled_indices[hi])
        dur_sec = float(end_f - start_f) / max(fps, 1.0)
        if dur_sec < min_shot_sec and (hi - lo) < 2:
            continue

        # Estimate slow-motion vs live-speed motion density inside the shot
        m_vals = step_motions[lo + 1 : hi + 1]
        med_motion = float(np.median(m_vals)) if m_vals else 6.0
        # Live wide basketball action at 5Hz typically has gray MAD ~7-16; slow-mo ~2-6
        est_speed = float(np.clip(med_motion / 9.5, 0.45, 1.0))
        raw_shots.append({
            "scan_lo": lo,
            "scan_hi": hi,
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_sec": dur_sec,
            "est_speed_ratio": round(est_speed, 3),
        })

    if not raw_shots:
        raw_shots.append({
            "scan_lo": 0,
            "scan_hi": n - 1,
            "start_frame": int(sampled_indices[0]),
            "end_frame": int(sampled_indices[-1]),
            "duration_sec": float(sampled_indices[-1] - sampled_indices[0]) / max(fps, 1.0),
            "est_speed_ratio": 1.0,
        })
    return raw_shots


def collect_players_in_frame(
    pose_res: Any,
    det_res: Any,
    frame_w: int,
    frame_h: int,
    scale_x: float,
    scale_y: float,
    bgr_frame: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Collect deduplicated active court players with 17-point skeletons and jersey HSV.
    Also returns `max_person_height_ratio` so extreme face close-ups can be filtered out.
    """
    merged: List[Tuple[np.ndarray, np.ndarray, float]] = []
    max_h_ratio = 0.0

    if pose_res is not None and pose_res.boxes is not None and len(pose_res.boxes) > 0:
        p_xyxy = pose_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        p_xyxy[:, [0, 2]] *= scale_x
        p_xyxy[:, [1, 3]] *= scale_y
        p_conf = (
            pose_res.boxes.conf.cpu().numpy().astype(np.float32)
            if pose_res.boxes.conf is not None
            else np.ones(len(p_xyxy), dtype=np.float32) * 0.5
        )
        p_cls = (
            pose_res.boxes.cls.cpu().numpy().astype(int)
            if pose_res.boxes.cls is not None
            else np.zeros(len(p_xyxy), dtype=int)
        )
        has_kpts = (
            getattr(pose_res, "keypoints", None) is not None
            and pose_res.keypoints.xy is not None
            and len(pose_res.keypoints.xy) == len(p_xyxy)
        )
        p_kpts = None
        if has_kpts:
            p_kpts = pose_res.keypoints.xy.cpu().numpy().astype(np.float32).copy()
            p_kpts[:, :, 0] *= scale_x
            p_kpts[:, :, 1] *= scale_y

        for i in range(len(p_xyxy)):
            if p_cls[i] != COCO_PERSON_CLS or p_conf[i] < 0.12:
                continue
            max_h_ratio = max(max_h_ratio, float(p_xyxy[i, 3] - p_xyxy[i, 1]) / max(float(frame_h), 1.0))
            kp = p_kpts[i] if p_kpts is not None else synthesize_box_keypoints(p_xyxy[i])
            merged.append((p_xyxy[i], kp, float(p_conf[i])))

    if det_res is not None and det_res.boxes is not None and len(det_res.boxes) > 0:
        d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        d_xyxy[:, [0, 2]] *= scale_x
        d_xyxy[:, [1, 3]] *= scale_y
        d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
        d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)
        for db, dc, cf in zip(d_xyxy, d_cls, d_conf):
            if dc != COCO_PERSON_CLS or float(cf) < 0.24:
                continue
            max_h_ratio = max(max_h_ratio, float(db[3] - db[1]) / max(float(frame_h), 1.0))
            if all(box_iou(db, mb) < 0.35 and box_iom(db, mb) < 0.55 for mb, _, _ in merged):
                merged.append((db, synthesize_box_keypoints(db), float(cf)))

    # Extreme face/bench close-up rejection (> 88% of frame height)
    if max_h_ratio > 0.88:
        return [], max_h_ratio

    players: List[Dict[str, Any]] = []
    for box, kp, conf in merged:
        if not is_replay_court_player_box(box, frame_w, frame_h):
            continue
        if is_referee_box(bgr_frame, box, frame_h, scale_x=scale_x, scale_y=scale_y):
            continue
        bh = max(1.0, float(box[3] - box[1]))
        hsv = extract_player_torso_hsv(bgr_frame, box, scale_x=scale_x, scale_y=scale_y)
        elev = compute_arm_elevation(kp, bh)
        players.append({
            "box": box,
            "kpts": kp,
            "conf": conf,
            "hsv": hsv,
            "elev": elev,
            "center": np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=np.float32),
        })
    return players, max_h_ratio


def analyze_contact_pair(
    players: List[Dict[str, Any]],
    ball_pt: Optional[np.ndarray],
    rim_pt: Optional[Tuple[int, int]],
    prev_center: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
) -> Tuple[np.ndarray, float, np.ndarray, Optional[np.ndarray]]:
    """
    Identify the primary opposing-jersey contact pair in a single frame and compute:
      - `roi_box`: `[x1, y1, x2, y2]` enclosing both interacting players
      - `saliency`: scalar contact/contest intensity score
      - `feats`: 12-D biomechanical feature vector
      - `pair_mid`: `(2,)` center for within-shot temporal smoothness
    """
    diag = float(np.hypot(frame_w, frame_h))
    default_roi = np.array([frame_w * 0.20, frame_h * 0.15, frame_w * 0.80, frame_h * 0.85], dtype=np.float32)
    feats = np.zeros(POSE_FEATURES, dtype=np.float32)

    if len(players) < 2:
        if len(players) == 1:
            return players[0]["box"].astype(np.float32), 0.08, feats, players[0]["center"]
        return default_roi, 0.0, feats, None

    rim_arr = np.array([float(rim_pt[0]), float(rim_pt[1])], dtype=np.float32) if rim_pt is not None else None
    best_pair: Optional[Tuple[int, int]] = None
    best_rank = -1e9
    best_stats = (diag, diag, 0.0)

    for i in range(len(players)):
        bi, ci, ki = players[i]["box"], players[i]["center"], players[i]["kpts"]
        bhi = max(1.0, float(bi[3] - bi[1]))
        for j in range(i + 1, len(players)):
            bj, cj, kj = players[j]["box"], players[j]["center"], players[j]["kpts"]
            bhj = max(1.0, float(bj[3] - bj[1]))

            if min(bhi, bhj) / max(bhi, bhj) < 0.50:
                continue
            cdist = float(np.linalg.norm(ci - cj))
            max_bh = max(bhi, bhj)
            if cdist > max(0.32 * diag, 1.75 * max_bh):
                continue

            cdiff = jersey_hsv_diff(players[i]["hsv"], players[j]["hsv"])
            if box_iom(bi, bj) >= 0.68 and cdiff < 26.0:
                continue

            iou = box_iou(bi, bj)
            arms_i = valid_points(ki, ARM_CONTACT_KPTS)
            arms_j = valid_points(kj, ARM_CONTACT_KPTS)
            upper_i = valid_points(ki, UPPER_BODY_KPTS)
            upper_j = valid_points(kj, UPPER_BODY_KPTS)
            torso_i = valid_points(ki, TORSO_KPTS)
            torso_j = valid_points(kj, TORSO_KPTS)

            arm_upper_d = min(
                min_point_set_distance(arms_i, upper_j, cdist),
                min_point_set_distance(arms_j, upper_i, cdist),
            )
            torso_d = min_point_set_distance(torso_i, torso_j, cdist)

            norm_arm = max(0.0, 1.0 - arm_upper_d / max(0.45 * max_bh, 1.0))
            norm_torso = max(0.0, 1.0 - torso_d / max(0.60 * max_bh, 1.0))
            elev = max(players[i]["elev"], players[j]["elev"])
            mid = 0.5 * (ci + cj)

            ball_prox = 0.0
            off_ball_pen = 0.0
            if ball_pt is not None:
                dball = min(
                    float(np.linalg.norm(mid - ball_pt)),
                    float(np.linalg.norm(ci - ball_pt)) + 20.0,
                    float(np.linalg.norm(cj - ball_pt)) + 20.0,
                )
                ball_prox = max(0.0, 1.0 - dball / max(0.24 * diag, 1.0))
                if dball > max(0.28 * diag, 380.0):
                    off_ball_pen = -0.35

            rim_bonus = 0.0
            if rim_arr is not None:
                drim = float(np.linalg.norm(mid - rim_arr))
                rim_bonus = 0.12 * max(0.0, 1.0 - drim / max(0.42 * diag, 1.0))

            smooth_bonus = 0.0
            if prev_center is not None:
                dprev = float(np.linalg.norm(mid - prev_center))
                smooth_bonus = 0.18 * float(np.exp(-dprev / max(0.18 * diag, 1.0)))

            team_bonus = 0.28 if cdiff >= 26.0 else -0.30
            rank = (
                0.34 * norm_arm
                + 0.24 * norm_torso
                + 0.24 * min(1.0, iou * 3.0)
                + 0.34 * ball_prox
                + 0.18 * max(0.0, min(1.0, elev))
                + team_bonus
                + rim_bonus
                + smooth_bonus
                + off_ball_pen
            )
            if rank > best_rank:
                best_rank = rank
                best_pair = (i, j)
                best_stats = (arm_upper_d, torso_d, iou)

    if best_pair is None:
        return default_roi, 0.0, feats, None

    idx_a, idx_b = best_pair
    pa, pb = players[idx_a], players[idx_b]
    # Higher elevated arm -> approximate shooter/ball-handler role; other -> contesting defender
    if pa["elev"] >= pb["elev"]:
        p_off, p_def = pa, pb
    else:
        p_off, p_def = pb, pa

    box_off, box_def = p_off["box"], p_def["box"]
    kpts_off, kpts_def = p_off["kpts"], p_def["kpts"]
    h_off = max(1.0, float(box_off[3] - box_off[1]))
    h_def = max(1.0, float(box_def[3] - box_def[1]))
    w_off = max(1.0, float(box_off[2] - box_off[0]))
    arm_upper_d, torso_d, pair_iou = best_stats

    roi_box = np.array(
        [
            min(float(box_off[0]), float(box_def[0])),
            min(float(box_off[1]), float(box_def[1])),
            max(float(box_off[2]), float(box_def[2])),
            max(float(box_off[3]), float(box_def[3])),
        ],
        dtype=np.float32,
    )
    pair_mid = 0.5 * (p_off["center"] + p_def["center"])

    # Compute 12 biomechanical features (dims 9 & 10 are refined temporally across the shot)
    arms_off = valid_points(kpts_off, ARM_CONTACT_KPTS)
    arms_def = valid_points(kpts_def, ARM_CONTACT_KPTS)
    arm_d = min_point_set_distance(arms_off, arms_def, arm_upper_d)
    arm_contact = float(np.clip(1.0 - arm_d / max(h_off * 0.28, 1.0), 0.0, 1.0))

    cyl_pen = float(
        np.clip(
            0.50 * (1.0 - torso_d / max(h_off * 0.36, 1.0)) + 0.50 * min(1.0, pair_iou * 2.4),
            0.0,
            1.0,
        )
    )

    feet_off = valid_points(kpts_off, LOWER_BODY_KPTS)
    feet_def = valid_points(kpts_def, LOWER_BODY_KPTS)
    feet_d = min_point_set_distance(feet_off, feet_def, 0.5 * (h_off + h_def))
    landing_viol = float(np.clip(1.0 - feet_d / max(h_off * 0.28, 1.0), 0.0, 1.0))

    sh_def = valid_points(kpts_def, (5, 6))
    wr_def = valid_points(kpts_def, (9, 10))
    if len(sh_def) > 0 and len(wr_def) > 0:
        dx = float(np.mean(wr_def[:, 0]) - np.mean(sh_def[:, 0]))
        dy = float(np.mean(wr_def[:, 1]) - np.mean(sh_def[:, 1]))
        reach_toward = (float(p_off["center"][0]) - float(p_def["center"][0])) * dx
        vert_dev = float(np.clip(abs(dx) / max(abs(dy) + 1.0, 1.0), 0.0, 1.0)) if reach_toward > 0 else 0.0
    else:
        vert_dev = 0.15

    lat_pos = float(np.clip(abs(float(p_def["center"][0]) - float(p_off["center"][0])) / max(w_off * 1.2, 1.0), 0.0, 1.0))
    off_disp = float(np.clip(pair_iou * 1.5, 0.0, 1.0))

    if ball_pt is not None and len(wr_def) > 0:
        def_ball = float(np.clip(1.0 - float(np.min(np.linalg.norm(wr_def - ball_pt, axis=-1))) / max(h_def * 0.75, 1.0), 0.0, 1.0))
    else:
        def_ball = 0.20

    wr_off = valid_points(kpts_off, (9, 10))
    if ball_pt is not None and len(wr_off) > 0:
        off_ball = float(np.clip(1.0 - float(np.min(np.linalg.norm(wr_off - ball_pt, axis=-1))) / max(h_off * 0.65, 1.0), 0.0, 1.0))
    else:
        off_ball = 0.45

    ball_vs_body = float(np.clip(cyl_pen * (1.0 - 0.5 * def_ball), 0.0, 1.0))
    contact_core = 0.35 * arm_contact + 0.30 * cyl_pen + 0.20 * landing_viol + 0.15 * vert_dev
    apex_score = float(np.clip(0.75 * contact_core + 0.25 * max(0.0, p_off["elev"]), 0.0, 1.0))

    feats[0] = arm_contact
    feats[1] = cyl_pen
    feats[2] = landing_viol
    feats[3] = vert_dev
    feats[4] = lat_pos
    feats[5] = off_disp
    feats[6] = def_ball
    feats[7] = off_ball
    feats[8] = ball_vs_body
    feats[9] = 0.50
    feats[10] = float(np.clip(pair_iou * 1.2, 0.0, 1.0))
    feats[11] = apex_score

    saliency = float(max(0.0, 0.55 * contact_core + 0.45 * max(0.0, best_rank)))
    return roi_box, saliency, feats, pair_mid


def interpolate_ball_within_shot(
    raw_balls: List[Optional[Tuple[float, float, float]]],
    frame_indices: List[int],
    fps: float,
    diag: float,
) -> List[Optional[np.ndarray]]:
    """Linearly interpolate missing basketball detections strictly inside a single camera shot."""
    track: List[Optional[np.ndarray]] = [
        (np.array([b[0], b[1]], dtype=np.float32) if b is not None else None) for b in raw_balls
    ]
    anchors = [t for t, pt in enumerate(track) if pt is not None]
    for i in range(len(anchors) - 1):
        t0, t1 = anchors[i], anchors[i + 1]
        if t1 - t0 <= 1:
            continue
        dt = abs(frame_indices[t1] - frame_indices[t0]) / max(fps, 1.0)
        if dt > 1.4:
            continue
        p0, p1 = track[t0], track[t1]
        assert p0 is not None and p1 is not None
        if float(np.linalg.norm(p1 - p0)) > max(0.28 * diag, dt * 1100.0):
            continue
        denom = max(float(frame_indices[t1] - frame_indices[t0]), 1.0)
        for k in range(t0 + 1, t1):
            alpha = float(frame_indices[k] - frame_indices[t0]) / denom
            track[k] = (1.0 - alpha) * p0 + alpha * p1
    return track


def compute_square_crop(
    roi_box: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin_scale: float = 1.22,
) -> Tuple[int, int, int, int]:
    """Convert `[x1, y1, x2, y2]` into a square crop clamped inside `[0, W] x [0, H]`."""
    x1, y1, x2, y2 = float(roi_box[0]), float(roi_box[1]), float(roi_box[2]), float(roi_box[3])
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, min(frame_w, frame_h) * 0.28) * margin_scale
    side = min(side, float(min(frame_w, frame_h)))
    half = 0.5 * side
    nx1 = max(0.0, min(cx - half, frame_w - side))
    ny1 = max(0.0, min(cy - half, frame_h - side))
    nx2 = min(float(frame_w), nx1 + side)
    ny2 = min(float(frame_h), ny1 + side)
    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))


def smooth_boxes_ema(boxes: np.ndarray, alpha: float = 0.32) -> np.ndarray:
    """Bidirectional EMA box smoothing."""
    if len(boxes) <= 1:
        return boxes.copy()
    fwd = np.zeros_like(boxes, dtype=np.float32)
    fwd[0] = boxes[0]
    for t in range(1, len(boxes)):
        fwd[t] = alpha * boxes[t] + (1.0 - alpha) * fwd[t - 1]
    bwd = np.zeros_like(boxes, dtype=np.float32)
    bwd[-1] = fwd[-1]
    for t in range(len(boxes) - 2, -1, -1):
        bwd[t] = alpha * fwd[t] + (1.0 - alpha) * bwd[t + 1]
    return bwd


def decode_replay_scan_frames(
    video_path: Path,
    scan_fps: float = 5.0,
    max_clip_sec: float = 60.0,
    max_scan_samples: int = 160,
    scan_w: int = 800,
    scan_h: int = 450,
) -> Optional[Dict[str, Any]]:
    """Decode downscaled scan frames across the entire replay video at constant physical rate (`scan_fps`)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Adapt scan resolution to never upscale low-res SD clips unnecessarily
    eff_scan_w = min(scan_w, max(320, frame_w))
    eff_scan_h = min(scan_h, max(180, frame_h))

    max_horizon = int(round(fps * max_clip_sec))
    eff_total = min(total_frames, max_horizon) if total_frames > 0 else max_horizon
    eff_total = max(eff_total, 2)

    desired_samples = int(round((eff_total / max(fps, 1.0)) * scan_fps))
    num_samples = int(np.clip(desired_samples, 16, max_scan_samples))
    target_indices = sorted(
        set(np.linspace(0, max(0, eff_total - 1), num=min(num_samples, eff_total)).round().astype(int).tolist())
    )
    target_set = set(target_indices)
    max_needed = target_indices[-1] if target_indices else 0

    sampled_indices: List[int] = []
    small_frames: List[np.ndarray] = []

    idx = 0
    while idx <= max_needed:
        if idx in target_set:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != eff_scan_w or frame.shape[0] != eff_scan_h:
                small = cv2.resize(frame, (eff_scan_w, eff_scan_h), interpolation=cv2.INTER_AREA)
            else:
                small = frame
            small_frames.append(small)
            sampled_indices.append(idx)
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()

    if not small_frames:
        return None

    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": max(total_frames, idx),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "scale_x": float(frame_w) / float(eff_scan_w),
        "scale_y": float(frame_h) / float(eff_scan_h),
        "sampled_indices": sampled_indices,
        "small_frames": small_frames,
    }


def extract_replay_action_tubes(
    video_path: Path | str,
    pose_model: Any,
    det_model: Optional[Any] = None,
    angles_per_clip: int = 2,
    base_window_sec: float = 1.6,
    num_frames: int = 16,
    crop_size: int = 224,
    scan_fps: float = 5.0,
    max_scan_samples: int = 160,
    yolo_batch_size: int = 16,
    device: str = "auto",
) -> List[Dict[str, Any]]:
    """
    Extract up to `angles_per_clip` shot-bounded `(3, num_frames, crop_size, crop_size)` RGB
    action tubes + `(num_frames, 12)` biomechanical pose sequences from a single multi-angle
    NBA Replay Archive video clip.

    Returns a list of dicts sorted by descending contact saliency:
      - `video`: `torch.Tensor` uint8 `(3, T, H, W)`
      - `pose_features`: `torch.Tensor` float32 `(T, 12)`
      - `angle_idx`: int (`0`, `1`, ...)
      - `shot_start_frame`, `shot_end_frame`, `peak_frame`: int
      - `sampled_indices`: `List[int]`
      - `crop_boxes`: `List[Tuple[int, int, int, int]]`
      - `fps`, `est_speed_ratio`, `saliency`: float
    """
    vpath = Path(video_path)
    scan = decode_replay_scan_frames(
        video_path=vpath,
        scan_fps=scan_fps,
        max_scan_samples=max_scan_samples,
    )
    if scan is None:
        raise RuntimeError(f"Could not decode frames from {vpath}")

    small_frames: List[np.ndarray] = scan["small_frames"]
    sampled_indices: List[int] = scan["sampled_indices"]
    fps: float = float(scan["fps"])
    frame_w: int = int(scan["frame_w"])
    frame_h: int = int(scan["frame_h"])
    scale_x: float = float(scan["scale_x"])
    scale_y: float = float(scan["scale_y"])
    diag = float(np.hypot(frame_w, frame_h))

    shots = segment_replay_shots(small_frames, sampled_indices, fps=fps)

    yolo_dev = select_yolo_device(device)
    batch_sz = max(1, int(yolo_batch_size))
    all_pose: List[Any] = []
    all_det: List[Any] = []

    with torch.inference_mode():
        for b_start in range(0, len(small_frames), batch_sz):
            b_imgs = small_frames[b_start : b_start + batch_sz]
            all_pose.extend(
                pose_model.predict(b_imgs, conf=0.12, imgsz=640, verbose=False, device=yolo_dev)
            )
            if det_model is not None and det_model is not pose_model:
                all_det.extend(
                    det_model.predict(
                        b_imgs,
                        conf=0.14,
                        imgsz=640,
                        classes=[COCO_PERSON_CLS, COCO_SPORTS_BALL_CLS],
                        verbose=False,
                        device=yolo_dev,
                    )
                )
            else:
                all_det.extend([None] * len(b_imgs))

    # Score every shot strictly within its own camera-cut boundaries
    shot_candidates: List[Dict[str, Any]] = []
    for shot in shots:
        lo, hi = int(shot["scan_lo"]), int(shot["scan_hi"])
        s_indices = sampled_indices[lo : hi + 1]
        s_frames = small_frames[lo : hi + 1]
        if len(s_indices) == 0:
            continue

        rims = [detect_rim_hsv(fr, frame_w, frame_h) for fr in s_frames]
        raw_balls = [
            extract_verified_ball(
                all_det[lo + k],
                scale_x=scale_x,
                scale_y=scale_y,
                frame_w=frame_w,
                frame_h=frame_h,
                bgr_frame=s_frames[k],
                rim_pt=rims[k],
            )
            for k in range(len(s_frames))
        ]
        ball_track = interpolate_ball_within_shot(raw_balls, s_indices, fps=fps, diag=diag)

        s_rois: List[np.ndarray] = []
        s_scores: List[float] = []
        s_feats: List[np.ndarray] = []
        valid_court_frames = 0
        prev_center: Optional[np.ndarray] = None

        for k in range(len(s_frames)):
            players, _ = collect_players_in_frame(
                pose_res=all_pose[lo + k],
                det_res=all_det[lo + k],
                frame_w=frame_w,
                frame_h=frame_h,
                scale_x=scale_x,
                scale_y=scale_y,
                bgr_frame=s_frames[k],
            )
            if len(players) >= 2:
                valid_court_frames += 1
            roi, sal, fvec, prev_center = analyze_contact_pair(
                players=players,
                ball_pt=ball_track[k],
                rim_pt=rims[k],
                prev_center=prev_center,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            s_rois.append(roi)
            s_scores.append(sal)
            s_feats.append(fvec)

        if valid_court_frames == 0:
            continue

        # Interpolate trajectories strictly over `[start_frame .. end_frame]`
        start_f, end_f = int(shot["start_frame"]), int(shot["end_frame"])
        shot_len = max(1, end_f - start_f + 1)
        local_x = np.arange(start_f, end_f + 1, dtype=np.float32)
        scan_x = np.array(s_indices, dtype=np.float32)

        roi_arr = np.stack(s_rois, axis=0)
        dense_rois = np.zeros((shot_len, 4), dtype=np.float32)
        for c in range(4):
            dense_rois[:, c] = np.interp(local_x, scan_x, roi_arr[:, c])

        feat_arr = np.stack(s_feats, axis=0)
        dense_feats = np.zeros((shot_len, POSE_FEATURES), dtype=np.float32)
        for d in range(POSE_FEATURES):
            dense_feats[:, d] = np.interp(local_x, scan_x, feat_arr[:, d])

        # Time-normalized closing velocity (dim 9) and impact jolt (dim 10)
        dt_frame = 1.0 / max(fps, 1.0)
        if shot_len >= 3:
            dcyl_dt = np.gradient(dense_feats[:, 1], dt_frame)
            roi_cy = 0.5 * (dense_rois[:, 1] + dense_rois[:, 3])
            dcy_dt = np.abs(np.gradient(roi_cy, dt_frame)) / max(float(frame_h) * 0.60, 1.0)
            dense_feats[:, 9] = np.clip(0.5 + 0.14 * dcyl_dt, 0.0, 1.0)
            dense_feats[:, 10] = np.clip(dcy_dt, 0.0, 1.0)

        dense_scores = np.interp(local_x, scan_x, np.array(s_scores, dtype=np.float32))
        k_len = max(3, int(round(fps * 0.40)))
        smooth_scores = np.convolve(dense_scores, np.ones(k_len, dtype=np.float32) / k_len, mode="same")
        peak_local = int(np.argmax(smooth_scores))
        peak_frame = start_f + peak_local
        peak_saliency = float(smooth_scores[peak_local])

        # Adapt window length for slow-motion shots so physical motion coverage stays consistent
        est_speed = float(shot["est_speed_ratio"])
        eff_window_sec = float(np.clip(base_window_sec / max(est_speed, 0.50), base_window_sec, base_window_sec * 1.75))
        half_w = max(num_frames // 2, int(round(fps * eff_window_sec * 0.5)))

        w_start = max(start_f, peak_frame - half_w)
        w_end = min(end_f, w_start + 2 * half_w)
        w_start = max(start_f, w_end - 2 * half_w)

        tube_indices = np.linspace(w_start, w_end, num=num_frames).round().astype(int).tolist()
        stabilized_rois = smooth_boxes_ema(dense_rois, alpha=0.30)

        crop_boxes = [
            compute_square_crop(stabilized_rois[min(shot_len - 1, max(0, f - start_f))], frame_w, frame_h)
            for f in tube_indices
        ]
        sampled_feats = np.stack(
            [dense_feats[min(shot_len - 1, max(0, f - start_f))] for f in tube_indices],
            axis=0,
        )

        shot_candidates.append({
            "shot_start_frame": start_f,
            "shot_end_frame": end_f,
            "peak_frame": peak_frame,
            "sampled_indices": tube_indices,
            "crop_boxes": crop_boxes,
            "pose_features": torch.from_numpy(sampled_feats).float(),
            "fps": fps,
            "est_speed_ratio": est_speed,
            "saliency": peak_saliency,
        })

    # Fallback if no multi-player shot passed filters (use full clip midpoint)
    if not shot_candidates:
        total_f = int(scan["total_frames"])
        mid = total_f // 2
        half_w = max(num_frames // 2, int(round(fps * base_window_sec * 0.5)))
        w_start = max(0, mid - half_w)
        w_end = min(max(0, total_f - 1), mid + half_w)
        tube_indices = np.linspace(w_start, w_end, num=num_frames).round().astype(int).tolist()
        default_crop = compute_square_crop(
            np.array([frame_w * 0.18, frame_h * 0.14, frame_w * 0.82, frame_h * 0.86], dtype=np.float32),
            frame_w,
            frame_h,
        )
        shot_candidates.append({
            "shot_start_frame": 0,
            "shot_end_frame": max(0, total_f - 1),
            "peak_frame": mid,
            "sampled_indices": tube_indices,
            "crop_boxes": [default_crop for _ in tube_indices],
            "pose_features": torch.zeros((num_frames, POSE_FEATURES), dtype=torch.float32),
            "fps": fps,
            "est_speed_ratio": 1.0,
            "saliency": 0.0,
        })

    # Keep top `angles_per_clip` highest-saliency distinct camera shots
    shot_candidates.sort(key=lambda s: float(s["saliency"]), reverse=True)
    selected = shot_candidates[: max(1, int(angles_per_clip))]

    # Single Pass-2 sequential decode covering all needed frames across selected shots
    needed_frames: Set[int] = set()
    for s in selected:
        needed_frames.update(s["sampled_indices"])
    max_frame_needed = max(needed_frames) if needed_frames else 0

    decoded_full: Dict[int, np.ndarray] = {}
    cap2 = cv2.VideoCapture(str(vpath))
    cur_f = 0
    while cur_f <= max_frame_needed:
        if cur_f in needed_frames:
            ret, frm = cap2.read()
            if not ret:
                break
            decoded_full[cur_f] = frm
        else:
            if not cap2.grab():
                break
        cur_f += 1
    cap2.release()

    fallback_bgr = next(iter(decoded_full.values())) if decoded_full else np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    results: List[Dict[str, Any]] = []
    for angle_idx, s in enumerate(selected):
        rgb_frames: List[np.ndarray] = []
        for f_idx, (x1, y1, x2, y2) in zip(s["sampled_indices"], s["crop_boxes"]):
            frm = decoded_full.get(f_idx, fallback_bgr)
            crop = frm[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frm
            resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            rgb_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        stacked = np.stack(rgb_frames, axis=0)  # (T, H, W, 3)
        video_tensor = torch.from_numpy(stacked).permute(3, 0, 1, 2).contiguous()  # (3, T, H, W) uint8
        results.append({
            "video": video_tensor,
            "pose_features": s["pose_features"],
            "angle_idx": angle_idx,
            "shot_start_frame": s["shot_start_frame"],
            "shot_end_frame": s["shot_end_frame"],
            "peak_frame": s["peak_frame"],
            "sampled_indices": s["sampled_indices"],
            "crop_boxes": s["crop_boxes"],
            "fps": s["fps"],
            "est_speed_ratio": s["est_speed_ratio"],
            "saliency": s["saliency"],
        })

    return results


def synthesize_fallback_tube(
    play_id: str,
    label: int,
    angle_idx: int = 0,
    num_frames: int = 16,
    crop_size: int = 224,
) -> Dict[str, Any]:
    """Deterministic offline synthetic tube for smoke-testing when raw MP4s are absent."""
    seed = int(hashlib.md5(f"{play_id}_{angle_idx}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    t_axis = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    env = np.exp(-((t_axis - 0.52) ** 2) / 0.05).astype(np.float32)
    scale = 0.78 if label == 1 else 0.30

    feats = np.zeros((num_frames, POSE_FEATURES), dtype=np.float32)
    for d in range(POSE_FEATURES):
        feats[:, d] = np.clip(scale * env + 0.12 + rng.normal(0.0, 0.04, size=num_frames), 0.0, 1.0)

    frames = np.zeros((num_frames, crop_size, crop_size, 3), dtype=np.uint8)
    for t in range(num_frames):
        canvas = np.full((crop_size, crop_size, 3), (36, 42, 50), dtype=np.uint8)
        sep = int((1.0 - env[t] * scale) * 38)
        c1, c2 = crop_size // 2 - max(6, sep), crop_size // 2 + max(6, sep)
        cv2.rectangle(canvas, (c1 - 20, 62), (c1 + 20, 168), (220, 145, 70), -1)
        cv2.rectangle(canvas, (c2 - 20, 64), (c2 + 20, 168), (75, 165, 230), -1)
        frames[t] = canvas

    return {
        "video": torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous(),
        "pose_features": torch.from_numpy(feats).float(),
        "angle_idx": angle_idx,
        "shot_start_frame": 0,
        "shot_end_frame": 120,
        "peak_frame": 60,
        "sampled_indices": list(range(12, 12 + num_frames * 4, 4)),
        "crop_boxes": [(320, 140, 820, 640) for _ in range(num_frames)],
        "fps": 60.0,
        "est_speed_ratio": 1.0,
        "saliency": float(scale),
        "is_synthetic": True,
    }


def select_balanced_clips(rows: List[Dict[str, str]], max_clips: int) -> List[Dict[str, str]]:
    """Prioritize rows whose local raw MP4 exists and interleave `foul` and `no_foul` 1:1."""
    def _has_mp4(r: Dict[str, str]) -> int:
        vp = resolve_project_path(r.get("video_path", ""))
        return 0 if (vp.exists() and vp.stat().st_size > 10_240) else 1

    ordered = sorted(rows, key=_has_mp4)
    if max_clips <= 0 or len(ordered) <= max_clips:
        return ordered
    pos = [r for r in ordered if int(r.get("label", 0)) == 1]
    neg = [r for r in ordered if int(r.get("label", 0)) == 0]
    half = max(1, max_clips // 2)
    sel_pos = pos[:half]
    sel_neg = neg[: max_clips - len(sel_pos)]
    if len(sel_pos) + len(sel_neg) < max_clips:
        rem = max_clips - (len(sel_pos) + len(sel_neg))
        sel_pos.extend(pos[len(sel_pos) : len(sel_pos) + rem])
    out: List[Dict[str, str]] = []
    for i in range(max(len(sel_pos), len(sel_neg))):
        if i < len(sel_neg):
            out.append(sel_neg[i])
        if i < len(sel_pos):
            out.append(sel_pos[i])
    return out[:max_clips]


def write_processed_csv(csv_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROCESSED_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def preprocess_replay_dataset(
    metadata_csv: Path = BASE_DIR / "data" / "replay_metadata.csv",
    processed_dir: Path = BASE_DIR / "data" / "processed",
    processed_metadata_csv: Path = BASE_DIR / "data" / "processed" / "processed_metadata.csv",
    test_dir: Optional[Path] = None,
    yolo_weights: str = "yolo11n.pt",
    pose_weights: str = "yolo11n-pose.pt",
    angles_per_clip: int = 2,
    base_window_sec: float = 1.6,
    num_frames: int = 16,
    crop_size: int = 224,
    scan_fps: float = 5.0,
    max_scan_samples: int = 160,
    max_clips: int = 0,
    yolo_batch_size: int = 16,
    device: str = "auto",
    synthetic_fallback: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process raw NBA Replay Archive MP4s into shot-bounded multi-angle `.pt` action tubes.
    Strictly excludes any video stems present in `data/test/` to prevent data leakage.
    """
    metadata_csv = resolve_project_path(metadata_csv)
    processed_dir = resolve_project_path(processed_dir)
    processed_metadata_csv = resolve_project_path(processed_metadata_csv)
    active_test_dir = resolve_project_path(test_dir) if test_dir is not None else resolve_project_path(BASE_DIR / "data" / "test")

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Replay metadata CSV not found: {metadata_csv}. Run `1_scrape_replays.py` first."
        )

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Strict held-out test segregation
    test_stems: Set[str] = set()
    if active_test_dir.exists():
        test_stems = {
            p.stem
            for p in active_test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")
        }
    if test_stems:
        all_rows = [
            r for r in all_rows
            if r.get("play_id", "").strip() not in test_stems
            and Path(r.get("video_path", "")).stem not in test_stems
        ]
        logger.info("Excluded %d held-out test stems from training preprocessing.", len(test_stems))

    rows = select_balanced_clips(all_rows, max_clips=max_clips)

    pose_model = None
    det_model = None
    has_raw_video = any(resolve_project_path(r.get("video_path", "")).exists() for r in rows)
    if has_raw_video and YOLO is not None:
        pose_p = resolve_project_path(pose_weights)
        det_p = resolve_project_path(yolo_weights)
        active_pose = str(pose_p) if pose_p.exists() else pose_weights
        active_det = str(det_p) if det_p.exists() else yolo_weights
        yolo_dev = select_yolo_device(device)
        if yolo_dev == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 12.0 and yolo_batch_size < 64:
                yolo_batch_size = 64
        logger.info("Loading YOLO models (device=%s, batch=%d): pose=%s | det=%s", yolo_dev, yolo_batch_size, active_pose, active_det)
        pose_model = YOLO(active_pose)
        det_model = YOLO(active_det)

    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "foul").mkdir(parents=True, exist_ok=True)
    (processed_dir / "no_foul").mkdir(parents=True, exist_ok=True)

    processed_rows: List[Dict[str, Any]] = []
    logger.info(
        "Preprocessing %d replay clips (angles_per_clip=%d, scan_fps=%.1f, crop=%dx%d)...",
        len(rows),
        angles_per_clip,
        scan_fps,
        crop_size,
        crop_size,
    )

    for idx, row in enumerate(rows, start=1):
        play_id = str(row["play_id"]).strip()
        game_id = str(row.get("game_id", "")).strip() or play_id.split("_")[0]
        label = int(row["label"])
        class_dir = "foul" if label == 1 else "no_foul"
        video_path = resolve_project_path(row.get("video_path", ""))

        # Check if cached non-synthetic `.pt` files already exist for this play_id
        expected_first = processed_dir / class_dir / (f"{play_id}_a0.pt" if angles_per_clip > 1 else f"{play_id}.pt")
        cached_pts = sorted((processed_dir / class_dir).glob(f"{play_id}_a*.pt")) if angles_per_clip > 1 else ([expected_first] if expected_first.exists() else [])
        valid_cached: List[Dict[str, Any]] = []
        for cp in cached_pts:
            try:
                payload = torch.load(cp, map_location="cpu", weights_only=False)
                if isinstance(payload, dict) and not payload.get("is_synthetic", False):
                    valid_cached.append({
                        "sample_id": cp.stem,
                        "play_id": play_id,
                        "game_id": game_id,
                        "season": row.get("season", ""),
                        "trigger": row.get("trigger", ""),
                        "outcome_category": row.get("outcome_category", ""),
                        "angle_idx": int(payload.get("angle_idx", 0)),
                        "shot_start_frame": int(payload.get("shot_start_frame", 0)),
                        "shot_end_frame": int(payload.get("shot_end_frame", 0)),
                        "peak_frame": int(payload.get("peak_frame", 0)),
                        "fps": round(float(payload.get("fps", 60.0)), 2),
                        "est_speed_ratio": round(float(payload.get("est_speed_ratio", 1.0)), 3),
                        "label": label,
                        "video_path": row.get("video_path", ""),
                        "tensor_path": (Path("data") / "processed" / class_dir / cp.name).as_posix(),
                    })
            except Exception:
                continue

        if valid_cached:
            processed_rows.extend(valid_cached)
            continue

        if video_path.exists() and video_path.stat().st_size > 10_240 and pose_model is not None:
            try:
                tubes = extract_replay_action_tubes(
                    video_path=video_path,
                    pose_model=pose_model,
                    det_model=det_model,
                    angles_per_clip=angles_per_clip,
                    base_window_sec=base_window_sec,
                    num_frames=num_frames,
                    crop_size=crop_size,
                    scan_fps=scan_fps,
                    max_scan_samples=max_scan_samples,
                    yolo_batch_size=yolo_batch_size,
                    device=device,
                )
            except Exception as exc:
                logger.warning("[%d/%d] Failed preprocessing %s: %s", idx, len(rows), play_id, exc)
                continue

            for t_item in tubes:
                a_idx = int(t_item["angle_idx"])
                sample_id = f"{play_id}_a{a_idx}" if angles_per_clip > 1 else play_id
                out_pt = processed_dir / class_dir / f"{sample_id}.pt"
                torch.save(
                    {
                        "video": t_item["video"],
                        "pose_features": t_item["pose_features"],
                        "label": label,
                        "sample_id": sample_id,
                        "play_id": play_id,
                        "game_id": game_id,
                        "angle_idx": a_idx,
                        "shot_start_frame": t_item["shot_start_frame"],
                        "shot_end_frame": t_item["shot_end_frame"],
                        "peak_frame": t_item["peak_frame"],
                        "sampled_indices": t_item["sampled_indices"],
                        "crop_boxes": t_item["crop_boxes"],
                        "fps": t_item["fps"],
                        "est_speed_ratio": t_item["est_speed_ratio"],
                        "saliency": t_item["saliency"],
                        "is_synthetic": False,
                    },
                    out_pt,
                )
                processed_rows.append({
                    "sample_id": sample_id,
                    "play_id": play_id,
                    "game_id": game_id,
                    "season": row.get("season", ""),
                    "trigger": row.get("trigger", ""),
                    "outcome_category": row.get("outcome_category", ""),
                    "angle_idx": a_idx,
                    "shot_start_frame": t_item["shot_start_frame"],
                    "shot_end_frame": t_item["shot_end_frame"],
                    "peak_frame": t_item["peak_frame"],
                    "fps": round(float(t_item["fps"]), 2),
                    "est_speed_ratio": round(float(t_item["est_speed_ratio"]), 3),
                    "label": label,
                    "video_path": row.get("video_path", ""),
                    "tensor_path": (Path("data") / "processed" / class_dir / f"{sample_id}.pt").as_posix(),
                })

            if idx % 10 == 0 or idx == len(rows):
                write_processed_csv(processed_metadata_csv, processed_rows)
                logger.info("[%d/%d] Processed %s -> %d angle tube(s) (total tubes: %d)", idx, len(rows), play_id, len(tubes), len(processed_rows))

        elif synthetic_fallback:
            for a_idx in range(max(1, angles_per_clip)):
                t_item = synthesize_fallback_tube(play_id, label, angle_idx=a_idx, num_frames=num_frames, crop_size=crop_size)
                sample_id = f"{play_id}_a{a_idx}" if angles_per_clip > 1 else play_id
                out_pt = processed_dir / class_dir / f"{sample_id}.pt"
                torch.save(
                    {
                        **t_item,
                        "label": label,
                        "sample_id": sample_id,
                        "play_id": play_id,
                        "game_id": game_id,
                    },
                    out_pt,
                )
                processed_rows.append({
                    "sample_id": sample_id,
                    "play_id": play_id,
                    "game_id": game_id,
                    "season": row.get("season", ""),
                    "trigger": row.get("trigger", ""),
                    "outcome_category": row.get("outcome_category", ""),
                    "angle_idx": a_idx,
                    "shot_start_frame": t_item["shot_start_frame"],
                    "shot_end_frame": t_item["shot_end_frame"],
                    "peak_frame": t_item["peak_frame"],
                    "fps": 60.0,
                    "est_speed_ratio": 1.0,
                    "label": label,
                    "video_path": row.get("video_path", ""),
                    "tensor_path": (Path("data") / "processed" / class_dir / f"{sample_id}.pt").as_posix(),
                })
        else:
            logger.debug("Skipping missing raw MP4: %s", video_path)

    write_processed_csv(processed_metadata_csv, processed_rows)
    logger.info("Stage 2 complete: %d action tubes indexed in %s", len(processed_rows), processed_metadata_csv)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return processed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Multi-Angle Shot-Segmented Action-Tube & Pose Extractor for NBA Replay Archive"
    )
    parser.add_argument("--metadata-csv", type=Path, default=BASE_DIR / "data" / "replay_metadata.csv")
    parser.add_argument("--processed-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument(
        "--processed-metadata-csv",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "processed_metadata.csv",
    )
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "data" / "test")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n.pt")
    parser.add_argument("--pose-weights", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--angles-per-clip", type=int, default=2, help="Distinct camera angles extracted per replay clip (default: 2)")
    parser.add_argument("--window-sec", type=float, default=1.6, help="Base temporal window in seconds around contact apex (default: 1.6)")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames sampled per action tube (default: 16)")
    parser.add_argument("--crop-size", type=int, default=224, help="Square RGB crop size (default: 224)")
    parser.add_argument("--scan-fps", type=float, default=5.0, help="Physical scan frequency in Hz (default: 5.0)")
    parser.add_argument("--max-scan-samples", type=int, default=160, help="Max scan frames per replay clip (default: 160)")
    parser.add_argument("--max-clips", type=int, default=0, help="Max replay clips to process (0 = all downloaded)")
    parser.add_argument("--yolo-batch-size", type=int, default=16, help="YOLO batch size (default: 16)")
    parser.add_argument("--device", type=str, default="auto", help="YOLO device ('auto', 'cuda', or 'cpu')")
    parser.add_argument("--synthetic-fallback", action="store_true", help="Generate deterministic synthetic .pt tensors when MP4s are absent")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    preprocess_replay_dataset(
        metadata_csv=cli_args.metadata_csv,
        processed_dir=cli_args.processed_dir,
        processed_metadata_csv=cli_args.processed_metadata_csv,
        test_dir=cli_args.test_dir,
        yolo_weights=cli_args.yolo_weights,
        pose_weights=cli_args.pose_weights,
        angles_per_clip=cli_args.angles_per_clip,
        base_window_sec=cli_args.window_sec,
        num_frames=cli_args.num_frames,
        crop_size=cli_args.crop_size,
        scan_fps=cli_args.scan_fps,
        max_scan_samples=cli_args.max_scan_samples,
        max_clips=cli_args.max_clips,
        yolo_batch_size=cli_args.yolo_batch_size,
        device=cli_args.device,
        synthetic_fallback=cli_args.synthetic_fallback,
    )
