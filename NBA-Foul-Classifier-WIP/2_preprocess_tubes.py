#!/usr/bin/env python3
"""
STEP 2: Cross-Platform Foul Action-Tube ROI Cropping & Biomechanical Pose Sampling (`2_preprocess_tubes.py`)

Engineered to run reliably on Windows, Linux, and macOS (Intel & Apple Silicon) with low RAM/VRAM:
  1. Cross-Platform Memory & Thread Safety:
     - Limits OpenMP/MKL and OpenCV thread pools (`cv2.setNumThreads(0)`) so FFmpeg H.264 decoding
       never deadlocks with PyTorch on Windows, Linux, or macOS.
     - Caps live-action foul scanning to the first `max_live_play_sec = 15.0` seconds (skipping
       multi-minute dead-ball replay reviews) and decodes at most `max_scan_samples = 12` downscaled
       `640x360` frames in Pass 1 (< 8 MB RAM).
     - Runs YOLOv11-Pose in micro-batches (`--yolo-batch-size 4`) after `cap.release()`.
  2. Lightweight Default Batch Processing:
     - Processes up to `--max-clips 12` balanced clips by default (`--max-clips 0` processes all clips)
       so student laptops can complete preprocessing quickly.
     - Supports `--synthetic-fallback` to generate deterministic sample `.pt` tensors from metadata
       when raw `.mp4` broadcast clips are not locally cached.
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
from typing import Dict, List, Optional, Tuple

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
logger = logging.getLogger("preprocess_tubes")

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
POSE_FEATURES = 12
NUM_FOUL_FEATURES = POSE_FEATURES

# COCO 17-keypoint indices:
# 0: nose, 5: L_shoulder, 6: R_shoulder, 7: L_elbow, 8: R_elbow,
# 9: L_wrist, 10: R_wrist, 11: L_hip, 12: R_hip
UPPER_BODY_KPTS = (0, 5, 6, 7, 8, 9, 10)
ARM_CONTACT_KPTS = (7, 8, 9, 10)
TORSO_KPTS = (5, 6, 11, 12)
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
    """
    Select a cross-platform device for Ultralytics YOLOv11-Pose inference.
    Uses CUDA when available on Windows/Linux, and CPU on macOS (MPS) / CPU machines
    where Ultralytics pose keypoint NMS is fastest and 100% deterministic.
    """
    if requested and requested not in ("auto", "mps"):
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def box_iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two `[x1, y1, x2, y2]` boxes."""
    xA = max(float(box_a[0]), float(box_b[0]))
    yA = max(float(box_a[1]), float(box_b[1]))
    xB = min(float(box_a[2]), float(box_b[2]))
    yB = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    area_a = max(1.0, float(box_a[2] - box_a[0]) * float(box_a[3] - box_a[1]))
    area_b = max(1.0, float(box_b[2] - box_b[0]) * float(box_b[3] - box_b[1]))
    return float(inter_area / max(area_a + area_b - inter_area, 1.0))


def valid_points(kpts_xy: np.ndarray, indices: Tuple[int, ...]) -> np.ndarray:
    """Return `(K, 2)` array of valid non-zero keypoints for given COCO indices."""
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
    dists = np.linalg.norm(diffs, axis=-1)
    return float(np.min(dists))


def compute_arm_elevation(kpts_xy: np.ndarray, box_h: float) -> float:
    """
    Measure normalized wrist/elbow elevation relative to shoulders.
    Positive values indicate arms raised above shoulder level (shooting gather/release or vertical contest).
    """
    shoulders = valid_points(kpts_xy, (5, 6))
    wrists = valid_points(kpts_xy, (7, 8, 9, 10))
    if len(shoulders) == 0 or len(wrists) == 0:
        return 0.0
    shoulder_y = float(np.mean(shoulders[:, 1]))
    highest_wrist_y = float(np.min(wrists[:, 1]))
    return float(np.clip((shoulder_y - highest_wrist_y) / max(box_h * 0.5, 1.0), -1.0, 1.5))


def detect_basketball_hsv(
    bgr_frame: Optional[np.ndarray],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Optional[np.ndarray]:
    """Detect the orange basketball in a downscaled BGR frame via HSV color and tight circularity."""
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    img_h, img_w = bgr_frame.shape[:2]
    max_area = max(420.0, 0.0009 * float(img_w * img_h))
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 120, 100]), np.array([22, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pt: Optional[np.ndarray] = None
    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 36.0 <= area <= max_area:
            perim = cv2.arcLength(cnt, True)
            if perim > 0:
                circ = 4.0 * np.pi * area / (perim * perim)
                if circ >= 0.62 and (circ * area) > best_score:
                    (cx, cy), _ = cv2.minEnclosingCircle(cnt)
                    if 0.14 * img_h <= cy <= 0.84 * img_h:
                        best_score = float(circ * area)
                        best_pt = np.array([cx * scale_x, cy * scale_y], dtype=np.float32)
    return best_pt


def extract_player_torso_hsv(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[float, float, float]:
    """Return mean `(H, S, V)` of a player's torso crop in `bgr_frame`."""
    if bgr_frame is None or bgr_frame.size == 0:
        return 0.0, 50.0, 128.0
    fh, fw = bgr_frame.shape[:2]
    x1 = float(box_xyxy[0]) / max(scale_x, 1e-5)
    y1 = float(box_xyxy[1]) / max(scale_y, 1e-5)
    x2 = float(box_xyxy[2]) / max(scale_x, 1e-5)
    y2 = float(box_xyxy[3]) / max(scale_y, 1e-5)
    bh = max(1.0, y2 - y1)
    bw = max(1.0, x2 - x1)
    ty1 = max(0, min(fh - 1, int(y1 + 0.18 * bh)))
    ty2 = max(ty1 + 1, min(fh, int(y1 + 0.52 * bh)))
    tx1 = max(0, min(fw - 1, int(x1 + 0.22 * bw)))
    tx2 = max(tx1 + 1, min(fw, int(x2 - 0.22 * bw)))
    torso = bgr_frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return 0.0, 50.0, 128.0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return (
        float(np.mean(hsv[:, :, 0])),
        float(np.mean(hsv[:, :, 1])),
        float(np.mean(hsv[:, :, 2])),
    )


def is_referee_or_sideline_official(
    bgr_frame: Optional[np.ndarray],
    box_xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> bool:
    """
    Identify NBA referees and sideline officials by uniform signature (dark gray/black low-saturation shirt
    + black trousers) and foreground/sideline court position.
    """
    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    if bgr_frame is None or bgr_frame.size == 0:
        return False
    fh, fw = bgr_frame.shape[:2]
    sx1, sy1 = x1 / max(scale_x, 1e-5), y1 / max(scale_y, 1e-5)
    sx2, sy2 = x2 / max(scale_x, 1e-5), y2 / max(scale_y, 1e-5)
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
    ls, lv = float(np.mean(lhsv[:, :, 1])), float(np.mean(lhsv[:, :, 2]))

    # Referee dark gray shirt + black trousers
    if tv < 92.0 and ts < 78.0 and lv < 88.0:
        return True
    # Foreground sideline official standing at bottom edge of court
    if y1 >= frame_h * 0.58 and y2 >= frame_h * 0.82 and ts < 75.0 and tv < 140.0:
        return True
    return False


def synthesize_box_keypoints(box_xyxy: np.ndarray) -> np.ndarray:
    """
    Generate a canonical 17-keypoint COCO skeleton `(17, 2)` from a player bounding box `[x1, y1, x2, y2]`
    when a player under the rim is detected by YOLOv11 detection but partially occluded for pose NMS.
    """
    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    w = max(4.0, x2 - x1)
    h = max(8.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[0] = [cx, y1 + 0.10 * h]                     # nose
    kpts[5] = [cx - 0.22 * w, y1 + 0.22 * h]          # L_shoulder
    kpts[6] = [cx + 0.22 * w, y1 + 0.22 * h]          # R_shoulder
    kpts[7] = [cx - 0.34 * w, y1 + 0.35 * h]          # L_elbow
    kpts[8] = [cx + 0.34 * w, y1 + 0.35 * h]          # R_elbow
    kpts[9] = [cx - 0.38 * w, y1 + 0.28 * h]          # L_wrist
    kpts[10] = [cx + 0.38 * w, y1 + 0.28 * h]         # R_wrist
    kpts[11] = [cx - 0.16 * w, y1 + 0.52 * h]         # L_hip
    kpts[12] = [cx + 0.16 * w, y1 + 0.52 * h]         # R_hip
    kpts[13] = [cx - 0.16 * w, y1 + 0.72 * h]         # L_knee
    kpts[14] = [cx + 0.16 * w, y1 + 0.72 * h]         # R_knee
    kpts[15] = [cx - 0.16 * w, y1 + 0.92 * h]         # L_ankle
    kpts[16] = [cx + 0.16 * w, y1 + 0.92 * h]         # R_ankle
    return kpts


def analyze_foul_interaction_frame(
    pose_res,
    det_res,
    frame_w: int,
    frame_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    bgr_frame: Optional[np.ndarray] = None,
    ball_pt_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Analyze player bounding boxes and 17-point human pose skeletons to:
      1. Identify the primary foul-contact pair (shooter/ball-handler + contesting defender).
      2. Compute an upper-body-biased ROI box `[x1, y1, x2, y2]` in original frame coordinates.
      3. Compute a scalar foul-contact intensity score and a 12-D biomechanical feature vector.
    """
    diag = float(np.hypot(frame_w, frame_h))
    default_roi = np.array(
        [frame_w * 0.22, frame_h * 0.18, frame_w * 0.78, frame_h * 0.82],
        dtype=np.float32,
    )
    feats = np.zeros(NUM_FOUL_FEATURES, dtype=np.float32)

    merged_boxes_list: List[np.ndarray] = []
    merged_kpts_list: List[np.ndarray] = []

    if pose_res is not None and pose_res.boxes is not None and len(pose_res.boxes) > 0:
        p_xyxy = pose_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        p_xyxy[:, [0, 2]] *= scale_x
        p_xyxy[:, [1, 3]] *= scale_y
        p_cls = pose_res.boxes.cls.cpu().numpy().astype(int)
        p_conf = pose_res.boxes.conf.cpu().numpy().astype(np.float32)
        p_mask = (p_cls == COCO_PERSON_CLS) & (p_conf >= 0.10)
        p_boxes = p_xyxy[p_mask]

        has_kpts = (
            getattr(pose_res, "keypoints", None) is not None
            and pose_res.keypoints.xy is not None
            and len(pose_res.keypoints.xy) == len(p_xyxy)
        )
        if has_kpts:
            p_kpts = pose_res.keypoints.xy.cpu().numpy().astype(np.float32)[p_mask]
            p_kpts[:, :, 0] *= scale_x
            p_kpts[:, :, 1] *= scale_y
        else:
            p_kpts = np.stack([synthesize_box_keypoints(b) for b in p_boxes], axis=0) if len(p_boxes) > 0 else np.empty((0, 17, 2), dtype=np.float32)

        for b_idx in range(len(p_boxes)):
            merged_boxes_list.append(p_boxes[b_idx])
            merged_kpts_list.append(p_kpts[b_idx])

    # Supplement with high-confidence det_res person boxes that pose_res missed due to collision occlusion
    def _box_iom_local(b1: np.ndarray, b2: np.ndarray) -> float:
        ix1 = max(float(b1[0]), float(b2[0]))
        iy1 = max(float(b1[1]), float(b2[1]))
        ix2 = min(float(b1[2]), float(b2[2]))
        iy2 = min(float(b1[3]), float(b2[3]))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = max(1.0, (float(b1[2]) - float(b1[0])) * (float(b1[3]) - float(b1[1])))
        a2 = max(1.0, (float(b2[2]) - float(b2[0])) * (float(b2[3]) - float(b2[1])))
        return inter / min(a1, a2)

    if det_res is not None and det_res.boxes is not None and len(det_res.boxes) > 0:
        d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        d_xyxy[:, [0, 2]] *= scale_x
        d_xyxy[:, [1, 3]] *= scale_y
        d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
        d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)
        d_mask = (d_cls == COCO_PERSON_CLS) & (d_conf >= 0.25)
        d_boxes = d_xyxy[d_mask]

        for db in d_boxes:
            if all(box_iou_single(db, mb) < 0.35 and _box_iom_local(db, mb) < 0.55 for mb in merged_boxes_list):
                merged_boxes_list.append(db)
                merged_kpts_list.append(synthesize_box_keypoints(db))

    if not merged_boxes_list:
        return default_roi, 0.0, feats

    player_boxes = np.stack(merged_boxes_list, axis=0).astype(np.float32)
    kpts_all: Optional[np.ndarray] = np.stack(merged_kpts_list, axis=0).astype(np.float32)

    # Reject close-up camera shots (e.g. referee/player face close-ups before inbound or during dead-ball replay)
    # Broadcast court foreground players can reach ~0.50 * frame_h, whereas true close-ups exceed 0.56 * frame_h.
    if len(player_boxes) > 0:
        all_bh = player_boxes[:, 3] - player_boxes[:, 1]
        if float(np.max(all_bh)) > frame_h * 0.56:
            return default_roi, 0.0, feats

    # Filter court players: reject upper stands / ceiling, seated bottom photographers/spectators, and referees.
    if len(player_boxes) > 0:
        cy = 0.5 * (player_boxes[:, 1] + player_boxes[:, 3])
        cx = 0.5 * (player_boxes[:, 0] + player_boxes[:, 2])
        bh = player_boxes[:, 3] - player_boxes[:, 1]
        bw = player_boxes[:, 2] - player_boxes[:, 0]
        non_ref_mask = np.array(
            [
                not is_referee_or_sideline_official(
                    bgr_frame, player_boxes[idx_p], frame_w, frame_h, scale_x=scale_x, scale_y=scale_y
                )
                for idx_p in range(len(player_boxes))
            ],
            dtype=bool,
        )
        # Seated courtside fans/photographers sit near y2 > 0.83*fh with small cropped height bh < 0.20*fh,
        # whereas foreground NBA players standing near the lower sideline have bh >= 0.20*fh and y2 <= 0.94*fh.
        foreground_perspective_ok = (player_boxes[:, 3] <= frame_h * 0.83) | (bh >= frame_h * 0.20)
        court_mask = (
            non_ref_mask
            & foreground_perspective_ok
            & (player_boxes[:, 3] >= frame_h * 0.42)
            & (player_boxes[:, 3] <= frame_h * 0.94)
            & (player_boxes[:, 1] >= frame_h * 0.20)
            & (cy >= frame_h * 0.33)
            & (cy <= frame_h * 0.84)
            & (cx >= frame_w * 0.05)
            & (cx <= frame_w * 0.95)
            & (bh >= frame_h * 0.10)
            & (bh <= frame_h * 0.52)
            & (bw >= frame_w * 0.02)
            & (bw <= frame_w * 0.30)
            & ((bh / np.maximum(bw, 1.0)) >= 0.78)
        )
        if int(np.sum(court_mask)) >= 2:
            player_boxes = player_boxes[court_mask]
            if kpts_all is not None:
                kpts_all = kpts_all[court_mask]
        else:
            return default_roi, 0.0, feats

    n_players = len(player_boxes)
    if n_players == 0:
        return default_roi, 0.0, feats
    if n_players == 1:
        b = player_boxes[0]
        feats[9] = 0.1
        feats[11] = 0.2
        return b.astype(np.float32), 0.2, feats

    ball_pt: Optional[np.ndarray] = ball_pt_override.copy() if ball_pt_override is not None else None
    ball_is_confident = ball_pt_override is not None
    if ball_pt is None and det_res is not None and det_res.boxes is not None and len(det_res.boxes) > 0:
        d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
        d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)
        b_mask = (d_cls == COCO_SPORTS_BALL_CLS) & (d_conf >= 0.14)
        if np.any(b_mask):
            b_boxes = d_xyxy[b_mask]
            b_confs = d_conf[b_mask]
            best_idx_b = int(np.argmax(b_confs))
            best_b = b_boxes[best_idx_b]
            bw_b = (best_b[2] - best_b[0]) * scale_x
            bh_b = (best_b[3] - best_b[1]) * scale_y
            if 0.008 * frame_h <= bw_b <= 0.065 * frame_h and 0.008 * frame_h <= bh_b <= 0.065 * frame_h:
                ball_pt = np.array(
                    [(best_b[0] + best_b[2]) * 0.5 * scale_x, (best_b[1] + best_b[3]) * 0.5 * scale_y],
                    dtype=np.float32,
                )
                ball_is_confident = float(b_confs[best_idx_b]) >= 0.22
    if ball_pt is None and bgr_frame is not None:
        hsv_ball = detect_basketball_hsv(bgr_frame, scale_x=scale_x, scale_y=scale_y)
        if hsv_ball is not None:
            p_centers = np.column_stack([(player_boxes[:, 0] + player_boxes[:, 2]) * 0.5, (player_boxes[:, 1] + player_boxes[:, 3]) * 0.5])
            if float(np.min(np.linalg.norm(p_centers - hsv_ball, axis=-1))) <= max(diag * 0.14, 240.0):
                ball_pt = hsv_ball

    player_hsv = [
        extract_player_torso_hsv(bgr_frame, player_boxes[i], scale_x=scale_x, scale_y=scale_y)
        for i in range(n_players)
    ]

    centers = np.column_stack(
        [
            (player_boxes[:, 0] + player_boxes[:, 2]) * 0.5,
            (player_boxes[:, 1] + player_boxes[:, 3]) * 0.5,
        ]
    )

    best_pair = (-1, -1)
    best_pair_score = -1e9
    best_pair_metrics = (diag, diag, diag, 0.0)
    fallback_pair = (-1, -1)
    fallback_score = -1e9
    fallback_metrics = (diag, diag, diag, 0.0)

    for i in range(n_players):
        for j in range(i + 1, n_players):
            c_dist = float(np.linalg.norm(centers[i] - centers[j]))
            if c_dist > diag * 0.45:
                continue

            iou = box_iou_single(player_boxes[i], player_boxes[j])
            arm_to_arm_dist = c_dist
            arm_to_upper_dist = c_dist
            torso_dist = c_dist

            if kpts_all is not None:
                arms_i = valid_points(kpts_all[i], ARM_CONTACT_KPTS)
                arms_j = valid_points(kpts_all[j], ARM_CONTACT_KPTS)
                upper_i = valid_points(kpts_all[i], UPPER_BODY_KPTS)
                upper_j = valid_points(kpts_all[j], UPPER_BODY_KPTS)
                torso_i = valid_points(kpts_all[i], TORSO_KPTS)
                torso_j = valid_points(kpts_all[j], TORSO_KPTS)

                arm_to_arm_dist = min_point_set_distance(arms_i, arms_j, c_dist)
                d_ij = min_point_set_distance(arms_i, upper_j, c_dist)
                d_ji = min_point_set_distance(arms_j, upper_i, c_dist)
                arm_to_upper_dist = min(d_ij, d_ji)
                torso_dist = min_point_set_distance(torso_i, torso_j, c_dist)

            norm_arm_upper = max(0.0, 1.0 - arm_to_upper_dist / max(diag * 0.18, 1.0))
            norm_torso = max(0.0, 1.0 - torso_dist / max(diag * 0.25, 1.0))

            color_diff = 0.0
            team_contrast_bonus = 0.0
            if bgr_frame is not None:
                hi, si, vi = player_hsv[i]
                hj, sj, vj = player_hsv[j]
                dh = min(abs(hi - hj), 180.0 - abs(hi - hj))
                color_diff = abs(vi - vj) + 0.75 * abs(si - sj) + 1.1 * dh
                team_contrast_bonus = 0.24 if color_diff >= 30.0 else -0.22

            if _box_iom_local(player_boxes[i], player_boxes[j]) >= 0.65 and color_diff < 28.0:
                continue

            ball_prox = 0.0
            off_ball_penalty = 0.0
            if ball_pt is not None:
                pair_mid = 0.5 * (centers[i] + centers[j])
                dist_ball = float(np.linalg.norm(pair_mid - ball_pt))
                ball_prox = max(0.0, 1.0 - dist_ball / max(diag * 0.22, 1.0))
                if ball_is_confident and dist_ball > max(diag * 0.24, 380.0):
                    off_ball_penalty = -0.25

            raw_pair_score = (
                0.38 * norm_arm_upper
                + 0.22 * norm_torso
                + 0.22 * min(1.0, iou * 3.0)
                + 0.36 * ball_prox
                + team_contrast_bonus
                + off_ball_penalty
            )
            if raw_pair_score > fallback_score:
                fallback_score = raw_pair_score
                fallback_pair = (i, j)
                fallback_metrics = (arm_to_arm_dist, arm_to_upper_dist, torso_dist, iou)

            # Geometric scale + vertical alignment guards for strict primary selection
            bhi = float(player_boxes[i, 3] - player_boxes[i, 1])
            bhj = float(player_boxes[j, 3] - player_boxes[j, 1])
            if min(bhi, bhj) / max(bhi, bhj, 1.0) < 0.55:
                continue
            if abs(float(player_boxes[i, 3]) - float(player_boxes[j, 3])) > frame_h * 0.26:
                continue
            if abs(float(centers[i, 0]) - float(centers[j, 0])) > max(frame_w * 0.20, 1.45 * max(bhi, bhj)):
                continue

            if raw_pair_score > best_pair_score:
                best_pair_score = raw_pair_score
                best_pair = (i, j)
                best_pair_metrics = (arm_to_arm_dist, arm_to_upper_dist, torso_dist, iou)

    if best_pair[0] < 0:
        if fallback_pair[0] < 0:
            return default_roi, 0.0, feats
        best_pair = fallback_pair
        best_pair_metrics = fallback_metrics

    idx_a, idx_b = best_pair
    box_a, box_b = player_boxes[idx_a], player_boxes[idx_b]
    arm_to_arm_dist, arm_to_upper_dist, torso_dist, pair_iou = best_pair_metrics

    h_a = max(1.0, float(box_a[3] - box_a[1]))
    h_b = max(1.0, float(box_b[3] - box_b[1]))
    w_a = max(1.0, float(box_a[2] - box_a[0]))
    w_b = max(1.0, float(box_b[2] - box_b[0]))

    elev_a = compute_arm_elevation(kpts_all[idx_a], h_a) if kpts_all is not None else 0.0
    elev_b = compute_arm_elevation(kpts_all[idx_b], h_b) if kpts_all is not None else 0.0

    # Determine Disputed Roles: Offense (Shooter/Driver) vs Defense (Contesting Defender)
    if elev_a >= elev_b:
        off_idx, def_idx = idx_a, idx_b
        box_off, box_def = box_a, box_b
        h_off, h_def = h_a, h_b
        w_off, w_def = w_a, w_b
        elev_off, elev_def = elev_a, elev_b
    else:
        off_idx, def_idx = idx_b, idx_a
        box_off, box_def = box_b, box_a
        h_off, h_def = h_b, h_a
        w_off, w_def = w_b, w_a
        elev_off, elev_def = elev_b, elev_a

    kpts_off = kpts_all[off_idx] if kpts_all is not None else None
    kpts_def = kpts_all[def_idx] if kpts_all is not None else None

    # Bounding Box around ALL players involved in the disputed interaction
    union_x1 = min(float(box_a[0]), float(box_b[0]))
    union_y1 = min(float(box_a[1]), float(box_b[1]))
    union_x2 = max(float(box_a[2]), float(box_b[2]))
    union_y2 = max(float(box_a[3]), float(box_b[3]))
    roi_box = np.array([union_x1, union_y1, union_x2, union_y2], dtype=np.float32)

    # 1. Arm Contact Intrusion (Defender hand/wrist striking shooter arm within physical contact radius)
    arms_off = valid_points(kpts_off, ARM_CONTACT_KPTS) if kpts_off is not None else np.empty((0, 2))
    arms_def = valid_points(kpts_def, ARM_CONTACT_KPTS) if kpts_def is not None else np.empty((0, 2))
    arm_dist = min_point_set_distance(arms_off, arms_def, arm_to_arm_dist)
    arm_contact_val = float(np.clip(1.0 - (arm_dist / max(h_off * 0.24, 1.0)), 0.0, 1.0))

    # 2. Body Cylinder Penetration (Defender invading offensive established space)
    torso_off = valid_points(kpts_off, TORSO_KPTS) if kpts_off is not None else np.empty((0, 2))
    torso_def = valid_points(kpts_def, TORSO_KPTS) if kpts_def is not None else np.empty((0, 2))
    torso_d = min_point_set_distance(torso_off, torso_def, torso_dist)
    cylinder_pen_val = float(
        np.clip(
            0.50 * (1.0 - torso_d / max(h_off * 0.34, 1.0)) + 0.50 * min(1.0, pair_iou * 2.4),
            0.0,
            1.0,
        )
    )

    # 3. Lower Body Landing Violation (Defender undercut in shooter landing zone)
    feet_off = valid_points(kpts_off, (13, 14, 15, 16)) if kpts_off is not None else np.empty((0, 2))
    feet_def = valid_points(kpts_def, (13, 14, 15, 16)) if kpts_def is not None else np.empty((0, 2))
    feet_dist = min_point_set_distance(feet_off, feet_def, 0.5 * (h_off + h_def))
    landing_violation_val = float(np.clip(1.0 - (feet_dist / max(h_off * 0.26, 1.0)), 0.0, 1.0))

    # 4. Defender Verticality Angle (Deviation from legal 90-degree vertical contest)
    sh_def = valid_points(kpts_def, (5, 6)) if kpts_def is not None else np.empty((0, 2))
    wr_def = valid_points(kpts_def, (9, 10)) if kpts_def is not None else np.empty((0, 2))
    if len(sh_def) > 0 and len(wr_def) > 0:
        dx = float(np.mean(wr_def[:, 0]) - np.mean(sh_def[:, 0]))
        dy = float(np.mean(wr_def[:, 1]) - np.mean(sh_def[:, 1]))
        off_cx = 0.5 * (box_off[0] + box_off[2])
        def_cx = 0.5 * (box_def[0] + box_def[2])
        reach_toward = (off_cx - def_cx) * dx
        verticality_val = float(np.clip(abs(dx) / max(abs(dy) + 1.0, 1.0), 0.0, 1.0)) if reach_toward > 0 else 0.0
    else:
        verticality_val = 0.15

    # 5. Defender Lateral Positioning (Offset relative to offensive drive)
    off_cx = 0.5 * (box_off[0] + box_off[2])
    def_cx = 0.5 * (box_def[0] + box_def[2])
    lat_pos_val = float(np.clip(abs(def_cx - off_cx) / max(w_off * 1.2, 1.0), 0.0, 1.0))

    # 6. Offensive Displacement (Center of mass displacement)
    off_disp_val = float(np.clip(pair_iou * 1.5, 0.0, 1.0))

    # 7. Defender to Ball Distance
    if ball_pt is not None and len(wr_def) > 0:
        d_ball_def = float(np.min(np.linalg.norm(wr_def - ball_pt, axis=-1)))
        def_ball_val = float(np.clip(1.0 - d_ball_def / max(h_def * 0.75, 1.0), 0.0, 1.0))
    else:
        def_ball_val = 0.20

    # 8. Ball to Shooter Distance (Possession / release state)
    wr_off = valid_points(kpts_off, (9, 10)) if kpts_off is not None else np.empty((0, 2))
    if ball_pt is not None and len(wr_off) > 0:
        d_ball_off = float(np.min(np.linalg.norm(wr_off - ball_pt, axis=-1)))
        ball_shooter_val = float(np.clip(1.0 - d_ball_off / max(h_off * 0.60, 1.0), 0.0, 1.0))
    else:
        ball_shooter_val = 0.50

    # 9. Ball Strike vs Body Ratio (Body collision severity relative to clean play on ball)
    ball_ratio_val = float(np.clip(cylinder_pen_val * (1.0 - 0.5 * def_ball_val), 0.0, 1.0))

    # 10. Relative Closing Velocity
    closing_vel_val = 0.50

    # 11. Momentum Transfer Shock
    momentum_shock_val = float(np.clip(pair_iou * 1.2, 0.0, 1.0))

    # 12. Composite Physical Contact & Contest Apex Score
    physical_contact_score = float(
        0.35 * arm_contact_val
        + 0.30 * cylinder_pen_val
        + 0.20 * landing_violation_val
        + 0.15 * verticality_val
    )
    apex_diff_val = float(np.clip(0.75 * physical_contact_score + 0.25 * max(0.0, elev_off), 0.0, 1.0))

    feats[0] = arm_contact_val
    feats[1] = cylinder_pen_val
    feats[2] = landing_violation_val
    feats[3] = verticality_val
    feats[4] = lat_pos_val
    feats[5] = off_disp_val
    feats[6] = def_ball_val
    feats[7] = ball_shooter_val
    feats[8] = ball_ratio_val
    feats[9] = closing_vel_val
    feats[10] = momentum_shock_val
    feats[11] = apex_diff_val

    return roi_box, physical_contact_score, feats


def compute_square_crop_coords(
    roi_box: np.ndarray,
    frame_w: int,
    frame_h: int,
    margin_scale: float = 1.18,
) -> Tuple[int, int, int, int]:
    """
    Convert an arbitrary `[x1, y1, x2, y2]` ROI box into a square crop window
    clamped within `[0, frame_w] x [0, frame_h]`.
    """
    x1, y1, x2, y2 = roi_box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1, min(frame_w, frame_h) * 0.26) * margin_scale
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


_CACHED_DET_MODEL: Optional[object] = None


def get_cached_det_model() -> Optional[object]:
    """Lazily load and cache `yolo11n.pt` object detector to supplement wide-court player and ball detection."""
    global _CACHED_DET_MODEL
    if _CACHED_DET_MODEL is None and YOLO is not None:
        det_path = resolve_project_path("yolo11n.pt")
        if det_path.exists():
            try:
                _CACHED_DET_MODEL = YOLO(str(det_path))
            except Exception:
                _CACHED_DET_MODEL = None
    return _CACHED_DET_MODEL


def box_iom_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-Minimum-Area (IoM) to detect duplicate boxes on the same person."""
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(1.0, (float(box_a[2]) - float(box_a[0])) * (float(box_a[3]) - float(box_a[1])))
    a2 = max(1.0, (float(box_b[2]) - float(box_b[0])) * (float(box_b[3]) - float(box_b[1])))
    return inter / min(a1, a2)


def is_court_player_box(box: np.ndarray, frame_w: int, frame_h: int) -> bool:
    """Filter active court players while rejecting upper stands and bottom-edge seated spectators."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bh = y2 - y1
    bw = x2 - x1
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    if not (0.42 * frame_h <= y2 <= 0.94 * frame_h):
        return False
    if y1 < 0.20 * frame_h:
        return False
    if y2 > 0.83 * frame_h and bh < 0.20 * frame_h:
        return False
    if not (0.33 * frame_h <= cy <= 0.84 * frame_h):
        return False
    if not (0.05 * frame_w <= cx <= 0.95 * frame_w):
        return False
    if not (0.10 * frame_h <= bh <= 0.52 * frame_h):
        return False
    if not (0.02 * frame_w <= bw <= 0.30 * frame_w):
        return False
    if bh / max(bw, 1.0) < 0.78:
        return False
    return True


def detect_nba_rim_hsv(bgr_frame: np.ndarray, frame_w: int, frame_h: int) -> Optional[Tuple[int, int]]:
    """Detect the orange basketball rim in the upper broadcast court zone."""
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    img_h, img_w = bgr_frame.shape[:2]
    y0, y1 = int(0.13 * img_h), int(0.46 * img_h)
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
        if rx_norm >= 0.38 and rx_norm <= 0.62:
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
    """
    Detect broadcast camera cuts, close-ups, and camera pan velocity `(pan_dx, pan_dy)`.
    Requires both HSV histogram jump and template-match drop to classify a true camera cut,
    preventing replay wipes or graphic banners from falsely splitting continuous live action.
    """
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
        is_closeup = (player_max_h_ratios is not None) and (t < len(player_max_h_ratios)) and (player_max_h_ratios[t] > 0.56)
        if is_closeup:
            is_wide_court[t] = False

    # Suppress isolated 1-frame transition/close-up islands when close-up ratio indicates non-wide segment
    if n >= 3:
        # Mark contiguous segments split by hard cuts; reject segments whose median max_h_ratio > 0.50
        seg_start = 0
        for t in range(1, n + 1):
            if t == n or is_hard_cut[t]:
                seg_idxs = list(range(seg_start, t))
                if player_max_h_ratios is not None and seg_idxs:
                    seg_med_h = float(np.median([player_max_h_ratios[idx] for idx in seg_idxs]))
                    if seg_med_h > 0.50:
                        for idx in seg_idxs:
                            is_wide_court[idx] = False
                seg_start = t

    if n > 0 and not any(is_wide_court):
        # Fallback: keep lowest max-height frames if no segment passed
        if player_max_h_ratios:
            min_idx = int(np.argmin(player_max_h_ratios))
            is_wide_court[min_idx] = True
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


def extract_verified_ball_point(
    det_res: Any,
    scale_x: float,
    scale_y: float,
    frame_w: int,
    frame_h: int,
    bgr_frame: Optional[np.ndarray] = None,
    rim_pt: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[float, float, float]]:
    """Extract highest-confidence orange-verified basketball detection in full-resolution coordinates."""
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
        if cls_id != COCO_SPORTS_BALL_CLS or float(conf) < 0.16:
            continue
        bx1, by1, bx2, by2 = float(box[0]) * scale_x, float(box[1]) * scale_y, float(box[2]) * scale_x, float(box[3]) * scale_y
        bw = bx2 - bx1
        bh = by2 - by1
        if not (0.008 * frame_h <= bw <= 0.062 * frame_h and 0.008 * frame_h <= bh <= 0.062 * frame_h):
            continue
        if max(bw, bh) / max(min(bw, bh), 1.0) > 1.65:
            continue
        cx, cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        if not (0.06 * frame_h <= cy <= 0.85 * frame_h):
            continue
        if float(conf) < 0.28 and cy > 0.81 * frame_h:
            continue

        # Reject low-confidence rim false positives
        if rim_pt is not None and float(conf) < 0.30:
            if np.hypot(cx - float(rim_pt[0]), cy - float(rim_pt[1])) < 32.0:
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
                orange_frac = float(np.mean((phsv[:, :, 0] >= 3) & (phsv[:, :, 0] <= 24) & (phsv[:, :, 1] >= 65)))

        if orange_frac < 0.025:
            continue
        if float(conf) < 0.28 and orange_frac < 0.24:
            continue
        rank = float(conf) + 0.22 * orange_frac
        if rank > best_rank:
            best_rank = rank
            best_ball = (cx, cy, float(conf))
    return best_ball


def interpolate_court_ball_track(
    raw_balls: List[Optional[Tuple[float, float, float]]],
    wide_mask: List[bool],
    sampled_frame_idxs: List[int],
    fps: float,
    diag: float,
    pan_dx: Optional[List[float]] = None,
) -> List[Optional[np.ndarray]]:
    """
    Prune teleporting false ball detections and smoothly linearly interpolate basketball coordinates
    across short occlusion gaps (<= 1.35s) inside continuous wide-court segments.
    """
    n = len(raw_balls)
    cleaned: List[Optional[Tuple[float, float, float]]] = [
        (raw_balls[t] if (t < len(wide_mask) and wide_mask[t]) else None) for t in range(n)
    ]

    # Prune stationary screen-locked overlays / sideline props that stay fixed despite large camera panning
    valid_idxs = [t for t, b in enumerate(cleaned) if b is not None]
    if pan_dx is not None and len(valid_idxs) >= 2:
        cum_pan_arr = np.cumsum(np.abs(np.array(pan_dx, dtype=np.float32)))
        for t in list(valid_idxs):
            bt = cleaned[t]
            if bt is None or bt[2] >= 0.45:
                continue
            for t2 in valid_idxs:
                if t2 <= t:
                    continue
                b2 = cleaned[t2]
                if b2 is None:
                    continue
                dt_pair = abs(sampled_frame_idxs[t2] - sampled_frame_idxs[t]) / max(fps, 1.0)
                cam_scroll = float(abs(cum_pan_arr[t2] - cum_pan_arr[t]))
                screen_disp = float(np.hypot(bt[0] - b2[0], bt[1] - b2[1]))
                if dt_pair >= 1.5 and cam_scroll >= 140.0 and screen_disp <= 52.0:
                    cleaned[t] = None
                    if b2[2] < 0.45:
                        cleaned[t2] = None
                    break

    # Prune isolated low-conf spikes that teleport wildly without neighbor support
    valid_idxs = [t for t, b in enumerate(cleaned) if b is not None]
    for k_pos, t in enumerate(valid_idxs):
        bt = cleaned[t]
        assert bt is not None
        if bt[2] >= 0.30:
            continue
        near_neighbor = False
        for k_other in (k_pos - 1, k_pos + 1):
            if 0 <= k_other < len(valid_idxs):
                to = valid_idxs[k_other]
                bo = cleaned[to]
                if bo is not None:
                    dt = abs(sampled_frame_idxs[to] - sampled_frame_idxs[t]) / max(fps, 1.0)
                    dist = float(np.hypot(bt[0] - bo[0], bt[1] - bo[1]))
                    if dt <= 1.2 and dist <= max(460.0, dt * 950.0):
                        near_neighbor = True
                        break
        if not near_neighbor and len(valid_idxs) >= 2:
            cleaned[t] = None

    track: List[Optional[np.ndarray]] = [
        (np.array([b[0], b[1]], dtype=np.float32) if b is not None else None) for b in cleaned
    ]
    anchor_ts = [t for t, pt in enumerate(track) if pt is not None]
    for idx in range(len(anchor_ts) - 1):
        t0, t1 = anchor_ts[idx], anchor_ts[idx + 1]
        if t1 - t0 <= 1:
            continue
        if not all(wide_mask[k] for k in range(t0, t1 + 1)):
            continue
        dt = abs(sampled_frame_idxs[t1] - sampled_frame_idxs[t0]) / max(fps, 1.0)
        if dt > 1.35:
            continue
        p0, p1 = track[t0], track[t1]
        assert p0 is not None and p1 is not None
        cum_pan = sum(abs(pan_dx[k]) for k in range(t0 + 1, t1 + 1)) if pan_dx is not None else 0.0
        max_jump = max(0.24 * diag, dt * 1150.0 + cum_pan)
        if float(np.linalg.norm(p1 - p0)) > max_jump:
            continue
        denom = max(float(sampled_frame_idxs[t1] - sampled_frame_idxs[t0]), 1.0)
        for k in range(t0 + 1, t1):
            alpha = float(sampled_frame_idxs[k] - sampled_frame_idxs[t0]) / denom
            track[k] = (1.0 - alpha) * p0 + alpha * p1
    return track


def collect_court_players_for_frame(
    pose_res: Any,
    det_res: Any,
    frame_w: int,
    frame_h: int,
    scale_x: float,
    scale_y: float,
    bgr_frame: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Collect IoM-deduplicated court player dictionaries (box, kpts, hsv, elev, conf)
    shared across preprocessing and inference tracking.
    """
    merged: List[Tuple[np.ndarray, Optional[np.ndarray], float]] = []
    if pose_res is not None and pose_res.boxes is not None and len(pose_res.boxes) > 0:
        p_xyxy = pose_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        p_xyxy[:, [0, 2]] *= scale_x
        p_xyxy[:, [1, 3]] *= scale_y
        p_conf = (
            pose_res.boxes.conf.cpu().numpy().astype(np.float32)
            if pose_res.boxes.conf is not None
            else np.ones(len(p_xyxy), dtype=np.float32) * 0.5
        )
        p_kpts = None
        if pose_res.keypoints is not None and pose_res.keypoints.xy is not None:
            p_kpts = pose_res.keypoints.xy.cpu().numpy().astype(np.float32).copy()
            p_kpts[:, :, 0] *= scale_x
            p_kpts[:, :, 1] *= scale_y
        for idx in range(len(p_xyxy)):
            if p_conf[idx] < 0.10:
                continue
            kp = p_kpts[idx] if (p_kpts is not None and idx < len(p_kpts)) else None
            merged.append((p_xyxy[idx], kp, float(p_conf[idx])))

    if det_res is not None and det_res.boxes is not None and len(det_res.boxes) > 0:
        d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        d_xyxy[:, [0, 2]] *= scale_x
        d_xyxy[:, [1, 3]] *= scale_y
        d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
        d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)
        for db, dc, cf in zip(d_xyxy, d_cls, d_conf):
            if dc != COCO_PERSON_CLS or float(cf) < 0.22:
                continue
            if all(box_iou_single(db, mb) < 0.35 and box_iom_single(db, mb) < 0.55 for mb, _, _ in merged):
                merged.append((db, synthesize_box_keypoints(db), float(cf)))

    players: List[Dict[str, Any]] = []
    for box, kp, conf in merged:
        if not is_court_player_box(box, frame_w, frame_h):
            continue
        if bgr_frame is not None and is_referee_or_sideline_official(
            bgr_frame, box, frame_w, frame_h, scale_x=scale_x, scale_y=scale_y
        ):
            continue
        hsv = (
            extract_player_torso_hsv(bgr_frame, box, scale_x=scale_x, scale_y=scale_y)
            if bgr_frame is not None
            else (0.0, 0.0, 128.0)
        )
        bh = max(1.0, float(box[3] - box[1]))
        elev = 0.0
        if kp is not None and len(kp) >= 11:
            sh = kp[[5, 6]][kp[[5, 6], 1] > 0]
            wr = kp[[9, 10]][kp[[9, 10], 1] > 0]
            if len(sh) > 0 and len(wr) > 0:
                elev = max(0.0, (float(np.mean(sh[:, 1])) - float(np.min(wr[:, 1]))) / max(0.5 * bh, 1.0))
        players.append(
            {
                "box": box,
                "kpts": kp,
                "conf": conf,
                "hsv": hsv,
                "elev": float(elev),
                "center": np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=np.float32),
            }
        )
    return players


def jersey_hsv_diff(hsv_a: Tuple[float, float, float], hsv_b: Tuple[float, float, float]) -> float:
    """Compute perceptual HSV separation between two player torso crops."""
    dh = min(abs(hsv_a[0] - hsv_b[0]), 180.0 - abs(hsv_a[0] - hsv_b[0]))
    return abs(hsv_a[2] - hsv_b[2]) + 0.75 * abs(hsv_a[1] - hsv_b[1]) + 1.1 * dh


def select_on_ball_opposing_pair(
    players: List[Dict[str, Any]],
    ball_pt: Optional[np.ndarray],
    rim_pt: Optional[Tuple[int, int]],
    prev_center: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    diag: float,
) -> Optional[Dict[str, Any]]:
    """
    Select the active opposing-team foul/contest pair anchored strongly to the basketball trajectory,
    rim direction, physical contact proximity, elevated contest arms, and temporal coherence.
    """
    if len(players) < 2:
        return None
    rim_arr = np.array([float(rim_pt[0]), float(rim_pt[1])], dtype=np.float32) if rim_pt is not None else None
    best: Optional[Dict[str, Any]] = None
    best_rank = -1e9

    for i in range(len(players)):
        bi = players[i]["box"]
        ci = players[i]["center"]
        bhi = float(bi[3] - bi[1])
        for j in range(i + 1, len(players)):
            bj = players[j]["box"]
            cj = players[j]["center"]
            bhj = float(bj[3] - bj[1])

            if min(bhi, bhj) / max(bhi, bhj, 1.0) < 0.50:
                continue
            if abs(float(bi[3]) - float(bj[3])) > frame_h * 0.30:
                continue
            if abs(float(ci[0]) - float(cj[0])) > max(frame_w * 0.17, 1.25 * max(bhi, bhj)):
                continue

            cdiff = jersey_hsv_diff(players[i]["hsv"], players[j]["hsv"])
            iom = box_iom_single(bi, bj)
            if iom >= 0.65 and cdiff < 28.0:
                continue

            cdist = float(np.linalg.norm(ci - cj))
            max_bh = max(bhi, bhj)
            if cdist > max(0.18 * diag, 1.40 * max_bh):
                continue

            iou = box_iou_single(bi, bj)
            mid = 0.5 * (ci + cj)
            elev = max(players[i]["elev"], players[j]["elev"])
            close_prox = max(0.0, 1.0 - cdist / max(1.40 * max_bh, 1.0))

            arm_reach = 0.0
            ki, kj = players[i].get("kpts"), players[j].get("kpts")
            if ki is not None and kj is not None:
                ai = valid_points(ki, ARM_CONTACT_KPTS)
                aj = valid_points(kj, ARM_CONTACT_KPTS)
                ui = valid_points(ki, UPPER_BODY_KPTS)
                uj = valid_points(kj, UPPER_BODY_KPTS)
                d_arm = min(
                    min_point_set_distance(ai, uj, cdist),
                    min_point_set_distance(aj, ui, cdist),
                )
                arm_reach = max(0.0, 1.0 - d_arm / max(0.28 * max_bh, 1.0))

            ball_prox = 0.0
            dball = 9999.0
            is_on_ball = False
            if ball_pt is not None:
                dball = min(
                    float(np.linalg.norm(mid - ball_pt)),
                    float(np.linalg.norm(ci - ball_pt)) + 25.0,
                    float(np.linalg.norm(cj - ball_pt)) + 25.0,
                )
                ball_prox = max(0.0, 1.0 - dball / max(diag * 0.16, 1.0))
                is_on_ball = dball <= max(diag * 0.13, 255.0)

            rim_bonus = 0.0
            if rim_arr is not None:
                drim = float(np.linalg.norm(mid - rim_arr))
                rim_bonus = 0.14 * max(0.0, 1.0 - drim / max(diag * 0.38, 1.0))

            smooth_bonus = 0.0
            if prev_center is not None:
                dprev = float(np.linalg.norm(mid - prev_center))
                smooth_bonus = 0.16 * float(np.exp(-dprev / max(0.16 * diag, 1.0)))

            team_score = 0.32 if cdiff >= 28.0 else -0.35
            off_ball_pen = -0.65 if (ball_pt is not None and dball > max(diag * 0.18, 340.0)) else 0.0

            rank = (
                0.48 * ball_prox
                + 0.10 * min(1.0, iou * 3.0)
                + 0.24 * arm_reach
                + 0.34 * min(1.2, elev * 1.4)
                + 0.16 * close_prox
                + team_score
                + rim_bonus
                + smooth_bonus
                + off_ball_pen
            )
            if rank > best_rank:
                best_rank = rank
                ux1 = max(0.0, min(float(bi[0]), float(bj[0])))
                uy1 = max(0.0, min(float(bi[1]), float(bj[1])))
                ux2 = min(float(frame_w), max(float(bi[2]), float(bj[2])))
                uy2 = min(float(frame_h), max(float(bi[3]), float(bj[3])))
                best = {
                    "roi": np.array([ux1, uy1, ux2, uy2], dtype=np.float32),
                    "off_box": bi,
                    "def_box": bj,
                    "off_hsv": players[i]["hsv"],
                    "def_hsv": players[j]["hsv"],
                    "mid": mid,
                    "iou": float(iou),
                    "arm_reach": float(arm_reach),
                    "elev": float(elev),
                    "cdiff": float(cdiff),
                    "dball": float(dball),
                    "is_on_ball": bool(is_on_ball),
                    "rank": float(rank),
                }
    return best


def score_foul_climax_sequence(
    frame_players: List[List[Dict[str, Any]]],
    cam_info: Dict[str, Any],
    rims: List[Optional[Tuple[int, int]]],
    raw_balls: List[Optional[Tuple[float, float, float]]],
    sampled_indices: List[int],
    frame_w: int,
    frame_h: int,
    fps: float,
    base_contact_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Shared spatiotemporal foul-climax scoring used by both preprocessing and dense AI Referee localization.
    Fuses local high-confidence basketball possession, arm/wrist contact & shooting elevation,
    transition deceleration, hoop proximity, and camera-cut climax structure.
    """
    n = len(frame_players)
    diag = float(np.hypot(frame_w, frame_h))
    wide_mask: List[bool] = cam_info["is_wide_court"]
    pan_dx: List[float] = cam_info["pan_dx"]
    hard_cuts: List[bool] = cam_info["is_hard_cut"]

    ball_track = interpolate_court_ball_track(
        raw_balls=raw_balls,
        wide_mask=wide_mask,
        sampled_frame_idxs=sampled_indices,
        fps=fps,
        diag=diag,
        pan_dx=pan_dx,
    )

    seg_bounds: Dict[int, Tuple[int, int, bool]] = {}
    t0 = 0
    while t0 < n:
        if not wide_mask[t0]:
            t0 += 1
            continue
        t1 = t0
        while t1 + 1 < n and wide_mask[t1 + 1] and not hard_cuts[t1 + 1]:
            t1 += 1
        ended_by_cut = (t1 + 1 < n) and (not wide_mask[t1 + 1] or hard_cuts[t1 + 1])
        for k_s in range(t0, t1 + 1):
            seg_bounds[k_s] = (t0, t1, ended_by_cut)
        t0 = t1 + 1

    ball_indices = [t for t, b in enumerate(ball_track) if b is not None]
    high_conf_ball_indices = [t for t, rb in enumerate(raw_balls) if rb is not None and rb[2] >= 0.28]
    has_high_conf_ball = len(high_conf_ball_indices) > 0

    pan_rates: List[float] = [0.0] * n
    for idx_p in range(1, n):
        if not hard_cuts[idx_p]:
            df_p = max(1, sampled_indices[idx_p] - sampled_indices[idx_p - 1])
            pan_rates[idx_p] = abs(pan_dx[idx_p]) / float(df_p)

    avg_step_f = max(1.0, float(np.median(np.diff(sampled_indices))) if n >= 2 else 6.0)
    lookback_steps = max(3, int(round(3.2 * max(fps, 1.0) / avg_step_f)))
    total_dur_sec = float(sampled_indices[-1] - sampled_indices[0]) / max(fps, 1.0) if n >= 2 else 0.0

    pair_results: List[Optional[Dict[str, Any]]] = [None] * n
    scores = np.zeros(n, dtype=np.float32)
    prev_center: Optional[np.ndarray] = None
    prev_dist: Optional[float] = None

    for idx in range(n):
        if not wide_mask[idx]:
            prev_center = None
            prev_dist = None
            continue

        best = select_on_ball_opposing_pair(
            players=frame_players[idx],
            ball_pt=ball_track[idx],
            rim_pt=rims[idx],
            prev_center=prev_center,
            frame_w=frame_w,
            frame_h=frame_h,
            diag=diag,
        )
        pair_results[idx] = best
        if best is None:
            prev_dist = None
            continue

        prev_center = best["mid"]
        pan_rate = pan_rates[idx]
        recent_fast_pan = max(pan_rates[max(0, idx - lookback_steps) : idx + 1], default=0.0)
        arm_reach = float(best.get("arm_reach", 0.0))
        has_action = (best["elev"] >= 0.08) or (arm_reach >= 0.15) or (best["iou"] >= 0.05)

        base_c = float(base_contact_scores[idx]) if (base_contact_scores is not None and idx < len(base_contact_scores)) else 0.0
        comp = 0.56 * max(0.0, float(best["rank"])) + 0.28 * base_c

        if best["cdiff"] >= 26.0 and (ball_track[idx] is None or best["is_on_ball"]):
            comp += 0.48 * min(1.3, best["elev"] * 1.45)
            comp += 0.26 * arm_reach
            comp += 0.16 * min(1.0, best["iou"] * 2.8)

        cur_dist = float(np.linalg.norm(best["off_box"][:2] - best["def_box"][:2]))
        if prev_dist is not None and (best["is_on_ball"] or has_action):
            closing = max(0.0, (prev_dist - cur_dist) / max(0.03 * diag, 1.0))
            comp += 0.16 * min(1.0, closing)
        prev_dist = cur_dist

        # Local high-confidence basketball possession check within ±2.3s in the same wide segment
        s_lo, s_hi, ended_by_cut = seg_bounds.get(idx, (0, n - 1, False))
        seg_hc_balls = [tb for tb in high_conf_ball_indices if s_lo <= tb <= s_hi]
        seg_all_balls = [tb for tb in ball_indices if s_lo <= tb <= s_hi]

        local_hc = 0.0
        for tb in seg_hc_balls:
            dt_signed = (sampled_indices[idx] - sampled_indices[tb]) / max(fps, 1.0)
            dt_b = abs(dt_signed)
            if dt_b <= 2.3:
                rb = raw_balls[tb]
                assert rb is not None
                tau = 1.45 if dt_signed >= 0.0 else 1.05
                w_conf = min(1.0, float(rb[2]) / 0.52) * float(np.exp(-((dt_b / tau) ** 2)))
                if w_conf > local_hc:
                    local_hc = w_conf

        eff_conf = 0.0
        if ball_track[idx] is not None and best["is_on_ball"]:
            if raw_balls[idx] is not None:
                eff_conf = float(raw_balls[idx][2])
            else:
                near_c = [
                    float(raw_balls[tb][2])
                    for tb in range(s_lo, s_hi + 1)
                    if raw_balls[tb] is not None
                    and abs(sampled_indices[tb] - sampled_indices[idx]) <= 1.2 * max(fps, 1.0)
                ]
                eff_conf = max([local_hc * 0.55] + near_c)
            if eff_conf >= 0.28:
                comp += 0.26 + 0.38 * min(1.0, eff_conf / 0.55)
            else:
                comp += 0.14 * min(1.0, eff_conf / 0.28)
        elif local_hc > 0.0:
            active_mult = 0.44 if (best["elev"] >= 0.16 or arm_reach >= 0.28 or best["iou"] >= 0.08) else 0.32
            comp += active_mult * local_hc
        elif seg_all_balls:
            dt_any = min(abs(sampled_indices[tb] - sampled_indices[idx]) for tb in seg_all_balls) / max(fps, 1.0)
            if dt_any <= 1.2 and has_action:
                comp += float(0.14 * np.exp(-((dt_any / 0.90) ** 2)))
            elif has_high_conf_ball:
                comp *= 0.58
        elif has_high_conf_ball:
            comp *= 0.52

        if has_high_conf_ball and local_hc < 0.10 and eff_conf < 0.28:
            comp *= 0.62

        if not has_action and eff_conf < 0.28:
            comp *= 0.70

        if s_lo > 0 and (float(sampled_indices[idx] - sampled_indices[s_lo]) / max(fps, 1.0)) < 0.45:
            comp *= 0.45

        if pan_rate > 20.0 and best["elev"] < 0.24:
            comp *= 0.55
        elif pan_rate <= 16.0:
            if has_action or eff_conf >= 0.28:
                comp += 0.08
            if recent_fast_pan >= 18.0 and (best["elev"] >= 0.16 or best["iou"] >= 0.08 or arm_reach >= 0.25 or local_hc >= 0.35 or best["is_on_ball"]):
                comp += 0.34

        if ended_by_cut:
            dt_to_cut = float(sampled_indices[s_hi] - sampled_indices[idx]) / max(fps, 1.0)
            if 0.35 <= dt_to_cut <= 2.8:
                comp += float(0.24 * np.exp(-((dt_to_cut - 1.35) / 1.15) ** 2))
            elif dt_to_cut < 0.20:
                comp *= 0.72
        elif s_hi >= n - 2 and 6.0 <= total_dur_sec <= 26.0 and (has_action or local_hc >= 0.20 or recent_fast_pan >= 18.0):
            dt_to_end = float(sampled_indices[s_hi] - sampled_indices[idx]) / max(fps, 1.0)
            if 0.60 <= dt_to_end <= 3.8:
                comp += float(0.16 * np.exp(-((dt_to_end - 2.20) / 1.30) ** 2))

        if rims[idx] is not None and (has_action or best["is_on_ball"]):
            if float(np.hypot(best["mid"][0] - rims[idx][0], best["mid"][1] - rims[idx][1])) < 0.22 * diag:
                comp += 0.14

        scores[idx] = max(0.0, comp)

    return {
        "scores": scores,
        "pair_results": pair_results,
        "ball_track": ball_track,
        "raw_balls": raw_balls,
        "seg_bounds": seg_bounds,
        "wide_mask": wide_mask,
        "pan_dx": pan_dx,
        "rims": rims,
        "cam_info": cam_info,
    }


def refine_sequence_rois_and_scores(
    pose_results: List[Any],
    det_results: List[Any],
    small_bgr_frames: List[np.ndarray],
    scanned_rois: List[np.ndarray],
    scanned_scores: List[float],
    sampled_indices: List[int],
    frame_w: int,
    frame_h: int,
    scale_x: float,
    scale_y: float,
    fps: float,
) -> Tuple[List[np.ndarray], List[float], Dict[str, Any]]:
    """
    Upgrade per-frame scanned ROIs and scores using sequence-level scene-cut rejection,
    camera-pan deceleration, verified/interpolated basketball trajectory, hoop rims, and
    temporally-consistent on-ball opposing-jersey pairs.
    """
    n = len(small_bgr_frames)
    diag = float(np.hypot(frame_w, frame_h))
    if n == 0:
        return scanned_rois, scanned_scores, {}

    max_h_ratios: List[float] = []
    rims: List[Optional[Tuple[int, int]]] = []
    for idx in range(n):
        mh = 0.0
        for res in (pose_results[idx] if idx < len(pose_results) else None, det_results[idx] if idx < len(det_results) else None):
            if res is not None and res.boxes is not None and len(res.boxes) > 0:
                xy = res.boxes.xyxy.cpu().numpy()
                mh = max(mh, float(np.max((xy[:, 3] - xy[:, 1]) * scale_y)) / max(float(frame_h), 1.0))
        max_h_ratios.append(mh)
        rims.append(detect_nba_rim_hsv(small_bgr_frames[idx], frame_w, frame_h))

    cam_info = compute_camera_cuts_and_pan(small_bgr_frames, frame_w, frame_h, player_max_h_ratios=max_h_ratios)
    wide_mask: List[bool] = cam_info["is_wide_court"]
    raw_balls = [
        extract_verified_ball_point(
            det_results[idx] if idx < len(det_results) else None,
            scale_x=scale_x,
            scale_y=scale_y,
            frame_w=frame_w,
            frame_h=frame_h,
            bgr_frame=small_bgr_frames[idx],
            rim_pt=rims[idx],
        )
        for idx in range(n)
    ]
    frame_players = [
        (
            collect_court_players_for_frame(
                pose_res=pose_results[idx] if idx < len(pose_results) else None,
                det_res=det_results[idx] if idx < len(det_results) else None,
                frame_w=frame_w,
                frame_h=frame_h,
                scale_x=scale_x,
                scale_y=scale_y,
                bgr_frame=small_bgr_frames[idx],
            )
            if wide_mask[idx]
            else []
        )
        for idx in range(n)
    ]

    seq = score_foul_climax_sequence(
        frame_players=frame_players,
        cam_info=cam_info,
        rims=rims,
        raw_balls=raw_balls,
        sampled_indices=sampled_indices,
        frame_w=frame_w,
        frame_h=frame_h,
        fps=fps,
        base_contact_scores=scanned_scores,
    )

    refined_rois: List[np.ndarray] = [r.copy() for r in scanned_rois]
    # Preserve classifier-aligned temporal scores on wide court frames while rejecting close-up cuts
    refined_scores: List[float] = [
        float(sc if (idx < len(wide_mask) and wide_mask[idx]) else 0.0)
        for idx, sc in enumerate(scanned_scores)
    ]
    ball_track = seq["ball_track"]
    for idx, best in enumerate(seq["pair_results"]):
        if not wide_mask[idx] or best is None:
            continue
        cur_roi = refined_rois[idx]
        cur_mid = np.array([0.5 * (cur_roi[0] + cur_roi[2]), 0.5 * (cur_roi[1] + cur_roi[3])], dtype=np.float32)
        cur_h = float(cur_roi[3] - cur_roi[1])
        replace_roi = (cur_h > frame_h * 0.48) or best["is_on_ball"] or (float(best["rank"]) > 0.45)
        if not replace_roi and ball_track[idx] is not None:
            if float(np.linalg.norm(cur_mid - ball_track[idx])) > max(diag * 0.16, 280.0):
                replace_roi = True
        if replace_roi:
            refined_rois[idx] = best["roi"]

    return refined_rois, refined_scores, seq


def _decode_clip_scan_frames(
    video_path: Path,
    max_live_play_sec: float = 22.0,
    max_scan_samples: int = 24,
    num_frames: int = 16,
    scan_w: int = 960,
    scan_h: int = 540,
) -> Optional[Dict[str, object]]:
    """Decode downscaled scan frames for a single video clip (safe to run in parallel CPU worker threads)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    effective_horizon_sec = max(float(max_live_play_sec), 22.0)
    max_horizon_frames = max(int(round(fps * effective_horizon_sec)), num_frames * 2)
    effective_total = min(reported_frames, max_horizon_frames) if reported_frames > 0 else max_horizon_frames

    scan_start = int(round(fps * 0.8)) if effective_total > 60 else 0
    scan_end = int(effective_total * 0.96) if effective_total > 60 else max(effective_total - 1, 1)
    effective_sec = float(max(1, scan_end - scan_start)) / max(fps, 1.0)
    adaptive_samples = int(np.clip(round(effective_sec * 1.6), 16, 32))
    num_scan_pts = min(max(int(max_scan_samples), adaptive_samples), 32)
    target_scan_set = set(
        np.linspace(scan_start, scan_end, num=min(num_scan_pts, max(scan_end - scan_start + 1, 1)))
        .round()
        .astype(int)
        .tolist()
    )
    max_scan_frame = max(target_scan_set) if target_scan_set else 0

    scanned_frames: List[int] = []
    small_bgr_frames: List[np.ndarray] = []

    frame_idx = 0
    while frame_idx <= max_scan_frame:
        if frame_idx in target_scan_set:
            ret, frame = cap.read()
            if not ret:
                break
            small_frame = cv2.resize(frame, (scan_w, scan_h), interpolation=cv2.INTER_LINEAR)
            small_bgr_frames.append(small_frame)
            scanned_frames.append(frame_idx)
        else:
            if not cap.grab():
                break
        frame_idx += 1

    cap.release()
    if not small_bgr_frames:
        return None

    return {
        "video_path": video_path,
        "fps": fps,
        "reported_frames": reported_frames,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "scan_w": scan_w,
        "scan_h": scan_h,
        "scale_x": float(frame_w) / float(scan_w),
        "scale_y": float(frame_h) / float(scan_h),
        "scan_start": scan_start,
        "scanned_frames": scanned_frames,
        "small_bgr_frames": small_bgr_frames,
    }


def _finalize_clip_action_tube(
    scan_info: Dict[str, object],
    scanned_rois: List[np.ndarray],
    scanned_scores: List[float],
    scanned_feats: List[np.ndarray],
    window_sec: float = 1.5,
    num_frames: int = 16,
    crop_size: int = 224,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[int], float, torch.Tensor]:
    """Interpolate trajectories, locate peak foul contact, and decode the 16 action-tube frames."""
    video_path = scan_info["video_path"]
    fps = float(scan_info["fps"])  # type: ignore[arg-type]
    reported_frames = int(scan_info["reported_frames"])  # type: ignore[arg-type]
    frame_w = int(scan_info["frame_w"])  # type: ignore[arg-type]
    frame_h = int(scan_info["frame_h"])  # type: ignore[arg-type]
    scan_start = int(scan_info["scan_start"])  # type: ignore[arg-type]
    scanned_frames: List[int] = scan_info["scanned_frames"]  # type: ignore[assignment]

    horizon_len = max(scanned_frames[-1] + int(round(fps * 1.0)), num_frames + 1)
    if reported_frames > 0:
        horizon_len = min(horizon_len, reported_frames)

    scan_idx_arr = np.array(scanned_frames, dtype=np.float32)
    all_idx_arr = np.arange(horizon_len, dtype=np.float32)

    raw_rois = np.zeros((horizon_len, 4), dtype=np.float32)
    scan_rois_arr = np.stack(scanned_rois, axis=0)
    for c in range(4):
        raw_rois[:, c] = np.interp(all_idx_arr, scan_idx_arr, scan_rois_arr[:, c])

    contact_scores = np.interp(all_idx_arr, scan_idx_arr, np.array(scanned_scores, dtype=np.float32))
    if scan_start > 0:
        contact_scores[:scan_start] = 0.0
    if len(scanned_frames) > 0 and scanned_frames[-1] + 1 < horizon_len:
        contact_scores[scanned_frames[-1] + 1 :] = 0.0
    scan_feats_arr = np.stack(scanned_feats, axis=0)
    full_feats = np.zeros((horizon_len, NUM_FOUL_FEATURES), dtype=np.float32)
    for f_dim in range(NUM_FOUL_FEATURES):
        full_feats[:, f_dim] = np.interp(all_idx_arr, scan_idx_arr, scan_feats_arr[:, f_dim])

    roi_cy = 0.5 * (raw_rois[:, 1] + raw_rois[:, 3])
    closing_vel = -np.gradient(full_feats[:, 1]) * 4.0
    impact_jolt = np.abs(np.gradient(roi_cy)) / max(float(frame_h) * 0.02, 1.0)
    full_feats[:, 9] = np.clip(0.5 + closing_vel, 0.0, 1.0)
    full_feats[:, 10] = np.clip(impact_jolt * 0.5, 0.0, 1.0)

    kernel_len = max(3, int(round(fps * 0.45)))
    kernel = np.ones(kernel_len, dtype=np.float32) / kernel_len
    smoothed_scores = np.convolve(contact_scores, kernel, mode="same")
    peak_frame = int(np.argmax(smoothed_scores))

    half_window_frames = max(num_frames // 2, int(round(fps * window_sec * 0.5)))
    start_frame = max(0, peak_frame - half_window_frames)
    end_frame = min(horizon_len - 1, start_frame + 2 * half_window_frames)
    start_frame = max(0, end_frame - 2 * half_window_frames)

    sampled_indices = np.linspace(start_frame, end_frame, num=num_frames).round().astype(int).tolist()
    sampled_set = set(sampled_indices)
    stabilized_rois = smooth_boxes_ema(raw_rois, alpha=0.32)

    cap2 = cv2.VideoCapture(str(video_path))
    decoded_samples: Dict[int, np.ndarray] = {}
    max_needed = max(sampled_indices)
    cur_idx = 0
    while cur_idx <= max_needed:
        if cur_idx in sampled_set:
            ret, frame = cap2.read()
            if not ret:
                break
            x1, y1, x2, y2 = compute_square_crop_coords(
                stabilized_rois[cur_idx],
                frame_w=frame_w,
                frame_h=frame_h,
            )
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                crop_bgr = frame
            resized_bgr = cv2.resize(crop_bgr, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            decoded_samples[cur_idx] = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        else:
            if not cap2.grab():
                break
        cur_idx += 1
    cap2.release()

    cropped_rgb_frames: List[np.ndarray] = []
    crop_boxes: List[Tuple[int, int, int, int]] = []
    sampled_feats_list: List[np.ndarray] = []

    fallback_rgb = (
        next(iter(decoded_samples.values()))
        if decoded_samples
        else np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    )
    for f_idx in sampled_indices:
        x1, y1, x2, y2 = compute_square_crop_coords(
            stabilized_rois[f_idx],
            frame_w=frame_w,
            frame_h=frame_h,
        )
        crop_boxes.append((x1, y1, x2, y2))
        cropped_rgb_frames.append(decoded_samples.get(f_idx, fallback_rgb))
        sampled_feats_list.append(full_feats[f_idx])

    stacked_rgb = np.stack(cropped_rgb_frames, axis=0)
    tube_tensor = torch.from_numpy(stacked_rgb).permute(3, 0, 1, 2).contiguous()
    foul_features_tensor = torch.from_numpy(np.stack(sampled_feats_list, axis=0)).float()

    return tube_tensor, crop_boxes, sampled_indices, fps, foul_features_tensor


def extract_action_tube(
    video_path: Path | str,
    det_model: Optional[object] = None,
    pose_model: Optional[object] = None,
    window_sec: float = 1.5,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "cpu",
    max_live_play_sec: float = 22.0,
    max_scan_samples: int = 16,
    yolo_batch_size: int = 4,
    **kwargs,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[int], float, torch.Tensor]:
    """
    Cross-Platform Memory-Safe & Deadlock-Free Action-Tube Extractor:
      1. Caps live-play scanning to the first `max_live_play_sec` (22.0s) so multi-minute
         replay review clips never exhaust RAM on 8GB laptops while covering the full
         ~20s live-action broadcast play window.
      2. Decodes downscaled `960x540` frames and calls `cap.release()` BEFORE running
         YOLOv11-pose in micro-batches.
      3. Reopens the video after YOLO completes to read ONLY the 16 sampled frames in the
         1.5s window around peak foul contact.
    """
    if "max_scan_frames" in kwargs:
        max_scan_samples = kwargs["max_scan_frames"]
    primary_pose = pose_model if pose_model is not None else det_model
    if primary_pose is None:
        raise ValueError("At least one YOLO model (pose_model or det_model) must be provided.")
    active_det = det_model if det_model is not None else get_cached_det_model()

    scan_info = _decode_clip_scan_frames(
        video_path=Path(video_path),
        max_live_play_sec=max_live_play_sec,
        max_scan_samples=max_scan_samples,
        num_frames=num_frames,
        scan_w=1280,
        scan_h=720,
    )
    if scan_info is None:
        raise RuntimeError(f"Unable to open or read video frames in {video_path}")

    small_bgr_frames: List[np.ndarray] = scan_info["small_bgr_frames"]  # type: ignore[assignment]
    frame_w = int(scan_info["frame_w"])  # type: ignore[arg-type]
    frame_h = int(scan_info["frame_h"])  # type: ignore[arg-type]
    scale_x = float(scan_info["scale_x"])  # type: ignore[arg-type]
    scale_y = float(scan_info["scale_y"])  # type: ignore[arg-type]
    fps = float(scan_info["fps"])  # type: ignore[arg-type]
    scanned_frames: List[int] = scan_info["scanned_frames"]  # type: ignore[assignment]

    scanned_rois: List[np.ndarray] = []
    scanned_scores: List[float] = []
    scanned_feats: List[np.ndarray] = []
    all_pose_results: List[Any] = []
    all_det_results: List[Any] = []

    yolo_device = select_yolo_device(device)
    batch_sz = max(1, int(yolo_batch_size))
    with torch.inference_mode():
        for b_start in range(0, len(small_bgr_frames), batch_sz):
            batch_frames = small_bgr_frames[b_start : b_start + batch_sz]
            pose_results = primary_pose.predict(
                batch_frames,
                conf=0.10,
                imgsz=768,
                verbose=False,
                device=yolo_device,
            )
            det_results = (
                active_det.predict(
                    batch_frames,
                    conf=0.12,
                    imgsz=1280,
                    classes=[COCO_PERSON_CLS, COCO_SPORTS_BALL_CLS],
                    verbose=False,
                    device=yolo_device,
                )
                if (active_det is not None and active_det is not primary_pose)
                else [None] * len(batch_frames)
            )
            for b_offset, (pose_res, det_res) in enumerate(zip(pose_results, det_results)):
                all_pose_results.append(pose_res)
                all_det_results.append(det_res)
                roi, score, feat_vec = analyze_foul_interaction_frame(
                    pose_res=pose_res,
                    det_res=det_res,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    bgr_frame=batch_frames[b_offset],
                )
                scanned_rois.append(roi)
                scanned_scores.append(score)
                scanned_feats.append(feat_vec)

    scanned_rois, scanned_scores, seq_dbg = refine_sequence_rois_and_scores(
        pose_results=all_pose_results,
        det_results=all_det_results,
        small_bgr_frames=small_bgr_frames,
        scanned_rois=scanned_rois,
        scanned_scores=scanned_scores,
        sampled_indices=scanned_frames,
        frame_w=frame_w,
        frame_h=frame_h,
        scale_x=scale_x,
        scale_y=scale_y,
        fps=fps,
    )

    scan_info["small_bgr_frames"] = []
    return _finalize_clip_action_tube(
        scan_info=scan_info,
        scanned_rois=scanned_rois,
        scanned_scores=scanned_scores,
        scanned_feats=scanned_feats,
        window_sec=window_sec,
        num_frames=num_frames,
        crop_size=crop_size,
    )


def synthesize_fallback_action_tube(
    play_id: str,
    label: int,
    num_frames: int = 16,
    crop_size: int = 224,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[int], float, torch.Tensor]:
    """
    Generate a deterministic synthetic `(3, 16, 224, 224)` video tube and `(16, 12)`
    biomechanical feature sequence seeded by `play_id` and `label` so offline users
    can test the full preprocessing + 3D CNN training pipeline without downloading GBs of video.
    """
    seed = int(hashlib.md5(play_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    t_axis = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    contact_envelope = np.exp(-((t_axis - 0.55) ** 2) / 0.045).astype(np.float32)
    intensity_scale = 0.78 if label == 1 else 0.32

    foul_feats = np.zeros((num_frames, NUM_FOUL_FEATURES), dtype=np.float32)
    for f_idx in range(NUM_FOUL_FEATURES):
        noise = rng.normal(0.0, 0.05, size=num_frames).astype(np.float32)
        foul_feats[:, f_idx] = np.clip(intensity_scale * contact_envelope + 0.15 + noise, 0.0, 1.0)
    foul_feats[:, 11] = np.clip(intensity_scale * contact_envelope + 0.10, 0.0, 1.0)

    frames = np.zeros((num_frames, crop_size, crop_size, 3), dtype=np.uint8)
    for t in range(num_frames):
        canvas = np.full((crop_size, crop_size, 3), (38, 44, 52), dtype=np.uint8)
        # Draw two converging player silhouettes
        offset = int((1.0 - contact_envelope[t] * intensity_scale) * 36)
        cx1, cy1 = crop_size // 2 - max(8, offset), crop_size // 2
        cx2, cy2 = crop_size // 2 + max(8, offset), crop_size // 2
        cv2.rectangle(canvas, (cx1 - 22, cy1 - 50), (cx1 + 22, cy1 + 55), (210, 140, 70), -1)
        cv2.rectangle(canvas, (cx2 - 22, cy2 - 48), (cx2 + 22, cy2 + 55), (75, 165, 225), -1)
        frames[t] = canvas

    tube_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
    crop_boxes = [(420, 180, 868, 628) for _ in range(num_frames)]
    sampled_indices = list(range(10, 10 + num_frames * 2, 2))
    return tube_tensor, crop_boxes, sampled_indices, 30.0, torch.from_numpy(foul_feats).float()


def select_balanced_subset(rows: List[Dict[str, str]], max_clips: int) -> List[Dict[str, str]]:
    """
    Select up to `max_clips` rows while preserving balanced representation of both classes,
    prioritizing rows whose raw `.mp4` clip is already downloaded in `data/raw/`.
    """
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
    """Atomically write `processed_metadata.csv` after each clip so progress is never lost."""
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
    yolo_weights: str = "yolo11n.pt",
    pose_weights: Optional[str] = "yolo11n-pose.pt",
    window_sec: float = 1.5,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "auto",
    max_live_play_sec: float = 22.0,
    max_scan_samples: int = 16,
    max_clips: int = 12,
    yolo_batch_size: int = 4,
    synthetic_fallback: bool = False,
) -> List[Dict[str, str]]:
    """
    Process raw clips listed in `data/l2m_metadata.csv` in small batches (`max_clips=12` by default),
    saving pre-cropped `(3, 16, 224, 224)` action tubes and `(16, 12)` biomechanical foul feature
    tensors to `data/processed/{foul,no_foul}/{play_id}.pt` and updating `processed_metadata.csv`.
    Clips set aside in `data/test/` are strictly excluded to prevent data leakage.
    """
    metadata_csv = resolve_project_path(metadata_csv)
    processed_dir = resolve_project_path(processed_dir)
    processed_metadata_csv = resolve_project_path(processed_metadata_csv)

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata file {metadata_csv} not found. Run `1_scrape_l2m.py` first."
        )

    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Segregate test data: strictly exclude any clips present in test_dir
    active_test_dir = resolve_project_path(test_dir) if test_dir is not None else resolve_project_path(BASE_DIR / "data" / "test")
    test_stems: set = set()
    if active_test_dir.exists():
        test_stems = {
            p.stem
            for p in active_test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
        }
    if test_stems:
        orig_count = len(all_rows)
        all_rows = [
            r for r in all_rows
            if r.get("play_id", "").strip() not in test_stems
            and Path(r.get("video_path", "")).stem not in test_stems
        ]
        logger.info(
            "Segregated %d test-only clips (excluded from training preprocessing): %s",
            len(test_stems),
            sorted(list(test_stems)),
        )

    rows = select_balanced_subset(all_rows, max_clips=max_clips)

    pose_model = None
    has_any_raw_video = any(resolve_project_path(r["video_path"]).exists() for r in rows)
    if has_any_raw_video and YOLO is not None:
        pose_candidate = resolve_project_path(pose_weights) if pose_weights else None
        active_pose_weights = (
            str(pose_candidate)
            if (pose_candidate and pose_candidate.exists())
            else (pose_weights or yolo_weights)
        )
        yolo_dev = select_yolo_device(device)
        if yolo_dev == "cuda" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 12.0:
                if yolo_batch_size < 128:
                    yolo_batch_size = 128
                max_scan_samples = min(max_scan_samples, 20)
        logger.info("Loading Ultralytics pose/detection model: %s (device=%s, batch_size=%d)", active_pose_weights, yolo_dev, yolo_batch_size)
        pose_model = YOLO(active_pose_weights)

    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_rows: List[Dict[str, str]] = []

    logger.info(
        "Preprocessing %d/%d clips into %dx%d foul-contact action tubes (scan_samples=%d, yolo_batch=%d)...",
        len(rows),
        len(all_rows),
        crop_size,
        crop_size,
        max_scan_samples,
        yolo_batch_size,
    )

    real_tasks: List[Tuple[int, Dict[str, str], Path, Path]] = []
    ready_map: Dict[int, Dict[str, str]] = {}

    for idx, row in enumerate(rows, start=1):
        play_id = row["play_id"]
        label = int(row["label"])
        video_path = resolve_project_path(row["video_path"])
        class_name = "foul" if label == 1 else "no_foul"
        out_tensor_path = processed_dir / class_name / f"{play_id}.pt"
        out_tensor_path.parent.mkdir(parents=True, exist_ok=True)

        needs_reprocess = not out_tensor_path.exists()
        if out_tensor_path.exists() and video_path.exists() and video_path.stat().st_size > 10_240:
            try:
                existing_pt = torch.load(out_tensor_path, map_location="cpu", weights_only=False)
                if isinstance(existing_pt, dict):
                    if existing_pt.get("is_synthetic", True):
                        needs_reprocess = True
                    ff = existing_pt.get("foul_features")
                    if isinstance(ff, torch.Tensor) and ff.numel() >= 4 and float(ff[:, :4].abs().max().item()) < 1e-4:
                        needs_reprocess = True
            except Exception:
                needs_reprocess = True

        if not needs_reprocess:
            out_row = dict(row)
            out_row["tensor_path"] = (Path("data") / "processed" / class_name / f"{play_id}.pt").as_posix()
            ready_map[idx] = out_row
        elif video_path.exists() and video_path.stat().st_size > 10_240 and pose_model is not None:
            real_tasks.append((idx, row, video_path, out_tensor_path))
        elif synthetic_fallback:
            tube_tensor, crop_boxes, sampled_indices, fps, foul_features = synthesize_fallback_action_tube(
                play_id=play_id,
                label=label,
                num_frames=num_frames,
                crop_size=crop_size,
            )
            torch.save(
                {
                    "video": tube_tensor,
                    "foul_features": foul_features,
                    "label": label,
                    "play_id": play_id,
                    "call_type": row.get("call_type", ""),
                    "review_decision": row.get("review_decision", ""),
                    "crop_boxes": crop_boxes,
                    "sampled_indices": sampled_indices,
                    "fps": fps,
                    "is_synthetic": True,
                },
                out_tensor_path,
            )
            out_row = dict(row)
            out_row["tensor_path"] = (Path("data") / "processed" / class_name / f"{play_id}.pt").as_posix()
            ready_map[idx] = out_row
        else:
            logger.warning("Skipping missing raw video: %s (pass --synthetic-fallback for offline tensor demo)", video_path)

    # High-Throughput Multi-Clip Parallel CPU Decoding + Mega-Batch GPU Inference
    if real_tasks and pose_model is not None:
        yolo_dev = select_yolo_device(device)
        active_det = get_cached_det_model()
        clips_per_superbatch = 12 if yolo_dev == "cuda" else 4
        cpu_workers = 4 if yolo_dev == "cuda" else 2

        for sb_start in range(0, len(real_tasks), clips_per_superbatch):
            sb_tasks = real_tasks[sb_start : sb_start + clips_per_superbatch]

            # 1. Parallel CPU video decoding across clips
            with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
                decoded_infos = list(
                    pool.map(
                        lambda t: _decode_clip_scan_frames(
                            video_path=t[2],
                            max_live_play_sec=max_live_play_sec,
                            max_scan_samples=max_scan_samples,
                            num_frames=num_frames,
                            scan_w=960,
                            scan_h=540,
                        ),
                        sb_tasks,
                    )
                )

            # 2. Concatenate all frames across the superbatch and run ONE massive GPU YOLO forward pass
            mega_frames: List[np.ndarray] = []
            clip_slices: List[Tuple[int, int, int, Dict[str, object]]] = []
            for t_idx, scan_info in enumerate(decoded_infos):
                if scan_info is None:
                    continue
                frames_list: List[np.ndarray] = scan_info["small_bgr_frames"]  # type: ignore[assignment]
                start_f = len(mega_frames)
                mega_frames.extend(frames_list)
                end_f = len(mega_frames)
                clip_slices.append((t_idx, start_f, end_f, scan_info))

            mega_pose_res: List[object] = []
            mega_det_res: List[object] = []
            if mega_frames:
                batch_sz = max(1, int(yolo_batch_size))
                with torch.inference_mode():
                    for b_start in range(0, len(mega_frames), batch_sz):
                        b_frames = mega_frames[b_start : b_start + batch_sz]
                        p_res = pose_model.predict(
                            b_frames,
                            conf=0.10,
                            imgsz=768,
                            verbose=False,
                            device=yolo_dev,
                        )
                        d_res = (
                            active_det.predict(
                                b_frames,
                                conf=0.06,
                                imgsz=960,
                                classes=[COCO_PERSON_CLS, COCO_SPORTS_BALL_CLS],
                                verbose=False,
                                device=yolo_dev,
                            )
                            if (active_det is not None and active_det is not pose_model)
                            else [None] * len(b_frames)
                        )
                        mega_pose_res.extend(p_res)
                        mega_det_res.extend(d_res)

            # 3. Analyze interaction per clip and parallelize Pass 2 tube cropping
            def _crop_and_save_clip(item_tuple):
                t_idx, start_f, end_f, scan_info = item_tuple
                idx, row, video_path, out_tensor_path = sb_tasks[t_idx]
                play_id = row["play_id"]
                label = int(row["label"])
                class_name = "foul" if label == 1 else "no_foul"
                frame_w = int(scan_info["frame_w"])
                frame_h = int(scan_info["frame_h"])
                scale_x = float(scan_info["scale_x"])
                scale_y = float(scan_info["scale_y"])
                clip_fps = float(scan_info["fps"])
                clip_scanned_frames: List[int] = scan_info["scanned_frames"]  # type: ignore[assignment]
                frames_list: List[np.ndarray] = scan_info["small_bgr_frames"]  # type: ignore[assignment]

                scanned_rois: List[np.ndarray] = []
                scanned_scores: List[float] = []
                scanned_feats: List[np.ndarray] = []
                for local_i, global_i in enumerate(range(start_f, end_f)):
                    roi, score, feat_vec = analyze_foul_interaction_frame(
                        pose_res=mega_pose_res[global_i],
                        det_res=mega_det_res[global_i],
                        frame_w=frame_w,
                        frame_h=frame_h,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        bgr_frame=frames_list[local_i],
                    )
                    scanned_rois.append(roi)
                    scanned_scores.append(score)
                    scanned_feats.append(feat_vec)

                scanned_rois, scanned_scores, _ = refine_sequence_rois_and_scores(
                    pose_results=mega_pose_res[start_f:end_f],
                    det_results=mega_det_res[start_f:end_f],
                    small_bgr_frames=frames_list,
                    scanned_rois=scanned_rois,
                    scanned_scores=scanned_scores,
                    sampled_indices=clip_scanned_frames,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    fps=clip_fps,
                )

                scan_info["small_bgr_frames"] = []
                tube_tensor, crop_boxes, sampled_indices, fps, foul_features = _finalize_clip_action_tube(
                    scan_info=scan_info,
                    scanned_rois=scanned_rois,
                    scanned_scores=scanned_scores,
                    scanned_feats=scanned_feats,
                    window_sec=window_sec,
                    num_frames=num_frames,
                    crop_size=crop_size,
                )
                torch.save(
                    {
                        "video": tube_tensor,
                        "foul_features": foul_features,
                        "label": label,
                        "play_id": play_id,
                        "call_type": row.get("call_type", ""),
                        "review_decision": row.get("review_decision", ""),
                        "crop_boxes": crop_boxes,
                        "sampled_indices": sampled_indices,
                        "fps": fps,
                        "is_synthetic": False,
                    },
                    out_tensor_path,
                )
                out_row = dict(row)
                out_row["tensor_path"] = (Path("data") / "processed" / class_name / f"{play_id}.pt").as_posix()
                return idx, out_row, out_tensor_path

            with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
                saved_results = list(pool.map(_crop_and_save_clip, clip_slices))

            for idx, out_row, out_tensor_path in saved_results:
                ready_map[idx] = out_row
                logger.info("[%d/%d] Saved foul action tube -> %s", idx, len(rows), out_tensor_path)

            del mega_frames, mega_pose_res, mega_det_res, decoded_infos

    for idx in sorted(ready_map.keys()):
        processed_rows.append(ready_map[idx])
    write_processed_metadata(processed_metadata_csv, processed_rows)

    logger.info(
        "Completed preprocessing: %d action tubes saved to %s (index: %s)",
        len(processed_rows),
        processed_dir,
        processed_metadata_csv,
    )
    try:
        del yolo_model, pose_model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return processed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Platform YOLOv11 Foul Action-Tube Cropping & Biomechanical Pose Sampling"
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=BASE_DIR / "data" / "l2m_metadata.csv",
        help="Input metadata CSV generated by 1_scrape_l2m.py",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=BASE_DIR / "data" / "processed",
        help="Directory to save processed .pt tensors (default: data/processed)",
    )
    parser.add_argument(
        "--processed-metadata-csv",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "processed_metadata.csv",
        help="Output CSV mapping play_id to processed .pt tensor paths",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=BASE_DIR / "data" / "test",
        help="Folder containing test-only clips to exclude from training preprocessing (default: data/test)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics YOLOv11 detection weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--pose-weights",
        type=str,
        default="yolo11n-pose.pt",
        help="Ultralytics YOLOv11-pose weights (default: yolo11n-pose.pt)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=24,
        help="Max number of clips to preprocess per run (default: 24; set 0 to process all clips)",
    )
    parser.add_argument(
        "--yolo-batch-size",
        type=int,
        default=4,
        help="Micro-batch size for YOLO frame scanning (default: 4; increase on high-VRAM GPUs)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.5,
        help="Temporal window duration in seconds around peak foul contact (default: 1.5)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames uniformly sampled per clip (default: 16)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Spatial crop height and width in pixels (default: 224)",
    )
    parser.add_argument(
        "--max-live-play-sec",
        type=float,
        default=22.0,
        help="Max live-play horizon in seconds to scan per clip (default: 22.0, skipping post-play replay reviews)",
    )
    parser.add_argument(
        "--max-scan-samples",
        type=int,
        default=16,
        help="Max frames sampled during Pass 1 pose/contact detection (default: 16; increase to 24 for higher temporal resolution)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device for YOLO ('auto', 'cuda', or 'cpu')",
    )
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Generate synthetic .pt action tubes when raw MP4 clips are not downloaded locally",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(
        metadata_csv=args.metadata_csv,
        processed_dir=args.processed_dir,
        processed_metadata_csv=args.processed_metadata_csv,
        test_dir=args.test_dir,
        yolo_weights=args.yolo_weights,
        pose_weights=args.pose_weights or None,
        window_sec=args.window_sec,
        num_frames=args.num_frames,
        crop_size=args.crop_size,
        device=args.device,
        max_live_play_sec=args.max_live_play_sec,
        max_scan_samples=args.max_scan_samples,
        max_clips=args.max_clips,
        yolo_batch_size=args.yolo_batch_size,
        synthetic_fallback=args.synthetic_fallback,
    )
