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

import argparse
import csv
import gc
import hashlib
import logging
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


def analyze_foul_interaction_frame(
    pose_res,
    det_res,
    frame_w: int,
    frame_h: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
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

    active_res = pose_res if (pose_res is not None and pose_res.boxes is not None and len(pose_res.boxes) > 0) else det_res
    if active_res is None or active_res.boxes is None or len(active_res.boxes) == 0:
        return default_roi, 0.0, feats

    xyxy = active_res.boxes.xyxy.cpu().numpy().astype(np.float32)
    xyxy[:, [0, 2]] *= scale_x
    xyxy[:, [1, 3]] *= scale_y
    cls = active_res.boxes.cls.cpu().numpy().astype(int)
    conf = active_res.boxes.conf.cpu().numpy().astype(np.float32)

    person_mask = (cls == COCO_PERSON_CLS) & (conf >= 0.25)
    player_boxes = xyxy[person_mask]

    kpts_all: Optional[np.ndarray] = None
    if (
        pose_res is not None
        and getattr(pose_res, "keypoints", None) is not None
        and pose_res.keypoints.xy is not None
        and len(pose_res.keypoints.xy) == len(xyxy)
    ):
        raw_kpts = pose_res.keypoints.xy.cpu().numpy().astype(np.float32)[person_mask]
        raw_kpts[:, :, 0] *= scale_x
        raw_kpts[:, :, 1] *= scale_y
        kpts_all = raw_kpts

    # Filter court players: reject foreground sideline (referees, bench personnel, photographers)
    # and background stands / ceiling. In broadcast basketball camera views, the active court floor
    # spans y in [0.22*H, 0.68*H] with y2 <= 0.72*H.
    if len(player_boxes) > 0:
        cy = 0.5 * (player_boxes[:, 1] + player_boxes[:, 3])
        cx = 0.5 * (player_boxes[:, 0] + player_boxes[:, 2])
        bh = player_boxes[:, 3] - player_boxes[:, 1]
        bw = player_boxes[:, 2] - player_boxes[:, 0]
        court_mask = (
            (player_boxes[:, 3] <= frame_h * 0.72)
            & (player_boxes[:, 1] >= frame_h * 0.15)
            & (cy >= frame_h * 0.22)
            & (cy <= frame_h * 0.68)
            & (cx >= frame_w * 0.10)
            & (cx <= frame_w * 0.90)
            & (bh >= frame_h * 0.08)
            & (bh <= frame_h * 0.55)
            & (bw >= frame_w * 0.02)
            & (bw <= frame_w * 0.35)
            & ((bh / np.maximum(bw, 1.0)) >= 0.90)
        )
        if int(np.sum(court_mask)) >= 2:
            player_boxes = player_boxes[court_mask]
            if kpts_all is not None:
                kpts_all = kpts_all[court_mask]
        elif int(np.sum(court_mask)) == 1:
            player_boxes = player_boxes[court_mask]
            if kpts_all is not None:
                kpts_all = kpts_all[court_mask]
        elif len(player_boxes) > 2:
            box_heights = player_boxes[:, 3] - player_boxes[:, 1]
            max_h = float(np.max(box_heights))
            h_mask = box_heights >= (max_h * 0.38)
            if int(np.sum(h_mask)) >= 2:
                player_boxes = player_boxes[h_mask]
                if kpts_all is not None:
                    kpts_all = kpts_all[h_mask]

    n_players = len(player_boxes)
    if n_players == 0:
        return default_roi, 0.0, feats
    if n_players == 1:
        b = player_boxes[0]
        feats[9] = 0.1
        feats[11] = 0.2
        return b.astype(np.float32), 0.2, feats

    ball_pt: Optional[np.ndarray] = None
    if det_res is not None and det_res.boxes is not None and len(det_res.boxes) > 0:
        d_xyxy = det_res.boxes.xyxy.cpu().numpy().astype(np.float32)
        d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
        d_conf = det_res.boxes.conf.cpu().numpy().astype(np.float32)
        b_mask = (d_cls == COCO_SPORTS_BALL_CLS) & (d_conf >= 0.15)
        if np.any(b_mask):
            b_boxes = d_xyxy[b_mask]
            b_confs = d_conf[b_mask]
            best_b = b_boxes[int(np.argmax(b_confs))]
            ball_pt = np.array(
                [(best_b[0] + best_b[2]) * 0.5 * scale_x, (best_b[1] + best_b[3]) * 0.5 * scale_y],
                dtype=np.float32,
            )

    centers = np.column_stack(
        [
            (player_boxes[:, 0] + player_boxes[:, 2]) * 0.5,
            (player_boxes[:, 1] + player_boxes[:, 3]) * 0.5,
        ]
    )

    best_pair = (0, 1)
    best_pair_score = -1.0
    best_pair_metrics = (diag, diag, diag, 0.0)

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
            ball_prox = 0.0
            if ball_pt is not None:
                pair_mid = 0.5 * (centers[i] + centers[j])
                ball_prox = max(0.0, 1.0 - float(np.linalg.norm(pair_mid - ball_pt)) / max(diag * 0.25, 1.0))

            pair_score = (
                0.45 * norm_arm_upper
                + 0.25 * norm_torso
                + 0.20 * min(1.0, iou * 3.0)
                + 0.10 * ball_prox
            )
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (i, j)
                best_pair_metrics = (arm_to_arm_dist, arm_to_upper_dist, torso_dist, iou)

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

    # 1. Arm Contact Intrusion (Defender hand/wrist striking shooter arm)
    arms_off = valid_points(kpts_off, ARM_CONTACT_KPTS) if kpts_off is not None else np.empty((0, 2))
    arms_def = valid_points(kpts_def, ARM_CONTACT_KPTS) if kpts_def is not None else np.empty((0, 2))
    arm_dist = min_point_set_distance(arms_off, arms_def, arm_to_arm_dist)
    arm_contact_val = float(np.clip(1.0 - (arm_dist / max(h_off * 0.40, 1.0)), 0.0, 1.0))

    # 2. Body Cylinder Penetration (Defender invading offensive established space)
    torso_off = valid_points(kpts_off, TORSO_KPTS) if kpts_off is not None else np.empty((0, 2))
    torso_def = valid_points(kpts_def, TORSO_KPTS) if kpts_def is not None else np.empty((0, 2))
    torso_d = min_point_set_distance(torso_off, torso_def, torso_dist)
    cylinder_pen_val = float(np.clip(0.55 * (1.0 - torso_d / max(h_off * 0.65, 1.0)) + 0.45 * min(1.0, pair_iou * 2.5), 0.0, 1.0))

    # 3. Lower Body Landing Violation (Defender undercut in shooter landing zone)
    feet_off = valid_points(kpts_off, (13, 14, 15, 16)) if kpts_off is not None else np.empty((0, 2))
    feet_def = valid_points(kpts_def, (13, 14, 15, 16)) if kpts_def is not None else np.empty((0, 2))
    feet_dist = min_point_set_distance(feet_off, feet_def, 0.5 * (h_off + h_def))
    landing_violation_val = float(np.clip(1.0 - (feet_dist / max(h_off * 0.45, 1.0)), 0.0, 1.0))

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

    # 9. Ball Strike vs Body Ratio (Clean play on ball vs illegal body collision)
    if ball_pt is not None:
        c_def = 0.5 * (box_def[:2] + box_def[2:])
        c_off = 0.5 * (box_off[:2] + box_off[2:])
        d_ball = float(np.linalg.norm(c_def - ball_pt))
        d_body = float(np.linalg.norm(c_def - c_off))
        ball_ratio_val = float(np.clip(d_ball / max(d_body + 1.0, 1.0), 0.0, 1.0))
    else:
        ball_ratio_val = 0.35

    # 10. Relative Closing Velocity
    closing_vel_val = 0.50

    # 11. Momentum Transfer Shock
    momentum_shock_val = float(np.clip(pair_iou * 1.2, 0.0, 1.0))

    # 12. Airborne Contest Apex Differential
    apex_diff_val = float(np.clip(0.5 + 0.5 * (elev_off - elev_def), 0.0, 1.0))

    # Composite physical contact score
    physical_contact_score = float(
        0.35 * arm_contact_val
        + 0.30 * cylinder_pen_val
        + 0.20 * landing_violation_val
        + 0.15 * verticality_val
    )

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


def extract_action_tube(
    video_path: Path | str,
    det_model: Optional[object] = None,
    pose_model: Optional[object] = None,
    window_sec: float = 1.5,
    num_frames: int = 16,
    crop_size: int = 224,
    device: str = "cpu",
    max_live_play_sec: float = 15.0,
    max_scan_samples: int = 12,
    yolo_batch_size: int = 4,
    **kwargs,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], List[int], float, torch.Tensor]:
    """
    Cross-Platform Memory-Safe & Deadlock-Free Action-Tube Extractor:
      1. Caps live-play scanning to the first `max_live_play_sec` (15.0s) so multi-minute
         replay review clips never exhaust RAM on 8GB laptops.
      2. Decodes at most `max_scan_samples` (12) downscaled `640x360` frames and calls
         `cap.release()` BEFORE running YOLOv11-pose in micro-batches (`yolo_batch_size=4`).
      3. Reopens the video after YOLO completes to read ONLY the 16 sampled frames in the
         1.5s window around peak foul contact.
    """
    if "max_scan_frames" in kwargs:
        max_scan_samples = kwargs["max_scan_frames"]
    if "ema_alpha" in kwargs:
        pass
    primary_pose = pose_model if pose_model is not None else det_model
    if primary_pose is None:
        raise ValueError("At least one YOLO model (pose_model or det_model) must be provided.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or np.isnan(fps):
        fps = 30.0
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    max_horizon_frames = max(int(round(fps * max_live_play_sec)), num_frames * 2)
    effective_total = min(reported_frames, max_horizon_frames) if reported_frames > 0 else max_horizon_frames

    scan_start = int(effective_total * 0.12) if effective_total > 40 else 0
    scan_end = int(effective_total * 0.88) if effective_total > 40 else max(effective_total - 1, 1)
    target_scan_set = set(
        np.linspace(scan_start, scan_end, num=min(max_scan_samples, max(scan_end - scan_start + 1, 1)))
        .round()
        .astype(int)
        .tolist()
    )
    max_scan_frame = max(target_scan_set) if target_scan_set else 0

    scan_w, scan_h = 640, 360
    scale_x = float(frame_w) / float(scan_w)
    scale_y = float(frame_h) / float(scan_h)

    scanned_frames: List[int] = []
    small_bgr_frames: List[np.ndarray] = []

    frame_idx = 0
    actual_total = 0
    while frame_idx <= max_scan_frame:
        if frame_idx in target_scan_set:
            ret, frame = cap.read()
            if not ret:
                break
            small_bgr_frames.append(cv2.resize(frame, (scan_w, scan_h), interpolation=cv2.INTER_LINEAR))
            scanned_frames.append(frame_idx)
        else:
            if not cap.grab():
                break
        frame_idx += 1
        actual_total = frame_idx

    # Release FFmpeg decoder BEFORE invoking PyTorch/Ultralytics
    cap.release()

    if not small_bgr_frames:
        raise RuntimeError(f"No readable video frames in {video_path}")

    scanned_rois: List[np.ndarray] = []
    scanned_scores: List[float] = []
    scanned_feats: List[np.ndarray] = []

    yolo_device = select_yolo_device(device)
    batch_sz = max(1, int(yolo_batch_size))
    with torch.inference_mode():
        for b_start in range(0, len(small_bgr_frames), batch_sz):
            batch_frames = small_bgr_frames[b_start : b_start + batch_sz]
            pose_results = primary_pose.predict(
                batch_frames,
                conf=0.25,
                imgsz=480,
                verbose=False,
                device=yolo_device,
            )
            for pose_res in pose_results:
                roi, score, feat_vec = analyze_foul_interaction_frame(
                    pose_res=pose_res,
                    det_res=None,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
                scanned_rois.append(roi)
                scanned_scores.append(score)
                scanned_feats.append(feat_vec)

    del small_bgr_frames

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

    # Pass 2: Open video and decode ONLY the 16 sampled frames
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

    stacked_rgb = np.stack(cropped_rgb_frames, axis=0)  # (16, 224, 224, 3)
    tube_tensor = torch.from_numpy(stacked_rgb).permute(3, 0, 1, 2).contiguous()  # (3, 16, 224, 224)
    foul_features_tensor = torch.from_numpy(np.stack(sampled_feats_list, axis=0)).float()  # (16, 12)

    return tube_tensor, crop_boxes, sampled_indices, fps, foul_features_tensor


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
    max_live_play_sec: float = 15.0,
    max_scan_samples: int = 12,
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
        logger.info("Loading Ultralytics pose/detection model: %s (device=%s)", active_pose_weights, yolo_dev)
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

    for idx, row in enumerate(rows, start=1):
        play_id = row["play_id"]
        label = int(row["label"])
        video_path = resolve_project_path(row["video_path"])
        class_name = "foul" if label == 1 else "no_foul"
        out_tensor_path = processed_dir / class_name / f"{play_id}.pt"
        out_tensor_path.parent.mkdir(parents=True, exist_ok=True)

        used_synthetic = False
        needs_reprocess = not out_tensor_path.exists()
        if out_tensor_path.exists() and video_path.exists() and video_path.stat().st_size > 10_240:
            try:
                existing_pt = torch.load(out_tensor_path, map_location="cpu", weights_only=False)
                if isinstance(existing_pt, dict) and existing_pt.get("is_synthetic", True):
                    needs_reprocess = True
            except Exception:
                needs_reprocess = True

        if needs_reprocess:
            if video_path.exists() and video_path.stat().st_size > 10_240 and pose_model is not None:
                try:
                    tube_tensor, crop_boxes, sampled_indices, fps, foul_features = extract_action_tube(
                        video_path=video_path,
                        det_model=None,
                        pose_model=pose_model,
                        window_sec=window_sec,
                        num_frames=num_frames,
                        crop_size=crop_size,
                        device=device,
                        max_live_play_sec=max_live_play_sec,
                        max_scan_samples=max_scan_samples,
                        yolo_batch_size=yolo_batch_size,
                    )
                except Exception as exc:
                    logger.warning("Failed to preprocess %s: %s", video_path, exc)
                    if not synthetic_fallback:
                        continue
                    used_synthetic = True
                    tube_tensor, crop_boxes, sampled_indices, fps, foul_features = synthesize_fallback_action_tube(
                        play_id=play_id,
                        label=label,
                        num_frames=num_frames,
                        crop_size=crop_size,
                    )
                finally:
                    gc.collect()
            elif synthetic_fallback:
                used_synthetic = True
                tube_tensor, crop_boxes, sampled_indices, fps, foul_features = synthesize_fallback_action_tube(
                    play_id=play_id,
                    label=label,
                    num_frames=num_frames,
                    crop_size=crop_size,
                )
            else:
                logger.warning("Skipping missing raw video: %s (pass --synthetic-fallback for offline tensor demo)", video_path)
                continue

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
                    "is_synthetic": used_synthetic,
                },
                out_tensor_path,
            )
            logger.info(
                "[%d/%d] Saved foul action tube -> %s",
                idx,
                len(rows),
                out_tensor_path,
            )

        out_row = dict(row)
        rel_tensor = Path("data") / "processed" / class_name / f"{play_id}.pt"
        out_row["tensor_path"] = rel_tensor.as_posix()
        processed_rows.append(out_row)
        write_processed_metadata(processed_metadata_csv, processed_rows)

    logger.info(
        "Completed preprocessing: %d action tubes saved to %s (index: %s)",
        len(processed_rows),
        processed_dir,
        processed_metadata_csv,
    )
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
        default=15.0,
        help="Max live-play horizon in seconds to scan per clip (default: 15.0, skipping replay reviews)",
    )
    parser.add_argument(
        "--max-scan-samples",
        type=int,
        default=12,
        help="Max frames sampled during Pass 1 pose/contact detection (default: 12; increase to 24 for higher temporal resolution)",
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
