# ============================================================
# File: src/core/pose_eval.py
# Evaluación de pose estimada contra ground truth
#
# Proporciona:
#   - Cálculo de error de pose (traslación y rotación geodésica)
#   - Carga de GT y pose estimada desde archivos JSON
#   - Clasificación de frames por umbrales configurables
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.data_loader import load_controller_pose
from src.core.io_utils import read_json
from src.core.transforms import (
    as_T,
    make_T,
    pose_dict_to_T44,
    quat_to_rotmat_xyzw,
    rotation_angle_deg,
)


# -------------------------
# Matemática de pose
# -------------------------
def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Dict[str, float]:
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

    trans_m = float(np.linalg.norm(t_est - t_gt))
    R_diff = R_est @ R_gt.T
    rot_deg = rotation_angle_deg(R_diff)
    return {"trans_error_m": trans_m, "rot_error_deg": rot_deg}


def delta_T(T_prev: np.ndarray, T_now: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_prev) @ T_now


def temporal_error(
    T_est_prev: np.ndarray,
    T_est_now: np.ndarray,
    T_gt_prev: np.ndarray,
    T_gt_now: np.ndarray,
) -> Dict[str, float]:
    d_est = delta_T(T_est_prev, T_est_now)
    d_gt = delta_T(T_gt_prev, T_gt_now)
    return pose_error(d_est, d_gt)


# -------------------------
# Carga de GT / salidas
# -------------------------

def load_gt_T(metadata_path: Path, side: str) -> Optional[np.ndarray]:
    """
    Carga la pose GT del controlador desde metadata.json y devuelve T 4×4.

    Usa la función centralizada load_controller_pose que soporta tanto el formato plano (ControllerPoseRight) como el anidado (ControllerPose.Right).

    Devuelve None si el archivo no existe o no contiene la pose solicitada.
    """
    if not metadata_path.exists():
        return None
    meta = read_json(metadata_path)
    try:
        center, quat = load_controller_pose(meta, side=side)
    except (KeyError, ValueError):
        return None
    R = quat_to_rotmat_xyzw(quat)
    return make_T(R, center)


def load_est_T(pose_best_path: Path) -> Optional[np.ndarray]:
    if not pose_best_path.exists():
        return None
    d = read_json(pose_best_path)
    if "transformation" not in d:
        return None
    return as_T(d["transformation"])


# -------------------------
# Clasificación
# -------------------------
@dataclass
class Thresholds:
    # Error absoluto
    max_trans_m_ok: float = 0.05
    max_rot_deg_ok: float = 10.0
    # Calidad geométrica
    min_fitness_ok: float = 0.15
    max_rmse_ok: float = 0.02
    # Datos mínimos
    min_segment_points_ok: int = 300


def classify_frame(
    *,
    have_est: bool,
    have_gt: bool,
    seg_n_points: Optional[int],
    fitness: Optional[float],
    rmse: Optional[float],
    trans_error_m: Optional[float],
    rot_error_deg: Optional[float],
    th: Thresholds,
) -> str:
    if not have_est:
        return "FAIL_NO_EST"
    if not have_gt:
        return "FAIL_NO_GT"

    if seg_n_points is not None and seg_n_points < th.min_segment_points_ok:
        return "FAIL_NO_DATA"

    if fitness is not None and fitness < th.min_fitness_ok:
        return "FAIL_BAD_ICP"
    if rmse is not None and rmse > th.max_rmse_ok:
        return "FAIL_BAD_ICP"

    if trans_error_m is not None and trans_error_m > th.max_trans_m_ok:
        return "FAIL_POSE"
    if rot_error_deg is not None and rot_error_deg > th.max_rot_deg_ok:
        return "FAIL_POSE"

    return "OK"