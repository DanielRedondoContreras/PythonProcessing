# ============================================================
# File: src/core/temporal_stabilizer.py
# Estabilización temporal de pose
#
# Proporciona:
#   1. Rechazo de saltos implausibles: descarta frames donde el desplazamiento (traslación o rotación) respecto al último frame aceptado excede un umbral físicamente plausible para la frecuencia de captura del dispositivo.
#   2. Filtro EMA (Exponential Moving Average) sobre traslación y SLERP-EMA sobre rotación del cuaternión, aplicados únicamente sobre la secuencia de frames aceptados.
#   3. Generación de la trayectoria estabilizada como lista de dicts (mismas columnas que trajectory_object_world.csv) más campos adicionales de diagnóstico).
#
# El módulo NO modifica los archivos per-frame (pose_best.json, pose_eval.json, etc.).  Actúa exclusivamente sobre la trayectoria agregada producida por run_session_summary.
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.transforms import (
    T_to_translation_quat_xyzw,
    quat_to_rotmat_xyzw,
    rotation_angle_deg,
)


# ----------------------------------------------------------------
# Quaternion SLERP (esférica lineal)
# ----------------------------------------------------------------

def _quat_dot(q1: np.ndarray, q2: np.ndarray) -> float:
    """Producto escalar de dos quaterniones [x, y, z, w]."""
    return float(np.dot(q1, q2))


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolación esférica lineal entre q0 y q1 con parámetro t ∈ [0, 1].

    Garantiza el camino corto (flipping si dot < 0).

    Parámetros
    ----------
    q0, q1 : np.ndarray, shape (4,)
        Quaterniones [x, y, z, w], normalizados internamente.
    t : float
        0 → q0, 1 → q1.

    Devuelve
    --------
    np.ndarray, shape (4,), normalizado.
    """
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))

    dot = _quat_dot(q0, q1)

    # Camino corto
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Si son prácticamente iguales, interpolación lineal (evita div/0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return _quat_normalize(result)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return _quat_normalize(s0 * q0 + s1 * q1)


# ----------------------------------------------------------------
# Detección de saltos implausibles
# ----------------------------------------------------------------

def _is_jump(
    t_prev: np.ndarray,
    q_prev: np.ndarray,
    t_curr: np.ndarray,
    q_curr: np.ndarray,
    max_trans_m: float,
    max_rot_deg: float,
) -> bool:
    """
    Devuelve True si la transición entre dos poses supera los umbrales de traslación o rotación.
    """
    dt = float(np.linalg.norm(t_curr - t_prev))
    if dt > max_trans_m:
        return True

    R_prev = quat_to_rotmat_xyzw(q_prev)
    R_curr = quat_to_rotmat_xyzw(q_curr)
    R_diff = R_curr @ R_prev.T
    dr = rotation_angle_deg(R_diff)
    if dr > max_rot_deg:
        return True

    return False


# ----------------------------------------------------------------
# Filtro EMA sobre trayectoria
# ----------------------------------------------------------------

def _ema_translation(
    t_prev_smooth: np.ndarray,
    t_curr_raw: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """EMA simple: t_smooth = alpha * t_raw + (1 - alpha) * t_prev_smooth."""
    return alpha * t_curr_raw + (1.0 - alpha) * t_prev_smooth


def _ema_rotation(
    q_prev_smooth: np.ndarray,
    q_curr_raw: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """EMA esférico vía SLERP: q_smooth = slerp(q_prev_smooth, q_curr_raw, alpha)."""
    return slerp(q_prev_smooth, q_curr_raw, alpha)


# ----------------------------------------------------------------
# API pública: estabilizar trayectoria
# ----------------------------------------------------------------

def stabilize_trajectory(
    raw_rows: List[Dict[str, Any]],
    *,
    ema_alpha: float = 0.4,
    max_trans_m: float = 0.15,
    max_rot_deg: float = 45.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Estabiliza la trayectoria cruda producida por ``build_trajectory_rows``.

    Lógica
    ------
    1. Recorre los frames en orden de ``frame_index``.
    2. Solo considera frames con ``valid_frame=True`` y pose no-None.
    3. Para el primer frame válido: lo acepta directamente como
       seed del filtro EMA.
    4. Para cada frame válido posterior:
       a. Si el salto respecto al último frame **aceptado** supera
          ``max_trans_m`` (m) o ``max_rot_deg`` (°), lo marca como
          ``rejected_jump=True`` y propaga la última pose suavizada.
       b. Si no hay salto, aplica EMA(alpha) sobre traslación y
          SLERP-EMA(alpha) sobre rotación.
    5. Frames con ``valid_frame=False`` se copian tal cual con
       ``stab_valid=False``.

    Parámetros
    ----------
    raw_rows : list of dict
        Filas producidas por ``build_trajectory_rows``.
    ema_alpha : float
        Factor de suavizado ∈ (0, 1]. 1.0 = sin suavizado (solo rechazo).
    max_trans_m : float
        Umbral máximo de traslación (m) entre frames consecutivos válidos.
    max_rot_deg : float
        Umbral máximo de rotación (°) entre frames consecutivos válidos.

    Devuelve
    --------
    (stabilized_rows, stats)
        stabilized_rows tiene las mismas columnas que raw_rows más:
          - stab_valid       (bool): True si el frame contribuyó al filtro.
          - rejected_jump    (bool): True si fue rechazado por salto.
          - stab_tx/ty/tz    (float|None): pose suavizada (traslación).
          - stab_qx/qy/qz/qw (float|None): pose suavizada (rotación).
        stats contiene métricas de estabilización para session_summary.
    """
    ema_alpha = max(0.01, min(1.0, ema_alpha))

    # Estado del filtro
    prev_t: Optional[np.ndarray] = None      # traslación cruda del último aceptado
    prev_q: Optional[np.ndarray] = None      # quaternión crudo del último aceptado
    smooth_t: Optional[np.ndarray] = None    # traslación suavizada
    smooth_q: Optional[np.ndarray] = None    # quaternión suavizado

    n_accepted = 0
    n_rejected_jump = 0
    n_invalid = 0
    jump_frames: List[int] = []

    stabilized: List[Dict[str, Any]] = []

    for row in raw_rows:
        out = dict(row)  # copia
        frame_idx = row.get("frame_index", -1)
        valid = bool(row.get("valid_frame", False))

        tx = row.get("tx")
        ty = row.get("ty")
        tz = row.get("tz")
        qx = row.get("qx")
        qy = row.get("qy")
        qz = row.get("qz")
        qw = row.get("qw")

        has_pose = all(v is not None for v in (tx, ty, tz, qx, qy, qz, qw))

        if (not valid) or (not has_pose):
            # Frame inválido o sin pose: propagar última pose suavizada si existe
            out["stab_valid"] = False
            out["rejected_jump"] = False
            if smooth_t is not None and smooth_q is not None:
                out["stab_tx"] = float(smooth_t[0])
                out["stab_ty"] = float(smooth_t[1])
                out["stab_tz"] = float(smooth_t[2])
                out["stab_qx"] = float(smooth_q[0])
                out["stab_qy"] = float(smooth_q[1])
                out["stab_qz"] = float(smooth_q[2])
                out["stab_qw"] = float(smooth_q[3])
            else:
                out["stab_tx"] = None
                out["stab_ty"] = None
                out["stab_tz"] = None
                out["stab_qx"] = None
                out["stab_qy"] = None
                out["stab_qz"] = None
                out["stab_qw"] = None
            n_invalid += 1
            stabilized.append(out)
            continue

        t_curr = np.array([tx, ty, tz], dtype=np.float64)
        q_curr = np.array([qx, qy, qz, qw], dtype=np.float64)

        if prev_t is None:
            # Primer frame válido: seed
            prev_t = t_curr.copy()
            prev_q = q_curr.copy()
            smooth_t = t_curr.copy()
            smooth_q = q_curr.copy()
            n_accepted += 1

            out["stab_valid"] = True
            out["rejected_jump"] = False
            out["stab_tx"] = float(smooth_t[0])
            out["stab_ty"] = float(smooth_t[1])
            out["stab_tz"] = float(smooth_t[2])
            out["stab_qx"] = float(smooth_q[0])
            out["stab_qy"] = float(smooth_q[1])
            out["stab_qz"] = float(smooth_q[2])
            out["stab_qw"] = float(smooth_q[3])
            stabilized.append(out)
            continue

        # Comprobar salto respecto al último frame ACEPTADO (crudo)
        if _is_jump(prev_t, prev_q, t_curr, q_curr, max_trans_m, max_rot_deg):
            # Rechazar: propagar la última pose suavizada
            n_rejected_jump += 1
            jump_frames.append(frame_idx)
            out["stab_valid"] = False
            out["rejected_jump"] = True
            out["stab_tx"] = float(smooth_t[0])
            out["stab_ty"] = float(smooth_t[1])
            out["stab_tz"] = float(smooth_t[2])
            out["stab_qx"] = float(smooth_q[0])
            out["stab_qy"] = float(smooth_q[1])
            out["stab_qz"] = float(smooth_q[2])
            out["stab_qw"] = float(smooth_q[3])
            stabilized.append(out)
            continue

        # Aceptar: actualizar EMA
        smooth_t = _ema_translation(smooth_t, t_curr, ema_alpha)
        smooth_q = _ema_rotation(smooth_q, q_curr, ema_alpha)

        prev_t = t_curr.copy()
        prev_q = q_curr.copy()
        n_accepted += 1

        out["stab_valid"] = True
        out["rejected_jump"] = False
        out["stab_tx"] = float(smooth_t[0])
        out["stab_ty"] = float(smooth_t[1])
        out["stab_tz"] = float(smooth_t[2])
        out["stab_qx"] = float(smooth_q[0])
        out["stab_qy"] = float(smooth_q[1])
        out["stab_qz"] = float(smooth_q[2])
        out["stab_qw"] = float(smooth_q[3])
        stabilized.append(out)

    # --- Métricas de estabilización ---
    stats: Dict[str, Any] = {
        "ema_alpha": ema_alpha,
        "max_trans_m": max_trans_m,
        "max_rot_deg": max_rot_deg,
        "n_frames_total": len(raw_rows),
        "n_accepted": n_accepted,
        "n_rejected_jump": n_rejected_jump,
        "n_invalid_or_no_pose": n_invalid,
        "rejected_jump_frames": jump_frames,
    }

    return stabilized, stats