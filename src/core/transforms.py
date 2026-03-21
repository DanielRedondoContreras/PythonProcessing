# ============================================================
# File: src/core/transforms.py
# Matemática SE(3)/SO(3) centralizada
#
# Módulo único para todas las operaciones de transformación rígida del repositorio.  
#
# Convención de quaterniones: orden [x, y, z, w] (Unity / Hamilton).
# ============================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


# ----------------------------------------------------------------
# Quaternion → Matriz de rotación
# ----------------------------------------------------------------

def quat_to_rotmat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convierte un quaternion [x, y, z, w] a matriz de rotación 3×3.

    Normaliza internamente para robustez.  Si la norma es prácticamente nula devuelve la identidad.

    Parámetros
    ----------
    q_xyzw : np.ndarray, shape (4,)
        Quaternion en orden [x, y, z, w].

    Devuelve
    --------
    R : np.ndarray, shape (3, 3), dtype float64
    """
    q = np.asarray(q_xyzw, dtype=np.float64).ravel()
    if q.shape[0] != 4:
        raise ValueError(f"Se esperaba quaternion de 4 elementos, recibido shape={q.shape}")

    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / n
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def quat_dict_to_array(q: Dict[str, Any]) -> np.ndarray:
    """
    Extrae un quaternion [x, y, z, w] desde un dict Unity
    ``{"x": …, "y": …, "z": …, "w": …}``.

    Devuelve
    --------
    np.ndarray, shape (4,), dtype float64
    """
    return np.array(
        [float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])],
        dtype=np.float64,
    )


def quat_dict_to_rotmat(q: Dict[str, Any]) -> np.ndarray:
    """
    Conveniencia: dict quaternion → matriz de rotación 3×3.

    Equivale a ``quat_to_rotmat_xyzw(quat_dict_to_array(q))``.
    """
    return quat_to_rotmat_xyzw(quat_dict_to_array(q))


# ----------------------------------------------------------------
# Quaternion — operaciones
# ----------------------------------------------------------------

def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Producto Hamilton de dos quaterniones en orden [x, y, z, w].

    Devuelve el quaternion resultado normalizado.
    """
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).ravel()
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).ravel()

    q = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    return q


# ----------------------------------------------------------------
# Construcción de matrices SE(3)
# ----------------------------------------------------------------

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construye una matriz homogénea 4×4 a partir de R (3×3) y t (3,).
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def pose_dict_to_T44(pose: Dict[str, Any]) -> np.ndarray:
    """
    Construye T 4×4 desde un dict de pose Unity con claves
    ``Position`` y ``Rotation``.

    Parámetros
    ----------
    pose : dict
        Debe contener ``pose["Position"]`` con ``{x, y, z}``
        y ``pose["Rotation"]`` con ``{x, y, z, w}``.

    Devuelve
    --------
    T : np.ndarray, shape (4, 4), dtype float64
    """
    p = pose["Position"]
    t = np.array([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float64)
    R = quat_dict_to_rotmat(pose["Rotation"])
    return make_T(R, t)


def as_T(x: Any) -> np.ndarray:
    """
    Convierte una lista/array a una matriz 4×4 de tipo float64, lanzando ``ValueError`` si la forma no es (4, 4).
    """
    T = np.array(x, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Se esperaba transformación 4×4, recibido shape={T.shape}")
    return T


# ----------------------------------------------------------------
# Descomposición de matrices SE(3)
# ----------------------------------------------------------------

def T_to_Rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae (R, t) de una matriz homogénea 4×4.
    """
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy()


def T_to_translation_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Descompone una T 4×4 en traslación (3,) y quaternion [x, y, z, w] (4,).

    Usa el algoritmo de Shepperd para robustez numérica.

    Devuelve
    --------
    t : np.ndarray, shape (3,)
    q_xyzw : np.ndarray, shape (4,)
    """
    T = np.asarray(T, dtype=np.float64)
    t = T[:3, 3].copy()

    R = T[:3, :3]
    tr = float(np.trace(R))

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    return t, q


# ----------------------------------------------------------------
# Ángulos y errores de rotación
# ----------------------------------------------------------------

def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Ángulo geodésico (en grados) de una matriz de rotación 3×3.

    Usa la fórmula ``θ = arccos((tr(R) − 1) / 2)`` con clamp numérico para evitar errores de dominio.
    """
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Matriz de rotación 3×3 por la fórmula de Rodrigues: rotación de ``angle_rad`` radianes alrededor de ``axis``.

    El eje se normaliza internamente.
    """
    axis = np.asarray(axis, dtype=np.float64).ravel()
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    return R


# ----------------------------------------------------------------
# Errores de pose
# ----------------------------------------------------------------

def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Dict[str, float]:
    """
    Calcula error de traslación (m) y rotación geodésica (deg) entre dos transformaciones SE(3).

    Devuelve
    --------
    dict con claves ``"trans_error_m"`` y ``"rot_error_deg"``.
    """
    t_est, t_gt = T_est[:3, 3], T_gt[:3, 3]
    R_est, R_gt = T_est[:3, :3], T_gt[:3, :3]

    trans_m = float(np.linalg.norm(t_est - t_gt))
    R_diff = R_est @ R_gt.T
    rot_deg = rotation_angle_deg(R_diff)
    return {"trans_error_m": trans_m, "rot_error_deg": rot_deg}


def delta_T(T_prev: np.ndarray, T_now: np.ndarray) -> np.ndarray:
    """
    Transformación relativa: ``T_prev⁻¹ · T_now``.
    """
    return np.linalg.inv(T_prev) @ T_now


def temporal_error(
    T_est_prev: np.ndarray,
    T_est_now: np.ndarray,
    T_gt_prev: np.ndarray,
    T_gt_now: np.ndarray,
) -> Dict[str, float]:
    """
    Error de pose sobre los *incrementos* temporales entre dos frames consecutivos (estimado vs ground truth).
    """
    d_est = delta_T(T_est_prev, T_est_now)
    d_gt = delta_T(T_gt_prev, T_gt_now)
    return pose_error(d_est, d_gt)


# ----------------------------------------------------------------
# Validación SO(3) / SE(3)
# ----------------------------------------------------------------

def validate_rotation_matrix(
    R: np.ndarray,
    *,
    det_eps: float = 1e-5,
    orth_eps: float = 1e-5,
) -> None:
    """
    Valida que R sea una rotación propia (SO(3)).

    Lanza ``AssertionError`` si:
      - ``|det(R) − 1| > det_eps``
      - ``‖R·Rᵀ − I‖_F > orth_eps``
    """
    det = float(np.linalg.det(R))
    assert abs(det - 1.0) < det_eps, f"det(R) no es 1 (det={det:.8f})"

    ortho_error = float(np.linalg.norm(R @ R.T - np.eye(3)))
    assert ortho_error < orth_eps, f"R no es ortogonal (error={ortho_error:.8f})"


def se3_sanity_check(
    T: np.ndarray,
    *,
    orth_eps: float = 1e-2,
    det_eps: float = 1e-2,
) -> Dict[str, Any]:
    """
    Sanity checks formales para una transformación SE(3).

    Devuelve un dict con:
      - ``se3_ok``       : bool
      - ``se3_reason``   : str  ("ok" | "bad_shape" | "nan_or_inf" |
                                  "non_orthogonal" | "det_not_1" | "exception")
      - ``det_R``        : float | None
      - ``orth_err_fro`` : float | None
      - ``has_nan``      : bool | None
    """
    out: Dict[str, Any] = {
        "se3_ok": False,
        "se3_reason": "unknown",
        "det_R": None,
        "orth_err_fro": None,
        "has_nan": True,
    }

    try:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            out["se3_reason"] = "bad_shape"
            return out

        if not np.isfinite(T).all():
            out["se3_reason"] = "nan_or_inf"
            out["has_nan"] = True
            return out

        out["has_nan"] = False
        R = T[:3, :3]
        I = np.eye(3, dtype=np.float64)
        orth_err = float(np.linalg.norm(R.T @ R - I, ord="fro"))
        det_R = float(np.linalg.det(R))

        out["orth_err_fro"] = orth_err
        out["det_R"] = det_R

        if orth_err > float(orth_eps):
            out["se3_reason"] = "non_orthogonal"
            return out

        if abs(det_R - 1.0) > float(det_eps):
            out["se3_reason"] = "det_not_1"
            return out

        out["se3_ok"] = True
        out["se3_reason"] = "ok"
        return out
    except Exception:
        out["se3_reason"] = "exception"
        return out