# ============================================================
# File: src/core/data_loader.py
# Carga y validación de datos de entrada del pipeline
#
# Proporciona:
#   - Carga de imágenes estéreo y metadatos por frame
#   - Construcción y ajuste de matrices de intrínsecos
#   - Extracción centralizada de poses del controlador
#   - Carga de K_rect desde rectification_debug.json
#   - Cálculo de Z_gt (distancia cámara-controlador)
# ============================================================

import warnings

import cv2
import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.io_utils import read_json
from src.core.transforms import (
    pose_dict_to_T44,
    rotation_angle_deg,
    validate_rotation_matrix,
)

# -----------------------------
# Funciones
# -----------------------------

#Validación de las imágenes
def validate_images(left, right):
    """Valida que dos imágenes estéreo tengan la misma resolución, sean grayscale y uint8."""
    assert left.shape == right.shape, "Left y Right no tienen la misma resolución"
    assert left.ndim == 2 and right.ndim == 2, "Las imágenes deben ser grayscale"
    assert left.dtype == np.uint8 and right.dtype == np.uint8, "Las imágenes deben ser uint8"

#Construcción de matriz K
def build_K(intrinsics):
    """Construye la matriz de intrínsecos K (3×3) desde un dict con fx, fy, cx, cy."""
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return K

#Validación de intrínsecos
def validate_intrinsics(intrinsics, image_shape):
    """Valida que los intrínsecos sean coherentes con el tamaño de la imagen."""
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    h, w = image_shape

    assert fx > 0 and fy > 0

    assert abs(cx - w/2) < 50

#Ajustar intrínsecos al tamaño real de la imagen
def adjust_intrinsics_to_image(intrinsics, image_shape, mode: str = "scale_y"):
    """
    Ajusta intrínsecos cuando (width,height) del SDK no coincide con el tamaño real del PNG.

    mode:
      - "crop_y": asume recorte vertical simétrico (quita arriba/abajo)
      - "scale_y": asume escalado solo en Y (altura), manteniendo ancho
    """
    h_img, w_img = image_shape  # (alto, ancho) imagen real
    intr = dict(intrinsics)

    w_intr = int(intr.get("width", w_img))
    h_intr = int(intr.get("height", h_img))

    # caso que SDK coincida con PNG
    if w_intr == w_img and h_intr == h_img:
        intr["width"] = w_img
        intr["height"] = h_img
        return intr

    # caso que SDK NO coincida con PNG (por ejemplo, intrínsecos 1280x1280 vs imagen 1280x960)
    if w_intr != w_img:
        raise ValueError(
            f"Mismatch de ancho: intr_width={w_intr}, img_width={w_img}. "
        )

    # Aplicamos crop
    if mode == "crop_y":
        offset_y = (h_intr - h_img) / 2.0  
        intr["cy"] = intr["cy"] - offset_y
        
    # Aplicamos scale
    elif mode == "scale_y":
        sy = h_img / float(h_intr)          
        intr["fy"] = intr["fy"] * sy
        intr["cy"] = intr["cy"] * sy
    else:
        raise ValueError(f"Modo desconocido: {mode}")

    intr["width"] = w_img
    intr["height"] = h_img
    return intr


#Validación del baseline 
def validate_baseline(baseline_m):
    """Valida que la baseline estéreo esté dentro del rango esperado (0.05–0.08 m)."""
    assert 0.05 < baseline_m < 0.08, f"Baseline fuera de rango: {baseline_m} m"

#Transformación Relativa Estéreo 
def compute_relative_transform(T_W_L, T_W_R):
    """Calcula la transformación relativa estéreo (R, t, baseline) entre ambas cámaras."""
    T_R_L = np.linalg.inv(T_W_R) @ T_W_L
    R_rel = T_R_L[:3,:3]
    t_rel = T_R_L[:3,3]

    baseline_m = np.linalg.norm(t_rel)

    validate_rotation_matrix(R_rel)
    validate_baseline(baseline_m)

    return R_rel, t_rel, baseline_m

#Entrar en frames de sesión 
def get_frame_dir(session_path: Path, frame_index: int) -> Path:
    """Devuelve la ruta al directorio de un frame concreto, validando su existencia."""
    frame_dir = session_path / f"frame_{frame_index:06d}"
    if not frame_dir.exists():
        raise FileNotFoundError(f"No existe el frame: {frame_dir}")
    return frame_dir

#Cargar imágenes (left/right) de cada frame
def load_images_from_frame(frame_dir: Path):
    """Carga las imágenes left.png y right.png desde el directorio de un frame."""
    left_path = frame_dir / "left.png"
    right_path = frame_dir / "right.png"

    print("Leyendo:", left_path)
    print("Leyendo:", right_path)

    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

    if left is None or right is None:
        raise ValueError("OpenCV no pudo leer las imágenes.")

    validate_images(left, right)
    return left, right

#Cargar metadatos de cada frame
def load_metadata_from_frame(frame_dir: Path):
    """Carga y parsea el archivo metadata.json de un frame."""
    meta_path = frame_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No se encontró metadata.json en: {meta_path}")

    print("Leyendo:", meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        import json
        metadata = json.load(f)

    return metadata


# ---------------------------------------------------------------
# Carga centralizada de K_rect desde rectification_debug.json
# ---------------------------------------------------------------

def _warn_fx_fy_mismatch(fx: float, fy: float, source: str) -> None:
    """Emite warning si |fx - fy| > 1 px (no esperado en rectificación estándar)."""
    diff = abs(fx - fy)
    if diff > 1.0:
        warnings.warn(
            f"K_rect ({source}): |fx - fy| = {diff:.2f} px > 1 px. "
            "En rectificación estéreo estándar con CALIB_ZERO_DISPARITY, ambos valores deberían ser prácticamente iguales. "
            "Verificar que los intrínsecos rectificados son correctos.",
            stacklevel=3,
        )


def _k_rect_from_p1(p1_raw: object, source: str) -> tuple[np.ndarray, str]:
    """Extrae K_rect (3×3) desde una matriz de proyección P1 (3×4)."""
    p1 = np.asarray(p1_raw, dtype=np.float64)
    if p1.shape != (3, 4):
        raise ValueError(
            f"{source} invalida. Se esperaba shape (3,4), recibida={p1.shape}"
        )

    fx = float(p1[0, 0])
    fy = float(p1[1, 1])
    cx = float(p1[0, 2])
    cy = float(p1[1, 2])

    if not np.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"{source}[0,0] (fx) invalido: {fx}")
    if not np.isfinite(fy) or fy <= 0.0:
        raise ValueError(f"{source}[1,1] (fy) invalido: {fy}")
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError(f"{source}[0,2]/[1,2] invalidos: cx={cx}, cy={cy}")

    k_rect = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _warn_fx_fy_mismatch(fx, fy, source)
    return k_rect, source


def load_k_rect_from_rectification_debug(path: Path) -> tuple[np.ndarray, str]:
    """
    Reconstruye la matriz de intrínsecos rectificada K_rect (3×3) desde rectification_debug.json.

    Cadena de fallback canónica:
      1. K_rect directo (3×3)
      2. P1 a nivel raíz (3×4) → extraer fx, fy, cx, cy
      3. rectify_info.P1 → mismo proceso
      4. fx_rect_px / fy_rect_px / cx_rect_px / cy_rect_px individuales

    Devuelve
    --------
    (K_rect, source_tag) donde source_tag indica qué campo se usó.
    """
    rect_dbg = read_json(path)

    # 1) K_rect directo
    raw_k_rect = rect_dbg.get("K_rect")
    if raw_k_rect is not None:
        k_rect = np.asarray(raw_k_rect, dtype=np.float64)
        if k_rect.shape != (3, 3):
            raise ValueError(
                f"K_rect invalida en {path}. "
                f"shape esperada=(3,3), recibida={k_rect.shape}"
            )
        _warn_fx_fy_mismatch(float(k_rect[0, 0]), float(k_rect[1, 1]), "K_rect")
        return k_rect, "K_rect"

    # 2) P1 a nivel raíz
    if "P1" in rect_dbg:
        return _k_rect_from_p1(rect_dbg["P1"], "P1")

    # 3) rectify_info.P1
    rectify_info = rect_dbg.get("rectify_info", {})
    if "P1" in rectify_info:
        return _k_rect_from_p1(rectify_info["P1"], "rectify_info.P1")

    # 4) Campos individuales fx/fy/cx/cy
    fx = rect_dbg.get("fx_rect_px")
    fy = rect_dbg.get("fy_rect_px", fx)
    cx = rect_dbg.get("cx_rect_px")
    cy = rect_dbg.get("cy_rect_px")

    if fx is not None and cx is not None and cy is not None:
        fx = float(fx)
        fy = float(fy)
        cx = float(cx)
        cy = float(cy)
        if not np.isfinite(fx) or fx <= 0.0:
            raise ValueError(f"fx_rect_px invalido en {path}: {fx}")
        if not np.isfinite(fy) or fy <= 0.0:
            raise ValueError(f"fy_rect_px invalido en {path}: {fy}")
        if not np.isfinite(cx) or not np.isfinite(cy):
            raise ValueError(f"cx_rect_px/cy_rect_px invalidos en {path}: {cx}, {cy}")

        k_rect = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        _warn_fx_fy_mismatch(fx, fy, "fx/fy/cx/cy_rect_px")
        return k_rect, "fx/fy/cx/cy_rect_px"

    keys = sorted(rect_dbg.keys())
    raise ValueError(
        "No se pudo obtener K_rect desde rectification_debug.json. "
        "Se esperaba K_rect, P1, rectify_info.P1 o fx/fy/cx/cy_rect_px. "
        f"Claves disponibles: {keys}"
    )


# ---------------------------------------------------------------
# Extracción centralizada de pose del controlador
# ---------------------------------------------------------------

def _find_controller_pose_dict(
    metadata: Dict[str, Any], side: str
) -> Optional[Dict[str, Any]]:
    """
    Localiza la pose del controlador dentro de metadata.

    Soporta dos formatos de metadata:
      1. Formato plano:   metadata["ControllerPoseRight"]  (o Left)
      2. Formato anidado: metadata["ControllerPose"]["Right"] (o Left)

    Devuelve Position/Rotation, o None si no existe la información en metadata.
    """
    side = side.lower().strip()
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")

    # 1) Formato plano: ControllerPoseRight / ControllerPoseLeft
    k_flat = "ControllerPoseRight" if side == "right" else "ControllerPoseLeft"
    if k_flat in metadata and isinstance(metadata[k_flat], dict):
        return metadata[k_flat]

    # 2) Formato anidado: ControllerPose.Right / ControllerPose.Left
    cp = metadata.get("ControllerPose")
    if isinstance(cp, dict):
        k_nested = "Right" if side == "right" else "Left"
        if k_nested in cp and isinstance(cp[k_nested], dict):
            return cp[k_nested]

    return None


def load_controller_pose(
    metadata: Dict[str, Any], side: str = "right"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae la pose del controlador.

    Soporta ambos formatos de metadata (plano y anidado).

    Parámetros
    ----------
    metadata : dict
        Contenido completo de metadata.json (ya parseado).
    side : str
        "right" o "left".

    Devuelve
    --------
    (center_xyz, quat_xyzw) : tuple[np.ndarray, np.ndarray]
        center_xyz — shape (3,), dtype float64
        quat_xyzw  — shape (4,), dtype float64, orden [x, y, z, w]

    Lanza
    -----
    KeyError si no se encuentra la pose o faltan campos Position/Rotation.
    """
    pose = _find_controller_pose_dict(metadata, side)
    if pose is None:
        k_flat = "ControllerPoseRight" if side.lower() == "right" else "ControllerPoseLeft"
        raise KeyError(
            f"No se encontró pose del controlador ('{k_flat}' ni "
            f"'ControllerPose.{side.capitalize()}') en metadata."
        )

    pos = pose.get("Position", {})
    rot = pose.get("Rotation", {})

    required_pos = ("x", "y", "z")
    required_rot = ("x", "y", "z", "w")

    if any(k not in pos for k in required_pos):
        raise KeyError(f"Position invalida para controlador '{side}': faltan campos {required_pos}")
    if any(k not in rot for k in required_rot):
        raise KeyError(f"Rotation invalida para controlador '{side}': faltan campos {required_rot}")

    center = np.array(
        [float(pos["x"]), float(pos["y"]), float(pos["z"])],
        dtype=np.float64,
    )
    quat = np.array(
        [float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"])],
        dtype=np.float64,
    )
    return center, quat


# -----------------------------
# Función principal
# -----------------------------

def load_and_validate_frame(session_path: Path, frame_index: int):
    """
    Carga y valida un frame concreto de una sesión.
    Esta es la función "real" que usará el pipeline.
    """
    frame_dir = get_frame_dir(session_path, frame_index)

    # 1) Cargar imágenes y metadatos del frame
    left_img, right_img = load_images_from_frame(frame_dir)
    meta = load_metadata_from_frame(frame_dir)

    # 2) Ajustar intrínsecos al tamaño real
    left_intr  = adjust_intrinsics_to_image(meta["LeftIntrinsics"],  left_img.shape,  mode="crop_y")
    right_intr = adjust_intrinsics_to_image(meta["RightIntrinsics"], right_img.shape, mode="crop_y")

    # 3) Validar intrínsecos
    validate_intrinsics(left_intr, left_img.shape)
    validate_intrinsics(right_intr, right_img.shape)

    # 4) Construir K
    K_left = build_K(left_intr)
    K_right = build_K(right_intr)

    # 5) Transformaciones cámara->mundo
    T_W_L = pose_dict_to_T44(meta["CameraPoseLeft"])
    T_W_R = pose_dict_to_T44(meta["CameraPoseRight"])

    # 6) Transformación relativa
    R_rel, t_rel, baseline_m = compute_relative_transform(T_W_L, T_W_R)

    # 7) Validación geométrica
    angle_deg = rotation_angle_deg(R_rel)
    tx, ty, tz = t_rel[0], t_rel[1], t_rel[2]

    assert angle_deg < 5.0, f"Ángulo relativo demasiado alto: {angle_deg:.2f} grados"
    assert abs(tx) > 5 * abs(ty), "La traslación no es principalmente en X (|tx| no domina a |ty|)"
    assert abs(tx) > 5 * abs(tz), "La traslación no es principalmente en X (|tx| no domina a |tz|)"

    return {
        "frame_index": frame_index,
        "frame_dir": frame_dir,
        "left_img": left_img,
        "right_img": right_img,
        "meta": meta,
        "K_left": K_left,
        "K_right": K_right,
        "T_W_L": T_W_L,
        "T_W_R": T_W_R,
        "R_rel": R_rel,
        "t_rel": t_rel,
        "baseline_m": baseline_m,
        "angle_deg": angle_deg,
        "tx_ty_tz": (tx, ty, tz)
    }


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def compute_Z_gt_camera_to_controller(
    metadata: Dict[str, Any],
    T_W_L: np.ndarray,
    controller: str = "right",
) -> float:
    """
    Calcula Z_gt = distancia euclídea (m) entre la cámara izquierda (posición en mundo, extraída de T_W_L) y el controlador (posición en mundo, extraída de metadata).

    Parámetros
    ----------
    metadata : dict
        Contenido de metadata.json ya parseado.
    T_W_L : np.ndarray, shape (4,4)
        Transformación mundo←cámara_izquierda.
    controller : str
        "right" o "left".

    Devuelve
    --------
    float — distancia euclídea en metros.
    """
    cam_pos = np.array(
        [float(T_W_L[0, 3]), float(T_W_L[1, 3]), float(T_W_L[2, 3])],
        dtype=np.float64,
    )

    ctrl = (
        metadata.get("ControllerPoseLeft", None)
        if controller.lower() == "left"
        else metadata.get("ControllerPoseRight", None)
    )
    if ctrl is None or "Position" not in ctrl:
        raise RuntimeError(
            "No se encontró ControllerPoseLeft/Right en metadata para calcular Z_gt."
        )

    p = ctrl["Position"]
    ctrl_pos = np.array(
        [float(p["x"]), float(p["y"]), float(p["z"])],
        dtype=np.float64,
    )
    return float(np.linalg.norm(ctrl_pos - cam_pos))