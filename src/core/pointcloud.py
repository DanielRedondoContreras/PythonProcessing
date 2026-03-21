# ============================================================
# File: src/core/pointcloud.py
# Generación de nubes de puntos 3D desde mapas de profundidad
#
# Proporciona:
#   - Backproyección de profundidad a coordenadas 3D de cámara
#   - Transformación de coordenadas OpenCV ↔ Unity
#   - Downsampling por voxel grid
#   - Escritura de archivos PLY ASCII
#   - Métricas de nube de puntos (bbox, estadísticas Z)
# ============================================================

import json
import numpy as np

from pathlib import Path


DEPTH_FILENAME = "depth_m.npy"
DEPTH_MASK_FILENAME = "depth_valid_mask.npy"

CAMERA_NPY_FILENAME = "pointcloud_camera_opencv.npy"
WORLD_NPY_FILENAME = "pointcloud_world_unity.npy"
CAMERA_PLY_FILENAME = "pointcloud_camera_rect.ply"
WORLD_PLY_FILENAME = "pointcloud_world_unity.ply"
METRICS_FILENAME = "pointcloud_metrics.json"


def validate_depth_and_mask(depth_m: np.ndarray, depth_valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valida y convierte depth_m y depth_valid_mask a los tipos esperados (float32, bool)."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m debe ser 2D (H, W). Recibido: shape={depth.shape}")

    mask = np.asarray(depth_valid_mask).astype(bool)
    if mask.shape != depth.shape:
        raise ValueError(
            "depth_valid_mask debe tener la misma shape que depth_m. "
            f"mask={mask.shape}, depth={depth.shape}"
        )

    return depth, mask


def _validate_k_rect(K_rect: np.ndarray) -> tuple[float, float, float, float]:
    """Valida la matriz K_rect (3×3) y extrae los parámetros fx, fy, cx, cy."""
    K = np.asarray(K_rect, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K_rect debe tener shape (3, 3). Recibido: {K.shape}")

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if not np.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"K_rect[0,0] (fx) invalido: {fx}")
    if not np.isfinite(fy) or fy <= 0.0:
        raise ValueError(f"K_rect[1,1] (fy) invalido: {fy}")
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise ValueError(f"K_rect[0,2]/K_rect[1,2] invalidos: cx={cx}, cy={cy}")

    return fx, fy, cx, cy


def _validate_transform_4x4(T_world_from_camera: np.ndarray) -> np.ndarray:
    """Valida que T_world_from_camera tenga shape (4, 4) y tipo float64."""
    T = np.asarray(T_world_from_camera, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            "T_world_from_camera debe tener shape (4, 4). "
            f"Recibido: {T.shape}"
        )
    return T


def _safe_output_path(out_dir: Path, filename: str) -> Path:
    """Construye una ruta de salida segura, verificando que no escape de out_dir."""
    root = Path(out_dir).resolve()
    path = (root / filename).resolve()

    if root not in path.parents:
        raise ValueError(
            f"Ruta de salida fuera de out_dir: {path}. "
            f"out_dir permitido: {root}"
        )

    return path


def _bounds_summary(points: np.ndarray) -> dict | None:
    if points.size == 0:
        return None

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    size = pmax - pmin

    return {
        "min_xyz_m": [float(pmin[0]), float(pmin[1]), float(pmin[2])],
        "max_xyz_m": [float(pmax[0]), float(pmax[1]), float(pmax[2])],
        "size_xyz_m": [float(size[0]), float(size[1]), float(size[2])],
    }


def _load_depth_inputs_from_out_dir(
    out_dir: Path,
    depth_filename: str = DEPTH_FILENAME,
    mask_filename: str = DEPTH_MASK_FILENAME,
) -> tuple[np.ndarray, np.ndarray]:
    depth_path = _safe_output_path(out_dir, depth_filename)
    mask_path = _safe_output_path(out_dir, mask_filename)

    if not depth_path.exists():
        raise FileNotFoundError(
            f"No existe {depth_filename} en {out_dir}. "
        )

    if not mask_path.exists():
        raise FileNotFoundError(
            f"No existe {mask_filename} en {out_dir}. "
        )

    depth_m = np.load(depth_path).astype(np.float32)
    depth_valid_mask = np.load(mask_path).astype(bool)

    return depth_m, depth_valid_mask


def build_pointcloud_camera_opencv(
    depth_m: np.ndarray,
    depth_valid_mask: np.ndarray,
    K_rect: np.ndarray,
    input_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backprojection a camara OpenCV rectificada:
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
      Z = depth

    Retorna:
      - puntos Nx3 float32 en camara OpenCV
      - mascara final usada (H, W)
    """
    depth, valid = validate_depth_and_mask(depth_m, depth_valid_mask)
    fx, fy, cx, cy = _validate_k_rect(K_rect)

    final_mask = valid & np.isfinite(depth) & (depth > 0.0)

    if input_mask is not None:
        extra = np.asarray(input_mask).astype(bool)
        if extra.shape != depth.shape:
            raise ValueError(
                "input_mask debe tener la misma shape que depth_m. "
                f"mask={extra.shape}, depth={depth.shape}"
            )
        final_mask &= extra

    v, u = np.where(final_mask)
    if v.size == 0:
        return np.zeros((0, 3), dtype=np.float32), final_mask

    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - np.float32(cx)) * z / np.float32(fx)
    y = (v.astype(np.float32) - np.float32(cy)) * z / np.float32(fy)

    points = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    return points, final_mask


def filter_points_z(points: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Filtra puntos por rango Z. Devuelve subconjunto con Z en [z_min, z_max]."""
    if points.size == 0:
        return points

    z = points[:, 2]
    m = np.isfinite(z) & (z >= float(z_min)) & (z <= float(z_max))
    return points[m]


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """
    Downsample por voxel. Si voxel_m <= 0, no aplica filtro.
    """
    if points.size == 0 or voxel_m is None or voxel_m <= 0:
        return points

    v = float(voxel_m)
    grid = np.floor(points / v).astype(np.int64)
    _, idx = np.unique(grid, axis=0, return_index=True)
    return points[np.sort(idx)]


def points_opencv_cam_to_unity_cam(points_cv: np.ndarray) -> np.ndarray:
    """
    Wrapper de compatibilidad con pipeline previo.
    """
    if points_cv.size == 0:
        return points_cv

    pts = points_cv.astype(np.float64, copy=False)
    out = pts.copy()
    out[:, 1] *= -1.0
    return out.astype(np.float32)


def transform_points(T_W_C: np.ndarray, points_C: np.ndarray) -> np.ndarray:
    """
    Aplica T_W_C (4x4) a puntos Nx3.
    """
    if points_C.size == 0:
        return points_C

    T = _validate_transform_4x4(T_W_C)
    R = T[:3, :3]
    t = T[:3, 3]
    pts = points_C.astype(np.float64, copy=False)
    out = (pts @ R.T) + t
    return out.astype(np.float32)


def write_ply_xyz(path: Path, points_xyz: np.ndarray):
    """
    Escribe un PLY ASCII minimo con XYZ.
    """
    path = Path(path)
    pts = np.asarray(points_xyz, dtype=np.float32)
    n = int(pts.shape[0])

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    ) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def pointcloud_metrics(points: np.ndarray) -> dict:
    """Calcula métricas básicas (n_points, bbox, estadísticas Z) de una nube de puntos."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return {
            "n_points": 0,
            "z_min": None,
            "z_max": None,
            "z_mean": None,
            "z_std": None,
            "bbox_min": None,
            "bbox_max": None,
            "bounds": None,
        }

    z = pts[:, 2]
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    return {
        "n_points": int(pts.shape[0]),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "bbox_min": [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])],
        "bbox_max": [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])],
        "bounds": _bounds_summary(pts),
    }