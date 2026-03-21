# ============================================================
# File: src/core/cad_model.py
# Preparación y validación del modelo CAD del controlador
#
# Proporciona:
#   - Carga de CAD desde PLY/OBJ/STL como nube de puntos
#   - Preprocesado: voxel downsample + estimación de normales
#   - Métricas básicas (extensión, escala, nº de puntos)
#   - Validación de escala para el controlador Touch Plus
# ============================================================

from __future__ import annotations

import numpy as np
import open3d as o3d

from dataclasses import dataclass
from pathlib import Path



@dataclass
class CadPrepConfig:
    voxel_size: float = 0.005
    sample_points: int = 20000
    # Validación de tamaño aproximado del controlador Touch Plus
    min_extent_m: float = 0.05
    max_extent_m: float = 0.35
    # Normales (para ICP point-to-plane)
    estimate_normals: bool = True
    normal_radius_factor: float = 2.5  
    normal_max_nn: int = 30


def _try_read_mesh(path: Path) -> o3d.geometry.TriangleMesh | None:
    """Intenta leer un archivo como malla triangular. Devuelve None si falla o está vacío."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is None:
        return None
    if len(mesh.vertices) == 0:
        return None
    return mesh


def _try_read_pointcloud(path: Path) -> o3d.geometry.PointCloud | None:
    """Intenta leer un archivo como nube de puntos. Devuelve None si falla o está vacío."""
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd is None:
        return None
    if len(pcd.points) == 0:
        return None
    return pcd


def load_cad_as_pointcloud(path: str | Path, cfg: CadPrepConfig) -> o3d.geometry.PointCloud:
    """
    Carga un CAD desde .ply/.obj/.stl/.fbx y devuelve un PointCloud.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    # Intento 1: leer como nube
    pcd = _try_read_pointcloud(path)
    if pcd is not None:
        return pcd

    # Intento 2: leer como malla
    mesh = _try_read_mesh(path)
    if mesh is None:
        raise ValueError(f"No se pudo leer como nube ni como malla: {path}")

    # Limpieza mínima de malla
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    # Sample uniforme (suficiente para preparar ICP)
    pcd = mesh.sample_points_uniformly(number_of_points=cfg.sample_points)
    return pcd


def preprocess_cad_pointcloud(pcd: o3d.geometry.PointCloud, cfg: CadPrepConfig) -> o3d.geometry.PointCloud:
    """
    Downsample + normales.
    """
    pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)

    # Quita NaNs si los hubiese
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("La nube CAD está vacía tras el voxel_down_sample.")
    mask = np.isfinite(pts).all(axis=1)
    if not mask.all():
        pcd = pcd.select_by_index(np.where(mask)[0])

    if cfg.estimate_normals:
        radius = cfg.normal_radius_factor * cfg.voxel_size
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=cfg.normal_max_nn)
        )
        # Orienta normales 
        pcd.orient_normals_consistent_tangent_plane(k=cfg.normal_max_nn)

    return pcd


def cad_basic_metrics(pcd: o3d.geometry.PointCloud) -> dict:
    """
    Métricas mínimas para validar escala y salud del modelo.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(aabb.get_extent())  # (dx,dy,dz) en metros
    center = np.asarray(aabb.get_center())
    n = len(pcd.points)
    return {
        "n_points": int(n),
        "aabb_extent_m": extent.tolist(),
        "aabb_center_m": center.tolist(),
        "extent_max_m": float(extent.max()),
        "extent_min_m": float(extent.min()),
    }


def validate_cad_scale(metrics: dict, cfg: CadPrepConfig) -> None:
    """
    Validación básica: el tamaño del objeto no puede ser absurdo.
    """
    emax = metrics["extent_max_m"]
    if not (cfg.min_extent_m <= emax <= cfg.max_extent_m):
        raise ValueError(
            "Escala CAD sospechosa.\n"
            f"- extent_max_m={emax:.4f} m\n"
            f"- esperado aprox entre [{cfg.min_extent_m}, {cfg.max_extent_m}] m\n"
        )