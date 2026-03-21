# ============================================================
# File: src/core/segmentation.py
# Segmentación espacial del controlador mediante geometría
# y clustering
#
# Proporciona:
#   - Recorte por OBB orientada (pose GT del controlador)
#   - Filtrado por banda Z
#   - Eliminación de planos dominantes (RANSAC)
#   - Eliminación de outliers (estadístico / radio)
#   - Clustering DBSCAN y selección del cluster óptimo
#   - Estimación de normales sobre la nube segmentada
# ============================================================

from __future__ import annotations

import numpy as np
import open3d as o3d

from src.core.transforms import quat_multiply_xyzw, quat_to_rotmat_xyzw


def crop_obb_world(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    size_xyz: np.ndarray,
) -> np.ndarray:
    """
    Recorta una OBB (Oriented Bounding Box) centrada en center_xyz, orientada por quat_xyzw y con tamano size_xyz (ancho, alto, fondo) en metros.

    points_xyz: (N,3) en world
    center_xyz: (3,) en world
    quat_xyzw: (4,) [x,y,z,w] en world
    size_xyz: (3,) dimensiones de la caja en el frame local del controller
    """
    R = quat_to_rotmat_xyzw(quat_xyzw)  # local->world
    Rt = R.T  # world->local
    half = size_xyz / 2.0

    # world -> local del controller
    p_local = (points_xyz - center_xyz[None, :]) @ Rt.T  # (N,3)

    # AABB en frame local
    mask = (
        (p_local[:, 0] >= -half[0])
        & (p_local[:, 0] <= half[0])
        & (p_local[:, 1] >= -half[1])
        & (p_local[:, 1] <= half[1])
        & (p_local[:, 2] >= -half[2])
        & (p_local[:, 2] <= half[2])
    )

    return points_xyz[mask]


def _empty_xyz() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _pcd_num_points(pcd: o3d.geometry.PointCloud) -> int:
    return int(np.asarray(pcd.points).shape[0])


def _points_to_pcd(points_xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def filter_z_band(
    points_xyz: np.ndarray, z_min: float, z_max: float, axis: int = 2
) -> np.ndarray:
    """
    Filtra puntos por una banda de profundidad sobre el eje indicado.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return _empty_xyz()
    if axis < 0 or axis > 2:
        return _empty_xyz()
    if z_min > z_max:
        z_min, z_max = z_max, z_min
    mask = (pts[:, axis] >= z_min) & (pts[:, axis] <= z_max)
    return pts[mask]


def remove_dominant_planes_o3d(
    pcd: o3d.geometry.PointCloud,
    n_planes: int,
    dist_thresh: float,
    ransac_n: int,
    num_iter: int,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """
    Elimina iterativamente los planos dominantes (suelo/pared) con RANSAC.
    """
    metrics = {
        "n_removed": 0,
        "plane_models": [],
        "n_remaining": [],
        "n_inliers": [],
        "warning": None,
    }

    if o3d is None:
        metrics["warning"] = "open3d_not_available"
        return pcd, metrics

    remaining = pcd
    if n_planes <= 0:
        metrics["warning"] = "remove_planes_disabled"
        metrics["n_remaining"].append(_pcd_num_points(remaining))
        return remaining, metrics

    if _pcd_num_points(remaining) == 0:
        metrics["warning"] = "empty_input"
        return remaining, metrics

    for _ in range(int(n_planes)):
        n_now = _pcd_num_points(remaining)
        if n_now < max(3, int(ransac_n)):
            metrics["warning"] = "not_enough_points_for_plane"
            break

        model, inliers = remaining.segment_plane(
            distance_threshold=float(dist_thresh),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iter),
        )
        inliers = np.asarray(inliers, dtype=np.int64)
        n_inliers = int(inliers.shape[0])
        if n_inliers == 0:
            metrics["warning"] = "no_plane_found"
            break

        metrics["plane_models"].append([float(x) for x in model])
        metrics["n_inliers"].append(n_inliers)
        metrics["n_removed"] += n_inliers

        remaining = remaining.select_by_index(inliers.tolist(), invert=True)
        metrics["n_remaining"].append(_pcd_num_points(remaining))

        if _pcd_num_points(remaining) == 0:
            metrics["warning"] = "all_points_removed_by_planes"
            break

    return remaining, metrics


def remove_outliers_o3d(
    pcd: o3d.geometry.PointCloud,
    method: str,
    nb_neighbors: int,
    std_ratio: float,
    radius: float,
    min_points: int,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """
    Elimina outliers por metodo estadistico, por radio o no aplica filtro.
    """
    n_before = _pcd_num_points(pcd)
    method_in = (method or "none").lower()
    metrics = {
        "method": method_in,
        "n_before": n_before,
        "n_after": n_before,
        "n_removed": 0,
        "warning": None,
    }

    if o3d is None:
        metrics["warning"] = "open3d_not_available"
        return pcd, metrics

    if n_before == 0:
        metrics["warning"] = "empty_input"
        return pcd, metrics

    filtered = pcd
    if method_in == "none":
        pass
    elif method_in == "statistical":
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
        )
        filtered = pcd.select_by_index(ind)
    elif method_in == "radius":
        _, ind = pcd.remove_radius_outlier(
            nb_points=int(min_points), radius=float(radius)
        )
        filtered = pcd.select_by_index(ind)
    else:
        metrics["warning"] = "unknown_method_fallback_none"
        filtered = pcd

    n_after = _pcd_num_points(filtered)
    metrics["n_after"] = n_after
    metrics["n_removed"] = n_before - n_after
    return filtered, metrics


def cluster_dbscan_o3d(
    pcd: o3d.geometry.PointCloud, eps: float, min_points: int
) -> tuple[list[o3d.geometry.PointCloud], dict]:
    """
    Ejecuta DBSCAN sobre la nube y devuelve los clusters como lista de pcd.
    """
    metrics = {
        "n_clusters": 0,
        "sizes": [],
        "labels_count": {},
        "warning": None,
    }

    if o3d is None:
        metrics["warning"] = "open3d_not_available"
        return [], metrics

    n_pts = _pcd_num_points(pcd)
    if n_pts == 0:
        metrics["warning"] = "empty_input"
        return [], metrics

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(eps),
            min_points=int(min_points),
            print_progress=False,
        ),
        dtype=np.int64,
    )
    if labels.shape[0] != n_pts:
        metrics["warning"] = "dbscan_labels_size_mismatch"
        return [], metrics

    uniq, counts = np.unique(labels, return_counts=True)
    metrics["labels_count"] = {int(l): int(c) for l, c in zip(uniq, counts)}

    clusters = []
    sizes = []
    for label in sorted(int(x) for x in uniq if x >= 0):
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        cluster_pcd = pcd.select_by_index(idx.tolist())
        clusters.append(cluster_pcd)
        sizes.append(int(idx.size))

    metrics["n_clusters"] = len(clusters)
    metrics["sizes"] = sizes
    if len(clusters) == 0:
        metrics["warning"] = "no_clusters_found"
    return clusters, metrics


# ==============================================================
# select_cluster con filtro de tamaño mínimo
# ==============================================================

def select_cluster(
    clusters: list[o3d.geometry.PointCloud],
    strategy: str,
    ref_center: np.ndarray | None,
    min_cluster_size: int = 0,
) -> tuple[o3d.geometry.PointCloud | None, dict]:
    """
    Selecciona un cluster por tamano o por cercania al centro de referencia.

    Fase 4a: cuando strategy='nearest_center' y min_cluster_size > 0, clusters con menos de min_cluster_size puntos antes de la selección por distancia, evitando que un cluster diminuto sea elegid sobre uno grande simplemente porque su centroide está más cerca.
    """
    metrics = {
        "strategy": strategy,
        "chosen_idx": None,
        "chosen_size": 0,
        "distances": None,
        "min_cluster_size": int(min_cluster_size),
        "n_filtered_by_size": 0,
        "warning": None,
    }

    if len(clusters) == 0:
        metrics["warning"] = "empty_clusters"
        return None, metrics

    strategy_in = (strategy or "largest").lower()
    chosen_idx = None

    if strategy_in == "largest":
        sizes = [int(np.asarray(c.points).shape[0]) for c in clusters]
        chosen_idx = int(np.argmax(sizes))
    elif strategy_in == "nearest_center":
        if ref_center is None:
            metrics["warning"] = "ref_center_none_fallback_largest"
            sizes = [int(np.asarray(c.points).shape[0]) for c in clusters]
            chosen_idx = int(np.argmax(sizes))
            strategy_in = "largest"
        else:
            rc = np.asarray(ref_center, dtype=np.float64).reshape(-1)
            if rc.shape[0] != 3:
                metrics["warning"] = "invalid_ref_center_fallback_largest"
                sizes = [int(np.asarray(c.points).shape[0]) for c in clusters]
                chosen_idx = int(np.argmax(sizes))
                strategy_in = "largest"
            else:
                # filtrar clusters por tamaño mínimo antes de seleccionar por distancia al centro de referencia.
                min_sz = int(min_cluster_size)
                eligible_indices = list(range(len(clusters)))

                if min_sz > 0:
                    eligible_indices = [
                        i for i in eligible_indices
                        if int(np.asarray(clusters[i].points).shape[0]) >= min_sz
                    ]
                    n_filtered = len(clusters) - len(eligible_indices)
                    metrics["n_filtered_by_size"] = n_filtered

                    if len(eligible_indices) == 0:
                        # Todos los clusters son demasiado pequeños.
                        # Fallback: elegir el mayor de todos.
                        metrics["warning"] = "all_clusters_below_min_size_fallback_largest"
                        sizes = [int(np.asarray(c.points).shape[0]) for c in clusters]
                        chosen_idx = int(np.argmax(sizes))
                        strategy_in = "largest"

                # Si eligible_indices no está vacío y no hemos hecho fallback
                if chosen_idx is None:
                    distances = []
                    for i in eligible_indices:
                        aabb = clusters[i].get_axis_aligned_bounding_box()
                        cc = np.asarray(aabb.get_center(), dtype=np.float64)
                        distances.append(float(np.linalg.norm(cc - rc)))

                    best_eligible = int(np.argmin(distances))
                    chosen_idx = eligible_indices[best_eligible]

                    # Reportar distancias de TODOS los clusters (para diagnóstico)
                    all_distances = []
                    for c in clusters:
                        aabb = c.get_axis_aligned_bounding_box()
                        cc = np.asarray(aabb.get_center(), dtype=np.float64)
                        all_distances.append(float(np.linalg.norm(cc - rc)))
                    metrics["distances"] = all_distances
    else:
        metrics["warning"] = "unknown_strategy_fallback_largest"
        sizes = [int(np.asarray(c.points).shape[0]) for c in clusters]
        chosen_idx = int(np.argmax(sizes))
        strategy_in = "largest"

    chosen = clusters[chosen_idx]
    metrics["strategy"] = strategy_in
    metrics["chosen_idx"] = chosen_idx
    metrics["chosen_size"] = int(np.asarray(chosen.points).shape[0])
    return chosen, metrics


def estimate_normals_for_points(
    points_xyz: np.ndarray,
    normal_radius: float,
    normal_max_nn: int,
    orient_k: int = 50,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Estima normales para una nube XYZ y devuelve:
      - pcd con normales
      - matriz Nx6 float32: [x y z nx ny nz]
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    pcd = _points_to_pcd(pts)
    if _pcd_num_points(pcd) == 0:
        return pcd, np.empty((0, 6), dtype=np.float32)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=int(normal_max_nn),
        )
    )

    k = int(orient_k) if int(orient_k) > 0 else int(normal_max_nn)
    pcd.orient_normals_consistent_tangent_plane(k=k)

    xyz = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    nx6 = np.concatenate([xyz, nrm], axis=1).astype(np.float32)
    return pcd, nx6


def presegment_and_cluster(
    points_xyz: np.ndarray,
    voxel_m: float,
    z_min: float,
    z_max: float,
    remove_planes: int,
    plane_dist_thresh: float,
    outlier_method: str,
    outlier_nb_neighbors: int,
    outlier_std_ratio: float,
    outlier_radius: float,
    outlier_min_points: int,
    dbscan_eps: float,
    dbscan_min_points: int,
    select_strategy: str,
    ref_center: np.ndarray | None,
    min_cluster_size: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Pipeline de presegmentacion geometrica + clustering robusto.
    Orden: z-band -> voxel -> remove planes -> outliers -> DBSCAN -> select cluster.

    parámetro min_cluster_size propagado a select_cluster.
    """
    plane_ransac_n = 3
    plane_num_iter = 1000


    metrics = {
        "n_in": 0,
        "n_after_z": 0,
        "n_after_voxel": 0,
        "n_after_planes": 0,
        "n_after_outliers": 0,
        "planes": {},
        "outliers": {},
        "dbscan": {
            "n_clusters": 0,
            "sizes": [],
            "labels_count": {},
            "chosen": {},
        },
        "params": {
            "voxel_m": float(voxel_m),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "remove_planes": int(remove_planes),
            "plane_dist_thresh": float(plane_dist_thresh),
            "plane_ransac_n": int(plane_ransac_n),
            "plane_num_iter": int(plane_num_iter),
            "outlier_method": str(outlier_method),
            "outlier_nb_neighbors": int(outlier_nb_neighbors),
            "outlier_std_ratio": float(outlier_std_ratio),
            "outlier_radius": float(outlier_radius),
            "outlier_min_points": int(outlier_min_points),
            "dbscan_eps": float(dbscan_eps),
            "dbscan_min_points": int(dbscan_min_points),
            "select_strategy": str(select_strategy),
            "min_cluster_size": int(min_cluster_size),
            "ref_center": None
            if ref_center is None
            else np.asarray(ref_center, dtype=np.float64).reshape(-1).tolist(),
        },
        "warning": None,
    }

    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        metrics["warning"] = "invalid_points_shape"
        return _empty_xyz(), metrics

    metrics["n_in"] = int(pts.shape[0])
    if metrics["n_in"] == 0:
        metrics["warning"] = "empty_input_points"
        return _empty_xyz(), metrics

    if o3d is None:
        metrics["warning"] = "open3d_not_available"
        return _empty_xyz(), metrics

    # 1) Filtro por banda de profundidad
    pts_z = filter_z_band(pts, z_min=z_min, z_max=z_max, axis=2)
    metrics["n_after_z"] = int(pts_z.shape[0])
    if metrics["n_after_z"] == 0:
        metrics["warning"] = "empty_after_z_band"
        return _empty_xyz(), metrics

    # 2) Voxel downsample
    pcd = _points_to_pcd(pts_z)
    if float(voxel_m) > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_m))
    metrics["n_after_voxel"] = _pcd_num_points(pcd)
    if metrics["n_after_voxel"] == 0:
        metrics["warning"] = "empty_after_voxel"
        return _empty_xyz(), metrics

    # 3) Eliminacion de planos dominantes
    pcd, plane_metrics = remove_dominant_planes_o3d(
        pcd=pcd,
        n_planes=int(remove_planes),
        dist_thresh=float(plane_dist_thresh),
        ransac_n=int(plane_ransac_n),
        num_iter=int(plane_num_iter),
    )
    metrics["planes"] = plane_metrics
    metrics["n_after_planes"] = _pcd_num_points(pcd)
    if metrics["n_after_planes"] == 0:
        metrics["warning"] = "empty_after_plane_removal"
        return _empty_xyz(), metrics

    # 4) Eliminacion de outliers
    pcd, outlier_metrics = remove_outliers_o3d(
        pcd=pcd,
        method=outlier_method,
        nb_neighbors=outlier_nb_neighbors,
        std_ratio=outlier_std_ratio,
        radius=outlier_radius,
        min_points=outlier_min_points,
    )
    metrics["outliers"] = outlier_metrics
    metrics["n_after_outliers"] = _pcd_num_points(pcd)
    if metrics["n_after_outliers"] == 0:
        metrics["warning"] = "empty_after_outlier_removal"
        return _empty_xyz(), metrics

    # 5) Clustering DBSCAN y seleccion de cluster final
    clusters, db_metrics = cluster_dbscan_o3d(
        pcd=pcd, eps=float(dbscan_eps), min_points=int(dbscan_min_points)
    )
    chosen, chosen_metrics = select_cluster(
        clusters=clusters,
        strategy=select_strategy,
        ref_center=ref_center,
        min_cluster_size=int(min_cluster_size),
    )
    db_metrics["chosen"] = chosen_metrics
    metrics["dbscan"] = db_metrics

    if chosen is None:
        metrics["warning"] = "no_cluster_selected"
        return _empty_xyz(), metrics

    pts_selected = np.asarray(chosen.points, dtype=np.float64)
    if pts_selected.ndim != 2 or pts_selected.shape[1] != 3:
        metrics["warning"] = "invalid_selected_cluster_shape"
        return _empty_xyz(), metrics

    return pts_selected, metrics


# ==============================================================
# ==============================================================
# ==============================================================

def choose_best_pose_for_roi(
    pts_world: np.ndarray,
    center: np.ndarray,
    quat: np.ndarray,
    roi_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, int, np.ndarray]:
    """
    Selecciona la hipótesis de pose (centro + quaternion) que maximiza el número de puntos capturados dentro de la OBB.

    Evalúa 4 candidatos: identidad, yaw 180° en Y, flip-Z y flip-X.

    Parámetros
    ----------
    pts_world : (N, 3) — nube de puntos en coordenadas mundo.
    center    : (3,)   — centro del controlador (mundo).
    quat      : (4,)   — quaternion [x, y, z, w] del controlador.
    roi_size  : (3,)   — dimensiones de la OBB (m).

    Devuelve
    --------
    (center_best, quat_best, tag_best, n_best, pts_best_xyz)
    """
    center = np.array(center, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)
    roi_size = np.array(roi_size, dtype=np.float64)

    quat_norm = np.linalg.norm(quat)
    if quat_norm > 0.0:
        quat = quat / quat_norm

    x, y, z = center.tolist()
    q_yaw180 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # Solo candidatos geométricamente consistentes:
    #   - "id": pose GT directa
    #   - "yaw180Y": rotación 180° en Y aplicada tanto a posición como quaternion
    # Los antiguos "flipZ" y "flipX" invertían posición sin ajustar quaternion,
    # creando OBBs geométricamente incoherentes.
    candidates = [
        ("id", center, quat),
        ("yaw180Y", np.array([-x, y, -z], dtype=np.float64), quat_multiply_xyzw(q_yaw180, quat)),
    ]

    center_best = center
    quat_best = quat
    tag_best = "id"
    n_best = -1
    pts_best_xyz = np.empty((0, 3), dtype=np.float64)

    for tag_i, center_i, quat_i in candidates:
        pts_roi_i = crop_obb_world(pts_world, center_i, quat_i, roi_size)
        pts_xyz_i = _to_xyz(pts_roi_i)
        n_i = int(pts_xyz_i.shape[0])

        if n_i > n_best:
            n_best = n_i
            center_best = center_i
            quat_best = quat_i
            tag_best = tag_i
            pts_best_xyz = pts_xyz_i

    return center_best, quat_best, tag_best, int(n_best), pts_best_xyz


def _to_xyz(points_xyz: np.ndarray | None) -> np.ndarray:
    """Normaliza un array de puntos a shape (N, 3) float64; devuelve vacío si es inválido."""
    if points_xyz is None:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float64)
    return pts