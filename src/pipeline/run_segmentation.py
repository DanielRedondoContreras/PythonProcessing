import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from src.core.data_loader import load_controller_pose
from src.core.io_utils import read_json
from src.core.segmentation import (
    _to_xyz,
    choose_best_pose_for_roi,
    estimate_normals_for_points,
    presegment_and_cluster,
)


def _write_ply_xyz(path: Path, points_xyz: np.ndarray | None) -> None:
    pts = _to_xyz(points_xyz)
    print(f"[WRITE_PLY] path={path} pts_shape={pts.shape} pts_dtype={pts.dtype}")
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"[WRITE_PLY] path={path} len(pcd.points)={len(pcd.points)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    print(f"[WRITE_PLY] path={path} write_ok={ok}")

def _save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _build_cli_params(args: argparse.Namespace) -> dict:
    return {
        "in_ply": str(args.in_ply),
        "metadata": str(args.metadata),
        "out_ply": str(args.out_ply),
        "side": str(args.side),
        "use_roi_gt": bool(args.use_roi_gt),
        "roi_sx": float(args.roi_sx),
        "roi_sy": float(args.roi_sy),
        "roi_sz": float(args.roi_sz),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "voxel": float(args.voxel),
        "remove_planes": int(args.remove_planes),
        "plane_dist": float(args.plane_dist),
        "outliers": str(args.outliers),
        "nb_neighbors": int(args.nb_neighbors),
        "std_ratio": float(args.std_ratio),
        "radius": float(args.radius),
        "min_points": int(args.min_points),
        "dbscan_eps": float(args.dbscan_eps),
        "dbscan_minpts": int(args.dbscan_minpts),
        "select": str(args.select),
        "min_cluster_size": int(args.min_cluster_size),
    }


def _write_failure(
    out_ply: Path,
    metrics_path: Path,
    base_metrics: dict,
    reason: str,
    core_metrics: dict | None = None,
) -> None:
    _write_ply_xyz(out_ply, np.empty((0, 3), dtype=np.float64))

    out = dict(base_metrics)
    out["ok"] = False
    out["reason"] = reason
    out["n_out"] = 0
    if core_metrics is not None:
        out["core_metrics"] = core_metrics
    _save_metrics(metrics_path, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segmentacion de objeto por ROI opcional + presegmentacion geometrica + clustering."
    )
    ap.add_argument("--in-ply", required=True, help="Ruta a pointcloud_world_unity.ply")
    ap.add_argument("--metadata", required=True, help="Ruta a metadata.json del frame")
    ap.add_argument("--out-ply", required=True, help="Salida object_segmented.ply")
    ap.add_argument("--side", default="right", choices=["left", "right"], help="ControllerPoseRight/Left")

    ap.add_argument(
        "--use-roi-gt",
        action="store_true",
        help="Si se activa, recorta por OBB usando pose del controller",
    )
    # Defaults alineados con YAML corregido (0.25 → 0.30)
    ap.add_argument("--roi-sx", "--sx", dest="roi_sx", type=float, default=0.30, help="Tamano OBB X (m)")
    ap.add_argument("--roi-sy", "--sy", dest="roi_sy", type=float, default=0.30, help="Tamano OBB Y (m)")
    ap.add_argument("--roi-sz", "--sz", dest="roi_sz", type=float, default=0.30, help="Tamano OBB Z (m)")

    # Defaults alineados con YAML corregido (z_min 0.0 → 0.15, z_max 5.0 → 1.5)
    ap.add_argument("--z-min", type=float, default=0.15, help="Banda Z minima (m)")
    ap.add_argument("--z-max", type=float, default=2.0, help="Banda Z maxima (m)")
    # Voxel alineado con YAML corregido (0.005 → 0.0015)
    ap.add_argument("--voxel", type=float, default=0.0015, help="Voxel downsample (m)")
    ap.add_argument("--remove-planes", type=int, default=0, help="Numero de planos dominantes a eliminar")
    ap.add_argument("--plane-dist", type=float, default=0.015, help="Umbral de distancia para RANSAC de plano")

    ap.add_argument("--outliers", choices=["none", "statistical", "radius"], default="statistical")
    ap.add_argument("--nb-neighbors", type=int, default=30, help="Solo outliers=statistical")
    ap.add_argument("--std-ratio", type=float, default=2.0, help="Solo outliers=statistical")
    ap.add_argument("--radius", type=float, default=0.02, help="Solo outliers=radius")
    ap.add_argument("--min-points", type=int, default=10, help="Solo outliers=radius")

    ap.add_argument("--dbscan-eps", type=float, default=0.03)
    ap.add_argument("--dbscan-minpts", type=int, default=50)
    ap.add_argument("--select", choices=["largest", "nearest_center"], default="nearest_center")

    # Filtro de tamaño mínimo para select_cluster
    ap.add_argument("--min-cluster-size", type=int, default=200,
                    help="Tamaño mínimo (puntos) de cluster elegible en nearest_center (Fase 4a)")

    ap.add_argument("--estimate-normals", action="store_true",
                    help="Calcular normales y guardar PLY con normales + Nx6 (.npy)")
    ap.add_argument("--normal-radius", type=float, default=0.02,
                    help="Radio (m) para estimate_normals (ej 0.01-0.03)")
    ap.add_argument("--normal-max-nn", type=int, default=50,
                    help="Max vecinos para estimate_normals")
    ap.add_argument("--normal-orient-k", type=int, default=50,
                    help="k para orient_normals_consistent_tangent_plane")
    
    args = ap.parse_args()

    in_ply = Path(args.in_ply)
    metadata_path = Path(args.metadata)
    out_ply = Path(args.out_ply)
    out_dir = out_ply.parent
    metrics_path = out_dir / "segmentation_metrics.json"
    roi_raw_path = out_dir / "object_roi_raw.ply"
    preseg_clean_path = out_dir / "object_preseg_clean.ply"
    
    seg_normals_ply = out_dir / "object_segmented_normals.ply"
    seg_nx6_npy = out_dir / "object_segmented_nx6.npy"

    base_metrics = {
        "ok": False,
        "reason": None,
        "cli": _build_cli_params(args),
        "inputs": {
            "in_ply": str(in_ply),
            "metadata": str(metadata_path),
        },
        "outputs": {
            "object_segmented_ply": str(out_ply),
            "segmentation_metrics_json": str(metrics_path),
            "object_roi_raw_ply": str(roi_raw_path),
            "object_preseg_clean_ply": str(preseg_clean_path),
        },
    }

    if not in_ply.exists():
        _write_failure(out_ply, metrics_path, base_metrics, reason=f"missing_input_ply:{in_ply}")
        return
    if not metadata_path.exists():
        _write_failure(out_ply, metrics_path, base_metrics, reason=f"missing_metadata:{metadata_path}")
        return

    pcd = o3d.io.read_point_cloud(str(in_ply))
    pts_full = _to_xyz(np.asarray(pcd.points, dtype=np.float64))
    if pts_full.shape[0] == 0:
        _write_failure(out_ply, metrics_path, base_metrics, reason="empty_input_cloud")
        return

    # Usa load_controller_pose centralizado (recibe dict)
    try:
        meta_dict = read_json(metadata_path)
        center, quat = load_controller_pose(meta_dict, side=args.side)
    except Exception as exc:
        _write_failure(out_ply, metrics_path, base_metrics, reason=f"metadata_pose_error:{exc}")
        return

    pts_input = pts_full
    roi_size = np.array([args.roi_sx, args.roi_sy, args.roi_sz], dtype=np.float64)
    n_after_roi = int(pts_full.shape[0])

    if args.use_roi_gt:
        center, quat, tag, n_best, pts_input = choose_best_pose_for_roi(pts_full, center, quat, roi_size)
        n_after_roi = int(pts_input.shape[0])

        print("[ROI-GT AUTO] chosen:", tag, "n_roi:", n_best)
        print("[ROI-GT AUTO] center_xyz:", center.tolist())
        print("[ROI-GT AUTO] quat_xyzw:", quat.tolist())
        if n_best < 20:
            print("[ROI-GT AUTO] WARNING: all hypotheses produced very few points")
        print("[ROI-GT AUTO] n_after_roi:", n_after_roi)
        print("[ROI-GT AUTO] pts_input_shape:", pts_input.shape, "dtype:", pts_input.dtype)

        _write_ply_xyz(roi_raw_path, pts_input)
        if n_after_roi == 0:
            fail = dict(base_metrics)
            fail["pose"] = {
                "side": args.side,
                "center_xyz": center.tolist(),
                "quat_xyzw": quat.tolist(),
            }
            fail["roi"] = {
                "used": True,
                "size_xyz_m": roi_size.tolist(),
                "n_in": int(pts_full.shape[0]),
                "n_out": 0,
            }
            _write_failure(out_ply, metrics_path, fail, reason="empty_after_roi")
            return

    ref_center = center if args.select == "nearest_center" else None
    pts_selected, core_metrics = presegment_and_cluster(
        points_xyz=pts_input,
        voxel_m=float(args.voxel),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        remove_planes=int(args.remove_planes),
        plane_dist_thresh=float(args.plane_dist),
        outlier_method=str(args.outliers),
        outlier_nb_neighbors=int(args.nb_neighbors),
        outlier_std_ratio=float(args.std_ratio),
        outlier_radius=float(args.radius),
        outlier_min_points=int(args.min_points),
        dbscan_eps=float(args.dbscan_eps),
        dbscan_min_points=int(args.dbscan_minpts),
        select_strategy=str(args.select),
        ref_center=ref_center,
        min_cluster_size=int(args.min_cluster_size),
    )
    pts_selected = _to_xyz(pts_selected)

    _write_ply_xyz(preseg_clean_path, pts_selected)

    out_metrics = dict(base_metrics)
    out_metrics["pose"] = {
        "side": args.side,
        "center_xyz": center.tolist(),
        "quat_xyzw": quat.tolist(),
    }
    out_metrics["roi"] = {
        "used": bool(args.use_roi_gt),
        "size_xyz_m": roi_size.tolist(),
        "n_in": int(pts_full.shape[0]),
        "n_out": int(n_after_roi),
    }
    out_metrics["n_in"] = int(pts_input.shape[0])
    out_metrics["n_out"] = int(pts_selected.shape[0])
    out_metrics["core_metrics"] = core_metrics

    if pts_selected.shape[0] == 0:
        reason = core_metrics.get("warning") or "empty_after_presegment_and_cluster"
        _write_failure(out_ply, metrics_path, out_metrics, reason=reason, core_metrics=core_metrics)
        return

    _write_ply_xyz(out_ply, pts_selected)
    
    if args.estimate_normals:
        pcd_n, nx6 = estimate_normals_for_points(
            pts_selected,
            normal_radius=float(args.normal_radius),
            normal_max_nn=int(args.normal_max_nn),
            orient_k=int(args.normal_orient_k),
        )

        o3d.io.write_point_cloud(str(seg_normals_ply), pcd_n, write_ascii=False)
        np.save(seg_nx6_npy, nx6)

        out_metrics["outputs"]["object_segmented_normals_ply"] = str(seg_normals_ply)
        out_metrics["outputs"]["object_segmented_nx6_npy"] = str(seg_nx6_npy)

        out_metrics["normals"] = {
            "enabled": True,
            "radius_m": float(args.normal_radius),
            "max_nn": int(args.normal_max_nn),
            "orient_k": int(args.normal_orient_k),
            "n_points": int(nx6.shape[0]),
        }
    
    out_metrics["ok"] = True
    out_metrics["reason"] = None
    _save_metrics(metrics_path, out_metrics)

    print(
        "[SEGMENTATION]",
        {
            "ok": True,
            "in_total": int(pts_full.shape[0]),
            "in_pipeline": int(pts_input.shape[0]),
            "out_segmented": int(pts_selected.shape[0]),
            "use_roi_gt": bool(args.use_roi_gt),
            "min_cluster_size": int(args.min_cluster_size),
            "out_ply": str(out_ply),
            "metrics": str(metrics_path),
        },
    )


if __name__ == "__main__":
    main()