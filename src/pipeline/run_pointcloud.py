import argparse
import json
from pathlib import Path

import numpy as np

from src.core.data_loader import load_k_rect_from_rectification_debug
from src.core.io_utils import require_file
from src.core.transforms import pose_dict_to_T44
from src.core.pointcloud import (
    CAMERA_NPY_FILENAME,
    CAMERA_PLY_FILENAME,
    DEPTH_FILENAME,
    DEPTH_MASK_FILENAME,
    METRICS_FILENAME,
    WORLD_NPY_FILENAME,
    WORLD_PLY_FILENAME,
    build_pointcloud_camera_opencv,
    pointcloud_metrics,
    points_opencv_cam_to_unity_cam,
    transform_points,
    validate_depth_and_mask,
    voxel_downsample,
    write_ply_xyz,
)


def _load_metadata_for_frame(session: str, frame_idx: int, out_dir: Path) -> tuple[dict, Path]:
    frame_name = f"frame_{frame_idx:06d}"
    candidates = [
        out_dir / "metadata.json",
        out_dir / "frame_metadata.json",
        Path("data") / "raw" / session / frame_name / "metadata.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path

    expected = "\n - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "No se encontro metadata para calcular T_world_from_camera. "
        "Se intentaron estas rutas:\n - " + expected
    )


def _extract_t_world_from_camera(metadata: dict, metadata_path: Path) -> tuple[np.ndarray, str]:
    if "T_world_from_camera" in metadata:
        t_mat = np.asarray(metadata["T_world_from_camera"], dtype=np.float64)
        source = "metadata.T_world_from_camera"
    elif "T_W_L" in metadata:
        t_mat = np.asarray(metadata["T_W_L"], dtype=np.float64)
        source = "metadata.T_W_L"
    elif "CameraPoseLeft" in metadata:
        t_mat = pose_dict_to_T44(metadata["CameraPoseLeft"]).astype(np.float64)
        source = "metadata.CameraPoseLeft"
    else:
        raise ValueError(
            f"No se encontro T_world_from_camera en {metadata_path}. "
            "Se esperaba T_world_from_camera, T_W_L o CameraPoseLeft."
        )

    if t_mat.shape != (4, 4):
        raise ValueError(
            f"T_world_from_camera invalida en {metadata_path}. "
            f"shape esperada=(4,4), recibida={t_mat.shape}"
        )

    return t_mat, source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera nube de puntos desde depth_m.npy + depth_valid_mask.npy usando K_rect y T_world_from_camera del frame."
    )
    parser.add_argument("--session", required=True, help="Nombre de sesion dentro de data/raw")
    parser.add_argument("--frame", type=int, default=0, help="Indice de frame (0..)")
    parser.add_argument("--z-min", type=float, default=0.3, help="Z minima (m)")
    parser.add_argument("--z-max", type=float, default=5.0, help="Z maxima (m)")
    parser.add_argument("--voxel", type=float, default=0.005, help="Voxel (m). 0 => sin downsample")
    args = parser.parse_args()

    out_dir = Path("data") / "processed" / args.session / f"frame_{args.frame:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_path = require_file(out_dir / DEPTH_FILENAME, DEPTH_FILENAME)
    mask_path = require_file(out_dir / DEPTH_MASK_FILENAME, DEPTH_MASK_FILENAME)
    rectification_path = require_file(out_dir / "rectification_debug.json", "rectification_debug.json")

    depth_m, depth_valid_mask = validate_depth_and_mask(
        np.load(depth_path).astype(np.float32, copy=False),
        np.load(mask_path),
    )

    k_rect, k_rect_source = load_k_rect_from_rectification_debug(rectification_path)

    metadata, metadata_path = _load_metadata_for_frame(args.session, args.frame, out_dir)
    t_world_from_camera, t_source = _extract_t_world_from_camera(metadata, metadata_path)

    z_mask = np.isfinite(depth_m) & (depth_m >= float(args.z_min)) & (depth_m <= float(args.z_max))

    points_camera_opencv, final_mask = build_pointcloud_camera_opencv(
        depth_m=depth_m,
        depth_valid_mask=depth_valid_mask,
        K_rect=k_rect,
        input_mask=z_mask,
    )

    points_camera_opencv = voxel_downsample(points_camera_opencv, float(args.voxel))

    points_camera_unity = points_opencv_cam_to_unity_cam(points_camera_opencv)
    points_world_unity = transform_points(t_world_from_camera, points_camera_unity)

    camera_npy = out_dir / CAMERA_NPY_FILENAME
    world_npy = out_dir / WORLD_NPY_FILENAME
    camera_ply = out_dir / CAMERA_PLY_FILENAME
    world_ply = out_dir / WORLD_PLY_FILENAME
    metrics_path = out_dir / METRICS_FILENAME

    np.save(camera_npy, points_camera_opencv.astype(np.float32, copy=False))
    np.save(world_npy, points_world_unity.astype(np.float32, copy=False))
    write_ply_xyz(camera_ply, points_camera_opencv)
    write_ply_xyz(world_ply, points_world_unity)

    metrics = {
        "z_min_m": float(args.z_min),
        "z_max_m": float(args.z_max),
        "voxel_m": float(args.voxel),
        "depth_shape": [int(depth_m.shape[0]), int(depth_m.shape[1])],
        "n_pixels_total": int(depth_m.size),
        "n_valid_depth_mask": int(np.count_nonzero(depth_valid_mask)),
        "n_valid_after_masks": int(np.count_nonzero(final_mask)),
        "k_rect_source": k_rect_source,
        "metadata_source": str(metadata_path),
        "t_world_from_camera_source": t_source,
        "K_rect": np.asarray(k_rect, dtype=np.float64).tolist(),
        "camera_opencv": pointcloud_metrics(points_camera_opencv),
        "world_unity": pointcloud_metrics(points_world_unity),
        "outputs": {
            "pointcloud_camera_opencv_npy": camera_npy.name,
            "pointcloud_world_unity_npy": world_npy.name,
            "pointcloud_camera_rect_ply": camera_ply.name,
            "pointcloud_world_unity_ply": world_ply.name,
            "pointcloud_metrics_json": metrics_path.name,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[POINTCLOUD INPUT]", depth_path, "shape=", depth_m.shape)
    print("[POINTCLOUD INPUT]", rectification_path, "K_rect_source=", k_rect_source)
    print("[POINTCLOUD INPUT]", metadata_path, "T_source=", t_source)
    print("[OK] Guardado:", out_dir)
    print(
        "[POINTCLOUD]",
        {
            "n_points_camera": int(points_camera_opencv.shape[0]),
            "n_points_world": int(points_world_unity.shape[0]),
            "camera_npy": camera_npy.name,
            "world_npy": world_npy.name,
            "camera_ply": camera_ply.name,
            "world_ply": world_ply.name,
            "metrics": metrics_path.name,
        },
    )


if __name__ == "__main__":
    main()