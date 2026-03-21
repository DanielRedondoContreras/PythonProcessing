# ============================================================
# File: src/pipeline/run_ppf_match.py
# Fase 6 — Matching PPF + ICP (OpenCV Surface Matching)
#
# Ejecuta PPFMatcher sobre un frame concreto:
#   - Carga nube segmentada + modelo CAD
#   - Pre-filtra escena (crop, view clamp, densidad)
#   - PPF matching + ICP refinement
#   - Persiste pose_best.json, aligned_model.ply, match_meta.json
# ============================================================

import argparse
import json
import open3d as o3d
import sys
# Necesario para la importación de numpy en el scope de aligned_model
import numpy as np


from pathlib import Path

from src.core.ppf_match import PPFMatcher
from src.core.data_loader import load_controller_pose
from src.core.io_utils import read_json


def build_paths(session: str, frame: int, side: str):
    frame_str = f"frame_{frame:06d}"
    base = Path("data") / "processed" / session / frame_str
    return {
        "base": base,
        "scene_segmented": base / "object_segmented.ply",
        "scene_full": base / "pointcloud_world_unity.ply",
        "meta_processed": base / "metadata.json",
        "meta_raw": Path("data") / "raw" / session / frame_str / "metadata.json",
        "model_ready": Path("data")
        / "cad"
        / ("right_controller_ready.ply" if side == "right" else "left_controller_ready.ply"),
        "seg_metrics": base / "segmentation_metrics.json",
        "output_dir": base / "ppf_match",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fase 6 — Matching PPF + ICP (OpenCV Surface Matching)"
    )

    parser.add_argument("--session", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--side", type=str, default="right", choices=["right", "left"])

    # --- PPF detector ---
    parser.add_argument("--ppf-rel-sampling", type=float, default=0.05,
                        help="PPF: relativeSamplingStep (0.03–0.08)")
    parser.add_argument("--ppf-rel-distance", type=float, default=0.05,
                        help="PPF: relativeDistanceStep (0.03–0.08)")

    # --- PPF matching ---
    parser.add_argument("--ppf-scene-sample-step", type=float, default=0.025,
                        help="Fracción de puntos de escena usados (1/40 = 0.025)")
    parser.add_argument("--ppf-scene-distance", type=float, default=0.05,
                        help="Distancia relativa para submuestreo de escena")
    parser.add_argument("--ppf-top-n", type=int, default=5,
                        help="Top-N hipótesis PPF a refinar con ICP")

    # --- ICP refinement ---
    parser.add_argument("--icp-iterations", type=int, default=100)
    parser.add_argument("--icp-tolerance", type=float, default=0.005)
    parser.add_argument("--icp-rejection-scale", type=float, default=2.5)
    parser.add_argument("--icp-num-levels", type=int, default=4)

    # --- Scene pre-filtering ---
    parser.add_argument("--crop-radius", type=float, default=0.14)
    parser.add_argument("--crop-min-points", type=int, default=300)
    parser.add_argument("--view-clamp", action="store_true")
    parser.add_argument("--view-near", type=float, default=0.10)
    parser.add_argument("--view-far", type=float, default=0.60)
    parser.add_argument("--use-left-camera", action="store_true")
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--density-radius", type=float, default=0.02)
    parser.add_argument("--density-min-p", type=float, default=5.0)
    parser.add_argument("--density-max-p", type=float, default=95.0)

    # --- Diagnóstico ---
    parser.add_argument("--save-subclouds", action="store_true")

    args = parser.parse_args()

    paths = build_paths(args.session, args.frame, args.side)

    # --- Selección de nube de escena ---
    used_full_scene_fallback = False
    if paths["scene_segmented"].exists():
        scene_path = paths["scene_segmented"]
        print(f"[INFO] Usando nube segmentada: {scene_path}")
    elif paths["scene_full"].exists():
        scene_path = paths["scene_full"]
        used_full_scene_fallback = True
        print("[INFO] Usando nube completa (fallback).")
    else:
        raise FileNotFoundError("No existe nube de escena.")

    # --- Metadata ---
    meta_path = None
    if paths["meta_processed"].exists():
        meta_path = paths["meta_processed"]
    elif paths["meta_raw"].exists():
        meta_path = paths["meta_raw"]
    else:
        print("[WARN] No se encontró metadata.json (no habrá view-clamp)")

    # --- Extraer centro GT del controlador para crop esférico (CAUSA #5) ---
    crop_center = None
    if meta_path is not None:
        try:
            meta_dict = read_json(meta_path)
            center_xyz, _ = load_controller_pose(meta_dict, side=args.side)
            crop_center = center_xyz
            print(f"[INFO] crop_center desde GT ({args.side}): {center_xyz.tolist()}")
        except (KeyError, ValueError, FileNotFoundError) as e:
            print(f"[WARN] No se pudo extraer centro GT para crop: {e}")

    # --- Matcher ---
    matcher = PPFMatcher(
        model_ply_path=str(paths["model_ready"]),
        scene_ply_path=str(scene_path),
        metadata_path=str(meta_path) if meta_path else None,
        # PPF
        ppf_rel_sampling=args.ppf_rel_sampling,
        ppf_rel_distance=args.ppf_rel_distance,
        ppf_scene_sample_step=args.ppf_scene_sample_step,
        ppf_scene_distance=args.ppf_scene_distance,
        ppf_top_n=args.ppf_top_n,
        # ICP
        icp_iterations=args.icp_iterations,
        icp_tolerance=args.icp_tolerance,
        icp_rejection_scale=args.icp_rejection_scale,
        icp_num_levels=args.icp_num_levels,
        # Pre-filtering
        crop_radius=args.crop_radius,
        crop_min_points=args.crop_min_points,
        crop_center=crop_center,
        view_clamp_enable=bool(args.view_clamp),
        view_clamp_near=args.view_near,
        view_clamp_far=args.view_far,
        use_right_camera=(not args.use_left_camera),
        density_enable=bool(args.density),
        density_radius=args.density_radius,
        density_min_percentile=args.density_min_p,
        density_max_percentile=args.density_max_p,
        save_subclouds=bool(args.save_subclouds),
    )

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- match_meta.json ---
    match_meta = {
        "session": args.session,
        "frame": int(args.frame),
        "side": str(args.side),
        "scene_path": str(scene_path),
        "has_segmented_input": bool(paths["scene_segmented"].exists()),
        "has_full_scene_input": bool(paths["scene_full"].exists()),
        "used_full_scene_fallback": bool(used_full_scene_fallback),
        "model_ready_path": str(paths["model_ready"]),
        "metadata_path": str(meta_path) if meta_path else None,
        "method": "ppf_icp_opencv",
    }
    with open(output_dir / "match_meta.json", "w", encoding="utf-8") as f:
        json.dump(match_meta, f, indent=2)

    # --- Matching ---
    result = matcher.run(output_dir=output_dir)

    # --- Debug: añadir used_full_scene_fallback ---
    debug_dict = result.get("debug", {})
    if not isinstance(debug_dict, dict):
        debug_dict = {}
    debug_dict["used_full_scene_fallback"] = bool(used_full_scene_fallback)

    # --- pose_best.json ---
    T = result["transformation"]
    output = {
        "transformation": T.tolist() if hasattr(T, "tolist") else T,
        "fitness": float(result["fitness"]),
        "rmse": float(result["rmse"]),
        "score": float(result["score"]),
        "debug": debug_dict,
    }
    with open(output_dir / "pose_best.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # --- aligned_model.ply ---
    try:
        model_pcd = o3d.io.read_point_cloud(str(paths["model_ready"]))
        if len(model_pcd.points) > 0:
            T_np = T if isinstance(T, np.ndarray) else np.array(T, dtype=np.float64)
            model_pcd.transform(T_np)
            o3d.io.write_point_cloud(
                str(output_dir / "aligned_model.ply"), model_pcd, write_ascii=False,
            )
    except Exception as e:
        print(f"[WARN] No se pudo generar aligned_model.ply: {e}")

    # --- Log ---
    print(f"[OK] PPF+ICP matching completado.")
    print(f"  fitness={result['fitness']:.4f}  rmse={result['rmse']:.6f}  "
          f"score={result['score']:.0f}")
    print(f"  → {output_dir / 'pose_best.json'}")
    print(f"  → {output_dir / 'aligned_model.ply'}")

if __name__ == "__main__":
    main()