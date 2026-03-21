from __future__ import annotations

import argparse
import json
import open3d as o3d

from pathlib import Path

from src.core.cad_model import (
    CadPrepConfig,
    load_cad_as_pointcloud,
    preprocess_cad_pointcloud,
    cad_basic_metrics,
    validate_cad_scale,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Ruta al CAD limpio (ej: data/cad/touch_plus_clean.ply)")
    ap.add_argument("--out", dest="out_path", required=True, help="Ruta salida (ej: data/cad/touch_plus_ready.ply)")
    ap.add_argument("--voxel", type=float, default=0.005, help="Voxel size en metros (default 0.005)")
    ap.add_argument("--sample-points", type=int, default=20000, help="Puntos si el input es malla (default 20000)")
    ap.add_argument("--no-normals", action="store_true", help="No calcular normales")
    ap.add_argument("--viz", action="store_true", help="Visualizar resultado")
    args = ap.parse_args()

    cfg = CadPrepConfig(
        voxel_size=args.voxel,
        sample_points=args.sample_points,
        estimate_normals=not args.no_normals,
    )

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = load_cad_as_pointcloud(in_path, cfg)
    pcd = preprocess_cad_pointcloud(pcd, cfg)

    metrics = cad_basic_metrics(pcd)
    validate_cad_scale(metrics, cfg)

    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"No se pudo escribir: {out_path}")

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("OK: CAD preparado")
    print(f"IN : {in_path}")
    print(f"OUT: {out_path}")
    print(f"MET: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    if args.viz:
        o3d.visualization.draw_geometries([pcd], window_name="CAD ready (Open3D)")


if __name__ == "__main__":
    main()