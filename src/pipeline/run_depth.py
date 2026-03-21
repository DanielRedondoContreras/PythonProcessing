import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.core.data_loader import load_k_rect_from_rectification_debug
from src.core.depth import compute_depth_from_disparity, depth_preview_uint8
from src.core.io_utils import read_json, require_file


def _load_baseline(rect_dbg_path: Path) -> tuple[float, str]:
    """
    Carga baseline_m desde rectification_debug.json.

    Busca primero ``baseline_eff_m`` (campo canónico escrito por run_disparity.py), con fallback a ``baseline_m``.

    Devuelve
    --------
    (baseline_m, source_key) donde source_key indica qué campo se usó.
    """
    rect_dbg = read_json(rect_dbg_path)

    baseline_val = rect_dbg.get("baseline_eff_m")
    source_key = "baseline_eff_m"
    if baseline_val is None:
        baseline_val = rect_dbg.get("baseline_m")
        source_key = "baseline_m"

    if baseline_val is None:
        raise ValueError(
            f"No se encontro baseline_eff_m/baseline_m en {rect_dbg_path}."
        )

    baseline_m = float(baseline_val)
    if not np.isfinite(baseline_m) or baseline_m <= 0.0:
        raise ValueError(f"baseline invalida en {rect_dbg_path}: {baseline_val}")

    return baseline_m, source_key


def main():
    parser = argparse.ArgumentParser(
        description="Calcula profundidad desde disparity_rect_wls.npy usando parametros de rectificacion ya guardados."
    )
    parser.add_argument("--session", required=True, help="Nombre de sesion dentro de data/raw (ej: session_YYYYMMDD_HHMMSS)")
    parser.add_argument("--frame", type=int, default=0, help="Indice de frame (0..)")
    parser.add_argument("--min-disp", type=float, default=1.0, help="Disparidad minima (px) para considerar valida")
    parser.add_argument("--z-min", type=float, default=0.3, help="Profundidad minima (m) para recorte/preview")
    parser.add_argument("--z-max", type=float, default=5.0, help="Profundidad maxima (m) para recorte/preview")
    args = parser.parse_args()

    out_dir = Path("data") / "processed" / args.session / f"frame_{args.frame:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    disp_path = require_file(out_dir / "disparity_rect_wls.npy", "disparity_rect_wls.npy")
    mask_path = require_file(out_dir / "disparity_rect_wls_valid_mask.npy", "disparity_rect_wls_valid_mask.npy")
    rect_dbg_path = require_file(out_dir / "rectification_debug.json", "rectification_debug.json")
    require_file(out_dir / "left_rect.png", "left_rect.png")

    disparity = np.load(disp_path).astype(np.float32, copy=False)
    if disparity.ndim != 2:
        raise ValueError(f"disparity_rect_wls.npy debe ser 2D. shape recibida: {disparity.shape}")

    valid_mask = np.load(mask_path).astype(bool, copy=False)
    if valid_mask.shape != disparity.shape:
        raise ValueError(
            "disparity_rect_wls_valid_mask.npy debe tener la misma shape que disparity_rect_wls.npy. "
            f"mask={valid_mask.shape}, disp={disparity.shape}"
        )

    k_rect, k_rect_source = load_k_rect_from_rectification_debug(rect_dbg_path)
    baseline_m, baseline_source = _load_baseline(rect_dbg_path)

    depth_m, depth_valid_mask, depth_stats = compute_depth_from_disparity(
        disparity_px=disparity,
        K_rect=k_rect,
        baseline_m=baseline_m,
        valid_mask=valid_mask,
        min_disp_px=float(args.min_disp),
        z_min_m=float(args.z_min),
        z_max_m=float(args.z_max),
    )

    depth_u8 = depth_preview_uint8(
        depth_m=depth_m,
        depth_valid_mask=depth_valid_mask,
        z_min_m=float(args.z_min),
        z_max_m=float(args.z_max),
    )

    np.save(out_dir / "depth_m.npy", depth_m.astype(np.float32, copy=False))
    np.save(out_dir / "depth_valid_mask.npy", depth_valid_mask.astype(bool, copy=False))
    cv2.imwrite(str(out_dir / "depth_preview.png"), depth_u8)

    metrics = dict(depth_stats)
    metrics.update(
        {
            "disp_input": disp_path.name,
            "mask_input": mask_path.name,
            "rectification_input": rect_dbg_path.name,
            "k_rect_source": k_rect_source,
            "baseline_source": baseline_source,
            "disp_shape": [int(disparity.shape[0]), int(disparity.shape[1])],
            "valid_ratio_input_mask": float(np.mean(valid_mask)),
            "valid_ratio_depth_mask": float(np.mean(depth_valid_mask)),
            "z_min_m": float(args.z_min),
            "z_max_m": float(args.z_max),
        }
    )

    with open(out_dir / "depth_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[DEPTH INPUT]", disp_path, "shape=", disparity.shape)
    print("[OK] Guardado:", out_dir)
    print("[DEPTH METRICS]", metrics)


if __name__ == "__main__":
    main()