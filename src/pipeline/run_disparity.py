import argparse
import cv2
import json
import numpy as np

from pathlib import Path

from src.core.data_loader import compute_Z_gt_camera_to_controller, load_and_validate_frame
from src.core.disparity import (
    auto_sgbm_range_from_Z,
    compute_disparity_sgbm,
    compute_disparity_wls,
    disparity_metrics,
    disparity_metrics_on_mask,
    disparity_to_uint8,
    save_disp_png_auto,
)
from src.core.rectification import Rt_left_to_right_from_Twc, rectify_pair, y_error_orb


def main():
    parser = argparse.ArgumentParser(
        description="Disparidad RAW y RECT (SGBM + WLS) con A1 (GT-range), A2 (presets) y B (GT band filter)."
    )

    parser.add_argument("--session", required=True,
                        help="Nombre de sesión dentro de data/raw (ej: session_YYYYMMDD_HHMMSS)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Índice de frame (0..)")

    parser.add_argument("--num-disparities", type=int, default=256,
                        help="Número de disparidades (múltiplo de 16)")
    parser.add_argument("--min-disparity", type=int, default=0,
                        help="minDisparity SGBM (se ignora si --use-gt-range)")

    parser.add_argument("--alpha", type=float, default=0.0,
                        help="alpha en stereoRectify (0.0 recorta a válido, 1.0 máxima área)")
    parser.add_argument("--lr-max-diff", type=float, default=1.0,
                        help="(compat) LR max diff para WLS")

    # ----------------  rango Z_GT -----------------
    parser.add_argument("--use-gt-range", action="store_true",
                        help="Acota minDisparity/numDisparities usando Z_gt del controlador.")
    parser.add_argument("--controller", choices=["right", "left"], default="right",
                        help="Controlador usado para Z_gt.")
    parser.add_argument("--z-band", type=float, default=0.25,
                        help="Banda en metros: Z_gt ± z_band.")

    # ----------------  presets SGBM ----------------
    parser.add_argument("--preset", choices=["smooth", "detail"], default="detail",
                        help="Preset SGBM: smooth (objeto liso) / detail (detalle).")

    # ----------------  CLAHE ----------------
    parser.add_argument("--no-clahe", action="store_true",
                        help="Desactiva CLAHE (Contrast Limited Adaptive Histogram Equalization) en SGBM/WLS.")

    # ----------------  filtro por banda (usando Z_GT) ----------------
    parser.add_argument("--gt-band-filter", action="store_true",
                        help="Filtra la salida final (disp_wls) al rango [d_min, d_max] derivado de Z_gt±banda.")
    parser.add_argument("--band-margin-px", type=float, default=2.0,
                        help="Margen en píxeles para relajar el filtro (tolerancia al ruido/cuantización).")

    args = parser.parse_args()

    use_clahe = not args.no_clahe

    # ----------------------------------------------------------------
    # 0) Carga del frame (imágenes + metadata + K + poses)
    # ----------------------------------------------------------------
    session_path = Path("data") / "raw" / args.session
    data = load_and_validate_frame(session_path, args.frame)

    left_raw = data["left_img"]
    right_raw = data["right_img"]

    out_dir = Path("data") / "processed" / args.session / f"frame_{args.frame:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1) Validación: error epipolar vertical en RAW (ORB+RANSAC)
    # ----------------------------------------------------------------
    yerr_raw = y_error_orb(left_raw, right_raw)
    with open(out_dir / "y_error_orb_raw.json", "w", encoding="utf-8") as f:
        json.dump(yerr_raw, f, indent=2)

    # ----------------------------------------------------------------
    # 2) Rectificación estéreo (usa R,t derivados de poses T_W_L y T_W_R)
    # ----------------------------------------------------------------
    R_lr, t_lr = Rt_left_to_right_from_Twc(data["T_W_L"], data["T_W_R"])
    left_rect, right_rect, _, rect_dbg = rectify_pair(
        left_raw,
        right_raw,
        data["K_left"],
        data["K_right"],
        R_lr,
        t_lr,
        alpha=float(args.alpha),
        return_debug=True,
    )

    # Parámetros rectificados (P1/P2) y baseline efectivo (en metros)
    P1 = rect_dbg["P1"]
    P2 = rect_dbg["P2"]
    fx_rect = float(P1[0, 0])
    baseline_eff = -float(P2[0, 3]) / float(P2[0, 0]) if float(P2[0, 0]) != 0 else 0.0

    print("[RECT P1 fx,fy,cx,cy] ", float(P1[0, 0]), float(P1[1, 1]), float(P1[0, 2]), float(P1[1, 2]))
    print("[RECT P2 fx,fy,cx,cy] ", float(P2[0, 0]), float(P2[1, 1]), float(P2[0, 2]), float(P2[1, 2]))
    print("[RECT baseline_eff_m] ", baseline_eff)

    # ----------------------------------------------------------------
    # 3) rango por GT (Z_GT) -> (d_min, d_max) -> (minDisp, numDisp)
    # ----------------------------------------------------------------
    gt_range_dbg = {}
    z_min_floor_used = 0.20
    max_num_disp_used = 512

    # Por defecto (sin activar GT-range)
    args.min_disparity = int(args.min_disparity)
    args.num_disparities = int(args.num_disparities)

    if args.use_gt_range:
        # Z_gt: distancia cámara izquierda (world) a controller (world)
        Z_gt = compute_Z_gt_camera_to_controller(
            metadata=data["meta"],
            T_W_L=data["T_W_L"],
            controller=args.controller,
        )

        # Auto-rango en disparidad derivado de banda de Z
        Z_min, Z_max, d_min, d_max, min_disp_auto, num_disp_auto, z_min_floor_used, max_num_disp_used = auto_sgbm_range_from_Z(
            fx_px=fx_rect,
            baseline_m=baseline_eff,
            Z_gt=Z_gt,
            z_band=float(args.z_band),
            z_min_floor=z_min_floor_used,
            max_num_disp=max_num_disp_used,
        )

        args.min_disparity = int(min_disp_auto)
        args.num_disparities = int(num_disp_auto)

        print("[GT] controller:", args.controller)
        print("[GT] Z_gt(m):", float(Z_gt))
        print("[GT] Z_band(m):", float(args.z_band), "=> Z_min/Z_max:", float(Z_min), float(Z_max))
        print("[GT] Z_min_floor(m):", float(z_min_floor_used))
        print("[GT] d_min/d_max(px):", float(d_min), float(d_max))
        print("[GT] minDisparity/numDisparities:", int(args.min_disparity), int(args.num_disparities))

        gt_range_dbg = {
            "controller": args.controller,
            "z_band_m": float(args.z_band),
            "z_min_floor_m": float(z_min_floor_used),
            "max_num_disp": int(max_num_disp_used),
            "fx_rect_px": float(fx_rect),
            "baseline_eff_m": float(baseline_eff),
            "Z_gt_m": float(Z_gt),
            "Z_min_m": float(Z_min),
            "Z_max_m": float(Z_max),
            "d_min_px": float(d_min),
            "d_max_px": float(d_max),
            "min_disparity": int(args.min_disparity),
            "num_disparities": int(args.num_disparities),
        }
        with open(out_dir / "gt_disparity_range.json", "w", encoding="utf-8") as f:
            json.dump(gt_range_dbg, f, indent=2)

    # Visualización
    max_disp_vis = float(int(args.min_disparity) + int(args.num_disparities))

    # ----------------------------------------------------------------
    # 4) preset SGBM (passthrough)
    # ----------------------------------------------------------------
    if args.preset == "smooth":
        # Más robusto para superficies lisas (pierde detalle fino)
        sgbm_cfg = dict(
            block_size=11,
            uniqueness_ratio=10,
            speckle_window_size=150,
            speckle_range=2,
            disp12_max_diff=1,
        )
    else:
        # Más detalle (preserva bordes)
        sgbm_cfg = dict(
            block_size=7,
            uniqueness_ratio=16,
            speckle_window_size=120,
            speckle_range=2,
            disp12_max_diff=1,
        )

    with open(out_dir / "sgbm_preset.json", "w", encoding="utf-8") as f:
        json.dump({"preset": args.preset, **sgbm_cfg}, f, indent=2)

    # ----------------------------------------------------------------
    # 5) Disparidad RAW (SGBM) - referencia (no rectificada)
    # ----------------------------------------------------------------
    disp_raw = compute_disparity_sgbm(
        left_raw,
        right_raw,
        min_disparity=int(args.min_disparity),
        num_disparities=int(args.num_disparities),
        clahe=use_clahe,
        **sgbm_cfg,
    )
    metrics_raw = disparity_metrics(disp_raw)

    np.save(out_dir / "disparity_raw.npy", disp_raw)
    cv2.imwrite(str(out_dir / "disparity_raw.png"), disparity_to_uint8(disp_raw, max_disp=max_disp_vis))
    with open(out_dir / "disparity_metrics_raw.json", "w", encoding="utf-8") as f:
        json.dump(metrics_raw, f, indent=2)

    # ----------------------------------------------------------------
    # 6) ROI común válido tras rectificación (evita bordes negros)
    # ----------------------------------------------------------------
    valid_mask_common = rect_dbg["valid_mask_common"]
    x0, y0, w, h = rect_dbg["roi_common"]

    left_rect_roi = left_rect[y0:y0 + h, x0:x0 + w]
    right_rect_roi = right_rect[y0:y0 + h, x0:x0 + w]
    valid_mask_roi = valid_mask_common[y0:y0 + h, x0:x0 + w]

    cv2.imwrite(str(out_dir / "left_rect.png"), left_rect_roi)
    cv2.imwrite(str(out_dir / "right_rect.png"), right_rect_roi)

    # ----------------------------------------------------------------
    # 7) Validación rectificación: error epipolar en ROI rectificado
    # ----------------------------------------------------------------
    yerr_rect = y_error_orb(left_rect_roi, right_rect_roi)
    with open(out_dir / "y_error_orb_rect.json", "w", encoding="utf-8") as f:
        json.dump(yerr_rect, f, indent=2)

    # ----------------------------------------------------------------
    # 8) Disparidad RECT (SGBM) - en ROI rectificado
    # ----------------------------------------------------------------
    disp_rect = compute_disparity_sgbm(
        left_rect_roi,
        right_rect_roi,
        min_disparity=int(args.min_disparity),
        num_disparities=int(args.num_disparities),
        clahe=use_clahe,
        **sgbm_cfg,
    )
    metrics_rect = disparity_metrics(disp_rect)

    metrics_sgbm_valid = disparity_metrics_on_mask(disp_rect, valid_mask_roi, int(args.min_disparity))
    print("[METRICS SGBM] ", {
        "completeness_valid": metrics_sgbm_valid["completeness_valid"],
        "completeness_strict": metrics_sgbm_valid["completeness_strict"],
        "mean_valid": metrics_sgbm_valid["mean_valid"],
        "std_valid": metrics_sgbm_valid["std_valid"],
    })

    np.save(out_dir / "disparity_rect.npy", disp_rect)
    cv2.imwrite(str(out_dir / "disparity_rect.png"), disparity_to_uint8(disp_rect, max_disp=max_disp_vis))
    with open(out_dir / "disparity_metrics_rect.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rect, f, indent=2)

    # ----------------------------------------------------------------
    # 9) Disparidad RECT (WLS) - mejora suavidad respetando bordes
    # ----------------------------------------------------------------
    disp_wls, valid_wls = compute_disparity_wls(
        left_rect_roi,
        right_rect_roi,
        min_disparity=int(args.min_disparity),
        num_disparities=int(args.num_disparities),
        lr_max_diff=float(args.lr_max_diff),
        clahe=use_clahe,
        **sgbm_cfg,
    )

    metrics_wls_valid = disparity_metrics_on_mask(disp_wls, valid_mask_roi, int(args.min_disparity))
    print("[METRICS WLS ]", {
        "completeness_valid": metrics_wls_valid["completeness_valid"],
        "completeness_strict": metrics_wls_valid["completeness_strict"],
        "mean_valid": metrics_wls_valid["mean_valid"],
        "std_valid": metrics_wls_valid["std_valid"],
    })

    vals = disp_wls[disp_wls > 0]
    if vals.size:
        print("[WLS range] min/max:", float(vals.min()), float(vals.max()))
    else:
        print("[WLS range] sin valores > 0")

    np.save(out_dir / "disparity_rect_wls.npy", disp_wls)
    save_disp_png_auto(disp_wls, out_dir / "disparity_rect_wls.png")
    np.save(out_dir / "disparity_rect_wls_valid_mask.npy", valid_wls.astype(np.uint8))

    # ----------------------------------------------------------------
    # 10) SALIDA PRINCIPAL
    #     - Por defecto: disp_final = disp_wls
    #     - filtra por banda de distancia (GT) en disparidad
    # ----------------------------------------------------------------
    disp_final = disp_wls

    if args.use_gt_range and args.gt_band_filter:
        # d_min/d_max vienen de A1 (guardados en gt_range_dbg)
        d_min = float(gt_range_dbg.get("d_min_px", np.nan))
        d_max = float(gt_range_dbg.get("d_max_px", np.nan))

        if np.isfinite(d_min) and np.isfinite(d_max) and d_max > d_min:
            m = float(args.band_margin_px)

            # Banda relajada (tolerancia)
            d_lo = max(0.0, d_min - m)
            d_hi = d_max + m

            band_mask = (disp_wls >= d_lo) & (disp_wls <= d_hi)

            disp_final = disp_wls.copy()
            disp_final[~band_mask] = 0.0

            band_info = {
                "enabled": True,
                "controller": args.controller,
                "z_band_m": float(args.z_band),
                "d_min_px": d_min,
                "d_max_px": d_max,
                "margin_px": m,
                "d_lo_px": float(d_lo),
                "d_hi_px": float(d_hi),
                "kept_ratio": float(np.mean(band_mask)) if band_mask.size else 0.0,
            }
            with open(out_dir / "gt_band_filter.json", "w", encoding="utf-8") as f:
                json.dump(band_info, f, indent=2)
        else:
            # Fallback: si por lo que sea no hay d_min/d_max válidos, no filtramos
            with open(out_dir / "gt_band_filter.json", "w", encoding="utf-8") as f:
                json.dump({"enabled": False, "reason": "d_min/d_max no disponibles"}, f, indent=2)

    np.save(out_dir / "disparity_final.npy", disp_final)
    save_disp_png_auto(disp_final, out_dir / "disparity_final.png")

    # ----------------------------------------------------------------
    # 11) Métricas VALID + debug de rectificación
    # ----------------------------------------------------------------
    metrics_rect_valid = disparity_metrics_on_mask(disp_rect, valid_mask_roi, int(args.min_disparity))
    with open(out_dir / "disparity_metrics_rect_valid.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rect_valid, f, indent=2)

    metrics_wls_valid_full = disparity_metrics_on_mask(disp_wls, valid_mask_roi, int(args.min_disparity))
    with open(out_dir / "disparity_metrics_wls_valid.json", "w", encoding="utf-8") as f:
        json.dump(metrics_wls_valid_full, f, indent=2)

    rect_debug_out = {
        "roi_left": rect_dbg.get("roi_left", None),
        "roi_right": rect_dbg.get("roi_right", None),
        "roi_common": rect_dbg.get("roi_common", None),
        "valid_ratio": rect_dbg.get("valid_ratio", None),
        "alpha": float(args.alpha),
        "baseline_eff_m": float(baseline_eff),

        # Intrínsecos rectificados (lo que run_pointcloud necesita)
        "fx_rect_px": float(P1[0, 0]),
        "fy_rect_px": float(P1[1, 1]),
        "cx_rect_px": float(P1[0, 2]),
        "cy_rect_px": float(P1[1, 2]),

        # (recomendado) guardar P1/P2 para auditoría y compatibilidad
        "P1": np.asarray(P1, dtype=float).tolist(),
        "P2": np.asarray(P2, dtype=float).tolist(),
    }
    with open(out_dir / "rectification_debug.json", "w", encoding="utf-8") as f:
        json.dump(rect_debug_out, f, indent=2)

    print("[OK] Guardado:", out_dir)
    print("[METRICS RAW ]", metrics_raw)
    print("[METRICS RECT]", metrics_rect)
    print("[YERR RAW ]", yerr_raw)
    print("[YERR RECT]", yerr_rect)


if __name__ == "__main__":
    main()