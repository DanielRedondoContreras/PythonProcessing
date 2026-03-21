# ============================================================
# File: src/core/depth.py
# Conversión disparidad a profundidad métrica y previsualización
#
# Proporciona:
#   - Cálculo de profundidad: Z = (fx_rect * baseline) / disparidad
#   - Filtrado por rango válido [z_min, z_max]
#   - Previsualización 8-bit (cerca=blanco, lejos=negro)
#   - Métricas de cobertura y estadísticas de profundidad
# ============================================================

import numpy as np


def compute_depth_from_disparity(
    disparity_px: np.ndarray,
    K_rect: np.ndarray,
    baseline_m: float,
    valid_mask: np.ndarray | None = None,
    min_disp_px: float = 1.0,
    z_min_m: float = 0.2,
    z_max_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Calcula profundidad en metros desde disparidad rectificada.

    Formula:
        depth_m = (fx_rect * baseline_m) / disparity_px

    Reglas de validez:
    - Disparidad finita y mayor a min_disp_px.
    - Si se pasa valid_mask, se combina con las reglas anteriores.
    - Se invalida cualquier valor fuera del rango [z_min_m, z_max_m].
    """
    d = np.asarray(disparity_px, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"disparity_px debe ser 2D (H, W). Recibido: shape={d.shape}")

    K = np.asarray(K_rect, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K_rect debe tener shape (3, 3). Recibido: {K.shape}")

    fx_rect = float(K[0, 0])
    if not np.isfinite(fx_rect) or fx_rect <= 0.0:
        raise ValueError(f"K_rect[0,0] invalido: {fx_rect}")

    baseline = float(baseline_m)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(f"baseline_m invalido: {baseline_m}")

    if z_max_m <= z_min_m:
        raise ValueError(f"Rango Z invalido: z_min_m={z_min_m}, z_max_m={z_max_m}")

    valid_disp = np.isfinite(d) & (d > float(min_disp_px))

    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape != d.shape:
            raise ValueError(
                "valid_mask debe tener la misma shape que disparity_px. "
                f"mask={vm.shape}, disp={d.shape}"
            )
        valid_disp &= vm

    depth_m = np.full(d.shape, np.nan, dtype=np.float32)
    if np.any(valid_disp):
        depth_m[valid_disp] = (fx_rect * baseline) / d[valid_disp]

    finite_depth = np.isfinite(depth_m)
    depth_valid_mask = (
        valid_disp
        & finite_depth
        & (depth_m >= float(z_min_m))
        & (depth_m <= float(z_max_m))
    )

    depth_m[~depth_valid_mask] = np.nan

    n_total = int(depth_m.size)
    n_input_mask_valid = int(np.count_nonzero(valid_mask)) if valid_mask is not None else n_total
    n_disp_finite = int(np.count_nonzero(np.isfinite(d)))
    n_disp_gt_min = int(np.count_nonzero(np.isfinite(d) & (d > float(min_disp_px))))
    n_valid_final = int(np.count_nonzero(depth_valid_mask))

    if n_valid_final > 0:
        z_vals = depth_m[depth_valid_mask]
        z_real_min = float(np.min(z_vals))
        z_real_max = float(np.max(z_vals))
    else:
        z_real_min = float("nan")
        z_real_max = float("nan")

    valid_ratio = float(n_valid_final / n_total) if n_total > 0 else 0.0

    metrics = {
        "n_total_px": n_total,
        "n_input_mask_valid": n_input_mask_valid,
        "n_disp_finite": n_disp_finite,
        "n_disp_gt_min": n_disp_gt_min,
        "n_valid_depth": n_valid_final,
        "valid_ratio": valid_ratio,
        "valid_percent": valid_ratio * 100.0,
        "z_min_real_m": z_real_min,
        "z_max_real_m": z_real_max,
        "z_min_cfg_m": float(z_min_m),
        "z_max_cfg_m": float(z_max_m),
        "fx_rect_px": fx_rect,
        "baseline_m": baseline,
        "min_disp_px": float(min_disp_px),
    }

    return depth_m, depth_valid_mask, metrics


def depth_preview_uint8(
    depth_m: np.ndarray,
    depth_valid_mask: np.ndarray | None = None,
    *,
    z_min_m: float,
    z_max_m: float,
) -> np.ndarray:
    """
    Genera una previsualizacion 8-bit en escala de grises.

    Convencion:
    - Cerca  -> blanco
    - Lejos  -> negro
    - Invalido/NaN/Inf -> negro
    """
    z = np.asarray(depth_m, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"depth_m debe ser 2D (H, W). Recibido: shape={z.shape}")

    valid = np.isfinite(z)
    if depth_valid_mask is not None:
        vm = np.asarray(depth_valid_mask).astype(bool)
        if vm.shape != z.shape:
            raise ValueError(
                "depth_valid_mask debe tener la misma shape que depth_m. "
                f"mask={vm.shape}, depth={z.shape}"
            )
        valid &= vm

    if z_max_m <= z_min_m:
        raise ValueError(f"Rango Z invalido: z_min_m={z_min_m}, z_max_m={z_max_m}")

    valid &= (z >= float(z_min_m)) & (z <= float(z_max_m))

    if np.any(valid):
        z_valid = z[valid]
        vmin = max(float(np.percentile(z_valid, 5)), float(z_min_m))
        vmax = min(float(np.percentile(z_valid, 95)), float(z_max_m))
        if vmax <= vmin:
            vmax = vmin + 1e-3
    else:
        vmin = float(z_min_m)
        vmax = float(z_max_m)

    z_vis = np.full_like(z, fill_value=vmax, dtype=np.float32)
    z_vis[valid] = np.clip(z[valid], vmin, vmax)

    norm = 1.0 - (z_vis - vmin) / (vmax - vmin)
    img_u8 = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    img_u8[~valid] = 0
    return img_u8


def depth_metrics(depth_m: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Metricas simples sobre profundidad valida.
    """
    valid_mask = np.asarray(valid_mask).astype(bool)
    z = np.asarray(depth_m, dtype=np.float32)[valid_mask]
    z = z[np.isfinite(z)]

    coverage = float(np.mean(valid_mask)) if valid_mask.size else 0.0
    if z.size == 0:
        return {
            "coverage": coverage,
            "n_valid": 0,
            "z_mean": 0.0,
            "z_std": 0.0,
            "z_p10": 0.0,
            "z_p50": 0.0,
            "z_p90": 0.0,
        }

    return {
        "coverage": coverage,
        "n_valid": int(z.size),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "z_p10": float(np.percentile(z, 10)),
        "z_p50": float(np.percentile(z, 50)),
        "z_p90": float(np.percentile(z, 90)),
    }