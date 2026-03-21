# ============================================================
# File: src/core/disparity.py
# Cálculo de disparidad (SGBM + WLS) y métricas asociadas
#
# Proporciona:
#   - CLAHE para mejora de contraste en imágenes passthrough
#   - Disparidad SGBM (Semi-Global Block Matching)
#   - Disparidad SGBM + filtro WLS (cv2.ximgproc)
#   - Métricas de completitud y calidad de disparidad
#   - Autoajuste de rango SGBM desde Z_gt
# ============================================================

import math

import cv2
import numpy as np


# ==============================================================
# UTILIDADES GENERALES
# ==============================================================

def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Mejora el contraste local, muy útil en imágenes passthrough con baja textura o iluminación desigual.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(gray)


def disparity_to_uint8(disp: np.ndarray, max_disp: float = 192.0) -> np.ndarray:
    """
    Convierte un mapa de disparidad float32 (en píxeles) a imagen uint8 para visualización.
    """
    max_disp = float(max(max_disp, 1e-6))
    disp_clip = np.clip(disp, 0, max_disp)
    return (disp_clip / max_disp * 255.0).astype(np.uint8)


def disparity_metrics(disp: np.ndarray) -> dict:
    """
    Métricas globales simples sobre disparidad > 0.
    """
    valid = disp > 0
    completeness = float(np.mean(valid)) if disp.size else 0.0

    if np.any(valid):
        mean = float(np.mean(disp[valid]))
        std = float(np.std(disp[valid]))
    else:
        mean, std = 0.0, 0.0

    return {
        "completeness": completeness,
        "mean": mean,
        "std": std
    }


# ==============================================================
# DISPARIDAD SGBM
# ==============================================================

def compute_disparity_sgbm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 256,
    block_size: int = 9,
    uniqueness_ratio: int = 15,
    speckle_window_size: int = 50,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    clahe: bool = True,
) -> np.ndarray:
    """
    Calcula disparidad usando StereoSGBM.

    Parámetros importantes:
    - min_disparity: desplazamiento mínimo (acotado por rango Z_GT).
    - num_disparities: rango total (múltiplo de 16).
    - block_size: tamaño ventana correlación.
    - uniqueness_ratio: penaliza soluciones ambiguas.
    - speckle_*: elimina regiones pequeñas incoherentes.
    - disp12_max_diff: chequeo left-right.
    """

    # Validaciones básicas
    assert left_gray.ndim == 2 and right_gray.ndim == 2, "Imágenes deben ser grayscale"
    assert left_gray.shape == right_gray.shape, "Resolución izquierda/derecha debe coincidir"
    assert left_gray.dtype == np.uint8 and right_gray.dtype == np.uint8, "Imágenes deben ser uint8"
    assert int(num_disparities) % 16 == 0, "num_disparities debe ser múltiplo de 16"
    assert int(block_size) % 2 == 1, "block_size debe ser impar"

    # Mejora de contraste 
    if clahe:
        left_gray = apply_clahe(left_gray)
        right_gray = apply_clahe(right_gray)

    block_size = int(block_size)

    stereo = cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=int(num_disparities),
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        disp12MaxDiff=int(disp12_max_diff),
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Cálculo de disparidad 
    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # Limpieza básica
    disp[~np.isfinite(disp)] = 0.0
    disp[disp < 0] = 0.0

    # Filtro para reducir ruido aislado
    disp = cv2.medianBlur(disp, 5)

    # Eliminar micro-disparidades residuales
    disp[disp < 1.0] = 0.0

    return disp


# ==============================================================
# DISPARIDAD SGBM + WLS
# ==============================================================

def compute_disparity_wls(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 256,
    block_size: int = 9,
    uniqueness_ratio: int = 15,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
    disp12_max_diff: int = 1,
    lr_max_diff: float = 2.0,
    wls_lambda: float = 8000.0,
    wls_sigma: float = 1.5,
    clahe: bool = True,
):
    """
    SGBM + filtro WLS.

    - Se calculan disparidades izquierda y derecha.
    - Se aplica filtro WLS para suavizado respetando bordes.
    - Devuelve disparidad float32 y máscara válida.
    """

    if clahe:
        left_gray = apply_clahe(left_gray)
        right_gray = apply_clahe(right_gray)

    assert int(num_disparities) % 16 == 0
    assert int(block_size) % 2 == 1

    block_size = int(block_size)

    # Matcher izquierdo
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=int(num_disparities),
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        disp12MaxDiff=int(disp12_max_diff),
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    if not (hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher")):
        raise RuntimeError("Instala opencv-contrib-python para usar WLS.")

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    dispL = left_matcher.compute(left_gray, right_gray).astype(np.int16)
    dispR = right_matcher.compute(right_gray, left_gray).astype(np.int16)

    # Filtro WLS
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls.setLambda(float(wls_lambda))
    wls.setSigmaColor(float(wls_sigma))

    disp_wls = wls.filter(dispL, left_gray, None, dispR).astype(np.float32) / 16.0

    disp_wls[~np.isfinite(disp_wls)] = 0.0

    valid = disp_wls > 0.5
    disp_wls[~valid] = 0.0

    return disp_wls, valid


# ==============================================================
# Funciones reubicadas desde run_disparity.py
# ==============================================================

def disparity_metrics_on_mask(disp: np.ndarray, valid_mask: np.ndarray, min_disp: int) -> dict:
    """
    Métricas de disparidad SOLO en región válida (evita bordes negros).

    Devuelve completitud "real" (disp > 0.5) y "estricta" (disp > min_disp+1), más media/std sobre píxeles válidos.
    """
    valid_mask = valid_mask.astype(bool)

    if valid_mask.size == 0 or np.count_nonzero(valid_mask) == 0:
        return {
            "valid_ratio": 0.0,
            "completeness_valid": 0.0,
            "completeness_strict": 0.0,
            "mean_valid": 0.0,
            "std_valid": 0.0,
            "n_valid_pixels": 0,
            "n_disp_pos": 0,
            "n_disp_strict": 0,
        }

    disp_in_valid = disp[valid_mask]
    valid_ratio = float(np.mean(valid_mask))

    # 1) Completitud real
    disp_pos = disp_in_valid[disp_in_valid > 0.5]
    completeness_valid = float(disp_pos.size / disp_in_valid.size)

    # 2) Completitud estricta
    thr_strict = float(int(min_disp)) + 1.0
    disp_strict = disp_in_valid[disp_in_valid > thr_strict]
    completeness_strict = float(disp_strict.size / disp_in_valid.size)

    if disp_pos.size == 0:
        mean_valid = 0.0
        std_valid = 0.0
    else:
        mean_valid = float(np.mean(disp_pos))
        std_valid = float(np.std(disp_pos))

    return {
        "valid_ratio": valid_ratio,
        "completeness_valid": completeness_valid,
        "completeness_strict": completeness_strict,
        "mean_valid": mean_valid,
        "std_valid": std_valid,
        "n_valid_pixels": int(disp_in_valid.size),
        "n_disp_pos": int(disp_pos.size),
        "n_disp_strict": int(disp_strict.size),
        "threshold_strict_px": thr_strict,
    }


def save_disp_png_auto(disp: np.ndarray, out_path) -> None:
    """
    Visualización de disparidad por percentil (evita que 1 outlier aplaste el contraste).
    """
    if np.any(disp > 0):
        vmax = float(np.percentile(disp[disp > 0], 95))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0

    vis = (np.clip(disp, 0, vmax) / vmax * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), vis)


def auto_sgbm_range_from_Z(
    fx_px: float,
    baseline_m: float,
    Z_gt: float,
    z_band: float,
    z_min_floor: float = 0.20,
    max_num_disp: int = 512,
) -> tuple:
    """
    Autoajuste de rango SGBM (minDisparity, numDisparities) a partir de Z_gt.

    Fórmula base:  d = fx * b / Z

    Blindajes:
    - z_min_floor evita Z_min ridículo (explota d_max).
    - max_num_disp limita coste/memoria.

    Devuelve
    --------
    (Z_min, Z_max, d_min, d_max, min_disp, num_disp, z_min_floor, max_num_disp)
    """
    Z_min = float(Z_gt - z_band)
    Z_max = float(Z_gt + z_band)

    Z_min = max(Z_min, float(z_min_floor))
    Z_max = max(Z_max, Z_min + 1e-3)

    d_max = (fx_px * baseline_m) / Z_min
    d_min = (fx_px * baseline_m) / Z_max

    min_disp = int(math.floor(d_min))
    if min_disp < 0:
        min_disp = 0

    n = int(math.ceil(d_max) - min_disp)
    if n < 16:
        n = 16
    num_disp = int(((n + 15) // 16) * 16)

    if num_disp > int(max_num_disp):
        num_disp = int((int(max_num_disp) // 16) * 16)

    return (
        float(Z_min),
        float(Z_max),
        float(d_min),
        float(d_max),
        int(min_disp),
        int(num_disp),
        float(z_min_floor),
        int(max_num_disp),
    )