# ============================================================
# File: src/core/rectification.py
# Rectificación estéreo y evaluación de calidad epipolar
#
# Proporciona:
#   - Cambio de base Unity → OpenCV para poses de cámara
#   - Rectificación estéreo con cv2.stereoRectify
#   - Evaluación de calidad epipolar con ORB + RANSAC
#   - Máscara de píxeles válidos tras rectificación
# ============================================================

import numpy as np
import cv2

# Flag de debug temporal para rectificacion.
RECTIFY_DEBUG = False

# Cambio de base Unity -> OpenCV en coordenadas de camara.
# Unity usa Y hacia arriba, OpenCV usa Y hacia abajo.
UNITY_TO_OPENCV_BASIS = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def _build_valid_mask(map_x: np.ndarray, map_y: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """
    Construye una máscara booleana de píxeles válidos tras remap.
    Un píxel es válido si proviene de una coordenada válida de la imagen origina (y no de relleno negro por bordes).
    """
    h, w = shape_hw
    ones = np.full((h, w), 255, dtype=np.uint8)
    remapped = cv2.remap(
        ones,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return remapped > 0


def _intersection_roi(roi_a: tuple[int, int, int, int], roi_b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Intersección de dos ROI (x, y, w, h).
    """
    x_a, y_a, w_a, h_a = roi_a
    x_b, y_b, w_b, h_b = roi_b

    x0 = max(x_a, x_b)
    y0 = max(y_a, y_b)
    x1 = min(x_a + w_a, x_b + w_b)
    y1 = min(y_a + h_a, y_b + h_b)

    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    return int(x0), int(y0), int(w), int(h)


def Rt_left_to_right_from_Twc(T_W_L_u: np.ndarray, T_W_R_u: np.ndarray):
    """
    Devuelve (R, t) para OpenCV stereoRectify: X_right = R * X_left + t

    Entrada: poses camera->world en Unity.
    """
    # 1) Transformacion relativa correcta: T_R<-L = inv(T_W<-R) @ T_W<-L
    T_R_L_u = np.linalg.inv(T_W_R_u) @ T_W_L_u
    R_u = T_R_L_u[:3, :3].astype(np.float64)
    t_u = T_R_L_u[:3, 3].astype(np.float64)

    # 2) Cambio de base de Unity a OpenCV aplicado sobre la relativa.
    M = UNITY_TO_OPENCV_BASIS
    R_cv = M @ R_u @ M
    t_cv = (M @ t_u).reshape(3, 1)

    if RECTIFY_DEBUG:
        print("[RECTIFY] t_unity (R<-L):", t_u)
        print("[RECTIFY] t_opencv (R<-L):", t_cv.reshape(3))
        print("[RECTIFY] baseline_m:", f"{np.linalg.norm(t_cv):.6f}")
        print("[RECTIFY] det(R):", f"{np.linalg.det(R_cv):.8f}")

    return R_cv, t_cv


def rectify_pair(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    alpha: float = -1.0,
    return_debug: bool = False,
):
    """
    Rectifica el par estéreo (sin distorsión) y devuelve:
      - por defecto: (left_rect, right_rect, Q)
      - si return_debug=True: (left_rect, right_rect, Q, debug_info)

    debug_info incluye:
      - valid_mask_common: máscara booleana (H,W) válida en ambas imágenes rectificadas
      - valid_ratio: porcentaje de píxeles válidos comunes
      - roi_left, roi_right, roi_common
      - rectify_info: parámetros de rectificación serializables
    """
    h, w = left_gray.shape
    image_size = (w, h)
    D0 = np.zeros((5, 1), dtype=np.float64)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K_left,
        distCoeffs1=D0,
        cameraMatrix2=K_right,
        distCoeffs2=D0,
        imageSize=image_size,
        R=R,
        T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K_left, D0, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_right, D0, R2, P2, image_size, cv2.CV_32FC1)

    left_rect = cv2.remap(
        left_gray,
        map1x,
        map1y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    right_rect = cv2.remap(
        right_gray,
        map2x,
        map2y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if not return_debug:
        if RECTIFY_DEBUG:
            print("[RECTIFY] roi_left:", tuple(int(v) for v in roi1))
            print("[RECTIFY] roi_right:", tuple(int(v) for v in roi2))
        return left_rect, right_rect, Q

    # --- Debug/máscara válida común ---
    valid_left = _build_valid_mask(map1x, map1y, (h, w))
    valid_right = _build_valid_mask(map2x, map2y, (h, w))
    valid_common = np.logical_and(valid_left, valid_right)

    roi_left = tuple(int(v) for v in roi1)
    roi_right = tuple(int(v) for v in roi2)
    roi_common = _intersection_roi(roi_left, roi_right)

    debug_info = {
        "roi_left": roi_left,
        "roi_right": roi_right,
        "roi_common": roi_common,
        "valid_mask_common": valid_common,
        "valid_ratio": float(np.mean(valid_common)),

        # Matrices de rectificación 
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,

        # Parámetros serializables de rectificación por frame.
        "rectify_info": {
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "roi1": [int(v) for v in roi1],
            "roi2": [int(v) for v in roi2],
            "image_size": [int(w), int(h)],
            "alpha": float(alpha),
            "flags": int(cv2.CALIB_ZERO_DISPARITY),
        },
    }

    if RECTIFY_DEBUG:
        print("[RECTIFY] roi_left:", roi_left)
        print("[RECTIFY] roi_right:", roi_right)
        print("[RECTIFY] roi_common:", roi_common)
        print("[RECTIFY] valid_ratio:", f"{debug_info['valid_ratio']:.6f}")

    return left_rect, right_rect, Q, debug_info


# ==============================================================
# ==============================================================
# ==============================================================

def y_error_orb(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    nfeatures: int = 3000,
) -> dict | None:
    """
    Estima el error epipolar vertical (|dy|) con ORB + RANSAC(F).

    Útil para evaluar la calidad de la rectificación estéreo: tras una rectificación perfecta, |dy| debería ser ~0 px.

    Devuelve None si no hay suficientes correspondencias.
    """
    left_u8 = left_gray.astype(np.uint8) if left_gray.dtype != np.uint8 else left_gray
    right_u8 = right_gray.astype(np.uint8) if right_gray.dtype != np.uint8 else right_gray

    orb = cv2.ORB_create(nfeatures=nfeatures)
    k1, d1 = orb.detectAndCompute(left_u8, None)
    k2, d2 = orb.detectAndCompute(right_u8, None)

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 12:
        return None

    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return None

    in1 = pts1[mask.ravel() == 1][:, 0, :]
    in2 = pts2[mask.ravel() == 1][:, 0, :]

    if in1.shape[0] < 8:
        return None

    dy = np.abs(in1[:, 1] - in2[:, 1]).astype(np.float64)

    return {
        "median_abs_dy_px": float(np.median(dy)),
        "mean_abs_dy_px": float(np.mean(dy)),
        "p90_abs_dy_px": float(np.percentile(dy, 90)),
        "n_matches": int(len(good)),
        "n_inliers": int(dy.shape[0]),
    }