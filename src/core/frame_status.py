# ============================================================
# File: src/core/frame_status.py
# Diagnóstico y clasificación de frames del pipeline
#
# Proporciona:
#   - Sanity checks SE(3) para transformaciones estimadas
#   - Inferencia de lado del controlador desde metadata
#   - Clasificación de frames (válido/inválido) con razón de fallo
#   - Lectura de métricas de segmentación y centros de PLY
# ============================================================

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from src.core.io_utils import read_json, safe_read_json
from src.core.transforms import T_to_translation_quat_xyzw


# -------------------------
# SE(3) sanity checks
# -------------------------
def se3_sanity_check(
    T: np.ndarray,
    *,
    orth_eps: float = 1e-2,
    det_eps: float = 1e-2,
) -> Dict[str, Any]:
    """
    Sanity checks formales para una transformación SE(3)
    Devuelve:
      {
        "se3_ok": bool,
        "se3_reason": str,
        "det_R": float | None,
        "orth_err_fro": float | None,
        "has_nan": bool
      }
    """
    out: Dict[str, Any] = {
        "se3_ok": False,
        "se3_reason": "unknown",
        "det_R": None,
        "orth_err_fro": None,
        "has_nan": True,
    }

    try:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            out["se3_reason"] = "bad_shape"
            return out

        if not np.isfinite(T).all():
            out["se3_reason"] = "nan_or_inf"
            out["has_nan"] = True
            return out

        out["has_nan"] = False
        R = T[:3, :3]
        I = np.eye(3, dtype=np.float64)
        orth_err = float(np.linalg.norm(R.T @ R - I, ord="fro"))
        det_R = float(np.linalg.det(R))

        out["orth_err_fro"] = orth_err
        out["det_R"] = det_R

        if orth_err > float(orth_eps):
            out["se3_reason"] = "non_orthogonal"
            return out

        if abs(det_R - 1.0) > float(det_eps):
            out["se3_reason"] = "det_not_1"
            return out

        out["se3_ok"] = True
        out["se3_reason"] = "ok"
        return out
    except Exception:
        out["se3_reason"] = "exception"
        return out


# -------------------------
# Side inference from metadata
# -------------------------
def infer_side_present_from_metadata(meta: Dict[str, Any]) -> str:
    """
    Devuelve: "right" | "left" | "both" | "unknown" en función de la existencia de ControllerPoseRight/Left o ControllerPose.
    """
    try:
        has_r = False
        has_l = False

        if isinstance(meta.get("ControllerPoseRight"), dict):
            has_r = True
        if isinstance(meta.get("ControllerPoseLeft"), dict):
            has_l = True

        cp = meta.get("ControllerPose")
        if isinstance(cp, dict):
            if isinstance(cp.get("Right"), dict):
                has_r = True
            if isinstance(cp.get("Left"), dict):
                has_l = True

        if has_r and has_l:
            return "both"
        if has_r:
            return "right"
        if has_l:
            return "left"
        return "unknown"
    except Exception:
        return "unknown"


# -------------------------
# Frame status
# -------------------------
@dataclass
class FrameStatus:
    frame_index: int

    # presence flags
    has_raw_frame: bool
    has_segmented: bool
    used_full_scene_fallback: bool
    has_aligned_model: bool
    has_eval_centers: bool

    # inferred info
    side_present: str  # "right" | "left" | "both" | "unknown"

    # derived
    valid_frame: bool
    fail_reason: str  # "ok" | "no_raw" | "no_segment" | "segment_empty" | "fallback_full_scene" | "no_aligned" | "no_eval_centers" | ...
    fail_detail: Optional[str]

    # counts
    seg_n_points: Optional[int] = None

    # se3 sanity
    se3_ok: Optional[bool] = None
    se3_reason: Optional[str] = None
    det_R: Optional[float] = None
    orth_err_fro: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "has_raw_frame": self.has_raw_frame,
            "has_segmented": self.has_segmented,
            "used_full_scene_fallback": self.used_full_scene_fallback,
            "has_aligned_model": self.has_aligned_model,
            "has_eval_centers": self.has_eval_centers,
            "side_present": self.side_present,
            "valid_frame": self.valid_frame,
            "fail_reason": self.fail_reason,
            "fail_detail": self.fail_detail,
            "seg_n_points": self.seg_n_points,
            "se3_ok": self.se3_ok,
            "se3_reason": self.se3_reason,
            "det_R": self.det_R,
            "orth_err_fro": self.orth_err_fro,
        }


def derive_fail_reason(
    *,
    has_raw_frame: bool,
    has_segmented: bool,
    seg_n_points: Optional[int],
    used_full_scene_fallback: bool,
    has_aligned_model: bool,
    has_eval_centers: bool,
    require_segment: bool = True,
    require_eval_centers: bool = True,
) -> Tuple[str, Optional[str], bool]:
    """
    Devuelve: (fail_reason, fail_detail, valid_frame)
    """
    if not has_raw_frame:
        return "no_raw", "metadata not found (processed or raw)", False

    if require_segment:
        if not has_segmented:
            return "no_segment", "object_segmented.ply missing", False
        if seg_n_points is not None and seg_n_points <= 0:
            return "segment_empty", f"segmented n_points={seg_n_points}", False

    # filtros necesarios para evitar que fallback contamine las métricas 
    if used_full_scene_fallback:
        return "fallback_full_scene", "matched against full scene (seg missing/invalid)", False

    if not has_aligned_model:
        return "no_aligned", "aligned_model.ply or pose_best.json missing", False

    if require_eval_centers and not has_eval_centers:
        return "no_eval_centers", "eval_centers.json missing/not computed", False

    return "ok", None, True


def load_used_full_scene_fallback(ppf_dir: Path) -> bool:
    """
    Lee ppf_match/match_meta.json si existe. Devuelve False si no se encuentra.
    """
    meta = safe_read_json(ppf_dir / "match_meta.json")
    if not meta:
        return False
    return bool(meta.get("used_full_scene_fallback", False))


# ==============================================================
# Unificación de load_seg_n_points 
# ==============================================================

def load_seg_n_points(seg_metrics_path: Path) -> Optional[int]:
    """
    Intenta leer el nº de puntos segmentados de segmentation_metrics.json.

    Soporta todas las variantes de clave usadas históricamente en el pipeline, tanto en nivel raíz como anidadas bajo 'outputs'.

    Claves buscadas (en orden de prioridad):
        n_out, n_points_segmented, n_points, n_segmented, n, points, n_points_obj
    """
    d = safe_read_json(seg_metrics_path)
    if not d:
        return None

    _KEYS = ("n_out", "n_points_segmented", "n_points", "n_segmented", "n", "points", "n_points_obj")

    # variantes planas (nivel raíz)
    for k in _KEYS:
        if k in d:
            try:
                return int(d[k])
            except Exception:
                pass

    # variantes anidadas bajo 'outputs'
    if "outputs" in d and isinstance(d["outputs"], dict):
        for k in _KEYS:
            if k in d["outputs"]:
                try:
                    return int(d["outputs"][k])
                except Exception:
                    pass

    return None


def compute_centers_from_ply(ply_path: Path) -> Optional[np.ndarray]:
    """
    Centro (media XYZ) de un PLY sin depender de Open3D.
    Soporta:
      - PLY ASCII
      - PLY binary_little_endian (vértices float32/float64/int/uint)
    Devuelve np.ndarray shape (3,) o None si no se puede leer / vacío.
    """
    try:
        import struct

        ply_path = Path(ply_path)
        if not ply_path.exists():
            return None

        with ply_path.open("rb") as f:
            header_lines: List[bytes] = []
            while True:
                line = f.readline()
                if not line:
                    return None
                header_lines.append(line)
                if line.strip() == b"end_header":
                    break

            header = b"".join(header_lines).decode("ascii", errors="ignore").splitlines()

            fmt = None
            n_verts = None
            in_vertex_element = False
            props: List[Tuple[str, str]] = []

            for ln in header:
                s = ln.strip()
                if s.startswith("format"):
                    parts = s.split()
                    if len(parts) >= 3:
                        fmt = parts[1]
                elif s.startswith("element vertex"):
                    parts = s.split()
                    n_verts = int(parts[2])
                    in_vertex_element = True
                    props = []
                elif s.startswith("element") and not s.startswith("element vertex"):
                    in_vertex_element = False
                elif in_vertex_element and s.startswith("property"):
                    parts = s.split()
                    if len(parts) == 3:
                        ptype, pname = parts[1], parts[2]
                        props.append((ptype, pname))

            if fmt is None or n_verts is None or n_verts <= 0:
                return None

            names = [p[1] for p in props]
            if not all(k in names for k in ("x", "y", "z")):
                return None
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")

            def ply_type_to_struct(t: str) -> Tuple[str, int]:
                t = t.lower()
                if t in ("float", "float32"):
                    return ("f", 4)
                if t in ("double", "float64"):
                    return ("d", 8)
                if t in ("char", "int8"):
                    return ("b", 1)
                if t in ("uchar", "uint8"):
                    return ("B", 1)
                if t in ("short", "int16"):
                    return ("h", 2)
                if t in ("ushort", "uint16"):
                    return ("H", 2)
                if t in ("int", "int32"):
                    return ("i", 4)
                if t in ("uint", "uint32"):
                    return ("I", 4)
                raise ValueError(f"Tipo de propiedad PLY no soportado: {t}")

            if fmt == "ascii":
                sum_xyz = np.zeros(3, dtype=np.float64)
                count = 0
                for _ in range(n_verts):
                    line = f.readline()
                    if not line:
                        break
                    parts = line.decode("ascii", errors="ignore").strip().split()
                    if len(parts) < len(props):
                        continue
                    try:
                        x = float(parts[ix])
                        y = float(parts[iy])
                        z = float(parts[iz])
                    except Exception:
                        continue
                    if np.isfinite([x, y, z]).all():
                        sum_xyz += np.array([x, y, z], dtype=np.float64)
                        count += 1
                if count == 0:
                    return None
                return (sum_xyz / count).astype(np.float64)

            if fmt != "binary_little_endian":
                return None

            struct_chars = []
            for ptype, _pname in props:
                ch, _sz = ply_type_to_struct(ptype)
                struct_chars.append(ch)
            row_fmt = "<" + "".join(struct_chars)
            row_size = struct.calcsize(row_fmt)

            sum_xyz = np.zeros(3, dtype=np.float64)
            count = 0
            for _ in range(n_verts):
                data = f.read(row_size)
                if len(data) != row_size:
                    break
                vals = struct.unpack(row_fmt, data)
                x = float(vals[ix])
                y = float(vals[iy])
                z = float(vals[iz])
                if np.isfinite([x, y, z]).all():
                    sum_xyz += np.array([x, y, z], dtype=np.float64)
                    count += 1

            if count == 0:
                return None
            return (sum_xyz / count).astype(np.float64)

    except Exception:
        return None