# ============================================================
# File: src/pipeline/run_session_summary.py
# Session summary + trajectory
# Estabilización temporal de pose
#
# ============================================================

from __future__ import annotations

import argparse
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.io_utils import safe_read_json
from src.core.reporting import write_csv, write_json
from src.core.temporal_stabilizer import stabilize_trajectory
from src.core.transforms import T_to_translation_quat_xyzw


def _iter_frame_dirs(session_dir: Path) -> List[Path]:
    return sorted([p for p in session_dir.glob("frame_*") if p.is_dir()])


def _frame_index_from_dir(frame_dir: Path) -> int:
    # frame_000110 -> 110
    try:
        return int(frame_dir.name.split("_")[-1])
    except Exception:
        return -1


def build_summary_frames(session_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for frame_dir in _iter_frame_dirs(session_dir):
        ppf_dir = frame_dir / "ppf_match"
        pose_eval_path = ppf_dir / "pose_eval.json"
        if not pose_eval_path.exists():
            # incluir fila mínima para trazabilidad
            rows.append(
                {
                    "frame_index": _frame_index_from_dir(frame_dir),
                    "valid_frame": False,
                    "fail_reason": "no_pose_eval",
                    "fail_detail": "pose_eval.json missing",
                }
            )
            continue

        d = safe_read_json(pose_eval_path) or {}
        # flatten minimal fields for analysis convenience
        metrics = d.get("metrics", {}) if isinstance(d.get("metrics", {}), dict) else {}
        row = {
            "frame_index": int(d.get("frame_index", _frame_index_from_dir(frame_dir))),
            "valid_frame": bool(d.get("valid_frame", False)),
            "fail_reason": d.get("fail_reason", "unknown"),
            "fail_detail": d.get("fail_detail", None),
            "has_raw_frame": d.get("has_raw_frame", None),
            "has_segmented": d.get("has_segmented", None),
            "used_full_scene_fallback": d.get("used_full_scene_fallback", None),
            "has_aligned_model": d.get("has_aligned_model", None),
            "has_eval_centers": d.get("has_eval_centers", None),
            "side_present": d.get("side_present", None),
            "seg_n_points": d.get("seg_n_points", None),
            "se3_ok": d.get("se3_ok", None),
            "se3_reason": d.get("se3_reason", None),
            "det_R": d.get("det_R", None),
            "orth_err_fro": d.get("orth_err_fro", None),
            "fitness": metrics.get("fitness", None),
            "rmse": metrics.get("rmse", None),
            "score": metrics.get("score", None),
            "center_distance_m": metrics.get("center_distance_m", None),
        }
        rows.append(row)
    return rows


def build_session_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _count(pred):
        return int(sum(1 for r in rows if pred(r)))

    n_total = len(rows)
    n_valid = _count(lambda r: bool(r.get("valid_frame")))
    n_fallback = _count(lambda r: bool(r.get("used_full_scene_fallback")))

    # reason counts
    by_reason: Dict[str, int] = {}
    for r in rows:
        k = str(r.get("fail_reason", "unknown"))
        by_reason[k] = by_reason.get(k, 0) + 1

    # side counts
    by_side: Dict[str, int] = {}
    for r in rows:
        s = str(r.get("side_present", "unknown"))
        by_side[s] = by_side.get(s, 0) + 1

    # numeric summaries (only valid)
    def _num(name: str) -> Dict[str, Any]:
        vals = [r.get(name) for r in rows if r.get("valid_frame") and (r.get(name) is not None)]
        vals_f = [float(v) for v in vals if isinstance(v, (int, float, np.floating, np.integer))]
        if not vals_f:
            return {"count": 0}
        a = np.array(vals_f, dtype=np.float64)
        return {
            "count": int(a.size),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)),
        }

    return {
        "n_frames_total": n_total,
        "n_valid_frames": n_valid,
        "n_invalid_frames": int(n_total - n_valid),
        "n_fallback_full_scene": n_fallback,
        "fail_reason_counts": by_reason,
        "side_present_counts": by_side,
        "metrics_valid": {
            "fitness": _num("fitness"),
            "rmse": _num("rmse"),
            "score": _num("score"),
            "center_distance_m": _num("center_distance_m"),
        },
    }


def build_trajectory_rows(session_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for frame_dir in _iter_frame_dirs(session_dir):
        frame_index = _frame_index_from_dir(frame_dir)
        ppf_dir = frame_dir / "ppf_match"
        pose_eval = safe_read_json(ppf_dir / "pose_eval.json") or {}
        pose_best = safe_read_json(ppf_dir / "pose_best.json") or {}

        valid_frame = bool(pose_eval.get("valid_frame", False))
        fail_reason = pose_eval.get("fail_reason", "unknown")
        used_fallback = bool(pose_eval.get("used_full_scene_fallback", False))

        T = None
        if isinstance(pose_best.get("transformation"), list):
            try:
                T = np.array(pose_best["transformation"], dtype=np.float64)
            except Exception:
                T = None

        tx = ty = tz = qx = qy = qz = qw = None
        if T is not None and T.shape == (4, 4) and np.isfinite(T).all():
            t, q = T_to_translation_quat_xyzw(T)
            tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
            qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        metrics = pose_eval.get("metrics", {}) if isinstance(pose_eval.get("metrics", {}), dict) else {}
        row = {
            "frame_index": frame_index,
            "valid_frame": valid_frame,
            "fail_reason": fail_reason,
            "used_full_scene_fallback": used_fallback,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "fitness": metrics.get("fitness", None),
            "rmse": metrics.get("rmse", None),
            "score": metrics.get("score", None),
            "center_distance_m": metrics.get("center_distance_m", None),
            "se3_ok": pose_eval.get("se3_ok", None),
            "det_R": pose_eval.get("det_R", None),
            "orth_err_fro": pose_eval.get("orth_err_fro", None),
        }
        out.append(row)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3.6 - Session summary + trajectory + Fase 5a stabilization")
    parser.add_argument("--session", required=True, help="Nombre de carpeta de sesión (ej: session_20260223_160956)")
    parser.add_argument("--session-dir", default=None, help="Override session dir (default data/processed/<session>)")
    parser.add_argument("--out-dir", default=None, help="Where to write outputs (default session dir)")

    parser.add_argument("--write-csv", action="store_true", help="Also write summary_frames.csv")

    # ---------------------------------------------------------------
    # Parámetros de estabilización temporal
    # ---------------------------------------------------------------
    parser.add_argument(
        "--stab-ema-alpha", type=float, default=0.4,
        help="Factor de suavizado EMA para estabilización temporal (0.01–1.0). "
             "1.0 = sin suavizado (solo rechazo de saltos). Default: 0.4",
    )
    parser.add_argument(
        "--stab-max-trans-m", type=float, default=0.15,
        help="Umbral máximo de traslación (m) entre frames consecutivos válidos "
             "para rechazo de saltos. Default: 0.15",
    )
    parser.add_argument(
        "--stab-max-rot-deg", type=float, default=45.0,
        help="Umbral máximo de rotación (°) entre frames consecutivos válidos "
             "para rechazo de saltos. Default: 45.0",
    )

    args = parser.parse_args()

    session_dir = Path(args.session_dir) if args.session_dir else (Path("data") / "processed" / args.session)
    out_dir = Path(args.out_dir) if args.out_dir else session_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- summary_frames ---
    rows = build_summary_frames(session_dir)
    write_json(out_dir / "summary_frames.json", rows, indent=2)
    if args.write_csv:
        write_csv(out_dir / "summary_frames.csv", rows)

    # --- trajectory cruda ---
    traj_rows = build_trajectory_rows(session_dir)
    write_csv(out_dir / "trajectory_object_world.csv", traj_rows)

    # --- trajectory estabilizada ---
    stab_rows, stab_stats = stabilize_trajectory(
        traj_rows,
        ema_alpha=args.stab_ema_alpha,
        max_trans_m=args.stab_max_trans_m,
        max_rot_deg=args.stab_max_rot_deg,
    )
    write_csv(out_dir / "trajectory_stabilized.csv", stab_rows)

    # --- session_summary con bloque stabilization ---
    session_summary = build_session_summary(rows)
    session_summary["stabilization"] = stab_stats
    write_json(out_dir / "session_summary.json", session_summary, indent=2)

    # --- Impresión de resultados ---
    print(f"[OK] wrote: {out_dir / 'summary_frames.json'}")
    if args.write_csv:
        print(f"[OK] wrote: {out_dir / 'summary_frames.csv'}")
    print(f"[OK] wrote: {out_dir / 'session_summary.json'}")
    print(f"[OK] wrote: {out_dir / 'trajectory_object_world.csv'}")
    print(f"[OK] wrote: {out_dir / 'trajectory_stabilized.csv'}")

    # Resumen rápido de estabilización
    print(
        f"[STAB] aceptados={stab_stats['n_accepted']}, "
        f"rechazados_salto={stab_stats['n_rejected_jump']}, "
        f"inválidos={stab_stats['n_invalid_or_no_pose']}"
    )
    if stab_stats["rejected_jump_frames"]:
        print(f"[STAB] frames rechazados por salto: {stab_stats['rejected_jump_frames']}")


if __name__ == "__main__":
    main()