# ============================================================
# File: src/pipeline/run_pose_eval.py
# Unificación label / label_classifier
# Deduplicación load_seg_n_points
#
# Objetivo:
# - Generar métricas por frame + flags/razones de fallo
# - Sanity checks formales de SE(3)
# - Persistencia por frame: pose_eval.json (+ eval_centers.json si aplica)
# - Agregado por sesión: per_frame_metrics.jsonl/csv + summary.json
#
#   - Si valid_frame=True pero label_classifier indica FAIL_NO_DATA, FAIL_BAD_ICP o FAIL_POSE, se fuerza valid_frame=False y fail_reason refleja la razón del clasificador. 
#  Elimina la divergencia donde label=OK y label_classifier=FAIL_*.
#
#   - load_seg_n_points importado desde frame_status.py (función unificada con lista completa de claves). Eliminada copia local.
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.io_utils import parse_frame_spec, read_json
from src.core.ppf_match import SENTINEL_RMSE, SENTINEL_SCORE
from src.core.pose_eval import (
    Thresholds,
    classify_frame,
    load_est_T,
    load_gt_T,
    pose_error,
    temporal_error,
)
from src.core.reporting import summarize_numeric, summarize_success, top_k, write_csv, write_json, write_jsonl

from src.core.frame_status import (
    compute_centers_from_ply,
    derive_fail_reason,
    infer_side_present_from_metadata,
    load_seg_n_points,
    load_used_full_scene_fallback,
    se3_sanity_check,
)


def build_paths(session: str, frame: int) -> Dict[str, Path]:
    frame_str = f"frame_{frame:06d}"
    base = Path("data") / "processed" / session / frame_str
    ppf_dir = base / "ppf_match"

    return {
        "base": base,
        "ppf_dir": ppf_dir,
        "pose_best": ppf_dir / "pose_best.json",
        "aligned_model": ppf_dir / "aligned_model.ply",
        "pose_eval": ppf_dir / "pose_eval.json",
        "eval_centers": ppf_dir / "eval_centers.json",
        "match_meta": ppf_dir / "match_meta.json",
        "segmented_ply": base / "object_segmented.ply",
        "seg_metrics": base / "segmentation_metrics.json",
        "meta_processed": base / "metadata.json",
        "meta_raw": Path("data") / "raw" / session / frame_str / "metadata.json",
    }


def pick_metadata_path(paths: Dict[str, Path]) -> Optional[Path]:
    if paths["meta_processed"].exists():
        return paths["meta_processed"]
    if paths["meta_raw"].exists():
        return paths["meta_raw"]
    return None


# ==============================================================
# Claves de label_classifier que invalidan valid_frame
# ==============================================================
_CLASSIFIER_FAIL_LABELS = frozenset({"FAIL_NO_DATA", "FAIL_BAD_ICP", "FAIL_POSE"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Fase 3.6 - Evaluación cuantitativa + flags/razones por frame")

    ap.add_argument("--session", required=True)
    ap.add_argument("--frames", required=True, help='Ej: "110" | "0:115" | "100:130:2" | "100,105,110"')
    ap.add_argument("--side", default="right", choices=["right", "left"])

    # Umbrales para classify_frame (diagnóstico histórico)
    ap.add_argument("--ok-trans-m", type=float, default=0.05)
    ap.add_argument("--ok-rot-deg", type=float, default=10.0)
    ap.add_argument("--ok-min-fitness", type=float, default=0.15)
    ap.add_argument("--ok-max-rmse", type=float, default=0.02)
    ap.add_argument("--ok-min-seg-points", type=int, default=300)

    # Outputs sesión
    ap.add_argument("--outdir", type=str, default=None, help="Por defecto data/processed/<session>/pose_eval")
    ap.add_argument("--write-csv", action="store_true")

    # Escritura por frame
    ap.add_argument(
        "--write-per-frame",
        action="store_true",
        help="Escribe pose_eval.json (y eval_centers.json si aplica) en cada frame/ppf_match/",
    )

    args = ap.parse_args()
    frames = parse_frame_spec(args.frames)

    th = Thresholds(
        max_trans_m_ok=float(args.ok_trans_m),
        max_rot_deg_ok=float(args.ok_rot_deg),
        min_fitness_ok=float(args.ok_min_fitness),
        max_rmse_ok=float(args.ok_max_rmse),
        min_segment_points_ok=int(args.ok_min_seg_points),
    )

    outdir = Path(args.outdir) if args.outdir else (Path("data") / "processed" / args.session / "pose_eval")
    outdir.mkdir(parents=True, exist_ok=True)

    per_frame_rows: List[Dict[str, Any]] = []

    prev_est: Optional[np.ndarray] = None
    prev_gt: Optional[np.ndarray] = None
    prev_frame: Optional[int] = None

    for fr in frames:
        paths = build_paths(args.session, fr)

        # ---------------------------
        # Metadata & side inference (solo informativo; NO participa en valid_frame aquí)
        # ---------------------------
        meta_path = pick_metadata_path(paths)
        meta_dict = read_json(meta_path) if meta_path and meta_path.exists() else None
        side_present = infer_side_present_from_metadata(meta_dict) if isinstance(meta_dict, dict) else "unknown"

        # ---------------------------
        # GT / EST
        # ---------------------------
        T_gt = load_gt_T(meta_path, side=args.side) if meta_path else None
        T_est = load_est_T(paths["pose_best"])

        have_gt = T_gt is not None
        have_est = T_est is not None

        # ---------------------------
        # Segmentation info (Fase 4d: función unificada desde frame_status)
        # ---------------------------
        seg_n = load_seg_n_points(paths["seg_metrics"])
        has_segmented = paths["segmented_ply"].exists()

        # ---------------------------
        # used_full_scene_fallback
        # ---------------------------
        used_full_scene_fallback = load_used_full_scene_fallback(paths["ppf_dir"])
        if (not used_full_scene_fallback) and paths["pose_best"].exists():
            try:
                dpose = read_json(paths["pose_best"])
                dbg = dpose.get("debug", {})
                if isinstance(dbg, dict) and bool(dbg.get("used_full_scene_fallback", False)):
                    used_full_scene_fallback = True
            except Exception:
                pass

        # ---------------------------
        # Pose best metrics if exist
        # ---------------------------
        fitness = None
        rmse = None
        score = None
        if paths["pose_best"].exists():
            d = read_json(paths["pose_best"])
            fitness = d.get("fitness", None)
            rmse = d.get("rmse", None)
            score = d.get("score", None)

        # ---------------------------
        # Pose error (GT vs EST)
        # ---------------------------
        trans_error_m = None
        rot_error_deg = None
        if have_est and have_gt:
            pe = pose_error(T_est, T_gt)
            trans_error_m = pe["trans_error_m"]
            rot_error_deg = pe["rot_error_deg"]

        # ---------------------------
        # Temporal error (delta)
        # ---------------------------
        if prev_est is not None and prev_gt is not None and have_est and have_gt and prev_frame is not None:
            te = temporal_error(prev_est, T_est, prev_gt, T_gt)
            delta_frame = int(fr - prev_frame)
            dtrans_error_m = te["trans_error_m"]
            drot_error_deg = te["rot_error_deg"]
        else:
            delta_frame = None
            dtrans_error_m = None
            drot_error_deg = None

        # ---------------------------
        # label_classifier: (diagnóstico)
        # ---------------------------
        label_classifier = classify_frame(
            have_est=have_est,
            have_gt=have_gt,
            seg_n_points=seg_n,
            fitness=float(fitness) if fitness is not None else None,
            rmse=float(rmse) if rmse is not None else None,
            trans_error_m=trans_error_m,
            rot_error_deg=rot_error_deg,
            th=th,
        )

        # ---------------------------
        # SE(3) sanity checks formales
        # ---------------------------
        if T_est is not None:
            se3 = se3_sanity_check(T_est)
        else:
            se3 = {"se3_ok": None, "se3_reason": "no_est", "det_R": None, "orth_err_fro": None, "has_nan": None}

        # ---------------------------
        # eval_centers (si existe segmentado + aligned_model)
        # ---------------------------
        has_aligned_model = paths["aligned_model"].exists() and (T_est is not None)
        center_distance_m = None
        has_eval_centers = False

        if has_segmented and has_aligned_model:
            seg_center = compute_centers_from_ply(paths["segmented_ply"])
            model_center = compute_centers_from_ply(paths["aligned_model"])
            if seg_center is not None and model_center is not None:
                center_distance_m = float(np.linalg.norm(seg_center - model_center))
                has_eval_centers = True

                # Opcional: escribir eval_centers.json por frame
                if args.write_per_frame:
                    paths["ppf_dir"].mkdir(parents=True, exist_ok=True)
                    write_json(
                        paths["eval_centers"],
                        {
                            "frame": int(fr),
                            "seg_center": seg_center.tolist(),
                            "aligned_center": model_center.tolist(),
                            "center_distance_m": center_distance_m,
                        },
                        indent=2,
                    )

        # ---------------------------
        # FLAGS/REASONS/VALID_FRAME 
        # ---------------------------
        has_raw_frame = meta_path is not None and meta_path.exists()

        fail_reason, fail_detail, valid_frame = derive_fail_reason(
            has_raw_frame=bool(has_raw_frame),
            has_segmented=bool(has_segmented),
            seg_n_points=seg_n,
            used_full_scene_fallback=bool(used_full_scene_fallback),
            has_aligned_model=bool(has_aligned_model),
            has_eval_centers=bool(has_eval_centers),
            require_segment=True,
            require_eval_centers=True,
        )

        # ---------------------------
        # sentinels de métricas
        # Los valores sentinel provienen de ppf_match.py (SENTINEL_RMSE=1e9, SENTINEL_SCORE=-1e9) y se generan cuando no hay matching válido.
        # El umbral se calibra a un orden de magnitud por debajo para capturar cualquier valor anómalo similar.
        # ---------------------------
        SENTINEL_ABS = min(abs(SENTINEL_RMSE), abs(SENTINEL_SCORE)) / 10.0  # 1e8
        rmse_f = float(rmse) if rmse is not None else None
        score_f = float(score) if score is not None else None

        has_rmse_sentinel = (rmse_f is not None) and (abs(rmse_f) >= SENTINEL_ABS)
        has_score_sentinel = (score_f is not None) and (abs(score_f) >= SENTINEL_ABS)

        if valid_frame and (has_rmse_sentinel or has_score_sentinel):
            valid_frame = False
            fail_reason = "bad_metrics_sentinel"
            fail_detail = f"rmse={rmse_f} score={score_f}"

        # Gate adicional: si pasa todo pero SE(3) es inválida, se marca como inválido
        if valid_frame and (se3.get("se3_ok") is False):
            valid_frame = False
            fail_reason = "bad_se3"
            fail_detail = f"se3_reason={se3.get('se3_reason')}"

        # ---------------------------
        # Unificación label / label_classifier
        #
        # Si derive_fail_reason marca valid_frame=True pero label_classifier detecta un fallo geométrico (NO_DATA, BAD_ICP, POSE), forzamos valid_frame=False y registramos la razón del clasificador. 
        # Esto elimina la divergencia donde un frame puede ser label=OK y label_classifier=FAIL_*.
        # ---------------------------
        if valid_frame and label_classifier in _CLASSIFIER_FAIL_LABELS:
            valid_frame = False
            # Mapeo a fail_reason estilo snake_case coherente
            _clf_reason_map = {
                "FAIL_NO_DATA": "classifier_no_data",
                "FAIL_BAD_ICP": "classifier_bad_icp",
                "FAIL_POSE": "classifier_bad_pose",
            }
            fail_reason = _clf_reason_map.get(label_classifier, label_classifier.lower())
            fail_detail = f"label_classifier={label_classifier}"

        # ---------------------------
        # Label "oficial" 
        # ---------------------------
        if bool(valid_frame):
            label_status = "OK"
        else:
            label_status = f"FAIL_{fail_reason}"

        # ---------------------------
        # Row base (session-level)
        # ---------------------------
        row: Dict[str, Any] = {
            "session": args.session,
            "frame": fr,
            "side": args.side,
            "meta_path": str(meta_path) if meta_path else None,
            "pose_best_path": str(paths["pose_best"]),
            "seg_n_points": seg_n,
            "fitness": float(fitness) if fitness is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
            "score": float(score) if score is not None else None,
            "used_full_scene_fallback": bool(used_full_scene_fallback),
            "side_present": side_present,
            "center_distance_m": center_distance_m,
            "trans_error_m": float(trans_error_m) if trans_error_m is not None else None,
            "rot_error_deg": float(rot_error_deg) if rot_error_deg is not None else None,
            "delta_frame": delta_frame,
            "dtrans_error_m": float(dtrans_error_m) if dtrans_error_m is not None else None,
            "drot_error_deg": float(drot_error_deg) if drot_error_deg is not None else None,
            "label": label_status,
            "label_classifier": label_classifier,
            "valid_frame": bool(valid_frame),
            "fail_reason": fail_reason,
            "fail_detail": fail_detail,
            "has_raw_frame": bool(has_raw_frame),
            "has_segmented": bool(has_segmented),
            "has_aligned_model": bool(has_aligned_model),
            "has_eval_centers": bool(has_eval_centers),
            "se3_ok": se3.get("se3_ok"),
            "se3_reason": se3.get("se3_reason"),
            "det_R": se3.get("det_R"),
            "orth_err_fro": se3.get("orth_err_fro"),
        }

        per_frame_rows.append(row)

        # ---------------------------
        # pose_eval.json por frame (si --write-per-frame)
        # ---------------------------
        if args.write_per_frame:
            paths["ppf_dir"].mkdir(parents=True, exist_ok=True)
            write_json(
                paths["pose_eval"],
                {
                    "frame_index": int(fr),
                    "valid_frame": bool(valid_frame),
                    "fail_reason": fail_reason,
                    "fail_detail": fail_detail,
                    "has_raw_frame": bool(has_raw_frame),
                    "has_segmented": bool(has_segmented),
                    "used_full_scene_fallback": bool(used_full_scene_fallback),
                    "has_aligned_model": bool(has_aligned_model),
                    "has_eval_centers": bool(has_eval_centers),
                    "side_present": side_present,  # informativo
                    "seg_n_points": int(seg_n) if seg_n is not None else None,
                    "se3_ok": se3.get("se3_ok"),
                    "se3_reason": se3.get("se3_reason"),
                    "det_R": se3.get("det_R"),
                    "orth_err_fro": se3.get("orth_err_fro"),
                    "metrics": {
                        "fitness": float(fitness) if fitness is not None else None,
                        "rmse": float(rmse) if rmse is not None else None,
                        "score": float(score) if score is not None else None,
                        "center_distance_m": center_distance_m,
                        "trans_error_m": float(trans_error_m) if trans_error_m is not None else None,
                        "rot_error_deg": float(rot_error_deg) if rot_error_deg is not None else None,
                        "label": label_status,
                        "label_classifier": label_classifier,
                    },
                },
                indent=2,
            )

        # update prev si este frame tiene est+gt (para temporal consistente)
        if have_est and have_gt:
            prev_est = T_est
            prev_gt = T_gt
            prev_frame = fr

        # Log coherente: label_status (estado real) + classifier (diagnóstico)
        print(
            f"[FRAME {fr:06d}] label={label_status} (clf={label_classifier}) "
            f"valid={bool(valid_frame)} reason={fail_reason} "
            f"trans={row.get('trans_error_m', None)} rot={row.get('rot_error_deg', None)} "
            f"fit={row.get('fitness', None)} rmse={row.get('rmse', None)} seg_n={seg_n} "
            f"fallback={bool(used_full_scene_fallback)} se3_ok={se3.get('se3_ok')}"
        )

    # outputs por-frame (session-level)
    jsonl_path = outdir / "per_frame_metrics.jsonl"
    write_jsonl(jsonl_path, per_frame_rows)

    if args.write_csv:
        write_csv(outdir / "per_frame_metrics.csv", per_frame_rows)

    # summary
    trans_vals = [r["trans_error_m"] for r in per_frame_rows if r.get("trans_error_m") is not None]
    rot_vals = [r["rot_error_deg"] for r in per_frame_rows if r.get("rot_error_deg") is not None]
    dtrans_vals = [r["dtrans_error_m"] for r in per_frame_rows if r.get("dtrans_error_m") is not None]
    drot_vals = [r["drot_error_deg"] for r in per_frame_rows if r.get("drot_error_deg") is not None]
    fit_vals = [r["fitness"] for r in per_frame_rows if r.get("fitness") is not None]
    rmse_vals = [r["rmse"] for r in per_frame_rows if r.get("rmse") is not None]
    cd_vals = [r["center_distance_m"] for r in per_frame_rows if r.get("center_distance_m") is not None]

    summary: Dict[str, Any] = {
        "session": args.session,
        "side": args.side,
        "frames": frames,
        "thresholds": {
            "ok_trans_m": th.max_trans_m_ok,
            "ok_rot_deg": th.max_rot_deg_ok,
            "ok_min_fitness": th.min_fitness_ok,
            "ok_max_rmse": th.max_rmse_ok,
            "ok_min_seg_points": th.min_segment_points_ok,
        },
        "success": summarize_success(per_frame_rows),
        "pose_error_trans_m": summarize_numeric(trans_vals),
        "pose_error_rot_deg": summarize_numeric(rot_vals),
        "temporal_error_dtrans_m": summarize_numeric(dtrans_vals),
        "temporal_error_drot_deg": summarize_numeric(drot_vals),
        "fitness": summarize_numeric(fit_vals),
        "rmse": summarize_numeric(rmse_vals),
        "center_distance_m": summarize_numeric(cd_vals),
        "worst_frames_by_trans": top_k(per_frame_rows, "trans_error_m", k=10, descending=True),
        "worst_frames_by_rot": top_k(per_frame_rows, "rot_error_deg", k=10, descending=True),
        "worst_frames_by_rmse": top_k(per_frame_rows, "rmse", k=10, descending=True),
    }

    write_json(outdir / "summary.json", summary, indent=2)
    print(f"[OK] wrote: {jsonl_path}")
    if args.write_csv:
        print(f"[OK] wrote: {outdir / 'per_frame_metrics.csv'}")
    print(f"[OK] wrote: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()