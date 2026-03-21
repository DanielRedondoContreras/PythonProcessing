# ============================================================
# File: src/pipeline/run_session_pipeline.py
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.io_utils import parse_frame_spec, safe_read_json
from src.core.reporting import write_json


# ---------------------------------------------------------------
# Carga de configuración YAML 
# ---------------------------------------------------------------

def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """
    Carga un archivo YAML de configuración.
    Devuelve dict vacío si path es None o el archivo no existe.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Archivo de configuración no encontrado: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def _compute_config_hash(path: str) -> Optional[str]:
    """
    Calcula el SHA-256 del contenido del archivo YAML.
    Devuelve None si el archivo no existe o no se puede leer.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        content = p.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return None


def _apply_yaml_defaults(parser: argparse.ArgumentParser, cfg: Dict[str, Any]) -> None:
    """
    Establece los defaults del parser a partir del dict YAML.

    Mapeo sección YAML → argumentos CLI:
      - Claves raíz       → --<clave>
      - disparity.*        → --<clave>  (no_clahe)
      - depth.*            → --<clave>  (min_disp, z_min, z_max)
      - pointcloud.voxel   → --pc-voxel
      - segmentation.*     → --<clave>  (roi_sx, seg_voxel, etc.)
      - segmentation.z_min → --seg-z-min   (Fase 2.2)
      - segmentation.z_max → --seg-z-max   (Fase 2.2)
      - segmentation.min_cluster_size → --min-cluster-size (Fase 4a)
      - matching.*         → --<clave>  (ppf_rel_sampling, icp_iterations, etc.)
      - matching.allow_full_scene_fallback → --allow-full-scene-fallback (Fase 4b)
      - pose_eval.*        → --ok-<clave>
      - stabilization.*    → --stab-<clave>  (Fase 5a)

    Solo se modifican defaults; los argumentos CLI explícitos siguen teniendo prioridad (argparse los resuelve después).
    """
    new_defaults: Dict[str, Any] = {}

    # --- Claves raíz directas ---
    for key in ("session", "frames", "side", "robust", "strict",
                "skip_disparity", "skip_depth", "skip_pointcloud",
                "skip_segmentation", "skip_ppf", "skip_per_frame_eval",
                "write_summary_csv"):
        if key in cfg:
            new_defaults[key.replace("-", "_")] = cfg[key]

    # --- disparity ---
    disp = cfg.get("disparity", {})
    if isinstance(disp, dict):
        if "no_clahe" in disp:
            new_defaults["no_clahe"] = disp["no_clahe"]

    # --- depth ---
    depth = cfg.get("depth", {})
    if isinstance(depth, dict):
        if "min_disp" in depth:
            new_defaults["min_disp"] = depth["min_disp"]
        if "z_min" in depth:
            new_defaults["z_min"] = depth["z_min"]
        if "z_max" in depth:
            new_defaults["z_max"] = depth["z_max"]

    # --- pointcloud ---
    pc = cfg.get("pointcloud", {})
    if isinstance(pc, dict):
        if "voxel" in pc:
            new_defaults["pc_voxel"] = pc["voxel"]

    # --- segmentation ---
    seg = cfg.get("segmentation", {})
    if isinstance(seg, dict):
        _seg_map = {
            "use_roi_gt": "use_roi_gt",
            "roi_sx": "roi_sx",
            "roi_sy": "roi_sy",
            "roi_sz": "roi_sz",
            "voxel": "seg_voxel",
            "outliers": "outliers",
            "remove_planes": "remove_planes",
            "estimate_normals": "estimate_normals",
            # propagación de z_min/z_max de segmentación
            "z_min": "seg_z_min",
            "z_max": "seg_z_max",
            # min_cluster_size
            "min_cluster_size": "min_cluster_size",
        }
        for yaml_key, cli_key in _seg_map.items():
            if yaml_key in seg:
                new_defaults[cli_key] = seg[yaml_key]

    # --- matching (PPF + ICP) ---
    match = cfg.get("matching", {})
    if isinstance(match, dict):
        for yaml_key in (
            "ppf_rel_sampling", "ppf_rel_distance",
            "ppf_scene_sample_step", "ppf_scene_distance", "ppf_top_n",
            "icp_iterations", "icp_tolerance", "icp_rejection_scale", "icp_num_levels",
            "crop_radius", "crop_min_points",
            "view_clamp", "view_near", "view_far", "use_left_camera",
            "density", "density_radius", "density_min_p", "density_max_p",
            "save_subclouds",
        ):
            if yaml_key in match:
                new_defaults[yaml_key.replace("-", "_")] = match[yaml_key]

        # allow_full_scene_fallback
        if "allow_full_scene_fallback" in match:
            new_defaults["allow_full_scene_fallback"] = match["allow_full_scene_fallback"]

    # --- pose_eval ---
    pe = cfg.get("pose_eval", {})
    if isinstance(pe, dict):
        _pe_map = {
            "ok_trans_m": "ok_trans_m",
            "ok_rot_deg": "ok_rot_deg",
            "ok_min_fitness": "ok_min_fitness",
            "ok_max_rmse": "ok_max_rmse",
            "ok_min_seg_points": "ok_min_seg_points",
        }
        for yaml_key, cli_key in _pe_map.items():
            if yaml_key in pe:
                new_defaults[cli_key] = pe[yaml_key]

    # --- Estabilización ---
    stab = cfg.get("stabilization", {})
    if isinstance(stab, dict):
        _stab_map = {
            "ema_alpha": "stab_ema_alpha",
            "max_trans_m": "stab_max_trans_m",
            "max_rot_deg": "stab_max_rot_deg",
        }
        for yaml_key, cli_key in _stab_map.items():
            if yaml_key in stab:
                new_defaults[cli_key] = stab[yaml_key]

    # Aplicar todos los defaults al parser
    if new_defaults:
        parser.set_defaults(**new_defaults)


# ---------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------

def _collect_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import numpy as np

        versions["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import open3d as o3d

        versions["open3d"] = o3d.__version__
    except Exception:
        pass
    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except Exception:
        pass
    return versions


def _run_module(module: str, args: List[str], *, robust: bool, warn_prefix: str) -> bool:
    """
    Ejecuta: python -m <module> <args>
    Devuelve True si OK, False si falla.
    En modo robust=True, NO lanza excepción (solo avisa).
    """
    cmd = [sys.executable, "-m", module] + args
    print("\n> " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"{warn_prefix} falló: {module}")
        if robust:
            return False
        raise


# ---------------------------------------------------------------
# config_diff: comparar parameters_final actual vs previo
# ---------------------------------------------------------------

def _compute_config_diff(
    current: Dict[str, Any],
    reference_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Compara el parameters_final.json actual contra la versión previa almacenada en ``reference_path``.

    Devuelve None si no existe referencia previa.
    Devuelve un dict con las diferencias:
      - "changed": dict de claves que cambiaron, con {old, new}
      - "added":   claves presentes en actual pero no en referencia
      - "removed": claves presentes en referencia pero no en actual
      - "identical": bool indicando si son idénticos

    Trabaja sobre una versión aplanada (dot-notation) de ambos dicts para facilitar la comparación de secciones anidadas.
    """
    ref_data = safe_read_json(reference_path)
    if ref_data is None:
        return None

    def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    flat_ref = _flatten(ref_data)
    flat_cur = _flatten(current)

    all_keys = set(flat_ref.keys()) | set(flat_cur.keys())

    changed: Dict[str, Dict[str, Any]] = {}
    added: List[str] = []
    removed: List[str] = []

    for key in sorted(all_keys):
        in_ref = key in flat_ref
        in_cur = key in flat_cur

        if in_ref and in_cur:
            # Comparar con tolerancia para floats
            v_ref = flat_ref[key]
            v_cur = flat_cur[key]
            if isinstance(v_ref, float) and isinstance(v_cur, float):
                if abs(v_ref - v_cur) > 1e-12:
                    changed[key] = {"old": v_ref, "new": v_cur}
            elif v_ref != v_cur:
                changed[key] = {"old": v_ref, "new": v_cur}
        elif in_cur and not in_ref:
            added.append(key)
        elif in_ref and not in_cur:
            removed.append(key)

    return {
        "identical": len(changed) == 0 and len(added) == 0 and len(removed) == 0,
        "changed": changed,
        "added": added,
        "removed": removed,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run session pipeline (end-to-end) + outputs Fase 3.6 (flags/reasons/trajectory/session summary)"
    )

    # --- Configuración por archivo ---
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta a archivo YAML de configuración (e.g. config/default_pipeline.yaml). "
             "Los valores del YAML actúan como defaults; los argumentos CLI tienen prioridad.",
    )

    parser.add_argument("--session", required=False, default=None)
    parser.add_argument("--frames", required=False, default=None, help='e.g. "0:115" or "100,105,110"')
    parser.add_argument("--side", default="right", choices=["right", "left"])

    # Comportamiento robusto (por defecto ON)
    parser.add_argument(
        "--robust",
        action="store_true",
        help="No aborta el barrido si falla una etapa (como tu PowerShell).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abortar en el primer error (equivalente a robust=False).",
    )

    # ------------------------------------------------------------
    # Etapas opcionales (si ya existen outputs, puedes saltarlas)
    # ------------------------------------------------------------
    parser.add_argument("--skip-disparity", action="store_true")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-pointcloud", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-ppf", action="store_true")
    parser.add_argument("--skip-per-frame-eval", action="store_true")
    parser.add_argument("--write-summary-csv", action="store_true")

    # ------------------------------------------------------------
    # Params DISPARIDAD 
    # ------------------------------------------------------------
    parser.add_argument("--no-clahe", action="store_true",
                        help="Desactiva CLAHE en SGBM/WLS (propagado a run_disparity).")

    # ------------------------------------------------------------
    # Params DEPTH 
    # ------------------------------------------------------------
    parser.add_argument("--min-disp", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=0.20)
    parser.add_argument("--z-max", type=float, default=2.0)

    # ------------------------------------------------------------
    # Params POINTCLOUD 
    # ------------------------------------------------------------
    parser.add_argument("--pc-voxel", type=float, default=0.002)

    # ------------------------------------------------------------
    # Params SEGMENTATION 
    # defaults alineados con YAML corregido
    # ------------------------------------------------------------
    parser.add_argument("--use-roi-gt", action="store_true")
    parser.add_argument("--roi-sx", type=float, default=0.30)
    parser.add_argument("--roi-sy", type=float, default=0.30)
    parser.add_argument("--roi-sz", type=float, default=0.30)
    parser.add_argument("--seg-voxel", type=float, default=0.0015)
    parser.add_argument("--outliers", type=str, default="statistical")    # Fase 2.3: "none" → "statistical"
    parser.add_argument("--remove-planes", type=int, default=0)           # YAML actual: 0
    parser.add_argument("--estimate-normals", action="store_true")

    # nuevos argumentos para banda Z de segmentación
    parser.add_argument("--seg-z-min", type=float, default=0.15,
                        help="Banda Z mínima para segmentación (m). Distinto de --z-min de depth.")
    parser.add_argument("--seg-z-max", type=float, default=2.0,
                        help="Banda Z máxima para segmentación (m). Distinto de --z-max de depth.")

    # min_cluster_size para select_cluster en segmentación
    parser.add_argument("--min-cluster-size", type=int, default=200,
                        help="Tamaño mínimo (puntos) de cluster elegible en nearest_center (Fase 4a)")

    # ------------------------------------------------------------
    # Params PPF + ICP MATCHING (OpenCV Surface Matching)
    # ------------------------------------------------------------
    # PPF detector
    parser.add_argument("--ppf-rel-sampling", type=float, default=0.05,
                        help="PPF: relativeSamplingStep (0.03–0.08)")
    parser.add_argument("--ppf-rel-distance", type=float, default=0.05,
                        help="PPF: relativeDistanceStep (0.03–0.08)")
    # PPF matching
    parser.add_argument("--ppf-scene-sample-step", type=float, default=0.025,
                        help="Fracción de puntos de escena usados (1/40=0.025)")
    parser.add_argument("--ppf-scene-distance", type=float, default=0.05)
    parser.add_argument("--ppf-top-n", type=int, default=5,
                        help="Top-N hipótesis PPF a refinar con ICP")
    # ICP refinement
    parser.add_argument("--icp-iterations", type=int, default=100)
    parser.add_argument("--icp-tolerance", type=float, default=0.005)
    parser.add_argument("--icp-rejection-scale", type=float, default=2.5)
    parser.add_argument("--icp-num-levels", type=int, default=4)
    # Scene pre-filtering
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
    parser.add_argument("--save-subclouds", action="store_true")

    # opt-in para permitir matching con escena completa (fallback)
    parser.add_argument(
        "--allow-full-scene-fallback",
        action="store_true",
        help="Permite que el matching use la nube completa si la segmentación falla.",
    )

    # ------------------------------------------------------------
    # Params POSE_EVAL thresholds
    # ------------------------------------------------------------
    parser.add_argument("--ok-trans-m", type=float, default=0.05)
    parser.add_argument("--ok-rot-deg", type=float, default=10.0)
    parser.add_argument("--ok-min-fitness", type=float, default=0.15)
    parser.add_argument("--ok-max-rmse", type=float, default=0.02)
    parser.add_argument("--ok-min-seg-points", type=int, default=300)

    # ------------------------------------------------------------
    # Params ESTABILIZACIÓN TEMPORAL
    # ------------------------------------------------------------
    parser.add_argument("--stab-ema-alpha", type=float, default=0.4,
                        help="Factor de suavizado EMA para estabilización temporal (0.01–1.0). "
                             "1.0 = sin suavizado (solo rechazo de saltos). Default: 0.4")
    parser.add_argument("--stab-max-trans-m", type=float, default=0.15,
                        help="Umbral máximo de traslación (m) entre frames consecutivos válidos "
                             "para rechazo de saltos. Default: 0.15")
    parser.add_argument("--stab-max-rot-deg", type=float, default=45.0,
                        help="Umbral máximo de rotación (°) entre frames consecutivos válidos "
                             "para rechazo de saltos. Default: 45.0")

    # ----------------------------------------------------------------
    # doble parseo para inyectar defaults desde YAML
    #   1) parse_known_args → obtener --config
    #   2) si hay YAML, inyectar sus valores como defaults
    #   3) parse_args → CLI explícito > YAML > hardcoded
    # ----------------------------------------------------------------
    pre_args, _ = parser.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)

    # ----------------------------------------------------------------
    # Guarda de trazabilidad
    # ----------------------------------------------------------------
    config_loaded = False
    config_hash: Optional[str] = None
    config_warning = False

    if pre_args.config is None:
        # No se proporcionó --config
        config_warning = True
        print(
            "[WARN] Ejecutando sin archivo de configuración. "
            "Se usarán defaults hardcoded. "
            "Usa --config para resultados reproducibles."
        )
        if pre_args.strict:
            parser.error(
                "Modo --strict activo y no se proporcionó --config. "
                "Proporciona un archivo de configuración YAML o desactiva --strict."
            )
    else:
        # Se proporcionó --config
        if cfg:
            # YAML cargado con éxito (dict no vacío)
            config_loaded = True
            config_hash = _compute_config_hash(pre_args.config)
            _apply_yaml_defaults(parser, cfg)
            print(f"[INFO] Configuración cargada desde: {pre_args.config}")
            if config_hash:
                print(f"[INFO] SHA-256 del config: {config_hash[:16]}...")
        else:
            # --config fue proporcionado pero el archivo no existe o su contenido es vacío/inválido. _load_yaml_config ya emitió un WARN. Registramos como no cargado.
            config_warning = True
            print(
                "[WARN] --config proporcionado pero no se pudo cargar. "
                "Se usarán defaults hardcoded."
            )
            if pre_args.strict:
                parser.error(
                    "Modo --strict activo y el archivo de configuración "
                    f"'{pre_args.config}' no se pudo cargar."
                )

    args = parser.parse_args()

    # --- Validación de argumentos obligatorios ---
    if not args.session:
        parser.error("--session es obligatorio (vía CLI o YAML)")
    if not args.frames:
        parser.error("--frames es obligatorio (vía CLI o YAML)")

    robust = bool(args.robust) and (not args.strict)

    session_dir = Path("data") / "processed" / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Leer parameters_final.json previo ANTES de sobrescribir, para calcular config_diff después.
    # ---------------------------------------------------------------
    prev_params_path = session_dir / "parameters_final.json"
    prev_params_exists = prev_params_path.exists()

    # -----------------------------
    # Reproducibilidad 
    # -----------------------------
    parameters_final = {
        "session": args.session,
        "frames": args.frames,
        "side": args.side,
        "config_file": pre_args.config,
        "config_loaded": config_loaded,
        "config_hash": config_hash,
        "config_warning": config_warning,
        "disparity": {"no_clahe": bool(args.no_clahe)},
        "depth": {"min_disp": args.min_disp, "z_min": args.z_min, "z_max": args.z_max},
        "pointcloud": {"voxel": args.pc_voxel},
        "segmentation": {
            "use_roi_gt": bool(args.use_roi_gt),
            "roi_sx": args.roi_sx,
            "roi_sy": args.roi_sy,
            "roi_sz": args.roi_sz,
            "voxel": args.seg_voxel,
            "outliers": args.outliers,
            "remove_planes": args.remove_planes,
            "estimate_normals": bool(args.estimate_normals),
            "z_min": args.seg_z_min,       
            "z_max": args.seg_z_max,       
            "min_cluster_size": args.min_cluster_size,  
        },
        "ppf_match": {
            "ppf_rel_sampling": args.ppf_rel_sampling,
            "ppf_rel_distance": args.ppf_rel_distance,
            "ppf_scene_sample_step": args.ppf_scene_sample_step,
            "ppf_scene_distance": args.ppf_scene_distance,
            "ppf_top_n": args.ppf_top_n,
            "icp_iterations": args.icp_iterations,
            "icp_tolerance": args.icp_tolerance,
            "icp_rejection_scale": args.icp_rejection_scale,
            "icp_num_levels": args.icp_num_levels,
            "crop_radius": args.crop_radius,
            "crop_min_points": args.crop_min_points,
            "view_clamp": bool(args.view_clamp),
            "view_near": args.view_near,
            "view_far": args.view_far,
            "use_left_camera": bool(args.use_left_camera),
            "density": bool(args.density),
            "density_radius": args.density_radius,
            "density_min_p": args.density_min_p,
            "density_max_p": args.density_max_p,
            "save_subclouds": bool(args.save_subclouds),
            "allow_full_scene_fallback": bool(args.allow_full_scene_fallback),
        },
        "pose_eval_thresholds": {
            "ok_trans_m": args.ok_trans_m,
            "ok_rot_deg": args.ok_rot_deg,
            "ok_min_fitness": args.ok_min_fitness,
            "ok_max_rmse": args.ok_max_rmse,
            "ok_min_seg_points": args.ok_min_seg_points,
        },
        # Parámetros de estabilización temporal
        "stabilization": {
            "ema_alpha": args.stab_ema_alpha,
            "max_trans_m": args.stab_max_trans_m,
            "max_rot_deg": args.stab_max_rot_deg,
        },
    }
    write_json(session_dir / "parameters_final.json", parameters_final, indent=2)

    # ---------------------------------------------------------------
    # config_diff en run_manifest.json
    # ---------------------------------------------------------------
    config_diff = None
    if prev_params_exists:
        config_diff = _compute_config_diff(parameters_final, prev_params_path)
        if config_diff is not None:
            if config_diff["identical"]:
                print("[INFO] config_diff: parámetros idénticos a la ejecución previa.")
            else:
                n_changed = len(config_diff.get("changed", {}))
                n_added = len(config_diff.get("added", []))
                n_removed = len(config_diff.get("removed", []))
                print(
                    f"[INFO] config_diff: {n_changed} cambiados, "
                    f"{n_added} añadidos, {n_removed} eliminados vs. ejecución previa."
                )
                for key, vals in config_diff.get("changed", {}).items():
                    print(f"  Δ {key}: {vals['old']} → {vals['new']}")

    run_manifest = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "command": " ".join(sys.argv),
        "versions": _collect_versions(),
        # diff contra la ejecución previa
        "config_diff": config_diff,
    }
    write_json(session_dir / "run_manifest.json", run_manifest, indent=2)

    frames = parse_frame_spec(args.frames)

    # -----------------------------
    # Per-frame pipeline (robusto)
    # -----------------------------
    for f in frames:
        f6 = f"{f:06d}"

        print("\n===============================")
        print(f"FRAME {f}")
        print("===============================")

        # 0) Si no existe el frame RAW, saltar (igual que tu PowerShell)
        raw_dir = Path("data") / "raw" / args.session / f"frame_{f6}"
        if not raw_dir.exists():
            print(f"[SKIP] No existe: {raw_dir}")
            continue

        # 1) DISPARIDAD (propaga --no-clahe)
        if not args.skip_disparity:
            disp_args = ["--session", args.session, "--frame", str(f)]
            if args.no_clahe:
                disp_args += ["--no-clahe"]
            ok = _run_module(
                "src.pipeline.run_disparity",
                disp_args,
                robust=robust,
                warn_prefix=f"[WARN][FRAME {f}] run_disparity",
            )
            if (not ok) and robust:
                continue

        # 2) PROFUNDIDAD
        if not args.skip_depth:
            ok = _run_module(
                "src.pipeline.run_depth",
                [
                    "--session",
                    args.session,
                    "--frame",
                    str(f),
                    "--min-disp",
                    str(args.min_disp),
                    "--z-min",
                    str(args.z_min),
                    "--z-max",
                    str(args.z_max),
                ],
                robust=robust,
                warn_prefix=f"[WARN][FRAME {f}] run_depth",
            )
            if (not ok) and robust:
                continue

        # 3) NUBE DE PUNTOS
        if not args.skip_pointcloud:
            ok = _run_module(
                "src.pipeline.run_pointcloud",
                ["--session", args.session, "--frame", str(f), "--voxel", str(args.pc_voxel)],
                robust=robust,
                warn_prefix=f"[WARN][FRAME {f}] run_pointcloud",
            )
            if (not ok) and robust:
                continue

        # 4) SEGMENTACIÓN
        # propaga --z-min / --z-max con los valores de seg_z_min / seg_z_max
        # propaga --min-cluster-size
        if not args.skip_segmentation:
            in_ply = Path("data") / "processed" / args.session / f"frame_{f6}" / "pointcloud_world_unity.ply"
            meta = Path("data") / "raw" / args.session / f"frame_{f6}" / "metadata.json"
            out_ply = Path("data") / "processed" / args.session / f"frame_{f6}" / "object_segmented.ply"

            seg_args = [
                "--in-ply",
                str(in_ply),
                "--metadata",
                str(meta),
                "--out-ply",
                str(out_ply),
                "--side",
                args.side,
                "--roi-sx",
                str(args.roi_sx),
                "--roi-sy",
                str(args.roi_sy),
                "--roi-sz",
                str(args.roi_sz),
                "--voxel",
                str(args.seg_voxel),
                "--outliers",
                str(args.outliers),
                "--remove-planes",
                str(args.remove_planes),
                # banda Z de segmentación
                "--z-min",
                str(args.seg_z_min),
                "--z-max",
                str(args.seg_z_max),
                # min_cluster_size
                "--min-cluster-size",
                str(args.min_cluster_size),
            ]
            if args.use_roi_gt:
                seg_args += ["--use-roi-gt"]
            if args.estimate_normals:
                seg_args += ["--estimate-normals"]

            ok = _run_module(
                "src.pipeline.run_segmentation",
                seg_args,
                robust=robust,
                warn_prefix=f"[WARN][FRAME {f}] run_segmentation",
            )
            # el resultado de segmentación se evalúa en el gate a continuación. Si falla y no hay fallback, saltamos matching.
            if (not ok) and (not robust):
                pass

        # ---------------------------------------------------------------
        # Gate pre-matching
        #
        # Tras la segmentación, lee segmentation_metrics.json y verifica:
        #   (a) ok == true
        #   (b) n_out >= ok_min_seg_points
        # Si no se cumple y --allow-full-scene-fallback NO está activo, registra el frame como fallido y salta directamente a evaluación sin ejecutar matching. Esto evita matching costoso sobre nubes inválidas y elimina los sentinels como mecanismo de señalización.
        # ---------------------------------------------------------------
        seg_gate_passed = True  # por defecto asumimos OK (skip_segmentation → no gate)

        if not args.skip_segmentation:
            seg_metrics_path = (
                Path("data") / "processed" / args.session / f"frame_{f6}" / "segmentation_metrics.json"
            )
            seg_met = safe_read_json(seg_metrics_path)

            if seg_met is None:
                seg_gate_passed = False
                print(f"[GATE][FRAME {f}] No segmentation_metrics.json — segmentación probablemente falló")
            else:
                seg_ok = bool(seg_met.get("ok", False))
                seg_n_out = 0
                try:
                    seg_n_out = int(seg_met.get("n_out", 0) or 0)
                except (TypeError, ValueError):
                    seg_n_out = 0

                if not seg_ok or seg_n_out < args.ok_min_seg_points:
                    seg_gate_passed = False
                    print(
                        f"[GATE][FRAME {f}] Segmentación insuficiente: "
                        f"ok={seg_ok} n_out={seg_n_out} (min={args.ok_min_seg_points})"
                    )

        # 5) MATCHING 
        if not args.skip_ppf:
            if not seg_gate_passed and not args.allow_full_scene_fallback:
                print(
                    f"[GATE][FRAME {f}] Matching omitido — segmentación insuficiente y "
                    f"--allow-full-scene-fallback no activo. Saltando a evaluación."
                )
            else:
                if not seg_gate_passed:
                    print(
                        f"[GATE][FRAME {f}] Segmentación insuficiente pero "
                        f"--allow-full-scene-fallback activo — ejecutando matching con posible fallback."
                    )

                ppf_args = [
                    "--session",
                    args.session,
                    "--frame",
                    str(f),
                    "--side",
                    args.side,
                    # PPF detector
                    "--ppf-rel-sampling",
                    str(args.ppf_rel_sampling),
                    "--ppf-rel-distance",
                    str(args.ppf_rel_distance),
                    # PPF matching
                    "--ppf-scene-sample-step",
                    str(args.ppf_scene_sample_step),
                    "--ppf-scene-distance",
                    str(args.ppf_scene_distance),
                    "--ppf-top-n",
                    str(args.ppf_top_n),
                    # ICP
                    "--icp-iterations",
                    str(args.icp_iterations),
                    "--icp-tolerance",
                    str(args.icp_tolerance),
                    "--icp-rejection-scale",
                    str(args.icp_rejection_scale),
                    "--icp-num-levels",
                    str(args.icp_num_levels),
                    # Scene pre-filtering
                    "--crop-radius",
                    str(args.crop_radius),
                    "--crop-min-points",
                    str(args.crop_min_points),
                    "--view-near",
                    str(args.view_near),
                    "--view-far",
                    str(args.view_far),
                    "--density-radius",
                    str(args.density_radius),
                    "--density-min-p",
                    str(args.density_min_p),
                    "--density-max-p",
                    str(args.density_max_p),
                ]
                if args.view_clamp:
                    ppf_args += ["--view-clamp"]
                if args.use_left_camera:
                    ppf_args += ["--use-left-camera"]
                if args.density:
                    ppf_args += ["--density"]
                if args.save_subclouds:
                    ppf_args += ["--save-subclouds"]

                ok = _run_module(
                    "src.pipeline.run_ppf_match",
                    ppf_args,
                    robust=robust,
                    warn_prefix=f"[WARN][FRAME {f}] run_ppf_match",
                )
                if (not ok) and robust:
                    continue

        # 6) EVALUACIÓN per-frame (pose_eval.json + eval_centers.json si aplica)
        if not args.skip_per_frame_eval:
            ok = _run_module(
                "src.pipeline.run_pose_eval",
                [
                    "--session",
                    args.session,
                    "--frames",
                    str(f),
                    "--side",
                    args.side,
                    "--write-per-frame",
                    "--ok-trans-m",
                    str(args.ok_trans_m),
                    "--ok-rot-deg",
                    str(args.ok_rot_deg),
                    "--ok-min-fitness",
                    str(args.ok_min_fitness),
                    "--ok-max-rmse",
                    str(args.ok_max_rmse),
                    "--ok-min-seg-points",
                    str(args.ok_min_seg_points),
                ],
                robust=robust,
                warn_prefix=f"[WARN][FRAME {f}] run_pose_eval",
            )
            if (not ok) and robust:
                continue

    # 7) Session-level summary + trajectory + estabilización
    sum_args = [
        "--session", args.session,
        # parámetros de estabilización
        "--stab-ema-alpha", str(args.stab_ema_alpha),
        "--stab-max-trans-m", str(args.stab_max_trans_m),
        "--stab-max-rot-deg", str(args.stab_max_rot_deg),
    ]
    if args.write_summary_csv:
        sum_args += ["--write-csv"]

    _run_module(
        "src.pipeline.run_session_summary",
        sum_args,
        robust=False,  # aquí interesa fallar si no se genera el resumen
        warn_prefix="[WARN] run_session_summary",
    )

    print("\n[OK] Session pipeline completed.")
    print(f" - {session_dir / 'parameters_final.json'}")
    print(f" - {session_dir / 'run_manifest.json'}")
    print(f" - {session_dir / 'summary_frames.json'}")
    print(f" - {session_dir / 'session_summary.json'}")
    print(f" - {session_dir / 'trajectory_object_world.csv'}")
    print(f" - {session_dir / 'trajectory_stabilized.csv'}")


if __name__ == "__main__":
    main()