# ============================================================
# File: src/core/reporting.py
# Escritura de métricas, resúmenes y archivos de salida
#
# Proporciona:
#   - Escritura de JSON, JSONL y CSV
#   - Funciones de resumen estadístico (media, mediana, percentiles)
#   - Agregación de tasas de éxito por label
#   - Selección de peores frames (top-k)
# ============================================================

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def ensure_dir(path: Path) -> None:
    """Crea el directorio (y padres) si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Escribe un objeto Python como JSON con indentación configurable."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Escribe una secuencia de dicts como archivo JSONL (una línea JSON por fila)."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """Escribe una lista de dicts como archivo CSV con cabecera automática."""
    ensure_dir(path.parent)
    if not rows:
        # Escribe cabecera mínima si se especifica
        with path.open("w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return

    if fieldnames is None:
        # Union ordenada: keys de la primera fila + el resto en orden de aparición
        seen = set()
        fieldnames = []
        for k in rows[0].keys():
            seen.add(k)
            fieldnames.append(k)
        for r in rows[1:]:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _nanpercentile(x: np.ndarray, q: float) -> float:
    x = x.astype(np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _nanmean(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _nanmedian(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    """Calcula estadísticas descriptivas (media, mediana, percentiles, min, max) sobre una lista de valores numéricos."""
    a = np.array(values, dtype=np.float64)
    return {
        "mean": _nanmean(a),
        "median": _nanmedian(a),
        "p90": _nanpercentile(a, 90),
        "p95": _nanpercentile(a, 95),
        "p99": _nanpercentile(a, 99),
        "min": float(np.nanmin(a)) if np.any(~np.isnan(a)) else float("nan"),
        "max": float(np.nanmax(a)) if np.any(~np.isnan(a)) else float("nan"),
        "count": float(np.sum(~np.isnan(a))),
    }


def summarize_success(rows: List[Dict[str, Any]], label_key: str = "label") -> Dict[str, Any]:
    """Contabiliza frames por label y calcula la tasa de éxito (ok_rate)."""
    total = len(rows)
    counts: Dict[str, int] = {}
    for r in rows:
        lab = str(r.get(label_key, "UNKNOWN"))
        counts[lab] = counts.get(lab, 0) + 1

    ok = counts.get("OK", 0)
    return {
        "n_frames": total,
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "ok_rate": (ok / total) if total > 0 else 0.0,
    }


def top_k(rows: List[Dict[str, Any]], key: str, k: int = 10, descending: bool = True) -> List[Dict[str, Any]]:
    """Devuelve los k frames con mayor (o menor) valor en la clave indicada, ignorando NaN."""
    def getv(r: Dict[str, Any]) -> float:
        v = r.get(key, float("nan"))
        try:
            return float(v)
        except Exception:
            return float("nan")

    # filtra NaN
    filtered = []
    for r in rows:
        v = getv(r)
        if not np.isnan(v):
            filtered.append((v, r))

    filtered.sort(key=lambda t: t[0], reverse=descending)
    return [r for _, r in filtered[:k]]