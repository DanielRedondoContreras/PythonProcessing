# ============================================================
# File: src/core/io_utils.py
# IO unificado de JSON + validación de archivos
# parse_frame_spec centralizado
#
# Centraliza:
#   - read_json()        
#   - safe_read_json()   
#   - require_file()     
#   - parse_frame_spec() 
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    """Lectura estándar de un archivo JSON. Lanza excepción si no existe o es inválido."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Lectura con fallback a None si el archivo no existe o falla el parseo."""
    try:
        if not path.exists():
            return None
        return read_json(path)
    except Exception:
        return None


def require_file(path: Path, name: str) -> Path:
    """Valida que un archivo existe. Lanza FileNotFoundError con mensaje descriptivo si no."""
    if not path.exists():
        raise FileNotFoundError(f"Falta el archivo requerido '{name}' en: {path}")
    return path


# ---------------------------------------------------------------
# parse_frame_spec centralizado
# ---------------------------------------------------------------

def parse_frame_spec(spec: str) -> List[int]:
    """
    Parsea una especificación de frames desde CLI.

    Formatos aceptados:
      - "110"            → [110]
      - "100:130"        → rango inclusivo [100, 130]
      - "100:130:2"      → rango con step
      - "100,105,110"    → lista explícita
      - combinaciones:   "0:10,50,80:90"

    Devuelve
    --------
    Lista ordenada de índices de frame (sin duplicados).
    """
    spec = spec.strip()
    if "," in spec:
        out: List[int] = []
        for s in spec.split(","):
            s = s.strip()
            if not s:
                continue
            out.extend(parse_frame_spec(s))
        return sorted(set(out))

    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("Rango de frames inválido. Utiliza start:end o start:end:step")
        a = int(parts[0])
        b = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise ValueError("step no puede ser 0")
        if b < a and step > 0:
            step = -step
        # Inclusivo
        return list(range(a, b + (1 if step > 0 else -1), step))

    return [int(spec)]
