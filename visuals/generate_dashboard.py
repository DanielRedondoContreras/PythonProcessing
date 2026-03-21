#!/usr/bin/env python3
"""
============================================================================
Archivo : visuals/generate_dashboard.py
Proyecto: TFG — Implementación de técnicas de visión artificial para la
          detección de objetos en contextos de realidad mixta

Propósito
---------
Este script genera un dashboard HTML interactivo a partir de los resultados producidos por el pipeline de estimación de pose 6-DOF.
El HTML resultante puede abrirse directamente en cualquier navegador sin necesidad de un servidor web.

Arquitectura del dashboard
--------------------------
Se optó por una implementación en JavaScript vanilla para garantizar:
  1. Funcionamiento offline — solo se requiere Three.js vía CDN para la vista 3D
  2. Portabilidad — un único archivo .html sin dependencias locales.
  3. Reproducibilidad — los datos de la sesión quedan embebidos en el HTML.

Flujo de generación
-------------------
  1. Se lee el CSV de trayectoria (trajectory_object_world.csv) producido por run_session_summary.py.
  2. Cada fila se convierte a un formato compacto (campos abreviados) para minimizar el tamaño del HTML resultante.
  3. Se lee la plantilla JavaScript (dashboard_app.js) y se sustituyen los placeholders con los datos serializados.
  4. Se ensambla el HTML final concatenando: CSS embebido, estructura HTML, CDN de Three.js, y el bloque <script> con el JavaScript inyectado.

Placeholders en la plantilla
-----------------------------
  __TRAJECTORY_JSON__  →  Array JSON compacto con los datos de trayectoria
  __SESSION_NAME__     →  Nombre de la sesión (p.ej. session_20260223_160956)
  __FRAME_MIN__        →  Índice del primer frame procesado
  __FRAME_MAX__        →  Índice del último frame procesado
  __FRAME_COUNT__      →  Número total de frames en la sesión

Formato compacto de cada fila de trayectoria
--------------------------------------------
  { f, v, r, tx, ty, tz, fit, rmse, sc, cd, se3 }

  f    = frame_index         (int)
  v    = valid_frame         (1/0)
  r    = fail_reason         (string)
  tx   = traslación X        (float, metros)
  ty   = traslación Y        (float, metros)
  tz   = traslación Z        (float, metros)
  fit  = fitness ICP         (float, [0,1])
  rmse = RMSE ICP            (float, metros)
  sc   = score combinado     (float, ≤0)
  cd   = center_distance_m   (float o null)
  se3  = se3_ok              (1/0)

Uso
---
  Desde la raíz del repositorio:
    python visuals/generate_dashboard.py --session session_XXXXXXXX_XXXXXX

  Salida por defecto:
    visuals/dashboard_<session>.html

Dependencias
------------
  - Python ≥ 3.8 (módulos estándar: argparse, csv, json, math, sys, pathlib)
  - Archivo de plantilla: visuals/dashboard_app.js
  - Datos de entrada: data/processed/<session>/trajectory_object_world.csv
============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Lectura y parseo del CSV de trayectoria
# ─────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> List[Dict[str, Any]]:
    """Lee un archivo CSV y convierte automáticamente los tipos de cada celda.

    Se aplica la siguiente jerarquía de conversión:
      vacío/None → None
      "True"/"true" → bool True
      "False"/"false" → bool False
      entero parseable → int
      float parseable (excluyendo NaN) → float
      resto → str

    Parámetros
    ----------
    path : Path
        Ruta al archivo CSV con cabecera.

    Retorna
    -------
    List[Dict[str, Any]]
        Lista de diccionarios, uno por fila del CSV.
    """
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                if v is None or v.strip() == "":
                    parsed[k] = None
                elif v in ("True", "true"):
                    parsed[k] = True
                elif v in ("False", "false"):
                    parsed[k] = False
                else:
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        try:
                            fv = float(v)
                            parsed[k] = None if math.isnan(fv) else fv
                        except ValueError:
                            parsed[k] = v
            rows.append(parsed)
    return rows


# ─────────────────────────────────────────────────────────────
# Conversión a formato compacto para inyección en JavaScript
# ─────────────────────────────────────────────────────────────

def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte una fila del CSV de trayectoria al formato compacto del dashboard.

    Se utilizan nombres de campo abreviados para reducir el tamaño del HTML generado, dado que los datos se embeben directamente en el archivo.

    Parámetros
    ----------
    row : Dict[str, Any]
        Fila del CSV con campos completos (frame_index, valid_frame, etc.).

    Retorna
    -------
    Dict[str, Any]
        Diccionario con campos compactos: f, v, r, tx, ty, tz, fit, rmse,
        sc, cd, se3.
    """
    def _n(val, default=0):
        """Convierte un valor a float, retornando default si es None o no finito."""
        if val is None:
            return default
        try:
            f = float(val)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    # center_distance_m requiere tratamiento especial: puede ser None cuando la segmentación falla y no se calcula distancia de centroide.
    cd_raw = row.get("center_distance_m")
    cd = None
    if cd_raw is not None:
        try:
            cd_f = float(cd_raw)
            cd = round(cd_f, 4) if math.isfinite(cd_f) else None
        except (TypeError, ValueError):
            cd = None

    return {
        "f":    int(row.get("frame_index", 0)),
        "v":    1 if row.get("valid_frame") else 0,
        "r":    row.get("fail_reason", "unknown") or "unknown",
        "tx":   round(_n(row.get("tx")), 4),
        "ty":   round(_n(row.get("ty")), 4),
        "tz":   round(_n(row.get("tz")), 4),
        "fit":  round(_n(row.get("fitness")), 4),
        "rmse": round(_n(row.get("rmse")), 5),
        "sc":   round(_n(row.get("score")), 4),
        "cd":   cd,
        "se3":  1 if row.get("se3_ok") else 0,
    }


def _to_js(obj: Any) -> str:
    """Serializa un objeto Python a JSON compatible con JavaScript."""
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)


# ─────────────────────────────────────────────────────────────
# CSS embebido del dashboard
# ─────────────────────────────────────────────────────────────
# Se utiliza CSS minificado para reducir el tamaño del HTML.
# Las clases siguen un esquema compacto de nombres:
#   .hdr = header, .ct = content, .sec = section, .crd = card,
#   .ab/.ai = arch box/inner, .pl = pill, .ph = phase, etc.

_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}\
body{background:#0a0f1e;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif}\
::-webkit-scrollbar{width:6px;height:6px}\
::-webkit-scrollbar-track{background:#111827}\
::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}\
.hdr{background:linear-gradient(135deg,#111827,#0a0f1e);border-bottom:1px solid #1e293b;padding:24px 32px 16px}\
.hdr h1{font-size:20px;font-weight:800;color:#f8fafc}\
.hdr .sub{color:#94a3b8;font-size:12px;margin-top:2px}.hdr .sub b{color:#06b6d4}\
.tabs{display:flex;gap:4px;margin-top:16px}\
.tb{background:0;border:1px solid transparent;border-radius:6px;padding:8px 16px;cursor:pointer;\
color:#94a3b8;font:500 13px/1 inherit;transition:.2s}\
.tb.on{background:#06b6d422;border-color:#06b6d455;color:#06b6d4;font-weight:700}\
.ct{padding:24px 32px;max-width:1100px;margin:0 auto}\
.sec{margin-bottom:28px}\
.sec h2{font-size:16px;font-weight:800;color:#f8fafc;margin-bottom:4px}\
.sec .d{font-size:12px;color:#94a3b8;margin-bottom:16px}\
.H{display:none!important}\
.ft{border-top:1px solid #1e293b;padding:16px 32px;text-align:center;color:#94a3b8;font-size:11px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}\
.g2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.crd{background:#111827;border-radius:8px;padding:16px;display:flex;justify-content:space-between;align-items:center}\
.crd .v{font-size:24px;font-weight:800;font-family:monospace}\
.crd .l{font-size:11px;color:#94a3b8;margin-bottom:4px}\
.crd .s{font-size:10px;color:#94a3b8;margin-top:2px}\
.crd .i{font-size:32px;opacity:.5}
.ab{background:#1a2238;border:1px solid #334155;border-radius:8px;padding:12px}\
.ai{background:#111827;border:1px solid #1e293b;border-radius:6px;padding:10px}\
.al{font-weight:700;font-size:12px;font-family:monospace;margin-bottom:6px}\
.as{border-left:3px solid;padding-left:12px;margin-bottom:10px}\
.as .h{color:#94a3b8;font-size:10px;margin-bottom:4px}\
.pl{display:inline-block;padding:2px 10px;border-radius:12px;font-size:10px;\
font-family:monospace;font-weight:600;margin:2px 3px;border:1px solid}
.ph{border-radius:8px;padding:12px;margin-bottom:8px}\
.ph .t{font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}\
.ph .ps{display:flex;flex-wrap:wrap;gap:4px}
.fb{border-radius:6px;padding:6px 14px;font-size:11px;font-family:monospace;\
font-weight:600;text-align:center;min-width:140px;display:inline-block}
.cp{background:#111827;border-radius:8px;padding:12px;border:1px solid #1e293b}\
.cp .n{font-family:monospace;font-weight:700;font-size:12px}\
.cp .cd{color:#94a3b8;font-size:11px;margin-top:4px;line-height:1.5}
.st{background:#111827;border-radius:8px;padding:14px}\
.st .sn{font-weight:700;font-size:13px;margin-bottom:8px}\
.st .sf{color:#94a3b8;font-size:11px;padding:3px 0;border-bottom:1px solid #1e293b;font-family:monospace}
.fr{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:16px}\
.fx{background:#111827;border-radius:8px;padding:12px;text-align:center}\
.fx .fn{font-weight:700;font-size:13px;font-family:monospace}\
.fx .fd{color:#94a3b8;font-size:10px;margin-top:6px}
.cb{background:#111827;border:1px solid #1e293b;border-radius:8px;padding:20px}\
.ce{font-family:monospace;font-size:13px;text-align:center;line-height:2.2}\
.cg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:16px}\
.ci{text-align:center;padding:8px;border-radius:6px}
.tw{overflow-x:auto;max-height:400px;overflow-y:auto}\
table{width:100%;border-collapse:collapse;font-size:10px}\
thead{position:sticky;top:0;background:#0a0f1e;z-index:1}\
th{padding:8px 4px;text-align:right;color:#06b6d4;font-weight:700;font-size:9px;text-transform:uppercase}\
td{padding:6px 4px;text-align:right;font-family:monospace;border-bottom:1px solid #1e293b}\
.bd{padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700}
.cc{background:#111827;border-radius:8px;padding:16px;border:1px solid #1e293b}\
.cc .ct2{font-weight:700;font-size:13px;margin-bottom:12px}
.mt{width:100%;border-collapse:collapse;font-size:11px}\
.mt th{padding:10px 6px;text-align:left;color:#06b6d4;font-weight:700;font-size:10px;\
text-transform:uppercase;border-bottom:2px solid #06b6d4}\
.mt td{padding:8px 6px;border-bottom:1px solid #1e293b}
.pf{display:flex;flex-direction:column;align-items:center;gap:4px;\
background:#111827;border:1px solid #1e293b;border-radius:8px;padding:16px}
.ci2{background:#111827;border-radius:8px;padding:12px}\
.ci2 .tg{padding:3px 10px;border-radius:4px;font-weight:700;font-size:12px;font-family:monospace}\
.ci2 .dd{color:#e2e8f0;font-size:11px;margin-top:8px;line-height:1.5}
#tc{width:100%;height:500px;border-radius:8px;overflow:hidden;border:1px solid #1e293b;background:#0a0f1e}
.lr{display:flex;gap:20px;margin-top:8px;justify-content:center}\
.lr .li{display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8}\
.lr .dt{width:10px;height:10px;border-radius:50%}
.zh{text-align:center;font-size:11px;color:#94a3b8;margin-top:4px}
svg.ch{width:100%;height:220px}svg.cht{width:100%;height:280px}"""


# ─────────────────────────────────────────────────────────────
# Función principal de generación
# ─────────────────────────────────────────────────────────────

def generate(session: str, repo_root: Path, output: Optional[Path] = None) -> Path:
    """Genera el dashboard HTML para una sesión concreta.

    Parámetros
    ----------
    session : str
        Nombre de la sesión (e.g. "session_XXXXXXXX_XXXXXX").
    repo_root : Path
        Directorio raíz del repositorio.
    output : Path, opcional
        Ruta de salida. Si no se especifica, se escribe en
        visuals/dashboard_<session>.html.

    Retorna
    -------
    Path
        Ruta del archivo HTML generado.
    """
    session_dir = repo_root / "data" / "processed" / session
    visuals_dir = repo_root / "visuals"
    template_path = visuals_dir / "dashboard_app.js"

    if not session_dir.exists():
        print(f"[ERROR] Directorio de sesión no encontrado: {session_dir}")
        sys.exit(1)
    if not template_path.exists():
        print(f"[ERROR] Plantilla JS no encontrada: {template_path}")
        sys.exit(1)

    # ── Lectura y conversión de la trayectoria ──
    trajectory = _read_csv(session_dir / "trajectory_object_world.csv")
    if not trajectory:
        print("[WARN] trajectory_object_world.csv vacío o no encontrado")

    compact = [_compact_row(r) for r in trajectory]

    frame_indices = [r["f"] for r in compact]
    frame_min = min(frame_indices) if frame_indices else 0
    frame_max = max(frame_indices) if frame_indices else 0
    frame_count = len(compact)

    # ── Serialización de datos para inyección en JS ──
    # Se formatea con una línea por objeto para facilitar la depuración
    traj_js = "[\n" + ",\n".join("  " + _to_js(row) for row in compact) + "\n]"

    # ── Lectura de la plantilla y sustitución de placeholders ──
    js_code = template_path.read_text(encoding="utf-8-sig")
    js_code = js_code.replace("__TRAJECTORY_JSON__", traj_js)
    js_code = js_code.replace("__SESSION_NAME__", session)
    js_code = js_code.replace("__FRAME_MIN__", str(frame_min))
    js_code = js_code.replace("__FRAME_MAX__", str(frame_max))
    js_code = js_code.replace("__FRAME_COUNT__", str(frame_count))

    # ── Ensamblaje del HTML ──
    parts = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html lang="es">')
    parts.append('<head>')
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append(f'<title>Dashboard \u2014 {session}</title>')
    parts.append(f'<style>\n{_CSS}\n</style>')
    parts.append('</head>')
    parts.append('<body>')

    # Cabecera con nombre de sesión y rango de frames
    parts.append('<div class="hdr">')
    parts.append('<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">')
    parts.append('<div style="background:linear-gradient(135deg,#06b6d4,#8b5cf6);'
                 'border-radius:8px;width:36px;height:36px;display:flex;'
                 'align-items:center;justify-content:center;font-size:18px">&#x1F4D0;</div>')
    parts.append('<div>')
    parts.append('<h1>Stereo Vision Pipeline &#8212; Documentaci\u00f3n T\u00e9cnica</h1>')
    parts.append(f'<div class="sub">Sesi\u00f3n: <b>{session}</b> &#183; '
                 f'Frames {frame_min}&#8211;{frame_max} ({frame_count} frames)</div>')
    parts.append('</div></div>')

    # Pestañas de navegación
    parts.append('<div class="tabs">')
    parts.append('<button class="tb on" onclick="T(\'a\')">&#x1F3D7;&#xFE0F; Arquitectura</button>')
    parts.append('<button class="tb" onclick="T(\'f\')">&#x1F504; Flujo de Datos</button>')
    parts.append('<button class="tb" onclick="T(\'m\')">&#x1F4CA; M\u00e9tricas</button>')
    parts.append('<button class="tb" onclick="T(\'v\')">&#x1F310; Visualizaci\u00f3n 3D</button>')
    parts.append('</div></div>')

    # Contenedores de contenido (uno por pestaña)
    parts.append('<div class="ct" id="ta"></div>')
    parts.append('<div class="ct H" id="tf"></div>')
    parts.append('<div class="ct H" id="tm"></div>')
    parts.append('<div class="ct H" id="tv"></div>')

    # Pie de página
    parts.append(f'<div class="ft">TFG: Visi\u00f3n Artificial para Estimaci\u00f3n de Pose '
                 f'en Realidad Mixta &#183; {session} &#183; '
                 f'Frames {frame_min}&#8211;{frame_max}</div>')

    # Three.js (única dependencia CDN — para la visualización 3D)
    parts.append('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>')

    # Bloque JavaScript con datos inyectados
    parts.append('<script>')
    parts.append(js_code)
    parts.append('</script>')
    parts.append('</body>')
    parts.append('</html>')

    html = "\n".join(parts)

    # ── Escritura del archivo de salida ──
    if output is None:
        output = visuals_dir / ("dashboard_" + session + ".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


# ─────────────────────────────────────────────────────────────
# Interfaz de línea de comandos
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un dashboard HTML interactivo para una sesión procesada.",
        epilog="Ejemplo: python visuals/generate_dashboard.py --session session_20260223_160956",
    )
    parser.add_argument("--session", required=True,
                        help="Nombre de la sesión (e.g. session_20260223_160956)")
    parser.add_argument("--output", default=None,
                        help="Ruta de salida (por defecto: visuals/dashboard_<session>.html)")
    args = parser.parse_args()

    # Detección automática de la raíz del repositorio
    repo_root = Path.cwd()
    if not (repo_root / "data").exists():
        script_dir = Path(__file__).resolve().parent
        if (script_dir.parent / "data").exists():
            repo_root = script_dir.parent

    print(f"[INFO] Repositorio: {repo_root}")
    print(f"[INFO] Sesión:      {args.session}")

    output_path = Path(args.output) if args.output else None
    result = generate(args.session, repo_root, output_path)

    print(f"\n[OK] Dashboard generado: {result}")
    print(f"     Abrir en navegador para visualizar.")


if __name__ == "__main__":
    main()