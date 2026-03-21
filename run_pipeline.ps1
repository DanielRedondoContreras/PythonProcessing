# ================================================================
#  PIPELINE COMPLETO - Vision Artificial MR
#  Ejecutar: .\run_pipeline.ps1
# ================================================================
#
#  +----------------------------------------------------------+
#  |           ZONA DE CONFIGURACION DE USUARIO               |
#  |              (Modificar según la sesión)                  |
#  +----------------------------------------------------------+

$SESSION = "session_20260223_160956"   # <-- NOMBRE DE LA SESION
$FRAMES  = "88:115"                     # <-- RANGO DE FRAMES (p.ej. "0:115", "0,5,10")
$SIDE    = "right"                     # <-- CONTROLADOR ("right" o "left")

# ================================================================

# Activar entorno virtual si existe
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

#--config 
python -m src.pipeline.run_session_pipeline `
    --config config\default_pipeline.yaml `
    --session $SESSION `
    --frames  $FRAMES `
    --side    $SIDE `
    --robust

# ================================================================
#  DASHBOARD INTERACTIVO
#
#  Genera un archivo HTML autocontenido con visualizacion interactiva de la sesion (arquitectura, flujo, metricas, 3D).
#
#  Requiere:
#    - visuals/dashboard_app.js  (plantilla JavaScript)
#    - visuals/generate_dashboard.py  (generador)
#    - data/processed/<session>/trajectory_object_world.csv
#
#  Salida:
#    - visuals/dashboard_<session>.html
#
#  Para generar manualmente el dashboard de una sesion concreta:
#    python visuals/generate_dashboard.py --session session_20260223_160956
# ================================================================

Write-Host "`n[DASHBOARD] Generando dashboard interactivo..."
python visuals/generate_dashboard.py --session $SESSION