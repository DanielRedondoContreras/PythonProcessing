# TFG: Implementación de técnicas de visión artificial para la detección de objetos en contextos de realidad mixta

## Descripción

Pipeline offline en Python para la estimación de pose 6-DOF de un controlador Meta Quest Touch Plus a partir de vídeo estereoscópico capturado mediante la Passthrough Camera API del visor Meta Quest 3.

El sistema reconstruye nubes de puntos 3D desde pares estéreo, segmenta el controlador de la escena, y registra un modelo CAD contra la nube segmentada mediante Point Pair Feature (PPF) voting + refinamiento ICP (OpenCV Surface Matching, Drost et al. 2010). La pose estimada se compara frame a frame contra el ground truth proporcionado por el SDK del dispositivo (tracking nativo), generando métricas de error en traslación y rotación, calidad de registro y consistencia temporal.

## Tecnologías

| Componente | Tecnología |
|---|---|
| Hardware de captura | Meta Quest 3 |
| Motor de captura | Unity (C#) |
| Procesado offline | Python ≥ 3.10 |
| Visión estéreo | OpenCV (opencv-contrib-python) |
| Registro 3D / Pose | OpenCV Surface Matching (PPF + ICP) |
| Geometría 3D | Open3D |
| Álgebra SE(3) | NumPy |
| Configuración | PyYAML |
| Modelo CAD | Blender → exportación PLY |
| Visualización | Dashboard HTML autocontenido (JavaScript vanilla + Recharts vía CDN) |

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/Python_Processing.git
cd Python_Processing

# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Requisitos previos

- Python ≥ 3.10
- El paquete `opencv-contrib-python` es necesario (no basta `opencv-python`) porque el pipeline usa `cv2.ximgproc` para el filtro WLS de disparidad.

## Datos de entrada

El pipeline espera sesiones capturadas con la siguiente estructura:

```
data/raw/session_XXXXXXXX_XXXXXX/
├── frame_000000/
│   ├── left.png          # Imagen izquierda (grayscale uint8)
│   ├── right.png         # Imagen derecha (grayscale uint8)
│   └── metadata.json     # Intrínsecos, poses de cámaras y controladores, timestamp
├── frame_000001/
│   └── ...
└── ...
```

Además, el modelo CAD del controlador debe estar preparado en:

```
data/cad/
├── right_controller_ready.ply     # Nube de puntos del modelo CAD preparada
└── ppf/
    ├── right_controller_ppf.npz   # Modelo PPF pre-entrenado
    └── right_controller_ppf_meta.json
```

## Ejecución

### Pipeline completo (una sesión)

```powershell
# Editar sesión, rango de frames y lado en run_pipeline.ps1
.\run_pipeline.ps1
```

O directamente:

```powershell
python -m src.pipeline.run_session_pipeline `
    --config config\default_pipeline.yaml `
    --session session_20260223_160956 `
    --frames "88:115" `
    --side right `
    --robust
```

### Preparación del modelo CAD

```powershell
# Preparar nube de puntos desde malla PLY
python -m src.pipeline.run_cad_model --mesh data/cad/right_controller.ply
```

El entrenamiento del modelo PPF se realiza automáticamente en cada ejecución del matching (inline en `PPFMatcher.run()`), por lo que no requiere un paso previo independiente.

### Dashboard de resultados

```powershell
python visuals/generate_dashboard.py --session session_20260223_160956
# Genera: visuals/dashboard_session_20260223_160956.html
```

## Pipeline: etapas

El pipeline procesa cada frame secuencialmente a través de seis fases, orquestadas por `run_session_pipeline.py`. Cada fase es también ejecutable de forma independiente.

### Fase 1 — Rectificación estéreo

**Módulo:** `src/core/rectification.py` → `src/pipeline/run_disparity.py`

Calcula mapas de rectificación estéreo a partir de los intrínsecos y poses de ambas cámaras (extraídos de `metadata.json`). Aplica el cambio de base Unity→OpenCV (inversión de eje Y). Genera imágenes rectificadas alineadas epipolármente.

### Fase 2 — Disparidad

**Módulo:** `src/core/disparity.py` → `src/pipeline/run_disparity.py`

Calcula el mapa de disparidad mediante SGBM (Semi-Global Block Matching). Opcionalmente aplica CLAHE como preproceso para mejorar textura en escenas con baja iluminación. Aplica filtro WLS (Weighted Least Squares) usando `cv2.ximgproc` para suavizado con preservación de bordes.

### Fase 3 — Profundidad

**Módulo:** `src/core/depth.py` → `src/pipeline/run_depth.py`

Convierte disparidad a profundidad métrica (metros) usando la focal rectificada y la baseline estéreo. Aplica filtro por rango válido (`z_min`–`z_max`) y genera máscara de validez.

### Fase 4 — Nube de puntos

**Módulo:** `src/core/pointcloud.py` → `src/pipeline/run_pointcloud.py`

Backproyecta el mapa de profundidad a coordenadas 3D de cámara (frame OpenCV) y transforma a coordenadas mundo (frame Unity) usando la pose del casco. Aplica downsampling por voxel grid. Genera nubes en formato PLY y NPY.

### Fase 5 — Segmentación

**Módulo:** `src/core/segmentation.py` → `src/pipeline/run_segmentation.py`

Aísla el controlador de la escena mediante:
1. Recorte por ROI orientada (OBB) centrada en la pose GT del controlador (cuando `use_roi_gt: true`).
2. Filtrado por banda Z.
3. Eliminación de outliers (statistical o radius).
4. Clustering DBSCAN y selección del cluster más cercano al centro esperado, con tamaño mínimo configurable.
5. Estimación de normales sobre la nube segmentada final.

### Fase 6 — Registro y estimación de pose

**Módulo:** `src/core/ppf_match.py` → `src/pipeline/run_ppf_match.py`

Registra el modelo CAD contra la nube segmentada para estimar la transformación SE(3) del controlador, implementando el pipeline OpenCV Surface Matching (Drost et al. 2010):

1. **Pre-filtrado de escena:** Crop esférico por radio alrededor del centroide, view clamp por eje de cámara (near/far), y filtro de densidad local por percentil.
2. **Entrenamiento PPF:** Construye una hash table de Point Pair Features (distancia + 4 ángulos entre normales) sobre todos los pares de puntos del modelo CAD (`cv2.ppf_match_3d_PPF3DDetector.trainModel`).
3. **Matching PPF:** Para cada par de puntos de la escena, busca pares similares en la hash table del modelo y vota en un espacio de poses 2D. Las hipótesis se agrupan por clustering y se seleccionan las top-N por número de votos (`detector.match`).
4. **Refinamiento ICP:** ICP point-to-plane multi-nivel sobre las top-N hipótesis del votador PPF (`cv2.ppf_match_3d_ICP.registerModelToScene`).
5. **Selección del mejor candidato:** La pose con mayor número de votos y menor residual tras ICP.
6. **Evaluación de calidad:** Fitness y RMSE calculados con `evaluate_registration` de Open3D para compatibilidad con los umbrales del pipeline de evaluación.

### Evaluación per-frame

**Módulo:** `src/core/pose_eval.py` + `src/core/frame_status.py` → `src/pipeline/run_pose_eval.py`

Para cada frame computa:
- Error de traslación (m) y rotación geodésica (°) respecto al ground truth.
- Fitness y RMSE del registro ICP.
- Distancia entre centroides (modelo alineado vs. escena segmentada).
- Clasificación del frame: `OK`, `FAIL_NO_DATA`, `FAIL_BAD_ICP`, `FAIL_POSE`.
- Sanity checks SE(3): ortonormalidad de la rotación estimada.

### Agregación y estabilización temporal

**Módulo:** `src/core/temporal_stabilizer.py` → `src/pipeline/run_session_summary.py`

Tras procesar todos los frames:
1. Genera trayectoria cruda (`trajectory_object_world.csv`).
2. Aplica estabilización temporal:
   - **Rechazo de saltos:** Frames donde el desplazamiento entre consecutivos excede umbrales físicamente plausibles se marcan como rechazados.
   - **Filtro EMA:** Media móvil exponencial sobre traslación y SLERP-EMA sobre rotación.
3. Genera trayectoria estabilizada (`trajectory_stabilized.csv`).
4. Genera resumen de sesión (`session_summary.json`) con tasas de éxito, estadísticas agregadas y parámetros efectivos.

### Dashboard

**Módulo:** `visuals/generate_dashboard.py` + `visuals/dashboard_app.js`

Genera un archivo HTML autocontenido e interactivo con:
- Esquema del pipeline con diagrama de flujo.
- Gráficos de error de traslación y rotación por frame.
- Gráfico de métricas de calidad (fitness, RMSE).
- Visualización 3D de la trayectoria estimada vs. ground truth.
- Tabla de clasificación por frame con diagnóstico detallado.

## Configuración

Toda la experimentación se controla desde `config/default_pipeline.yaml`, que es la fuente única de verdad para los parámetros del pipeline. El sistema de prioridad de resolución es:

```
CLI explícito > YAML > defaults hardcoded del script
```

Cada ejecución registra en `parameters_final.json` los parámetros efectivos utilizados (incluyendo hash SHA-256 del YAML). El archivo `run_manifest.json` incluye un `config_diff` contra la ejecución previa.

## Marcos de referencia

| Marco | Descripción |
|---|---|
| `W` | Mundo Unity (Y-up, left-handed) |
| `C_left` | Cámara izquierda original |
| `C_left_rect` | Cámara izquierda rectificada (OpenCV: Y-down, Z-forward) |
| `M` | Modelo CAD |
| `Controller_GT` | Frame del controlador según tracking nativo del SDK |

Las transformaciones entre marcos se componen en SE(3). El cambio de base Unity↔OpenCV se aplica mediante `UNITY_TO_OPENCV_BASIS = diag(1, -1, 1)`. Los quaterniones siguen la convención `[x, y, z, w]` (Unity/Hamilton).

## Estructura del repositorio

```
Python_Processing/
├── config/
│   └── default_pipeline.yaml          # Configuración centralizada
├── data/
│   ├── cad/                           # Modelo CAD y PPF pre-entrenado
│   ├── processed/                     # Salidas del pipeline por sesión
│   └── raw/                           # Sesiones capturadas (imágenes + metadata)
├── src/
│   ├── core/                          # Módulos funcionales
│   │   ├── cad_model.py               # Preparación del modelo CAD
│   │   ├── data_loader.py             # Carga y validación de datos de entrada
│   │   ├── depth.py                   # Conversión disparidad → profundidad
│   │   ├── disparity.py               # Cálculo de disparidad (SGBM + WLS)
│   │   ├── frame_status.py            # Diagnóstico y clasificación de frames
│   │   ├── io_utils.py                # Utilidades de E/S (JSON, parseo de frames)
│   │   ├── pointcloud.py              # Backproyección y nube de puntos
│   │   ├── pose_eval.py               # Evaluación de pose contra GT
│   │   ├── ppf_match.py               # Estimación de pose 6-DOF (PPF + ICP, OpenCV Surface Matching)
│   │   ├── rectification.py           # Rectificación estéreo
│   │   ├── reporting.py               # Escritura de métricas y resúmenes
│   │   ├── segmentation.py            # Segmentación espacial del controlador
│   │   ├── temporal_stabilizer.py     # Estabilización temporal (EMA + SLERP)
│   │   └── transforms.py             # Matemática SE(3)/SO(3) centralizada
│   └── pipeline/                      # Orquestadores (CLI) por fase
│       ├── run_cad_model.py           # Preparación del modelo CAD
│       ├── run_depth.py               # Fase 3: profundidad
│       ├── run_disparity.py           # Fases 1-2: rectificación + disparidad
│       ├── run_pointcloud.py          # Fase 4: nube de puntos
│       ├── run_pose_eval.py           # Evaluación per-frame + agregado
│       ├── run_ppf_match.py           # Fase 6: matching + registro
│       ├── run_segmentation.py        # Fase 5: segmentación
│       ├── run_session_pipeline.py    # Orquestador principal de sesión
│       └── run_session_summary.py     # Resumen + trayectoria + estabilización
├── visuals/
│   ├── dashboard_app.js               # Plantilla JavaScript del dashboard
│   └── generate_dashboard.py          # Generador de dashboard HTML
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.ps1                   # Script de ejecución principal (PowerShell)
```

## Métricas reportadas

### Per-frame

- Error de traslación (m)
- Error de rotación geodésica (°)
- Fitness ICP (proporción de inliers)
- RMSE ICP (m)
- Distancia entre centroides (m)
- Puntos segmentados disponibles
- Clasificación: `OK` / `FAIL_NO_DATA` / `FAIL_BAD_ICP` / `FAIL_POSE`

### Por sesión

- Tasa de éxito (`ok_rate`)
- Media, mediana, p90, p95 de error de traslación y rotación
- Número de frames rechazados por salto temporal
- Estadísticas de la trayectoria estabilizada

## Salidas del pipeline

Cada frame procesado genera sus resultados en `data/processed/<session>/frame_XXXXXX/`. Los archivos principales de sesión son:

| Archivo | Descripción |
|---|---|
| `parameters_final.json` | Parámetros efectivos de la ejecución |
| `run_manifest.json` | Manifiesto con diff de configuración |
| `session_summary.json` | Resumen estadístico de la sesión |
| `trajectory_object_world.csv` | Trayectoria cruda estimada |
| `trajectory_stabilized.csv` | Trayectoria con estabilización temporal |

## Licencia

Proyecto académico — Trabajo de Fin de Grado.