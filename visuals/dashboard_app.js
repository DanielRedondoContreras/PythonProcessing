// ============================================================================
// Archivo : visuals/dashboard_app.js
// Proyecto: TFG — Implementación de técnicas de visión artificial para la
//           detección de objetos en contextos de realidad mixta
//
// Propósito
// ---------
// Plantilla JavaScript que define la interfaz interactiva del dashboard de resultados. 
// El contenido es JavaScript vanilla ES5, compatible con cualquier navegador moderno sin necesidad de transpilación.
//
// Relación con generate_dashboard.py
// -----------------------------------
// Este archivo actúa como plantilla: contiene placeholders que el script Python generate_dashboard.py sustituye con los datos de cada sesión antes de embeber el código en el HTML final. Los placeholders son:
//
//   __TRAJECTORY_JSON__  →  Array JSON con los datos compactos de trayectoria
//   __SESSION_NAME__     →  Identificador de la sesión procesada
//   __FRAME_MIN__        →  Índice del primer frame
//   __FRAME_MAX__        →  Índice del último frame
//   __FRAME_COUNT__      →  Número total de frames
//
// Formato compacto de datos
// -------------------------
// Cada elemento del array de trayectoria utiliza campos abreviados para reducir el tamaño del HTML resultante:
//
//   f    = frame_index           (índice del frame)
//   v    = valid_frame           (1 = válido, 0 = fallido)
//   r    = fail_reason           (motivo de fallo o "ok")
//   tx   = traslación X          (metros, espacio mundo Unity)
//   ty   = traslación Y          (metros, espacio mundo Unity)
//   tz   = traslación Z          (metros, espacio mundo Unity)
//   fit  = fitness ICP            ([0,1], fracción de correspondencias)
//   rmse = RMSE ICP               (metros)
//   sc   = score combinado        (≤0, métrica compuesta fitness/rmse)
//   cd   = center_distance_m      (metros, null si no disponible)
//   se3  = se3_ok                 (1 = transformación SE(3) válida)
//
// Estructura del dashboard (4 pestañas)
// -------------------------------------
//   1. Arquitectura — Mapa de datos del repositorio, mapa de código,
//      descripción de cada módulo core y script de pipeline.
//   2. Flujo de Datos — Diagrama del pipeline de procesamiento,
//      archivos producidos por etapa, marcos de referencia y cadena SE(3).
//   3. Métricas — Resumen estadístico de la sesión, guía de métricas
//      con umbrales de calidad, lógica de clasificación de frames.
//   4. Visualización 3D — Trayectoria 3D interactiva (Three.js),
//      gráficos SVG de traslación y métricas por frame, tabla detallada.
//
// Dependencias externas
// ---------------------
//   - Three.js r128 (CDN) — solo para la vista 3D; se degrada
//     graciosamente si no está disponible (gráficos 2D siguen funcionando).
//
// Convención de nombres de funciones
// -----------------------------------
// Se utilizan nombres compactos para las funciones de construcción del DOM:
//   bA = build Architecture tab      bF = build Flow tab
//   bM = build Metrics tab           bV = build Visualization tab
//   bT = build Trajectory table      rC = render Charts (SVG)
//   i3 = init Three.js scene         SC = Single Chart (SVG)
//   ML = Multi-Line chart (SVG)
//   A  = media aritmética            S  = desviación estándar
//   P  = pill (badge HTML)           T  = Tab switcher
// ============================================================================


// ═══════════════════════════════════════════════════════════════
// DATOS DE SESIÓN (inyectados por generate_dashboard.py)
// ═══════════════════════════════════════════════════════════════

var D = __TRAJECTORY_JSON__;

// Metadatos de sesión (sustituidos en tiempo de generación)
var _SN   = '__SESSION_NAME__';     // Nombre de sesión
var _FMIN = __FRAME_MIN__;          // Primer frame
var _FMAX = __FRAME_MAX__;          // Último frame

// Subconjunto de frames válidos (valid_frame == 1)
var V = D.filter(function(d) { return d.v; });


// ═══════════════════════════════════════════════════════════════
// FUNCIONES AUXILIARES
// ═══════════════════════════════════════════════════════════════

/**
 * Calcula la media aritmética de un array numérico.
 * Retorna 0 si el array está vacío.
 */
function A(a) {
  return a.length ? a.reduce(function(s, x) { return s + x; }, 0) / a.length : 0;
}

/**
 * Calcula la desviación estándar poblacional de un array numérico.
 * Retorna 0 si el array está vacío.
 */
function S(a) {
  var m = A(a);
  return a.length ? Math.sqrt(a.reduce(function(s, x) { return s + (x - m) * (x - m); }, 0) / a.length) : 0;
}

/**
 * Genera el HTML de un badge tipo "pill" con color personalizado.
 * Se utiliza extensivamente en los mapas de archivos y código.
 */
function P(l, c) {
  return '<span class="pl" style="background:' + c + '22;color:' + c + ';border-color:' + c + '44">' + l + '</span>';
}

/**
 * Controla la navegación entre pestañas del dashboard.
 * La pestaña de visualización 3D inicializa Three.js y los gráficos
 * SVG de forma diferida (solo al acceder por primera vez).
 */
function T(id) {
  ['a', 'f', 'm', 'v'].forEach(function(t) {
    document.getElementById('t' + t).classList.toggle('H', t !== id);
  });
  document.querySelectorAll('.tb').forEach(function(b, i) {
    b.classList.toggle('on', ['a', 'f', 'm', 'v'][i] === id);
  });
  // Inicialización diferida de la pestaña de visualización
  if (id === 'v' && !window._vi) {
    window._vi = 1;
    i3();   // Escena Three.js
    rC();   // Gráficos SVG
  }
}


// ═══════════════════════════════════════════════════════════════
// GRÁFICOS SVG — Renderizado manual sin dependencias externas
// ═══════════════════════════════════════════════════════════════
// Se implementan gráficos SVG directamente en lugar de utilizar bibliotecas como Recharts o Chart.js para garantizar el funcionamiento offline del dashboard.

/**
 * SC — Single Chart: renderiza un gráfico de línea o barras en un <svg>.
 *
 * Parámetros:
 *   el   — Elemento SVG destino
 *   data — Array de objetos con campo 'f' (frame) y el campo 'key'
 *   key  — Nombre del campo a graficar (e.g. 'fit', 'rmse')
 *   col  — Color del gráfico (hex string)
 *   yMn  — Mínimo del eje Y (undefined = auto)
 *   yMx  — Máximo del eje Y (undefined = auto)
 *   fmt  — Función formateadora de etiquetas del eje Y
 *   bar  — Si es truthy, se renderiza como gráfico de barras
 */
function SC(el, data, key, col, yMn, yMx, fmt, bar) {
  var svg = el;
  if (!svg) return;
  var W = svg.clientWidth || 500, H = svg.clientHeight || 220;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var p = {l: 50, r: 15, t: 10, b: 30}, cw = W - p.l - p.r, ch = H - p.t - p.b;
  var vs = data.map(function(d) { return d[key]; }).filter(function(v) { return v !== null; });
  if (yMn === void 0) yMn = Math.min.apply(null, vs);
  if (yMx === void 0) yMx = Math.max.apply(null, vs);
  var rng = yMx - yMn || 1;
  yMn -= rng * .05;
  yMx += rng * .05;
  rng = yMx - yMn;
  var n = data.length, bw = cw / n;
  function sX(i) { return p.l + i * bw + bw / 2; }
  function sY(v) { return p.t + ch * (1 - (v - yMn) / rng); }

  // Rejilla horizontal y etiquetas del eje Y
  var h = '<g stroke="#1e293b" stroke-width="0.5">';
  for (var i = 0; i < 5; i++) {
    var yy = p.t + ch * i / 4;
    h += '<line x1="' + p.l + '" y1="' + yy + '" x2="' + (W - p.r) + '" y2="' + yy + '"/>';
    var lb = fmt ? fmt(yMx - rng * i / 4) : (yMx - rng * i / 4).toFixed(4);
    h += '<text x="' + (p.l - 4) + '" y="' + (yy + 3) + '" fill="#94a3b8" font-size="9" text-anchor="end">' + lb + '</text>';
  }
  h += '</g><g>';
  // Etiquetas del eje X (índices de frame)
  data.forEach(function(d, i) {
    if (n > 15 && i % 2) return;  // Mostrar alternadas si hay muchos frames
    h += '<text x="' + sX(i) + '" y="' + (H - 5) + '" fill="#94a3b8" font-size="9" text-anchor="middle">' + d.f + '</text>';
  });
  h += '</g>';

  if (bar) {
    // Modo barras (utilizado para center_distance_m)
    data.forEach(function(d, i) {
      var v = d[key];
      if (v === null) return;
      h += '<rect x="' + (sX(i) - bw * .3) + '" y="' + sY(v) + '" width="' + (bw * .6) + '" height="' + (ch * (v - yMn) / rng) + '" fill="' + col + '" fill-opacity="0.6" rx="2"/>';
    });
  } else {
    // Modo línea con área de relleno degradado
    var pts = [];
    data.forEach(function(d, i) {
      if (d[key] !== null) pts.push({x: sX(i), y: sY(d[key])});
    });
    if (pts.length > 1) {
      h += '<defs><linearGradient id="g' + key + '" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="' + col + '" stop-opacity="0.3"/><stop offset="100%" stop-color="' + col + '" stop-opacity="0"/></linearGradient></defs>';
      var ar = 'M' + pts[0].x + ',' + pts[0].y;
      pts.forEach(function(q, i) { if (i > 0) ar += 'L' + q.x + ',' + q.y; });
      ar += 'L' + pts[pts.length - 1].x + ',' + (p.t + ch) + 'L' + pts[0].x + ',' + (p.t + ch) + 'Z';
      h += '<path d="' + ar + '" fill="url(#g' + key + ')"/>';
      var ln = 'M' + pts[0].x + ',' + pts[0].y;
      pts.forEach(function(q, i) { if (i > 0) ln += 'L' + q.x + ',' + q.y; });
      h += '<path d="' + ln + '" fill="none" stroke="' + col + '" stroke-width="2"/>';
      pts.forEach(function(q) {
        h += '<circle cx="' + q.x + '" cy="' + q.y + '" r="3" fill="' + col + '"/>';
      });
    }
  }
  svg.innerHTML = h;
}

/**
 * ML — Multi-Line chart: renderiza varias series en un mismo <svg>.
 * Se utiliza para graficar tx, ty, tz simultáneamente.
 *
 * Parámetros:
 *   el   — Elemento SVG destino
 *   data — Array de objetos de datos
 *   ks   — Array de nombres de campo (e.g. ['tx','ty','tz'])
 *   cs   — Array de colores correspondientes
 *   ns   — Array de nombres para la leyenda
 */
function ML(el, data, ks, cs, ns) {
  var svg = el;
  if (!svg) return;
  var W = svg.clientWidth || 500, H = svg.clientHeight || 280;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  var p = {l: 50, r: 15, t: 10, b: 45}, cw = W - p.l - p.r, ch = H - p.t - p.b;

  // Cálculo del rango Y global (abarca todas las series)
  var av = [];
  ks.forEach(function(k) {
    data.forEach(function(d) { if (d[k] !== null) av.push(d[k]); });
  });
  var yMn = Math.min.apply(null, av), yMx = Math.max.apply(null, av), rng = yMx - yMn || 1;
  yMn -= rng * .05;
  yMx += rng * .05;
  rng = yMx - yMn;
  var n = data.length, bw = cw / n;
  function sX(i) { return p.l + i * bw + bw / 2; }
  function sY(v) { return p.t + ch * (1 - (v - yMn) / rng); }

  // Rejilla y etiquetas
  var h = '<g stroke="#1e293b" stroke-width="0.5">';
  for (var i = 0; i < 5; i++) {
    var yy = p.t + ch * i / 4;
    h += '<line x1="' + p.l + '" y1="' + yy + '" x2="' + (W - p.r) + '" y2="' + yy + '"/>';
    h += '<text x="' + (p.l - 4) + '" y="' + (yy + 3) + '" fill="#94a3b8" font-size="9" text-anchor="end">' + (yMx - rng * i / 4).toFixed(1) + '</text>';
  }
  h += '</g><g>';
  data.forEach(function(d, i) {
    if (n > 15 && i % 2) return;
    h += '<text x="' + sX(i) + '" y="' + (H - 25) + '" fill="#94a3b8" font-size="9" text-anchor="middle">' + d.f + '</text>';
  });
  h += '</g>';

  // Renderizado de cada serie con su color
  ks.forEach(function(k, ki) {
    var pts = [];
    data.forEach(function(d, i) {
      if (d[k] !== null) pts.push({x: sX(i), y: sY(d[k])});
    });
    if (pts.length > 1) {
      var ln = 'M' + pts[0].x + ',' + pts[0].y;
      pts.forEach(function(q, i) { if (i > 0) ln += 'L' + q.x + ',' + q.y; });
      h += '<path d="' + ln + '" fill="none" stroke="' + cs[ki] + '" stroke-width="2"/>';
      pts.forEach(function(q) {
        h += '<circle cx="' + q.x + '" cy="' + q.y + '" r="2" fill="' + cs[ki] + '"/>';
      });
    }
  });

  // Leyenda inferior
  var lx = p.l + 10;
  ks.forEach(function(k, ki) {
    h += '<circle cx="' + lx + '" cy="' + (H - 8) + '" r="4" fill="' + cs[ki] + '"/>';
    h += '<text x="' + (lx + 8) + '" y="' + (H - 5) + '" fill="#94a3b8" font-size="10">' + ns[ki] + '</text>';
    lx += 100;
  });
  svg.innerHTML = h;
}


// ═══════════════════════════════════════════════════════════════
// PESTAÑA 1: ARQUITECTURA
// ═══════════════════════════════════════════════════════════════
// Muestra la estructura de archivos del repositorio, el mapa de código organizado por fases, y la descripción de cada módulo.

function bA() {
  var ta = document.getElementById('ta');

  // ── Mapa de datos: estructura de directorios de data/ ──
  // Refleja fielmente los archivos generados por cada fase del pipeline
  ta.innerHTML = '<div class="sec"><h2>Mapa de Datos del Repositorio</h2>'
    + '<div class="d">Estructura de directorios data/ con todos los archivos generados por el pipeline</div>'
    + '<div style="display:grid;grid-template-columns:2fr 1fr;gap:12px">'
    + '<div class="ab">'
    + '<div class="al" style="color:#f59e0b">data/processed/&lt;session&gt;/</div>'
    + '<div class="ai" style="margin-bottom:8px">'
    + '<div class="al" style="color:#06b6d4">frame_XXXXXX/</div>'
    // Imágenes rectificadas y previsualizaciones de disparidad/profundidad
    + '<div class="as" style="border-color:#22d3ee">'
    + '<div class="h">Im\u00e1genes rectificadas y previsualizaciones</div>'
    + ['left_rect.png', 'right_rect.png', 'disparity_raw.png', 'disparity_rect.png',
       'disparity_rect_wls.png', 'disparity_final.png', 'depth_preview.png'].map(function(f) { return P(f, '#22d3ee'); }).join('')
    + '</div>'
    // Archivos JSON de métricas por etapa
    + '<div class="as" style="border-color:#f97316">'
    + '<div class="h">M\u00e9tricas y metadatos</div>'
    + ['rectification_debug.json', 'disparity_metrics_raw.json', 'disparity_metrics_rect.json',
       'disparity_metrics_rect_valid.json', 'disparity_metrics_wls_valid.json',
       'depth_metrics.json', 'pointcloud_metrics.json', 'segmentation_metrics.json',
       'sgbm_preset.json', 'y_error_orb_raw.json', 'y_error_orb_rect.json'].map(function(f) { return P(f, '#f97316'); }).join('')
    + '</div>'
    // Nubes de puntos en formato PLY (Open3D)
    + '<div class="as" style="border-color:#2dd4bf">'
    + '<div class="h">Nubes de puntos</div>'
    + ['pointcloud_world_unity.ply', 'pointcloud_camera_rect.ply',
       'object_roi_raw.ply', 'object_preseg_clean.ply',
       'object_segmented_normals.ply', 'object_segmented.ply'].map(function(f) { return P(f, '#2dd4bf'); }).join('')
    + '</div>'
    // Arrays NumPy binarios
    + '<div class="as" style="border-color:#ec4899">'
    + '<div class="h">Arrays binarios (.npy)</div>'
    + ['disparity_raw.npy', 'disparity_rect.npy', 'disparity_rect_wls.npy', 'disparity_final.npy',
       'disparity_rect_wls_valid_mask.npy', 'depth_m.npy', 'depth_valid_mask.npy',
       'pointcloud_camera_opencv.npy', 'pointcloud_world_unity.npy',
       'object_segmented_nx6.npy'].map(function(f) { return P(f, '#ec4899'); }).join('')
    + '</div>'
    // Subdirectorio ppf_match/ dentro de cada frame
    + '<div class="ai" style="margin-top:6px">'
    + '<div class="al" style="color:#8b5cf6">ppf_match/</div>'
    + ['eval_centers.json', 'match_meta.json', 'pose_best.json', 'pose_eval.json'].map(function(f) { return P(f, '#f97316'); }).join('')
    + P('aligned_model.ply', '#2dd4bf')
    + ['scene_face.ply', 'scene_handle.ply', 'scene_ring.ply'].map(function(f) { return P(f, '#2dd4bf'); }).join('')
    + '</div>'
    + '</div>'
    // Archivos a nivel de sesión (fuera de frame_XXXXXX/)
    + '<div class="as" style="border-color:#22c55e">'
    + '<div class="h">Archivos a nivel de sesi\u00f3n</div>'
    + ['parameters_final.json', 'run_manifest.json', 'session_summary.json',
       'summary_frames.json', 'per_frame_metrics.jsonl'].map(function(f) { return P(f, '#f97316'); }).join('')
    + P('trajectory_object_world.csv', '#22c55e')
    + P('trajectory_stabilized.csv', '#22c55e')
    + '</div>'
    + '</div>'
    // Panel derecho: datos crudos y modelo CAD
    + '<div style="display:flex;flex-direction:column;gap:12px">'
    + '<div class="ab">'
    + '<div class="al" style="color:#06b6d4">data/raw/&lt;session&gt;/frame_XXXXXX/</div>'
    + P('left.png', '#22d3ee') + P('right.png', '#22d3ee') + P('metadata.json', '#f97316')
    + '<div style="color:#94a3b8;font-size:9px;margin-top:6px">Poses c\u00e1maras/mandos, intr\u00ednsecos, timestamps</div>'
    + '</div>'
    + '<div class="ab">'
    + '<div class="al" style="color:#10b981">data/cad/</div>'
    + P('Touch Plus MetaQuest Controllers.blend', '#ef4444')
    + P('right_controller.ply', '#2dd4bf')
    + P('right_controller_ready.ply', '#2dd4bf')
    + P('right_controller_ready_metrics.json', '#f97316')
    + '<div style="color:#94a3b8;font-size:9px;margin-top:6px">Modelo CAD del controlador Meta Quest 3</div>'
    + '</div>'
    + '<div class="ab">'
    + '<div class="al" style="color:#8b5cf6">data/cad/ppf/</div>'
    + P('right_controller_ppf_meta.json', '#f97316')
    + P('right_controller_sampled_normals.ply', '#2dd4bf')
    + P('right_controller_ppf.npz', '#ec4899')
    + '<div style="color:#94a3b8;font-size:9px;margin-top:6px">Descriptores PPF pre-entrenados [Legacy: entrenamiento ahora inline]</div>'
    + '</div>'
    + '</div></div></div>';

  // ── Mapa de código: módulos organizados por fase del pipeline ──
  var ph = [
    {id: 'p1', c: '#06b6d4', t: 'Fase 1 \u2014 Carga de datos y utilidades',
     i: ['data_loader.py', 'io_utils.py', 'transforms.py', 'reporting.py']},
    {id: 'p2', c: '#8b5cf6', t: 'Fase 2 \u2014 Reconstrucci\u00f3n est\u00e9reo 3D',
     i: ['rectification.py', 'disparity.py', 'depth.py', 'pointcloud.py']},
    {id: 'p3', c: '#f59e0b', t: 'Fase 3 \u2014 Estimaci\u00f3n de pose',
     i: ['cad_model.py', 'segmentation.py', 'ppf_match.py']},
    {id: 'p4', c: '#10b981', t: 'Fase 4 \u2014 Evaluaci\u00f3n y resultados',
     i: ['pose_eval.py', 'frame_status.py', 'temporal_stabilizer.py']},
    {id: 'pc', c: '#94a3b8', t: 'Configuraci\u00f3n y ejecuci\u00f3n',
     i: ['default_pipeline.yaml', 'run_pipeline.ps1', 'requirements.txt', 'README.md']}
  ];

  // Diagrama del flujo de ejecución del pipeline
  var pipeSteps = [
    '<div class="fb" style="background:#e879f922;border:2px solid #e879f9;color:#e879f9;padding:8px 20px;font-size:12px;font-weight:800;margin-bottom:8px">run_session_pipeline.py</div>',
    '<div style="color:#94a3b8;font-size:10px;margin-bottom:6px">Orquestador \u2014 ejecuta secuencialmente:</div>',
    '<div class="fb" style="background:#8b5cf618;border:1px solid #8b5cf644;color:#8b5cf6">run_disparity.py</div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div class="fb" style="background:#8b5cf618;border:1px solid #8b5cf644;color:#8b5cf6">run_depth.py</div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div class="fb" style="background:#8b5cf618;border:1px solid #8b5cf644;color:#8b5cf6">run_pointcloud.py</div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div class="fb" style="background:#f59e0b18;border:1px solid #f59e0b44;color:#f59e0b">run_segmentation.py</div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div class="fb" style="background:#f59e0b18;border:1px solid #f59e0b44;color:#f59e0b">run_ppf_match.py <span style="font-size:9px;opacity:0.7">(PPF + ICP)</span></div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div class="fb" style="background:#ef444418;border:1px solid #ef444444;color:#ef4444">run_pose_eval.py</div>',
    '<div style="color:#94a3b8;font-size:16px">\u2193</div>',
    '<div class="fb" style="background:#e879f918;border:1px solid #e879f944;color:#e879f9">run_session_summary.py</div>'
  ];

  ta.innerHTML += '<div class="sec"><h2>Mapa de C\u00f3digo del Repositorio</h2>'
    + '<div class="d">M\u00f3dulos core organizados por fase, flujo de ejecuci\u00f3n del pipeline y scripts auxiliares</div>'
    + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
    + '<div>'
    + '<div style="color:#06b6d4;font-weight:800;font-size:14px;margin-bottom:12px">src/core/ \u2014 M\u00f3dulos Core</div>'
    + ph.slice(0, 4).map(function(p) {
        return '<div class="ph" style="background:' + p.c + '08;border:1px solid ' + p.c + '33">'
          + '<div class="t" style="color:' + p.c + '">' + p.t + '</div>'
          + '<div class="ps">' + p.i.map(function(i) { return P(i, p.c); }).join('') + '</div></div>';
      }).join('')
    + '<div style="color:#ef4444;font-weight:800;font-size:14px;margin:20px 0 12px">Scripts auxiliares</div>'
    + ph.slice(4).map(function(p) {
        return '<div class="ph" style="background:' + p.c + '08;border:1px solid ' + p.c + '33">'
          + '<div class="t" style="color:' + p.c + '">' + p.t + '</div>'
          + '<div class="ps">' + p.i.map(function(i) { return P(i, p.c); }).join('') + '</div></div>';
      }).join('')
    + '</div>'
    + '<div>'
    + '<div style="color:#e879f9;font-weight:800;font-size:14px;margin-bottom:12px">src/pipeline/ \u2014 Flujo de Ejecuci\u00f3n</div>'
    + '<div class="pf">' + pipeSteps.join('') + '</div>'
    + '</div></div></div>';

  // ── Descripción de cada módulo core (src/core/) ──
  var core = [
    {n: 'transforms.py', d: 'Matem\u00e1tica SE(3)/SO(3) centralizada: quaterniones, matrices 4\u00d74, errores de pose, validaci\u00f3n de rotaciones', c: '#06b6d4'},
    {n: 'data_loader.py', d: 'Carga y parsing de datos brutos: metadata JSON, poses del controlador, intr\u00ednsecos de c\u00e1mara', c: '#06b6d4'},
    {n: 'io_utils.py', d: 'Lectura/escritura JSON unificada, validaci\u00f3n de archivos, parse de especificaci\u00f3n de frames', c: '#06b6d4'},
    {n: 'reporting.py', d: 'Escritura de JSON/JSONL/CSV, funciones de resumen estad\u00edstico, agregaci\u00f3n', c: '#06b6d4'},
    {n: 'rectification.py', d: 'Rectificaci\u00f3n est\u00e9reo con OpenCV: stereoRectify, evaluaci\u00f3n de calidad epipolar (ORB)', c: '#8b5cf6'},
    {n: 'disparity.py', d: 'C\u00e1lculo de disparidad SGBM + filtro WLS. CLAHE para mejora de contraste. M\u00e9tricas de completitud', c: '#8b5cf6'},
    {n: 'depth.py', d: 'Conversi\u00f3n disparidad\u2192profundidad: Z = fx\u00b7b/d. Validaci\u00f3n de rango Z y m\u00e9tricas de cobertura', c: '#8b5cf6'},
    {n: 'pointcloud.py', d: 'Backprojection a nube 3D, transformaci\u00f3n c\u00e1mara\u2192mundo, voxel downsampling, escritura PLY', c: '#8b5cf6'},
    {n: 'cad_model.py', d: 'Preparaci\u00f3n del modelo CAD: lectura malla/nube, voxel downsample, normales, validaci\u00f3n de tama\u00f1o', c: '#f59e0b'},
    {n: 'segmentation.py', d: 'Crop OBB basada en pose GT, filtrado Z, DBSCAN clustering, filtro de outliers estad\u00edstico', c: '#f59e0b'},
    {n: 'ppf_match.py', d: 'Estimaci\u00f3n de pose 6-DOF: PPF voting (Drost 2010) + ICP multi-nivel (OpenCV Surface Matching)', c: '#f59e0b'},
    {n: 'pose_eval.py', d: 'Evaluaci\u00f3n por frame: carga GT/EST, calcula errores de traslaci\u00f3n y rotaci\u00f3n, clasifica el frame', c: '#10b981'},
    {n: 'frame_status.py', d: 'Determinaci\u00f3n de validez de cada frame: chequeos SE(3), presencia de datos, agregaci\u00f3n de metadatos', c: '#10b981'},
    {n: 'temporal_stabilizer.py', d: 'Filtro EMA + SLERP para estabilizaci\u00f3n temporal. Rechazo de saltos implausibles entre frames', c: '#10b981'}
  ];

  // ── Descripción de cada script de pipeline (src/pipeline/) ──
  var pipe = [
    {n: 'run_session_pipeline.py', d: 'Orquestador principal: ejecuta todo el pipeline por frame con configuraci\u00f3n YAML. Prioridad: CLI > YAML > defaults', c: '#e879f9'},
    {n: 'run_disparity.py', d: 'Calcula disparidad SGBM + WLS para un frame. Genera mapas .npy y visualizaciones .png', c: '#8b5cf6'},
    {n: 'run_depth.py', d: 'Convierte disparidad a profundidad. Genera depth_m.npy, m\u00e1scara y m\u00e9tricas', c: '#8b5cf6'},
    {n: 'run_pointcloud.py', d: 'Backprojection a nube de puntos en coordenadas c\u00e1mara y mundo. Exporta PLY y NPY', c: '#8b5cf6'},
    {n: 'run_cad_model.py', d: 'Prepara el modelo CAD: downsample, normales, validaci\u00f3n. Genera right_controller_ready.ply', c: '#f59e0b'},
    {n: 'run_segmentation.py', d: 'Segmenta la nube de puntos: ROI + DBSCAN + filtros. Genera object_segmented.ply', c: '#f59e0b'},
    {n: 'run_ppf_match.py', d: 'Registro global: PPF voting + ICP multi-nivel (OpenCV Surface Matching). Genera pose_best.json', c: '#f59e0b'},
    {n: 'run_pose_eval.py', d: 'Eval\u00faa pose estimada vs GT: errores de traslaci\u00f3n/rotaci\u00f3n, clasifica frame. Genera pose_eval.json', c: '#ef4444'},
    {n: 'run_session_summary.py', d: 'Agrega resultados por sesi\u00f3n: trayectoria CSV, estabilizaci\u00f3n temporal, session_summary.json', c: '#e879f9'}
  ];

  function cd(arr) {
    return arr.map(function(c) {
      return '<div class="cp" style="border-left:3px solid ' + c.c + '">'
        + '<div class="n" style="color:' + c.c + '">' + c.n + '</div>'
        + '<div class="cd">' + c.d + '</div></div>';
    }).join('');
  }

  ta.innerHTML += '<div class="sec"><h2>Descripci\u00f3n de Componentes Core</h2>'
    + '<div class="d">Cada m\u00f3dulo de src/core/ y su responsabilidad</div>'
    + '<div class="g2">' + cd(core) + '</div></div>';

  ta.innerHTML += '<div class="sec"><h2>Descripci\u00f3n de Scripts de Pipeline</h2>'
    + '<div class="d">Cada runner de src/pipeline/ y su funci\u00f3n</div>'
    + '<div class="g2">' + cd(pipe) + '</div></div>';
}


// ═══════════════════════════════════════════════════════════════
// PESTAÑA 2: FLUJO DE DATOS
// ═══════════════════════════════════════════════════════════════
// Diagrama del pipeline, archivos por etapa, y cadena SE(3).

function bF() {
  var tf = document.getElementById('tf');

  // ── Fases del pipeline con método y símbolo ──
  var steps = [
    {l: 'Captura Est\u00e9reo',   s: 'Meta Quest 3',    i: '\u{1F4F7}', c: '#06b6d4'},
    {l: 'Rectificaci\u00f3n',     s: 'stereoRectify',    i: '\u{1F527}', c: '#06b6d4'},
    {l: 'Disparidad',             s: 'SGBM + WLS',       i: '\u{1F4CA}', c: '#8b5cf6'},
    {l: 'Profundidad',            s: 'Z = fx\u00b7b / d', i: '\u{1F4CF}', c: '#8b5cf6'},
    {l: 'Nube de Puntos',         s: 'Backprojection',   i: '\u2601\uFE0F', c: '#f59e0b'},
    {l: 'Segmentaci\u00f3n',      s: 'ROI + DBSCAN',     i: '\u2702\uFE0F', c: '#f59e0b'},
    {l: 'Registro ICP',           s: 'PPF + ICP',        i: '\u{1F3AF}', c: '#10b981'},
    {l: 'Evaluaci\u00f3n',        s: 'SE(3) error',      i: '\u{1F4CB}', c: '#ef4444'},
    {l: 'Estabilizaci\u00f3n',    s: 'EMA + SLERP',      i: '\u{1F504}', c: '#e879f9'}
  ];

  var fd = '<div style="display:flex;gap:6px;overflow-x:auto;padding:8px 0">';
  steps.forEach(function(s, i) {
    fd += '<div style="min-width:100px;background:#111827;border:1.5px solid ' + s.c + ';border-radius:8px;padding:10px;text-align:center;flex-shrink:0">'
      + '<div style="font-size:22px">' + s.i + '</div>'
      + '<div style="color:' + s.c + ';font-size:9px;font-weight:700;margin:4px 0">' + s.l + '</div>'
      + '<div style="color:#94a3b8;font-size:8px">' + s.s + '</div>'
      + '<div style="background:' + s.c + '12;border-radius:3px;padding:2px;margin-top:6px;color:' + s.c + ';font-size:7px">Fase ' + (i + 1) + '</div>'
      + '</div>';
    if (i < steps.length - 1) {
      fd += '<div style="display:flex;align-items:center;color:#334155;font-size:16px;flex-shrink:0">\u2192</div>';
    }
  });
  fd += '</div>';
  fd += '<div style="background:#06b6d406;border-radius:4px;padding:8px;text-align:center;margin-top:8px;color:#06b6d4;font-size:10px">'
    + 'Imagen Est\u00e9reo \u2192 Disparidad \u2192 Profundidad \u2192 Nube 3D \u2192 Segmento \u2192 Registro \u2192 Pose 6D \u2192 Trayectoria</div>';

  // ── Archivos producidos por cada fase ──
  var stgs = [
    {n: 'Entrada', c: '#06b6d4', f: ['left.png / right.png', 'metadata.json (poses, intr\u00ednsecos)', 'default_pipeline.yaml']},
    {n: 'Fase 1: Reconstrucci\u00f3n', c: '#8b5cf6', f: ['left_rect.png / right_rect.png', 'rectification_debug.json', 'disparity_raw / rect / wls', 'depth_m.npy + mask', 'depth_metrics.json']},
    {n: 'Fase 2: Nube + Seg.', c: '#f59e0b', f: ['pointcloud_world_unity.ply', 'pointcloud_metrics.json', 'object_segmented.ply', 'segmentation_metrics.json']},
    {n: 'Fase 3: Registro', c: '#10b981', f: ['pose_best.json (T 4\u00d74)', 'match_meta.json', 'eval_centers.json', 'aligned_model.ply']},
    {n: 'Fase 4: Evaluaci\u00f3n', c: '#ef4444', f: ['pose_eval.json', 'per_frame_metrics.jsonl', 'summary.json']},
    {n: 'Fase 5: Agregaci\u00f3n', c: '#e879f9', f: ['trajectory_object_world.csv', 'trajectory_stabilized.csv', 'session_summary.json', 'parameters_final.json']}
  ];

  var sg = '<div class="g3">' + stgs.map(function(s) {
    return '<div class="st" style="border:1px solid ' + s.c + '33;border-top:3px solid ' + s.c + '">'
      + '<div class="sn" style="color:' + s.c + '">' + s.n + '</div>'
      + s.f.map(function(f) { return '<div class="sf">\u{1F4C4} ' + f + '</div>'; }).join('')
      + '</div>';
  }).join('') + '</div>';

  // ── Marcos de referencia del sistema ──
  // Cinco frames coordenados utilizados en el pipeline
  var frames = [
    {f: 'W (Mundo)',       d: 'Unity World. Y-up',               c: '#06b6d4'},
    {f: 'C_left',          d: 'C\u00e1mara izquierda original',  c: '#8b5cf6'},
    {f: 'C_left_rect',     d: 'C\u00e1mara rectificada (OpenCV)', c: '#8b5cf6'},
    {f: 'M (Modelo)',      d: 'Frame del modelo CAD',             c: '#f59e0b'},
    {f: 'Controller_GT',   d: 'Tracking nativo SDK',              c: '#10b981'}
  ];

  var rf = '<div class="fr">' + frames.map(function(f) {
    return '<div class="fx" style="border:1px solid ' + f.c + '33">'
      + '<div class="fn" style="color:' + f.c + '">' + f.f + '</div>'
      + '<div class="fd">' + f.d + '</div></div>';
  }).join('') + '</div>';

  // Cadena de composición SE(3): T_world←model = T_world←cam_rect · T_cam_rect←model
  var chain = [
    {l: 'T_world\u2190cam',          d: 'Pose del casco',            c: '#06b6d4'},
    {l: 'T_cam\u2190model (ICP)',     d: 'Resultado del registro',    c: '#8b5cf6'},
    {l: 'T_world\u2190model',        d: 'Pose final estimada',       c: '#f59e0b'}
  ];

  rf += '<div class="cb">'
    + '<div class="ce"><span style="color:#f59e0b">T_world\u2190model</span>'
    + ' = <span style="color:#06b6d4">T_world\u2190cam_rect</span>'
    + ' \u00b7 <span style="color:#8b5cf6">T_cam_rect\u2190model</span></div>'
    + '<div class="cg">' + chain.map(function(c) {
        return '<div class="ci" style="background:' + c.c + '11">'
          + '<div style="color:' + c.c + ';font-family:monospace;font-size:11px;font-weight:700">' + c.l + '</div>'
          + '<div style="color:#94a3b8;font-size:10px;margin-top:4px">' + c.d + '</div></div>';
      }).join('')
    + '</div></div>';

  tf.innerHTML = '<div class="sec"><h2>Pipeline de Procesamiento</h2>'
    + '<div class="d">Flujo completo desde captura est\u00e9reo hasta trayectoria estabilizada</div>' + fd + '</div>'
    + '<div class="sec"><h2>Flujo de Archivos por Etapa</h2>'
    + '<div class="d">Archivos producidos y consumidos por cada fase</div>' + sg + '</div>'
    + '<div class="sec"><h2>Marcos de Referencia y Cadena SE(3)</h2>'
    + '<div class="d">Transformaciones geom\u00e9tricas del sistema</div>' + rf + '</div>';
}


// ═══════════════════════════════════════════════════════════════
// PESTAÑA 3: MÉTRICAS
// ═══════════════════════════════════════════════════════════════
// Resumen estadístico, guía de interpretación y clasificación.

function bM() {
  var tm = document.getElementById('tm');

  // Extracción de vectores de métricas de los frames válidos
  var fV = V.map(function(d) { return d.fit; });
  var rV = V.map(function(d) { return d.rmse; });
  var cV = V.filter(function(d) { return d.cd !== null; }).map(function(d) { return d.cd; });

  // ── Tarjetas resumen de la sesión ──
  var cards = [
    {l: 'Frames Totales', v: '' + D.length,
     s: V.length + ' v\u00e1lidos \u00b7 ' + (D.length - V.length) + ' fallidos', c: '#06b6d4', i: '\u{1F3AC}'},
    {l: 'Tasa de \u00c9xito', v: Math.round(V.length / D.length * 100) + '%',
     s: V.length + '/' + D.length, c: '#10b981', i: '\u2705'},
    {l: 'Fitness Medio', v: A(fV).toFixed(3),
     s: '\u03c3 = ' + S(fV).toFixed(3), c: '#8b5cf6', i: '\u{1F4CA}'},
    {l: 'RMSE Medio', v: (A(rV) * 1000).toFixed(2) + ' mm',
     s: '\u03c3 = ' + (S(rV) * 1000).toFixed(2) + ' mm', c: '#f59e0b', i: '\u{1F4CF}'},
    {l: 'Center Dist Media', v: (A(cV) * 100).toFixed(2) + ' cm',
     s: '\u03c3 = ' + (S(cV) * 100).toFixed(2) + ' cm', c: '#ef4444', i: '\u{1F3AF}'},
    {l: 'SE(3) V\u00e1lido', v: '100%',
     s: 'Transformaciones v\u00e1lidas', c: '#e879f9', i: '\u{1F522}'}
  ];

  var ch = '<div class="g3">' + cards.map(function(c) {
    return '<div class="crd" style="border:1px solid ' + c.c + '33;border-left:3px solid ' + c.c + '">'
      + '<div><div class="l">' + c.l + '</div>'
      + '<div class="v" style="color:' + c.c + '">' + c.v + '</div>'
      + '<div class="s">' + c.s + '</div></div>'
      + '<div class="i">' + c.i + '</div></div>';
  }).join('') + '</div>';

  // ── Tabla de referencia de métricas con umbrales ──
  // Los umbrales coinciden con los definidos en default_pipeline.yaml (sección pose_eval)
  var mt = [
    {n: 'fitness',          f: 'pose_best.json',       d: 'Fracci\u00f3n correspondencias ICP',            r: '[0,1]',      g: '> 0.5',       a: '0.15\u20130.5',       b: '< 0.15'},
    {n: 'rmse',             f: 'pose_best.json',       d: 'Error cuadr\u00e1tico medio ICP',               r: '[0,\u221e)', g: '< 0.004',     a: '0.004\u20130.02',     b: '> 0.02'},
    {n: 'score',            f: 'pose_best.json',       d: 'Votos PPF del mejor candidato',                  r: '[0,\u221e)',   g: '> 100',       a: '50\u2013100',          b: '< 50'},
    {n: 'center_distance_m',f: 'eval_centers.json',    d: 'Distancia centroide alineado vs segmento',      r: '[0,\u221e)', g: '< 0.02',      a: '0.02\u20130.05',      b: '> 0.05'},
    {n: 'trans_error_m',    f: 'pose_eval.json',       d: 'Error traslaci\u00f3n ||t_est - t_gt||\u2082',  r: '[0,\u221e)', g: '< 0.02',      a: '0.02\u20130.05',      b: '> 0.05'},
    {n: 'rot_error_deg',    f: 'pose_eval.json',       d: 'Error rotaci\u00f3n geod\u00e9sico',            r: '[0,180]',    g: '< 5\u00b0',   a: '5\u00b0\u201310\u00b0',b: '> 10\u00b0'},
    {n: 'det_R',            f: 'trajectory CSV',       d: 'det(R) \u2014 debe ser 1.0',                    r: '{1.0}',      g: '|det-1|<1e-6',a: '|det-1|<0.01',        b: '|det-1|>0.01'},
    {n: 'n_out',            f: 'segmentation_metrics',  d: 'Puntos segmentados finales',                    r: '[0,\u221e)', g: '> 1000',      a: '300\u20131000',       b: '< 300'},
    {n: 'ok_rate',          f: 'session_summary',       d: 'Tasa de \u00e9xito',                           r: '[0,1]',      g: '> 0.8',       a: '0.5\u20130.8',        b: '< 0.5'}
  ];

  var tb = '<table class="mt"><thead><tr>'
    + '<th>M\u00e9trica</th><th>Archivo</th><th>Descripci\u00f3n</th><th>Rango</th><th>Excelente</th><th>Aceptable</th><th>Problema</th>'
    + '</tr></thead><tbody>'
    + mt.map(function(m) {
        return '<tr>'
          + '<td style="color:#f8fafc;font-family:monospace;font-weight:600">' + m.n + '</td>'
          + '<td style="color:#94a3b8;font-family:monospace;font-size:10px">' + m.f + '</td>'
          + '<td style="color:#e2e8f0;max-width:200px">' + m.d + '</td>'
          + '<td style="color:#94a3b8;font-family:monospace">' + m.r + '</td>'
          + '<td style="color:#10b981;font-family:monospace">' + m.g + '</td>'
          + '<td style="color:#f59e0b;font-family:monospace">' + m.a + '</td>'
          + '<td style="color:#ef4444;font-family:monospace">' + m.b + '</td>'
          + '</tr>';
      }).join('')
    + '</tbody></table>';

  // ── Guía de clasificación de frames ──
  // Categorías asignadas por pose_eval.py según los umbrales del YAML
  var cls = [
    {l: 'OK',            c: '#10b981', d: 'Fitness \u2265 0.15, RMSE \u2264 0.02, trans \u2264 0.05m, rot \u2264 10\u00b0, seg \u2265 300'},
    {l: 'FAIL_NO_EST',   c: '#ef4444', d: 'No se obtuvo estimaci\u00f3n de pose'},
    {l: 'FAIL_NO_GT',    c: '#ef4444', d: 'No se encontr\u00f3 ground truth del SDK'},
    {l: 'FAIL_NO_DATA',  c: '#f59e0b', d: 'Puntos segmentados < 300'},
    {l: 'FAIL_BAD_ICP',  c: '#f97316', d: 'Fitness < 0.15 o RMSE > 0.02m'},
    {l: 'FAIL_POSE',     c: '#ef4444', d: 'Error traslaci\u00f3n > 0.05m o rotaci\u00f3n > 10\u00b0'}
  ];

  var cg = '<div class="g2">' + cls.map(function(c) {
    return '<div class="ci2" style="border:1px solid ' + c.c + '33">'
      + '<span class="tg" style="background:' + c.c + '22;color:' + c.c + '">' + c.l + '</span>'
      + '<div class="dd">' + c.d + '</div></div>';
  }).join('') + '</div>';

  tm.innerHTML = '<div class="sec"><h2>Resumen de la Sesi\u00f3n</h2>'
    + '<div class="d">' + _SN + ' \u2014 frames ' + _FMIN + '\u2013' + _FMAX + '</div>'
    + ch + '</div>'
    + '<div class="sec"><h2>Gu\u00eda de M\u00e9tricas</h2>'
    + '<div class="d">Referencia para interpretar cada m\u00e9trica</div>' + tb + '</div>'
    + '<div class="sec"><h2>Clasificaci\u00f3n de Frames</h2>'
    + '<div class="d">L\u00f3gica del clasificador pose_eval</div>' + cg + '</div>';
}


// ═══════════════════════════════════════════════════════════════
// PESTAÑA 4: VISUALIZACIÓN 3D Y GRÁFICOS
// ═══════════════════════════════════════════════════════════════
// Trayectoria interactiva Three.js, gráficos SVG, tabla detallada.

/**
 * Construye la estructura HTML de la pestaña de visualización.
 * Los gráficos SVG y la escena 3D se inicializan de forma diferida.
 */
function bV() {
  var tv = document.getElementById('tv');
  tv.innerHTML = '<div class="sec"><h2>Trayectoria 3D del Controlador</h2>'
    + '<div class="d">Pose estimada en espacio mundo Unity</div>'
    + '<div id="tc"></div>'
    + '<div class="lr">'
    + '<div class="li"><div class="dt" style="background:#10b981"></div>Fitness > 0.5</div>'
    + '<div class="li"><div class="dt" style="background:#f59e0b"></div>Fitness 0.3\u20130.5</div>'
    + '<div class="li"><div class="dt" style="background:#ef4444"></div>Fitness < 0.3 / Inv\u00e1lido</div>'
    + '</div>'
    + '<div class="zh">\u{1F5B1}\uFE0F Arrastra para rotar \u00b7 \u2328\uFE0F W/S \u00f3 +/- \u00f3 \u2191/\u2193 para zoom</div>'
    + '</div>'
    + '<div class="sec"><h2>Componentes de Traslaci\u00f3n</h2>'
    + '<div class="d">tx, ty, tz por frame</div>'
    + '<div class="cc"><div class="ct2" style="color:#f8fafc">\u{1F4D0} Traslaci\u00f3n (tx, ty, tz)</div>'
    + '<svg id="ct" class="cht"></svg></div></div>'
    + '<div class="sec"><h2>M\u00e9tricas de Calidad por Frame</h2>'
    + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
    + '<div class="cc"><div class="ct2" style="color:#06b6d4">\u{1F4C8} Fitness</div><svg id="cf" class="ch"></svg></div>'
    + '<div class="cc"><div class="ct2" style="color:#8b5cf6">\u{1F4CF} RMSE</div><svg id="cr" class="ch"></svg></div>'
    + '<div class="cc"><div class="ct2" style="color:#10b981">\u{1F3AF} Center Distance</div><svg id="cc" class="ch"></svg></div>'
    + '<div class="cc"><div class="ct2" style="color:#f59e0b">\u{1F3C5} Score</div><svg id="cs" class="ch"></svg></div>'
    + '</div></div>'
    + '<div class="sec"><h2>Tabla de Trayectoria Detallada</h2><div class="tw" id="tt"></div></div>';
}

/**
 * Construye la tabla HTML con los datos detallados de cada frame.
 * El coloreado refleja la calidad: verde (OK+), amarillo (OK), rojo (FAIL/WARN).
 */
function bT() {
  var h = '<table><thead><tr>'
    + '<th>Frame</th><th>Estado</th><th>Raz\u00f3n</th><th>tx</th><th>ty</th><th>tz</th>'
    + '<th>Fitness</th><th>RMSE</th><th>Score</th><th>Ctr Dist</th><th>SE(3)</th>'
    + '</tr></thead><tbody>';

  D.forEach(function(d) {
    var bc = !d.v ? '#ef4444' : d.fit > 0.5 ? '#10b981' : d.fit > 0.3 ? '#f59e0b' : '#ef4444';
    var bl = !d.v ? 'FAIL' : d.fit > 0.5 ? 'OK+' : d.fit > 0.3 ? 'OK' : 'WARN';
    var fc = d.fit > 0.5 ? '#10b981' : d.fit > 0.3 ? '#f59e0b' : '#ef4444';
    var cc = d.cd !== null ? (d.cd < 0.05 ? '#10b981' : '#ef4444') : '#94a3b8';

    h += '<tr>'
      + '<td style="color:#f8fafc;font-weight:600">' + d.f + '</td>'
      + '<td><span class="bd" style="background:' + bc + '22;color:' + bc + '">' + bl + '</span></td>'
      + '<td style="color:#94a3b8;font-size:9px">' + d.r + '</td>'
      + '<td style="color:#e2e8f0">' + d.tx.toFixed(4) + '</td>'
      + '<td style="color:#e2e8f0">' + d.ty.toFixed(4) + '</td>'
      + '<td style="color:#e2e8f0">' + d.tz.toFixed(4) + '</td>'
      + '<td style="color:' + fc + '">' + d.fit.toFixed(4) + '</td>'
      + '<td style="color:#e2e8f0">' + d.rmse.toFixed(5) + '</td>'
      + '<td style="color:#e2e8f0">' + d.sc.toFixed(4) + '</td>'
      + '<td style="color:' + cc + '">' + (d.cd !== null ? (d.cd * 100).toFixed(2) + 'cm' : '\u2014') + '</td>'
      + '<td style="color:' + (d.se3 ? '#10b981' : '#ef4444') + '">' + (d.se3 ? '\u2713' : '\u2717') + '</td>'
      + '</tr>';
  });

  h += '</tbody></table>';
  document.getElementById('tt').innerHTML = h;
}

/**
 * Renderiza todos los gráficos SVG de la pestaña de visualización.
 * Se invoca de forma diferida (solo cuando el usuario accede a la pestaña).
 */
function rC() {
  SC(document.getElementById('cf'), V, 'fit', '#06b6d4', 0, 1, function(v) { return v.toFixed(2); });
  SC(document.getElementById('cr'), V, 'rmse', '#8b5cf6', void 0, void 0, function(v) { return v.toFixed(4); });
  SC(document.getElementById('cc'), V, 'cd', '#10b981', 0, void 0, function(v) { return (v * 100).toFixed(0) + 'cm'; }, true);
  SC(document.getElementById('cs'), V, 'sc', '#f59e0b', void 0, void 0, function(v) { return v.toFixed(2); });
  ML(document.getElementById('ct'), V, ['tx', 'ty', 'tz'],
     ['#ef4444', '#10b981', '#06b6d4'], ['tx (X)', 'ty (Y)', 'tz (Z)']);
}


// ═══════════════════════════════════════════════════════════════
// ESCENA THREE.JS — Visualización 3D de la trayectoria
// ═══════════════════════════════════════════════════════════════
// Se renderiza la trayectoria del controlador en espacio mundo Unity.
// Los frames válidos se muestran como esferas (coloreadas por fitness), y los inválidos como cruces rojas. La cámara orbita alrededor del centroide de la trayectoria.
// Si Three.js no está disponible (sin conexión), se muestra un mensaje informativo sin interrumpir el funcionamiento del resto del dashboard.

function i3() {
  var el = document.getElementById('tc');
  if (typeof THREE === 'undefined') {
    el.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;'
      + 'color:#94a3b8;font-size:14px;padding:40px;text-align:center">'
      + 'Three.js no disponible (requiere conexi\u00f3n a internet para la vista 3D).'
      + '<br>Los gr\u00e1ficos 2D funcionan sin conexi\u00f3n.</div>';
    return;
  }

  var w = el.clientWidth, h = 500;
  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0f1e);
  var cam = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
  var ren = new THREE.WebGLRenderer({antialias: true});
  ren.setSize(w, h);
  ren.setPixelRatio(window.devicePixelRatio);
  el.appendChild(ren.domElement);

  // Rejilla de referencia
  scene.add(new THREE.GridHelper(4, 20, 0x1e293b, 0x111827));

  // Ejes coordenados (X=rojo, Y=verde, Z=azul)
  function mL(p, c) {
    return new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(p),
      new THREE.LineBasicMaterial({color: c})
    );
  }
  scene.add(mL([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0.5, 0, 0)], 0xef4444));
  scene.add(mL([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0.5, 0)], 0x10b981));
  scene.add(mL([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0.5)], 0x06b6d4));

  // Centroide de la trayectoria válida (target de la cámara orbital)
  var tg = new THREE.Vector3(
    A(V.map(function(d) { return d.tx; })),
    A(V.map(function(d) { return d.ty; })),
    A(V.map(function(d) { return d.tz; }))
  );

  // Línea de trayectoria continua
  var pts = V.map(function(d) { return new THREE.Vector3(d.tx, d.ty, d.tz); });
  if (pts.length > 1) scene.add(mL(pts, 0x06b6d4));

  // Esferas en posiciones válidas (tamaño y color según fitness)
  V.forEach(function(d) {
    var c = d.fit > 0.5 ? 0x10b981 : d.fit > 0.3 ? 0xf59e0b : 0xef4444;
    var sp = new THREE.Mesh(
      new THREE.SphereGeometry(d.fit > 0.5 ? 0.008 : 0.006, 8, 8),
      new THREE.MeshBasicMaterial({color: c})
    );
    sp.position.set(d.tx, d.ty, d.tz);
    scene.add(sp);
  });

  // Cruces rojas en posiciones de frames inválidos
  D.filter(function(d) { return !d.v; }).forEach(function(d) {
    var s = 0.02;
    scene.add(mL([new THREE.Vector3(d.tx - s, d.ty - s, d.tz), new THREE.Vector3(d.tx + s, d.ty + s, d.tz)], 0xef4444));
    scene.add(mL([new THREE.Vector3(d.tx + s, d.ty - s, d.tz), new THREE.Vector3(d.tx - s, d.ty + s, d.tz)], 0xef4444));
  });

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));

  // Control orbital manual (sin OrbitControls — no disponible en r128 CDN)
  var St = {theta: 0.5, phi: 1.0, dist: 2.5};

  function upd() {
    cam.position.set(
      tg.x + St.dist * Math.sin(St.phi) * Math.cos(St.theta),
      tg.y + St.dist * Math.cos(St.phi),
      tg.z + St.dist * Math.sin(St.phi) * Math.sin(St.theta)
    );
    cam.lookAt(tg);
  }
  upd();

  // Interacción con ratón (arrastrar para rotar)
  var md = false, px = 0, py = 0;

  ren.domElement.addEventListener('mousedown', function(e) {
    md = true; px = e.clientX; py = e.clientY;
  });
  ren.domElement.addEventListener('mouseup', function() { md = false; });
  ren.domElement.addEventListener('mousemove', function(e) {
    if (!md) return;
    St.theta += (e.clientX - px) * 0.01;
    St.phi = Math.max(0.2, Math.min(Math.PI - 0.2, St.phi - (e.clientY - py) * 0.01));
    px = e.clientX;
    py = e.clientY;
    upd();
  });

  // Interacción con teclado (zoom)
  window.addEventListener('keydown', function(e) {
    if (e.key === 'w' || e.key === 'W' || e.key === '+' || e.key === 'ArrowUp') {
      St.dist = Math.max(0.3, St.dist - 0.15); upd();
    }
    if (e.key === 's' || e.key === 'S' || e.key === '-' || e.key === 'ArrowDown') {
      St.dist = Math.min(8, St.dist + 0.15); upd();
    }
  });

  // Bucle de renderizado
  function an() { requestAnimationFrame(an); ren.render(scene, cam); }
  an();
}


// ═══════════════════════════════════════════════════════════════
// INICIALIZACIÓN — Se ejecuta al cargar la página
// ═══════════════════════════════════════════════════════════════
// Se construyen las 4 pestañas. 
// La pestaña de Arquitectura es la activa por defecto. 
// La pestaña de Visualización 3D se inicializa
// de forma diferida para evitar cargar Three.js innecesariamente.

bA();    // Pestaña 1: Arquitectura
bF();    // Pestaña 2: Flujo de Datos
bM();    // Pestaña 3: Métricas
bV();    // Pestaña 4: Visualización (estructura HTML solamente)
bT();    // Tabla de trayectoria (dentro de pestaña 4)