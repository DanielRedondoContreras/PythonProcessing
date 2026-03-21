# ============================================================
# File: src/core/ppf_match.py
# Estimación de pose 6-DOF mediante PPF + ICP (OpenCV Surface Matching)
#
# Implementa el pipeline recomendado por OpenCV (Drost et al. 2010):
#   1. Entrenamiento PPF: construye hash table de point-pair features
#      del modelo CAD (offline, pero se re-entrena cada invocación
#      porque el detector no es serializable desde Python).
#   2. Matching PPF: vota en espacio de poses 2D por cada par de
#      puntos de la escena contra el modelo entrenado.
#   3. Refinamiento ICP: point-to-plane ICP multi-nivel sobre las
#      top-N hipótesis de pose del votador PPF.
#   4. Evaluación: fitness/RMSE consistentes con Open3D para
#      compatibilidad con el pipeline de evaluación existente.
#
# Pre-filtrado de escena (conservado del pipeline anterior):
#   - Crop esférico por radio alrededor del centroide
#   - View clamp por eje de cámara (near/far)
#   - Filtro de densidad local (percentil)
#
# Referencia:
#   https://docs.opencv.org/4.x/d9/d25/group__surface__matching.html
# ============================================================

import json
import numpy as np
import cv2
import open3d as o3d

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.transforms import pose_dict_to_T44


# ---------------------------------------------------------------
# Sentinels para candidatos sin matching válido.
# (importados por run_pose_eval.py — no eliminar)
# ---------------------------------------------------------------
SENTINEL_RMSE: float = 1e9
SENTINEL_SCORE: float = -1e9


# ---------------------------------------------------------------
# Proyección a SO(3) — FIX para matrices no-ortogonales de ICP
# ---------------------------------------------------------------

def _project_to_SO3(T: np.ndarray) -> np.ndarray:
    """
    Proyecta la submatriz de rotación de una T 4×4 sobre SO(3)
    usando descomposición SVD (Polar Decomposition).

    Necesario porque cv2.ppf_match_3d_ICP puede devolver matrices
    con R no ortogonal (acumulación de error numérico en la
    linearización point-to-plane iterativa multi-nivel).

    Garantiza det(R)=+1 y ortogonalidad exacta.
    """
    T = np.array(T, dtype=np.float64)
    R = T[:3, :3].copy()
    U, S, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    T[:3, :3] = R_proj
    return T


# ---------------------------------------------------------------
# Utilidades de conversión Open3D ↔ OpenCV PPF
# ---------------------------------------------------------------

def pcd_to_nx6(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convierte Open3D PointCloud (con normales) a Nx6 float32 para OpenCV PPF."""
    pts = np.asarray(pcd.points, dtype=np.float32)
    if not pcd.has_normals():
        raise ValueError("PointCloud debe tener normales para PPF matching.")
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    return np.hstack([pts, nrm]).astype(np.float32)


def pca_axis(points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve (axis0, evals) donde axis0 es el eje principal (unit vector)."""
    if points_xyz.shape[0] < 10:
        return np.array([0, 1, 0], dtype=np.float64), np.array([0, 0, 0], dtype=np.float64)
    c = points_xyz.mean(axis=0)
    X = points_xyz - c
    C = (X.T @ X) / max(points_xyz.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    axis0 = evecs[:, 0]
    axis0 = axis0 / (np.linalg.norm(axis0) + 1e-12)
    return axis0, evals


# ---------------------------------------------------------------
# PPFMatcher
# ---------------------------------------------------------------

class PPFMatcher:
    """
    Estimación de pose 6-DOF mediante OpenCV Surface Matching (PPF + ICP).

    Pipeline:
      1. Carga modelo y escena como nubes Open3D
      2. Pre-filtra la escena (crop, view clamp, densidad)
      3. Asegura normales en ambas nubes
      4. Convierte a Nx6 float32 para OpenCV
      5. Entrena PPF3DDetector con el modelo
      6. Ejecuta detector.match() sobre la escena
      7. Refina top-N hipótesis con ICP point-to-plane multi-nivel
      8. Evalúa fitness/RMSE con Open3D para compatibilidad downstream
      9. Devuelve la mejor pose en el formato esperado por pose_eval
    """

    def __init__(
        self,
        model_ply_path: str,
        scene_ply_path: str,
        metadata_path: Optional[str] = None,

        # --- PPF detector ---
        ppf_rel_sampling: float = 0.05,
        ppf_rel_distance: float = 0.05,

        # --- PPF matching ---
        ppf_scene_sample_step: float = 0.025,
        ppf_scene_distance: float = 0.05,
        ppf_top_n: int = 5,

        # --- ICP refinement (cv2.ppf_match_3d_ICP) ---
        icp_iterations: int = 100,
        icp_tolerance: float = 0.005,
        icp_rejection_scale: float = 2.5,
        icp_num_levels: int = 4,

        # --- Scene pre-filtering ---
        crop_radius: float = 0.14,
        crop_min_points: int = 300,
        crop_center: Optional[np.ndarray] = None,
        view_clamp_enable: bool = True,
        view_clamp_near: float = 0.10,
        view_clamp_far: float = 0.60,
        use_right_camera: bool = True,
        density_enable: bool = True,
        density_radius: float = 0.02,
        density_min_percentile: float = 5.0,
        density_max_percentile: float = 95.0,

        # --- Diagnóstico ---
        save_subclouds: bool = True,

        # --- Evaluación (voxel para evaluate_registration) ---
        eval_voxel: float = 0.005,
    ):
        self.model_path = Path(model_ply_path)
        self.scene_path = Path(scene_ply_path)
        self.metadata_path = Path(metadata_path) if metadata_path else None

        # PPF detector
        self.ppf_rel_sampling = float(ppf_rel_sampling)
        self.ppf_rel_distance = float(ppf_rel_distance)

        # PPF matching
        self.ppf_scene_sample_step = float(ppf_scene_sample_step)
        self.ppf_scene_distance = float(ppf_scene_distance)
        self.ppf_top_n = int(ppf_top_n)

        # ICP
        self.icp_iterations = int(icp_iterations)
        self.icp_tolerance = float(icp_tolerance)
        self.icp_rejection_scale = float(icp_rejection_scale)
        self.icp_num_levels = int(icp_num_levels)

        # Pre-filtering
        self.crop_radius = float(crop_radius)
        self.crop_min_points = int(crop_min_points)
        self.crop_center = np.asarray(crop_center, dtype=np.float64).ravel() if crop_center is not None else None
        self.view_clamp_enable = bool(view_clamp_enable)
        self.view_clamp_near = float(view_clamp_near)
        self.view_clamp_far = float(view_clamp_far)
        self.use_right_camera = bool(use_right_camera)
        self.density_enable = bool(density_enable)
        self.density_radius = float(density_radius)
        self.density_min_percentile = float(density_min_percentile)
        self.density_max_percentile = float(density_max_percentile)

        self.save_subclouds = bool(save_subclouds)
        self.eval_voxel = float(eval_voxel)

    # ============================================================
    # I/O
    # ============================================================

    def _load(self, path: Path) -> o3d.geometry.PointCloud:
        if not path.exists():
            raise FileNotFoundError(path)
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise ValueError(f"Nube vacía: {path}")
        return pcd

    def _load_metadata(self) -> Optional[Dict]:
        if self.metadata_path is None or not self.metadata_path.exists():
            return None
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    # ============================================================
    # PRE-FILTRADO DE ESCENA
    # ============================================================

    @staticmethod
    def _centroid(pcd: o3d.geometry.PointCloud) -> np.ndarray:
        return np.mean(np.asarray(pcd.points), axis=0)

    def _crop_scene_by_radius(self, scene: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if self.crop_radius <= 0.0:
            return scene
        n0 = len(scene.points)
        if n0 == 0:
            return scene

        # FIX CAUSA #5: usar centro GT del controlador si está disponible,
        # en lugar del centroide geométrico de la escena (que puede estar
        # desplazado por ruido o puntos de fondo).
        if self.crop_center is not None:
            c = self.crop_center
            c_source = "gt_controller"
        else:
            c = self._centroid(scene)
            c_source = "scene_centroid"

        pts = np.asarray(scene.points)
        d2 = np.sum((pts - c) ** 2, axis=1)
        mask = d2 <= (self.crop_radius ** 2)
        idx = np.where(mask)[0].astype(np.int32)

        if idx.size < self.crop_min_points:
            print(f"[CROP-R] skip: before={n0} after={idx.size} r={self.crop_radius:.3f} center={c_source}")
            return scene

        cropped = scene.select_by_index(idx.tolist())
        print(f"[CROP-R] applied: before={n0} after={len(cropped.points)} r={self.crop_radius:.3f} center={c_source}")
        return cropped

    def _view_clamp(self, scene: o3d.geometry.PointCloud, meta: Optional[Dict]) -> o3d.geometry.PointCloud:
        if not self.view_clamp_enable or meta is None:
            return scene

        cam_key = "CameraPoseRight" if self.use_right_camera else "CameraPoseLeft"
        if cam_key not in meta:
            print(f"[CLAMP-Z] skip: '{cam_key}' no encontrado en metadata")
            return scene

        T_w_c = pose_dict_to_T44(meta[cam_key])
        T_c_w = np.linalg.inv(T_w_c)

        pts_w = np.asarray(scene.points)
        pts_c = (T_c_w[:3, :3] @ pts_w.T + T_c_w[:3, 3:4]).T

        z = pts_c[:, 2]
        mask = (z >= self.view_clamp_near) & (z <= self.view_clamp_far)
        idx = np.where(mask)[0].astype(np.int32)

        if idx.size < self.crop_min_points:
            print(f"[CLAMP-Z] skip: after={idx.size} (min={self.crop_min_points})")
            return scene

        clamped = scene.select_by_index(idx.tolist())
        print(f"[CLAMP-Z] applied: before={len(scene.points)} after={len(clamped.points)}")
        return clamped

    def _density_filter(self, scene: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if not self.density_enable:
            return scene
        n0 = len(scene.points)
        if n0 < 50:
            return scene

        kdt = o3d.geometry.KDTreeFlann(scene)
        r = self.density_radius
        counts = np.zeros(n0, dtype=np.int32)
        for i in range(n0):
            _, idx, _ = kdt.search_radius_vector_3d(scene.points[i], r)
            counts[i] = max(len(idx) - 1, 0)

        lo = np.percentile(counts, self.density_min_percentile)
        hi = np.percentile(counts, self.density_max_percentile)
        mask = (counts >= lo) & (counts <= hi)
        idx = np.where(mask)[0].astype(np.int32)

        if idx.size < self.crop_min_points:
            print(f"[DENS] skip: after={idx.size} (min={self.crop_min_points})")
            return scene

        out = scene.select_by_index(idx.tolist())
        print(f"[DENS] applied: before={n0} after={len(out.points)}")
        return out

    # ============================================================
    # NORMALES
    # ============================================================

    @staticmethod
    def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: float = 0.01, force_orient: bool = False) -> None:
        """
        Estima normales solo si la nube no las tiene.
        
        Si la nube ya tiene normales (ej: del pipeline de segmentación o del modelo CAD),
        solo se normalizan sin re-orientar, para preservar la consistencia establecida
        previamente. Esto evita que orient_normals_towards_camera_location destruya
        la orientación coherente que el upstream ya estableció.
        
        Si force_orient=True, orienta hacia el origen (útil solo para la primera vez).
        """
        has_valid = (pcd.has_normals() and 
                     np.asarray(pcd.normals).shape[0] == np.asarray(pcd.points).shape[0])
        
        if not has_valid:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
            )
            pcd.normalize_normals()
            # Orientar solo cuando acabamos de estimar (primera vez)
            pcd.orient_normals_towards_camera_location(np.zeros(3))
        else:
            pcd.normalize_normals()
            if force_orient:
                pcd.orient_normals_towards_camera_location(np.zeros(3))

    # ============================================================
    # SUBCLOUDS (diagnóstico)
    # ============================================================

    def _save_subclouds(self, scene: o3d.geometry.PointCloud, outdir: Path) -> None:
        if not self.save_subclouds:
            return
        n = len(scene.points)
        if n < 200:
            return

        pts = np.asarray(scene.points)
        c = pts.mean(axis=0)
        axis0, _ = pca_axis(pts)

        proj = (pts - c) @ axis0
        proj_min, proj_max = float(proj.min()), float(proj.max())
        proj_norm = (proj - proj_min) / (proj_max - proj_min + 1e-12)

        v = pts - c
        radial = np.linalg.norm(v - np.outer(proj, axis0), axis=1)

        handle_mask = (proj_norm <= 0.55) & (radial <= np.percentile(radial, 60))
        ring_mask = (proj_norm >= 0.65) & (radial >= np.percentile(radial, 70))
        face_mask = ~(handle_mask | ring_mask)

        def _write(mask, name):
            idx = np.where(mask)[0].astype(np.int32).tolist()
            if len(idx) < 50:
                return
            sub = scene.select_by_index(idx)
            o3d.io.write_point_cloud(str(outdir / f"scene_{name}.ply"), sub)

        _write(handle_mask, "handle")
        _write(ring_mask, "ring")
        _write(face_mask, "face")

    # ============================================================
    # RESULTADO VACÍO
    # ============================================================

    @staticmethod
    def _empty_result(reason: str = "no_matches") -> Dict:
        return {
            "transformation": np.eye(4),
            "fitness": 0.0,
            "rmse": SENTINEL_RMSE,
            "score": SENTINEL_SCORE,
            "debug": {
                "ppf_match_failed": True,
                "fail_reason": reason,
                "n_results_raw": 0,
                "n_results_refined": 0,
            },
        }

    # ============================================================
    # RUN
    # ============================================================

    def run(self, output_dir: Optional[Path] = None) -> Dict:
        # 1) Carga
        model = self._load(self.model_path)
        scene = self._load(self.scene_path)
        meta = self._load_metadata()

        n_scene_raw = len(scene.points)

        # 2) Pre-filtrado de escena
        scene = self._crop_scene_by_radius(scene)
        scene = self._view_clamp(scene, meta)
        scene = self._density_filter(scene)

        n_scene_filtered = len(scene.points)

        if n_scene_filtered < self.crop_min_points:
            print(f"[PPF] Escena insuficiente tras filtrado: {n_scene_filtered} puntos")
            return self._empty_result("scene_too_small")

        # Subclouds (diagnóstico)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_subclouds(scene, output_dir)

        # 3) Normales
        #    Modelo: force_orient=True para garantizar orientación consistente
        #    Escena: preservar normales del pipeline de segmentación (ya orientadas)
        self._ensure_normals(model, radius=0.005, force_orient=True)
        self._ensure_normals(scene, radius=0.01, force_orient=False)

        # 4) Conversión a Nx6 float32
        model_nx6 = pcd_to_nx6(model)
        scene_nx6 = pcd_to_nx6(scene)

        n_model = model_nx6.shape[0]
        n_scene = scene_nx6.shape[0]

        print(f"[PPF] modelo={n_model} pts, escena={n_scene} pts "
              f"(raw={n_scene_raw}, filtrada={n_scene_filtered})")

        # 5) Entrenamiento PPF
        print(f"[PPF] Entrenando detector (rel_sampling={self.ppf_rel_sampling}, "
              f"rel_distance={self.ppf_rel_distance})...")
        detector = cv2.ppf_match_3d_PPF3DDetector(
            self.ppf_rel_sampling,
            self.ppf_rel_distance,
        )
        detector.trainModel(model_nx6)

        # 6) Matching PPF
        print(f"[PPF] Matching (scene_sample={self.ppf_scene_sample_step}, "
              f"scene_dist={self.ppf_scene_distance})...")
        try:
            results = detector.match(
                scene_nx6,
                self.ppf_scene_sample_step,
                self.ppf_scene_distance,
            )
        except cv2.error as e:
            print(f"[PPF] Error en matching: {e}")
            return self._empty_result("ppf_match_error")

        if results is None or len(results) == 0:
            print("[PPF] Sin hipótesis de pose del votador PPF.")
            return self._empty_result("no_ppf_hypotheses")

        n_raw = len(results)
        print(f"[PPF] {n_raw} hipótesis de pose encontradas.")

        # 7) Refinamiento ICP sobre top-N
        top_n = min(self.ppf_top_n, n_raw)
        print(f"[PPF] Refinando top-{top_n} con ICP "
              f"(iter={self.icp_iterations}, levels={self.icp_num_levels})...")

        icp = cv2.ppf_match_3d_ICP(
            self.icp_iterations,
            self.icp_tolerance,
            self.icp_rejection_scale,
            self.icp_num_levels,
        )
        try:
            _, results_refined = icp.registerModelToScene(
                model_nx6, scene_nx6, results[:top_n]
            )
        except cv2.error as e:
            print(f"[PPF] Error en ICP refinement: {e}")
            # Usar resultado PPF sin refinar
            results_refined = results[:top_n]

        if results_refined is None or len(results_refined) == 0:
            print("[PPF] ICP no produjo resultados válidos.")
            return self._empty_result("icp_failed")

        n_refined = len(results_refined)

        # 8) Proyectar TODOS los candidatos a SO(3) + evaluar con Open3D
        #    Seleccionar por fitness post-ICP (no por numVotes del votador PPF)
        model_down = model.voxel_down_sample(self.eval_voxel)
        scene_down = scene.voxel_down_sample(self.eval_voxel)

        candidates_info = []
        best_idx = 0
        best_fitness = -1.0
        best_rmse = float('inf')

        for i, r in enumerate(results_refined):
            T_i = _project_to_SO3(np.array(r.pose, dtype=np.float64))
            ev_i = o3d.pipelines.registration.evaluate_registration(
                model_down, scene_down, self.eval_voxel * 1.5, T_i,
            )
            f_i = float(ev_i.fitness)
            rmse_i = float(ev_i.inlier_rmse)

            candidates_info.append({
                "rank": i,
                "numVotes": int(r.numVotes),
                "residual": float(r.residual),
                "fitness": f_i,
                "rmse": rmse_i,
            })

            # Selección por fitness (desempate por menor RMSE)
            if f_i > best_fitness or (f_i == best_fitness and rmse_i < best_rmse):
                best_idx = i
                best_fitness = f_i
                best_rmse = rmse_i

        best = results_refined[best_idx]
        T_best = _project_to_SO3(np.array(best.pose, dtype=np.float64))
        fitness = best_fitness
        rmse = best_rmse

        print(f"[PPF] Mejor pose (rank={best_idx}): votes={best.numVotes}, "
              f"residual={best.residual:.6f}, fitness={fitness:.4f}, rmse={rmse:.6f}")

        out = {
            "transformation": T_best,
            "fitness": fitness,
            "rmse": rmse,
            "score": float(best.numVotes),
            "debug": {
                "method": "ppf_icp_opencv",
                "ppf_rel_sampling": self.ppf_rel_sampling,
                "ppf_rel_distance": self.ppf_rel_distance,
                "ppf_scene_sample_step": self.ppf_scene_sample_step,
                "ppf_scene_distance": self.ppf_scene_distance,
                "ppf_top_n": self.ppf_top_n,
                "icp_iterations": self.icp_iterations,
                "icp_tolerance": self.icp_tolerance,
                "icp_rejection_scale": self.icp_rejection_scale,
                "icp_num_levels": self.icp_num_levels,
                "crop_radius": self.crop_radius,
                "view_clamp_enable": self.view_clamp_enable,
                "density_enable": self.density_enable,
                "n_model": n_model,
                "n_scene_raw": n_scene_raw,
                "n_scene_filtered": n_scene_filtered,
                "n_results_raw": n_raw,
                "n_results_refined": n_refined,
                "top_result_votes": int(best.numVotes),
                "top_result_residual": float(best.residual),
                "selected_rank": best_idx,
                "selection_criterion": "max_fitness",
                "eval_voxel": self.eval_voxel,
                "candidates": candidates_info,
            },
        }
        return out