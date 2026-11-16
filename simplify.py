import numpy as np
import pymeshlab
import trimesh
from trimesh import grouping

class WeightedQEMSimplifier:
    """Legacy name: vertex scalars are prepared, but QEM quality weighting is off.

    Preserve this setting for the existing experiment protocol. Learned saliency
    affects the subsequent surface refinement in main.py.
    """
    def __init__(self, mesh, weights):
        self.mesh = mesh
        self.weights = np.asarray(weights, dtype=float).copy()

    def simplify(self, target_faces):
        print(f"Initializing PyMeshLab for DENTAL optimized simplification (Target: {target_faces})...")

        # --- 1. 计算简化比例 ---
        original_faces = len(self.mesh.faces)
        if original_faces == 0: return self.mesh

        ratio = target_faces / original_faces
        print(f"  - Ratio: {ratio:.1%} ({target_faces}/{original_faces})")
        # 动态调整简化策略

        # 1. (0% - 8%)
        if ratio <= 0.08:
            feature_multiplier = 3.0 if ratio <= 0.03 else 2.5
            boundary_weight = 1000.0
            clip_min = 0.2
            quality_thr = 0.3
            print(f"    [Strategy] LOW Ratio ({ratio:.2f}) -> Moderate Feature Protection")

        elif ratio <= 0.2:
            feature_multiplier = 2.0
            boundary_weight = 1000.0
            clip_min = 0.2
            quality_thr = 0.3
            print(f"    [Strategy] LOW-MID Ratio ({ratio:.2f}) -> Balanced Protection")
        # 2.  (8% - 30%)
        elif ratio <= 0.3:
            feature_multiplier = 1.5
            boundary_weight = 1000.0
            clip_min = 0.3
            quality_thr = 0.4
            print(f"    [Strategy] MID Ratio ({ratio:.2f}) -> Smoothness & Uniformity First")

        # 3. (> 30%)
        else:
            feature_multiplier = 1.5
            boundary_weight = 1000.0
            clip_min = 0.2
            quality_thr = 0.5
            print(f"    [Strategy] HIGH Ratio ({ratio:.2f}) -> Guided QEM + Quality Lock")

        # --- 2. 识别边界 ---
        edges = self.mesh.edges_sorted
        groups = grouping.group_rows(edges, require_count=1)
        boundary_indices = np.unique(edges[groups].flatten())
        print(f"  - Detected {len(boundary_indices)} boundary vertices.")

        # --- 3. 动态特征阈值 ---
        target_vertices = target_faces / 2
        # 在中高比例下，减少特征预算的比例，让更多点用于平滑
        budget_ratio = 0.7 if ratio <= 0.05 else 0.4

        feature_budget = int(target_vertices * budget_ratio)
        total_vertices = len(self.weights)
        top_percentile = 100 * (1.0 - (feature_budget / (total_vertices + 1e-6)))
        top_percentile = np.clip(top_percentile, 80.0, 99.0)

        adaptive_threshold = np.percentile(self.weights, top_percentile)

        # --- 4. 应用权重 ---

        # A. 基础截断 (应用 clip_min)
        self.weights = np.clip(self.weights, clip_min, 1.0)

        # B. 特征放大
        mask = self.weights > adaptive_threshold
        self.weights[mask] *= feature_multiplier

        # C. 边界加权 (永远锁死)
        self.weights[boundary_indices] = boundary_weight

        # D. 最终截断
        self.weights = np.clip(self.weights, clip_min, boundary_weight)

        # E. [改进] 映射非边界权重到 [1, 10]，与 pn 基线一致的 QEM 权重范围
        non_boundary = np.ones(len(self.weights), dtype=bool)
        non_boundary[boundary_indices] = False
        w_nb = self.weights[non_boundary]
        w_min, w_max = (w_nb.min(), w_nb.max()) if len(w_nb) else (0.0, 0.0)
        if w_max > w_min:
            self.weights[non_boundary] = 1.0 + 9.0 * (w_nb - w_min) / (w_max - w_min)
        else:
            self.weights[non_boundary] = 1.0

        print(f"    Weight Stats: Mean={self.weights.mean():.2f}, Min={self.weights.min():.2f}, Max={self.weights.max():.2f}")

        # --- 5. PyMeshLab 执行 ---
        vertices = self.mesh.vertices.astype(np.float64)
        faces = self.mesh.faces.astype(np.int32)
        weights_expanded = self.weights.astype(np.float64).reshape(-1, 1)

        m = pymeshlab.Mesh(vertices, faces, v_scalar_array=weights_expanded)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "dental_mesh")

        try:
            ms.apply_filter('cleaning_repair_non_manifold_edges')
            ms.apply_filter('cleaning_repair_non_manifold_vertices')
        except: pass

        print(f"Simplifying...")
        try:
            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetfacenum=target_faces,

                qualityweight=False,  # Explicitly retain the existing unweighted QEM protocol.

                # 边界保护
                preserveboundary=True,
                boundaryweight=1000.0,

                # 开启所有优化
                preservenormal=True,
                planarquadric=True,
                optimalplacement=True,

                # 动态质量阈值
                qualitythr=quality_thr,

                autoclean=True
            )

        except Exception as e:
            print(f"Simplification Error: {e}")
            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetfacenum=target_faces,
                preserveboundary=True,
                boundaryweight=1000.0
            )

        return self._export_trimesh(ms)

    def _export_trimesh(self, ms):
        out_mesh = ms.current_mesh()
        return trimesh.Trimesh(vertices=out_mesh.vertex_matrix(), faces=out_mesh.face_matrix())
