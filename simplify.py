import numpy as np
import pymeshlab
import trimesh
from trimesh import grouping

class WeightedQEMSimplifier:

    def __init__(self, mesh, weights):
        self.mesh = mesh
        self.weights = np.asarray(weights, dtype=float).copy()

    def simplify(self, target_faces):
        print(f"Initializing PyMeshLab for DENTAL optimized simplification (Target: {target_faces})...")


        original_faces = len(self.mesh.faces)
        if original_faces == 0: return self.mesh

        ratio = target_faces / original_faces
        print(f"  - Ratio: {ratio:.1%} ({target_faces}/{original_faces})")


        edges = self.mesh.edges_sorted
        groups = grouping.group_rows(edges, require_count=1)
        boundary_indices = np.unique(edges[groups].flatten())
        print(f"  - Detected {len(boundary_indices)} boundary vertices.")


        target_vertices = target_faces / 2

        budget_ratio = 0.7 if ratio <= 0.05 else 0.4

        feature_budget = int(target_vertices * budget_ratio)
        total_vertices = len(self.weights)
        top_percentile = 100 * (1.0 - (feature_budget / (total_vertices + 1e-6)))
        top_percentile = np.clip(top_percentile, 80.0, 99.0)

        adaptive_threshold = np.percentile(self.weights, top_percentile)


        self.weights = np.clip(self.weights, clip_min, 1.0)


        mask = self.weights > adaptive_threshold
        self.weights[mask] *= feature_multiplier


        self.weights[boundary_indices] = boundary_weight


        self.weights = np.clip(self.weights, clip_min, boundary_weight)


        non_boundary = np.ones(len(self.weights), dtype=bool)
        non_boundary[boundary_indices] = False
        w_nb = self.weights[non_boundary]
        w_min, w_max = (w_nb.min(), w_nb.max()) if len(w_nb) else (0.0, 0.0)
        if w_max > w_min:
            self.weights[non_boundary] = 1.0 + 9.0 * (w_nb - w_min) / (w_max - w_min)
        else:
            self.weights[non_boundary] = 1.0

        print(f"    Weight Stats: Mean={self.weights.mean():.2f}, Min={self.weights.min():.2f}, Max={self.weights.max():.2f}")


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

                qualityweight=False,


                preserveboundary=True,
                boundaryweight=1000.0,


                preservenormal=True,
                planarquadric=True,
                optimalplacement=True,


                qualitythr=quality_thr,

                autoclean=True
            )

        except Exception as e:
            print(f"Simplification Error: {e}")
            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetfacenum=target_faces,
                preserveboundary=True,
                boundaryweight=10.0
            )

        return self._export_trimesh(ms)

    def _export_trimesh(self, ms):
        out_mesh = ms.current_mesh()
        return trimesh.Trimesh(vertices=out_mesh.vertex_matrix(), faces=out_mesh.face_matrix())
