import heapq
import math

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

        elif ratio <= 0.3:
            feature_multiplier = 1.5
            boundary_weight = 1000.0
            clip_min = 0.3
            quality_thr = 0.4
            print(f"    [Strategy] MID Ratio ({ratio:.2f}) -> Smoothness & Uniformity First")


        else:
            feature_multiplier = 1.5
            boundary_weight = 1000.0
            clip_min = 0.2
            quality_thr = 0.5
            print(f"    [Strategy] HIGH Ratio ({ratio:.2f}) -> Guided QEM + Quality Lock")


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
                boundaryweight=1000.0
            )

        return self._export_trimesh(ms)

    def _export_trimesh(self, ms):
        out_mesh = ms.current_mesh()
        return trimesh.Trimesh(vertices=out_mesh.vertex_matrix(), faces=out_mesh.face_matrix())


def protection_multiplier(ratio, *, eta, r0, gamma):
    """Compute the saliency multiplier in manuscript Eq. (16)."""
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('eta must be finite and positive')
    if not 0 < r0 < 1:
        raise ValueError('r0 must be in (0, 1)')
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma must be finite and nonnegative')
    if not 0 < ratio <= 1:
        raise ValueError('face retention ratio must be in (0, 1]')
    if ratio >= r0:
        return gamma
    exponent = (1 - ratio) / eta
    if exponent > math.log(np.finfo(float).max):
        raise ValueError('eta is too small for the requested face retention ratio')
    return math.exp(exponent)


class PaperQEMSimplifier:
    """Greedy QEM with the endpoint saliency costs and boundary rule of the manuscript."""

    def __init__(self, mesh, weights, *, eta, r0, gamma):
        self.mesh = mesh
        self.weights = np.asarray(weights, dtype=float).reshape(-1).copy()
        if (len(self.weights) != len(mesh.vertices) or not np.isfinite(self.weights).all()
                or np.any(self.weights < 0) or np.any(self.weights > 1)):
            raise ValueError('Expected one saliency score in [0, 1] per input vertex')
        self.eta = eta
        self.r0 = r0
        self.gamma = gamma

    def simplify(self, target_faces, *, ratio=None):
        input_faces = len(self.mesh.faces)
        if not 0 < target_faces <= input_faces:
            raise ValueError('Target faces must be between 1 and the input face count')
        ratio = target_faces / input_faces if ratio is None else ratio
        multiplier = protection_multiplier(ratio, eta=self.eta, r0=self.r0, gamma=self.gamma)
        if target_faces == input_faces:
            return self.mesh.copy()
        return _PaperCollapse(self.mesh, self.weights, multiplier).run(target_faces)


class _PaperCollapse:
    def __init__(self, mesh, weights, multiplier):
        self.source = np.asarray(mesh.vertices)
        self.center = self.source.mean(axis=0)
        self.scale = np.ptp(self.source, axis=0).max()
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Expected a finite mesh with nonzero extent')
        self.vertices = (self.source - self.center) / self.scale
        self.faces = np.array(mesh.faces, dtype=np.intp, copy=True)
        self.live = np.ones(len(self.faces), dtype=bool)
        self.face_count = len(self.faces)
        self.weights = weights.copy()
        self.multiplier = multiplier

        edges = mesh.edges_unique
        edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edges))
        self.boundary = np.zeros(len(self.vertices), dtype=bool)
        self.boundary[edges[edge_counts == 1].ravel()] = True
        self.weights[self.boundary] = 10.0
        self.nonmanifold = np.zeros(len(self.vertices), dtype=bool)
        self.nonmanifold[edges[edge_counts > 2].ravel()] = True

        self.incident = [set() for _ in self.vertices]
        self.neighbors = [set() for _ in self.vertices]
        for index, triangle in enumerate(self.faces.tolist()):
            for vertex in triangle:
                self.incident[vertex].add(index)
        for a, b in edges.tolist():
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)
        self.revision = [0] * len(self.vertices)
        self.rejected = {}
        self.fallback_tokens = {}
        self.next_token = 0

        triangles = self.vertices[self.faces]
        self.normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(self.normals, axis=1)
        unit = self.normals / np.where(length > 0, length, 1.0)[:, None]
        planes = np.column_stack((unit, -np.einsum('ij,ij->i', unit, triangles[:, 0])))
        plane_quadrics = planes[:, :, None] * planes[:, None, :]
        self.quadrics = np.zeros((len(self.vertices), 4, 4))
        for corner in self.faces.T:
            np.add.at(self.quadrics, corner, plane_quadrics)

        self.queue = []
        for block_start in range(0, len(edges), 65536):
            block = edges[block_start:block_start + 65536]
            self.queue.extend(self._candidates(block))
        heapq.heapify(self.queue)

    def _allowed(self, a, b):
        return ~(self.nonmanifold[a] | self.nonmanifold[b] | (self.boundary[a] & self.boundary[b]))

    def _cost(self, quadrics, positions, a, b):
        homogeneous = np.pad(positions, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
        error = np.einsum('npi,nij,npj->np', homogeneous, quadrics, homogeneous)
        weight = np.maximum(self.weights[a], self.weights[b])
        # For large multipliers, divide every priority by the same constant
        # to keep the ordering in Eq. (15) without overflowing.
        if self.multiplier > 1:
            factor = 1 / self.multiplier + weight
        else:
            factor = 1 + self.multiplier * weight
        return np.maximum(error, 0) * factor[:, None]

    def _positions(self, a, b):
        quadrics = self.quadrics[a] + self.quadrics[b]
        start, end = self.vertices[a], self.vertices[b]
        middle = (start + end) * 0.5
        matrix = quadrics[:, :3, :3]
        best = middle.copy()
        magnitude = np.max(np.abs(matrix), axis=(1, 2))
        solvable = np.abs(np.linalg.det(matrix)) > 1e-12 * magnitude ** 3
        best[solvable] = np.linalg.solve(matrix[solvable], -quadrics[solvable, :3, 3:4])[:, :, 0]
        positions = np.stack((best, start, end, middle), axis=1)
        positions[self.boundary[a]] = start[self.boundary[a], None, :]
        positions[self.boundary[b]] = end[self.boundary[b], None, :]
        return positions, self._cost(quadrics, positions, a, b)

    def _entry(self, a, b, cost, position, token=0):
        return (cost, a, b, self.revision[a], self.revision[b], token, *position)

    def _candidates(self, edges):
        if not len(edges):
            return []
        a, b = np.asarray(edges, dtype=np.intp).T
        allowed = self._allowed(a, b)
        if not allowed.any():
            return []
        a, b = a[allowed], b[allowed]
        positions, costs = self._positions(a, b)
        rows = np.arange(len(a))
        best = costs.argmin(axis=1)
        return [self._entry(left, right, cost, position)
                for left, right, cost, position in zip(a.tolist(), b.tolist(),
                                                       costs[rows, best].tolist(), positions[rows, best].tolist())]

    def _current(self, entry):
        _, a, b, revision_a, revision_b, token, *_ = entry
        return (self.revision[a] == revision_a and self.revision[b] == revision_b
                and (token == 0 or self.fallback_tokens.get((a, b)) == token))

    def _normals_at(self, face_ids, a, b, positions):
        faces = self.faces[face_ids]
        triangles = np.repeat(self.vertices[faces][None, :], len(positions), axis=0)
        triangles[:, (faces == a) | (faces == b)] = positions[:, None, :]
        return np.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0])

    def _reject(self, a, b):
        self.rejected.setdefault(a, set()).add(b)
        self.rejected.setdefault(b, set()).add(a)

    def _collapse(self, keep, drop, position, common, retained, normals):
        affected = self.neighbors[keep] | self.neighbors[drop] | {keep, drop}
        for index in common:
            for vertex in self.faces[index].tolist():
                self.incident[vertex].remove(index)
        self.live[list(common)] = False
        self.face_count -= len(common)

        for index in self.incident[drop]:
            row = self.faces[index]
            row[row == drop] = keep
            self.incident[keep].add(index)
        self.incident[drop].clear()
        new_neighbors = set(self.faces[list(self.incident[keep])].ravel().tolist()) - {keep}
        for vertex in self.neighbors[drop]:
            self.neighbors[vertex].remove(drop)
        for vertex in self.neighbors[keep] - new_neighbors:
            self.neighbors[vertex].discard(keep)
        for vertex in new_neighbors:
            self.neighbors[vertex].add(keep)
        self.neighbors[keep] = new_neighbors
        self.neighbors[drop].clear()

        self.vertices[keep] = position
        self.weights[keep] = max(self.weights[keep], self.weights[drop])
        self.quadrics[keep] += self.quadrics[drop]
        self.normals[retained] = normals
        self.revision[keep] += 1
        self.revision[drop] += 1

        # Pending costs only change at the surviving endpoint. Previously
        # rejected edges in the changed triangles may become legal again.
        refresh = {tuple(sorted((keep, vertex))) for vertex in new_neighbors}
        for vertex in affected:
            for other in self.rejected.pop(vertex, ()):
                self.rejected[other].discard(vertex)
                edge = tuple(sorted((vertex, other)))
                self.fallback_tokens.pop(edge, None)
                if other in self.neighbors[vertex]:
                    refresh.add(edge)
        for entry in self._candidates(sorted(refresh)):
            heapq.heappush(self.queue, entry)

    def _try_collapse(self, entry):
        _, a, b, _, _, _, x, y, z = entry
        common = self.incident[a] & self.incident[b]
        shared_neighbors = self.neighbors[a] & self.neighbors[b]
        if (len(common) != 2 or len(shared_neighbors) != 2
                or self.face_count <= len(common)):
            self._reject(a, b)
            return
        # A tetrahedron (or an equivalent local link) would create duplicate
        # triangles on collapse even though the shared neighbor count is two.
        link_a, link_b = shared_neighbors
        link_faces = self.incident[link_a] & self.incident[link_b]
        if link_faces & self.incident[a] and link_faces & self.incident[b]:
            self._reject(a, b)
            return

        retained = sorted(self.incident[a] ^ self.incident[b])
        position = np.array([x, y, z])
        normals = self._normals_at(retained, a, b, position[None, :])[0]
        old_normals = self.normals[retained]
        if np.any(np.einsum('ij,ij->i', normals, old_normals) <= 0):
            positions, costs = self._positions(np.array([a]), np.array([b]))
            candidates = positions[0]
            alternatives = self._normals_at(retained, a, b, candidates)
            valid = np.all(np.einsum('pij,ij->pi', alternatives, old_normals) > 0, axis=1)
            self._reject(a, b)
            if valid.any():
                best = np.argmin(np.where(valid, costs[0], np.inf))
                self.next_token += 1
                self.fallback_tokens[(a, b)] = self.next_token
                fallback = self._entry(a, b, costs[0, best], candidates[best].tolist(), self.next_token)
                heapq.heappush(self.queue, fallback)
            return

        keep, drop = (b, a) if self.boundary[b] else (a, b)
        self._collapse(keep, drop, position, common, retained, normals)

    def run(self, target_faces):
        while self.face_count > target_faces and self.queue:
            candidate = heapq.heappop(self.queue)
            if self._current(candidate):
                self._try_collapse(candidate)
            if len(self.queue) > 3 * self.face_count + 1024:
                self.queue = [entry for entry in self.queue if self._current(entry)]
                heapq.heapify(self.queue)

        if self.face_count > target_faces:
            print(f'QEM stopped at {self.face_count} faces because no valid collapse remains '
                  f'(target: {target_faces})')
        coordinates = self.vertices * self.scale + self.center
        locked = self.boundary | self.nonmanifold
        coordinates[locked] = self.source[locked]
        live_faces = self.faces[self.live]
        used, indices = np.unique(live_faces, return_inverse=True)
        return trimesh.Trimesh(vertices=coordinates[used], faces=indices.reshape(-1, 3), process=False)
