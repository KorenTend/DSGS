"""Normal-direction fitting of a decimated surface to the original scan."""
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

_FACE_QUADRATURE = np.array([[2 / 3, 1 / 6, 1 / 6],
                             [1 / 6, 2 / 3, 1 / 6],
                             [1 / 6, 1 / 6, 2 / 3]])


def _angles(triangles):
    # Stabilized angle deficit from the incident triangle angles.
    edges = np.roll(triangles, -1, axis=1) - triangles
    edges /= np.linalg.norm(edges, axis=2)[:, :, None] + 1e-8
    cosine = -(edges * np.roll(edges, 1, axis=1)).sum(axis=2)
    return np.arccos(cosine.clip(-1, 1))


def _deficit(mesh):
    sums = np.bincount(mesh.faces.ravel(), _angles(mesh.triangles).ravel(),
                       minlength=len(mesh.vertices))
    return 2 * np.pi - sums


def _closest_surface(mesh, points):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype='float32')),
                        o3d.core.Tensor(np.asarray(mesh.faces, dtype='uint32')))
    return scene.compute_closest_points(o3d.core.Tensor(points.astype('float32')))


def _quadrature(mesh, dense=False):
    if not dense:
        return mesh.triangles_center, mesh.area_faces
    points = np.einsum('qk,fkj->fqj', _FACE_QUADRATURE, mesh.triangles).reshape(-1, 3)
    return points, np.repeat(mesh.area_faces / 3, 3)


def _normal_step(original, mesh, symmetric=False, dense=False):
    """Area quadrature of the source surface, with no evaluation ROI or RNG."""
    centers, areas = _quadrature(original, dense)
    hit = _closest_surface(mesh, centers)
    ids = hit['primitive_ids'].numpy()
    faces = mesh.faces[ids]
    uv = hit['primitive_uvs'].numpy()
    bary = np.column_stack((1 - uv.sum(axis=1), uv))
    normals = mesh.face_normals[ids]
    residual = ((centers - hit['points'].numpy()) * normals).sum(axis=1)
    basis = (mesh.vertex_normals[faces] * normals[:, None]).sum(axis=2)
    weight = areas[:, None] * bary
    n = len(mesh.vertices)
    rhs = np.bincount(faces.ravel(), (weight * basis * residual[:, None]).ravel(), minlength=n)
    mass = np.bincount(faces.ravel(), (weight * basis**2).ravel(), minlength=n)
    if symmetric:
        # Iteratively reweight the two area integrals toward absolute distance.
        floor = max(np.median(mesh.edges_unique_length) * 0.01, 1e-12)
        weight /= np.sqrt(residual[:, None]**2 + floor**2) * original.area
        rhs = np.bincount(faces.ravel(), (weight * basis * residual[:, None]).ravel(), minlength=n)
        mass = np.bincount(faces.ravel(), (weight * basis**2).ravel(), minlength=n)
        reverse_points, reverse_areas = _quadrature(mesh, dense)
        reverse = _closest_surface(original, reverse_points)
        normals = original.face_normals[reverse['primitive_ids'].numpy()]
        residual = ((reverse['points'].numpy() - reverse_points) * normals).sum(axis=1)
        repeats = 3 if dense else 1
        reverse_faces = np.repeat(mesh.faces, repeats, axis=0)
        if dense:
            reverse_bary = np.tile(_FACE_QUADRATURE, (len(mesh.faces), 1))
        else:
            reverse_bary = np.full((len(mesh.faces), 3), 1 / 3)
        basis = (mesh.vertex_normals[reverse_faces] * normals[:, None]).sum(axis=2)
        weight = reverse_areas[:, None] * reverse_bary / (mesh.area * np.sqrt(residual[:, None]**2 + floor**2))
        rhs += np.bincount(reverse_faces.ravel(), (weight * basis * residual[:, None]).ravel(), minlength=n)
        mass += np.bincount(reverse_faces.ravel(), (weight * basis**2).ravel(), minlength=n)
    step = np.divide(rhs, mass, out=np.zeros(n), where=mass > 0)
    shortest = np.full(n, np.inf)
    for corner in mesh.edges_unique.T:
        np.minimum.at(shortest, corner, mesh.edges_unique_length)
    return mesh.vertex_normals * np.clip(step, -0.1 * shortest, 0.1 * shortest)[:, None]


def _covered_moves(source, proposed, back, owner, dist, roi, hd_limit, roi_limit):
    """Keep the existing nearest vertex as a witness for each source distance bound."""
    after = np.linalg.norm(source - proposed[owner], axis=1)
    allowed = back <= hd_limit
    allowed[owner[after > hd_limit]] = False
    allowed[owner[roi & (after > roi_limit)]] = False
    roi_delta = np.bincount(owner[roi], (after - dist)[roi], minlength=len(proposed))
    return allowed & (roi_delta <= 0)


def refine_surface(original, simplified, guide_vertices, saliency, passes=12):
    """Prioritize large fitting residuals at salient vertices within the legacy bounds."""
    mesh = simplified.copy()
    source = original.vertices
    tree = cKDTree(source)
    target_curv = _deficit(original)
    roi = abs(target_curv) > np.percentile(abs(target_curv), 90)
    if not roi.any():
        return mesh
    faces = mesh.faces
    n = len(mesh.vertices)
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    fixed = np.unique(mesh.edges_unique[counts != 2])
    rings = mesh.vertex_neighbors
    saliency = np.asarray(saliency).reshape(-1)
    if len(saliency) != len(guide_vertices) or not np.isfinite(saliency).all():
        raise ValueError("Expected one finite saliency weight per guide vertex")
    guide_index = cKDTree(guide_vertices).query(mesh.vertices)[1]
    priority = 1.0 + np.clip(saliency[guide_index], 0, 1)
    for _ in range(passes):
        verts = mesh.vertices
        delta = _normal_step(original, mesh)
        proposed = verts + delta
        dist, owner = cKDTree(verts).query(source)
        back, nearest = tree.query(verts)
        new_back, new_nearest = tree.query(proposed)
        hd = max(dist.max(), back.max())
        allowed = _covered_moves(source, proposed, new_back, owner, dist, roi, hd, dist[roi].max())
        allowed[fixed] = False

        # Each row describes changing just one vertex. Sum the angle change
        # across incident faces before evaluating each affected angle deficit.
        tri = verts[faces]
        old_angles = _angles(tri)
        rows, cols, values = [], [], []
        for j in range(3):
            moved = tri.copy()
            moved[:, j] = proposed[faces[:, j]]
            cross = np.cross(moved[:, 1] - moved[:, 0], moved[:, 2] - moved[:, 0])
            allowed[faces[(cross * mesh.face_normals).sum(axis=1) <= 0, j]] = False
            rows.append(np.repeat(faces[:, j], 3))
            cols.append(faces.ravel())
            values.append((old_angles - _angles(moved)).ravel())
        changes = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n)).tocsr().tocoo()
        error = _deficit(mesh) - target_curv[nearest]
        updated = error[changes.col] + changes.data
        diag = changes.row == changes.col
        moved_ids = changes.row[diag]
        updated[diag] += target_curv[nearest[moved_ids]] - target_curv[new_nearest[moved_ids]]
        cost = np.bincount(changes.row, abs(updated) - abs(error[changes.col]), minlength=n)
        allowed &= (cost <= 0) & (np.linalg.norm(delta, axis=1) > 0)
        # Disjoint rings make the curvature changes additive.
        used = np.zeros(n, dtype=bool)
        selected = []
        for i in np.argsort(-priority * np.linalg.norm(delta, axis=1), kind='stable'):
            if allowed[i] and not used[i] and not used[rings[i]].any():
                used[i] = True
                used[rings[i]] = True
                selected.append(i)
        if not selected:
            break
        mesh.vertices[selected] = proposed[selected]
    # The disjoint updates can leave curvature headroom. Take a coupled normal
    # step if its actual surface residual decreases within the original bounds.
    initial_dist = cKDTree(simplified.vertices).query(source)[0]
    initial_back, initial_nearest = tree.query(simplified.vertices)
    hd_limit = max(initial_dist.max(), initial_back.max())
    roi_limit = initial_dist[roi].max()
    curvature_limit = np.abs(_deficit(simplified) - target_curv[initial_nearest]).mean()
    # First reproduce the centroid fit, then refine its surface integral with
    # three interior quadrature points per face. Evaluation samples stay independent.
    for dense in (False, True):
        centers, areas = _quadrature(original, dense)

        def residual(candidate):
            closest = _closest_surface(candidate, centers)['points'].numpy()
            forward = np.average(np.linalg.norm(centers - closest, axis=1), weights=areas)
            points, weights = _quadrature(candidate, dense)
            closest = _closest_surface(original, points)['points'].numpy()
            backward = np.average(np.linalg.norm(points - closest, axis=1), weights=weights)
            return forward, backward

        before = residual(mesh)
        accepted_steps = 0
        for _ in range(24 if dense else 8):
            delta = _normal_step(original, mesh, symmetric=True, dense=dense)
            delta[fixed] = 0
            dist, owner = cKDTree(mesh.vertices).query(source)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125):
                proposed = mesh.vertices + scale * delta
                back, nearest = tree.query(proposed)
                reject = ~_covered_moves(source, proposed, back, owner, dist, roi, hd_limit, roi_limit)
                proposed[reject] = mesh.vertices[reject]
                candidate = mesh.copy()
                candidate.vertices = proposed
                if np.any((candidate.face_normals * mesh.face_normals).sum(axis=1) <= 0):
                    continue
                nearest[reject] = tree.query(proposed[reject])[1]
                curvature = np.abs(_deficit(candidate) - target_curv[nearest]).mean()
                if curvature > curvature_limit:
                    continue
                after = residual(candidate)
                if after[0] < before[0] and after[1] < before[1]:
                    mesh, before = candidate, after
                    accepted = True
                    accepted_steps += 1
                    break
            if not accepted:
                break
        print(f"Surface quadrature ({3 if dense else 1} points/face): "
              f"{accepted_steps} accepted steps, directional means={before}")
    return mesh
