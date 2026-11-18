import numpy as np
import trimesh
from sklearn.neighbors import KDTree

def compute_curvature(mesh, *, paper=False):
    if paper:
        # Use the curvature magnitude as described in §3.2 and §4.2.
        curvature = np.abs(mesh.vertex_defects)
        lower_bound, upper_bound = np.percentile(curvature, [50, 90])
        if upper_bound == lower_bound:
            return np.zeros_like(curvature)
        normalized = (curvature - lower_bound) / (upper_bound - lower_bound)
        return np.sqrt(np.clip(normalized, 0.0, 1.0))

    vertices = mesh.vertices
    edges = mesh.edges_unique


    n0 = mesh.vertex_normals[edges[:, 0]]
    n1 = mesh.vertex_normals[edges[:, 1]]
    dot = np.sum(n0 * n1, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.arccos(dot)


    curvature = np.zeros(len(vertices))
    np.add.at(curvature, edges[:, 0], angles)
    np.add.at(curvature, edges[:, 1], angles)


    lower_bound = np.percentile(curvature, 50)
    upper_bound = np.percentile(curvature, 90)

    curvature = np.clip(curvature, lower_bound, upper_bound)
    curvature = (curvature - lower_bound) / (upper_bound - lower_bound + 1e-8)


    curvature = curvature ** 0.5

    return curvature

def fps_sampling(points, k):
    n = len(points)
    indices = np.zeros(k, dtype=int)
    indices[0] = np.random.randint(0, n)
    distances = np.sum((points - points[indices[0]])**2, axis=1)
    for i in range(1, k):
        farthest = np.argmax(distances)
        indices[i] = farthest
        new_dists = np.sum((points - points[farthest])**2, axis=1)
        distances = np.minimum(distances, new_dists)
    return indices

def get_patches(mesh, global_curvature, config, *, paper=False):
    vertices = mesh.vertices
    normals = mesh.vertex_normals


    z_coords = vertices[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    z_range = z_max - z_min
    if paper:
        global_height = np.divide(z_coords - z_min, z_range, out=np.zeros_like(z_coords), where=z_range > 0)
    else:
        global_height = (z_coords - z_min) / (z_range + 1e-8)


    if paper:
        center_count = min(config.NUM_PATCHES, len(vertices))
        centers = vertices[fps_sampling(vertices, center_count)]
        tree = KDTree(vertices)
        neighbor_count = min(config.PATCH_POINTS, len(vertices))
        indices_list = tree.query(centers, k=neighbor_count, return_distance=False)
    else:
        num_guide = min(config.NUM_POINTS_GUIDE, len(vertices))
        guide_idx = np.random.choice(len(vertices), num_guide, replace=False)
        guide_points = vertices[guide_idx]

        center_indices_local = fps_sampling(guide_points, config.NUM_PATCHES)
        centers = guide_points[center_indices_local]

        bbox_diag = np.linalg.norm(vertices.max(0) - vertices.min(0))
        radius = (bbox_diag / np.sqrt(config.NUM_PATCHES)) * config.OVERLAP_RATIO
        tree = KDTree(vertices)
        indices_list = tree.query_radius(centers, r=radius)

    patches = []

    for idx in indices_list:
        if paper:
            sample_idx = idx if len(idx) == config.PATCH_POINTS else np.resize(idx, config.PATCH_POINTS)
        elif len(idx) < 100:
            continue
        elif len(idx) >= config.PATCH_POINTS:
            sample_idx = np.random.choice(idx, config.PATCH_POINTS, replace=False)
        else:
            sample_idx = np.random.choice(idx, config.PATCH_POINTS, replace=True)

        pts = vertices[sample_idx]
        nms = normals[sample_idx]
        curv = global_curvature[sample_idx]
        height = global_height[sample_idx]


        local_center = np.mean(pts, axis=0)
        pts_centered = pts - local_center
        scale = np.max(np.linalg.norm(pts_centered, axis=1)) + 1e-6
        pts_norm = pts_centered / scale

        patches.append({
            'points_norm': pts_norm,
            'normals': nms,
            'curvature': curv,
            'height': height,
            'center': local_center,
            'scale': scale,
            'original_indices': sample_idx
        })

    return patches

def stitch_weights(mesh, patches, patch_weights_list, *, paper=False):
    full_weights_sum = np.zeros(len(mesh.vertices))
    full_weights_count = np.full(len(mesh.vertices), 0.0 if paper else 1e-6)

    for i, patch in enumerate(patches):
        orig_idx = patch['original_indices']
        weights = patch_weights_list[i].flatten()

        pts_norm = patch['points_norm']
        dists = np.linalg.norm(pts_norm, axis=1)
        confidences = np.clip(1.0 - dists, 0.0, 1.0)

        np.add.at(full_weights_sum, orig_idx, weights * confidences)
        np.add.at(full_weights_count, orig_idx, confidences)

    final_weights = np.divide(full_weights_sum, full_weights_count, out=np.zeros_like(full_weights_sum),
                              where=full_weights_count > 0)
    return np.clip(final_weights, 0, 1)
