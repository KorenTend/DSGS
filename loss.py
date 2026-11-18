import torch
import torch.nn.functional as F

PAIRWISE_CHUNK_SIZE = 512


def knn_points(k, points):

    inner = -2 * torch.matmul(points, points.transpose(2, 1))
    xx = torch.sum(points**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1]
    return idx[:, :, 1:]


def _collapse_neighbors(points, k):
    """Exact KNN with the query itself excluded, including at duplicate points."""
    point_count = points.shape[1]
    neighbors = []
    with torch.no_grad():
        for start in range(0, point_count, PAIRWISE_CHUNK_SIZE):
            query = points[:, start:start + PAIRWISE_CHUNK_SIZE]
            distances = torch.cdist(query, points, compute_mode='donot_use_mm_for_euclid_dist')
            rows = torch.arange(query.shape[1], device=points.device)
            distances[:, rows, start + rows] = torch.inf
            neighbors.append(distances.topk(k, largest=False).indices)
    return torch.cat(neighbors, dim=1)


def virtual_decimation_loss(points, weights, curvature, k=8, curv_beta=0.0, height=None, *, paper=False):

    B, N, C = points.shape


    idx = _collapse_neighbors(points, k) if paper else knn_points(k, points)


    idx_flat = idx.reshape(B, N * k).unsqueeze(-1).expand(-1, -1, C)
    neighbors = torch.gather(points, 1, idx_flat).reshape(B, N, k, C)


    collapsed_pos = torch.mean(neighbors, dim=2)


    raw_error = torch.sum((points - collapsed_pos) ** 2, dim=-1)
    if not paper:
        raw_error = raw_error * 1000.0


    curv_val = curvature.squeeze(-1)
    importance = 1.0 + curv_beta * curv_val
    if height is not None and not paper:
        height_val = height.squeeze(-1)
        importance = importance + 1.0 * height_val


    weighted_error = raw_error * importance


    w = weights.squeeze(-1)
    loss = (1.0 - w) * weighted_error

    return torch.mean(loss)

def view_consistency_loss(weights, weights_rot):
    return torch.mean((weights - weights_rot)**2)

def sparsity_loss(weights):
    return torch.mean(torch.abs(weights))

def edge_preserving_loss(weights, curvature, k=8):

    B, N = weights.shape[:2]
    idx = knn_points(k, curvature.view(B, N, 1))

    w_expand = weights.view(B, N, 1)
    idx_flat = idx.reshape(B, N * k).unsqueeze(-1).expand(-1, -1, 1)
    w_neighbors = torch.gather(w_expand, 1, idx_flat).reshape(B, N, k)

    curv_expand = curvature.view(B, N, 1)
    curv_neighbors = torch.gather(curv_expand, 1, idx_flat).reshape(B, N, k)

    curv_diff = torch.abs(curv_expand - curv_neighbors)
    edge_weight = torch.exp(-5.0 * curv_diff)

    w_diff = (w_expand - w_neighbors) ** 2
    loss = torch.mean(edge_weight * w_diff)
    return loss


def _chamfer_matches(p1, p2):
    """Find both nearest-neighbor maps without retaining a dense autograd graph."""
    forward_indices = []
    reverse_distances = p2.new_full(p2.shape[:-1], torch.inf)
    reverse_indices = torch.zeros(p2.shape[:-1], dtype=torch.long, device=p2.device)
    with torch.no_grad():
        for start in range(0, p1.shape[1], PAIRWISE_CHUNK_SIZE):
            query = p1[:, start:start + PAIRWISE_CHUNK_SIZE]
            distances = torch.cdist(query, p2, compute_mode='donot_use_mm_for_euclid_dist')
            forward_indices.append(distances.argmin(dim=2))
            chunk_distances, chunk_indices = distances.min(dim=1)
            reverse_indices = torch.where(chunk_distances < reverse_distances,
                                          chunk_indices + start, reverse_indices)
            reverse_distances = torch.minimum(reverse_distances, chunk_distances)
    return torch.cat(forward_indices, dim=1), reverse_indices


def chamfer_distance_charbonnier(p1, p2, eps=1e-3, *, paper=False):
    if paper:
        forward_indices, reverse_indices = _chamfer_matches(p1, p2)
        forward_matches = p2.gather(1, forward_indices.unsqueeze(-1).expand_as(p1))
        reverse_matches = p1.gather(1, reverse_indices.unsqueeze(-1).expand_as(p2))
        forward = ((p1 - forward_matches).square().sum(dim=-1) + eps**2).sqrt()
        reverse = ((p2 - reverse_matches).square().sum(dim=-1) + eps**2).sqrt()
        return forward.mean() + reverse.mean()

    diff = p1.unsqueeze(2) - p2.unsqueeze(1)
    dist = torch.sum(diff**2, dim=-1)
    min_dist_p1, _ = torch.min(dist, dim=2)
    min_dist_p2, _ = torch.min(dist, dim=1)
    dist_p1 = torch.sqrt(min_dist_p1 + eps**2)
    dist_p2 = torch.sqrt(min_dist_p2 + eps**2)
    return torch.mean(dist_p1) + torch.mean(dist_p2)


def chamfer_distance(p1, p2):
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)
    dist = torch.sum(diff**2, dim=-1)
    min_dist_p1, _ = torch.min(dist, dim=2)
    min_dist_p2, _ = torch.min(dist, dim=1)
    return torch.mean(min_dist_p1) + torch.mean(min_dist_p2)
