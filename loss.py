import torch
import torch.nn.functional as F

def knn_points(k, points):

    inner = -2 * torch.matmul(points, points.transpose(2, 1))
    xx = torch.sum(points**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1]
    return idx[:, :, 1:]

def virtual_decimation_loss(points, weights, curvature, k=8, curv_beta=0.0, height=None):

    B, N, C = points.shape


    idx = knn_points(k, points)


    idx_flat = idx.reshape(B, N * k).unsqueeze(-1).expand(-1, -1, C)
    neighbors = torch.gather(points, 1, idx_flat).reshape(B, N, k, C)


    collapsed_pos = torch.mean(neighbors, dim=2)


    raw_error = torch.sum((points - collapsed_pos) ** 2, dim=-1) * 1000.0


    curv_val = curvature.squeeze(-1)
    importance = 1.0 + curv_beta * curv_val
    if height is not None:
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


def chamfer_distance_charbonnier(p1, p2, eps=1e-3):

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
