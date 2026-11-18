import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

def knn(x, k):

    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):

    batch_size = x.size(0)
    num_points = x.size(1)

    x = x.reshape(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x.transpose(2, 1), k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()


    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)


    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DeformableEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(DeformableEdgeConv, self).__init__()
        self.k = k
        self.out_channels = out_channels


        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)


        self.p_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)


        self.m_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.m_conv.weight, 0)

    def forward(self, x):

        x_graph = get_graph_feature(x, k=self.k)

        offset = self.p_conv(x_graph)
        mask = self.sigmoid(self.m_conv(x_graph))

        diff_feat = x_graph[:, :x_graph.shape[1]//2, :, :]
        center_feat = x_graph[:, x_graph.shape[1]//2:, :, :]

        diff_feat_deformed = diff_feat + offset
        x_graph_deformed = torch.cat([diff_feat_deformed, center_feat], dim=1)

        out = self.conv(x_graph_deformed)
        out = out * mask
        out = out.max(dim=-1, keepdim=False)[0]

        return self.leaky_relu(self.bn(out))


class GAM_Attention_Point(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention_Point, self).__init__()


        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )


        self.spatial_attention = nn.Sequential(


            nn.Conv1d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm1d(int(in_channels / rate)),
            nn.ReLU(inplace=True),

            nn.Conv1d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(in_channels)
        )

    def forward(self, x):

        b, c, n = x.shape


        x_permute = x.permute(0, 2, 1)


        x_att_permute = self.channel_attention(x_permute)


        x_channel_att = x_att_permute.permute(0, 2, 1).sigmoid()


        x = x * x_channel_att


        x_spatial_att = self.spatial_attention(x).sigmoid()

        out = x * x_spatial_att

        return out


class DGCNN_SaliencyEncoder(nn.Module):
    def __init__(self, k=20, output_dim=1):
        super(DGCNN_SaliencyEncoder, self).__init__()
        self.k = k
        input_channels = Config.INPUT_DIM * 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(512)


        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv3 = DeformableEdgeConv(in_channels=64, out_channels=128, k=k)
        self.conv4 = DeformableEdgeConv(in_channels=128, out_channels=256, k=k)


        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.attention = GAM_Attention_Point(in_channels=1024, rate=4)


        self.linear1 = nn.Linear(1024, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(128, output_dim)


        self.blend_alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, curvature_raw, ablate=None):

        x_g = get_graph_feature(x, k=self.k)
        x = self.conv1(x_g)
        x1 = x.max(dim=-1, keepdim=False)[0]


        x_g = get_graph_feature(x1.permute(0, 2, 1), k=self.k)
        x = self.conv2(x_g)
        x2 = x.max(dim=-1, keepdim=False)[0]


        x3 = self.conv3(x2.permute(0, 2, 1))
        x4 = self.conv4(x3.permute(0, 2, 1))


        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x_cat)
        global_feat = x.max(dim=2)[0]


        num_points = x.size(2)
        global_feat_expand = global_feat.unsqueeze(2).repeat(1, 1, num_points)
        x_seg = torch.cat((x1, x2, x3, x4, global_feat_expand), dim=1)


        x_seg = self.attention(x_seg)


        x_seg = x_seg.permute(0, 2, 1).contiguous()
        x_seg = F.leaky_relu(self.bn6(self.linear1(x_seg).transpose(2, 1))).transpose(2, 1)
        x_seg = self.dp1(x_seg)
        x_seg = F.leaky_relu(self.bn7(self.linear2(x_seg).transpose(2, 1))).transpose(2, 1)
        x_seg = self.dp2(x_seg)

        logits = self.linear3(x_seg)
        saliency = torch.sigmoid(logits)


        blend_weight = torch.sigmoid(self.blend_alpha)
        saliency_final = blend_weight * saliency + (1 - blend_weight) * curvature_raw

        return saliency_final, global_feat


@torch.no_grad()
def edge_neighbors(features, k, chunk_size=512):
    """Find nearest other vertices in each patch's current feature space."""
    batch_size, num_points, _ = features.shape
    k = min(k, num_points - 1)
    neighbors = torch.empty(batch_size, num_points, k, dtype=torch.long, device=features.device)
    squares = features.square().sum(dim=-1)
    transposed = features.transpose(1, 2)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        query = features[:, start:end]
        scores = torch.bmm(query, transposed)
        scores.mul_(2).sub_(squares[:, start:end, None]).sub_(squares[:, None])
        rows = torch.arange(end - start, device=features.device)
        scores[:, rows, start + rows] = -torch.inf
        neighbors[:, start:end] = scores.topk(k, dim=-1, sorted=False).indices
    return neighbors


class PaperEdgeConv(nn.Module):
    """Dynamic graph convolution with center and relative features (Eqs. 5–6)."""

    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, features):
        indices = edge_neighbors(features, self.k)
        patches = torch.arange(features.shape[0], device=features.device)[:, None, None]
        adjacent = features[patches, indices]
        centers = features.unsqueeze(2)
        edges = torch.cat((centers.expand_as(adjacent), adjacent - centers), dim=-1)
        return self.mlp(edges.permute(0, 3, 1, 2)).max(dim=-1).values.transpose(1, 2)


class VertexAttention(nn.Module):
    """A shared MLP applies one learned scalar gate per vertex (Eqs. 7–8)."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, features):
        return features * self.gate(features)


class PaperSaliencyEncoder(nn.Module):
    def __init__(self, use_curvature=True):
        super().__init__()
        self.use_curvature = use_curvature
        self.edge_convs = nn.ModuleList([
            PaperEdgeConv(8 if use_curvature else 7, 64),
            PaperEdgeConv(64, 64),
            PaperEdgeConv(64, 128),
            PaperEdgeConv(128, 256),
        ])
        self.attention = VertexAttention()
        self.saliency_head = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.blend_alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self, features, curvature, ablate=None):
        if not self.use_curvature:
            # The ablation removes the input channel while retaining Eq. (10)'s prior.
            features = torch.cat((features[..., :6], features[..., 7:]), dim=-1)
        layers = []
        for edge_conv in self.edge_convs:
            features = edge_conv(features)
            layers.append(features)
        attended = self.attention(torch.cat(layers, dim=-1))
        saliency = self.saliency_head(attended)
        blend = self.blend_alpha.sigmoid()
        return blend * saliency + (1 - blend) * curvature, attended.max(dim=1).values


class FoldingNetDecoder(nn.Module):
    def __init__(self, input_dim=512, paper=False):
        super(FoldingNetDecoder, self).__init__()
        self.paper = paper
        layers = [nn.Linear(input_dim + 2, 512), nn.ReLU()]
        if paper:
            layers.append(nn.Dropout(0.5))
        layers.extend((nn.Linear(512, 512), nn.ReLU()))
        if paper:
            layers.append(nn.Dropout(0.5))
        self.mlp = nn.Sequential(*layers, nn.Linear(512, 3))

    def forward(self, feature, num_points=2048):
        batch_size = feature.shape[0]
        if self.paper:
            width = math.ceil(math.sqrt(num_points))
            height = math.ceil(num_points / width)
            horizontal = torch.linspace(-1, 1, width, device=feature.device, dtype=feature.dtype)
            vertical = torch.linspace(-1, 1, height, device=feature.device, dtype=feature.dtype)
            grid = torch.cartesian_prod(vertical, horizontal)[:num_points].expand(batch_size, -1, -1)
        else:
            grid = torch.rand(batch_size, num_points, 2).to(feature.device)
        feature_expand = feature.unsqueeze(1).expand(-1, num_points, -1)
        cat_input = torch.cat([feature_expand, grid], dim=-1)
        return self.mlp(cat_input)

class DentalSaliencyNet(nn.Module):
    def __init__(self, paper=False, use_curvature=True):
        super(DentalSaliencyNet, self).__init__()
        self.encoder = PaperSaliencyEncoder(use_curvature) if paper else DGCNN_SaliencyEncoder()
        self.decoder = FoldingNetDecoder(input_dim=512, paper=paper)

    def forward(self, x, curvature, x_rot=None, ablate=None):
        weights, global_feat = self.encoder(x, curvature, ablate=ablate)
        recon_points = self.decoder(global_feat, num_points=x.size(1))

        weights_rot = None
        if x_rot is not None:
            weights_rot, _ = self.encoder(x_rot, curvature, ablate=ablate)

        return weights, recon_points, weights_rot
