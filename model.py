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

class FoldingNetDecoder(nn.Module):
    def __init__(self, input_dim=512):
        super(FoldingNetDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, feature, num_points=2048):
        B = feature.shape[0]
        grid = torch.rand(B, num_points, 2).to(feature.device)
        feature_expand = feature.unsqueeze(1).repeat(1, num_points, 1)
        cat_input = torch.cat([feature_expand, grid], dim=-1)
        return self.mlp(cat_input)

class DentalSaliencyNet(nn.Module):
    def __init__(self):
        super(DentalSaliencyNet, self).__init__()
        self.encoder = DGCNN_SaliencyEncoder()
        self.decoder = FoldingNetDecoder(input_dim=512)

    def forward(self, x, curvature, x_rot=None, ablate=None):
        weights, global_feat = self.encoder(x, curvature, ablate=ablate)
        recon_points = self.decoder(global_feat, num_points=x.size(1))

        weights_rot = None
        if x_rot is not None:
            weights_rot, _ = self.encoder(x_rot, curvature, ablate=ablate)

        return weights, recon_points, weights_rot
