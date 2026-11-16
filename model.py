# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

def knn(x, k):
    """
    纯 PyTorch 实现 KNN
    x: (B, N, C)
    返回: (B, N, k) 索引
    """
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    DGCNN 核心: 构造静态图特征 (EdgeConv Input)
    返回: (B, 2*C, N, k) -> [x_j - x_i, x_i]
    """
    batch_size = x.size(0)
    num_points = x.size(1)
    
    x = x.reshape(batch_size, -1, num_points)      # (B, C, N)
    
    if idx is None:
        idx = knn(x.transpose(2, 1), k=k)       # (B, N, k)
        
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()   # (B, N, C)
    
    # 收集邻居特征 x_j
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # 中心点 x_i
    
    # EdgeConv 特征: [x_j - x_i, x_i]
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (B, 2*C, N, k)

# =========================================================================
# Deformable EdgeConv (已修复 BatchNorm 维度问题)
# =========================================================================
class DeformableEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=20):
        super(DeformableEdgeConv, self).__init__()
        self.k = k
        self.out_channels = out_channels
        
        # 1. 基础卷积
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels) # [Fix] 这里的输出是 (B, C, N)，必须用 1d
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        # 2. 偏移学习层
        self.p_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        # 3. 调制学习层
        self.m_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1) 
        self.sigmoid = nn.Sigmoid()
        
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.m_conv.weight, 0)

    def forward(self, x):
        # x: (B, N, C)
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

# =========================================================================
# [新增] GAM Attention (适配 Point Cloud 版本)
# =========================================================================
class GAM_Attention_Point(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention_Point, self).__init__()

        # 1. 通道注意力子模块
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        # 2. 空间注意力子模块 (针对点云修改为 Conv1d)
        self.spatial_attention = nn.Sequential(
            # 使用7x7卷积核进行空间特征的降维处理
            # [修改] Conv2d -> Conv1d
            nn.Conv1d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm1d(int(in_channels / rate)), # [修改] BN2d -> BN1d
            nn.ReLU(inplace=True),
            # 使用7x7卷积核进行空间特征的升维处理
            nn.Conv1d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(in_channels)
        )

    def forward(self, x):
        """
        x: (B, C, N) 点云特征
        """
        b, c, n = x.shape 

        # --- Channel Attention ---
        # 调整形状适配 Linear: (B, C, N) -> (B, N, C)
        x_permute = x.permute(0, 2, 1)
        
        # 应用通道注意力
        x_att_permute = self.channel_attention(x_permute)
        
        # 恢复形状并生成权重: (B, N, C) -> (B, C, N)
        x_channel_att = x_att_permute.permute(0, 2, 1).sigmoid()
        
        # 特征加权
        x = x * x_channel_att

        # --- Spatial Attention ---
        # 直接在 (B, C, N) 上应用 Conv1d
        x_spatial_att = self.spatial_attention(x).sigmoid()
        
        out = x * x_spatial_att

        return out

# =========================================================================
# DGCNN Saliency Encoder
# =========================================================================
class DGCNN_SaliencyEncoder(nn.Module):
    def __init__(self, k=20, output_dim=1):
        super(DGCNN_SaliencyEncoder, self).__init__()
        self.k = k
        input_channels = Config.INPUT_DIM * 2

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(512)

        # Layers
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        # Deformable EdgeConvs
        self.conv3 = DeformableEdgeConv(in_channels=64, out_channels=128, k=k)
        self.conv4 = DeformableEdgeConv(in_channels=128, out_channels=256, k=k)

        # Aggregation
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # [改进] 多尺度特征融合: 64(x1) + 64(x2) + 128(x3) + 256(x4) + 512(global) = 1024
        self.attention = GAM_Attention_Point(in_channels=1024, rate=4)

        # Prediction Head (1024-dim input)
        self.linear1 = nn.Linear(1024, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(128, output_dim)

        # [改进] 可学习曲率融合权重
        self.blend_alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self, x, curvature_raw, ablate=None):
        # Layer 1
        x_g = get_graph_feature(x, k=self.k)
        x = self.conv1(x_g)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # Layer 2
        x_g = get_graph_feature(x1.permute(0, 2, 1), k=self.k)
        x = self.conv2(x_g)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # Layer 3 & 4 (Deformable)
        x3 = self.conv3(x2.permute(0, 2, 1)) # (B, 128, N)
        x4 = self.conv4(x3.permute(0, 2, 1)) # (B, 256, N)

        # Global Feature
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x_cat)
        global_feat = x.max(dim=2)[0]

        # [改进] 多尺度特征融合
        num_points = x.size(2)
        global_feat_expand = global_feat.unsqueeze(2).repeat(1, 1, num_points)
        x_seg = torch.cat((x1, x2, x3, x4, global_feat_expand), dim=1) # (B, 1024, N)

        # GAM Attention
        x_seg = self.attention(x_seg)

        # MLP Prediction
        x_seg = x_seg.permute(0, 2, 1).contiguous()
        x_seg = F.leaky_relu(self.bn6(self.linear1(x_seg).transpose(2, 1))).transpose(2, 1)
        x_seg = self.dp1(x_seg)
        x_seg = F.leaky_relu(self.bn7(self.linear2(x_seg).transpose(2, 1))).transpose(2, 1)
        x_seg = self.dp2(x_seg)

        logits = self.linear3(x_seg)
        saliency = torch.sigmoid(logits)

        # 可学习曲率融合
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