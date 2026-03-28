import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    1D 残差块：包含两个卷积层和一条跳跃连接 (Shortcut Connection)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，bias=False 因为后续使用了 BatchNorm
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU() # 使用前沿的 GELU 激活函数
        
        # 第二个卷积层，stride 固定为 1
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 跳跃连接路径：
        # 如果步长不为 1（导致时间维长度变短）或输入输出通道不一致，需要用 1x1 卷积对齐残差路径的形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # x shape: (B, C_in, L_in)
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 加上跳跃连接（如果形状不同，shortcut 会将其调整为与 out 相同形状）
        out += self.shortcut(identity)
        out = self.gelu(out)
        
        return out # out shape: (B, C_out, L_out)

class ConformerAMC(nn.Module):
    """
    ConformerAMC: 结合 ResNet 局部特征提取与 Transformer 全局建模的调制分类模型。
    这是一种现代化的混合架构 Baseline，旨在利用 ResNet 的高吞吐和 Transformer 在长序列上的注意力。
    
    输入形状: (Batch, 2, 1024)
    输出形状: (Batch, 24)
    """
    def __init__(self, num_classes=24, d_model=128, nhead=8, num_layers=4):
        super(ConformerAMC, self).__init__()
        
        # ==========================================
        # 1. 前端: CNN ResNet Backbone (局部特征提取 & 时间降采样)
        # 输入: (B, 2, 1024)
        # 通过卷积的 stride 进行降采样，大幅减少后续 Transformer 的计算量
        # ==========================================
        
        # 初始特征映射：(2, 1024) -> (32, 1024)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        
        # 利用你定义的 ResidualBlock 堆叠，并在每层进行 2 倍降采样
        # 此时 d_model = 128 (来自参数输入)
        self.resnet_layers = nn.Sequential(
            ResidualBlock(32, 64, stride=2),   # shape -> (B, 64, 512)
            ResidualBlock(64, d_model, stride=2), # shape -> (B, d_model=128, 256)
            ResidualBlock(d_model, d_model, stride=2), # shape -> (B, 128, 128)
            ResidualBlock(d_model, d_model, stride=2)  # shape -> (B, 128, 64)
        )
        
        # CNN 处理后的序列长度是固定的 (针对 1024 输入来说是 64)
        processed_length = 64
        
        # ==========================================
        # 2. 桥接与位置编码 (Bridge & Positional Encoding)
        # Transformer 本身无序，需要加上相对位置信息
        # ==========================================
        
        # 定义一个可学习的位置编码参数，形状为 (1, Sequence_Length, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, processed_length, d_model) * 0.02)        
        # ==========================================
        # 3. 后端: Transformer Encoder (全局依赖关系建模)
        # 输入: (B, 64, d_model=128)
        # ==========================================
        
        # 定义 Transformer Encoder 层 ( batch_first=True 简化了张量排列，需要 PyTorch >= 1.9)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, # 前馈网络扩 4 倍，标准配置
            dropout=0.1, 
            activation="gelu",
            batch_first=True # 极重要参数！直接接受 (Batch, Seq, Feature) 输入
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ==========================================
        # 4. 分类头 (Classifier Head)
        # 将 Transformer 输出的时间序列聚合为分类预测
        # ==========================================
        
        self.classifier_head = nn.Sequential(
            # 使用层归一化防止过拟合

            # 全球平均池化 (聚合时间维度，把 64 个点的特征压缩为 1 个点的特征) -> shape (B, d_model)
            nn.LayerNorm(d_model), # 层归一化提升训练稳定性
            nn.Linear(d_model, d_model),# 全连接层提升特征表达能力
            nn.GELU(), # 激活函数
            nn.Dropout(0.3), # Dropout 放在 GELU 之后 Linear 之前
            nn.Linear(d_model, num_classes) #  Logits -> shape (B, 24)
        )

    def forward(self, x):
        # 1. 输入数据 x shape: [B, 2, 1024]

        # --- CNN Backbone 阶段 (局部特征与降采样) ---
        x = self.initial_conv(x)      # shape: [B, 32, 1024]
        x = self.resnet_layers(x)     # shape: [B, 128, 64] (时间维从 1024 压到 64)

        # --- 桥接阶段 (维度置换与位置编码) ---
        # 转换形状以满足 Transformer Batch First 要求: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)        # shape: [B, 64, 128]
        
        # 注入可学习的位置信息 (自动广播机制)
        x = x + self.pos_embedding   # shape: [B, 64, 128]

        # --- Transformer Encoder 阶段 (全局建模) ---
        x = self.transformer_encoder(x) # shape: [B, 64, 128]

        # --- 分类聚合阶段 ---
        # 为了配合 AdaptiveAvgPool1d 需要将维度换回: (B, L, C) -> (B, C, L)
        x = x.mean(dim=1)        # shape: [B, 128, 64]
        
        logits = self.classifier_head(x) # shape: [B, 24]
        
        return logits
