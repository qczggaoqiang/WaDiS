import torch
import torch.nn as nn
import warnings
import math
import torch.nn.functional as F


# ========== Haar DWT/IWT 基础逻辑 ==========
def dwt_init(x):
    """Haar小波变换的前向实现"""
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


def iwt_init(x):
    """Haar小波逆变换的实现"""
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# ========== HREB核心组件 ==========
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class CrossAttention(nn.Module):
    """交叉注意力机制"""

    def __init__(self, dim, num_heads=8, dropout=0.):
        super(CrossAttention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除")

        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = DepthwiseSeparableConv(dim, dim)
        self.key = DepthwiseSeparableConv(dim, dim)
        self.value = DepthwiseSeparableConv(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """重塑张量以适应多头注意力"""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_input, key_value_input):
        # 计算Q, K, V
        mixed_query_layer = self.query(query_input)
        mixed_key_layer = self.key(key_value_input)
        mixed_value_layer = self.value(key_value_input)

        # 展平空间维度以进行注意力计算
        B, C, H, W = mixed_query_layer.shape
        mixed_query_layer = mixed_query_layer.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        mixed_key_layer = mixed_key_layer.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        mixed_value_layer = mixed_value_layer.view(B, C, -1).transpose(1, 2)  # (B, HW, C)

        # 多头注意力变换
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (B, num_heads, HW, head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 应用注意力权重
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 重塑回原始形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.transpose(1, 2).view(B, C, H, W)

        return context_layer


class ProgressiveDilatedResidualBlock(nn.Module):
    """渐进式扩张残差块"""

    def __init__(self, channels):
        super(ProgressiveDilatedResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x

        out = self.leaky_relu(self.conv1(x))
        out = self.leaky_relu(self.conv2(out))
        out = self.leaky_relu(self.conv3(out))
        out = self.leaky_relu(self.conv4(out))
        out = self.conv5(out)

        return out + residual


class HREB(nn.Module):
    """高频残差增强块 (High-Frequency Residual Enhancement Block)"""

    def __init__(self, in_channels, out_channels):
        super(HREB, self).__init__()

        # 初始深度可分离卷积
        self.initial_conv_LH = DepthwiseSeparableConv(in_channels, out_channels)
        self.initial_conv_HL = DepthwiseSeparableConv(in_channels, out_channels)
        self.initial_conv_HH = DepthwiseSeparableConv(in_channels, out_channels)

        # 交叉注意力模块
        self.cross_attn_LH_to_HH = CrossAttention(out_channels, num_heads=8)
        self.cross_attn_HL_to_HH = CrossAttention(out_channels, num_heads=8)

        # 融合HH的卷积
        self.fusion_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)

        # 渐进式扩张残差块
        self.dilated_block_LH = ProgressiveDilatedResidualBlock(out_channels)
        self.dilated_block_HL = ProgressiveDilatedResidualBlock(out_channels)
        self.dilated_block_HH = ProgressiveDilatedResidualBlock(out_channels)

        # 最终深度可分离卷积
        self.final_conv_LH = DepthwiseSeparableConv(out_channels, in_channels)
        self.final_conv_HL = DepthwiseSeparableConv(out_channels, in_channels)
        self.final_conv_HH = DepthwiseSeparableConv(out_channels, in_channels)

    def forward(self, I_LH, I_HL, I_HH):
        # 初始深度可分离卷积
        feat_LH = self.initial_conv_LH(I_LH)
        feat_HL = self.initial_conv_HL(I_HL)
        feat_HH = self.initial_conv_HH(I_HH)

        # 交叉注意力：LH和HL的信息聚合到HH
        attn_LH_to_HH = self.cross_attn_LH_to_HH(feat_HH, feat_LH)
        attn_HL_to_HH = self.cross_attn_HL_to_HH(feat_HH, feat_HL)

        # 融合增强的HH特征
        enhanced_HH = self.fusion_conv(torch.cat([attn_LH_to_HH, attn_HL_to_HH], dim=1))

        # 渐进式扩张残差处理
        refined_LH = self.dilated_block_LH(feat_LH)
        refined_HL = self.dilated_block_HL(feat_HL)
        refined_HH = self.dilated_block_HH(enhanced_HH)

        # 最终深度可分离卷积压缩通道
        hat_I_LH = self.final_conv_LH(refined_LH)
        hat_I_HL = self.final_conv_HL(refined_HL)
        hat_I_HH = self.final_conv_HH(refined_HH)

        return hat_I_LH, hat_I_HL, hat_I_HH


# ========== 多尺度小波编码器 ==========
class MultiScaleWaveletEncoder(nn.Module):
    """多尺度小波编码器，完全按照论文实现"""

    def __init__(self, in_channels=3, hreb_out_channels=64):
        super(MultiScaleWaveletEncoder, self).__init__()

        self.dwt = DWT()
        self.iwt = IWT()

        # HREB模块用于两个分解层级
        self.hreb_level1 = HREB(in_channels, hreb_out_channels)  # 第一层
        self.hreb_level2 = HREB(in_channels, hreb_out_channels)  # 第二层

    def forward(self, I):
        """
        按照论文公式实现：
        输入: I ∈ R^{H×W×3}
        输出: φ(I) = x_c (条件编码)
        """
        n, c, h, w = I.shape

        # ===== 第一层分解 (l=1) =====
        # 公式(1): {I_LL^l, I_LH^l, I_HL^l, I_HH^l} = DWT(I)
        dwt_level1 = self.dwt(I)  # (4n, c, h/2, w/2)
        I_LL_1 = dwt_level1[:n]  # 低频近似
        I_HL_1 = dwt_level1[n:2 * n]  # 水平高频
        I_LH_1 = dwt_level1[2 * n:3 * n]  # 垂直高频
        I_HH_1 = dwt_level1[3 * n:]  # 对角高频

        # 公式(2): {hat_I_HL^l, hat_I_HH^l, hat_I_LH^l} = HREB(I_HL^l, I_HH^l, I_LH^l)
        hat_I_LH_1, hat_I_HL_1, hat_I_HH_1 = self.hreb_level1(I_LH_1, I_HL_1, I_HH_1)

        # ===== 第二层分解 (l+1=2) =====
        # 公式(3): {I_LL^{l+1}, I_LH^{l+1}, I_HL^{l+1}, I_HH^{l+1}} = DWT(I_LL^l)
        dwt_level2 = self.dwt(I_LL_1)  # (4n, c, h/4, w/4)
        I_LL_2 = dwt_level2[:n]  # 低频近似
        I_HL_2 = dwt_level2[n:2 * n]  # 水平高频
        I_LH_2 = dwt_level2[2 * n:3 * n]  # 垂直高频
        I_HH_2 = dwt_level2[3 * n:]  # 对角高频

        # 公式(4): {hat_I_HL^{l+1}, hat_I_HH^{l+1}, hat_I_LH^{l+1}} = HREB(I_HL^{l+1}, I_HH^{l+1}, I_LH^{l+1})
        hat_I_LH_2, hat_I_HL_2, hat_I_HH_2 = self.hreb_level2(I_LH_2, I_HL_2, I_HH_2)

        # ===== 重构和融合 =====
        # 公式(5): I' = IDWT(I_LL^l, hat_I_LH^l, hat_I_HL^l, hat_I_HH^l)
        recon_input_1 = torch.cat([I_LL_1, hat_I_HL_1, hat_I_LH_1, hat_I_HH_1], dim=0)
        I_prime = self.iwt(recon_input_1)  # (n, c, h, w)

        # 公式(6): I'' = IDWT(I_LL^{l+1}, hat_I_LH^{l+1}, hat_I_HL^{l+1}, hat_I_HH^{l+1})
        recon_input_2 = torch.cat([I_LL_2, hat_I_HL_2, hat_I_LH_2, hat_I_HH_2], dim=0)
        I_double_prime = self.iwt(recon_input_2)  # (n, c, h/2, w/2)

        # 公式(7): I'~ = Down_8x(I'), I''~ = Down_4x(I'')
        # 下采样到相同尺寸
        target_h, target_w = h // 8, w // 8
        I_tilde_prime = F.interpolate(I_prime, size=(target_h, target_w), mode='bilinear', align_corners=False)
        I_tilde_double_prime = F.interpolate(I_double_prime, size=(target_h, target_w), mode='bilinear',
                                             align_corners=False)

        # 公式(8): I''' = Concat(I'~, I''~)
        I_triple_prime = torch.cat([I_tilde_prime, I_tilde_double_prime], dim=1)  # (n, 2c, h/8, w/8)

        # 公式(9): φ(I) = I''' = x_c
        x_c = I_triple_prime

        return x_c


# ========== 测试代码 ==========
if __name__ == "__main__":
    # 创建模型
    encoder = MultiScaleWaveletEncoder(in_channels=3, hreb_out_channels=64)

    # 测试输入
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256)  # 输入图像

    print("输入形状:", input_tensor.shape)

    # 前向传播
    with torch.no_grad():
        conditional_embedding = encoder(input_tensor)

    print("条件编码 φ(I) 形状:", conditional_embedding.shape)
    print("预期形状: ({}, {}, {}, {})".format(batch_size, 6, 32, 32))

    # 验证模型参数
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")