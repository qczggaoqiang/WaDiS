import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class FrequencyAwareGatingBlock(nn.Module):
    """
    频域感知门控块 (FAGB)

    严格按照论文Algorithm 1实现：
    1. LayerNorm
    2. 1x1 Conv (channel expansion)
    3. Split (分组处理)
    4. FFT到频域
    5. 可学习复数滤波器
    6. IFFT回空间域
    7. Polymerize (特征融合)
    8. DWC (深度可分离卷积)
    9. GELU门控机制
    10. 1x1 Conv (channel reduction)
    11. 残差连接
    """

    def __init__(self, in_channels, expansion_factor=2, num_groups=4):
        """
        Args:
            in_channels: 输入特征的通道数
            expansion_factor: 通道扩展因子
            num_groups: Split分组数量
        """
        super(FrequencyAwareGatingBlock, self).__init__()

        self.in_channels = in_channels
        self.expanded_channels = in_channels * expansion_factor
        self.num_groups = num_groups
        self.group_channels = self.expanded_channels // num_groups

        # Step 1: Layer Normalization
        self.layer_norm = LayerNorm([in_channels], elementwise_affine=True)

        # Step 2: 1x1 convolution for channel expansion
        self.conv1x1_expand = nn.Conv2d(in_channels, self.expanded_channels,
                                        kernel_size=1, stride=1, padding=0, bias=False)

        # Step 5: 可学习的频域滤波器
        # 使用可学习参数来调制频域响应
        self.freq_filter_real = nn.Parameter(torch.ones(1, self.expanded_channels, 1, 1))
        self.freq_filter_imag = nn.Parameter(torch.zeros(1, self.expanded_channels, 1, 1))

        # Step 7: Polymerize - 特征融合层
        self.polymerize_conv = nn.Conv2d(self.expanded_channels, self.expanded_channels,
                                         kernel_size=1, stride=1, padding=0, bias=False)

        # Step 8: 深度可分离卷积 (DWC)
        self.dwconv = nn.Conv2d(self.expanded_channels, self.expanded_channels,
                                kernel_size=3, stride=1, padding=1,
                                groups=self.expanded_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(self.expanded_channels, self.expanded_channels,
                                        kernel_size=1, stride=1, padding=0, bias=False)

        # Step 9: GELU激活函数
        self.gelu = nn.GELU()

        # Step 10: 最终的1x1卷积，降维回原始通道数
        self.conv1x1_final = nn.Conv2d(self.expanded_channels, in_channels,
                                       kernel_size=1, stride=1, padding=0, bias=False)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        # 频域滤波器初始化为接近恒等映射
        nn.init.ones_(self.freq_filter_real)
        nn.init.zeros_(self.freq_filter_imag)

        # 其他层使用默认初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, f):
        """
        Args:
            f: 输入特征 [B, C, H, W]

        Returns:
            f_out: 输出特征 [B, C, H, W]
        """
        B, C, H, W = f.shape
        identity = f  # 保存用于残差连接

        # Step 1: LayerNorm(f)
        f1 = f.permute(0, 2, 3, 1)  # [B, H, W, C]
        f1 = self.layer_norm(f1)
        f1 = f1.permute(0, 3, 1, 2)  # [B, C, H, W]

        # Step 2: Conv1x1(f1) - channel expansion
        f2 = self.conv1x1_expand(f1)  # [B, expanded_C, H, W]

        # Step 3: Split(f2) - 分组处理
        f3 = f2.view(B, self.num_groups, self.group_channels, H, W)

        # Step 4: F(f3) - FFT变换到频域
        f3_reshaped = f3.view(B * self.num_groups, self.group_channels, H, W)
        f_hat = torch.fft.fft2(f3_reshaped, dim=(-2, -1))  # 复数FFT

        # Step 5: Filter(f_hat) - 频域滤波
        # 应用可学习的复数滤波器
        filter_complex = torch.complex(
            self.freq_filter_real.expand_as(f3_reshaped),
            self.freq_filter_imag.expand_as(f3_reshaped)
        )
        f_hat_filtered = f_hat * filter_complex

        # Step 6: F^(-1)(f_hat_filtered) - IFFT回空间域
        f4 = torch.fft.ifft2(f_hat_filtered, dim=(-2, -1)).real

        # 重新整形回原始维度
        f4 = f4.view(B, self.expanded_channels, H, W)

        # Step 7: Polymerize(f4) - 特征融合
        f5 = self.polymerize_conv(f4)

        # Step 8: DWC(f5) - 深度可分离卷积
        f6 = self.dwconv(f5)
        f6 = self.pointwise_conv(f6)

        # Step 9: Gelu(f6) ⊙ f6 - 门控机制
        f6_gated = self.gelu(f6)
        f7 = f6_gated * f6  # 元素级门控

        # Step 10: Conv1x1(f7) - 降维
        f8 = self.conv1x1_final(f7)

        # Step 11: f + f8 - 残差连接
        f_out = identity + f8

        return f_out


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    batch_size = 2
    channels = 64
    height, width = 32, 32

    # 创建模型
    fagb_model = FrequencyAwareGatingBlock(in_channels=channels)

    # 创建输入
    x = torch.randn(batch_size, channels, height, width)

    # 前向传播
    with torch.no_grad():
        output = fagb_model(x)

    print(f"输入形状: {x.shape}")
    print(f"FAGB输出形状: {output.shape}")

    # 验证残差连接是否正确
    print(f"输出与输入形状一致: {output.shape == x.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in fagb_model.parameters() if p.requires_grad)
    print(f"FAGB参数量: {total_params:,}")

    # 测试梯度流
    x.requires_grad_(True)
    output = fagb_model(x)
    loss = output.sum()
    loss.backward()
    print(f"梯度计算正常: {x.grad is not None}")