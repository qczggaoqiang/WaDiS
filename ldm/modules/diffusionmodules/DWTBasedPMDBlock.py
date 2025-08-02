# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:04:26 2024

@author: admin
"""

import torch
import torch.nn as nn
import pywt
import numpy as np

# PMD Diffusion Function
def perona_malik_diffusion(u_LH, u_HL, k=1.0):
    """
    Perona-Malik Diffusion step.
    u_LH: horizontal detail coefficients
    u_HL: vertical detail coefficients
    k: diffusion control parameter (default 1.0)
    """
    # Compute gradient magnitude
    grad_mag = torch.sqrt(u_LH ** 2 + u_HL ** 2)

    # Compute diffusion coefficient
    g = 1 / (1 + (grad_mag / k) ** 2)

    # Apply diffusion to high-frequency components
    diffused_LH = g * u_LH
    diffused_HL = g * u_HL

    return diffused_LH, diffused_HL


# DWT-based PMD Block
class DWTBasedPMDBlock(nn.Module):
    def __init__(self, in_channels, wavelet='haar'):
        """
        DWT-based PMD Block
        wavelet: type of wavelet (default is 'haar')
        """
        super(DWTBasedPMDBlock, self).__init__()
        self.dwt_wavelet = wavelet

        # 使用 nn.ModuleList 替代 nn.Sequential 以便动态修改
        self.dwt_res_block = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        ])

    def dwt(self, x):
        """
        Discrete Wavelet Transform (DWT) for 2D feature maps.
        x: input tensor of shape (batch_size, channels, height, width)
        returns: list of DWT coefficients for each channel in each batch
        """
        batch_size, channels, height, width = x.shape
        dwt_coeffs = []
        for i in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                # Apply 2D DWT per channel
                coeffs = pywt.dwt2(x[i, c].detach().cpu().numpy(), self.dwt_wavelet)
                batch_coeffs.append(coeffs)
            dwt_coeffs.append(batch_coeffs)
        return dwt_coeffs

    def idwt(self, coeffs):
        """
        Inverse Discrete Wavelet Transform (IDWT).
        coeffs: list of DWT coefficients for each channel in each batch
        returns: reconstructed feature map as a tensor
        """
        batch_size = len(coeffs)
        channels = len(coeffs[0])
        recons_img = []
        for i in range(batch_size):
            img_recons = []
            for c in range(channels):
                u_LL, (u_LH, u_HL, u_HH) = coeffs[i][c]
                # Reconstruct the image for each channel using IDWT
                img_recons.append(pywt.idwt2((u_LL, (u_LH, u_HL, u_HH)), self.dwt_wavelet))
            recons_img.append(torch.tensor(np.stack(img_recons)))
        return torch.stack(recons_img)

    def forward(self, x):
        # 获取输入通道数
        #print(x.shape)
        in_channels = x.size(1)  # 输入的通道数

        # 动态生成适应通道数的卷积层
        # 在这里更新卷积层的输入输出通道数
        # self.res_block[0] = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.res_block[1] = nn.BatchNorm2d(in_channels)
        # self.res_block[3] = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.res_block[4] = nn.BatchNorm2d(in_channels)

        # 获取输入设备并将模型迁移到相同设备
        device = x.device
        self.to(device)  # 将模型迁移到输入设备

        # Perform DWT to get coefficients
        dwt_coeffs = self.dwt(x)

        # Extract LH and HL components for Perona-Malik Diffusion
        dwt_LH = []
        dwt_HL = []
        new_coeffs = []
        for i in range(len(dwt_coeffs)):
            batch_coeffs = []
            for j in range(len(dwt_coeffs[i])):
                u_LL, (u_LH, u_HL, u_HH) = dwt_coeffs[i][j]
                dwt_LH.append(torch.tensor(u_LH, dtype=torch.float32).to(device))  # 确保设备一致
                dwt_HL.append(torch.tensor(u_HL, dtype=torch.float32).to(device))  # 确保设备一致
                batch_coeffs.append((u_LL, (u_LH, u_HL, u_HH)))
            new_coeffs.append(batch_coeffs)

        # Convert LH and HL lists back to tensors
        dwt_LH = torch.stack(dwt_LH)
        dwt_HL = torch.stack(dwt_HL)

        # Apply Perona-Malik diffusion
        diffused_LH, diffused_HL = perona_malik_diffusion(dwt_LH, dwt_HL)

        # Replace original LH and HL components with diffused versions
        for i in range(len(dwt_coeffs)):
            for j in range(len(dwt_coeffs[i])):
                u_LL, (u_LH, u_HL, u_HH) = new_coeffs[i][j]
                new_coeffs[i][j] = (u_LL, (diffused_LH[j].cpu().numpy(), diffused_HL[j].cpu().numpy(), u_HH))

        # Perform inverse DWT to reconstruct the feature map
        diffused_feature_map = self.idwt(new_coeffs).to(device)  # 确保设备一致

        # Apply residual block
        out = self.dwt_res_block[0](diffused_feature_map)  # Apply first Conv2d
        out = self.dwt_res_block[1](out)  # Apply BatchNorm
        out = self.dwt_res_block[2](out)  # Apply ReLU
        out = self.dwt_res_block[3](out)  # Apply second Conv2d
        out = self.dwt_res_block[4](out)  # Apply second BatchNorm
        out = out + diffused_feature_map  # Add residual

        return out

# 测试代码
if __name__ == "__main__":
    # 生成随机的 64 通道特征图
    input_tensor = torch.randn((1, 64, 256, 256))
    #input_tensor = torch.randn((1, 64, 256, 256)).cuda()

    # 实例化 DWT-based PMD Block
    dwt_pmd_block = DWTBasedPMDBlock()
    #dwt_pmd_block = DWTBasedPMDBlock().cuda()

    # 执行前向传播
    output_tensor = dwt_pmd_block(input_tensor)

    print("Output shape:", output_tensor.shape)
