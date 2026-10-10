"""
FAFBlock —— Frequency-Aware Fusion Block.

用途：作为 FusedHE 中 FeatureFusion 的替代品，在 FusedDecoder 内按 stage 融合
ConvNeXt / ResNet 侧的高频分量 与 MiT / GroupMamba / MambaVision 侧的低频分量，
缓解粗暴 cat 带来的"低频主导、建筑细节被抹平"问题。

forward 接口与 FeatureFusion 一致：
    forward(x_mit, x_cnn) -> Tensor
    - x_mit: 低频分支特征（MiT / GroupMamba / MambaVision），通道数 = mit_in_channel
             （默认等于 out_channel，与 FusedHE 原逻辑一致）
    - x_cnn: 高频分支特征（ConvNeXt / ResNet），通道数 = in_channel
    - 输出通道数 = 2 * out_channel，与原 FeatureFusion 保持一致，下游 decoder 无需改动

流程（对应 FAFBlock.txt 第七节）:
  1. 通道对齐           Conv1x1 + BN + ReLU 两支都映射到 d = out_channel
  2. 高频提取           use_smooth=1: C_high = C_align - AvgPool(C_align)
                        use_smooth=0: 跳过平滑差分，C_high = C_align
  3. 低频提取           G_low  = G_align (第一版保持恒等，避免额外平滑过头)
  4. 拼接               cat(C_align, G_align)
  5. 空间门控           Sigmoid(Conv3x3(ReLU(Conv1x1(cat))))  -> (B,1,H,W)
  6. 通道门控           Sigmoid(MLP(GAP(cat)))               -> (B,d,1,1)
  7. 联合权重           W_c = (1-M_spat) * (1-M_chan),  W_g = M_spat * M_chan
  8. 分层差异化融合     α*(W_c * C_high) + β*(W_g * G_low) + λ*(C_align + G_align)
  9. 细化               Conv3x3 + BN + ReLU 投到 2*d 通道
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 每个 stage 的默认 (alpha, beta):
#   stage_idx = 1 → 最浅（分辨率最高），更信 ConvNeXt 的高频细节
#   stage_idx = 4 → 最深（语义层），更信 Mamba / MiT 的低频全局
# _DEFAULT_STAGE_COEF = {
#     1: (0.7, 0.3),
#     2: (0.6, 0.4),
#     3: (0.5, 0.5),
#     4: (0.4, 0.6),
# }

#DFC19
# _DEFAULT_STAGE_COEF = {
#     1: (0.579304, 0.420696),
#     2: (0.354298, 0.645702),
#     3: (0.445415, 0.554585),
#     4: (0.268313, 0.731687),
# }
#dfc23
# _DEFAULT_STAGE_COEF = {
#     1: (0.678834, 0.321166),
#     2: (0.509497, 0.490503),
#     3: (0.394169, 0.605831),
#     4: (0.327750, 0.672250),
# }
#vhg
_DEFAULT_STAGE_COEF = {
    1: (0.7, 0.3),
    2: (0.6, 0.4),
    3: (0.4, 0.6),
    4: (0.3, 0.7),
}


class _ChannelGate(nn.Module):
    """对拼接特征做 GAP → MLP → Sigmoid，得到通道维门控 (B, d, 1, 1)。"""

    def __init__(self, cat_dim: int, out_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(cat_dim // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Linear(cat_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, cat_feat: torch.Tensor) -> torch.Tensor:
        z = cat_feat.mean(dim=[2, 3])
        w = torch.sigmoid(self.mlp(z)).unsqueeze(-1).unsqueeze(-1)
        return w


class FAFBlock(nn.Module):
    """Frequency-Aware Fusion Block（drop-in 替换 FeatureFusion）。"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stage_idx: int,
        mit_in_channel: Optional[int] = None,
        smooth_kernel: int = 3,
        use_smooth: bool = True,
        lambda_res: float = 0.5,
        learnable_coef: bool = True,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        channel_reduction: int = 4,
    ):
        super().__init__()

        if mit_in_channel is None:
            mit_in_channel = out_channel

        self.use_smooth = bool(use_smooth)

        if smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel 必须是奇数，收到: {}".format(smooth_kernel))

        d = out_channel
        self.d = d
        self.stage_idx = stage_idx
        self.smooth_kernel = smooth_kernel
        self.smooth_pad = smooth_kernel // 2

        default_alpha, default_beta = _DEFAULT_STAGE_COEF.get(stage_idx, (0.5, 0.5))
        alpha = float(default_alpha if alpha is None else alpha)
        beta = float(default_beta if beta is None else beta)
        lambda_res = float(lambda_res)

        if learnable_coef:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            self.lambda_res = nn.Parameter(torch.tensor(lambda_res))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))
            self.register_buffer("lambda_res", torch.tensor(lambda_res))

        # Step 1: 通道对齐
        self.align_c = nn.Sequential(
            nn.Conv2d(in_channel, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.align_g = nn.Sequential(
            nn.Conv2d(mit_in_channel, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )

        # Step 5: 空间门控（基于拼接特征，单通道图）
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2 * d, d, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # Step 6: 通道门控
        self.channel_gate = _ChannelGate(2 * d, d, reduction=channel_reduction)

        # Step 9: 融合后细化并投回 2*d 通道，保持与 FeatureFusion 的输出契约
        self.refine = nn.Sequential(
            nn.Conv2d(d, 2 * d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(inplace=True),
        )

        smooth_hint = (
            "AvgPool diff -> c_high=c_align-c_smooth"
            if self.use_smooth
            else "c_high=c_align, no AvgPool diff"
        )
        print(
            "[FAFBlock] stage{}: use_smooth={} smooth_kernel={} ({})".format(
                stage_idx, int(self.use_smooth), self.smooth_kernel, smooth_hint
            ),
            flush=True,
        )

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        # stride=1 的平均池化，保持 H/W 不变；作为简单低通，用来抽取高频残差分量
        return F.avg_pool2d(
            x,
            kernel_size=self.smooth_kernel,
            stride=1,
            padding=self.smooth_pad,
        )

    def forward(
        self,
        x_mit: torch.Tensor,
        x_cnn: torch.Tensor,
        capture: Optional[dict] = None,
    ) -> torch.Tensor:
        # Step 1: 通道对齐
        c_align = self.align_c(x_cnn)
        g_align = self.align_g(x_mit)

        if capture is not None:
            capture["x_mit"] = x_mit.detach().cpu()
            capture["x_cnn"] = x_cnn.detach().cpu()

        # Step 2: ConvNeXt 侧高频分量（可选 smooth 差分提取）
        if self.use_smooth:
            c_smooth = self._smooth(c_align)
            c_high = c_align - c_smooth
        else:
            c_high = c_align

        # Step 3: Mamba / MiT 侧低频（恒等，第一版保持稳妥）
        g_low = g_align

        if capture is not None:
            capture["use_smooth"] = self.use_smooth
            if self.use_smooth:
                capture["c_smooth"] = c_smooth.detach().cpu()
            capture["c_high"] = c_high.detach().cpu()

        # Step 4: 拼接，供两个门控共用
        cat_feat = torch.cat([c_align, g_align], dim=1)

        if capture is not None:
            capture["cat_feat"] = cat_feat.detach().cpu()

        # Step 5: 空间门控（GM 更偏区域一致性 → M 大；ConvNeXt 更偏边缘 → 1-M 大）
        m_spatial = self.spatial_gate(cat_feat)           # (B,1,H,W)
        w_spatial_g = m_spatial
        w_spatial_c = 1.0 - m_spatial

        # Step 6: 通道门控
        m_channel = self.channel_gate(cat_feat)           # (B,d,1,1)
        w_channel_g = m_channel
        w_channel_c = 1.0 - m_channel

        # Step 7: 空间-通道联合权重（广播相乘）
        w_c = w_spatial_c * w_channel_c
        w_g = w_spatial_g * w_channel_g

        # Step 8: 分层差异化融合 + 残差项
        fused = (
            self.alpha * (w_c * c_high)
            + self.beta * (w_g * g_low)
            + self.lambda_res * (c_align + g_align)
        )

        # Step 9: 细化并投回 2*d 通道
        out = self.refine(fused)
        if capture is not None:
            capture["faf_out"] = out.detach().cpu()
        return out

    def extra_repr(self) -> str:
        # 仅用于打印时更直观；buffer 已用 tensor 存储，这里直接读 .item()
        a = float(self.alpha.detach().cpu().item())
        b = float(self.beta.detach().cpu().item())
        l = float(self.lambda_res.detach().cpu().item())
        return (
            "stage_idx={}, d={}, alpha={:.3f}, beta={:.3f}, lambda_res={:.3f}, "
            "smooth_kernel={}, use_smooth={}"
        ).format(self.stage_idx, self.d, a, b, l, self.smooth_kernel, self.use_smooth)
