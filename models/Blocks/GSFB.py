"""
GSFB —— Gated Skip Fusion Block.

用途：作为 FusedDecoder 中 DecoderBlock 的 drop-in 替代，用来处理跳跃链接部分：
      从编码器旁路过来的 skip 特征 与 上一层 decoder 输出上采样后的特征 之间的融合。

接口与 DecoderBlock 一致：
    forward(ffi_1, ffi_2) -> Tensor
    - ffi_1: 上一层 decoder 输出（较深语义、较低分辨率）
             通道数 = in_channel；空间尺寸若与 ffi_2 不同，内部自动 bilinear 上采样
    - ffi_2: 对应 stage 的 skip 特征（浅层高分辨率）
             通道数 = out_channel
    - 输出通道数 = out_channel；空间尺寸 = 2 × ffi_2.spatial
      （与原 DecoderBlock 末尾的 upsample 保持一致，下游 FusedDecoder 无需改动）

流程（对应 GSF-B.txt）:
  Step 1: D_up   = Up(ffi_1)                        （内部 F.interpolate 到 ffi_2 尺寸）
  Step 2: D_proj = Conv1x1 + BN + ReLU (D_up)
          S_proj = Conv1x1 + BN + ReLU (ffi_2)
  Step 3: Z      = Concat(D_proj, S_proj)
          M      = Sigmoid(Conv3x3(BN(ReLU(Conv1x1(Z)))))   (B, 1, H, W)
  Step 4: S_sel  = M * S_proj
  Step 5: F_fuse = Concat(D_proj, S_sel)                   (B, 2C, H, W)
          F      = ReLU(BN(Conv3x3(F_fuse)))                (B, C,  H, W)
  Step 6: F      = ReLU(BN(Conv3x3(F_fuse)))                (B, C,  H, W)
          F      = F + γ * residual                          （可学习残差，稳定训练）
          residual: 默认 γ·(D_proj+S_proj)；use_FF=true 时改为
                    γ·DecoderBlock(ffi_1, ffi_2)（3×3→concat→3×3，不含 upsample）
  Step 7: out    = Upsample×2(F)                           （为下一 stage 准备）

设计说明：
- 严格按 GSF-B.txt 思路实现空间门控（1 通道 M）；
- 在 gate 中间层加 BN 缓解 sigmoid 饱和；
- 第 6 步轻量残差 γ*(D+S)：γ 为可学习 scalar，初值 0.1，避免早期被门控压死梯度；
- 第 7 步 ×2 upsample 保持与原 DecoderBlock 输出契约完全一致（drop-in）。
"""
# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class GSFB(nn.Module):
#     """Gated Skip Fusion Block（drop-in 替换 DecoderBlock）。"""

#     def __init__(
#         self,
#         in_channel: int,
#         out_channel: int,
#         gate_reduction: int = 2,
#         residual_gamma_init: float = 0.1,
#         learnable_residual: bool = True,
#     ):
#         """
#         Args:
#             in_channel:  ffi_1（上一层 decoder 输出）通道数
#             out_channel: ffi_2（当前 skip）通道数，亦为 GSFB 输出通道数
#             gate_reduction: 门控中间层通道压缩比（>=1）
#             residual_gamma_init: 第 6 步残差门的初值
#             learnable_residual: 残差门 γ 是否可学习；False 时 register_buffer，不更新
#         """
#         super().__init__()

#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         c = out_channel

#         # Step 2: 通道对齐 —— 两支都映射到 C_i = out_channel
#         self.align_d = nn.Sequential(
#             nn.Conv2d(in_channel, c, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True),
#         )
#         self.align_s = nn.Sequential(
#             nn.Conv2d(out_channel, c, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True),
#         )

#         # Step 3: 空间门控 —— Conv1x1 → BN+ReLU → Conv3x3 → Sigmoid
#         #   输入 Z = cat(D_proj, S_proj) (B, 2C, H, W)
#         #   输出 M (B, 1, H, W)，每个位置表示"skip 该注入多少"
#         hidden = max(c // gate_reduction, 16)
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(2 * c, hidden, kernel_size=1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
#             nn.Sigmoid(),
#         )

#         # Step 5: 融合并通道对齐回 C_i
#         self.refine = nn.Sequential(
#             nn.Conv2d(2 * c, c, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c),
#             nn.ReLU(inplace=True),
#         )

#         # Step 6: 可学习残差门 γ —— 早期保梯度、后期可被训练压低
#         gamma = torch.tensor(float(residual_gamma_init))
#         if learnable_residual:
#             self.gamma = nn.Parameter(gamma)
#         else:
#             self.register_buffer("gamma", gamma)

#         # Step 7: 送下一 stage 的 ×2 upsample，与原 DecoderBlock 末尾一致
#         self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

#     def forward(self, ffi_1: torch.Tensor, ffi_2: torch.Tensor) -> torch.Tensor:
#         # Step 1: 空间对齐到 skip 尺寸；当前 FusedDecoder 会提前把 ffi_1 对齐好，
#         # 这里仍做一次形状检查兜底，便于未来调用方简化。
#         if ffi_1.shape[-2:] != ffi_2.shape[-2:]:
#             ffi_1 = F.interpolate(
#                 ffi_1, size=ffi_2.shape[-2:], mode="bilinear", align_corners=False
#             )

#         # Step 2: 通道对齐
#         d_proj = self.align_d(ffi_1)
#         s_proj = self.align_s(ffi_2)

#         # Step 3: 空间门控
#         z = torch.cat([d_proj, s_proj], dim=1)
#         m_spatial = self.spatial_gate(z)  # (B, 1, H, W)

#         # Step 4: 按门控筛选 skip
#         s_sel = m_spatial * s_proj

#         # Step 5: concat + 3×3 refine 投回 C_i
#         fused = self.refine(torch.cat([d_proj, s_sel], dim=1))

#         # Step 6: 轻量可学习残差，防止门控初期把信号掐死
#         fused = fused + self.gamma * (d_proj + s_proj)

#         # Step 7: ×2 upsample，匹配原 DecoderBlock 输出尺寸
#         return self.upsample(fused)

#     def extra_repr(self) -> str:
#         g = float(self.gamma.detach().cpu().item())
#         return "in_channel={}, out_channel={}, gamma={:.3f}".format(
#             self.in_channel, self.out_channel, g
#         )




from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from models.Blocks._config import load_block_cfg
except Exception:
    def load_block_cfg(name: str):
        return {}


def _parse_truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


class _ChannelGate(nn.Module):
    """
    轻量 SE-style 通道门控。
    对单个输入分支做通道重标定：
        x -> GAP -> MLP -> sigmoid -> x * w
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))          # B, C
        w = self.mlp(w).view(b, c, 1, 1)
        return x * w


class _OrthogonalDirectionGate(nn.Module):
    """
    BOSA-style 双向正交空间门控的轻量实现。

    输入:
        x: B, C, H, W

    思路:
        1. 沿 width 聚合，得到 height-aware context: B, C, H, 1
        2. 沿 height 聚合，得到 width-aware context:  B, C, 1, W
        3. 分别投影成 q_h / k_w
        4. 通过 q_h 和 k_w 的逐通道相关生成 B, 1, H, W 的方向空间注意力
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.hidden = hidden

        self.proj_h = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.proj_w = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # height direction context: B, C, H, 1
        avg_h = x.mean(dim=3, keepdim=True)
        max_h = x.max(dim=3, keepdim=True)[0]
        ctx_h = avg_h + max_h

        # width direction context: B, C, 1, W
        avg_w = x.mean(dim=2, keepdim=True)
        max_w = x.max(dim=2, keepdim=True)[0]
        ctx_w = avg_w + max_w

        q_h = self.proj_h(ctx_h)        # B, hidden, H, 1
        k_w = self.proj_w(ctx_w)        # B, hidden, 1, W

        # B, hidden, H, W -> B, 1, H, W
        attn = (q_h * k_w).sum(dim=1, keepdim=True) / math.sqrt(float(self.hidden))
        attn = F.relu(attn, inplace=True)

        return self.out(attn)


class GSFB(nn.Module):
    """Gated Skip Fusion Block（drop-in 替换 DecoderBlock）。"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        gate_reduction: int = 2,
        residual_gamma_init: float = 0.1,
        learnable_residual: bool = True,
        channel_gate: Optional[bool] = None,
        direction_gate: Optional[bool] = None,
        channel_reduction: int = 4,
        use_FF: Optional[bool] = None,
    ):
        """
        Args:
            in_channel:  ffi_1（上一层 decoder 输出）通道数
            out_channel: ffi_2（当前 skip）通道数，亦为 GSFB 输出通道数
            gate_reduction: 空间门控中间层通道压缩比
            residual_gamma_init: 残差门 gamma 初值
            learnable_residual: gamma 是否可学习
            channel_gate: 是否开启分支通道注意力；None 时读 block.yaml，缺省为 False
            direction_gate: 是否开启双向正交空间注意力；None 时读 block.yaml，缺省为 False
            channel_reduction: 通道注意力压缩比
            use_FF: 残差旁路是否走原版 DecoderBlock（3×3→concat→3×3）；
                    None 时读 block.yaml，缺省为 False（γ·(D_proj+S_proj)）
        """
        super().__init__()

        cfg = load_block_cfg("GSFB")
        if channel_gate is None:
            channel_gate = bool(cfg.get("channel_gate", False))
        if direction_gate is None:
            direction_gate = bool(cfg.get("direction_gate", False))
        if use_FF is None:
            use_FF = _parse_truthy(cfg.get("use_FF", False))

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_channel_gate = bool(channel_gate)
        self.use_direction_gate = bool(direction_gate)
        self.use_FF = bool(use_FF)

        c = out_channel

        # Step 2: 通道对齐 —— 两支都映射到 C_i = out_channel
        self.align_d = nn.Sequential(
            nn.Conv2d(in_channel, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.align_s = nn.Sequential(
            nn.Conv2d(out_channel, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # 可选：BOSA-style 的分支通道重标定
        if self.use_channel_gate:
            self.channel_gate_d = _ChannelGate(c, reduction=channel_reduction)
            self.channel_gate_s = _ChannelGate(c, reduction=channel_reduction)

        # 原始空间门控
        hidden = max(c // gate_reduction, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2 * c, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # 可选：双向正交空间门控
        if self.use_direction_gate:
            self.direction_gate = _OrthogonalDirectionGate(c, reduction=gate_reduction)

            # 方向门控与原始 spatial gate 的融合系数
            # sigmoid 后约为 0.5，初期不偏向任何一个 gate
            self.direction_mix = nn.Parameter(torch.tensor(0.0))

        # Step 5: 融合并通道对齐回 C_i
        self.refine = nn.Sequential(
            nn.Conv2d(2 * c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Step 6: 可学习残差门 gamma
        gamma = torch.tensor(float(residual_gamma_init))
        if learnable_residual:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

        # 可选：原版 DecoderBlock 式残差旁路（不含末端 ×2 upsample，与门控主路共享）
        if self.use_FF:
            self.ff_bypass_conv1 = nn.Sequential(
                nn.Conv2d(in_channel, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
            self.ff_bypass_conv2 = nn.Sequential(
                nn.Conv2d(2 * c, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )

        # Step 7: 与原 DecoderBlock 输出契约一致
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, ffi_1: torch.Tensor, ffi_2: torch.Tensor) -> torch.Tensor:
        # Step 1: 空间对齐到 skip 尺寸
        if ffi_1.shape[-2:] != ffi_2.shape[-2:]:
            ffi_1 = F.interpolate(
                ffi_1, size=ffi_2.shape[-2:], mode="bilinear", align_corners=False
            )

        # Step 2: 通道对齐
        d_proj = self.align_d(ffi_1)
        s_proj = self.align_s(ffi_2)

        # Step 2.5: 可选通道重标定
        # 这里是对 decoder 分支和 skip 分支分别做通道筛选，
        # 不重新拆频，不破坏 FAF 已经完成的高低频融合。
        if self.use_channel_gate:
            d_proj = self.channel_gate_d(d_proj)
            s_proj = self.channel_gate_s(s_proj)

        # Step 3: 原始空间门控
        z = torch.cat([d_proj, s_proj], dim=1)
        m_spatial = self.spatial_gate(z)  # B, 1, H, W

        # Step 3.5: 可选双向正交空间门控
        # 先用 decoder+skip 的融合语义构造方向上下文，再与原 spatial gate 融合。
        if self.use_direction_gate:
            dir_feat = d_proj + s_proj
            m_direction = self.direction_gate(dir_feat)  # B, 1, H, W

            mix = torch.sigmoid(self.direction_mix)
            m_spatial = (1.0 - mix) * m_spatial + mix * m_direction

        # Step 4: 按门控筛选 skip
        s_sel = m_spatial * s_proj

        # Step 5: concat + 3×3 refine 投回 C_i
        fused = self.refine(torch.cat([d_proj, s_sel], dim=1))

        # Step 6: 可学习残差旁路，防止门控初期把信号掐死
        if self.use_FF:
            d_ff = self.ff_bypass_conv1(ffi_1)
            ff_bypass = self.ff_bypass_conv2(torch.cat([d_ff, ffi_2], dim=1))
            fused = fused + self.gamma * ff_bypass
        else:
            fused = fused + self.gamma * (d_proj + s_proj)

        # Step 7: ×2 upsample
        return self.upsample(fused)

    def extra_repr(self) -> str:
        g = float(self.gamma.detach().cpu().item())
        return (
            "in_channel={}, out_channel={}, gamma={:.3f}, "
            "channel_gate={}, direction_gate={}, use_FF={}"
        ).format(
            self.in_channel,
            self.out_channel,
            g,
            self.use_channel_gate,
            self.use_direction_gate,
            self.use_FF,
        )