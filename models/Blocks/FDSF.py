"""
FDSF —— Frequency-Decoupled Skip Fusion.

定位
----
FAFBlock 已经在 encoder 侧把双主干的高/低频做完融合，FAF 输出的 skip
特征本质上是「频率融合后的混合特征」。decoder 侧不应该再去显式做
S^H = S - AvgPool(S) / S^L = AvgPool(S) 这种重复分解 —— 那会和 FAF 自相打架。

FDSF 的做法是 **decoder-guided adaptive skip injection**：
让 decoder 上层语义 D 来决定「该如何读取 FAF 融合后的 skip」，
通过空间门 + 通道门组合出像素+通道两个维度的注入权重，
而不去重新分解频率。

接口约定（drop-in 替换 GSFB / DecoderBlock）
-----------------------------------------
    forward(ffi_1, ffi_2) -> Tensor
    - ffi_1: 上一层 decoder 输出（深层语义、低分辨率），channels = in_channel
             空间尺寸若与 ffi_2 不同，内部自动 bilinear 上采样
    - ffi_2: 当前 stage 的 FAF skip 特征（浅层、高分辨率），channels = out_channel
    - 返回 tensor 通道 = out_channel，空间尺寸 = 2 × ffi_2 的 H/W
      （末端 ×2 上采样与原 DecoderBlock 末尾一致，下游 FusedDecoder 无需改动）

主流程
------
    D_proj = Conv1x1+BN+ReLU(ffi_1)        # (B, c, H, W)
    S_proj = Conv1x1+BN+ReLU(ffi_2)        # (B, c, H, W)
    Z      = cat(D_proj, S_proj)           # (B, 2c, H, W)
    M_spat = Sigmoid(Conv3x3(BN+ReLU(Conv1x1(Z))))  # (B, 1, H, W)
    M_chan = Sigmoid(MLP(GAP(Z)))                   # (B, c, 1, 1)
    M      = M_spat * M_chan                         # 广播到 (B, c, H, W)
    F_add  = D_proj + M * S_proj + γ * (D_proj + S_proj)
    F      = Conv3x3+BN+ReLU(F_add)        # 末端 refine
    out    = Upsample×2(F)

设计要点
--------
- 空间门 1 通道：负责「在哪些像素更需要 skip 注入」（边界 / 区域结构）
- 通道门 C 通道：负责「在哪些特征通道更需要 skip 注入」
- 残差项 γ*(D+S)：防止门控初期把信号掐死，γ 可学习初值 0.1（同 GSFB）
- 末端 3×3 refine：让加法融合后的特征再 mixing 一次（仅靠加法不够稳）
- 末端 ×2 上采样：与原 DecoderBlock / GSFB 输出契约严格对齐
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


class _ChannelGate(nn.Module):
    """GAP → MLP → Sigmoid 得到 (B, c, 1, 1)。"""

    def __init__(self, in_dim: int, out_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(in_dim // max(1, reduction), 16)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, cat_feat: torch.Tensor) -> torch.Tensor:
        z = cat_feat.mean(dim=[2, 3])  # GAP -> (B, in_dim)
        w = torch.sigmoid(self.mlp(z)).unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
        return w


class _LiteResRefine(nn.Module):
    """轻量残差细化块。

    DWConv3x3 + BN + ReLU
    PWConv1x1 + BN
    Residual Add
    ReLU

    相比普通双 3x3 ResBlock，参数量更低，同时保留局部空间混合能力。
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FDSF(nn.Module):
    """Frequency-Decoupled Skip Fusion（drop-in 替换 DecoderBlock / GSFB）。

    该版本兼容 V1 与 V2。V2 通过 block.yaml 中的额外参数开启：

        use_diff: true
        use_lite_res_refine: true
        use_gated_residual: true

    原版 V1:
        D_proj = Align_D(D)
        S_proj = Align_S(S)
        Z = Concat(D_proj, S_proj)
        M = SpatialGate(Z) * ChannelGate(Z)
        F_add = D_proj + M * S_proj + gamma * (D_proj + S_proj)
        Out = Upsample(Refine(F_add))

    V2:
        D_proj = Align_D(D)
        S_proj = Align_S(S)
        E = |D_proj - S_proj|
        Z = Concat(D_proj, S_proj, E)
        M = SpatialGate(Z) * ChannelGate(Z)
        F_add = D_proj + M * S_proj + gamma_d * D_proj + gamma_s * S_proj
              = (1 + gamma_d) * D_proj + (M + gamma_s) * S_proj
        Out = Upsample(LiteResRefine(F_add))

    Args:
        in_channel:  ffi_1（上一层 decoder 输出）通道数
        out_channel: ffi_2（当前 skip）通道数，亦为本模块输出通道数
        stage_idx: decoder 层级索引，用于 gated residual 的默认 gamma 初始化。
                   推荐约定：
                       3 = 最深层 skip fusion，更偏 decoder 语义
                       2 = 中间层 skip fusion，decoder/skip 均衡
                       1 = 最浅层 skip fusion，更偏 skip 细节
        gate_reduction: 空间/通道门中间层压缩比；None 时使用 block.yaml
        spatial_gate_kernel: 空间门末层卷积核（奇数）；None 时使用 block.yaml
        residual_gamma_init: V1 的 gamma 初值；None 时使用 block.yaml
        gamma_d_init: V2 gated residual 中 decoder 残差权重初值
        gamma_s_init: V2 gated residual 中 skip 残差权重初值
        learnable_residual: gamma / gamma_d / gamma_s 是否可学习
        use_channel_gate: 是否启用通道门
        use_spatial_gate: 是否启用空间门
        use_refine: 是否启用 refine；False 时为 Identity
        use_diff: 是否启用 Difference-aware 分支 E=|D-S|
        use_lite_res_refine: refine 是否使用轻量 DW+PW 残差块
        use_gated_residual: 是否用 gamma_d*D + gamma_s*S 替代 gamma*(D+S)

    任何未传入（None）的参数都会优先从 ``models/Blocks/block.yaml`` 的 `FDSF` 段读取。
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stage_idx: Optional[int] = None,
        gate_reduction: Optional[int] = None,
        spatial_gate_kernel: Optional[int] = None,
        residual_gamma_init: Optional[float] = None,
        gamma_d_init: Optional[float] = None,
        gamma_s_init: Optional[float] = None,
        learnable_residual: Optional[bool] = None,
        use_channel_gate: Optional[bool] = None,
        use_spatial_gate: Optional[bool] = None,
        use_refine: Optional[bool] = None,
        use_diff: Optional[bool] = None,
        use_lite_res_refine: Optional[bool] = None,
        use_gated_residual: Optional[bool] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("FDSF")

        gate_reduction = int(
            cfg.get("gate_reduction", 4)
            if gate_reduction is None
            else gate_reduction
        )
        spatial_gate_kernel = int(
            cfg.get("spatial_gate_kernel", 3)
            if spatial_gate_kernel is None
            else spatial_gate_kernel
        )
        residual_gamma_init = float(
            cfg.get("residual_gamma_init", 0.1)
            if residual_gamma_init is None
            else residual_gamma_init
        )
        learnable_residual = bool(
            cfg.get("learnable_residual", True)
            if learnable_residual is None
            else learnable_residual
        )
        use_channel_gate = bool(
            cfg.get("use_channel_gate", True)
            if use_channel_gate is None
            else use_channel_gate
        )
        use_spatial_gate = bool(
            cfg.get("use_spatial_gate", True)
            if use_spatial_gate is None
            else use_spatial_gate
        )
        use_refine = bool(
            cfg.get("use_refine", True)
            if use_refine is None
            else use_refine
        )

        # V2 开关，默认 False，确保不影响原版 FDSF。
        use_diff = bool(
            cfg.get("use_diff", False)
            if use_diff is None
            else use_diff
        )
        use_lite_res_refine = bool(
            cfg.get("use_lite_res_refine", False)
            if use_lite_res_refine is None
            else use_lite_res_refine
        )
        use_gated_residual = bool(
            cfg.get("use_gated_residual", False)
            if use_gated_residual is None
            else use_gated_residual
        )

        if spatial_gate_kernel % 2 == 0:
            raise ValueError("spatial_gate_kernel 必须是奇数，收到: {}".format(spatial_gate_kernel))

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stage_idx = stage_idx

        c = out_channel
        self.use_channel_gate = use_channel_gate
        self.use_spatial_gate = use_spatial_gate
        self.use_refine = use_refine
        self.use_diff = use_diff
        self.use_lite_res_refine = use_lite_res_refine
        self.use_gated_residual = use_gated_residual

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

        # V1: Z = [D, S]      -> 2c
        # V2: Z = [D, S, |D-S|] -> 3c
        gate_in_dim = 3 * c if use_diff else 2 * c

        if use_spatial_gate:
            hidden = max(c // gate_reduction, 16)
            pad = spatial_gate_kernel // 2
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(gate_in_dim, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, kernel_size=spatial_gate_kernel, padding=pad, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None

        if use_channel_gate:
            self.channel_gate = _ChannelGate(gate_in_dim, c, reduction=gate_reduction)
        else:
            self.channel_gate = None

        # Residual path:
        # - V1: gamma * (D + S)
        # - V2: gamma_d * D + gamma_s * S
        gamma = torch.tensor(float(residual_gamma_init))
        if learnable_residual:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

        gd_default, gs_default = self._default_gamma(stage_idx)
        gamma_d_init = float(
            cfg.get("gamma_d_init", gd_default)
            if gamma_d_init is None
            else gamma_d_init
        )
        gamma_s_init = float(
            cfg.get("gamma_s_init", gs_default)
            if gamma_s_init is None
            else gamma_s_init
        )

        gamma_d = torch.tensor(float(gamma_d_init))
        gamma_s = torch.tensor(float(gamma_s_init))
        if learnable_residual:
            self.gamma_d = nn.Parameter(gamma_d)
            self.gamma_s = nn.Parameter(gamma_s)
        else:
            self.register_buffer("gamma_d", gamma_d)
            self.register_buffer("gamma_s", gamma_s)

        if not use_refine:
            self.refine = nn.Identity()
        elif use_lite_res_refine:
            self.refine = _LiteResRefine(c)
        else:
            self.refine = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    @staticmethod
    def _default_gamma(stage_idx: Optional[int]) -> Tuple[float, float]:
        """V2 gated residual 的 stage-wise 默认初始化。

        gamma_d: decoder 语义保底权重
        gamma_s: skip 细节保底权重

        推荐 stage_idx:
            3 = 深层：更依赖 decoder 语义
            2 = 中层：二者均衡
            1 = 浅层：更依赖 skip 细节
        """
        if stage_idx is None:
            return 0.10, 0.10

        defaults = {
            3: (0.15, 0.05),
            2: (0.10, 0.10),
            1: (0.05, 0.15),
        }
        return defaults.get(stage_idx, (0.10, 0.10))

    def forward(self, ffi_1: torch.Tensor, ffi_2: torch.Tensor) -> torch.Tensor:
        # 空间对齐到 skip 尺寸；调用方一般已对齐，这里兜底。
        if ffi_1.shape[-2:] != ffi_2.shape[-2:]:
            ffi_1 = F.interpolate(
                ffi_1,
                size=ffi_2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        d_proj = self.align_d(ffi_1)
        s_proj = self.align_s(ffi_2)

        if self.use_diff:
            diff = torch.abs(d_proj - s_proj)
            z = torch.cat([d_proj, s_proj, diff], dim=1)
        else:
            z = torch.cat([d_proj, s_proj], dim=1)

        if self.spatial_gate is not None:
            m_spat = self.spatial_gate(z)  # (B, 1, H, W)
        else:
            m_spat = z.new_ones(z.size(0), 1, z.size(2), z.size(3))

        if self.channel_gate is not None:
            m_chan = self.channel_gate(z)  # (B, c, 1, 1)
        else:
            m_chan = z.new_ones(z.size(0), d_proj.size(1), 1, 1)

        # 通道门 × 空间门：广播到 (B, c, H, W)
        m = m_chan * m_spat

        if self.use_gated_residual:
            # V2: D + M·S + gamma_d·D + gamma_s·S
            fused = (
                d_proj
                + m * s_proj
                + self.gamma_d * d_proj
                + self.gamma_s * s_proj
            )
        else:
            # V1: D + M·S + gamma·(D + S)
            fused = d_proj + m * s_proj + self.gamma * (d_proj + s_proj)

        fused = self.refine(fused)
        return self.upsample(fused)

    def extra_repr(self) -> str:
        g = float(self.gamma.detach().cpu().item())
        gd = float(self.gamma_d.detach().cpu().item())
        gs = float(self.gamma_s.detach().cpu().item())
        return (
            "in_channel={}, out_channel={}, stage_idx={}, "
            "gamma={:.3f}, gamma_d={:.3f}, gamma_s={:.3f}, "
            "diff={}, gated_residual={}, lite_res_refine={}, "
            "spatial_gate={}, channel_gate={}"
        ).format(
            self.in_channel,
            self.out_channel,
            self.stage_idx,
            g,
            gd,
            gs,
            self.use_diff,
            self.use_gated_residual,
            self.use_lite_res_refine,
            self.spatial_gate is not None,
            self.channel_gate is not None,
        )












#V2 通过block.yaml的一些额外参数来开启
#use_diff: true\use_lite_res_refine: true\use_gated_residual
#FDSF 现在这个解码器模块整体偏简单
#它目前的核心就是：
# D_proj：解码器上层语义
# S_proj：GC-FAF encoder 传来的 skip 特征
# M：由 D_proj 和 S_proj 共同生成的空间-通道门控

# 原版是：

# D_proj = Align_D(D)
# S_proj = Align_S(S)

# Z = Concat(D_proj, S_proj)

# M_spat = SpatialGate(Z)
# M_chan = ChannelGate(Z)
# M = M_spat * M_chan

# F_add = D_proj + M * S_proj + γ * (D_proj + S_proj)

# F = Refine(F_add)
# Out = Upsample(F)
# V2 改成：

# D_proj = Align_D(D)
# S_proj = Align_S(S)

# E = |D_proj - S_proj|

# Z = Concat(D_proj, S_proj, E)

# M_spat = SpatialGate(Z)
# M_chan = ChannelGate(Z)
# M = M_spat * M_chan

# F_add = D_proj + M * S_proj + γ_d * D_proj + γ_s * S_proj

# F = LiteResRefine(F_add)

# Out = Upsample(F)

# 也就是：

# F_add = (1 + γ_d) * D_proj + (M + γ_s) * S_proj
# 2. 三个增强分别解决什么问题
# 增强一：Difference-aware 分支

# 加入：

# E = |D_proj - S_proj|

# 然后门控输入从：

# Z = Concat(D_proj, S_proj)

# 变成：

# Z = Concat(D_proj, S_proj, |D_proj - S_proj|)

# 它的作用是让门控知道：

# 哪些区域深层语义和浅层 skip 一致；
# 哪些区域二者差异很大，需要重点融合或修正。

# 增强二：gated residual 分开控制 D 和 S

# 原版：

# γ * (D_proj + S_proj)

# D 和 S 用同一个 γ，比较粗。

# V2：

# γ_d * D_proj + γ_s * S_proj

# 这样可以让 decoder 语义路径和 skip 细节路径有不同保底权重。

# 推荐初值：

# 深层 decoder stage：γ_d 稍大，γ_s 稍小
# 浅层 decoder stage：γ_d 稍小，γ_s 稍大

# 例如三层 decoder 可以这样：

# stage 3：γ_d = 0.15, γ_s = 0.05
# stage 2：γ_d = 0.10, γ_s = 0.10
# stage 1：γ_d = 0.05, γ_s = 0.15

# 这样逻辑很清楚：

# 深层更信 decoder 语义；
# 浅层更信 skip 细节。

# 增强三：refine 改成轻量残差块

# 原版：

# Conv3x3 + BN + ReLU

# V2 推荐改成：

# DWConv3x3 + BN + ReLU
# PWConv1x1 + BN
# Residual Add
# ReLU

# 也就是一个轻量深度可分离残差细化块。

# 它的作用是：

# 1. 增强融合后的局部空间混合；
# 2. 避免普通 3×3 refine 过薄；
# 3. 参数量比普通双 3×3 ResBlock 小很多；
# 4. 更适合边界和局部高度突变恢复。