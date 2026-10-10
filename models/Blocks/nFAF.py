"""
nFAF —— New Frequency-Aware Fusion Block（FAFBlock 改进版，v2 修订）。

v2 相对 v1 修复的病根
------------------------------------------------
v1 修对了 FAF 的两点（加性融合 → cat、refine 太浅 → 深度 refine），但自身仍有
四处自相矛盾/冗余，v2 一次性修复：

1. **对齐容量与假设自相矛盾**（v1 最大问题）：
   诊断说"1×1 做不了语义对齐"，但 v1 默认 dwsep（DW3×3 + PW1×1），跨通道部分
   仍是 1×1，翻译能力只比 FAF 多一层逐通道空间滤波；而参数预算却砸在 2d 通道的
   全 3×3 refine 上。v2 把预算挪回对齐：CNN 支默认 align_depth=2 个**全 3×3**
   （2048→1024→512，与原 FeatureFusion conv1+conv2 完全同构）；refine 第一层保留
   全 3×3（跨支混合的关键层），其后各层可选 dwsep 省参数。

2. **门控零和耦合 + 双门相乘压幅值**：
   v1 用 (1-m) 把两支强制竞争（建筑边缘恰恰两支都要），且 spat×chan 两个
   sigmoid 相乘初始期望 ~0.25，信号进 refine 前先被打折。v2 改为**两支各自独立
   的残差式门**：w = (1 + tanh(s)) ⊙ (1 + tanh(c))，门末层 zero-init，
   **初始严格恒等（w=1）**——初始状态即"原版 FeatureFusion"，门从不干预开始学偏置。
   空间门输出 2 通道（两支各一），通道门输出 2d（两支各 d），不再共享/互补。

3. **α/β/λ 在 cat 模式下不可辨识**：
   cat 后第一层 refine 是 Conv+BN，BN 把每支整体尺度归一化掉，三个无约束标量
   与 BN affine 自由度完全重叠，纯添噪还可能学成负数翻转语义。v2 整体删除
   （恒等初始化的残差门已涵盖其表达能力）。

4. **高频差分作用在 ReLU 之后**：
   v1 的 c_high = c_align - AvgPool(c_align) 算在 ReLU 后，负半轴信息已被砍，
   差出的"高频"有偏。v2 对齐末层只到 BN（pre-act），差分在带符号特征上做，
   ReLU 在差分之后才进入融合路径；且高频改为**残差注入**：
   c_used = c_align + γ_h·c_high，γ_h 可学习、初值 0（默认 use_smooth 仍关闭）。

管线级提示（不在本文件内，需改训练配置）：
   repvit native_mode=false 时 1×1 先把 512 升到 2048，进本模块又被压回 512，
   两次纯线性映射背靠背是低秩浪费。建议 native_mode=true 并把 in_channel_ffb
   改为 [512, 256, 128, 64]。

接口与 FAFBlock / FeatureFusion 完全一致（drop-in 替换）
------------------------------------------------
    forward(x_mit, x_cnn, capture=None) -> Tensor
    - x_mit: 低频/语义支（MiT / GroupMamba / MambaVision），通道 = mit_in_channel
    - x_cnn: 高频/细节支（ConvNeXt / ResNet / RepViT），通道 = in_channel
    - 输出通道 = 2 * out_channel，下游 decoder 无需改动

未显式传入（None）的参数会从 ``models/Blocks/block.yaml`` 的 `nFAF` 段读取。

消融指南（围绕"恒等基线 + 单变量加回"）：
    - 复刻原版 FeatureFusion：use_spatial_gate=false, use_channel_gate=false,
      use_smooth=false（align_depth=2 + refine_depth=2 即原版同构）
    - 在此之上逐项打开 spatial/channel gate、use_smooth 观察增益
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def _conv_bn(in_c: int, out_c: int, kernel: int, dwsep: bool, act: bool) -> nn.Module:
    """Conv(+BN)(+ReLU)。dwsep=True 且 kernel>1 时为 DW(k)+BN(+ReLU)+PW(1×1)+BN(+ReLU)。

    act=False 时末尾不加 ReLU（pre-activation 输出，供高频差分使用）。
    """
    pad = kernel // 2
    if (not dwsep) or kernel == 1:
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    layers = [
        nn.Conv2d(in_c, in_c, kernel_size=kernel, padding=pad, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _align_channel_schedule(in_c: int, out_c: int, depth: int) -> list:
    """对齐支路的通道日程：逐层减半但不低于 out_c，最后一层落到 out_c。

    depth=2, in=2048, out=512 → [2048, 1024, 512]（与原 FeatureFusion conv1+conv2 一致）
    depth=1 → [in, out]
    """
    chans = [in_c]
    for i in range(depth - 1):
        nxt = max(out_c, chans[-1] // 2)
        chans.append(nxt)
    chans.append(out_c)
    return chans


class _AlignBranch(nn.Module):
    """多层对齐；最后一层只到 BN（pre-act），forward 返回 (pre_act, post_relu)。"""

    def __init__(self, in_c: int, out_c: int, kernel: int, depth: int, dwsep: bool):
        super().__init__()
        chans = _align_channel_schedule(in_c, out_c, depth)
        blocks = []
        for i in range(len(chans) - 1):
            last = i == len(chans) - 2
            blocks.append(
                _conv_bn(chans[i], chans[i + 1], kernel=kernel, dwsep=dwsep and kernel > 1, act=not last)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        pre = self.blocks(x)
        return pre, F.relu(pre)


class _ResidualSpatialGate(nn.Module):
    """残差式空间门：输出 (B, 2, H, W) 的 tanh 残差，末层 zero-init → 初始为 0。"""

    def __init__(self, cat_dim: int, hidden: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv2d(cat_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, 2, kernel_size=kernel, padding=pad, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cat_feat: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.body(cat_feat)))


class _ResidualChannelGate(nn.Module):
    """残差式通道门：GAP → MLP → tanh，输出 (B, 2*d, 1, 1)，末层 zero-init。"""

    def __init__(self, cat_dim: int, d: int, reduction: int):
        super().__init__()
        hidden = max(cat_dim // max(1, reduction), 16)
        self.fc1 = nn.Linear(cat_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 2 * d, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cat_feat: torch.Tensor) -> torch.Tensor:
        z = cat_feat.mean(dim=[2, 3])
        w = torch.tanh(self.fc2(F.relu(self.fc1(z), inplace=True)))
        return w.unsqueeze(-1).unsqueeze(-1)


class nFAF(nn.Module):
    """New Frequency-Aware Fusion Block v2（drop-in 替换 FeatureFusion / FAFBlock）。"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stage_idx: int,
        mit_in_channel: Optional[int] = None,
        # 以下参数为 None 时从 block.yaml 的 nFAF 段读取
        align_mode: Optional[str] = None,
        align_kernel: Optional[int] = None,
        align_depth: Optional[int] = None,
        align_use_dwsep: Optional[bool] = None,
        fusion_mode: Optional[str] = None,
        use_spatial_gate: Optional[bool] = None,
        use_channel_gate: Optional[bool] = None,
        gate_reduction: Optional[int] = None,
        spatial_gate_kernel: Optional[int] = None,
        use_smooth: Optional[bool] = None,
        smooth_kernel: Optional[int] = None,
        smooth_gamma_init: Optional[float] = None,
        refine_depth: Optional[int] = None,
        refine_tail_dwsep: Optional[bool] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("nFAF")

        if mit_in_channel is None:
            mit_in_channel = out_channel

        align_mode = str(cfg.get("align_mode", "cnn") if align_mode is None else align_mode).lower()
        if align_mode not in ("cnn", "both", "none"):
            raise ValueError("align_mode 仅支持 cnn|both|none，收到: {}".format(align_mode))
        align_kernel = int(cfg.get("align_kernel", 3) if align_kernel is None else align_kernel)
        align_depth = int(cfg.get("align_depth", 2) if align_depth is None else align_depth)
        align_use_dwsep = bool(cfg.get("align_use_dwsep", False) if align_use_dwsep is None else align_use_dwsep)

        fusion_mode = str(cfg.get("fusion_mode", "cat") if fusion_mode is None else fusion_mode).lower()
        if fusion_mode not in ("cat", "add"):
            raise ValueError("fusion_mode 仅支持 cat|add，收到: {}".format(fusion_mode))

        use_spatial_gate = bool(cfg.get("use_spatial_gate", True) if use_spatial_gate is None else use_spatial_gate)
        use_channel_gate = bool(cfg.get("use_channel_gate", True) if use_channel_gate is None else use_channel_gate)
        gate_reduction = int(cfg.get("gate_reduction", 4) if gate_reduction is None else gate_reduction)
        spatial_gate_kernel = int(cfg.get("spatial_gate_kernel", 3) if spatial_gate_kernel is None else spatial_gate_kernel)

        use_smooth = bool(cfg.get("use_smooth", False) if use_smooth is None else use_smooth)
        smooth_kernel = int(cfg.get("smooth_kernel", 3) if smooth_kernel is None else smooth_kernel)
        smooth_gamma_init = float(cfg.get("smooth_gamma_init", 0.0) if smooth_gamma_init is None else smooth_gamma_init)

        refine_depth = int(cfg.get("refine_depth", 2) if refine_depth is None else refine_depth)
        refine_tail_dwsep = bool(cfg.get("refine_tail_dwsep", False) if refine_tail_dwsep is None else refine_tail_dwsep)

        if align_kernel % 2 == 0:
            raise ValueError("align_kernel 必须是奇数，收到: {}".format(align_kernel))
        if spatial_gate_kernel % 2 == 0:
            raise ValueError("spatial_gate_kernel 必须是奇数，收到: {}".format(spatial_gate_kernel))
        if smooth_kernel % 2 == 0:
            raise ValueError("smooth_kernel 必须是奇数，收到: {}".format(smooth_kernel))
        if align_depth < 1:
            raise ValueError("align_depth 必须 >= 1，收到: {}".format(align_depth))
        if refine_depth < 1:
            raise ValueError("refine_depth 必须 >= 1，收到: {}".format(refine_depth))

        d = out_channel
        self.d = d
        self.stage_idx = stage_idx
        self.align_mode = align_mode
        self.fusion_mode = fusion_mode
        self.use_spatial_gate = use_spatial_gate
        self.use_channel_gate = use_channel_gate
        self.use_smooth = use_smooth
        self.smooth_kernel = smooth_kernel
        self.smooth_pad = smooth_kernel // 2
        self.refine_depth = refine_depth

        # Step 1: 语义 + 通道对齐
        #   - CNN/高频支：align_depth 层全 3×3（默认 2 层，与原 FeatureFusion conv1+conv2 同构）
        #   - MiT/低频支：默认 1×1 仅通道对齐（align_mode=both 时同 CNN 支配置）
        c_kernel = align_kernel if align_mode in ("cnn", "both") else 1
        g_kernel = align_kernel if align_mode == "both" else 1
        c_depth = align_depth if c_kernel > 1 else 1
        g_depth = align_depth if g_kernel > 1 else 1
        self.align_c = _AlignBranch(in_channel, d, kernel=c_kernel, depth=c_depth, dwsep=align_use_dwsep)
        self.align_g = _AlignBranch(mit_in_channel, d, kernel=g_kernel, depth=g_depth, dwsep=align_use_dwsep)

        # Step 2: 高频残差注入系数（pre-act 差分，默认关闭；开启时 γ 从 0 学起）
        if use_smooth:
            self.smooth_gamma = nn.Parameter(torch.tensor(float(smooth_gamma_init)))
        else:
            self.smooth_gamma = None

        # Step 3: 残差式门控（两支独立，无 (1-m) 零和耦合；zero-init → 初始恒等）
        hidden = max(d // max(1, gate_reduction), 16)
        self.spatial_gate = (
            _ResidualSpatialGate(2 * d, hidden, kernel=spatial_gate_kernel) if use_spatial_gate else None
        )
        self.channel_gate = (
            _ResidualChannelGate(2 * d, d, reduction=gate_reduction) if use_channel_gate else None
        )

        # Step 4: 深度 refine。
        #   第一层始终全 3×3（跨支混合的关键层，不省）；其后各层可选 dwsep 省参数。
        #   - cat 融合：输入 2d，第一层 2d→2d
        #   - add 融合：输入 d，第一层 d→2d（顺带扩通道）
        refine_in = 2 * d if fusion_mode == "cat" else d
        layers = [_conv_bn(refine_in, 2 * d, kernel=3, dwsep=False, act=True)]
        for _ in range(refine_depth - 1):
            layers.append(_conv_bn(2 * d, 2 * d, kernel=3, dwsep=refine_tail_dwsep, act=True))
        self.refine = nn.Sequential(*layers)

        print(
            "[nFAF-v2] stage{}: align_mode={} (c_k={},depth={},dwsep={}) fusion={} "
            "spatial_gate={} channel_gate={} use_smooth={} refine_depth={} (tail_dwsep={})".format(
                stage_idx, align_mode, c_kernel, c_depth, int(align_use_dwsep),
                fusion_mode, int(use_spatial_gate), int(use_channel_gate),
                int(use_smooth), refine_depth, int(refine_tail_dwsep),
            ),
            flush=True,
        )

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
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
        # Step 1: 语义 + 通道对齐（pre 为带符号 pre-activation，align 为 ReLU 后）
        c_pre, c_align = self.align_c(x_cnn)
        g_pre, g_align = self.align_g(x_mit)

        if capture is not None:
            capture["x_mit"] = x_mit.detach().cpu()
            capture["x_cnn"] = x_cnn.detach().cpu()

        # Step 2: 高频残差注入（差分在 pre-act 带符号特征上做，无 ReLU 偏置）
        if self.use_smooth:
            c_high = c_pre - self._smooth(c_pre)
            c_used = c_align + self.smooth_gamma * c_high
        else:
            c_high = None
            c_used = c_align
        g_used = g_align

        if capture is not None:
            capture["use_smooth"] = self.use_smooth
            if c_high is not None:
                capture["c_smooth"] = (c_pre - c_high).detach().cpu()
                capture["c_high"] = c_high.detach().cpu()
            else:
                capture["c_high"] = c_used.detach().cpu()

        # Step 3: 残差式门控（初始 w=1 严格恒等；两支独立，无零和竞争）
        cat_feat = torch.cat([c_align, g_align], dim=1)
        if capture is not None:
            capture["cat_feat"] = cat_feat.detach().cpu()

        w_c = c_used.new_ones(1)
        w_g = c_used.new_ones(1)
        if self.spatial_gate is not None:
            s = self.spatial_gate(cat_feat)              # (B,2,H,W), 初始 0
            w_c = w_c * (1.0 + s[:, 0:1])
            w_g = w_g * (1.0 + s[:, 1:2])
        if self.channel_gate is not None:
            ch = self.channel_gate(cat_feat)             # (B,2d,1,1), 初始 0
            w_c = w_c * (1.0 + ch[:, : self.d])
            w_g = w_g * (1.0 + ch[:, self.d:])

        c_contrib = w_c * c_used
        g_contrib = w_g * g_used

        if capture is not None:
            capture["w_c"] = (w_c * torch.ones_like(c_used)).detach().cpu()
            capture["w_g"] = (w_g * torch.ones_like(g_used)).detach().cpu()

        # Step 4: 融合 + 深度 refine
        if self.fusion_mode == "cat":
            fused = torch.cat([c_contrib, g_contrib], dim=1)
        else:
            fused = c_contrib + g_contrib

        out = self.refine(fused)

        if capture is not None:
            capture["faf_out"] = out.detach().cpu()
        return out

    def extra_repr(self) -> str:
        gamma = (
            "{:.3f}".format(float(self.smooth_gamma.detach().cpu().item()))
            if self.smooth_gamma is not None
            else "off"
        )
        return (
            "stage_idx={}, d={}, align_mode={}, fusion_mode={}, "
            "spatial_gate={}, channel_gate={}, smooth_gamma={}, refine_depth={}"
        ).format(
            self.stage_idx, self.d, self.align_mode, self.fusion_mode,
            self.use_spatial_gate, self.use_channel_gate, gamma, self.refine_depth,
        )
