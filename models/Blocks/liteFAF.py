"""
liteFAF —— Lite Frequency-Aware Fusion（真·频域分离的轻量融合模块）。

设计动机
------------------------------------------------
nFAF v2 修复了 FAF 的病根，但本质仍是"FeatureFusion + 门控"，频率感知只停留在
可选的 use_smooth。liteFAF 把"频率"做实，核心思想：

    先显式分频，再同频段融合；低频以 Mamba/MiT 为主、高频以 CNN 为主，
    门控只做带内残差注入，从结构上避免跨语义空间的逐元素强融合。

流程
------------------------------------------------
    c = Align_c(x_cnn)            # 见下「分级对齐」，→ d
    g = Align_g(x_mit)            # 1×1+BN（无 ReLU），→ d

    # 显式分频（两种方式，均为零可学习参数、保持 H×W）：
    #   use_DWT=0：stride=1 AvgPool 低通，high = x - low（各向同性两带）
    #   use_DWT=1：无下采样 Haar，LL + LH/HL/HH 方向高频带（见 _HaarSplit）
    c_low, c_high = split(c)      # 高频带符号、无 ReLU 偏置
    g_low, g_high = split(g)

    # 带内融合：主支恒等保底 + 副支门控残差注入（无 (1-m) 零和竞争）
    L = g_low  + γ_L ⊙ c_low      # γ_L: GAP(cat(lows)) → MLP → tanh，(B,d,1,1)，zero-init
    H = c_high + γ_H ⊙ g_high     # γ_H: Conv(cat(highs)) → tanh，(B,1,H,W)，zero-init

    out = Refine(cat(L, H))       # 第一层全 3×3（跨支/跨频段混合），其后 dwsep，→ 2d

关键性质
------------------------------------------------
- γ 末层 zero-init：初始时 out = Refine(cat(g_low, c_high))，即"低频取 Mamba、
  高频取 CNN 的纯分频 cat 融合"，永远有一个不差的起点；门学的是"要从对面支借多少"。
- 分频为显式零参数操作，"频率感知"有真实依据，且高频在无 ReLU 的带符号特征上计算。
- 轻量：对齐默认 dwsep、refine 仅第一层全卷积，无 α/β/λ、无双门相乘。
  stage4 (in=2048, d=512) 参数量约为 nFAF v2 的 1/3、原 FeatureFusion 的 1/4。

分级对齐（strong_align_stages）
------------------------------------------------
γ 注入是逐元素加法，加法语义有效的前提是两支已被"翻译"到可共处的空间；纯线性
投影（DW+PW / 1×1）翻译能力弱。若某 stage 两支语义差异大（可视化观察 stage3/4
普遍如此），梯度会发现"开门干涉、关门安全"，γ 退化到 0，模块退化成硬路由，
跨支互补收益清零。因此对语义差异大的深层 stage 给 CNN 支换**非线性瓶颈翻译**：

    强对齐（stage ∈ strong_align_stages，默认 [3,4]）：
        Conv1×1(in→d)+BN+ReLU → Conv3×3(d→d)+BN      # 非线性 + 空间/跨通道混合
    轻对齐（其余 stage）：
        DW3×3+PW1×1+BN                                # 线性投影，维持 lite

末层都只到 BN（无 ReLU），分频仍在带符号特征上做。深层空间尺寸小，FLOPs 代价
可忽略；stage4 参数 +2.3M（瓶颈式，远低于全 3×3 直压 2048→512 的 +9.4M）。
Mamba/MiT 支保持 1×1：原版 FeatureFusion 仅翻译 CNN 支即可工作，证据上单侧翻译足够。

DWT 小波分频（use_DWT）
------------------------------------------------
AvgPool 差分的三个局限：只有低/高二分、高频无方向信息、无多频带表达。
use_DWT=1 时改用无下采样单层 Haar：
    LL（低频） + LH/HL/HH（垂直边缘/水平边缘/对角结构，方向高频带）
    - 分解无损：LL+LH+HL+HH = x
    - 每个方向带一个独立空间门 γ_k，门控可按方向调制注入强度
    - band_mix(1×1, 3d→d) 重组，权重初始化为三带求和（= x - LL），
      配合 γ zero-init，初始行为与 AvgPool 两带版完全一致（恒等起点不变）
注意：Haar 低通仅 2×2 均值（比 3×3 AvgPool 更"软"），方向也只有 3 个粗朝向；
收益取决于两支主干在不同朝向上的互补差异是否真实存在，故默认关闭、作为开关消融。

接口与 nFAF / FAFBlock / FeatureFusion 完全一致（drop-in 替换）
------------------------------------------------
    forward(x_mit, x_cnn, capture=None) -> Tensor
    - x_mit: 低频/语义支（MiT / GroupMamba / MambaVision），通道 = mit_in_channel
    - x_cnn: 高频/细节支（ConvNeXt / ResNet / RepViT），通道 = in_channel
    - 输出通道 = 2 * out_channel，下游 decoder 无需改动

未显式传入（None）的参数会从 ``models/Blocks/block.yaml`` 的 `liteFAF` 段读取。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def _align_conv(in_c: int, out_c: int, kernel: int, dwsep: bool) -> nn.Module:
    """轻对齐投影：线性（无 ReLU）+ 末端 BN，保留符号供分频使用。

    dwsep=True 且 kernel>1：DW(k) → PW(1×1) → BN（线性组合，等效一个可分离滤波器）
    否则：Conv(k) → BN
    """
    pad = kernel // 2
    if dwsep and kernel > 1:
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=kernel, padding=pad, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
    )


class _HaarSplit(nn.Module):
    """无下采样单层 Haar 分解（stride=1，replicate pad，保持 H×W）。

    四个 2×2 滤波器之和为单位冲激 → LL + LH + HL + HH = x（无损分解）。
    - LL：低频（2×2 均值）
    - LH：水平差分 → 响应垂直边缘
    - HL：垂直差分 → 响应水平边缘
    - HH：对角差分 → 响应对角结构
    相比 AvgPool 差分：高频带携带方向信息，门控可按方向独立调制。
    """

    def __init__(self):
        super().__init__()
        ll = [[0.25, 0.25], [0.25, 0.25]]
        lh = [[0.25, -0.25], [0.25, -0.25]]
        hl = [[0.25, 0.25], [-0.25, -0.25]]
        hh = [[0.25, -0.25], [-0.25, 0.25]]
        w = torch.tensor([ll, lh, hl, hh], dtype=torch.float32).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer("weight", w)

    def forward(self, x: torch.Tensor):
        b, c, h, wd = x.shape
        xp = F.pad(x, (0, 1, 0, 1), mode="replicate")
        w = self.weight.repeat(c, 1, 1, 1)        # (4c,1,2,2)，组序 [通道0 的 4 带, 通道1 的 4 带, ...]
        y = F.conv2d(xp, w, groups=c)             # (B, 4c, H, W)
        y = y.view(b, c, 4, h, wd)
        low = y[:, :, 0]
        bands = (y[:, :, 1], y[:, :, 2], y[:, :, 3])  # (LH, HL, HH)
        return low, bands


def _strong_align_conv(in_c: int, out_c: int, kernel: int) -> nn.Module:
    """强对齐（非线性瓶颈翻译）：Conv1×1(in→d)+BN+ReLU → Conv(k)(d→d)+BN。

    用于两支语义差异大的深层 stage：1×1 先降维省参数，3×3 做空间+跨通道的
    非线性语义翻译；末层只到 BN（无 ReLU），保证分频在带符号特征上进行。
    """
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=kernel, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
    )


class _LowGate(nn.Module):
    """低频带通道门：GAP(cat(g_low,c_low)) → MLP → tanh，(B,d,1,1)，zero-init。"""

    def __init__(self, d: int, reduction: int):
        super().__init__()
        hidden = max(2 * d // max(1, reduction), 16)
        self.fc1 = nn.Linear(2 * d, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, d, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cat_low: torch.Tensor) -> torch.Tensor:
        z = cat_low.mean(dim=[2, 3])
        w = torch.tanh(self.fc2(F.relu(self.fc1(z), inplace=True)))
        return w.unsqueeze(-1).unsqueeze(-1)


class _HighGate(nn.Module):
    """高频带空间门：Conv1×1 → ReLU → Conv(k)→out_ch → tanh，(B,out_ch,H,W)，zero-init。

    use_DWT=0 时 in_dim=2d、out_ch=1（单一高频带）；
    use_DWT=1 时 in_dim=6d、out_ch=3（LH/HL/HH 每个方向带一个独立门）。
    """

    def __init__(self, d: int, reduction: int, kernel: int, in_dim: int, out_ch: int = 1):
        super().__init__()
        hidden = max(d // max(1, reduction), 16)
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(hidden, out_ch, kernel_size=kernel, padding=pad, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cat_high: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.body(cat_high)))


class liteFAF(nn.Module):
    """Lite Frequency-Aware Fusion（drop-in 替换 FeatureFusion / FAFBlock / nFAF）。"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stage_idx: int,
        mit_in_channel: Optional[int] = None,
        # 以下参数为 None 时从 block.yaml 的 liteFAF 段读取
        align_kernel: Optional[int] = None,
        align_use_dwsep: Optional[bool] = None,
        strong_align: Optional[bool] = None,
        strong_align_stages: Optional[list] = None,
        use_DWT: Optional[bool] = None,
        lp_kernel: Optional[int] = None,
        gate_reduction: Optional[int] = None,
        spatial_gate_kernel: Optional[int] = None,
        refine_depth: Optional[int] = None,
        refine_tail_dwsep: Optional[bool] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("liteFAF")

        if mit_in_channel is None:
            mit_in_channel = out_channel

        align_kernel = int(cfg.get("align_kernel", 3) if align_kernel is None else align_kernel)
        align_use_dwsep = bool(cfg.get("align_use_dwsep", True) if align_use_dwsep is None else align_use_dwsep)
        if strong_align_stages is None:
            strong_align_stages = cfg.get("strong_align_stages", [3, 4])
        strong_align_stages = [int(s) for s in (strong_align_stages or [])]
        if strong_align is None:
            strong_align = stage_idx in strong_align_stages
        strong_align = bool(strong_align)
        use_DWT = bool(int(cfg.get("use_DWT", 0)) if use_DWT is None else use_DWT)
        lp_kernel = int(cfg.get("lp_kernel", 3) if lp_kernel is None else lp_kernel)
        gate_reduction = int(cfg.get("gate_reduction", 4) if gate_reduction is None else gate_reduction)
        spatial_gate_kernel = int(cfg.get("spatial_gate_kernel", 3) if spatial_gate_kernel is None else spatial_gate_kernel)
        refine_depth = int(cfg.get("refine_depth", 2) if refine_depth is None else refine_depth)
        refine_tail_dwsep = bool(cfg.get("refine_tail_dwsep", True) if refine_tail_dwsep is None else refine_tail_dwsep)

        if align_kernel % 2 == 0:
            raise ValueError("align_kernel 必须是奇数，收到: {}".format(align_kernel))
        if lp_kernel % 2 == 0:
            raise ValueError("lp_kernel 必须是奇数，收到: {}".format(lp_kernel))
        if spatial_gate_kernel % 2 == 0:
            raise ValueError("spatial_gate_kernel 必须是奇数，收到: {}".format(spatial_gate_kernel))
        if refine_depth < 1:
            raise ValueError("refine_depth 必须 >= 1，收到: {}".format(refine_depth))

        d = out_channel
        self.d = d
        self.stage_idx = stage_idx
        self.use_DWT = use_DWT
        self.lp_kernel = lp_kernel
        self.lp_pad = lp_kernel // 2
        self.refine_depth = refine_depth

        # Step 1: 分级对齐（末层均只到 BN，保留符号供分频）
        #   - 强对齐（深层，语义差异大）：1×1 瓶颈降维 + ReLU + 3×3 非线性翻译
        #   - 轻对齐（浅层）：线性 DW+PW 投影
        self.strong_align = strong_align
        if strong_align:
            self.align_c = _strong_align_conv(in_channel, d, kernel=align_kernel)
        else:
            self.align_c = _align_conv(in_channel, d, kernel=align_kernel, dwsep=align_use_dwsep)
        self.align_g = _align_conv(mit_in_channel, d, kernel=1, dwsep=False)

        # Step 2: 分频方式
        #   use_DWT=0：AvgPool 低通 + 差分（各向同性两带）
        #   use_DWT=1：无下采样 Haar，LL 低频 + LH/HL/HH 三个方向高频带，
        #              每带独立 γ 门；band_mix(1×1, 3d→d) 重组，权重初始化为三带求和
        #              （LH+HL+HH = x - LL），配合 γ zero-init，初始行为与两带版完全一致
        if use_DWT:
            self.dwt = _HaarSplit()
            self.band_mix = nn.Conv2d(3 * d, d, kernel_size=1, bias=False)
            with torch.no_grad():
                wm = torch.zeros(d, 3 * d, 1, 1)
                for j in range(d):
                    wm[j, j, 0, 0] = 1.0
                    wm[j, d + j, 0, 0] = 1.0
                    wm[j, 2 * d + j, 0, 0] = 1.0
                self.band_mix.weight.copy_(wm)
        else:
            self.dwt = None
            self.band_mix = None

        # Step 3: 带内残差门（zero-init → 初始为纯分频 cat 融合）
        self.low_gate = _LowGate(d, reduction=gate_reduction)
        if use_DWT:
            self.high_gate = _HighGate(
                d, reduction=gate_reduction, kernel=spatial_gate_kernel, in_dim=6 * d, out_ch=3,
            )
        else:
            self.high_gate = _HighGate(
                d, reduction=gate_reduction, kernel=spatial_gate_kernel, in_dim=2 * d, out_ch=1,
            )

        # Step 4: refine（第一层全 3×3 跨支/跨频段混合，其后 dwsep 省参数），输出 2d
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(2 * d, 2 * d, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(2 * d),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(refine_depth - 1):
            if refine_tail_dwsep:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(2 * d, 2 * d, kernel_size=3, padding=1, groups=2 * d, bias=False),
                        nn.BatchNorm2d(2 * d),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(2 * d, 2 * d, kernel_size=1, bias=False),
                        nn.BatchNorm2d(2 * d),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(2 * d, 2 * d, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(2 * d),
                        nn.ReLU(inplace=True),
                    )
                )
        self.refine = nn.Sequential(*layers)

        print(
            "[liteFAF] stage{}: align={} (k={},dwsep={}) split={} gate_red={} "
            "refine_depth={} (tail_dwsep={})".format(
                stage_idx,
                "strong(1x1+ReLU+3x3)" if strong_align else "lite(DW+PW)",
                align_kernel, int(align_use_dwsep),
                "haar-DWT(LL+LH/HL/HH)" if use_DWT else "avgpool(lp_k={})".format(lp_kernel),
                gate_reduction, refine_depth, int(refine_tail_dwsep),
            ),
            flush=True,
        )

    def _split(self, x: torch.Tensor):
        """零参数分频：low = AvgPool(x)（stride=1 低通），high = x - low。"""
        low = F.avg_pool2d(x, kernel_size=self.lp_kernel, stride=1, padding=self.lp_pad)
        return low, x - low

    def forward(
        self,
        x_mit: torch.Tensor,
        x_cnn: torch.Tensor,
        capture: Optional[dict] = None,
    ) -> torch.Tensor:
        # Step 1: 对齐到 d（带符号）
        c = self.align_c(x_cnn)
        g = self.align_g(x_mit)

        if capture is not None:
            capture["x_mit"] = x_mit.detach().cpu()
            capture["x_cnn"] = x_cnn.detach().cpu()

        # Step 2: 显式分频
        if self.use_DWT:
            c_low, c_bands = self.dwt(c)   # bands: (LH, HL, HH)，各 (B,d,H,W)
            g_low, g_bands = self.dwt(g)
            c_high = c - c_low             # = LH+HL+HH，供 capture / 门控语义参考
            g_high = g - g_low
        else:
            c_low, c_high = self._split(c)
            g_low, g_high = self._split(g)

        if capture is not None:
            capture["use_smooth"] = True
            capture["c_smooth"] = c_low.detach().cpu()
            capture["c_high"] = c_high.detach().cpu()
            capture["g_low"] = g_low.detach().cpu()
            capture["g_high"] = g_high.detach().cpu()

        # Step 3: 带内融合（主支恒等 + 副支残差注入）
        gamma_l = self.low_gate(torch.cat([g_low, c_low], dim=1))     # (B,d,1,1)
        band_low = g_low + gamma_l * c_low

        if self.use_DWT:
            # 每个方向带一个独立空间门 (B,3,H,W)，再经 band_mix(初始=三带求和)重组回 d
            gate_in = torch.cat(list(c_bands) + list(g_bands), dim=1)  # (B,6d,H,W)
            gamma_h = self.high_gate(gate_in)                          # (B,3,H,W)
            fused_bands = [
                c_bands[k] + gamma_h[:, k:k + 1] * g_bands[k] for k in range(3)
            ]
            band_high = self.band_mix(torch.cat(fused_bands, dim=1))
        else:
            gamma_h = self.high_gate(torch.cat([c_high, g_high], dim=1))  # (B,1,H,W)
            band_high = c_high + gamma_h * g_high

        if capture is not None:
            capture["gamma_l"] = (gamma_l * torch.ones_like(c_low)).detach().cpu()
            capture["gamma_h"] = (gamma_h * torch.ones_like(c_high[:, :1])).detach().cpu()

        # Step 4: 跨频段 refine，输出 2d
        fused = torch.cat([band_low, band_high], dim=1)
        if capture is not None:
            capture["cat_feat"] = fused.detach().cpu()

        out = self.refine(fused)

        if capture is not None:
            capture["faf_out"] = out.detach().cpu()
        return out

    def extra_repr(self) -> str:
        return (
            "stage_idx={}, d={}, strong_align={}, use_DWT={}, lp_kernel={}, refine_depth={}"
        ).format(
            self.stage_idx, self.d, self.strong_align, self.use_DWT,
            self.lp_kernel, self.refine_depth,
        )
