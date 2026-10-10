"""
RGHF —— Residual Gated High-Frequency（单主干编码器用）。

与 GRFA 的区别：
- GRFA 面向双主干融合（x_mit + x_cnn → 2d），RGHF 面向单主干 stage 输出（x → d）。
- 在每个 GroupMamba stage 结束后，从 pre-act 特征提取高频残差，经门控与逐通道 gamma 回补：

    out = x + freq_lambda * gamma * gate * edge_residual

gamma 默认零初始化：初始 out == x，便于从预训练 GroupMamba checkpoint 热启动。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def _conv_bn(in_c: int, out_c: int, kernel: int) -> nn.Module:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
    )


class _HaarSplit(nn.Module):
    """无下采样 Haar 分解，返回 LL 与三个方向高频带。"""

    def __init__(self):
        super().__init__()
        ll = [[0.25, 0.25], [0.25, 0.25]]
        lh = [[0.25, -0.25], [0.25, -0.25]]
        hl = [[0.25, 0.25], [-0.25, -0.25]]
        hh = [[0.25, -0.25], [-0.25, 0.25]]
        weight = torch.tensor([ll, lh, hl, hh], dtype=torch.float32).unsqueeze(1)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        xp = F.pad(x, (0, 1, 0, 1), mode="replicate")
        weight = self.weight.repeat(c, 1, 1, 1)
        y = F.conv2d(xp, weight, groups=c)
        y = y.view(b, c, 4, h, w)
        return y[:, :, 0], (y[:, :, 1], y[:, :, 2], y[:, :, 3])


class _BandRefine(nn.Module):
    """Depthwise 3x3 + BN + GELU + Pointwise 1x1 + BN，d -> d。"""

    def __init__(self, d: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RGHF(nn.Module):
    """单主干 stage 级门控高频残差增强（GroupMamba 等 NCHW 特征图）。"""

    def __init__(
        self,
        channels: int,
        stage_idx: int,
        lp_kernel: Optional[int] = None,
        shallow_lp_kernel: Optional[int] = None,
        deep_lp_kernel: Optional[int] = None,
        shallow_stages: Optional[list] = None,
        use_haar: Optional[bool] = None,
        gate_reduction: Optional[int] = None,
        gate_kernel: Optional[int] = None,
        gamma_init: Optional[float] = None,
        use_pre_conv: Optional[bool] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("RGHF")

        d = int(channels)
        self.d = d
        self.stage_idx = int(stage_idx)

        if shallow_stages is None:
            shallow_stages = cfg.get("shallow_stages", [1, 2])
        shallow_stages = [int(s) for s in (shallow_stages or [])]
        shallow_lp_kernel = int(cfg.get("shallow_lp_kernel", 5) if shallow_lp_kernel is None else shallow_lp_kernel)
        deep_lp_kernel = int(cfg.get("deep_lp_kernel", 3) if deep_lp_kernel is None else deep_lp_kernel)
        if lp_kernel is None:
            lp_kernel = shallow_lp_kernel if self.stage_idx in shallow_stages else deep_lp_kernel
        lp_kernel = int(lp_kernel)
        gate_reduction = int(cfg.get("gate_reduction", 4) if gate_reduction is None else gate_reduction)
        gate_kernel = int(cfg.get("gate_kernel", 3) if gate_kernel is None else gate_kernel)
        use_haar = bool(int(cfg.get("use_haar", 1)) if use_haar is None else use_haar)
        gamma_init = float(cfg.get("gamma_init", 0.0) if gamma_init is None else gamma_init)
        use_pre_conv = bool(int(cfg.get("use_pre_conv", 1)) if use_pre_conv is None else use_pre_conv)

        if lp_kernel % 2 == 0:
            raise ValueError("lp_kernel 必须是奇数，收到: {}".format(lp_kernel))
        if gate_kernel % 2 == 0:
            raise ValueError("gate_kernel 必须是奇数，收到: {}".format(gate_kernel))

        self.use_pre_conv = use_pre_conv
        if use_pre_conv:
            self.pre = _conv_bn(d, d, kernel=3)
        else:
            self.pre = nn.Identity()

        self.lp_kernel = lp_kernel
        self.lp_pad = lp_kernel // 2
        self.use_haar = use_haar
        if use_haar:
            self.haar = _HaarSplit()
            self.band_mix = nn.Conv2d(3 * d, d, kernel_size=1, bias=False)
            with torch.no_grad():
                weight = torch.zeros(d, 3 * d, 1, 1)
                for j in range(d):
                    weight[j, j, 0, 0] = 1.0
                    weight[j, d + j, 0, 0] = 1.0
                    weight[j, 2 * d + j, 0, 0] = 1.0
                self.band_mix.weight.copy_(weight)
        else:
            self.haar = None
            self.band_mix = None

        self.band_refine = _BandRefine(d)
        hidden = max(d // max(1, gate_reduction), 16)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * d, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=gate_kernel, padding=gate_kernel // 2, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1, d, 1, 1), gamma_init))
        self.freq_lambda = 1.0
        self.last_edge_gate: Optional[torch.Tensor] = None

        print(
            "[RGHF] stage{}: d={} split={} lp_kernel={} pre_conv={} gate_red={} gate_k={} gamma_init={:.4f}".format(
                self.stage_idx,
                d,
                "haar(LH+HL+HH)" if use_haar else "avgpool-preact",
                lp_kernel,
                use_pre_conv,
                gate_reduction,
                gate_kernel,
                gamma_init,
            ),
            flush=True,
        )

    def _high_frequency(self, x_pre: torch.Tensor) -> torch.Tensor:
        if self.use_haar:
            low, bands = self.haar(x_pre)
            return self.band_mix(torch.cat(list(bands), dim=1))
        low = F.avg_pool2d(x_pre, kernel_size=self.lp_kernel, stride=1, padding=self.lp_pad)
        return x_pre - low

    def forward(
        self,
        x: torch.Tensor,
        capture: Optional[dict] = None,
    ) -> torch.Tensor:
        base = x
        x_pre = self.pre(x) if self.use_pre_conv else x

        high = self._high_frequency(x_pre)
        edge_residual = self.band_refine(high)
        edge_gate = self.gate(torch.cat((base, edge_residual), dim=1))
        lam = float(self.freq_lambda)
        out = base + lam * self.gamma * (edge_gate * edge_residual)
        self.last_edge_gate = edge_gate

        if capture is not None:
            capture["x_in"] = x.detach().cpu()
            capture["base"] = base.detach().cpu()
            capture["x_pre"] = x_pre.detach().cpu()
            capture["c_high"] = high.detach().cpu()
            capture["edge_residual"] = edge_residual.detach().cpu()
            capture["edge_gate"] = edge_gate.detach().cpu()
            capture["gamma"] = self.gamma.detach().cpu()
            capture["freq_lambda"] = lam
            capture["rghf_out"] = out.detach().cpu()
        return out

    def extra_repr(self) -> str:
        return "stage_idx={}, d={}, use_haar={}, lp_kernel={}, freq_lambda={:.4f}, gamma_mean={:.4f}".format(
            self.stage_idx,
            self.d,
            self.use_haar,
            self.lp_kernel,
            float(self.freq_lambda),
            float(self.gamma.detach().mean().cpu().item()),
        )


def set_freq_lambda(module_or_model, lam: float) -> None:
    """推理时临时设置所有 RGHF 的频率残差放大系数 λ（out = x + λ * gamma * M * e）。"""
    lam = float(lam)
    if isinstance(module_or_model, RGHF):
        module_or_model.freq_lambda = lam
        return
    for m in module_or_model.modules():
        if isinstance(m, RGHF):
            m.freq_lambda = lam
