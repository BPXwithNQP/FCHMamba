"""
GRFA —— Gated Residual Frequency Augmentation.

Drop-in 替换 FeatureFusion：
    out = base + freq_lambda * gamma * gate * edge_residual
    freq_lambda 默认 1.0；测试时可临时放大频率残差（不改 gamma 权重）。

base 分支保持与原 FeatureFusion 权重同名、同形状，便于从基线 checkpoint
热启动；新增高频残差分支和逐通道 gamma 默认零初始化，初始输出严格等价于 base。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def _conv_bn_relu(in_c: int, out_c: int, kernel: int) -> nn.Module:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _conv_bn(in_c: int, out_c: int, kernel: int) -> nn.Module:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad, bias=True),
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
    """Depthwise 3x3 + BN + GELU + Pointwise 1x1 + BN，d -> 2d。"""

    def __init__(self, d: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False),
            nn.BatchNorm2d(d),
            nn.GELU(),
            nn.Conv2d(d, 2 * d, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class GRFA(nn.Module):
    """Gated Residual Frequency Augmentation（drop-in 替换 FeatureFusion / FAF 系列）。"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stage_idx: int,
        mit_in_channel: Optional[int] = None,
        lp_kernel: Optional[int] = None,
        shallow_lp_kernel: Optional[int] = None,
        deep_lp_kernel: Optional[int] = None,
        shallow_stages: Optional[list] = None,
        use_haar: Optional[bool] = None,
        gate_reduction: Optional[int] = None,
        gate_kernel: Optional[int] = None,
        gamma_init: Optional[float] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("GRFA")

        d = out_channel
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
        use_haar = bool(int(cfg.get("use_haar", 0)) if use_haar is None else use_haar)
        gamma_init = float(cfg.get("gamma_init", 0.0) if gamma_init is None else gamma_init)

        if lp_kernel % 2 == 0:
            raise ValueError("lp_kernel 必须是奇数，收到: {}".format(lp_kernel))
        if gate_kernel % 2 == 0:
            raise ValueError("gate_kernel 必须是奇数，收到: {}".format(gate_kernel))

        # base：与 FeatureFusion 同名同形状；conv2 末尾 ReLU 拆到 forward，便于 pre-act 分频。
        self.conv1 = _conv_bn_relu(in_channel, in_channel // 2, kernel=3)
        self.conv2 = _conv_bn(in_channel // 2, d, kernel=3)
        self.conv3 = _conv_bn_relu(2 * d, 2 * d, kernel=3)
        self.conv4 = _conv_bn_relu(2 * d, 2 * d, kernel=3)

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
            nn.Conv2d(4 * d, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=gate_kernel, padding=gate_kernel // 2, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1, 2 * d, 1, 1), gamma_init))
        self.freq_lambda = 1.0
        self.last_edge_gate: Optional[torch.Tensor] = None

        print(
            "[GRFA] stage{}: split={} lp_kernel={} gate_red={} gate_k={} gamma_init={:.4f}".format(
                self.stage_idx,
                "haar(LH+HL+HH)" if use_haar else "avgpool-preact",
                lp_kernel,
                gate_reduction,
                gate_kernel,
                gamma_init,
            ),
            flush=True,
        )

    def _high_frequency(self, c_pre: torch.Tensor) -> torch.Tensor:
        if self.use_haar:
            low, bands = self.haar(c_pre)
            return self.band_mix(torch.cat(list(bands), dim=1))
        low = F.avg_pool2d(c_pre, kernel_size=self.lp_kernel, stride=1, padding=self.lp_pad)
        return c_pre - low

    def forward(
        self,
        x_mit: torch.Tensor,
        x_cnn: torch.Tensor,
        capture: Optional[dict] = None,
    ) -> torch.Tensor:
        c_pre = self.conv2(self.conv1(x_cnn))
        c = F.relu(c_pre, inplace=False)
        base = self.conv4(self.conv3(torch.cat((x_mit, c), dim=1)))

        high = self._high_frequency(c_pre)
        edge_residual = self.band_refine(high)
        edge_gate = self.gate(torch.cat((base, edge_residual), dim=1))
        lam = float(self.freq_lambda)
        out = base + lam * self.gamma * (edge_gate * edge_residual)
        self.last_edge_gate = edge_gate

        if capture is not None:
            capture["x_mit"] = x_mit.detach().cpu()
            capture["x_cnn"] = x_cnn.detach().cpu()
            capture["base"] = base.detach().cpu()
            capture["c_high"] = high.detach().cpu()
            capture["edge_residual"] = edge_residual.detach().cpu()
            capture["edge_gate"] = edge_gate.detach().cpu()
            capture["gamma"] = self.gamma.detach().cpu()
            capture["freq_lambda"] = lam
            capture["faf_out"] = out.detach().cpu()
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
    """推理时临时设置所有 GRFA 的频率残差放大系数 λ（out = base + λ * gamma * M * e）。"""
    lam = float(lam)
    if isinstance(module_or_model, GRFA):
        module_or_model.freq_lambda = lam
        return
    for m in module_or_model.modules():
        if isinstance(m, GRFA):
            m.freq_lambda = lam
