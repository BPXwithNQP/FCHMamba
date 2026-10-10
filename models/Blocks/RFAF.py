"""
RFAF —— Residual Frequency-Aware Fusion。

out = base + α · freq
- base：原版 FeatureFusion 主融合（不动）
- freq：阶段化 DWT 频域残差；stage1/2 从对齐后 CNN 支补高频，stage3/4 从 MiT 支补低频
- α 每 stage 独立可学习，默认 0.01

接口与 FeatureFusion / nFAF / liteFAF 一致；细粒度参数读 block.yaml 的 RFAF 段。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def _conv_bn(in_c: int, out_c: int, kernel: int, act: bool = False) -> nn.Module:
    pad = kernel // 2
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=pad, bias=False),
        nn.BatchNorm2d(out_c),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _HaarDWT(nn.Module):
    """Haar DWT（stride=2），输出 LL/LH/HL/HH，各 (B,C,H/2,W/2)。"""

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
        if w % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode="reflect")
        xa = x[:, :, 0::2, 0::2]
        xb = x[:, :, 0::2, 1::2]
        xc = x[:, :, 1::2, 0::2]
        xd = x[:, :, 1::2, 1::2]
        ll = (xa + xb + xc + xd) * 0.25
        lh = (xa + xb - xc - xd) * 0.25
        hl = (xa - xb + xc - xd) * 0.25
        hh = (xa - xb - xc + xd) * 0.25
        return ll, lh, hl, hh


class _HaarIDWT(nn.Module):
    """Haar IDWT，由四子带重建 (B,C,H,W)。"""

    def forward(self, ll, lh, hl, hh):
        rec = torch.zeros(
            ll.size(0), ll.size(1), ll.size(2) * 2, ll.size(3) * 2,
            device=ll.device, dtype=ll.dtype,
        )
        rec[:, :, 0::2, 0::2] = ll + lh + hl + hh
        rec[:, :, 0::2, 1::2] = ll + lh - hl - hh
        rec[:, :, 1::2, 0::2] = ll - lh + hl - hh
        rec[:, :, 1::2, 1::2] = ll - lh - hl + hh
        return rec


class _BandRefine(nn.Module):
    """Depthwise 3×3 + BN + GELU + Pointwise 1×1 + BN（无最终激活）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RFAF(nn.Module):
    """Residual Frequency-Aware Fusion（drop-in 替换 FeatureFusion / FAFBlock / nFAF / liteFAF）。"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stage_idx: int,
        mit_in_channel: Optional[int] = None,
        high_reduce_channels: Optional[int] = None,
        low_reduce_channels: Optional[int] = None,
        high_freq_stages: Optional[list] = None,
        alpha_init: Optional[float] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("RFAF")

        if mit_in_channel is None:
            mit_in_channel = out_channel

        high_reduce_channels = int(
            cfg.get("high_reduce_channels", 32) if high_reduce_channels is None else high_reduce_channels
        )
        low_reduce_channels = int(
            cfg.get("low_reduce_channels", 64) if low_reduce_channels is None else low_reduce_channels
        )
        if high_freq_stages is None:
            high_freq_stages = cfg.get("high_freq_stages", [1, 2])
        high_freq_stages = [int(s) for s in (high_freq_stages or [])]
        alpha_init = float(cfg.get("alpha_init", 0.01) if alpha_init is None else alpha_init)

        d = out_channel
        self.d = d
        self.stage_idx = stage_idx
        self.mit_in_channel = mit_in_channel
        self.is_high_stage = stage_idx in high_freq_stages
        self.high_reduce_channels = high_reduce_channels
        self.low_reduce_channels = low_reduce_channels

        # 原版 FeatureFusion 主融合
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel // 2, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(d * 2, d * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d * 2),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(d * 2, d * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d * 2),
            nn.ReLU(inplace=True),
        )

        self.dwt = _HaarDWT()
        self.idwt = _HaarIDWT()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        if self.is_high_stage:
            c_red = high_reduce_channels
            self.reduce = _conv_bn(d, c_red, kernel=1, act=False)
            self.refine = _BandRefine(3 * c_red)
            self.expand = _conv_bn(c_red, 2 * d, kernel=1, act=False)
        else:
            c_red = low_reduce_channels
            self.reduce = _conv_bn(mit_in_channel, c_red, kernel=1, act=False)
            self.refine = _BandRefine(c_red)
            self.expand = _conv_bn(c_red, 2 * d, kernel=1, act=False)

        branch = "high(CNN)" if self.is_high_stage else "low(MiT)"
        print(
            "[RFAF] stage{}: branch={} reduce={} alpha_init={:.4f}".format(
                stage_idx, branch, c_red, alpha_init,
            ),
            flush=True,
        )

    def _base_fusion(self, x_mit: torch.Tensor, x_cnn: torch.Tensor):
        x_res_align = self.conv2(self.conv1(x_cnn))
        base = self.conv4(self.conv3(torch.cat((x_mit, x_res_align), dim=1)))
        return base, x_res_align

    def _high_freq_branch(self, x_res_align: torch.Tensor) -> torch.Tensor:
        c_red = self.high_reduce_channels
        z = self.reduce(x_res_align)
        ll, lh, hl, hh = self.dwt(z)
        h_band = torch.cat([lh, hl, hh], dim=1)
        h_refined = self.refine(h_band)
        lh_r, hl_r, hh_r = torch.split(h_refined, c_red, dim=1)
        ll_zero = torch.zeros_like(ll)
        f_small = self.idwt(ll_zero, lh_r, hl_r, hh_r)
        return self.expand(f_small)

    def _low_freq_branch(self, x_mit: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x_mit)
        ll, lh, hl, hh = self.dwt(z)
        ll_r = self.refine(ll)
        zero = torch.zeros_like(lh)
        f_small = self.idwt(ll_r, zero, zero, zero)
        return self.expand(f_small)

    def forward(
        self,
        x_mit: torch.Tensor,
        x_cnn: torch.Tensor,
        capture: Optional[dict] = None,
    ) -> torch.Tensor:
        base, x_res_align = self._base_fusion(x_mit, x_cnn)

        if self.is_high_stage:
            freq = self._high_freq_branch(x_res_align)
        else:
            freq = self._low_freq_branch(x_mit)

        out = base + self.alpha * freq

        if capture is not None:
            capture["x_mit"] = x_mit.detach().cpu()
            capture["x_cnn"] = x_cnn.detach().cpu()
            capture["base"] = base.detach().cpu()
            capture["freq"] = freq.detach().cpu()
            capture["alpha"] = float(self.alpha.detach().cpu().item())
            capture["faf_out"] = out.detach().cpu()
        return out

    def extra_repr(self) -> str:
        branch = "high" if self.is_high_stage else "low"
        return "stage_idx={}, d={}, branch={}, alpha={:.4f}".format(
            self.stage_idx, self.d, branch, float(self.alpha.detach().cpu().item()),
        )
