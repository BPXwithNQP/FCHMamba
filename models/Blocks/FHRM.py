"""FHRM — Frequency-aware Height Refinement Module.

在 DepthHead 输出的高度图上叠加三类特征域残差（低频一致性 / 高频边界 / warp 对齐），
并由分割概率 ``P_s`` 与由其得到的边界响应 ``P_e`` 进行空间加权。
设计说明见仓库内历史注释稿（FHRM 思路）；默认超参来自 ``block.yaml`` 的 ``FHRM`` 段。

注意：FHE 中 SegmentHead 末尾带 ReLU，配合 BCEWithLogitsLoss 训练；将其输出
``sigmoid`` 后作为 P_s 时取值范围仅 [0.5, 1)。本实现仍按注释稿 ``P_s = sigmoid(seg)``
处理，并将 ``seg_prob.detach()`` 以避免 depth_loss 倒灌干扰分割头；若希望 P_s 区分度
更好，可改用 ``--use_HL`` 提供的去 ReLU 分割头，或在 FHRM 内部启用 ``use_internal_gate``
让模块自带 1×1 分割导引头（见下方实现）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def _warp_feature_pixel_flow(x: torch.Tensor, flow_xy: torch.Tensor) -> torch.Tensor:
    """flow_xy: (B, 2, H, W)，分别为 dx、dy（像素位移，可加在采样网格上）。"""
    b, _, h, w = x.shape
    device, dtype = x.device, x.dtype
    ys = torch.linspace(0.0, float(h - 1), h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, float(w - 1), w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)
    sx = grid_x + flow_xy[:, 0]
    sy = grid_y + flow_xy[:, 1]
    gx = 2.0 * sx / float(max(w - 1, 1)) - 1.0
    gy = 2.0 * sy / float(max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class FHRMRefinement(nn.Module):
    def __init__(
        self,
        channels_in: int,
        lpf_kernel: Optional[int] = None,
        max_offset: Optional[float] = None,
        lambda_l_init: Optional[float] = None,
        lambda_h_init: Optional[float] = None,
        lambda_w_init: Optional[float] = None,
        learnable_lambdas: Optional[bool] = None,
        flow_mid_ratio: Optional[int] = None,
        residual_hidden_ratio: Optional[int] = None,
        clamp_pred: Optional[bool] = None,
        detach_seg: Optional[bool] = None,
        use_internal_gate: Optional[bool] = None,
        max_resid_h: Optional[float] = None,
        max_resid_w: Optional[float] = None,
        max_resid_l: Optional[float] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("FHRM")

        def _ig(key: str, override: Optional[int], cast=int, default=None):
            if override is not None:
                return cast(override)
            if default is not None:
                return cast(cfg.get(key, default))
            return cast(cfg[key])

        def _fg(key: str, override: Optional[float], default: float) -> float:
            if override is not None:
                return float(override)
            return float(cfg.get(key, default))

        def _bg(key: str, override: Optional[bool], default: bool) -> bool:
            if override is not None:
                return bool(override)
            return bool(cfg.get(key, default))

        self.lpf_kernel = _ig("lpf_kernel", lpf_kernel, int, 5)
        k = self.lpf_kernel
        if k < 1 or k % 2 == 0:
            raise ValueError("FHRM.lpf_kernel 必须为正奇数，收到 {}".format(k))

        self.max_offset = _fg("max_offset", max_offset, 1.0)
        self.clamp_pred = _bg("clamp_pred", clamp_pred, False)
        self.detach_seg = _bg("detach_seg", detach_seg, True)
        self.use_internal_gate = _bg("use_internal_gate", use_internal_gate, False)

        # 三路残差幅值上限（归一化高度空间）；通过 max_resid · tanh(conv) 限幅，避免高频/warp 爆冲。
        # 低频默认不限幅（low_branch 输出本身就稳，留无穷以兼容旧行为）；若 max_resid_l > 0 同样限幅。
        self.max_resid_l = _fg("max_resid_l", max_resid_l, 0.0)
        self.max_resid_h = _fg("max_resid_h", max_resid_h, 0.1)
        self.max_resid_w = _fg("max_resid_w", max_resid_w, 0.1)

        ll = _fg("lambda_l_init", lambda_l_init, 0.05)
        lh = _fg("lambda_h_init", lambda_h_init, 0.05)
        lw = _fg("lambda_w_init", lambda_w_init, 0.05)
        learn = _bg("learnable_lambdas", learnable_lambdas, True)

        self.lambda_l = nn.Parameter(torch.tensor(ll, dtype=torch.float32))
        self.lambda_h = nn.Parameter(torch.tensor(lh, dtype=torch.float32))
        self.lambda_w = nn.Parameter(torch.tensor(lw, dtype=torch.float32))
        if not learn:
            self.lambda_l.requires_grad_(False)
            self.lambda_h.requires_grad_(False)
            self.lambda_w.requires_grad_(False)

        flow_ratio = _ig("flow_mid_ratio", flow_mid_ratio, int, 4)
        hid_ratio = _ig("residual_hidden_ratio", residual_hidden_ratio, int, 2)

        c = int(channels_in)
        flow_hid = max(c // max(flow_ratio, 1), 32)
        res_hid = max(c // max(hid_ratio, 1), 16)

        self.low_branch = nn.Sequential(
            nn.Conv2d(c, res_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(res_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_hid, res_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(res_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_hid, 1, 1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(c, res_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(res_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_hid, res_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(res_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_hid, 1, 1),
        )
        flow_in = c * 2 + 1
        self.flow_head = nn.Sequential(
            nn.Conv2d(flow_in, flow_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(flow_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_hid, flow_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(flow_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_hid, 2, 1),
        )
        self.warp_branch = nn.Sequential(
            nn.Conv2d(c, res_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(res_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_hid, res_hid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(res_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_hid, 1, 1),
        )

        # 可选：FHRM 内部分割导引头（绕开外部 SegmentHead 末尾 ReLU 的怪癖）。
        # 仅作 P_s/P_e 用，不参与分割监督，因此区分度更好。
        if self.use_internal_gate:
            self.internal_seg = nn.Sequential(
                nn.Conv2d(c, res_hid, 3, 1, 1, bias=False),
                nn.BatchNorm2d(res_hid),
                nn.ReLU(inplace=True),
                nn.Conv2d(res_hid, 1, 1),
            )
        else:
            self.internal_seg = None

        sobel_x = torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _low_pass(self, feat: torch.Tensor) -> torch.Tensor:
        """AvgPool k×k，stride=1，保持空间尺寸。"""
        k = self.lpf_kernel
        pad = k // 2
        return F.avg_pool2d(feat, kernel_size=k, stride=1, padding=pad)

    def _edge_from_prob(self, p: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(p, self.sobel_x, padding=1)
        gy = F.conv2d(p, self.sobel_y, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        # sobel 在 p∈[0,1] 上的理论上界 ≈ sqrt(4^2+4^2) ≈ 5.66；
        # 用 tanh(α·mag) 做软归一化，既稳定又保留方向信息（α=2 比较温和）。
        return torch.tanh(2.0 * mag)

    def forward(
        self,
        feat: torch.Tensor,
        h_base: torch.Tensor,
        seg_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """feat: dec_final；h_base: DepthHead(feat)；seg_prob: SegmentHead 输出（logits，未 sigmoid）。"""
        if self.internal_seg is not None:
            p_s = torch.sigmoid(self.internal_seg(feat))
        elif seg_prob is None:
            p_s = torch.ones_like(h_base)
        else:
            if seg_prob.shape[-2:] != h_base.shape[-2:]:
                seg_prob = F.interpolate(
                    seg_prob, size=h_base.shape[-2:], mode="bilinear", align_corners=False
                )
            if self.detach_seg:
                seg_prob = seg_prob.detach()
            # FHE.SegmentHead 末尾是 ReLU(logits)，对其再 sigmoid 即为前景概率（范围受限 [0.5,1)）。
            p_s = torch.sigmoid(seg_prob)

        p_e = self._edge_from_prob(p_s)

        f_low = self._low_pass(feat)
        r_l = self.low_branch(f_low)
        if self.max_resid_l > 0:
            r_l = self.max_resid_l * torch.tanh(r_l)

        f_high = feat - self._low_pass(feat)
        r_h = self.high_branch(f_high)
        if self.max_resid_h > 0:
            r_h = self.max_resid_h * torch.tanh(r_h)

        f_dn = F.avg_pool2d(feat, kernel_size=2, stride=2)
        f_du = F.interpolate(f_dn, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        z = torch.cat([feat, f_du, p_s], dim=1)
        flow = torch.tanh(self.flow_head(z)) * self.max_offset
        f_w = _warp_feature_pixel_flow(feat, flow)
        r_w = self.warp_branch(feat - f_w)
        if self.max_resid_w > 0:
            r_w = self.max_resid_w * torch.tanh(r_w)

        # 三个 mask 都归一化到 [0,1]，让 λ 数值可直接横向对比。
        mask_l = 1.0 - p_e
        mask_h = p_e
        mask_w = 0.5 * (p_s + p_e)

        delta = (
            self.lambda_l * mask_l * r_l
            + self.lambda_h * mask_h * r_h
            + self.lambda_w * mask_w * r_w
        )
        out = h_base + delta
        if self.clamp_pred:
            out = out.clamp(0.0, 1.0)
        return out


__all__ = ["FHRMRefinement"]
