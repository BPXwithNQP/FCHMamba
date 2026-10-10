"""跨 FusedHE / HTC-DC 的统一验证与测试指标（米制深度）。

`UnifiedMetricAccumulator.finalize()` 返回的 dict 含：
  rmse, mae, rmse_m, rmse_nm, rmse_b, d1, d2, d3, b10, b0, m10, m0, iou, …

RMSE-M / RMSE-NM：有 building_mask（BLG）时仅由 mask 划分；无 mask 时用 GT 高度 >
`mask_depth_threshold`（默认 1 m）；若 `paper_building_min_m > 0` 则优先该阈值。

B10（MAFENet）定义：在**建筑像素**上，若 max(P/G, G/P) < `MAFENET_B10_MAX_RATIO`（默认 1.1）则计为
「好」；b10 为这类像素数 / 建筑像素总数。优先使用 `b10_building_mask`（数据集 mask/*_BLG.tif），
与 `building_mask`（可用于 RMSE-M 等）解耦；未提供时回退到 `building_mask`。

B0：与 B10 同集合（同分母：建筑像素），阈值为 1.0，即 max(P/G, G/P) < 1.0 的占比。

M10 / M0：与 B10 / B0 算法相同，但分母为**高度主体像素**（GT 高度 > `mask_depth_threshold`，默认 1 m），
不使用 building_mask / BLG。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# 论文/MAFENet：建筑深度比值阈值；亦用于 δ 指标与 B10 / B0
MAFENET_B10_MAX_RATIO = 1.1
MAFENET_B0_MAX_RATIO = 1.0

try:
    import cv2
except ImportError:
    cv2 = None


def _buildingwise_sq_errors(pred_hw: np.ndarray, gt_hw: np.ndarray, building_hw: np.ndarray):
    if cv2 is None:
        return 0.0, 0

    mask = (building_hw.astype(np.uint8) > 0).astype(np.uint8)
    if mask.sum() == 0:
        return 0.0, 0

    num_labels, labels = cv2.connectedComponents(mask)
    se_sum = 0.0
    n_inst = 0
    for lid in range(1, num_labels):
        sel = labels == lid
        gv = gt_hw[sel]
        pv = pred_hw[sel]
        if gv.size == 0:
            continue
        mg = float(np.median(gv))
        mp = float(np.median(pv))
        se_sum += (mp - mg) ** 2
        n_inst += 1

    return se_sum, n_inst


def _infer_ratio_floor(pred: torch.Tensor, target: torch.Tensor) -> float:
    m = float(torch.maximum(pred.max(), target.max()).detach().cpu())
    if m > 1.5:
        return 0.01
    return 1e-4


class UnifiedMetricAccumulator:
    """全数据集聚合：论文式 RMSE / RMSE-M/NM/B、δ1–3、IoU、MAE；与 utils.metrics.eval_depth 口径对齐。"""

    def __init__(
        self,
        mask_depth_threshold: float = 1.0,
        paper_building_min_m: float = 0.0,
        iou_gt_from_height_threshold: bool = False,
    ):
        self.mask_depth_threshold = float(mask_depth_threshold)
        self.paper_building_min_m = float(paper_building_min_m)
        # True：IoU 的 GT 侧固定为 gt > mask_depth_threshold，不用 BLG building_mask
        self.iou_gt_from_height_threshold = bool(iou_gt_from_height_threshold)

        self._se_all = 0.0
        self._n_all = 0
        self._ae_all = 0.0
        self._se_m = 0.0
        self._n_m = 0
        self._se_nm = 0.0
        self._n_nm = 0
        self._se_b = 0.0
        self._n_b = 0

        self._d1_n = 0
        self._d2_n = 0
        self._d3_n = 0
        self._d_d = 0

        # B10 / B0（MAFENet 风格）：建筑像素上 max(P/G, G/P) 与阈值 1.1 / 1.0 比较；同分母
        self._b10_n = 0
        self._b0_n = 0
        self._b10_d = 0

        # M10 / M0：高度主体像素（GT > mask_depth_threshold）上同 B10 / B0 口径
        self._m10_n = 0
        self._m0_n = 0
        self._m10_d = 0

        self._iou_inter = 0.0
        self._iou_union = 0.0

    def _building_height_threshold(self) -> float:
        """无 BLG 时：GT 高度 > 阈值视为建筑。默认与 mask_depth_threshold 一致（1 m）。"""
        if self.paper_building_min_m > 0:
            return self.paper_building_min_m
        return self.mask_depth_threshold

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        building_mask: Optional[torch.Tensor] = None,
        b10_building_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if pred.ndim == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if gt.ndim == 4 and gt.size(1) == 1:
            gt = gt.squeeze(1)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)

        diff = pred - gt
        se = diff * diff
        finite = torch.isfinite(pred) & torch.isfinite(gt)
        if finite.any():
            self._se_all += float(se[finite].sum().item())
            self._n_all += int(finite.sum().item())
            self._ae_all += float(torch.abs(diff[finite]).sum().item())

        if building_mask is not None:
            bm = building_mask
            if bm.ndim == 4 and bm.size(1) == 1:
                bm = bm.squeeze(1)
            m_b = bm > 0.5
        else:
            m_b = gt > self._building_height_threshold()

        m_b = m_b & finite
        m_nb = (~m_b) & finite

        if m_b.any():
            self._se_m += float(se[m_b].sum().item())
            self._n_m += int(m_b.sum().item())
        if m_nb.any():
            self._se_nm += float(se[m_nb].sum().item())
            self._n_nm += int(m_nb.sum().item())

        pred_np = pred.detach().float().cpu().numpy()
        gt_np = gt.detach().float().cpu().numpy()
        mb_np = m_b.detach().cpu().numpy()

        for b in range(pred_np.shape[0]):
            s_b, n_inst = _buildingwise_sq_errors(pred_np[b], gt_np[b], mb_np[b])
            self._se_b += s_b
            self._n_b += n_inst

        eps_r = _infer_ratio_floor(pred, gt)
        pred_r = torch.clamp(pred, min=eps_r)
        tgt_r = torch.clamp(gt, min=eps_r)
        ratio_all = torch.max((pred_r / tgt_r), (tgt_r / pred_r))
        thresh = ratio_all[finite]
        if thresh.numel() > 0:
            self._d1_n += int((thresh < 1.25).sum().item())
            self._d2_n += int((thresh < 1.25**2).sum().item())
            self._d3_n += int((thresh < 1.25**3).sum().item())
            self._d_d += int(thresh.numel())

        # B10 / B0：建筑像素（优先 b10_building_mask / BLG，与 RMSE-M 用的 building_mask 可不同）
        if b10_building_mask is not None:
            bm10 = b10_building_mask
            if bm10.ndim == 4 and bm10.size(1) == 1:
                bm10 = bm10.squeeze(1)
            m_b10 = (bm10 > 0.5) & finite
        else:
            m_b10 = m_b
        if m_b10.any():
            ratio_b = ratio_all[m_b10]
            if ratio_b.numel() > 0:
                self._b10_n += int((ratio_b < MAFENET_B10_MAX_RATIO).sum().item())
                self._b0_n += int((ratio_b < MAFENET_B0_MAX_RATIO).sum().item())
                self._b10_d += int(ratio_b.numel())

        # M10 / M0：高度主体像素（GT > mask_depth_threshold），不用 building_mask
        m_h = (gt > self.mask_depth_threshold) & finite
        if m_h.any():
            ratio_h = ratio_all[m_h]
            if ratio_h.numel() > 0:
                self._m10_n += int((ratio_h < MAFENET_B10_MAX_RATIO).sum().item())
                self._m0_n += int((ratio_h < MAFENET_B0_MAX_RATIO).sum().item())
                self._m10_d += int(ratio_h.numel())

        if self.iou_gt_from_height_threshold:
            seg = (gt > self.mask_depth_threshold) & finite
        elif building_mask is not None:
            bm = building_mask
            if bm.ndim == 4 and bm.size(1) == 1:
                bm = bm.squeeze(1)
            seg = (bm > 0.5) & finite
        else:
            seg = (gt > self.mask_depth_threshold) & finite

        binary = (pred > self.mask_depth_threshold) & finite
        inter = binary & seg
        union = binary | seg
        self._iou_inter += float(inter.sum().item())
        self._iou_union += float(union.sum().item())

    def finalize(self) -> dict:
        rmse = (
            (self._se_all / self._n_all) ** 0.5 if self._n_all > 0 else float("nan")
        )
        mae = self._ae_all / self._n_all if self._n_all > 0 else float("nan")
        rmse_m = (self._se_m / self._n_m) ** 0.5 if self._n_m > 0 else float("nan")
        rmse_nm = (self._se_nm / self._n_nm) ** 0.5 if self._n_nm > 0 else float("nan")
        rmse_b = (self._se_b / self._n_b) ** 0.5 if self._n_b > 0 else float("nan")

        if self._d_d > 0:
            d1 = self._d1_n / self._d_d
            d2 = self._d2_n / self._d_d
            d3 = self._d3_n / self._d_d
        else:
            d1 = d2 = d3 = float("nan")

        b10 = self._b10_n / self._b10_d if self._b10_d > 0 else float("nan")
        b0 = self._b0_n / self._b10_d if self._b10_d > 0 else float("nan")
        m10 = self._m10_n / self._m10_d if self._m10_d > 0 else float("nan")
        m0 = self._m0_n / self._m10_d if self._m10_d > 0 else float("nan")

        iou = (
            self._iou_inter / (self._iou_union + 1.0)
            if self._iou_union >= 0
            else float("nan")
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "rmse_m": rmse_m,
            "rmse_nm": rmse_nm,
            "rmse_b": rmse_b,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "b10": b10,
            "b0": b0,
            "m10": m10,
            "m0": m0,
            "iou": iou,
            "n_pixels_all": self._n_all,
            "n_pixels_building": self._n_m,
            "n_pixels_nonbuilding": self._n_nm,
            "n_pixels_height_subject": self._m10_d,
            "n_buildings": self._n_b,
        }
