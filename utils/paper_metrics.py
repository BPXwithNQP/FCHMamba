"""HTC-DC-Net 论文（TGRS 2023）第四节 Evaluation Metrics：
- RMSE：全体像素
- RMSE-M：仅建筑像素
- RMSE-NM：仅非建筑像素
- RMSE-B：建筑实例内高度中位数与 GT 中位数的 RMSE

与 FusedHE/HTC-DC 增强版统一：若数据加载提供 building_mask（数据集 mask/*_BLG.tif），
则建筑/非建筑划分 **仅** 由 BLG 决定，不再用 GT nDSM 高度阈值。

无 BLG 时（如部分 Vaihingen 划分）：回退为 GT nDSM > building_min_m（默认 1 m，与旧版一致）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None


class PaperMetricAccumulator:
    """跨 batch 累积，与论文表格同一聚合方式。"""

    def __init__(self, building_min_m: float = 1.0):
        self.building_min_m = float(building_min_m)
        self._se_all = 0.0
        self._n_all = 0
        self._se_m = 0.0
        self._n_m = 0
        self._se_nm = 0.0
        self._n_nm = 0
        self._se_b = 0.0
        self._n_b = 0

    def _building_height_threshold(self) -> float:
        return self.building_min_m if self.building_min_m > 0 else 1.0

    def update_batch(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        building_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        if gt.ndim == 4:
            gt = gt.squeeze(1)
        assert pred.shape == gt.shape, (pred.shape, gt.shape)

        diff = pred - gt
        se = diff * diff
        self._se_all += float(se.sum().item())
        self._n_all += int(se.numel())

        if building_mask is not None:
            if building_mask.ndim == 4:
                building_mask = building_mask.squeeze(1)
            m_b = building_mask > 0.5
        else:
            m_b = gt > self._building_height_threshold()

        m_b = m_b & torch.isfinite(pred) & torch.isfinite(gt)
        m_nb = (~m_b) & torch.isfinite(pred) & torch.isfinite(gt)

        if m_b.any():
            self._se_m += float(se[m_b].sum().item())
            self._n_m += int(m_b.sum().item())
        if m_nb.any():
            self._se_nm += float(se[m_nb].sum().item())
            self._n_nm += int(m_nb.sum().item())

        # RMSE-B：逐样本连通域 + 中位数（CPU / numpy）
        pred_np = pred.detach().float().cpu().numpy()
        gt_np = gt.detach().float().cpu().numpy()
        mb_np = m_b.detach().cpu().numpy()

        for b in range(pred_np.shape[0]):
            s_b, n_inst = _buildingwise_sq_errors(
                pred_np[b],
                gt_np[b],
                mb_np[b],
            )
            self._se_b += s_b
            self._n_b += n_inst

    def finalize(self) -> dict:
        out = {}
        out["rmse_paper"] = (
            (self._se_all / self._n_all) ** 0.5 if self._n_all > 0 else float("nan")
        )
        out["rmse_m"] = (self._se_m / self._n_m) ** 0.5 if self._n_m > 0 else float("nan")
        out["rmse_nm"] = (self._se_nm / self._n_nm) ** 0.5 if self._n_nm > 0 else float("nan")
        out["rmse_b"] = (self._se_b / self._n_b) ** 0.5 if self._n_b > 0 else float("nan")
        out["n_pixels_all"] = self._n_all
        out["n_pixels_building"] = self._n_m
        out["n_pixels_nonbuilding"] = self._n_nm
        out["n_buildings"] = self._n_b
        return out


def _buildingwise_sq_errors(pred_hw: np.ndarray, gt_hw: np.ndarray, building_hw: np.ndarray):
    """单张图：足迹连通域内 pred/gt 中位数差的平方和、建筑个数。"""
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
