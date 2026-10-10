"""
TBWHHead —— Truncated Bin Weighted Height Head.

定位
----
相比 DPLD，TBWH 有意**去掉数据集先验**：
  - 不读 ndsm_stats.pickle，不做 quantile 划分，不缓存 bin edges。
  - 直接在 [0, max_height] 上做 **均匀等宽分区**，让模型自己学高度分布。

保留的机制
----------
1. **Ground bin**：`[0, ground_bin_max_m]` 独占 bin 0。
2. **Top-k softmax 加权**：对概率最高的 top-k 个 bin 重归一加权中心得米制高度。
3. **按序号最矮的前 k 档压低 norm（k 即 yaml ``top_k``）**：预测米制 \(h\le e[k]\)→`pred_d=0`；
   ``partition_stair=true``（且 \(k\le n\_bins{-}2\)）时下一高度档映射 **norm [0,0.5]**、更高档映射 **[0.5,1]**；
   ``partition_stair=false`` 时在 \(h>e[k]\) 上线性铺满 norm **[0,1]**。
4. **constraint（yaml）**：为 false 时训练**不**施加高斯 KL，仅 focal bin 等其它项保留。

三数据集独立配置（via block.yaml TBWH 段）
-----------------------------------------
| 数据集       | n_bins | max_height | 截断阈值（>99.9%）|
|-------------|--------|------------|-----------------|
| dfc2019_512 |  128   |  76 m      |  76 m           |
| dfc2023_512 |  128   |  81.5 m    |  81.5 m         |
| VHG512      |   64   |  20.5 m    |  20.5 m         |

调用方（FHE 或 trainer）根据 dataset_name 从 block.yaml 读取 `19_n_bins` /
`23_n_bins` / `vhg_n_bins` 和对应 `max_height`，显式传入 TBWHHead 构造函数；
其余超参（top_k / sigma_m / ground_bin_max_m / head_reduction 等）统一读 block.yaml。

接口
----
forward(x) ->
    pred_d     (B, 1, H, W)  分段映射到 [0,1]，非简单 h/max_height
    bin_logits (B, n_bins, H, W)
    bin_prob   (B, n_bins, H, W)

buffer（随 state_dict 持久化）：
    bin_edges_m   (n_bins+1,)  米制 bin 边界
    bin_centers_m (n_bins,)    米制 bin 中心
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


def meters_to_tbwh_pred_normalized(
    h_meters: torch.Tensor,
    edges: torch.Tensor,
    max_h: float,
    partition_k: int,
    stair: bool,
    n_bins: int,
) -> torch.Tensor:
    """预测米制高度 → [0,1] norm：最矮连续的 partition_k 档（h≤edges[k]）输出全为 0。

    stair=True（且 k≤n_bins-2）：下一档映射 norm [0,0.5]，更高档映射 [0.5,1]。
    stair=False：h≤edges[k]→0；（edges[k], max_h] 线性铺满 norm [0,1]。
    """
    if h_meters.dim() == 4:
        h_2 = h_meters.squeeze(1)
        squeeze_4 = True
    else:
        h_2 = h_meters
        squeeze_4 = False
    k = max(0, min(int(partition_k), max(n_bins - 1, 0)))
    edges = edges.to(dtype=h_2.dtype, device=h_2.device)
    emax = torch.as_tensor(float(max_h), device=h_2.device, dtype=h_2.dtype)
    floor_eps = torch.finfo(h_2.dtype).eps * 1024 + float(max_h) * 1e-8

    ek = edges[k]
    zeros = torch.zeros_like(h_2)
    zero_mask = h_2 <= ek

    if stair and k <= max(n_bins - 2, 0):
        ek1 = edges[k + 1]
        denom1 = (ek1 - ek).clamp(min=floor_eps)
        denom2 = (emax - ek1).clamp(min=floor_eps)
        mid_mask = (h_2 > ek) & (h_2 <= ek1)
        high_mask = h_2 > ek1
        tier2 = (0.5 * (h_2 - ek) / denom1).clamp(0.0, 1.0)
        tier3 = (0.5 + 0.5 * ((h_2 - ek1) / denom2)).clamp(0.0, 1.0)
        pred = torch.where(mid_mask, tier2, torch.where(high_mask, tier3, zeros))
        pred = torch.where(zero_mask, zeros, pred)
    else:
        denom_tail = (emax - ek).clamp(min=floor_eps)
        above_mask = h_2 > ek
        tier_tail = ((h_2 - ek) / denom_tail).clamp(0.0, 1.0)
        pred = torch.where(above_mask, tier_tail, zeros)

    pred = pred.clamp(0.0, 1.0)
    return pred.unsqueeze(1) if squeeze_4 else pred


# ============================================================================
#  均匀 bin edges 构建
# ============================================================================

def build_uniform_bin_edges(
    n_bins: int,
    max_height: float,
    ground_bin_max_m: float = 1.0,
) -> torch.Tensor:
    """构造均匀等宽 bin edges。

    结构：
      bin 0  → [0, ground_bin_max_m]
      bin 1…n_bins-1 → [ground_bin_max_m, max_height] 均匀划分（n_bins-1 个等宽 bin）

    Returns:
        edges: (n_bins+1,) float32，首项=0，末项=max_height
    """
    if n_bins < 2:
        raise ValueError("n_bins={} 必须 >= 2（bin 0 为 ground bin，至少还需 1 个高度 bin）".format(n_bins))
    if max_height <= ground_bin_max_m:
        raise ValueError(
            "max_height={} 必须 > ground_bin_max_m={}".format(max_height, ground_bin_max_m)
        )
    n_inner = n_bins - 1
    inner_edges = torch.linspace(
        float(ground_bin_max_m), float(max_height), n_inner + 1, dtype=torch.float32
    )
    edges = torch.cat([torch.tensor([0.0], dtype=torch.float32), inner_edges])
    return edges  # (n_bins+1,)


# ============================================================================
#  TBWHHead
# ============================================================================

class TBWHHead(nn.Module):
    """Truncated Bin Weighted Height Head。

    Args:
        channels_in:      输入特征通道数（与 SegmentHead / DepthHead 一致）
        max_height:       米制最大高度（数据集相关，必须显式传入）
        n_bins:           bin 总数（数据集相关，必须显式传入）
        ground_bin_max_m: ground bin 上界（米），None 时读 block.yaml（默认 1.0）
        top_k:            top-k 加权的 k，None 时读 block.yaml（默认 5）
        sigma_m:          高斯软标签 σ（米），None 时读 block.yaml（默认 1.5）
        head_reduction:   bin_head 中间层压缩比，None 时读 block.yaml（默认 2）
    """

    def __init__(
        self,
        channels_in: int,
        max_height: float,
        n_bins: int,
        ground_bin_max_m: Optional[float] = None,
        top_k: Optional[int] = None,
        sigma_m: Optional[float] = None,
        head_reduction: Optional[int] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("TBWH")

        self.n_bins = int(n_bins)
        self.max_height = float(max_height)
        self.channels_in = int(channels_in)
        self.top_k = int(cfg.get("top_k", 5) if top_k is None else top_k)
        self.sigma_m = float(cfg.get("sigma_m", 1.5) if sigma_m is None else sigma_m)
        self.ground_bin_max_m = float(
            cfg.get("ground_bin_max_m", 1.0) if ground_bin_max_m is None else ground_bin_max_m
        )
        head_reduction = int(
            cfg.get("head_reduction", 2) if head_reduction is None else head_reduction
        )
        self.partition_stair_norm = bool(cfg.get("partition_stair", False))
        effective_k = min(self.top_k, self.n_bins - 1)
        self.partition_idx = int(max(0, effective_k))

        if self.partition_stair_norm and self.partition_idx > max(self.n_bins - 2, 0):
            print(
                "[TBWH] partition_stair=true 需要 top_k≤n_bins-2，当前将退化为分段线性铺满（与 stair=false 等价）",
                flush=True,
            )
            self.partition_stair_norm = False

        if self.top_k > self.n_bins:
            raise ValueError("top_k={} 不能大于 n_bins={}".format(self.top_k, self.n_bins))

        # bin head：channels_in → hidden → n_bins
        hidden = max(self.channels_in // max(1, head_reduction), 32)
        self.bin_head = nn.Sequential(
            nn.Conv2d(self.channels_in, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.n_bins, kernel_size=1),
        )

        # 均匀 bin edges / centers 注册为 buffer，随 state_dict 持久化
        edges_t = build_uniform_bin_edges(self.n_bins, self.max_height, self.ground_bin_max_m)
        centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
        self.register_buffer("bin_edges_m", edges_t)
        self.register_buffer("bin_centers_m", centers_t)

        print(
            "[TBWH] n_bins={}, max_height={}m, ground_bin=[0,{}m], softmax_top_k={}, sigma_m={}m, "
            "partition_flat_idx={}, partition_stair={}, bin_width≈{:.4f}m (inner {} bins)".format(
                self.n_bins,
                self.max_height,
                self.ground_bin_max_m,
                self.top_k,
                self.sigma_m,
                self.partition_idx,
                self.partition_stair_norm,
                (self.max_height - self.ground_bin_max_m) / max(self.n_bins - 1, 1),
                self.n_bins - 1,
            ),
            flush=True,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, channels_in, H, W) decoder 最后一层输出
        Returns:
            pred_d:    (B, 1, H, W)  分段归一化高度 [0,1]（partition_stair / partition_idx 映射）
            bin_logits:(B, n_bins, H, W)
            bin_prob:  (B, n_bins, H, W)  softmax 概率
        """
        bin_logits = self.bin_head(x)
        bin_prob = F.softmax(bin_logits, dim=1)

        # Top-k：在 bin 维度取概率最大的 k 个，重新归一化后加权 bin center
        topk_p, topk_idx = bin_prob.topk(self.top_k, dim=1)          # (B, k, H, W)
        topk_p_sum = topk_p.sum(dim=1, keepdim=True).clamp(min=1e-8)
        topk_p_norm = topk_p / topk_p_sum                             # 重新归一化
        topk_centers = self.bin_centers_m[topk_idx]                   # (B, k, H, W)
        h_meters = (topk_p_norm * topk_centers).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        pred_d = meters_to_tbwh_pred_normalized(
            h_meters,
            self.bin_edges_m,
            self.max_height,
            self.partition_idx,
            self.partition_stair_norm,
            self.n_bins,
        )
        return pred_d, bin_logits, bin_prob

    # ------------------------------------------------------------------
    def gaussian_soft_label(self, h_gt_m: torch.Tensor) -> torch.Tensor:
        """在米制 bin center 上构造高斯软标签。

        Args:
            h_gt_m: (B, H, W) 或 (B, 1, H, W) 米制 GT 高度
        Returns:
            q: (B, n_bins, H, W) 软标签概率，已沿 bin 维归一化（sum=1）
        """
        if h_gt_m.dim() == 4 and h_gt_m.size(1) == 1:
            h_gt_m = h_gt_m.squeeze(1)
        c = self.bin_centers_m.view(1, -1, 1, 1)    # (1, n_bins, 1, 1)
        h = h_gt_m.unsqueeze(1)                      # (B, 1, H, W)
        sigma = max(float(self.sigma_m), 1e-3)
        log_q = -((c - h) ** 2) / (2.0 * sigma * sigma)
        q = F.softmax(log_q, dim=1)
        return q

    # ------------------------------------------------------------------
    def gt_bin_indices(self, h_gt_m: torch.Tensor) -> torch.Tensor:
        """根据 bin_edges_m 把米制 GT 映射到 bin index。

        Args:
            h_gt_m: (B, H, W) 米制 GT
        Returns:
            (B, H, W) long，取值 [0, n_bins-1]
        """
        if h_gt_m.dim() == 4 and h_gt_m.size(1) == 1:
            h_gt_m = h_gt_m.squeeze(1)
        # 用 (n_bins-1) 个内部边界做 bucketize
        inner_edges = self.bin_edges_m[1:-1]  # (n_bins-1,)
        idx = torch.bucketize(h_gt_m.contiguous(), inner_edges, right=False)
        return idx.clamp(0, self.n_bins - 1)


__all__ = ["TBWHHead", "build_uniform_bin_edges", "meters_to_tbwh_pred_normalized"]
