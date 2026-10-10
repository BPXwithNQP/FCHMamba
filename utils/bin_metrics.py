"""高度 bin 档位（条件头 / DPLD / TBWH / HL+DPLD）的 top-k / top-1 命中率，供 train / test 共用。"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from models.Blocks._config import load_block_cfg
except Exception:
    load_block_cfg = None


def squeeze_depth_bhw(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4 and t.size(1) == 1:
        return t.squeeze(1)
    return t


def top_k_setting_for_bucket_metrics(
    args: Any,
    n_bins: int,
    section_override: Optional[str] = None,
) -> int:
    """与 DPLDHead / TBWHHead 所用 top_k 一致：默认读 block.yaml；test 可由 section_override 覆盖。"""
    if section_override == "TBWH":
        section = "TBWH"
    elif section_override == "DPLD":
        section = "DPLD"
    else:
        section = None
        # 条件分区：与 ConditionalCascadeHeads.condition_k 对齐（固定档宽 / AdaBins 共用）
        if bool(getattr(args, "use_condition", False)):
            try:
                ck = int(getattr(args, "condition_k", 0) or 0)
            except (TypeError, ValueError):
                ck = 0
            if ck > 0:
                return max(1, min(ck, n_bins))
        dpld_raw = getattr(args, "use_DPLD", 0)
        if isinstance(dpld_raw, bool):
            di = 1 if dpld_raw else 0
        else:
            try:
                di = int(dpld_raw)
            except (TypeError, ValueError):
                di = 0
        if di not in (1, 2):
            di = 0
        if di in (1, 2):
            section = "DPLD"
        elif int(getattr(args, "use_TBWH", 0)):
            section = "TBWH"
    if section is None or load_block_cfg is None:
        return max(1, min(10, n_bins))
    cfg = load_block_cfg(section)
    if not isinstance(cfg, dict):
        return max(1, min(10, n_bins))
    k_default = 10 if section == "DPLD" else 5
    k = int(cfg.get("top_k", k_default))
    return max(1, min(k, n_bins))


def infer_bin_metrics_section_from_model(model: torch.nn.Module) -> Optional[str]:
    """从已构建的 FHE 推断读哪段 YAML 的 top_k（兜底：ckpt 无 args 时用）。"""
    if getattr(model, "tbwh_head", None) is not None:
        return "TBWH"
    if getattr(model, "dpld_head", None) is not None:
        return "DPLD"
    ht = getattr(model, "hl_tail", None)
    if ht is not None and getattr(ht, "dpld_head", None) is not None:
        return "DPLD"
    return None


def batch_bin_bucket_accuracy(
    preds: dict,
    depth_gt_bhw: torch.Tensor,
    args: Any,
    device: torch.device,
    section_override: Optional[str] = None,
) -> Tuple[float, float, int]:
    """GT bin 是否在预测 softmax 的 top-k 档内（与 DepthSegLoss 中 bucketize / 固定档划分一致）。"""
    depth_gt_bhw = squeeze_depth_bhw(depth_gt_bhw)
    bin_prob = preds.get("bin_prob")
    if bin_prob is None and "bin_logits" in preds:
        bin_prob = F.softmax(preds["bin_logits"], dim=1)
    if bin_prob is None:
        return float("nan"), float("nan"), 0

    n_bins = int(bin_prob.size(1))
    top_k = top_k_setting_for_bucket_metrics(args, n_bins, section_override=section_override)
    max_h = float(getattr(args, "max_height_eval", 185.0))
    min_norm = float(getattr(args, "min_height_eval", 1e-4)) / max(1.0, max_h)

    d = depth_gt_bhw.to(device)
    if d.dim() != 3:
        return float("nan"), float("nan"), 0
    h_gt_m = d * max_h
    valid = d.ge(min_norm)
    nv = int(valid.sum().item())
    if nv == 0:
        return float("nan"), float("nan"), 0

    # ----- GT 档位：DPLD/TBWH（bin_edges_m）或条件头（固定 bin_len / AdaBins）-----
    if "bin_edges_m" in preds:
        edges = preds["bin_edges_m"]
        if edges.dim() == 2:
            inner_edges = edges[0, 1:-1].to(device)
        else:
            inner_edges = edges[1:-1].to(device)
        bin_gt = torch.bucketize(
            h_gt_m.contiguous(),
            inner_edges,
            right=False,
        ).clamp(0, n_bins - 1)
    elif bool(getattr(args, "use_condition", False)) and "bin_logits" in preds:
        if bool(getattr(args, "use_adabin", False)) and "bin_edges" in preds:
            bin_edges = preds["bin_edges"].to(device)
            if bin_edges.dim() != 2:
                return float("nan"), float("nan"), 0
            be_lo = bin_edges[:, :-1, None, None]
            be_hi = bin_edges[:, 1:, None, None]
            h_gt_exp = h_gt_m.unsqueeze(1)
            bin_ind = (h_gt_exp > be_lo) & (h_gt_exp <= be_hi)
            bin_ind[:, 0] = bin_ind[:, 0] | (
                (h_gt_exp.squeeze(1) >= 0) & (h_gt_exp.squeeze(1) <= be_hi[:, 0])
            )
            bin_gt = bin_ind.float().argmax(dim=1).clamp(0, n_bins - 1)
        elif not bool(getattr(args, "use_adabin", False)):
            bin_len = float(getattr(args, "bin_len", 6.0))
            if bin_len <= 0:
                return float("nan"), float("nan"), 0
            # 与 criterion.DepthSegLoss 固定档分支一致：(h_gt / bin_len).long().clamp(...)
            bin_gt = (h_gt_m / bin_len).long().clamp(0, n_bins - 1)
        else:
            return float("nan"), float("nan"), 0
    else:
        return float("nan"), float("nan"), 0

    bp = bin_prob.to(device)
    _, top_idx = bp.topk(min(top_k, n_bins), dim=1)
    gt_exp = bin_gt.unsqueeze(1)
    hit_topk = (top_idx == gt_exp).any(dim=1)
    hit_top1 = bp.argmax(dim=1) == bin_gt

    topk_m = float((hit_topk & valid).float().sum().item() / max(nv, 1))
    top1_m = float((hit_top1 & valid).float().sum().item() / max(nv, 1))
    return topk_m, top1_m, nv


__all__ = [
    "squeeze_depth_bhw",
    "top_k_setting_for_bucket_metrics",
    "infer_bin_metrics_section_from_model",
    "batch_bin_bucket_accuracy",
]
