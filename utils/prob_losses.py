"""
HTC 风格的概率分布约束损失：约束 bin 概率呈高斯/拉普拉斯等分布，对 GT 所在 bin 附近做平滑矫正。
与 HTC-DC-Net parts.losses 对齐，供 FusedHE 在 distribution_constraint 时使用。
"""
import math
import torch
import torch.nn.functional as F


def get_gaussian_cdf(std, mean, lower_edge, upper_edge, mode_prob):
    """高斯 CDF 在 bin 区间上的增量。"""
    cdf = 0.5 * (
        torch.erf((upper_edge - mean) / std / math.sqrt(2))
        - torch.erf((lower_edge - mean) / std / math.sqrt(2))
    ) - mode_prob
    return cdf


def gaussian_loss(probs, bin_edges, gt, masked=False, reduction=True):
    """
    正态分布矫正：根据 GT 所在 bin 与当前预测的 mode_prob，反推理想高斯 CDF 在各 bin 上的概率，
    令预测的 bin 概率分布拟合该 CDF（KL 散度），使概率在 GT 所在 bin 附近呈高斯平滑。
    probs: (B, n_bins, H, W), bin_edges: (B, n_bins+1), gt: (B, 1, H, W) 或 (B, H, W)，归一化 [0,1]
    """
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    with torch.no_grad():
        # bin_edges: (B, n_bins+1) -> (B, n_bins, 1, 1) 与 (B, 1, H, W) 广播
        be_lo = bin_edges[:, :-1, None, None]  # (B, n_bins, 1, 1)
        be_hi = bin_edges[:, 1:, None, None]
        bin_ind = (gt > be_lo) & (gt <= be_hi)
        mode_prob = (probs * bin_ind.float()).sum(dim=1, keepdim=True)
        bin_lower_edge = (bin_ind.float() * be_lo).sum(dim=1, keepdim=True)
        bin_upper_edge = (bin_ind.float() * be_hi).sum(dim=1, keepdim=True)
        # 避免除零：mode_prob 和 (bin_upper - bin_lower) 可能为 0
        safe_mode = mode_prob.clamp(min=1e-3, max=1 - 1e-3)
        width = (bin_upper_edge - bin_lower_edge).clamp(min=1e-6)
        # erfinv(safe_mode) 可能产生 inf，需 clamp
        denom = torch.erfinv(safe_mode).clamp(min=1e-3, max=5.0)
        std = width / 2.0 / math.sqrt(2) / denom
        std = std.clamp(min=1e-6)
        cdf_probs = get_gaussian_cdf(std, gt, be_lo, be_hi, 0)
        cdf_probs = torch.clamp(cdf_probs, 1e-3, 1 - 1e-3)
        cdf_probs = F.normalize(cdf_probs, p=1, dim=1)
    probs_masks = (probs > 0).float() if masked else torch.ones_like(probs)
    probs = torch.clamp(probs, 1e-3, 1 - 1e-3)
    probs = F.normalize(probs, p=1, dim=1)
    if reduction:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction="none") * probs_masks).sum(dim=1, keepdim=True).mean()
    else:
        loss = (F.kl_div(torch.log(probs), cdf_probs, reduction="none") * probs_masks).sum(dim=1, keepdim=True)
    return loss
