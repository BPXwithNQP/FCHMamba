import torch

# eval_depth 中「建筑前景」定义：
#   - 若传入 building_mask（数据集 BLG），则用 mask>0 的像素，不再用 GT 高度阈值；
#   - 否则沿用 target>mask_depth_threshold（默认 1m，与旧版及无 BLG 数据集一致）。
# 以下键在无前景像素时为 nan
MASK_DEPTH_METRIC_KEYS = (
    'abs_rel_mask', 'sq_rel_mask', 'rmse_mask', 'rmse_log_mask',
    'abs_rel_mask_3m', 'abs_rel_mask_5m', 'sq_rel_mask_3m', 'sq_rel_mask_5m',
    'b10_mask',
)


def _infer_ratio_floor(pred, target):
    """
    仅用于 δ 的 max(pred/gt, gt/pred)：pred/target 与 loss 同尺度。
    旧代码写死 0.0001：在米制下是 0.1mm，分母过小 → δ 异常偏低；归一化下语义不同。
    按数值量级自动选 ratio 下限（米制用 1cm，归一化保持 1e-4）。
    """
    m = float(torch.maximum(pred.max(), target.max()).detach().cpu())
    if m > 1.5:
        return 0.01  # 米
    return 1e-4  # 归一化


def eval_depth(
    pred,
    target,
    bin_num=10,
    bin_len=6.0,
    height_thresholds=None,
    mask_depth_threshold=1.0,
    ratio_floor=None,
    building_mask=None,
):
    assert pred.shape == target.shape
    pred_orig = pred.clone()
    target_orig = target.clone()

    def _prepare_building_mask(bm):
        if bm is None:
            return None
        m = bm
        if not torch.is_tensor(m):
            m = torch.as_tensor(m, device=pred_orig.device, dtype=torch.float32)
        else:
            m = m.to(device=pred_orig.device, dtype=torch.float32)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m.squeeze(1)
        # batch=1 时常为 (1, H, W) 而 pred/target 已与 train 中 depth 一同 squeeze 成 (H, W)
        while m.ndim > pred_orig.ndim and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.shape != pred_orig.shape:
            raise ValueError(
                "building_mask 形状 {} 与 pred/target {} 不一致".format(
                    tuple(m.shape), tuple(pred_orig.shape)
                )
            )
        return m

    # ---------- δ：单独用稳定 ratio 下限，不改变 RMSE 的物理含义 ----------
    eps_r = ratio_floor if ratio_floor is not None else _infer_ratio_floor(pred_orig, target_orig)
    pred_r = torch.clamp(pred_orig, min=eps_r)
    target_r = torch.clamp(target_orig, min=eps_r)
    thresh = torch.max((pred_r / target_r), (target_r / pred_r))
    denom = thresh.numel() if thresh.numel() > 0 else 1
    d1 = torch.sum(thresh < 1.25).float() / denom
    d2 = torch.sum(thresh < 1.25 ** 2).float() / denom
    d3 = torch.sum(thresh < 1.25 ** 3).float() / denom

    # ---------- 其余指标：原始 pred/GT；log 仅做数值正数保护 ----------
    log_floor = 1e-6
    bm = _prepare_building_mask(building_mask)
    if bm is not None:
        # 数据集 BLG：建筑/非建筑划分仅由 mask 决定（与论文 RMSE-M/NM 口径一致）
        mask = (bm > 0.5) & torch.isfinite(target_orig) & torch.isfinite(pred_orig)
    else:
        # 无 BLG 时（如 Vaihingen 部分划分）：与 HTC-DC 旧逻辑一致，GT 高度 > thr 为前景
        mask = (target_orig > mask_depth_threshold) & torch.isfinite(
            target_orig
        ) & torch.isfinite(pred_orig)
    pred_mask = pred_orig[mask]
    target_mask = target_orig[mask]
    n_mask = int(pred_mask.numel())

    volume = torch.min(
        (torch.sum(pred_orig) / torch.sum(target_orig)),
        (torch.sum(target_orig) / torch.sum(pred_orig)),
    )

    diff = pred_orig - target_orig
    diff_log = torch.log(torch.clamp(pred_orig, min=log_floor)) - torch.log(
        torch.clamp(target_orig, min=log_floor)
    )

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(torch.clamp(pred_orig, min=log_floor)) - torch.log10(torch.clamp(target_orig, min=log_floor))))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    # 预测二值前景：仍用 pred>thr 与 GT mask 算 IoU（与 HTC-DC basenets 一致）
    binary = pred_orig > mask_depth_threshold
    segment = mask
    binary = binary.float()
    segment = segment.float()
    intersection = (binary + segment) == 2
    union = (binary + segment) >= 1
    iou = (torch.sum(intersection)) / (torch.sum(union) + 1)
    seg_sum = torch.sum(segment)
    segment_recall = (torch.sum(intersection).float() / (seg_sum + 1e-8))

    device = pred.device
    nanv = torch.tensor(float('nan'), device=device)

    if n_mask == 0:
        abs_rel_mask = sq_rel_mask = rmse_mask = rmse_log_mask = nanv
        abs_rel_mask_3m = abs_rel_mask_5m = sq_rel_mask_3m = sq_rel_mask_5m = nanv
        b10_mask = nanv
    else:
        diff_mask = pred_mask - target_mask
        diff_log_mask = torch.log(torch.clamp(pred_mask, min=log_floor)) - torch.log(
            torch.clamp(target_mask, min=log_floor)
        )
        abs_rel_mask = torch.mean(torch.abs(diff_mask) / target_mask)
        sq_rel_mask = torch.mean(torch.pow(diff_mask, 2) / target_mask)
        rmse_mask = torch.sqrt(torch.mean(torch.pow(diff_mask, 2)))
        rmse_log_mask = torch.sqrt(torch.mean(torch.pow(diff_log_mask, 2)))

        # B10（MAFENet）：建筑像素 max(P/G, G/P) < 1.1 的占比
        pm_r = torch.clamp(pred_mask, min=eps_r)
        tm_r = torch.clamp(target_mask, min=eps_r)
        ratio_mask = torch.max(pm_r / tm_r, tm_r / pm_r)
        b10_mask = (ratio_mask < 1.1).float().mean()

        pm = pred_mask.clone()
        tm = target_mask.clone()
        tm[tm < 3] = 3
        pm[pm < 3] = 3
        diff_mask_3m = pm - tm
        abs_rel_mask_3m = torch.mean(torch.abs(diff_mask_3m) / tm)
        sq_rel_mask_3m = torch.mean(torch.pow(diff_mask_3m, 2) / tm)

        tm[tm < 5] = 5
        pm[pm < 5] = 5
        diff_mask_5m = pm - tm
        abs_rel_mask_5m = torch.mean(torch.abs(diff_mask_5m) / tm)
        sq_rel_mask_5m = torch.mean(torch.pow(diff_mask_5m, 2) / tm)

    # 按高度阈值分段的 Masked RMSE：只对 GT > th 的像素算 RMSE（严格 >）
    if height_thresholds is None:
        height_thresholds = [i * float(bin_len) for i in range(bin_num)]
    height_rmse = {}
    for th in height_thresholds:
        th_val = float(th)
        hmask = (target_orig > th_val)
        n = hmask.sum()
        if n > 0:
            hdiff = (pred_orig[hmask] - target_orig[hmask])
            val = torch.sqrt((hdiff ** 2).mean())
        else:
            val = torch.tensor(0.0, device=pred.device)
        key = f'rmse_>{th_val:.1f}m'
        height_rmse[key] = val

    out = {'d1': d1, 'd2': d2, 'd3': d3,
           'abs_rel_mask': abs_rel_mask, 'abs_rel_mask_3m': abs_rel_mask_3m, 'abs_rel_mask_5m': abs_rel_mask_5m,
           'sq_rel_mask': sq_rel_mask, 'sq_rel_mask_3m': sq_rel_mask_3m, 'sq_rel_mask_5m': sq_rel_mask_5m,
           'rmse': rmse, 'rmse_mask': rmse_mask,
           'b10_mask': b10_mask,
           'segment_iou': iou, 'segment_recall': segment_recall,
           'rmse_log': rmse_log, 'rmse_log_mask': rmse_log_mask,
           'log10': log10, 'silog': silog, 'volume': volume}
    out.update(height_rmse)
    return out
