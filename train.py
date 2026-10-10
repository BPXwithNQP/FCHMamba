import os
import sys
import random
import math

_NQP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _NQP_ROOT not in sys.path:
    sys.path.insert(0, _NQP_ROOT)

import utils.conda_glibcxx  # noqa: F401  # 须在 torch 之前（GLIBCXX / selective_scan）

import numpy as np
import torch
from typing import Dict, Optional
from nqp_utils import UnifiedMetricAccumulator
import utils.logging as logging

from utils.bin_metrics import batch_bin_bucket_accuracy, top_k_setting_for_bucket_metrics


def _squeeze_depth_channel(t: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W)->(B,H,W)；保留 batch 维，避免 batch=1 时裸 .squeeze() 变成 (H,W)。"""
    if t.dim() == 4 and t.size(1) == 1:
        return t.squeeze(1)
    return t


def _iter_grfa_edge_gates(model):
    for module in model.modules():
        gate = getattr(module, "last_edge_gate", None)
        if gate is not None and module.__class__.__name__ == "GRFA":
            yield gate


def _grfa_edge_gate_loss(model, depth_gt: torch.Tensor, args):
    weight = float(getattr(args, "grfa_edge_loss_weight", 0.0))
    if weight <= 0:
        return depth_gt.new_tensor(0.0), 0

    gates = list(_iter_grfa_edge_gates(model))
    if not gates:
        return depth_gt.new_tensor(0.0), 0

    gt = _squeeze_depth_channel(depth_gt).unsqueeze(1).float()
    gt_m = gt * float(getattr(args, "max_height_eval", 200))
    kx = gt_m.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = gt_m.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(gt_m, kx, padding=1)
    gy = torch.nn.functional.conv2d(gt_m, ky, padding=1)
    edge = (torch.sqrt(gx * gx + gy * gy + 1e-12) > float(getattr(args, "grfa_edge_threshold_m", 0.75))).float()

    losses = []
    for gate in gates:
        target = torch.nn.functional.interpolate(edge, size=gate.shape[-2:], mode="nearest")
        losses.append(torch.nn.functional.binary_cross_entropy(gate.float(), target))
    return weight * torch.stack(losses).mean(), len(losses)


def _depth_numpy_hw(x):
    """深度图转为 (H,W) float64 numpy；pred[:,0] 常为 (1,H,W)，必须 squeeze 否则 OpenCV colormap 报错。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError("val_vis 深度图应为二维 (H,W)，收到 shape={}".format(x.shape))
    return x.astype(np.float64, copy=False)


def _save_val_vis_sample(log_dir, epoch, rgb, depth_gt, preds, args, sample_id=None, sample_idx=0):
    """保存一张 val 样本。默认 RGB|GT|Pred；若 preds 含 pred_d_base（FHRM）则插入一行 H_base，
    即 RGB|GT|H_base|Pred，共用色标便于看屋顶突变与地面平整度差异。"""
    import cv2
    vis_dir = os.path.join(log_dir, "val_vis")
    os.makedirs(vis_dir, exist_ok=True)
    max_h = float(getattr(args, "max_height_eval", 185))

    si = int(sample_idx)
    rgb_0 = rgb[si].detach().cpu()
    depth_gt_0 = depth_gt[si : si + 1]

    gt_np = _depth_numpy_hw(depth_gt_0.squeeze() * max_h)

    pred_t = preds["pred_d"][si : si + 1]
    if pred_t.dim() == 4 and pred_t.size(1) == 1:
        pred_np = _depth_numpy_hw(pred_t[:, 0] * max_h)
    else:
        pred_np = _depth_numpy_hw(pred_t.squeeze() * max_h)

    pred_base_np = None
    if preds is not None and "pred_d_base" in preds:
        pb = preds["pred_d_base"][si : si + 1]
        if pb.dim() == 4 and pb.size(1) == 1:
            pred_base_np = _depth_numpy_hw(pb[:, 0] * max_h)
        else:
            pred_base_np = _depth_numpy_hw(pb.squeeze() * max_h)
    rgb_np = rgb_0.numpy()
    if rgb_np.shape[0] == 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
    if rgb_np.max() > 1.0:
        rgb_np = np.clip(rgb_np / 255.0, 0.0, 1.0)
    rgb_uint8 = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    arrs_for_v = [gt_np, pred_np]
    if pred_base_np is not None:
        arrs_for_v.extend([pred_base_np])
    vmin_h = float(min(np.min(a) for a in arrs_for_v))
    vmax_h = float(max(np.max(a) for a in arrs_for_v))
    if vmax_h <= vmin_h:
        vmin_h, vmax_h = 0.0, max_h

    def _to_colormap(arr, vmin, vmax):
        arr = np.asarray(arr, dtype=np.float64)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError("_to_colormap 需要 H×W，shape={}".format(arr.shape))
        arr = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        arr_uint8 = np.ascontiguousarray((arr * 255.0).astype(np.uint8))
        return cv2.applyColorMap(arr_uint8, cv2.COLORMAP_VIRIDIS)

    gt_cmap = _to_colormap(gt_np, vmin_h, vmax_h)
    cols = [rgb_bgr, gt_cmap]
    if pred_base_np is not None:
        cols.append(_to_colormap(pred_base_np, vmin_h, vmax_h))
    cols.append(_to_colormap(pred_np, vmin_h, vmax_h))

    h, w = rgb_bgr.shape[:2]
    resized = []
    for cimg in cols:
        if cimg.shape[0] != h or cimg.shape[1] != w:
            resized.append(cv2.resize(cimg, (w, h), interpolation=cv2.INTER_LINEAR))
        else:
            resized.append(cimg)

    concat = np.concatenate(resized, axis=1)
    fname = f"epoch_{epoch:03d}_sample_{sample_id}.png" if sample_id is not None else f"epoch_{epoch:03d}_contrast.png"
    save_path = os.path.join(vis_dir, fname)
    cv2.imwrite(save_path, concat)
    layout = "RGB|GT|H_base|Pred" if pred_base_np is not None else "RGB|GT|Pred"
    print("[val_vis] 已保存: {} （{}）".format(save_path, layout))


def _fhrm_log_epoch(
    tag: str,
    epoch: int,
    *,
    criterion_d,
    model: torch.nn.Module,
    total_avg: float,
    depth_avg: float,
    fhrm_aux_avg: float,
    delta_norm_avg: float,
    delta_meter_count: int,
) -> Optional[dict]:
    """FHRM epoch 末尾盯盘日志：λ / L_aux / |pred−H_base| / 总损失与深度主项之差。"""
    fhrm = getattr(model, "fhrm", None)
    if fhrm is None:
        return None
    ll = float(fhrm.lambda_l.detach().cpu().item())
    lh = float(fhrm.lambda_h.detach().cpu().item())
    lw = float(fhrm.lambda_w.detach().cpu().item())
    w_aux = float(getattr(criterion_d, "fhrm_aux_base_weight", 0.0))
    mse_base_unw = (
        float(fhrm_aux_avg / w_aux)
        if (w_aux > 1e-12 and math.isfinite(fhrm_aux_avg))
        else float("nan")
    )
    extra_vs_depth_only = total_avg - depth_avg
    print(
        "[FHRM][{} e={:03d}] λ_l={:.4f} λ_h={:.4f} λ_w={:.4f} | "
        "L_aux(=w_aux·MSE(H_base,GT)) avg={:.6f} ⇒ MSE(H_base)≈{:.6f} (w_aux={:.4g}) | "
        "Δ|pred−H_base|(归一 L1_mean) avg={}".format(
            tag,
            int(epoch),
            ll,
            lh,
            lw,
            float(fhrm_aux_avg),
            mse_base_unw,
            w_aux,
            "{:.6f}".format(float(delta_norm_avg))
            if delta_meter_count > 0 and math.isfinite(delta_norm_avg)
            else "n/a",
        ),
        flush=True,
    )
    print(
        "[FHRM][{} e={:03d}] loss_total − loss_depth(最终pred) = {:.6f} "
        "（内含 λ_seg·seg 与 bin/其它分项；FHRM 时亦含上方 L_aux；"
        "|pred−H_base|↑ 表明细化幅度大）".format(tag, int(epoch), float(extra_vs_depth_only)),
        flush=True,
    )
    lam_max = max(ll, lh, lw)
    if lam_max < 0.01:
        hint_lambda = "λ 整体仍很小 (<0.01) → 三路残差加权偏弱，或已被 tanh/max_resid 限制。"
    elif lam_max > 0.3:
        hint_lambda = "存在 λ>0.3 → 某分支主导加权，可看是否需要略调对应 max_resid_* 或警惕纹理过拟合。"
    else:
        hint_lambda = "λ 在中等区间（与 block.yaml 里 λ_init≈0.05 一致属正常）；往后盯各 λ 谁升谁降即可。"

    print(
        "[FHRM] 判读: {} "
        "loss_total−loss_depth 在无 bin/DPLD 时多与 L_aux+λ·seg 接近。"
        " val_vis：RGB|GT|H_base|Pred。".format(hint_lambda),
        flush=True,
    )
    dl1 = (
        float(delta_norm_avg)
        if delta_meter_count > 0 and math.isfinite(delta_norm_avg)
        else float("nan")
    )
    diag: Dict[str, float] = {
        "{}/fh/lambda_l".format(tag): ll,
        "{}/fh/lambda_h".format(tag): lh,
        "{}/fh/lambda_w".format(tag): lw,
        "{}/fh/L_aux_w_mse_batch_avg".format(tag): float(fhrm_aux_avg),
        "{}/fh/MSE_H_base_approx".format(tag): float(mse_base_unw),
        "{}/fh/delta_L1_mean_norm".format(tag): dl1,
        "{}/fh/total_minus_depth".format(tag): float(extra_vs_depth_only),
    }
    return diag


global global_step
global_step = 0


def state_dict_to_cpu(model: torch.nn.Module) -> dict:
    """将 state_dict 拷贝到 CPU，避免在 GPU 上 duplicate best 权重占显存。"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(train_loader, model, criterion_d, optimizer_d, device, epoch, args, option="full"):

    global global_step

    model.train()
    depth_loss = logging.AverageMeter()
    segment_loss = logging.AverageMeter()
    total_loss = logging.AverageMeter()
    bin_loss_meter = logging.AverageMeter()
    bin_topk_hit = logging.AverageMeter()
    bin_top1_acc = logging.AverageMeter()
    fhrm_aux_meter = logging.AverageMeter()
    fhrm_delta_meter = logging.AverageMeter()
    grfa_edge_meter = logging.AverageMeter()
    yaml_top_k_for_bins = None
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    metric_freq = max(1, int(getattr(args, "train_bin_metric_freq", 20)))
    progress_freq = max(1, int(getattr(args, "progress_freq", 10)))
    accum_steps = max(1, int(getattr(args, "accum_steps", 1)))
    disable_bin_metrics = bool(getattr(args, "disable_bin_metrics", False))
    n_batches = len(train_loader)
    optimizer_d.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        global_step += 1

        _nb = device.type == "cuda"
        input_RGB = batch["image"].to(device, non_blocking=_nb)
        depth_gt = batch["depth"].to(device, non_blocking=_nb)
        segment_gt = batch["segment"].to(device, non_blocking=_nb)

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(input_RGB)

            depth_gt = _squeeze_depth_channel(depth_gt)
            depth_seg = _squeeze_depth_channel(segment_gt)
            loss, loss_d, loss_s, loss_bin, _, loss_reg, loss_fhrm_aux = criterion_d(preds, depth_gt, depth_seg)
            loss_grfa_edge, grfa_edge_count = _grfa_edge_gate_loss(model, depth_gt, args)
            if grfa_edge_count > 0:
                loss = loss + loss_grfa_edge
            if accum_steps > 1:
                loss = loss / accum_steps

        should_log_batch = (
            batch_idx % progress_freq == 0
            or batch_idx == n_batches - 1
        )
        should_compute_bin_metric = (
            not disable_bin_metrics
            and (
                batch_idx % metric_freq == 0
                or batch_idx == n_batches - 1
            )
        )
        if should_compute_bin_metric:
            tk_hit, tk1_hit, nv_pix = batch_bin_bucket_accuracy(preds, depth_gt, args, device)
            if nv_pix > 0 and not np.isnan(tk_hit):
                bin_topk_hit.update(tk_hit, nv_pix)
                bin_top1_acc.update(tk1_hit, nv_pix)
                if yaml_top_k_for_bins is None and "bin_logits" in preds:
                    yaml_top_k_for_bins = top_k_setting_for_bucket_metrics(args, preds["bin_logits"].size(1))

        bs = input_RGB.size(0)
        if should_log_batch:
            total_loss_value = loss.item() * float(accum_steps) if accum_steps > 1 else loss.item()
            depth_loss.update(loss_d.item(), bs)
            total_loss.update(total_loss_value, bs)
            segment_loss.update(loss_s.item(), bs)
            bin_loss_meter.update(loss_bin.item(), bs)
            if grfa_edge_count > 0:
                grfa_edge_meter.update(loss_grfa_edge.item(), bs)
            if getattr(model, "fhrm", None) is not None and "pred_d_base" in preds:
                fhrm_aux_meter.update(loss_fhrm_aux.item(), bs)
                pd = preds["pred_d"].detach()
                pb = preds["pred_d_base"].detach()
                if pd.dim() == 4 and pd.size(1) == 1:
                    pd = pd.squeeze(1)
                if pb.dim() == 4 and pb.size(1) == 1:
                    pb = pb.squeeze(1)
                fhrm_delta_meter.update(torch.mean(torch.abs(pd - pb)).item(), bs)
        scaler.scale(loss).backward()
        is_accum_end = (
            (batch_idx + 1) % accum_steps == 0
            or batch_idx == n_batches - 1
        )
        if is_accum_end:
            if getattr(args, 'clip_max_norm', 0) > 0:
                scaler.unscale_(optimizer_d)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            scaler.step(optimizer_d)
            scaler.update()
            optimizer_d.zero_grad(set_to_none=True)

        if should_log_batch:
            prog_msg = 'Depth Loss: %.4f (%.4f)' % (total_loss.val, total_loss.avg)
            if bin_topk_hit.count > 0 and yaml_top_k_for_bins is not None:
                prog_msg += ' | BinTop%d: %.4f  Top1: %.4f' % (
                    yaml_top_k_for_bins,
                    bin_topk_hit.avg,
                    bin_top1_acc.avg,
                )
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch, prog_msg)

    fh_diag_train = _fhrm_log_epoch(
        "train",
        epoch,
        criterion_d=criterion_d,
        model=model,
        total_avg=float(total_loss.avg),
        depth_avg=float(depth_loss.avg),
        fhrm_aux_avg=float(fhrm_aux_meter.avg) if fhrm_aux_meter.count > 0 else float("nan"),
        delta_norm_avg=float(fhrm_delta_meter.avg) if fhrm_delta_meter.count > 0 else float("nan"),
        delta_meter_count=int(fhrm_delta_meter.count),
    )
    if grfa_edge_meter.count > 0:
        if fh_diag_train is None:
            fh_diag_train = {}
        fh_diag_train["train/grfa_edge_loss"] = float(grfa_edge_meter.avg)

    tk_out = bin_topk_hit.avg if bin_topk_hit.count > 0 else float("nan")
    t1_out = bin_top1_acc.avg if bin_top1_acc.count > 0 else float("nan")
    return (
        total_loss.avg,
        depth_loss.avg,
        segment_loss.avg,
        bin_loss_meter.avg,
        tk_out,
        t1_out,
        yaml_top_k_for_bins,
        model,
        fh_diag_train,
    )


def save_val_vis_from_cache(log_dir, epoch, vis_cache, args):
    """将 validate 缓存的单张 val 样本写出为对比图（由 main 在 new best 时调用）。"""
    if not vis_cache:
        return
    _save_val_vis_sample(
        log_dir,
        epoch,
        vis_cache["rgb"],
        vis_cache["depth_gt"],
        vis_cache["preds"],
        args,
        sample_id=vis_cache.get("sample_id"),
        sample_idx=vis_cache.get("sample_idx", 0),
    )


def validate(val_loader, model, criterion_d, device, epoch, args, log_dir, option="full", val_vis_target_idx=None):
    """
    验证指标统一由 nqp_utils.UnifiedMetricAccumulator 在全 val 上 pooled 计算（米制）。
    val_vis：验证时仅缓存一张样本（seed+epoch 随机；或 val_vis_target_idx 固定下标），
    由 main 在 new best 时调用 save_val_vis_from_cache 落盘。
    """
    depth_loss = logging.AverageMeter()
    segment_loss = logging.AverageMeter()
    total_loss = logging.AverageMeter()
    bin_loss_meter = logging.AverageMeter()
    bin_topk_hit = logging.AverageMeter()
    bin_top1_acc = logging.AverageMeter()
    fhrm_aux_meter = logging.AverageMeter()
    fhrm_delta_meter = logging.AverageMeter()
    yaml_top_k_for_bins = None
    model.eval()

    n_val_batches = len(val_loader)
    seed = int(getattr(args, "seed", 42))
    rng_vis = random.Random(seed + epoch * 10009)
    if val_vis_target_idx is None:
        vis_batch_idx = rng_vis.randrange(n_val_batches) if n_val_batches > 0 else -1
        fixed_dataset_idx = None
    else:
        vis_batch_idx = None
        fixed_dataset_idx = int(val_vis_target_idx)

    mask_thr = float(getattr(args, 'mask_depth_threshold', 1.0))
    paper_min = float(getattr(args, 'paper_building_min_m', 0.0))
    metric_acc = UnifiedMetricAccumulator(
        mask_depth_threshold=mask_thr, paper_building_min_m=paper_min,
    )
    vis_cache = None
    progress_freq = max(1, int(getattr(args, "progress_freq", 10)))
    disable_bin_metrics = bool(getattr(args, "disable_bin_metrics", False))

    _nb = device.type == "cuda"
    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch["image"].to(device, non_blocking=_nb)
        depth_gt = batch["depth"].to(device, non_blocking=_nb)
        segment_gt = batch["segment"].to(device, non_blocking=_nb)

        with torch.no_grad():
            preds = model(input_RGB)

        if not disable_bin_metrics:
            tk_hit, tk1_hit, nv_pix = batch_bin_bucket_accuracy(preds, depth_gt, args, device)
            if nv_pix > 0 and not np.isnan(tk_hit):
                bin_topk_hit.update(tk_hit, nv_pix)
                bin_top1_acc.update(tk1_hit, nv_pix)
                if yaml_top_k_for_bins is None and "bin_logits" in preds:
                    yaml_top_k_for_bins = top_k_setting_for_bucket_metrics(args, preds["bin_logits"].size(1))

        batch_size_cur = input_RGB.size(0)
        need_vis = False
        vis_i = 0
        if fixed_dataset_idx is None and n_val_batches > 0 and batch_idx == vis_batch_idx:
            need_vis = True
            vis_i = rng_vis.randrange(batch_size_cur)
        elif fixed_dataset_idx is not None:
            if batch_idx * batch_size_cur <= fixed_dataset_idx < (batch_idx + 1) * batch_size_cur:
                need_vis = True
                vis_i = fixed_dataset_idx - batch_idx * batch_size_cur

        if need_vis:
            preds_cpu = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in preds.items()
            }
            vis_cache = {
                "rgb": input_RGB.detach().cpu(),
                "depth_gt": depth_gt.detach().cpu(),
                "preds": preds_cpu,
                "sample_id": f"{batch_idx}_{vis_i}" if fixed_dataset_idx is None else fixed_dataset_idx,
                "sample_idx": vis_i,
            }

        depth_gt = _squeeze_depth_channel(depth_gt)
        depth_seg = _squeeze_depth_channel(segment_gt)
        loss, loss_d, loss_s, loss_bin, _, loss_reg, loss_fhrm_aux = criterion_d(preds, depth_gt, depth_seg)

        bs = input_RGB.size(0)
        depth_loss.update(loss_d.item(), bs)
        total_loss.update(loss.item(), bs)
        segment_loss.update(loss_s.item(), bs)
        bin_loss_meter.update(loss_bin.item(), bs)
        if getattr(model, "fhrm", None) is not None and "pred_d_base" in preds:
            fhrm_aux_meter.update(loss_fhrm_aux.item(), bs)
            pd = preds["pred_d"].detach()
            pb = preds["pred_d_base"].detach()
            if pd.dim() == 4 and pd.size(1) == 1:
                pd = pd.squeeze(1)
            if pb.dim() == 4 and pb.size(1) == 1:
                pb = pb.squeeze(1)
            fhrm_delta_meter.update(torch.mean(torch.abs(pd - pb)).item(), bs)

        pred_d = preds["pred_d"]
        if pred_d.dim() == 4 and pred_d.size(1) == 1:
            pred_d = pred_d.squeeze(1)
        pred_d_denorm = pred_d * args.max_height_eval
        depth_gt_denorm = depth_gt * args.max_height_eval
        bmask_v = batch.get("building_mask")
        if bmask_v is not None:
            bmask_v = bmask_v.to(device)
            if bmask_v.dim() == 4 and bmask_v.size(1) == 1:
                bmask_v = bmask_v.squeeze(1)
        metric_acc.update(pred_d_denorm, depth_gt_denorm, bmask_v)

        if batch_idx % progress_freq == 0 or batch_idx == len(val_loader) - 1:
            logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

    result_metrics = metric_acc.finalize()

    fh_diag_val = _fhrm_log_epoch(
        "val",
        epoch,
        criterion_d=criterion_d,
        model=model,
        total_avg=float(total_loss.avg),
        depth_avg=float(depth_loss.avg),
        fhrm_aux_avg=float(fhrm_aux_meter.avg) if fhrm_aux_meter.count > 0 else float("nan"),
        delta_norm_avg=float(fhrm_delta_meter.avg) if fhrm_delta_meter.count > 0 else float("nan"),
        delta_meter_count=int(fhrm_delta_meter.count),
    )

    vk = bin_topk_hit.avg if bin_topk_hit.count > 0 else float("nan")
    v1 = bin_top1_acc.avg if bin_top1_acc.count > 0 else float("nan")
    return (
        result_metrics,
        total_loss.avg,
        depth_loss.avg,
        segment_loss.avg,
        bin_loss_meter.avg,
        vk,
        v1,
        yaml_top_k_for_bins,
        model,
        fh_diag_val,
        vis_cache,
    )
