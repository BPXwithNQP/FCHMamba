"""
在 test 集上评估 FusedHE 原版模型：深度与分割指标，参数量与推理时间。
用法示例：
  python test.py --ckpt_path output/xxx/best.ckpt
  python test.py --ckpt_path output/xxx/best.ckpt --no_mask 5
  python test.py --ckpt_path output/xxx/best.ckpt --test_dir ../dataset/dfc2019/test
  python test.py --ckpt_path best.ckpt --test_dir /path/to/htcdc_dfc23/test_data_split_dirs/test.txt
  python test.py --ckpt_path output/xxx/best.ckpt --id GF2_NewDelhi_28.5557_77.1225_IMG
  # --id：按 stem 在「权重对应数据集根」全库解析（gbh_txt：data_dir 下 image/ndsm，不限于 test.txt）。
若 ckpt 为 main_dfc2023 保存的 dict（含 args），默认会把训练时的超参注入为 argparse 默认值，
仍可用命令行覆盖；纯 state_dict 时从权重形状/键名推断结构，并用 --no_ckpt_args 可禁用元数据合并。
默认使用 dataset_root/dataset_name/test，dataset_name 默认为 dfc2019。
gbh_txt 布局时：未显式传 --data_dir 等则按 dataset_root/dataset_name 解析 txt（不沿用 checkpoint 内保存的绝对路径）。
Test 指标统一由仓库根目录 nqp_utils（UnifiedMetricAccumulator）在米制上计算。
FLOPs：默认用 thop 估计（pip install thop），失败时尝试 torch.profiler。
"""
# 限制 BLAS/OpenMP 线程数，避免在 RLIMIT_NPROC 紧张时 pthread_create 失败导致 numpy 导入报错
import os
_env = os.environ
_env.setdefault("OPENBLAS_NUM_THREADS", "4")
_env.setdefault("OMP_NUM_THREADS", "4")
_env.setdefault("MKL_NUM_THREADS", "4")

# 必须在 numpy/torch 之前：优先加载 conda 的 libstdc++（selective_scan 扩展需 GLIBCXX_3.4.29+）
import utils.conda_glibcxx  # noqa: F401

import argparse
import sys
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader

_NQP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _NQP_ROOT not in sys.path:
    sys.path.insert(0, _NQP_ROOT)
from nqp_utils import (
    UnifiedMetricAccumulator,
    profile_model,
    count_parameters as nqp_count_parameters,
)

from main import get_args_parser
from utils.encoder_presets import resolve_encoder_args
from utils.dataset_config import (
    apply_test_dir_arg_for_gbh_txt,
    build_test_dataset,
    maybe_apply_ndsm_pickle_max_height,
    resolve_test_data_path,
    _resolve_gbh_dirs,
)
from utils.gbh_split_dataset import load_gbh_sample_by_stem
from models.Blocks.GRFA import set_freq_lambda
from models.Blocks.RGHF import set_freq_lambda as set_rghf_freq_lambda
from models.model import FHE

from utils.bin_metrics import (
    batch_bin_bucket_accuracy,
    infer_bin_metrics_section_from_model,
    top_k_setting_for_bucket_metrics,
)


def _infer_no_reshape_from_state(state):
    """从 decoder.block1 输入通道推断是否使用 MambaVision 原生通道（未对齐 MiT）。"""
    w = state.get('decoder.block1.conv1.0.weight')
    if w is None or not hasattr(w, 'shape') or len(w.shape) < 2:
        return False
    in_ch = int(w.shape[1])
    # Tiny 原生: 2*640=1280；Base 原生: 2*1024=2048；对齐 MiT: 2*512=1024
    return in_ch in (1280, 2048)


def _infer_bin_num_from_state(state):
    """从 state_dict 推断 condition_heads 的 bin_num（档数）。"""
    key_centers = 'condition_heads.centers'
    if key_centers in state:
        return state[key_centers].shape[0]
    key_bin = 'condition_heads.bin_head.2.weight'
    if key_bin in state:
        return state[key_bin].shape[0]
    return None


def _infer_bin_len_from_state(state):
    """从 state_dict 的 centers buffer 推断 bin_len。centers[i]=(i+0.5)*bin_len => bin_len≈centers[1]-centers[0]。"""
    key_centers = 'condition_heads.centers'
    if key_centers not in state:
        return None
    centers = state[key_centers]
    if centers.numel() >= 2:
        return (centers[1] - centers[0]).item()
    if centers.numel() == 1:
        return (2.0 * centers[0].item())
    return None


def _infer_use_conv_from_state(state):
    """从 state_dict 推断是否使用 ConvNeXt V2 替换 ResNet。
    返回 0 (ResNet), 1 (ConvNeXt native), 2 (ConvNeXt reshaped)。"""
    has_convnext = any('backbone.downsample_layers' in k for k in state)
    if not has_convnext:
        return 0
    has_adapter_params = any(
        '.adapters.' in k and 'weight' in k and '.mit.' not in k
        for k in state)
    return 2 if has_adapter_params else 1


def _infer_adabin_from_state(state):
    """检查 state_dict 中是否包含 adabin_head，推断 use_adabin。"""
    return any(k.startswith('adabin_head.') for k in state)


def _infer_decoder_cfg_from_state(state):
    """
    从 checkpoint 推断 FusedDecoder 的通道配置，修复“test.py 使用默认 args 导致结构不匹配”问题。

    主要根据以下权重的形状推断：
    - decoder.ff{i}.conv1.0.weight: [out=in/2, in=in, k, k]  => in_channel_ffb[i-1] = shape[1]
    - decoder.ff{i}.conv2.0.weight: [out=out_channel_ffb[i-1], ...] => out_channel_ffb[i-1] = shape[0]
    - decoder.block{i}.conv1.0.weight: [out=out_channel_decoderb[i-1], in=in_channel_decoderb[i-1], k, k]
    - decoder.conv1.0.weight: [out=out_decoder, in=..., k, k]
    """
    # ff* (1..4)
    out_channel_ffb = []
    in_channel_ffb = []
    for i in range(1, 5):
        w_ff_conv1 = f"decoder.ff{i}.conv1.0.weight"
        w_ff_conv2 = f"decoder.ff{i}.conv2.0.weight"
        if w_ff_conv1 not in state or w_ff_conv2 not in state:
            return None
        w1 = state[w_ff_conv1]
        w2 = state[w_ff_conv2]
        # w1: [out=int(in/2), in=in, k, k]
        if not hasattr(w1, "shape") or len(w1.shape) < 2:
            return None
        in_channel_ffb.append(int(w1.shape[1]))
        # w2: [out=out_channel_ffb, ...]
        out_channel_ffb.append(int(w2.shape[0]))

    # block* (1..3)
    out_channel_decoderb = []
    in_channel_decoderb = []
    for i in range(1, 4):
        w_block_conv1 = f"decoder.block{i}.conv1.0.weight"
        if w_block_conv1 not in state:
            return None
        w = state[w_block_conv1]
        if not hasattr(w, "shape") or len(w.shape) < 2:
            return None
        out_channel_decoderb.append(int(w.shape[0]))
        in_channel_decoderb.append(int(w.shape[1]))

    w_decoder_conv1 = "decoder.conv1.0.weight"
    if w_decoder_conv1 not in state:
        return None
    out_decoder = int(state[w_decoder_conv1].shape[0])

    return {
        "in_channel_ffb": in_channel_ffb,
        "out_channel_ffb": out_channel_ffb,
        "in_channel_decoderb": in_channel_decoderb,
        "out_channel_decoderb": out_channel_decoderb,
        "out_decoder": out_decoder,
    }


def _squeeze_depth_channel(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4 and t.size(1) == 1:
        return t.squeeze(1)
    return t


def _depth_numpy_hw(x):
    """确保深度图为二维 numpy 数组 (H,W)。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError("深度图应为二维 (H,W)，收到 shape={}".format(x.shape))
    return x.astype(np.float64, copy=False)


def _save_test_vis_sample(vis_dir, batch_idx, rgb, depth_gt, preds, args, sample_id=None):
    """保存单个 test 样本的可视化：RGB | GT | Pred"""
    os.makedirs(vis_dir, exist_ok=True)
    max_h = float(getattr(args, "max_height_eval", 185))

    # 取 batch 中第一个样本
    rgb_0 = rgb[0].detach().cpu()
    depth_gt_0 = depth_gt[0:1]

    gt_np = _depth_numpy_hw(depth_gt_0.squeeze() * max_h)

    pred_t = preds["pred_d"][0:1]
    if pred_t.dim() == 4 and pred_t.size(1) == 1:
        pred_np = _depth_numpy_hw(pred_t[:, 0] * max_h)
    else:
        pred_np = _depth_numpy_hw(pred_t.squeeze() * max_h)

    # 处理 RGB
    rgb_np = rgb_0.numpy()
    if rgb_np.shape[0] == 3:
        rgb_np = np.transpose(rgb_np, (1, 2, 0))
    if rgb_np.max() > 1.0:
        rgb_np = np.clip(rgb_np / 255.0, 0.0, 1.0)
    rgb_uint8 = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # 确定色标范围
    vmin_h = float(min(np.min(gt_np), np.min(pred_np)))
    vmax_h = float(max(np.max(gt_np), np.max(pred_np)))
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
    pred_cmap = _to_colormap(pred_np, vmin_h, vmax_h)

    cols = [rgb_bgr, gt_cmap, pred_cmap]

    h, w = rgb_bgr.shape[:2]
    resized = []
    for cimg in cols:
        if cimg.shape[0] != h or cimg.shape[1] != w:
            resized.append(cv2.resize(cimg, (w, h), interpolation=cv2.INTER_LINEAR))
        else:
            resized.append(cimg)

    concat = np.concatenate(resized, axis=1)
    fname = f"test_{batch_idx:04d}_{sample_id}.png" if sample_id else f"test_{batch_idx:04d}.png"
    save_path = os.path.join(vis_dir, fname)
    cv2.imwrite(save_path, concat)
    return save_path


def evaluate_test(test_loader, model, device, args, vis_output_dir: str = None):
    """米制、全 test pooled；指标由 UnifiedMetricAccumulator；附带条件头 / DPLD / TBWH 时写入 bin_topk/top1。

    Args:
        vis_output_dir: 如不为 None，则保存每张测试图片的可视化结果到该目录
    """
    model.eval()
    section_ov = infer_bin_metrics_section_from_model(model)
    bin_sum_hit = 0.0
    bin_sum_top1 = 0.0
    bin_n_valid = 0
    yaml_k_bins = None
    no_mask_thr = float(getattr(args, 'no_mask', 0.0))
    if no_mask_thr > 0:
        # 忽略数据集 BLG，用 GT nDSM（米）> 阈值 生成临时建筑掩码；IoU 预测侧二值化与 mask_depth_threshold 对齐
        mask_thr = no_mask_thr
    else:
        mask_thr = float(getattr(args, 'mask_depth_threshold', 1.0))
    max_h = float(getattr(args, 'max_height_eval', 185.0))
    paper_min = float(getattr(args, 'paper_building_min_m', 0.0))
    metric_acc = UnifiedMetricAccumulator(
        mask_depth_threshold=mask_thr, paper_building_min_m=paper_min,
    )
    mask_audit_rows = []
    vis_sample_count = 0  # 全量可视化计数器

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_RGB = batch['image'].to(device)
            depth_gt = _squeeze_depth_channel(batch["depth"].to(device))

            preds = model(input_RGB)

            # 处理 mask_test 参数对高度输出的修改
            mask_test_mode = int(getattr(args, 'mask_test', 0))
            if mask_test_mode > 0:
                pred_d = preds.get("pred_d")
                if pred_d is not None:
                    # 反归一化到米制（假设 max_h 已在 args 中）
                    max_h = float(getattr(args, 'max_height_eval', 185.0))
                    pred_d_denorm = pred_d * max_h

                    if mask_test_mode == 1:
                        # 使用分割副产物：将分割结果覆盖在高度输出图上
                        # 高度主体区域（>1的部分）设为1，不超过1的部分保留原值
                        pred_seg = preds.get("pred_seg")
                        if pred_seg is not None:
                            # 分割图转换为二值掩码（假设分割输出经过 sigmoid 后>0.5为建筑）
                            seg_mask = (pred_seg > 0.5).float()
                            # 对高度主体区域（>1）设为1，否则保留原值
                            height_mask = (pred_d_denorm > 1.0).float()
                            # 结合分割掩码和高度掩码
                            combined_mask = seg_mask * height_mask
                            # 在分割建筑区域内，高度>1的设为1，其他保留原值
                            pred_d_denorm = torch.where(
                                combined_mask.bool(),
                                torch.ones_like(pred_d_denorm),  # 设为1
                                pred_d_denorm  # 保留原值
                            )
                        # 重新归一化
                        preds["pred_d"] = pred_d_denorm / max_h

                    elif mask_test_mode == 2:
                        # 不使用分割副产物，直接对高度图设阈值
                        # 低于1的设为0
                        pred_d_denorm = torch.where(
                            pred_d_denorm < 1.0,
                            torch.zeros_like(pred_d_denorm),
                            pred_d_denorm
                        )
                        # 重新归一化
                        preds["pred_d"] = pred_d_denorm / max_h

            tk_m, top1_m, nv_b = batch_bin_bucket_accuracy(
                preds, depth_gt, args, device, section_override=section_ov,
            )
            if nv_b > 0:
                bin_sum_hit += tk_m * nv_b
                bin_sum_top1 += top1_m * nv_b
                bin_n_valid += nv_b
                if yaml_k_bins is None and "bin_logits" in preds:
                    yaml_k_bins = top_k_setting_for_bucket_metrics(
                        args, preds["bin_logits"].size(1), section_override=section_ov,
                    )

            pred_d = preds["pred_d"]
            if pred_d.dim() == 4 and pred_d.size(1) == 1:
                pred_d = pred_d.squeeze(1)
            pred_d_denorm = pred_d * max_h
            depth_gt_denorm = depth_gt * max_h

            blg_mask = batch.get('building_mask')
            if blg_mask is not None:
                blg_mask = blg_mask.to(device)
                if blg_mask.dim() == 4 and blg_mask.size(1) == 1:
                    blg_mask = blg_mask.squeeze(1)
            if no_mask_thr > 0:
                bmask = (depth_gt_denorm > no_mask_thr).to(dtype=torch.float32)
            else:
                bmask = blg_mask
            metric_acc.update(
                pred_d_denorm, depth_gt_denorm, bmask, b10_building_mask=blg_mask,
            )

            # 全量可视化保存
            if vis_output_dir is not None:
                batch_size = input_RGB.size(0)
                for i in range(batch_size):
                    # 获取样本名称
                    sample_name = None
                    if 'fname' in batch and i < len(batch['fname']):
                        sample_name = batch['fname'][i]
                        if isinstance(sample_name, str):
                            sample_name = os.path.splitext(os.path.basename(sample_name))[0]
                    if sample_name is None:
                        sample_name = f"sample_{batch_idx}_{i}"

                    _save_test_vis_sample(
                        vis_output_dir,
                        batch_idx * len(test_loader.dataset) + i,
                        input_RGB[i:i+1],
                        depth_gt[i:i+1],
                        {"pred_d": pred_d[i:i+1]},
                        args,
                        sample_id=sample_name
                    )

            t = depth_gt_denorm
            n_pix = int(t.numel())
            n_ge2 = int((t > mask_thr).sum().item())
            mask_audit_rows.append({
                'frac_ge2': n_ge2 / n_pix if n_pix else 0.0,
                'n_ge2': n_ge2,
                'n_pix': n_pix,
                'gt_min_m': float(t.min()),
                'gt_max_m': float(t.max()),
                'gt_mean_m': float(t.mean()),
            })

    n_batches = len(mask_audit_rows)
    frs = [r['frac_ge2'] for r in mask_audit_rows]
    mask_audit = {
        'mask_depth_threshold_m': mask_thr,
        'n_batches': n_batches,
        'empty_ge2_batches': sum(1 for r in mask_audit_rows if r['n_ge2'] == 0),
        'frac_ge2_min': min(frs) if frs else 0.0,
        'frac_ge2_max': max(frs) if frs else 0.0,
        'frac_ge2_mean': float(np.mean(frs)) if frs else 0.0,
        'gt_m_min': min(r['gt_min_m'] for r in mask_audit_rows) if mask_audit_rows else None,
        'gt_m_max': max(r['gt_max_m'] for r in mask_audit_rows) if mask_audit_rows else None,
        'rows': mask_audit_rows if getattr(args, 'verbose_mask_audit', False) else [],
    }

    unified = metric_acc.finalize()
    if bin_n_valid > 0 and yaml_k_bins is not None:
        unified = dict(unified)
        unified["bin_top{}_hit".format(yaml_k_bins)] = bin_sum_hit / bin_n_valid
        unified["bin_top1_acc"] = bin_sum_top1 / bin_n_valid
    height_thresholds = []
    return unified, height_thresholds, mask_audit, unified


_FAF_FF_BACKBONE_STAGES = (
    ("ff1", 4),
    ("ff2", 3),
    ("ff3", 2),
    ("ff4", 1),
)


def _normalize_id_stem(id_str: str) -> str:
    s = str(id_str).strip()
    low = s.lower()
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        if low.endswith(ext):
            s = s[: -len(ext)]
            break
    return s


def _stem_matches(key: str, stem: str) -> bool:
    key = key.strip()
    stem = stem.strip()
    if key == stem:
        return True

    def _strip_img(s: str) -> str:
        return s[: -len("_IMG")] if s.endswith("_IMG") else s

    return _strip_img(key) == _strip_img(stem)


def _load_folder_sample_by_stem_global(args, key: str) -> dict:
    """folder 布局：在 dataset_name 根、test/train/validation 下含 rgb/ 的目录中按文件名 stem 查找。"""
    from utils.local_dataset import LocalDataset

    roots = []
    base = os.path.join(args.dataset_root, args.dataset_name)
    for sub in ("", "test", "train", "validation"):
        roots.append(os.path.join(base, sub) if sub else base)
    last_err = None
    for root in roots:
        rgb_dir = os.path.join(root, "rgb")
        if not os.path.isdir(rgb_dir):
            continue
        try:
            ds = LocalDataset(root, args=args)
        except Exception as e:
            last_err = e
            continue
        for i in range(len(ds)):
            rgb_f, _ = ds._pairs[i]
            stem = os.path.splitext(rgb_f)[0]
            if _stem_matches(key, stem):
                print(
                    "[Test --id] folder 布局：于目录 {} 命中样本".format(root),
                    flush=True,
                )
                return ds[i]
    msg = "folder 布局下在 {} 的 rgb/ 中未找到 stem={!r}".format(base, key)
    if last_err is not None:
        msg += "（最后一次 LocalDataset 错误: {}）".format(last_err)
    raise FileNotFoundError(msg)


def _load_sample_for_single_id(args, id_str: str) -> dict:
    """按 --id 从数据集根全库加载（gbh_txt：data_dir；folder：多目录 rgb 扫描）。"""
    key = _normalize_id_stem(id_str)
    layout = getattr(args, "dataset_layout", "folder")
    if layout == "gbh_txt":
        data_dir, _, _, _ = _resolve_gbh_dirs(args)
        sample = load_gbh_sample_by_stem(data_dir, key, args)
        print(
            "[Test --id] gbh_txt：已从数据根目录全库解析（非仅 test.txt）: {}".format(data_dir),
            flush=True,
        )
        return sample
    return _load_folder_sample_by_stem_global(args, key)


def _safe_dir_name(stem: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", stem.strip())
    return s[:200] if len(s) > 200 else s


def _faf_activation_gray01(t: torch.Tensor, batch_idx: int = 0) -> np.ndarray:
    """(B,C,H,W) → (H,W) float32，按通道 |x| 均值 min-max 归一化到约 [0,1]。"""
    if t.dim() != 4:
        raise ValueError("_faf_activation_gray01 需要 (B,C,H,W)，收到 dim={}".format(t.dim()))
    x = t[batch_idx].detach().float().cpu().numpy()
    g = np.abs(x).mean(axis=0)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo < 1e-8:
        return np.zeros_like(g, dtype=np.float32)
    return ((g - lo) / (hi - lo)).astype(np.float32)


def _faf_lerp_bgr(gray01: np.ndarray, bgr_lo: np.ndarray, bgr_hi: np.ndarray) -> np.ndarray:
    """gray01: (H,W) in [0,1]；bgr_* shape (3,)_BGR。"""
    g = gray01.astype(np.float32)[..., np.newaxis]
    lo = bgr_lo.reshape(1, 1, 3).astype(np.float32)
    hi = bgr_hi.reshape(1, 1, 3).astype(np.float32)
    out = (1.0 - g) * lo + g * hi
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# BGR：低频 / MiT 浅蓝 → 蓝（提高 B、压低 R）
_FAF_BGR_BLUE_LO = np.array([255, 248, 240], dtype=np.float32)
_FAF_BGR_BLUE_HI = np.array([255, 130, 60], dtype=np.float32)

# BGR：高频 / CNN 浅橙 → 红（R 通道在 BGR 下为第三维，需拉高）
_FAF_BGR_ORANGE_LO = np.array([220, 235, 255], dtype=np.float32)
_FAF_BGR_ORANGE_HI = np.array([40, 45, 245], dtype=np.float32)

# 融合 / 差分：signed∈[-1,1] 蓝 ← 中性 → 红（中和：以中位数为 0，避免整幅偏一端）
_FAF_BGR_RB_NEG = np.array([255, 160, 95], dtype=np.float32)
_FAF_BGR_RB_NEU = np.array([250, 250, 250], dtype=np.float32)
_FAF_BGR_RB_POS = np.array([50, 55, 245], dtype=np.float32)


def _bgr_red_blue_from_signed(s: np.ndarray) -> np.ndarray:
    """s: (H,W) float，约 [-1,1]；负偏蓝、零近白、正偏红。"""
    s = np.clip(s.astype(np.float32), -1.0, 1.0)
    out = np.empty((*s.shape, 3), dtype=np.float32)
    lo = s <= 0
    hi = ~lo
    u = np.clip(s[lo] + 1.0, 0.0, 1.0)
    b_neg = _FAF_BGR_RB_NEG.reshape(1, 3)
    b_neu = _FAF_BGR_RB_NEU.reshape(1, 3)
    b_pos = _FAF_BGR_RB_POS.reshape(1, 3)
    out[lo] = (1.0 - u)[:, np.newaxis] * b_neg + u[:, np.newaxis] * b_neu
    u = np.clip(s[hi], 0.0, 1.0)
    out[hi] = (1.0 - u)[:, np.newaxis] * b_neu + u[:, np.newaxis] * b_pos
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _faf_fusion_bgr_rb_balanced(t: torch.Tensor, batch_idx: int = 0) -> np.ndarray:
    """FAF 输出：|x| 通道均值，以中位数 + 稳健尺度做 signed，再映射蓝-白-红，减轻整幅偏橙/偏蓝。"""
    x = t[batch_idx].detach().float().cpu().numpy()
    g = np.abs(x).mean(axis=0)
    med = float(np.median(g))
    mad = float(np.percentile(np.abs(g - med), 92))
    if mad < 1e-8:
        mad = max(float(np.std(g)), 1e-8)
    s = (g - med) / mad
    s = np.tanh(s * 0.85)
    return _bgr_red_blue_from_signed(s)


def _faf_bgr_for_key(t: torch.Tensor, vis_key: str, batch_idx: int = 0) -> np.ndarray:
    g = _faf_activation_gray01(t, batch_idx)
    if vis_key == "x_mit":
        return _faf_lerp_bgr(g, _FAF_BGR_BLUE_LO, _FAF_BGR_BLUE_HI)
    if vis_key in ("x_cnn", "c_high"):
        return _faf_lerp_bgr(g, _FAF_BGR_ORANGE_LO, _FAF_BGR_ORANGE_HI)
    if vis_key == "faf_out":
        return _faf_fusion_bgr_rb_balanced(t, batch_idx)
    return _faf_fusion_bgr_rb_balanced(t, batch_idx)


def _c_high_minus_mit_dw_mean_map(
    c_high: torch.Tensor, x_mit: torch.Tensor, batch_idx: int = 0
) -> np.ndarray:
    """
    mean(c_high) - mean(Depthwise-3x3-smoothed(x_mit))。
    对 x_mit 使用 kernel=3、groups=C 等价的逐通道 avg_pool（与同尺寸 c_high 对齐）。
    返回 (H,W) float32 numpy。
    """
    ch = c_high[batch_idx : batch_idx + 1].float()
    xm = x_mit[batch_idx : batch_idx + 1].float()
    mit_dw = F.avg_pool2d(xm, kernel_size=3, stride=1, padding=1)
    a = ch.mean(dim=1, keepdim=True)
    b = mit_dw.mean(dim=1, keepdim=True)
    diff = (a - b).squeeze(0).squeeze(0)
    return diff.detach().cpu().numpy()


def _diff_map_bgr_rb(diff_hw: np.ndarray) -> np.ndarray:
    """差分标量图：中位数归零 + 稳健尺度 + tanh 压缩，蓝-白-红。"""
    med = float(np.median(diff_hw))
    mad = float(np.percentile(np.abs(diff_hw - med), 92))
    if mad < 1e-12:
        mad = max(float(np.std(diff_hw)), 1e-12)
    s = ((diff_hw - med) / mad).astype(np.float32)
    s = np.tanh(s * 0.85)
    return _bgr_red_blue_from_signed(s)


_FAF_VIS_UPSIDE = 512


def _faf_write_bgr_png_pair(out_dir: str, fname: str, bgr: np.ndarray) -> None:
    """写入原分辨率 PNG，并额外写入双线性缩放到 {size}×{size} 的 `*_512.png`。"""
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, bgr)
    side = int(_FAF_VIS_UPSIDE)
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    if h == side and w == side:
        b_up = bgr
    else:
        b_up = cv2.resize(bgr, (side, side), interpolation=cv2.INTER_LINEAR)
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = ".png"
    cv2.imwrite(os.path.join(out_dir, "{}_{}{}".format(base, side, ext)), b_up)


def _save_faf_capture_pngs(out_dir: str, faf_capture: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stage_hint = {name: st for name, st in _FAF_FF_BACKBONE_STAGES}
    for ff_name in ("ff1", "ff2", "ff3", "ff4"):
        cap = faf_capture.get(ff_name) or {}
        st = stage_hint.get(ff_name, -1)
        prefix = "{}_backbone_stage{}_".format(ff_name, st)
        mapping = [
            ("x_mit", prefix + "input_mit_branch.png"),
            ("x_cnn", prefix + "input_cnn_branch.png"),
            ("c_high", prefix + "cnn_highfreq_after_extract.png"),
            ("faf_out", prefix + "faf_output.png"),
        ]
        for key, fname in mapping:
            if key not in cap or cap[key] is None:
                continue
            bgr = _faf_bgr_for_key(cap[key], key, 0)
            _faf_write_bgr_png_pair(out_dir, fname, bgr)
        if cap.get("c_high") is not None and cap.get("x_mit") is not None:
            diff = _c_high_minus_mit_dw_mean_map(cap["c_high"], cap["x_mit"], 0)
            bgr_d = _diff_map_bgr_rb(diff)
            _faf_write_bgr_png_pair(
                out_dir, prefix + "cnn_highfreq_minus_mit_after_dw3.png", bgr_d,
            )
    meta = os.path.join(out_dir, "faf_legend.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write(
            "ff1: 融合双主干第 4 级（最粗）；ff4: 第 1 级（最细）。\n"
            "x_mit: 浅蓝→蓝（|激活| 通道均值 min-max）。\n"
            "x_cnn、c_high: 浅橙→红。\n"
            "faf_out: 蓝-白-红（以中位数为中性、稳健分位为尺度 + tanh，减轻整幅偏一色）。\n"
            "cnn_highfreq_minus_mit_after_dw3: mean(c_high) - mean(AvgPool3x3(x_mit))，"
            "AvgPool3x3 为逐通道、depthwise 等价的平滑；着色同为蓝-白-红差分。\n"
            "每张 PNG 另有同内容缩放到 {}×{} 的文件：文件名后缀 _{}.png（双线性）。\n".format(
                _FAF_VIS_UPSIDE, _FAF_VIS_UPSIDE, _FAF_VIS_UPSIDE,
            )
        )


def run_single_id_faf_vis(args, model, device, id_str: str) -> None:
    if getattr(args, "enc_opt", "") != "fused":
        raise RuntimeError("--id 单图 FAF 可视化需要 enc_opt=fused（双主干 FusedEncoder）")
    dec = getattr(model, "decoder", None)
    _faf_like = dec is not None and (
        bool(getattr(dec, "faf_like", False))
        or bool(getattr(dec, "use_FAF", False))
        or bool(getattr(dec, "use_nFAF", False))
        or bool(getattr(dec, "use_liteFAF", False))
    )
    if not _faf_like:
        raise RuntimeError("当前权重未启用 FAF/nFAF，无法导出 FAF 中间图")

    sample = _load_sample_for_single_id(args, id_str)
    input_rgb = sample["image"].unsqueeze(0).to(device)
    stem_display = _normalize_id_stem(id_str)
    if sample.get("fname"):
        stem_display = os.path.splitext(os.path.basename(str(sample["fname"])))[0]

    vis_root = getattr(args, "faf_vis_dir", None)
    if vis_root:
        vis_root = os.path.abspath(os.path.expanduser(str(vis_root)))
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
        vis_root = os.path.join(ckpt_dir, "faf_vis", _safe_dir_name(stem_display))
    os.makedirs(vis_root, exist_ok=True)

    faf_capture: dict = {}
    with torch.no_grad():
        _ = model(input_rgb, faf_capture=faf_capture)

    _save_faf_capture_pngs(vis_root, faf_capture)

    print(
        "\n[Test --id] 已保存 FAF 可视化（原分辨率 + 各图 {}×{} 版本 *_{}.png）到:\n  {}".format(
            _FAF_VIS_UPSIDE, _FAF_VIS_UPSIDE, _FAF_VIS_UPSIDE, vis_root,
        ),
        flush=True,
    )
    print(
        "  对应关系: ff1→stage4, ff2→stage3, ff3→stage2, ff4→stage1（与 FusedDecoder 一致）",
        flush=True,
    )


def _fmt_metric(x):
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return f"{xf}"
    except (TypeError, ValueError):
        return str(x)
    return f"{xf:.6f}"


def _fmt2(x) -> str:
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return "nan"
        return f"{xf:.2f}"
    except (TypeError, ValueError):
        return str(x)


def _params_mb(num_params: int) -> float:
    return int(num_params) * 4.0 / (1024.0 * 1024.0)


def _truthy_flag(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def _warn_and_fix_dmsn_for_enc_opt(args, state, verbose=True) -> None:
    """从 checkpoint 推断 DMSN；命令行显式 --use_DMSN 时以 CLI 为准（支持消融）。"""
    has_dmsn = any(k.startswith("decoder.dmsn.") for k in state)
    cli_explicit = "use_DMSN" in _argv_explicit_longopt_dests()
    if has_dmsn:
        if not cli_explicit:
            args.use_DMSN = 1
            if verbose:
                print("[Test] 从 checkpoint 推断 use_DMSN=True（含 decoder.dmsn.* 权重）")
        elif verbose:
            if _truthy_flag(getattr(args, "use_DMSN", 0)):
                print("[Test] use_DMSN=True（命令行显式指定）")
            else:
                print(
                    "[Test] use_DMSN=False（命令行消融：构建 Identity，不经过 decoder.dmsn）"
                )
    elif _truthy_flag(getattr(args, "use_DMSN", 0)) and verbose:
        print(
            "[Test] 警告: use_DMSN=True 但 checkpoint 无 decoder.dmsn.* 权重"
            "（旧单主干训练未接入 DMSN，需重新训练）"
        )


def _resolve_cuda_device(args) -> torch.device:
    """使用当前机器可用的 CUDA；ckpt 内 cuda:3 等在单卡环境自动回退 cuda:0。"""
    if not torch.cuda.is_available():
        args.device = "cpu"
        return torch.device("cpu")
    n = torch.cuda.device_count()
    raw = str(getattr(args, "device", "cuda") or "cuda")
    idx = 0
    if raw.startswith("cuda") and ":" in raw:
        try:
            idx = int(raw.split(":", 1)[1])
        except ValueError:
            idx = 0
    if idx < 0 or idx >= n:
        if idx != 0:
            print(
                "[Test] checkpoint/args device={} 不可用，回退 cuda:0（共 {} 张 GPU）".format(
                    raw, n
                ),
                flush=True,
            )
        idx = 0
    dev = torch.device("cuda:{}".format(idx))
    args.device = str(dev)
    return dev


def print_bin_bucket_metrics(m: dict) -> None:
    """若 evaluate_test 写入了 bin_top{k}_hit / bin_top1_acc，则在 Test 末尾打印。"""
    hit_keys = sorted(
        k for k in m if isinstance(k, str) and k.startswith("bin_top") and k.endswith("_hit")
    )
    if not hit_keys and "bin_top1_acc" not in m:
        return
    print("\n【Bin 档位（条件头 / DPLD / TBWH，与 train 一致）】")
    print("  说明: 在 GT≥min_height 的像素上，GT 落入的高度区间档是否出现在预测 softmax 的 top-k 档内")
    for kh in hit_keys:
        print(f"  {kh}: {_fmt_metric(m[kh])}")
    if "bin_top1_acc" in m:
        print(f"  bin_top1_acc: {_fmt_metric(m['bin_top1_acc'])}  （argmax(bin_prob)==GT-bin）")


def _split_ckpt(ckpt):
    """将 torch.load 结果拆成 state_dict 与元数据（epoch、args 等）。"""
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return state, meta
    return ckpt, {}


def _saved_args_to_dict(obj):
    """将 checkpoint 中的 args / Namespace / dict 转为普通 dict，便于打印与合并。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return vars(obj)
    return None


def _extract_first_saved_args(meta):
    """从 checkpoint 元数据取出第一份训练超参 dict（与 _print_ckpt_hyperparams 查找顺序一致）。"""
    if not isinstance(meta, dict):
        return None
    for key in ("args", "hyper_parameters", "hparams", "config"):
        if key not in meta or meta[key] is None:
            continue
        d = _saved_args_to_dict(meta[key])
        if d:
            return d
    return None


def _peek_ckpt_path_from_argv(argv=None):
    """从 argv 解析 --ckpt_path，便于在 parse_args 前加载 ckpt 并 set_defaults。"""
    if argv is None:
        argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--ckpt_path" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--ckpt_path="):
            return a.split("=", 1)[1]
    return None


def _peek_test_dir_from_argv(argv=None):
    """从 argv 解析 --test_dir，用于在合并 checkpoint 内 args 后恢复用户指定的测试数据路径。"""
    if argv is None:
        argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--test_dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--test_dir="):
            return a.split("=", 1)[1]
    return None


def _argv_has_no_ckpt_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    return "--no_ckpt_args" in argv


def _argv_explicit_longopt_dests(argv=None):
    """列出 argv 中显式出现的 `--选项名`（转为 argparse 常见 dest：`-`→`_`），不含 `-短选项`。

    用于合并 ckpt 内嵌训练 args 时跳过这些字段，避免覆盖用户在命令行传入的值
    （例如 `--condition_k 0` 应在 evaluate_test / Bin 指标中生效）。
    """
    if argv is None:
        argv = sys.argv[1:]
    out = set()
    for tok in argv:
        if tok == "--":
            break
        if not tok.startswith("--") or len(tok) <= 2:
            continue
        body = tok[2:]
        name = body.split("=", 1)[0]
        if name:
            out.add(name.replace("-", "_"))
    return out


def _argv_has_explicit_gbh_paths(argv=None):
    """命令行是否显式指定 gbh 数据根目录或 txt 划分目录。"""
    if argv is None:
        argv = sys.argv[1:]
    prefixes = (
        "--data_dir",
        "--gbh_data_dir",
        "--data_split_dirs",
        "--gbh_split_dir",
        "--test_data_split_dirs",
        "--gbh_test_split_dir",
    )
    for a in argv:
        if a in prefixes:
            return True
        if any(a.startswith(p + "=") for p in prefixes):
            return True
    return False


def _reset_gbh_paths_for_test_if_needed(args) -> None:
    """未显式传路径时，不以 checkpoint 内保存的 data_dir 为准，而从 dataset_root/dataset_name 推导 txt 位置。"""
    if getattr(args, "dataset_layout", "folder") != "gbh_txt":
        return
    if getattr(args, "_test_dir_gbh_resolved", False):
        return
    if _argv_has_explicit_gbh_paths():
        return
    args.data_dir = None
    args.gbh_data_dir = None
    args.data_split_dirs = None
    args.gbh_split_dir = None
    args.test_data_split_dirs = None
    args.gbh_test_split_dir = None
    print(
        "[Test] 数据路径按 dataset_root/dataset_name 解析："
        "<dataset_root>/<dataset_name>/data_split_dirs/{train,val}.txt，"
        "<dataset_root>/<dataset_name>/test_data_split_dirs/test.txt"
        "（未沿用 checkpoint 内 data_dir 等绝对路径）",
        flush=True,
    )


def _valid_dest_names(parser):
    return {a.dest for a in parser._actions if getattr(a, "dest", None) is not None}


def _apply_ckpt_defaults_to_parser(parser, merged):
    """将 ckpt 中的训练超参设为 argparse 默认值；命令行显式传入的项仍会覆盖。"""
    names = _valid_dest_names(parser)
    kv = {k: v for k, v in merged.items() if k in names}
    if kv:
        parser.set_defaults(**kv)
    return len(kv)


def _infer_option_from_state(state):
    """是否存在分割头权重；与训练 option='full' / 'depth' 一致。"""
    if any(k.startswith("segment_head.") for k in state):
        return "full"
    return "depth"


def _state_has_any_prefix(state, prefixes) -> bool:
    keys = tuple(state.keys())
    return any(any(k.startswith(p) for p in prefixes) for k in keys)


def _infer_skip_fusion_flags_from_state(state) -> dict:
    """从 decoder.block* 键名推断 GSFB / FDSF（mit/conv/fused 共用）。"""
    out = {}
    if _state_has_any_prefix(state, ("decoder.block1.gamma_d",)):
        out["use_FDSF"] = True
        out["use_GSFB"] = 0
    elif _state_has_any_prefix(state, ("decoder.block1.align_d",)):
        out["use_GSFB"] = 1
    return out


def _infer_fusion_flags_from_state(state) -> dict:
    """从 decoder.ff* 键名推断 FusedDecoder 融合模块类型（覆盖 ckpt args 中的错误默认值）。"""
    out = {}
    if _state_has_any_prefix(state, ("decoder.ff1.gamma", "decoder.ff1.band_refine")):
        out["use_GRFA"] = True
        return out
    if _state_has_any_prefix(state, ("decoder.ff1.gamma_l", "decoder.ff1.gamma_h")):
        out["use_liteFAF"] = True
        return out
    if _state_has_any_prefix(state, ("decoder.ff1.freq",)) and _state_has_any_prefix(
        state, ("decoder.ff1.alpha",)
    ):
        out["use_RFAF"] = True
        return out
    if _state_has_any_prefix(state, ("decoder.ff1.align_c",)):
        out["use_nFAF"] = True
        return out
    if _state_has_any_prefix(state, ("decoder.ff1.alpha", "decoder.ff1.spatial_gate")):
        out["use_FAF"] = True
    return out


def _infer_native_proj_ff_from_state(state, args=None) -> bool:
    """显式 --native_proj_ff 或 ckpt args 标记时使用 ConvTranspose ff；勿仅凭权重形状推断。"""
    if args is not None and bool(getattr(args, "native_proj_ff", False)):
        return True
    return False


def _infer_legacy_single_decoder_ff_from_state(state) -> bool:
    """dfc23-GM 旧结构：ff 为 1×1 投影到 single_decoder 宽度 + 3×3 Conv2d，无 conv3。"""
    w = state.get("decoder.ff1.conv1.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 4:
        return False
    if int(w.shape[2]) != 1 or int(w.shape[3]) != 1:
        return False
    if "decoder.ff1.conv3.0.weight" in state:
        return False
    w2 = state.get("decoder.ff1.conv2.0.weight")
    if w2 is None or not hasattr(w2, "shape") or len(w2.shape) != 4:
        return False
    if int(w2.shape[2]) != 3 or int(w2.shape[3]) != 3:
        return False
    # conv1 输出宽度 != 输入（原生→固定 decoder 宽度）
    return int(w.shape[0]) != int(w.shape[1])


def _infer_native_align_ff_from_state(state, args=None) -> bool:
    """1×1 对齐 + 3×3 Conv2d refine（含 dfc23-GM 旧 single_decoder 结构）。"""
    if _infer_native_proj_ff_from_state(state, args=args):
        return False
    if _infer_legacy_single_decoder_ff_from_state(state):
        return True
    w = state.get("decoder.ff1.conv1.0.weight")
    if w is None or not hasattr(w, "shape") or len(w.shape) != 4:
        return False
    if int(w.shape[2]) != 1 or int(w.shape[3]) != 1:
        return False
    return "decoder.ff1.conv3.0.weight" not in state


def _needs_single_decoder_channel_map(args, state) -> bool:
    if bool(getattr(args, "native_proj_ff", False)):
        return True
    if bool(getattr(args, "native_align_ff", False)) and getattr(
        args, "single_decoder_channels", None
    ):
        return True
    return _infer_legacy_single_decoder_ff_from_state(state)


def _infer_native_groupmamba_encoder_from_state(state) -> bool:
    """encoder.backbone.* 且无 adapter 权重 → 原生 stage 通道输出。"""
    keys = tuple(state.keys())
    is_gm = any("encoder.backbone.patch_embed" in k for k in keys) or any(
        "mamba_g" in k for k in keys
    )
    if not is_gm:
        return False
    has_adapters = any(
        k.startswith("encoder.adapters.") or k.startswith("encoder.mit.adapters.")
        for k in keys
    )
    return not has_adapters


def _infer_encoder_native_flags_from_state(state, args, verbose=True) -> None:
    """根据 adapter 有无与主干类型，推断 RepViT/ConvNeXt/GroupMamba 是否走原生通道。"""
    keys = tuple(state.keys())
    has_top_adapters = any(k.startswith("encoder.adapters.") for k in keys)
    has_res_adapters = any(k.startswith("encoder.resnet.adapters.") for k in keys)
    has_mit_adapters = any(k.startswith("encoder.mit.adapters.") for k in keys)

    if _infer_native_groupmamba_encoder_from_state(state):
        args.native_groupmamba_encoder = True
        if isinstance(getattr(args, "encoder1_cfg", None), dict):
            args.encoder1_cfg["native_mode"] = True
        if isinstance(getattr(args, "encoder2_cfg", None), dict):
            args.encoder2_cfg["native_mode"] = True
        if verbose:
            print("[Test] 从 checkpoint 推断 GroupMamba 原生通道（无 adapter 权重）")

    enc_opt = getattr(args, "enc_opt", "fused")
    e1 = str(getattr(args, "encoder1_type", "") or getattr(args, "encoder1", "")).lower()

    if enc_opt in ("mit", "conv") and not has_top_adapters:
        if e1 == "repvit":
            args.no_reshape = True
            if isinstance(getattr(args, "encoder1_cfg", None), dict):
                args.encoder1_cfg["native_mode"] = True
            if verbose:
                print("[Test] 从 checkpoint 推断 RepViT native_mode=True")
        elif e1 == "convnext":
            args.use_conv = max(int(getattr(args, "use_conv", 0)), 1)
            if isinstance(getattr(args, "encoder1_cfg", None), dict):
                args.encoder1_cfg["native_mode"] = True
            if verbose:
                print("[Test] 从 checkpoint 推断 ConvNeXt native_mode=True")

    if enc_opt == "fused":
        if not has_res_adapters and any(k.startswith("encoder.resnet.") for k in keys):
            e1n = str(getattr(args, "encoder1_type", "")).lower()
            if e1n == "repvit":
                args.no_reshape = True
                if isinstance(getattr(args, "encoder1_cfg", None), dict):
                    args.encoder1_cfg["native_mode"] = True
                if verbose:
                    print("[Test] 从 checkpoint 推断 fused encoder1 RepViT native_mode=True")
            elif e1n == "convnext":
                args.use_conv = max(int(getattr(args, "use_conv", 0)), 1)
                if isinstance(getattr(args, "encoder1_cfg", None), dict):
                    args.encoder1_cfg["native_mode"] = True
                if verbose:
                    print("[Test] 从 checkpoint 推断 fused encoder1 ConvNeXt native_mode=True")
        if not has_mit_adapters and any(k.startswith("encoder.mit.") for k in keys):
            if isinstance(getattr(args, "encoder2_cfg", None), dict):
                args.encoder2_cfg["native_mode"] = True
            if verbose:
                print("[Test] 从 checkpoint 推断 fused encoder2 native_mode=True")


def _apply_structure_inference_from_state(args, state, verbose=True) -> None:
    """用 state_dict 形状/键名覆盖结构相关 args（通道、融合模块、原生主干）。"""
    fusion_flags = _infer_fusion_flags_from_state(state)
    for k, v in fusion_flags.items():
        setattr(args, k, v)
        if verbose:
            print("[Test] 从 checkpoint 推断 {}={}".format(k, v))

    skip_flags = _infer_skip_fusion_flags_from_state(state)
    for k, v in skip_flags.items():
        setattr(args, k, v)
        if verbose:
            print("[Test] 从 checkpoint 推断 {}={}".format(k, v))

    if _infer_native_proj_ff_from_state(state, args=args):
        args.native_proj_ff = True
        args.native_align_ff = False
        w1 = state.get("decoder.ff1.conv1.0.weight")
        if w1 is not None and hasattr(w1, "shape"):
            args.single_decoder_channels = [
                int(state[f"decoder.ff{i}.conv1.0.weight"].shape[0]) for i in range(1, 5)
            ]
        if verbose:
            print(
                "[Test] 从 checkpoint 推断 native_proj_ff=True（decoder ff: 1×1 proj + ConvTranspose）"
            )
    elif _infer_native_align_ff_from_state(state, args=args):
        args.native_align_ff = True
        args.native_proj_ff = False
        if _infer_legacy_single_decoder_ff_from_state(state):
            w1 = state.get("decoder.ff1.conv1.0.weight")
            if w1 is not None and hasattr(w1, "shape"):
                args.single_decoder_channels = [
                    int(state[f"decoder.ff{i}.conv1.0.weight"].shape[0]) for i in range(1, 5)
                ]
        if verbose:
            msg = "[Test] 从 checkpoint 推断 native_align_ff=True（decoder ff: 1×1+3×3 Conv2d）"
            if _infer_legacy_single_decoder_ff_from_state(state):
                msg += "，single_decoder 旧 GM 结构"
            print(msg)

    _infer_encoder_native_flags_from_state(state, args, verbose=verbose)
    _warn_and_fix_dmsn_for_enc_opt(args, state, verbose=verbose)


def _apply_decoder_cfg_to_args(args, state, decoder_cfg) -> None:
    """将 _infer_decoder_cfg_from_state 的结果写回 args；single_decoder ff 需单独映射通道语义。"""
    if decoder_cfg is None:
        return
    if _needs_single_decoder_channel_map(args, state):
        if _infer_native_proj_ff_from_state(state, args=args):
            args.native_proj_ff = True
            args.native_align_ff = False
        elif _infer_legacy_single_decoder_ff_from_state(state):
            args.native_align_ff = True
            args.native_proj_ff = False
        # ff: in=原生 encoder 通道(out_channel_ffb)，out=single_decoder_channels
        args.out_channel_ffb = list(decoder_cfg["in_channel_ffb"])
        args.single_decoder_channels = list(decoder_cfg["out_channel_ffb"])
        args.in_channel_decoderb = list(decoder_cfg["in_channel_decoderb"])
        args.out_channel_decoderb = list(decoder_cfg["out_channel_decoderb"])
        args.out_decoder = decoder_cfg["out_decoder"]
    else:
        for k, v in decoder_cfg.items():
            setattr(args, k, v)


def _print_decoder_cfg_inference(args, verbose=True) -> None:
    if not verbose:
        return
    if getattr(args, "native_proj_ff", False) or (
        getattr(args, "native_align_ff", False)
        and getattr(args, "single_decoder_channels", None)
    ):
        ff_tag = "native_proj_ff" if getattr(args, "native_proj_ff", False) else "native_align_ff"
        print(
            "[Test] 从 checkpoint 推断 decoder 通道 ({}): "
            "out_channel_ffb(原生)={}, single_decoder_channels={}, "
            "in_channel_decoderb={}, out_channel_decoderb={}, out_decoder={}".format(
                ff_tag,
                args.out_channel_ffb,
                args.single_decoder_channels,
                args.in_channel_decoderb,
                args.out_channel_decoderb,
                args.out_decoder,
            )
        )
    else:
        print(
            "[Test] 从 checkpoint 推断 decoder 通道: "
            "in_channel_ffb={}, out_channel_ffb={}, in_channel_decoderb={}, "
            "out_channel_decoderb={}, out_decoder={}".format(
                args.in_channel_ffb,
                args.out_channel_ffb,
                args.in_channel_decoderb,
                args.out_channel_decoderb,
                args.out_decoder,
            )
        )


def prepare_fhe_args_from_checkpoint(args, state, ckpt_dict=None, verbose=True):
    """
    合并 ckpt 内 args（若传入 ckpt_dict）、从 state 推断结构，并 resolve_encoder_args；
    再回写 decoder 通道（与 test main 一致）。供 test.py / all_test.py 共用。
    """
    if ckpt_dict is not None and isinstance(ckpt_dict, dict) and ckpt_dict.get("args") is not None:
        d = _saved_args_to_dict(ckpt_dict["args"])
        if d:
            skip_from_argv = _argv_explicit_longopt_dests()
            for k, v in d.items():
                if k in skip_from_argv:
                    continue
                setattr(args, k, v)

    decoder_cfg = _infer_decoder_cfg_from_state(state)

    if not getattr(args, "use_condition", False) and any(k.startswith("condition_heads.") for k in state):
        args.use_condition = True
        if verbose:
            print("[Test] 从 checkpoint 推断 use_condition=True")

    if _infer_adabin_from_state(state):
        args.use_adabin = True
        if verbose:
            print("[Test] 从 checkpoint 推断 use_adabin=True")

    if getattr(args, "use_condition", False):
        bin_num_ckpt = _infer_bin_num_from_state(state)
        bin_len_ckpt = _infer_bin_len_from_state(state)
        if bin_num_ckpt is not None:
            args.bin_num = int(bin_num_ckpt)
        if bin_len_ckpt is not None and not getattr(args, "use_adabin", False):
            args.bin_len = float(bin_len_ckpt)
        if verbose:
            print(
                "[Test] 条件头配置: bin_num={}, bin_len={}, adabin={}".format(
                    getattr(args, "bin_num", None),
                    getattr(args, "bin_len", None),
                    getattr(args, "use_adabin", False),
                )
            )

    # TBWH / DPLD：纯 state_dict 时 args 可能没有 use_TBWH，须与权重前缀一致否则会 load 失败或 bin 指标用错 yaml
    sk_tuple = tuple(state.keys())
    has_tbwh = any(k.startswith("tbwh_head.") for k in sk_tuple)
    has_dpld = any(k.startswith("dpld_head.") for k in sk_tuple) or any(
        k.startswith("hl_tail.dpld_head.") for k in sk_tuple
    )
    if has_tbwh:
        setattr(args, "use_TBWH", 1)
        setattr(args, "use_DPLD", 0)
        if verbose:
            print("[Test] 从 checkpoint 推断 height head = TBWH (tbwh_head.*)")
    elif has_dpld:
        setattr(args, "use_TBWH", 0)
        di = getattr(args, "use_DPLD", 0)
        if isinstance(di, bool):
            di = 1 if di else 0
        else:
            try:
                di = int(di)
            except (TypeError, ValueError):
                di = 0
        if di not in (1, 2):
            setattr(args, "use_DPLD", 1)
        if verbose:
            print("[Test] 从 checkpoint 推断 height head = DPLD (dpld_head.* / hl_tail.dpld_head.*)")

    _apply_structure_inference_from_state(args, state, verbose=verbose)

    if decoder_cfg is not None:
        _apply_decoder_cfg_to_args(args, state, decoder_cfg)
        _print_decoder_cfg_inference(args, verbose=verbose)

    state_keys = list(state.keys())
    # EfficientVMamba：outnorm 在 backbone 上（encoder.mit.backbone.outnorm0）；与 MambaVision 的 encoder.mit.backbone.patch_embed 共存，须先判 EVM
    ckpt_is_evm = any("encoder.mit.backbone.outnorm0" in k for k in state_keys)
    # GroupMamba：GroupMambaLayer 内含 mamba_g1…（勿与 MambaVision 的 patch_embed 推断混淆）
    ckpt_is_gm = any("mamba_g1" in k for k in state_keys) and any(
        "encoder.mit.backbone" in k for k in state_keys
    )
    ckpt_is_or = any(
        "encoder.mit.patch_embed1" in k or "encoder.mit.patch_embed2" in k for k in state_keys
    )
    ckpt_is_fused_mv = any("encoder.mit.backbone.patch_embed" in k for k in state_keys) and not ckpt_is_evm

    if ckpt_is_evm:
        args.encoder2 = "EfficientVMamba"
        if verbose:
            print("[Test] 检测到 EfficientVMamba（encoder.mit.backbone.outnorm*），encoder2=EfficientVMamba")
    elif ckpt_is_gm:
        args.encoder2 = "GroupMamba"
        if verbose:
            print("[Test] 检测到 GroupMamba（encoder.mit.backbone.*mamba_g*），encoder2=GroupMamba")
    elif ckpt_is_or and not ckpt_is_fused_mv:
        args.use_mv = False
        if verbose:
            print("[Test] 检测到 FusedHE-main-or 权重，使用 MiT 主干匹配结构")
    elif ckpt_is_fused_mv:
        args.use_mv = True
        has_outnorms = any("encoder.mit.outnorms." in k for k in state_keys)
        if has_outnorms:
            args.mv_variant = "tiny_ade20k"
            if verbose:
                print("[Test] 检测到 MambaVision-Tiny-ADE20K 权重")
        else:
            args.mv_variant = "base_1k"
            if verbose:
                print("[Test] 检测到 MambaVision-Base-1K 权重")
        if not getattr(args, "no_reshape", False) and _infer_no_reshape_from_state(state):
            args.no_reshape = True
            if verbose:
                print("[Test] 从 checkpoint 推断 no_reshape=True（MambaVision 原生通道 + 配套 decoder）")

    inferred_conv = _infer_use_conv_from_state(state)
    if inferred_conv > 0 and getattr(args, "use_conv", 0) == 0:
        args.use_conv = inferred_conv
        if verbose:
            print(
                "[Test] 从 checkpoint 推断 use_conv={}（ConvNeXt V2 {}）".format(
                    inferred_conv,
                    "原生通道" if inferred_conv == 1 else "适配到 ResNet 通道",
                )
            )

    if getattr(args, "_encoder_resolved", False):
        delattr(args, "_encoder_resolved")
    resolve_encoder_args(args)
    _apply_decoder_cfg_to_args(args, state, decoder_cfg)


def _print_faf_learned_coefficients(model, args) -> None:
    """
    在 FusedDecoder 使用频率融合模块时，打印各 ff* 中决定频率分支强度的参数。
    GRFA 使用逐通道 gamma 控制 gate * edge_residual；旧 FAFBlock 使用 α、β、λ_res。
    """
    if getattr(args, "enc_opt", "fused") != "fused":
        return
    dec = getattr(model, "decoder", None)
    if dec is None or not (
        bool(getattr(dec, "faf_like", False))
        or bool(getattr(dec, "use_FAF", False))
        or bool(getattr(dec, "use_nFAF", False))
        or bool(getattr(dec, "use_liteFAF", False))
    ):
        return
    # ff1 融合 stage4 … ff4 融合 stage1（与 FAFBlock.stage_idx 一致）
    _ff_to_mit_stage = {"ff1": 4, "ff2": 3, "ff3": 2, "ff4": 1}
    modules = [(name, getattr(dec, name, None)) for name in ("ff1", "ff2", "ff3", "ff4")]
    has_grfa_gamma = any(m is not None and hasattr(m, "gamma") for _, m in modules)
    if has_grfa_gamma:
        print("\n--- [Test] GRFA gamma（来自 checkpoint 中训练后的频率残差强度）---")
        print("  说明: out = base + gamma * (edge_gate * edge_residual)；gamma 若全为 0，则频率分支对输出无贡献。")
        print("  nonzero_ratio 使用阈值 1e-8；abs_mean 越大，频率残差平均缩放越明显。")
    else:
        learnable = bool(getattr(args, "faf_learnable_coef", True))
        if learnable:
            title = "--- [Test] FAF 系数（可学习：来自 checkpoint 中训练后的参数）---"
            hint = (
                "  faf_learnable_coef=True：alpha/beta/lambda_res 为 nn.Parameter，"
                "下面数值以加载权重后为准。"
            )
        else:
            title = "--- [Test] FAF 系数（不可学习：固定层级先验，与 FAFBlock._DEFAULT_STAGE_COEF 一致）---"
            hint = (
                "  faf_learnable_coef=False：alpha/beta/lambda_res 为 register_buffer，训练不更新；"
                "若与下表一致，即模型仍使用 FAFBlock.py 中 stage_idx 的默认 (α,β) 与默认 λ=0.5。"
            )
        print("\n" + title)
        print("  当前 faf_learnable_coef: {}".format(learnable))
        print(
            "  说明: alpha、beta 缩放门控后的高频/低频项，lambda_res 缩放 (c_align+g_align) 残差；"
            "α/β_归一 = alpha/(α+β)、beta/(α+β)（与 λ 无关）。"
        )
        print(hint)
    for name, m in modules:
        if m is None:
            continue
        st = int(getattr(m, "stage_idx", -1))
        mit_st = _ff_to_mit_stage.get(name, st)
        if hasattr(m, "gamma"):
            g = m.gamma.detach().float().cpu().view(-1)
            nz = float((g.abs() > 1e-8).float().mean().item())
            print(
                "  decoder.{}  (融合第 {} 级 stage, GRFA stage_idx={}): "
                "gamma_mean={:.8f}, abs_mean={:.8f}, std={:.8f}, min={:.8f}, max={:.8f}, nonzero_ratio={:.2%}".format(
                    name,
                    mit_st,
                    st,
                    float(g.mean().item()),
                    float(g.abs().mean().item()),
                    float(g.std(unbiased=False).item()),
                    float(g.min().item()),
                    float(g.max().item()),
                    nz,
                )
            )
            continue
        try:
            a = float(m.alpha.detach().cpu().item())
            b = float(m.beta.detach().cpu().item())
            lam = float(m.lambda_res.detach().cpu().item())
        except Exception as e:
            print("  {}: 无法读取系数 ({})".format(name, e))
            continue
        s = a + b
        if s > 1e-12:
            ra, rb = a / s, b / s
        else:
            ra, rb = float("nan"), float("nan")
        print(
            "  decoder.{}  (融合第 {} 级 stage, FAF stage_idx={}): "
            "alpha={:.6f}, beta={:.6f}, lambda_res={:.6f}  |  α/β 归一: {:.6f}, {:.6f}".format(
                name, mit_st, st, a, b, lam, ra, rb,
            )
        )


def _collect_rghf_modules(model) -> list:
    """收集 encoder 上各 stage 的 RGHF 模块（单主干或 fused 的 mit/resnet 支路）。"""
    out: list = []
    enc = getattr(model, "encoder", None)
    if enc is None:
        return out

    def _append(prefix: str, rghf_list) -> None:
        if rghf_list is None:
            return
        for i, m in enumerate(rghf_list):
            if m is not None and hasattr(m, "gamma"):
                out.append((prefix, i, m))

    _append("encoder.rghf", getattr(enc, "rghf", None))
    for branch in ("mit", "resnet"):
        sub = getattr(enc, branch, None)
        if sub is not None:
            _append("encoder.{}.rghf".format(branch), getattr(sub, "rghf", None))
    return out


def _print_rghf_learned_gamma(model, args) -> None:
    """打印 GroupMamba 编码器各 stage RGHF 的逐通道 gamma。"""
    modules = _collect_rghf_modules(model)
    if not modules:
        return
    print("\n--- [Test] RGHF gamma（来自 checkpoint 中训练后的频率残差强度）---")
    print("  说明: out = x + freq_lambda * gamma * (edge_gate * edge_residual)；")
    print("        gamma 若全为 0，则频率分支对 stage 输出无贡献。")
    print("  nonzero_ratio 使用阈值 1e-8；abs_mean 越大，频率残差平均缩放越明显。")
    for prefix, idx, m in modules:
        st = int(getattr(m, "stage_idx", idx + 1))
        d = int(getattr(m, "d", m.gamma.numel()))
        g = m.gamma.detach().float().cpu().view(-1)
        nz = float((g.abs() > 1e-8).float().mean().item())
        print(
            "  {}.{}  (encoder stage {}, d={}): "
            "gamma_mean={:.8f}, abs_mean={:.8f}, std={:.8f}, min={:.8f}, max={:.8f}, nonzero_ratio={:.2%}".format(
                prefix,
                idx,
                st,
                d,
                float(g.mean().item()),
                float(g.abs().mean().item()),
                float(g.std(unbiased=False).item()),
                float(g.min().item()),
                float(g.max().item()),
                nz,
            )
        )


def _print_ckpt_hyperparams(meta):
    """打印权重文件中除 state_dict 外的字段及嵌入的训练超参。"""
    print("\n--- [Test] 权重文件中的超参 / 元数据 ---")
    if not meta:
        print("  （无）当前文件为纯 state_dict，或旧版训练未嵌入 args/epoch。")
        print("  结构相关标志仍由下方「从 state_dict 推断」逻辑补齐。")
        return
    # 其它元数据（非 args）
    other_keys = [k for k in meta.keys() if k not in ("args", "hyper_parameters", "hparams", "config")]
    for k in sorted(other_keys):
        v = meta[k]
        try:
            s = repr(v)
            if len(s) > 120:
                s = s[:117] + "..."
        except Exception:
            s = f"<{type(v).__name__}>"
        print(f"  {k}: {s}")
    for key in ("args", "hyper_parameters", "hparams", "config"):
        if key not in meta or meta[key] is None:
            continue
        d = _saved_args_to_dict(meta[key])
        if not d:
            continue
        print(f"\n  [{key}]（训练时保存，共 {len(d)} 项）")
        for name in sorted(d.keys()):
            val = d[name]
            try:
                line = repr(val)
                if len(line) > 100:
                    line = line[:97] + "..."
            except Exception:
                line = f"<{type(val).__name__}>"
            print(f"    {name}: {line}")


def _apply_saved_args_to_namespace(args, saved_dict):
    """将保存的 dict 合并进 argparse Namespace，仅覆盖已存在的属性名。"""
    if not saved_dict:
        return 0
    n = 0
    for k, v in saved_dict.items():
        if hasattr(args, k):
            setattr(args, k, v)
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser('FusedHE test (original)', parents=[get_args_parser()])
    parser.add_argument('--ckpt_path', required=True, type=str, help='权重路径，如 output/xxx/best.ckpt')
    parser.add_argument('--test_dir', default=None, type=str,
                        help='folder 布局：test 目录（含 rgb/、dsm/）。gbh_txt：可传 test.txt 路径，如 '
                        '.../htcdc_dfc23/test_data_split_dirs/test.txt，将自动以 htcdc_dfc23 为 data_dir；'
                        '或传含 test.txt 的目录。不传则 dataset_root/dataset_name/test 或 gbh 默认 txt')
    parser.add_argument(
        '--no_mask',
        default=0.0,
        type=float,
        metavar='M',
        help='默认 0：用数据集 building_mask（若有）计算 RMSE-M/NM/B、IoU。'
        '设为正数（如 1 或 5）则忽略数据集 mask，改为用 GT nDSM（米）>M 临时生成建筑掩码。'
        'B10/B0 始终优先用 mask/*_BLG.tif（若存在）；M10/M0 始终用 GT>M。',
    )
    parser.add_argument('--verbose_mask_audit', action='store_true',
                        help='在核对报告中打印每个 batch 的 GT>阈值 比例与 GT 范围')
    parser.add_argument('--skip_flops', action='store_true',
                        help='跳过 FLOPs 统计（thop/profiler 较慢）')
    parser.add_argument('--flops_height', default=512, type=int,
                        help='FLOPs 统计用的输入高，默认与训练/推理一致')
    parser.add_argument('--flops_width', default=512, type=int,
                        help='FLOPs 统计用的输入宽')
    parser.add_argument('--no_ckpt_args', action='store_true',
                        help='不从 checkpoint 元数据注入训练超参（仅用命令行 + 权重形状/键名推断）')
    parser.add_argument(
        '--id',
        dest='test_image_id',
        default=None,
        type=str,
        help='单图 FAF 可视化：样本 stem（如 SV_Berlin_52.5070_13.4974_IMG，可不带 .tif）。'
        'gbh_txt：在 checkpoint 对应的 data_dir（dataset_root/dataset_name）下按 image/ndsm 全库解析，无需在 test.txt 中；'
        'folder：在 train/val/test 等含 rgb/ 的子目录中扫描。',
    )
    parser.add_argument(
        '--faf_vis_dir',
        default=None,
        type=str,
        help='--id 模式下 FAF PNG 输出目录；默认 <ckpt 所在目录>/faf_vis/<stem>/',
    )
    parser.add_argument(
        '--mask_test',
        default=0,
        type=int,
        choices=[0, 1, 2],
        help='高度图掩码测试模式。0: 默认不做处理；1: 使用分割副产物，'
             '将分割结果覆盖在高度输出图上（高度>1的区域设为1，否则保留）；'
             '2: 不使用分割副产物，直接对高度图设阈值（<1设为0）',
    )
    parser.add_argument(
        '--all_test_vis',
        default=0,
        type=int,
        metavar='N',
        help='随机抽取 N 张测试样本进行可视化保存（0 表示不启用，默认测试全部不保存可视化）',
    )
    parser.add_argument(
        '--vis_output_dir',
        default=None,
        type=str,
        help='全量可视化输出目录；默认 <ckpt 所在目录>/all_test_vis/',
    )
    parser.add_argument(
        '--freq_lambda',
        default=1.0,
        type=float,
        help='GRFA 推理时临时放大频率残差：out = base + λ * gamma * M * e（默认 1.0，仅 use_GRFA 生效）',
    )

    state = None
    ckpt_meta = {}
    ckpt_pre = None
    path_for_ckpt = _peek_ckpt_path_from_argv()
    if path_for_ckpt and os.path.isfile(path_for_ckpt) and not _argv_has_no_ckpt_args():
        ckpt_pre = torch.load(path_for_ckpt, map_location='cpu')
        state, ckpt_meta = _split_ckpt(ckpt_pre)
        merged = _extract_first_saved_args(ckpt_meta)
        if merged:
            n = _apply_ckpt_defaults_to_parser(parser, merged)
            print(f"\n[Test] 已从 checkpoint 元数据注入 {n} 项默认超参（命令行参数仍可覆盖）")

    args = parser.parse_args()
    apply_test_dir_arg_for_gbh_txt(args)
    _reset_gbh_paths_for_test_if_needed(args)

    if getattr(args, 'apply_ckpt_args', False):
        print("[Test] 提示: --apply_ckpt_args 已弃用，默认即会合并 ckpt 内训练超参。")

    if state is None:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        state, ckpt_meta = _split_ckpt(ckpt)
        ckpt_full = ckpt
    else:
        ckpt_full = ckpt_pre

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    _print_ckpt_hyperparams(ckpt_meta)

    merge_ckpt = (
        ckpt_full
        if (not getattr(args, "no_ckpt_args", False) and ckpt_full is not None)
        else None
    )
    prepare_fhe_args_from_checkpoint(args, state, ckpt_dict=merge_ckpt, verbose=True)

    # prepare_fhe_args_from_checkpoint 合并 ckpt 内 args 时会跳过命令行已显式传入的选项；
    # 若仍传了 --test_dir，在此再次套用路径解析。
    _cli_test_dir = _peek_test_dir_from_argv()
    if _cli_test_dir is not None:
        args.test_dir = os.path.abspath(os.path.expanduser(_cli_test_dir))
        apply_test_dir_arg_for_gbh_txt(args)
        _reset_gbh_paths_for_test_if_needed(args)
        print(
            "[Test] 已按命令行 --test_dir 覆盖 checkpoint 中的数据集路径（避免仍用训练时的 dataset_name）",
            flush=True,
        )

    opt_inf = _infer_option_from_state(state)
    if getattr(args, 'option', 'full') != opt_inf:
        print(f"[Test] 从 checkpoint 推断 option={opt_inf}（原 args.option={getattr(args, 'option', None)}）")
        args.option = opt_inf

    maybe_apply_ndsm_pickle_max_height(args)
    test_data = build_test_dataset(args)
    test_ref = resolve_test_data_path(args)

    # 处理 --all_test_vis 随机抽样
    all_test_vis_n = int(getattr(args, 'all_test_vis', 0))
    vis_output_dir = None
    if all_test_vis_n > 0:
        n_test_full = len(test_data)
        if all_test_vis_n >= n_test_full:
            # 如果指定数量大于等于总样本数，则测试全部并保存可视化
            print(f"[Test] --all_test_vis={all_test_vis_n} 大于等于总样本数 {n_test_full}，将测试全部样本并保存可视化")
        else:
            # 随机抽取指定数量的样本
            import torch.utils.data as torch_data
            indices = list(range(n_test_full))
            random.shuffle(indices)
            selected_indices = indices[:all_test_vis_n]
            test_data = torch_data.Subset(test_data, selected_indices)
            print(f"[Test] 已从 {n_test_full} 张测试样本中随机抽取 {all_test_vis_n} 张进行测试和可视化")
        # 设置可视化输出目录
        if getattr(args, 'vis_output_dir', None) is not None:
            vis_output_dir = args.vis_output_dir
        else:
            ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
            vis_output_dir = os.path.join(ckpt_dir, f'val_vis_{all_test_vis_n}')
        os.makedirs(vis_output_dir, exist_ok=True)
        print(f"[Test] 可视化将保存到: {vis_output_dir}")

    # num_workers=0、pin_memory=False：在进程/线程数接近 RLIMIT_NPROC 时避免 can't start new thread
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    n_test = len(test_data)
    print(f"[Test] 加载 test 集: {test_ref}, 样本数: {n_test}")

    device = _resolve_cuda_device(args)

    model = FHE(args)
    try:
        load_ok = model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        # PyTorch 的 strict=False 仍会对 size mismatch 抛异常，
        # 因此在不匹配时只保留 shape 完全一致的参数再加载。
        print(f"[Test] load_state_dict 发生参数形状不匹配，开始过滤后重试：{e}")
        model_state = model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in model_state and hasattr(v, 'shape') and hasattr(model_state[k], 'shape'):
                if tuple(model_state[k].shape) == tuple(v.shape):
                    filtered[k] = v
        load_ok = model.load_state_dict(filtered, strict=False)

    if load_ok.missing_keys:
        print(f"[Test] 未加载的 missing keys: {load_ok.missing_keys[:8]}...")
    if load_ok.unexpected_keys:
        print(f"[Test] 未使用的 unexpected keys: {load_ok.unexpected_keys[:8]}...")
    model.to(device)
    model.eval()
    e1 = getattr(args, "encoder1_type", "res")
    e2 = getattr(args, "encoder2_type", "mit")
    if getattr(args, "enc_opt", "fused") == "fused":
        print(f"[Test] 主干: encoder1={e1} + encoder2={e2}")
    else:
        print(f"[Test] 单编码器 enc_opt={getattr(args, 'enc_opt', '')}，主干 encoder1={e1}")
    print(f"[Test] 已加载权重: {args.ckpt_path}")

    freq_lambda = float(getattr(args, "freq_lambda", 1.0))
    if bool(getattr(getattr(model, "decoder", None), "use_GRFA", False)):
        set_freq_lambda(model, freq_lambda)
        if freq_lambda != 1.0:
            print(
                "[Test] GRFA 推理频率残差放大: freq_lambda={:.4f}（out = base + λ * gamma * M * e）".format(
                    freq_lambda,
                ),
                flush=True,
            )
    if _collect_rghf_modules(model):
        set_rghf_freq_lambda(model, freq_lambda)
        if freq_lambda != 1.0:
            print(
                "[Test] RGHF 推理频率残差放大: freq_lambda={:.4f}（out = x + λ * gamma * M * e）".format(
                    freq_lambda,
                ),
                flush=True,
            )

    _print_faf_learned_coefficients(model, args)
    _print_rghf_learned_gamma(model, args)

    if getattr(args, "test_image_id", None):
        run_single_id_faf_vis(args, model, device, args.test_image_id)
        print(
            "\n[Test] 已使用 --id 单图 FAF 可视化模式，已跳过全 test 集评估与 FLOPs。",
            flush=True,
        )
        print("\n--- Test 完成 ---")
        return

    print("\n--- Test 集评估（米制，nqp_utils 统一指标）---")
    if float(getattr(args, 'no_mask', 0.0)) > 0:
        print(
            "[Test] --no_mask={}：RMSE-M/NM/B、IoU 用 GT nDSM（米）>{}；"
            "B10/B0 仍用 mask/*_BLG.tif（若有）；M10/M0 用 GT>{} m".format(
                getattr(args, 'no_mask', 0.0),
                getattr(args, 'no_mask', 0.0),
                getattr(args, 'no_mask', 0.0),
                getattr(args, 'no_mask', 0.0),
            ),
            flush=True,
        )
    mask_test_mode = int(getattr(args, 'mask_test', 0))
    if mask_test_mode == 1:
        print(
            "[Test] --mask_test=1：使用分割副产物处理高度图，"
            "建筑区域（分割>0.5）内高度>1的部分设为1，其他保留原值",
            flush=True,
        )
    elif mask_test_mode == 2:
        print(
            "[Test] --mask_test=2：不使用分割副产物，直接对高度图设阈值，"
            "低于1的设为0",
            flush=True,
        )

    results, _height_thresholds, mask_audit, _pm = evaluate_test(test_loader, model, device, args, vis_output_dir=vis_output_dir)
    m = results

    if getattr(args, "verbose_mask_audit", False):
        th_m = mask_audit["mask_depth_threshold_m"]
        print("\n【前景掩码核对】")
        print(f"  mask_depth_threshold = {th_m} m")
        print(f"  batch 数: {mask_audit['n_batches']}")
        if mask_audit["rows"]:
            for i, r in enumerate(mask_audit["rows"]):
                print(
                    f"    batch {i}: GT>{th_m}m 占比={r['frac_ge2']:.4f}, "
                    f"GT∈[{r['gt_min_m']:.2f},{r['gt_max_m']:.2f}]m"
                )

    print()
    print(f"  RMSE:      {_fmt_metric(m['rmse'])}")
    print(f"  MAE:       {_fmt_metric(m.get('mae', float('nan')))}")
    print(f"  RMSE-M:    {_fmt_metric(m['rmse_m'])}")
    print(f"  RMSE-NM:   {_fmt_metric(m['rmse_nm'])}")
    print(f"  RMSE-B:    {_fmt_metric(m['rmse_b'])}")
    print(f"  δ1:        {m['d1']:.4f}")
    print(f"  δ2:        {m['d2']:.4f}")
    print(f"  δ3:        {m['d3']:.4f}")
    print(f"  B10:       {_fmt_metric(m.get('b10', float('nan')))}")
    print(f"  M10:       {_fmt_metric(m.get('m10', float('nan')))}")
    print(f"  IoU:       {m['iou']:.4f}")

    total_params = nqp_count_parameters(model)
    fh, fw = int(args.flops_height), int(args.flops_width)
    if getattr(args, "skip_flops", False):
        gflops_val = float("nan")
    else:
        prof = profile_model(model, fh, fw, inner=model)
        gflops_val = prof["gflops"]
        if gflops_val != gflops_val:
            gflops_val = float("nan")
    print(f"  GFLOPs:      {_fmt2(gflops_val)}")
    print(f"  Params (MB): {_fmt2(_params_mb(total_params))}")

    print("\n--- Test 完成 ---")


if __name__ == '__main__':
    main()
