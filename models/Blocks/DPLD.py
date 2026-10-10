"""
DPLD —— Dataset-Prior Local Distribution Head.

定位
----
替代原来 `--use_condition` 那一套「固定 N 档 × bin_len + 全局加权 + 残差」的高度头。
旧设计的痛点：
  - 远端 bin 的小概率泄漏会污染低高度像素（2m 像素出现 50m bin 1% 概率 ≈ 0.5m 误差）
  - bin 太少且固定米宽，δ 指标在低高度处不稳
  - 全局残差几乎只为整张图加一个偏置，作用十分有限

DPLD 的核心
-----------
1. **数据集先验 binEdge**：在训练前对训练集高度分布做一次 quantile 划分，
   生成 n_bins 个非均匀 bin。低高度密集区 bin 更密、长尾区 bin 更宽。
2. **单独的 ground bin**：把 [0, ground_bin_max_m]（默认 [0, 1m]）单独占用 bin 0，
   避免大量 0m 像素让 quantile 在 0 附近产生重复边界（user 自己提到的坑）。
3. **Top-k 加权**：取概率最大的 top-k 个 bin（默认 10），重新归一化后加权 bin center。
   远端低概率直接被丢弃，根治"远端污染"。
4. **高斯软标签分布约束**：以 GT 米制高度为中心、在 bin **米制 center** 上构造高斯软标签 q，
   用 KL(q || p) 让 p 在 GT 附近呈连续平滑响应，但又不强制 one-hot。
5. **不再有残差**：用 128 个密集 bin + 非均匀 binEdge + Top-k 后，残差冗余且可能引入噪声。

接口
----
forward(x, max_height) ->
    pred_d   (B, 1, H, W)  归一化高度 [0,1]
    bin_logits  (B, n_bins, H, W)
    bin_prob    (B, n_bins, H, W)
本模块同时注册 buffer：
    bin_edges_m   (n_bins+1,)  米制 bin edges
    bin_centers_m (n_bins,)    米制 bin centers
推理时不需要任何额外参数（buffer 随 state_dict 持久化）。

bin edges 来源
--------------
- 默认（``--use_DPLD`` 1）：从 ``args.data_dir/ndsm_stats.pickle`` 里的 ``count`` 直方图
  （HTC pickle 第 5 项）反推 quantile bin edges；并缓存到 ``args.data_dir/<cache_filename>``，再次启动可直接复用。
- ``--use_DPLD`` 2：每次启动按 ``train.txt``+``val.txt`` 遍历全部 nDSM 在线累积 ``count``
  （与 pickle 同源算法），再做 quantile；不读 pickle/磁盘 bin 缓存、不写 bin 缓存文件。
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


# ============================================================================
#  bin edges 的构建 / 缓存
# ============================================================================
def _bin_prior_from_count_hist(count: np.ndarray, edges_m: np.ndarray) -> np.ndarray:
    """根据 1m 间隔频次直方图与米制 bin edges 计算每个 bin 的归一化先验概率。

    count[i] = floor(h)=i 米的像素数（来自 ndsm_stats.pickle 第 5 项）。
    返回 shape (n_bins,) 的概率，sum=1。若 count 累计为 0 则退化为均匀分布。
    """
    n_bins = int(edges_m.size - 1)
    bin_probs = np.zeros(n_bins, dtype=np.float64)
    n_buckets = int(count.size)
    for i in range(n_bins):
        lo_m = float(edges_m[i])
        hi_m = float(edges_m[i + 1])
        lo_i = max(0, int(np.floor(lo_m)))
        hi_i = min(n_buckets, int(np.ceil(hi_m)))
        for j in range(lo_i, hi_i):
            ov = max(0.0, min(float(j + 1), hi_m) - max(float(j), lo_m))
            if ov > 0:
                bin_probs[i] += float(count[j]) * ov
    total = bin_probs.sum()
    if total <= 0:
        return np.ones(n_bins, dtype=np.float64) / max(n_bins, 1)
    return bin_probs / total


def _quantile_edges_from_count_hist(
    count: np.ndarray,
    n_inner_bins: int,
    lo_m: float,
    hi_m: float,
) -> np.ndarray:
    """根据 1m 间隔的频次直方图 count 在 [lo_m, hi_m] 上求 quantile 边界。

    count[i] 表示 floor(h)=i m 时的像素数（HTC ndsm_stats.pickle 的第 5 项口径）。
    返回长度 = n_inner_bins + 1 的米制边界，首项 = lo_m，末项 = hi_m。
    若 hi 处 count 为 0，则尾部边界会被 clamp 到 hi_m。
    """
    if hi_m <= lo_m:
        raise ValueError("hi_m={} 必须 > lo_m={}".format(hi_m, lo_m))
    if n_inner_bins < 1:
        raise ValueError("n_inner_bins={} 必须 >= 1".format(n_inner_bins))

    # cumsum 与对应右边界（米）
    n = int(np.ceil(hi_m))
    if count.size < n:
        count = np.pad(count, (0, n - count.size))
    count = np.asarray(count[: max(n, 1)], dtype=np.float64).copy()

    # 把 < lo_m 的桶清零（统一交给 ground bin）
    lo_i = int(np.floor(lo_m))
    if lo_i > 0:
        count[: min(lo_i, count.size)] = 0.0

    # 累加到 [lo, hi] 的总像素数；保险起见加 ε
    total = count.sum()
    if total <= 0:
        # 完全没有 >= lo_m 的像素，则退化为均匀
        return np.linspace(lo_m, hi_m, n_inner_bins + 1, dtype=np.float64)

    # 累计分布在「米」上的近似 inverse-CDF：count[i] 看作 (i, i+1] 区间内均匀分布
    cum = np.cumsum(count) / total
    # 我们在均匀网格上插值 inverse-CDF
    grid_m = np.arange(1, count.size + 1, dtype=np.float64)  # 区间右边界
    quantiles = np.linspace(0.0, 1.0, n_inner_bins + 1)
    edges = np.interp(quantiles, np.concatenate(([0.0], cum)), np.concatenate(([lo_m], grid_m)))

    # 强制首尾、单调
    edges[0] = lo_m
    edges[-1] = max(float(edges[-1]), float(hi_m))
    edges = np.maximum.accumulate(edges)

    # 处理重复边界：如果 quantile 落在同一米桶产生 < 1e-6 的退化，强制最小宽度
    min_w = max((hi_m - lo_m) / (n_inner_bins * 50.0), 1e-3)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1] + min_w:
            edges[i] = min(edges[i - 1] + min_w, hi_m)
    edges[-1] = max(edges[-1], hi_m)
    return edges


def _format_dpld_edges_head_tail_oneline(edges: np.ndarray, n_head: int = 20, n_tail: int = 20) -> str:
    """单行字符串：前 n_head 与后 n_tail 个米制 boundary（逗号分隔）。"""
    e = np.asarray(edges, dtype=np.float64).reshape(-1)

    def fmt_block(arr: np.ndarray) -> str:
        return ",".join("{:.6g}".format(float(x)) for x in arr)

    if e.size == 0:
        return "edges_m=[]"
    if e.size <= 40:
        return "edges_m=[{}]".format(fmt_block(e))
    nh = min(n_head, e.size)
    nt = min(n_tail, e.size)
    head = e[:nh]
    tail = e[-nt:]
    return "edges_m_head{}=[{}] ... tail{}=[{}]".format(n_head, fmt_block(head), n_tail, fmt_block(tail))


def _dpld_histogram_from_train_val_ndsm(args: Any) -> Tuple[np.ndarray, int, str]:
    """与 gbh_txt 一致：按 train.txt+val.txt 遍历全部 nDSM，累积 floor(h) 直方图（与 ndsm_stats 第 5 项口径相同）。"""
    from utils.dataset_config import _resolve_gbh_dirs
    from utils.gbh_split_dataset import _online_get_ndsm_stats, _paths_for_stats_train_val

    data_dir, split_dir, _, _ = _resolve_gbh_dirs(args)
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")
    if not os.path.isfile(train_txt) or not os.path.isfile(val_txt):
        raise FileNotFoundError(
            "use_DPLD=2 需要 HTC 划分文件（与 gbh_txt 一致）: {!r} 与 {!r} 均需存在".format(
                train_txt, val_txt
            )
        )
    _, ndsm_paths = _paths_for_stats_train_val(data_dir, train_txt, val_txt)
    if not ndsm_paths:
        raise RuntimeError("use_DPLD=2：train.txt+val.txt 未解析到任何 nDSM 路径（data_dir={}）".format(data_dir))
    _mean_n, _std_n, _minh, _maxh, count = _online_get_ndsm_stats(ndsm_paths)
    count = np.asarray(count, dtype=np.float64)
    return count, len(ndsm_paths), data_dir


def compute_or_load_dpld_bin_stats(
    args: Any,
    n_bins: int,
    ground_bin_max_m: float,
    max_height: float,
    fallback: str = "uniform",
    cache_filename: str = "dpld_bin_edges.pt",
    verbose: bool = True,
    online_rescan: bool = False,
) -> dict:
    """读取/构建 DPLD 用的非均匀 bin edges + 先验概率。

    返回 dict：
        bin_edges_m:     (n_bins+1,)  float32 米制 bin 边界
        bin_prior_probs: (n_bins,)    float32 数据集级 bin 概率，sum=1（用于 bin_head bias 初始化、loss 平衡等）

    优先级（online_rescan=False）：
    1. 命中 `args.data_dir/<cache_filename>` 且 schema/参数匹配 → 直接复用
    2. `args.data_dir/ndsm_stats.pickle` 存在 → 用 count 直方图做 quantile 划分 + 先验估计
    3. fallback="uniform" → 均匀划分 + 均匀先验；fallback="none" → 报错

    online_rescan=True（--use_DPLD 2）：跳过 1/2 的 pickle 与磁盘缓存，遍历 train+val nDSM 在线算 count 再 quantile；
    不写入 dpld 缓存文件。
    """
    data_dir = getattr(args, "data_dir", None)
    if data_dir is None:
        # gbh_data_dir 兜底
        data_dir = getattr(args, "gbh_data_dir", None)
    if data_dir is None:
        # folder layout：dataset_root/dataset_name
        ds_root = getattr(args, "dataset_root", None)
        ds_name = getattr(args, "dataset_name", None)
        if ds_root and ds_name:
            data_dir = os.path.join(ds_root, ds_name)

    if data_dir and os.path.isdir(data_dir):
        cache_path = os.path.join(data_dir, cache_filename)
    else:
        cache_path = None

    if online_rescan:
        count, n_ndsm, scan_dir = _dpld_histogram_from_train_val_ndsm(args)
        inner = _quantile_edges_from_count_hist(
            count, n_inner_bins=n_bins - 1, lo_m=float(ground_bin_max_m), hi_m=float(max_height)
        )
        edges_f = np.concatenate(([0.0], inner)).astype(np.float64)
        prior = _bin_prior_from_count_hist(count, edges_f).astype(np.float32)
        edges_out = edges_f.astype(np.float32)
        if verbose:
            line = _format_dpld_edges_head_tail_oneline(edges_out)
            print(
                "[DPLD] use_DPLD=2 已遍历 train+val 共 {} 张 nDSM（data_dir={}），"
                "quantile n_bins={} ground={}m max={}m | {}".format(
                    n_ndsm, scan_dir, n_bins, ground_bin_max_m, max_height, line,
                ),
                flush=True,
            )
        return {
            "bin_edges_m": torch.from_numpy(edges_out),
            "bin_prior_probs": torch.from_numpy(prior),
        }

    # 1) 缓存命中（要求同时含 bin_edges_m 与 bin_prior_probs，否则视为旧版缓存自动重算）
    if cache_path is not None and os.path.isfile(cache_path):
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if (
            isinstance(payload, dict)
            and int(payload.get("n_bins", -1)) == int(n_bins)
            and abs(float(payload.get("ground_bin_max_m", -1.0)) - float(ground_bin_max_m)) < 1e-6
            and abs(float(payload.get("max_height", -1.0)) - float(max_height)) < 1e-3
            and "bin_edges_m" in payload
            and "bin_prior_probs" in payload
        ):
            if verbose:
                print("[DPLD] 复用缓存 bin stats: {} (n_bins={})".format(cache_path, n_bins), flush=True)
            return {
                "bin_edges_m": torch.as_tensor(payload["bin_edges_m"], dtype=torch.float32),
                "bin_prior_probs": torch.as_tensor(payload["bin_prior_probs"], dtype=torch.float32),
            }

    # 2) 从 ndsm_stats.pickle 重建
    ndsm_pickle = os.path.join(data_dir, "ndsm_stats.pickle") if data_dir else None
    if ndsm_pickle and os.path.isfile(ndsm_pickle):
        try:
            payload = torch.load(ndsm_pickle, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(ndsm_pickle, map_location="cpu")
        if not isinstance(payload, (list, tuple)) or len(payload) < 5:
            raise RuntimeError(
                "ndsm_stats.pickle 格式异常（期望 [mean, std, minh, maxh, count]）: {}".format(ndsm_pickle)
            )
        count = payload[4]
        if torch.is_tensor(count):
            count = count.detach().cpu().numpy()
        count = np.asarray(count, dtype=np.float64)

        inner = _quantile_edges_from_count_hist(
            count, n_inner_bins=n_bins - 1, lo_m=float(ground_bin_max_m), hi_m=float(max_height)
        )
        edges = np.concatenate(([0.0], inner)).astype(np.float64)  # 加入 ground bin
        prior = _bin_prior_from_count_hist(count, edges).astype(np.float32)
        edges = edges.astype(np.float32)
        if verbose:
            top3_bin = np.argsort(-prior)[:3].tolist()
            print(
                "[DPLD] 已从 {} 构建 bin stats：n_bins={}, ground=[0,{}m], max={}m，"
                "首 5 edges={}，末 5 edges={}".format(
                    ndsm_pickle,
                    n_bins,
                    ground_bin_max_m,
                    max_height,
                    np.round(edges[:5], 3).tolist(),
                    np.round(edges[-5:], 3).tolist(),
                ),
                flush=True,
            )
            print(
                "[DPLD] 数据集先验：bin 0 (ground) 占 {:.2%}，最高 3 个 bin idx={}（prob={}）".format(
                    float(prior[0]),
                    top3_bin,
                    np.round(prior[top3_bin], 4).tolist(),
                ),
                flush=True,
            )
        if cache_path is not None:
            try:
                torch.save(
                    {
                        "schema_version": 2,
                        "n_bins": int(n_bins),
                        "ground_bin_max_m": float(ground_bin_max_m),
                        "max_height": float(max_height),
                        "bin_edges_m": torch.from_numpy(edges),
                        "bin_prior_probs": torch.from_numpy(prior),
                    },
                    cache_path,
                )
                if verbose:
                    print("[DPLD] 已缓存 bin stats 到 {}".format(cache_path), flush=True)
            except OSError as e:
                print("[DPLD] 缓存写入失败（忽略）: {}".format(e), flush=True)
        return {
            "bin_edges_m": torch.from_numpy(edges),
            "bin_prior_probs": torch.from_numpy(prior),
        }

    # 3) fallback：均匀 edges + 均匀 prior
    if fallback == "uniform":
        inner = np.linspace(float(ground_bin_max_m), float(max_height), n_bins, dtype=np.float64)
        edges = np.concatenate(([0.0], inner)).astype(np.float32)
        prior = (np.ones(n_bins, dtype=np.float32) / n_bins)
        if verbose:
            print(
                "[DPLD] 警告：未找到 ndsm_stats.pickle（data_dir={}），"
                "退回均匀 bin edges + 均匀先验仅作 sanity check".format(data_dir),
                flush=True,
            )
        return {"bin_edges_m": torch.from_numpy(edges), "bin_prior_probs": torch.from_numpy(prior)}

    raise FileNotFoundError(
        "DPLD 需要 ndsm_stats.pickle 来构建 quantile bin edges（data_dir={}），"
        "请先运行 gbh_txt + --gbh_auto_stats 生成，或者在 block.yaml 把 fallback 改为 uniform".format(data_dir)
    )


def compute_or_load_dpld_bin_edges(*args, **kwargs) -> torch.Tensor:
    """[兼容老接口] 仅返回 bin_edges_m（推荐改用 compute_or_load_dpld_bin_stats）。"""
    return compute_or_load_dpld_bin_stats(*args, **kwargs)["bin_edges_m"]


# ============================================================================
#  DPLD Head
# ============================================================================
class DPLDHead(nn.Module):
    """Dataset-Prior Local Distribution Head。

    Args:
        channels_in: 输入特征通道数（与 SegmentHead / DepthHead 一致 = channels_out / args.out_decoder）
        max_height: 米制最大高度（与原 ConditionalCascadeHeads 的 max_height 一致）
        bin_edges_m: 预计算好的 (n_bins+1,) 米制 bin edges；
                     None 时表示推理阶段会从 state_dict 中加载 buffer，
                     但训练时必须显式传入（一般由 FHE 构建时调 compute_or_load_dpld_bin_edges 取得）
        其余参数（n_bins, top_k, sigma_m, ground_bin_max_m, head_reduction）：
            None 时全部从 block.yaml 读取
    """

    def __init__(
        self,
        channels_in: int,
        max_height: float,
        bin_edges_m: Optional[torch.Tensor] = None,
        bin_prior_probs: Optional[torch.Tensor] = None,
        n_bins: Optional[int] = None,
        top_k: Optional[int] = None,
        sigma_m: Optional[float] = None,
        ground_bin_max_m: Optional[float] = None,
        head_reduction: Optional[int] = None,
        prior_init: Optional[bool] = None,
        prior_init_scale: Optional[float] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("DPLD")

        self.n_bins = int(cfg.get("n_bins", 128) if n_bins is None else n_bins)
        self.top_k = int(cfg.get("top_k", 10) if top_k is None else top_k)
        self.sigma_m = float(cfg.get("sigma_m", 1.5) if sigma_m is None else sigma_m)
        self.ground_bin_max_m = float(
            cfg.get("ground_bin_max_m", 1.0) if ground_bin_max_m is None else ground_bin_max_m
        )
        head_reduction = int(
            cfg.get("head_reduction", 2) if head_reduction is None else head_reduction
        )
        prior_init = bool(cfg.get("prior_init", True) if prior_init is None else prior_init)
        prior_init_scale = float(
            cfg.get("prior_init_scale", 0.05) if prior_init_scale is None else prior_init_scale
        )
        self.max_height = float(max_height)
        self.channels_in = int(channels_in)

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

        # bin edges / centers 注册为 buffer，参与 state_dict 持久化
        if bin_edges_m is not None:
            edges_t = torch.as_tensor(bin_edges_m, dtype=torch.float32)
            if edges_t.numel() != self.n_bins + 1:
                raise ValueError(
                    "bin_edges_m 长度={} 与 n_bins+1={} 不一致".format(edges_t.numel(), self.n_bins + 1)
                )
        else:
            # 占位（推理时由 state_dict 覆盖；训练时必须传入真实 edges）
            edges_t = torch.linspace(0.0, float(max_height), self.n_bins + 1, dtype=torch.float32)
        centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
        self.register_buffer("bin_edges_m", edges_t)
        self.register_buffer("bin_centers_m", centers_t)

        # bin 先验作 buffer 一起保存，方便 criterion 复用并保证 train/test 一致
        if bin_prior_probs is not None:
            prior_t = torch.as_tensor(bin_prior_probs, dtype=torch.float32)
            if prior_t.numel() != self.n_bins:
                raise ValueError(
                    "bin_prior_probs 长度={} 与 n_bins={} 不一致".format(prior_t.numel(), self.n_bins)
                )
            prior_t = prior_t / prior_t.sum().clamp(min=1e-8)
        else:
            prior_t = torch.full((self.n_bins,), 1.0 / max(self.n_bins, 1), dtype=torch.float32)
        self.register_buffer("bin_prior_probs", prior_t)

        # 关键修正：用数据集先验初始化 bin_head 末层 Conv2d
        # ----------------------------------------------------------
        # 默认 Kaiming 初始化下，所有 bin 的 logit 接近 0，softmax 均匀，
        # 这意味着 top-k 加权后 h_pred ≈ 多个分散 bin 中心的平均（DFC23 大约 10m），
        # 而 GT 上 69% 像素是 ground（h≈0m），导致前几个 epoch 模型要花大力气把分布
        # 从均匀拉到 ground 主导，收敛慢于 use_condition（后者有可学残差 R 一步抵消全局偏置）。
        #
        # 解决：把末层 conv 的 weight 缩小、bias 设成 log(prior)，
        #       这样初始 logit = ε·Wx + log(prior)，softmax(logit) ≈ prior。
        #       => 所有像素初始 bin_prob ≈ 数据集先验（ground bin 占大头），
        #          初始 h_pred ≈ E[h]（DFC23 上 ~2-3m，远小于 92.5m）。
        # ----------------------------------------------------------
        if prior_init:
            last_conv = self.bin_head[-1]
            with torch.no_grad():
                last_conv.weight.mul_(float(prior_init_scale))
                log_prior = torch.log(prior_t.clamp(min=1e-8))
                last_conv.bias.copy_(log_prior)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, channels_in, H, W) decoder 最后一层输出
        Returns:
            pred_d:    (B, 1, H, W)  归一化高度 [0,1]（除以 max_height）
            bin_logits:(B, n_bins, H, W)
            bin_prob:  (B, n_bins, H, W)  softmax 概率
        """
        bin_logits = self.bin_head(x)
        bin_prob = F.softmax(bin_logits, dim=1)

        # Top-k 在 bin 维度选最大的 k 个
        topk_p, topk_idx = bin_prob.topk(self.top_k, dim=1)         # (B, k, H, W)
        topk_p_sum = topk_p.sum(dim=1, keepdim=True).clamp(min=1e-8)
        topk_p_norm = topk_p / topk_p_sum                            # 重新归一化
        topk_centers = self.bin_centers_m[topk_idx]                  # (B, k, H, W) 米制
        h_meters = (topk_p_norm * topk_centers).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        pred_d = (h_meters / max(self.max_height, 1e-6)).clamp(0.0, 1.0)
        return pred_d, bin_logits, bin_prob

    # ------------------------------------------------------------------
    def gaussian_soft_label(self, h_gt_m: torch.Tensor) -> torch.Tensor:
        """在米制 bin center 上构造高斯软标签。

        Args:
            h_gt_m: (B, H, W) 或 (B, 1, H, W) 米制 GT 高度
        Returns:
            q: (B, n_bins, H, W) 软标签概率，已沿 bin 维归一化
        """
        if h_gt_m.dim() == 4 and h_gt_m.size(1) == 1:
            h_gt_m = h_gt_m.squeeze(1)
        c = self.bin_centers_m.view(1, -1, 1, 1)                     # (1, n_bins, 1, 1)
        h = h_gt_m.unsqueeze(1)                                       # (B, 1, H, W)
        sigma = max(float(self.sigma_m), 1e-3)
        logits = -((c - h) ** 2) / (2.0 * sigma * sigma)
        q = F.softmax(logits, dim=1)
        return q

    # ------------------------------------------------------------------
    def gt_bin_indices(self, h_gt_m: torch.Tensor) -> torch.Tensor:
        """根据 bin_edges_m 把米制 GT 分到 bin index。

        Args:
            h_gt_m: (B, H, W) 米制 GT
        Returns:
            (B, H, W) long，取值 [0, n_bins-1]
        """
        if h_gt_m.dim() == 4 and h_gt_m.size(1) == 1:
            h_gt_m = h_gt_m.squeeze(1)
        # 用 (n_bins-1) 个内部边界做 bucketize
        inner_edges = self.bin_edges_m[1:-1]                          # (n_bins-1,)
        # bucketize: h ∈ (-inf, e0] → 0；(e0, e1] → 1；...
        idx = torch.bucketize(h_gt_m.contiguous(), inner_edges, right=False)
        return idx.clamp(0, self.n_bins - 1)


__all__ = ["DPLDHead", "compute_or_load_dpld_bin_edges"]
