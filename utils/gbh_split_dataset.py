"""
HTC-DC 增强版风格的数据读取：data_dir 下 image/、ndsm/（可选 mask/），
由 txt 列出样本：每行可为「场景主干」（文件名 {stem}_IMG.tif）或已带 _IMG 后缀的 stem（文件名 {stem}.tif，dfc23 风格）。
统计量 image_stats.pickle / ndsm_stats.pickle 与原版命名一致。
输出 batch 字典与 LocalDataset 一致：image, depth, segment, fname；
若存在 mask/<stem>_BLG.tif，额外提供 building_mask（float 0/1，与 depth 同空间尺寸），
供论文 RMSE-M/NM/B 与 eval_depth 中「建筑前景」划分（不再依赖 GT 高度阈值）。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Any


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


def _read_lines(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def _first_existing_path(candidates: Sequence[str]) -> str:
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def _paths_for_stem_line(data_dir: str, stem: str) -> Tuple[str, str, str, str]:
    """
    由 txt 行构造 image / ndsm / mask 路径。

    - HTC 惯例：行内为场景主干（不含 _IMG），文件名为 {stem}_IMG.tif、{stem}_AGL.tif、{stem}_BLG.tif。
    - DFC23 等：行内已含「_IMG」后缀；实际磁盘上常见 {base}.tif（image/）与 {base}.tif（dsm/），
      亦兼容文档中的 {stem}.tif、ndsm/{base}_AGL.tif。
    """
    stem = stem.strip()
    if stem.endswith("_IMG"):
        base = stem[: -len("_IMG")]
        rgb = _first_existing_path(
            [
                os.path.join(data_dir, "image", base + ".tif"),
                os.path.join(data_dir, "image", stem + ".tif"),
            ]
        )
        ndsm = _first_existing_path(
            [
                os.path.join(data_dir, "dsm", base + ".tif"),
                os.path.join(data_dir, "ndsm", base + "_AGL.tif"),
                os.path.join(data_dir, "ndsm", base + ".tif"),
                os.path.join(data_dir, "dsm", base + "_AGL.tif"),
            ]
        )
        mask = _first_existing_path(
            [
                os.path.join(data_dir, "mask", base + "_BLG.tif"),
                os.path.join(data_dir, "mask", base + ".tif"),
            ]
        )
    else:
        rgb = os.path.join(data_dir, "image", stem + "_IMG.tif")
        ndsm = os.path.join(data_dir, "ndsm", stem + "_AGL.tif")
        mask = os.path.join(data_dir, "mask", stem + "_BLG.tif")
    return rgb, ndsm, mask


def load_gbh_sample_by_stem(
    data_dir: str,
    stem: str,
    args: Any,
    normalize_rgb: Optional[bool] = None,
) -> dict:
    """
    按 stem 从 HTC/DFC 风格 data_dir（image/、ndsm/、可选 mask/）加载一条样本，
    无须出现在 train/val/test 的 txt 中（全库按文件名规则解析）。

    stem 与 txt 行一致：可带或不带 _IMG 后缀（见 _paths_for_stem_line）。
    """
    stem = stem.strip()
    rgb_path, ndsm_path, mask_path = _paths_for_stem_line(data_dir, stem)
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(
            "全局查找未找到 RGB（请核对 stem 与 data_dir）: {}".format(rgb_path)
        )
    if not os.path.isfile(ndsm_path):
        raise FileNotFoundError("全局查找未找到 nDSM: {}".format(ndsm_path))
    mpath = mask_path if os.path.isfile(mask_path) else None

    if normalize_rgb is None:
        normalize_rgb = bool(getattr(args, "gbh_normalize_rgb", False))
    max_height = float(getattr(args, "max_height_eval", 200))
    min_height = float(getattr(args, "min_height_eval", 1e-4))
    rgb_mean = None
    rgb_std = None
    if normalize_rgb:
        img_stats = os.path.join(data_dir, "image_stats.pickle")
        if os.path.isfile(img_stats):
            try:
                rgb_mean, rgb_std = torch.load(img_stats, map_location="cpu", weights_only=False)
            except TypeError:
                rgb_mean, rgb_std = torch.load(img_stats, map_location="cpu")
    to_tensor = T.ToTensor()

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("无法读取 RGB: {}".format(rgb_path))
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(ndsm_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError("无法读取 nDSM: {}".format(ndsm_path))
    depth = np.asarray(depth, dtype=np.float32)

    image = to_tensor(image)
    depth_t = to_tensor(depth)

    nan_ind = torch.isnan(depth_t)
    depth_t[nan_ind] = min_height
    inf_in = torch.isinf(depth_t)
    depth_t[inf_in] = min_height

    gt_ind = depth_t <= 0.9
    depth_t[gt_ind] = min_height

    segment = torch.ones_like(depth_t, dtype=torch.float32)
    segment[gt_ind] = 0.0

    if rgb_mean is not None and rgb_std is not None:
        image = T.functional.normalize(image, rgb_mean, rgb_std)

    data = {
        "image": image,
        "depth": depth_t / max_height,
        "segment": segment,
        "fname": rgb_path,
    }
    if mpath is not None:
        m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError("无法读取 building mask: {}".format(mpath))
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[:, :, 0]
        dh, dw = int(depth_t.shape[-2]), int(depth_t.shape[-1])
        if m.shape[0] != dh or m.shape[1] != dw:
            m = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
        bmask = (m > 0).astype(np.float32)
        data["building_mask"] = torch.from_numpy(bmask).unsqueeze(0)

    return data


def _collect_paths_from_txt(
    data_dir: str, lines: Sequence[str]
) -> List[Tuple[str, str, str, Optional[str]]]:
    """返回 (stem, rgb_path, ndsm_path, mask_path 或 None)。"""
    out = []
    for stem in lines:
        rgb, ndsm, mask = _paths_for_stem_line(data_dir, stem)
        mpath = mask if os.path.isfile(mask) else None
        out.append((stem, rgb, ndsm, mpath))
    return out


def _paths_for_stats_train_val(
    data_dir: str, train_txt: str, val_txt: str
) -> Tuple[List[str], List[str]]:
    """与 HTC dataloaders 一致：train+val 的 RGB / nDSM 路径列表。"""
    stems = []
    for p in (train_txt, val_txt):
        if os.path.isfile(p):
            stems.extend(_read_lines(p))
    rgb_paths = []
    ndsm_paths = []
    for stem in stems:
        rgb, ndsm, _ = _paths_for_stem_line(data_dir, stem)
        rgb_paths.append(rgb)
        ndsm_paths.append(ndsm)
    return rgb_paths, ndsm_paths


def _online_get_image_stats(rgb_paths: List[str]) -> Tuple[List[float], List[float]]:
    sum_x = np.zeros(3, dtype=np.float64)
    sum_x2 = np.zeros(3, dtype=np.float64)
    n = 0
    for path in rgb_paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("无法读取 RGB: {}".format(path))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64).reshape(-1, 3)
        sum_x += img.sum(axis=0)
        sum_x2 += (img * img).sum(axis=0)
        n += img.shape[0]
    mean = sum_x / max(n, 1)
    std = np.sqrt(np.maximum(sum_x2 / max(n, 1) - mean * mean, 0.0))
    return mean.tolist(), std.tolist()


def _online_get_ndsm_stats(ndsm_paths: List[str]) -> Tuple[float, float, float, float, np.ndarray]:
    sum_x = 0.0
    sum_x2 = 0.0
    n = 0
    minh = 0.0
    maxh = 0.0
    for path in ndsm_paths:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError("无法读取 nDSM: {}".format(path))
        img = np.asarray(d, dtype=np.float64).flatten()
        img = np.nan_to_num(img)
        sum_x += img.sum()
        sum_x2 += (img ** 2).sum()
        n += img.size
        minh = float(min(img.min(), minh))
        maxh = float(max(img.max(), maxh))
    minh = 0.0 if minh < 0 else minh
    mean = sum_x / max(n, 1)
    std = float(np.sqrt(max(sum_x2 / max(n, 1) - mean ** 2, 0.0)))
    count = np.zeros(max(1, int(np.ceil(max(maxh, 1.0)))), dtype=np.float64)
    for path in ndsm_paths:
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.asarray(d, dtype=np.float64).flatten()
        img = np.nan_to_num(img)
        img = np.clip(np.floor(img), 0, None).astype(np.int64)
        res = np.bincount(img, minlength=len(count))
        if res.size > count.size:
            count = np.pad(count, (0, res.size - count.size))
        count[: res.size] += res
    return float(mean), std, float(minh), float(maxh), count


def ensure_htc_pickle_stats(
    data_dir: str, train_txt: str, val_txt: str, auto_compute: bool = False
) -> None:
    """若缺少 image_stats.pickle / ndsm_stats.pickle，可选按 train+val 在线计算（较慢）。"""
    img_pk = os.path.join(data_dir, "image_stats.pickle")
    ndsm_pk = os.path.join(data_dir, "ndsm_stats.pickle")
    need_img = not os.path.isfile(img_pk)
    need_ndsm = not os.path.isfile(ndsm_pk)
    if not need_img and not need_ndsm:
        return
    if not auto_compute:
        missing = []
        if need_img:
            missing.append(img_pk)
        if need_ndsm:
            missing.append(ndsm_pk)
        raise FileNotFoundError(
            "缺少统计文件: {}。请先放置 HTC 生成的 pickle，或训练时加 --gbh_auto_stats".format(
                ", ".join(missing)
            )
        )
    rgb_paths, ndsm_paths = _paths_for_stats_train_val(data_dir, train_txt, val_txt)
    if need_img:
        if not rgb_paths:
            raise RuntimeError("train/val 列表为空，无法计算 image_stats")
        mean, std = _online_get_image_stats(rgb_paths)
        torch.save([mean, std], img_pk)
        print("[gbh] 已写入 {}".format(img_pk), flush=True)
    if need_ndsm:
        if not ndsm_paths:
            raise RuntimeError("train/val 列表为空，无法计算 ndsm_stats")
        mean_n, std_n, minh, maxh, count = _online_get_ndsm_stats(ndsm_paths)
        torch.save([mean_n, std_n, minh, maxh, count], ndsm_pk)
        print("[gbh] 已写入 {}".format(ndsm_pk), flush=True)


class GbhTxtSplitDataset(Dataset):
    """单 split 文件（如 train.txt）对应的数据集。"""

    def __init__(
        self,
        data_dir: str,
        split_txt: str,
        args,
        normalize_rgb: bool = False,
    ):
        self.data_dir = data_dir
        self.args = args
        self.max_height = float(getattr(args, "max_height_eval", 200))
        self.min_height = float(getattr(args, "min_height_eval", 1e-4))
        self._normalize_rgb = normalize_rgb
        self._rgb_mean = None
        self._rgb_std = None
        if normalize_rgb:
            img_stats = os.path.join(data_dir, "image_stats.pickle")
            if os.path.isfile(img_stats):
                mean, std = torch.load(img_stats, map_location="cpu", weights_only=False)
                self._rgb_mean = mean
                self._rgb_std = std
        self._pairs = _collect_paths_from_txt(data_dir, _read_lines(split_txt))
        if not self._pairs:
            raise RuntimeError("split 为空或不存在: {}".format(split_txt))
        if getattr(args, "gbh_require_blg", False):
            missing = [st for st, _, _, mp in self._pairs if mp is None]
            if missing:
                raise FileNotFoundError(
                    "已启用 --gbh_require_blg，以下样本缺少 mask/*_BLG.tif（前 10 个）: {}".format(
                        missing[:10]
                    )
                )
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        stem, rgb_path, ndsm_path, mask_path = self._pairs[idx]

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError("无法读取 RGB: {}".format(rgb_path))
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(ndsm_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError("无法读取 nDSM: {}".format(ndsm_path))
        depth = np.asarray(depth, dtype=np.float32)

        image = self.to_tensor(image)
        depth_t = self.to_tensor(depth)

        nan_ind = torch.isnan(depth_t)
        depth_t[nan_ind] = self.min_height
        inf_in = torch.isinf(depth_t)
        depth_t[inf_in] = self.min_height

        # 与 utils/local_dataset.LocalDataset 一致：<=0.9m 视为无效地面
        gt_ind = depth_t <= 0.9
        depth_t[gt_ind] = self.min_height

        segment = torch.ones_like(depth_t, dtype=torch.float32)
        segment[gt_ind] = 0.0

        if self._rgb_mean is not None and self._rgb_std is not None:
            image = T.functional.normalize(image, self._rgb_mean, self._rgb_std)

        data = {
            "image": image,
            "depth": depth_t / self.max_height,
            "segment": segment,
            "fname": rgb_path,
        }
        if mask_path is not None:
            m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError("无法读取 building mask: {}".format(mask_path))
            m = np.asarray(m)
            if m.ndim == 3:
                m = m[:, :, 0]
            dh, dw = int(depth_t.shape[-2]), int(depth_t.shape[-1])
            if m.shape[0] != dh or m.shape[1] != dw:
                m = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
            bmask = (m > 0).astype(np.float32)
            data["building_mask"] = torch.from_numpy(bmask).unsqueeze(0)

        return data
