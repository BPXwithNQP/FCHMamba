"""
训练/测试集构建：支持旧版 folder 布局（train/validation/rgb）与 HTC-DC 风格 gbh_txt（image/ndsm + txt）。

gbh_txt 时路径与 HTC-DC 配置一致：data_dir、data_split_dirs、test_data_split_dirs（见 configs/VHGconfig.yaml）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch

from utils.local_dataset import LocalDataset

# 可通过 FCHMAMBA_DATA_DIR 覆盖；默认使用仓库下的 datasets 目录。
_DEFAULT_HTC_DATA_DIR = os.environ.get(
    "FCHMAMBA_DATA_DIR", "datasets/dfc2019_512"
)
_REP_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _REP_ROOT.parent


def _first_existing_dir(candidates: Sequence[Optional[str]], label: str = "path") -> str:
    """返回第一个存在的目录；均不存在时返回最后一个候选的绝对路径。"""
    tried: List[str] = []
    chosen: Optional[str] = None
    for raw in candidates:
        if raw is None or str(raw).strip() == "":
            continue
        p = os.path.abspath(os.path.expanduser(str(raw)))
        if p in tried:
            continue
        tried.append(p)
        if os.path.isdir(p):
            chosen = p
            break
    if chosen is None:
        chosen = tried[-1] if tried else ""
    elif tried and chosen != tried[0]:
        print(
            "[dataset] {} 不可用 ({})，回退: {}".format(label, tried[0], chosen),
            flush=True,
        )
    return chosen


def add_gbh_dataset_args(parser) -> Any:
    parser.add_argument(
        "--dataset_layout",
        type=str,
        default="gbh_txt",
        choices=["folder", "gbh_txt"],
        help="folder: dataset_root/dataset_name/{train,validation}/rgb；"
        "gbh_txt: 与 HTC-DC 相同，由 data_dir + data_split_dirs + test_data_split_dirs 指定",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="HTC 风格数据根目录（含 image/、ndsm/、可选 pickle），与 VHGconfig 的 data_dir 一致；"
        "默认未指定时为 dataset_root/dataset_name，再否则 {}".format(_DEFAULT_HTC_DATA_DIR),
    )
    parser.add_argument(
        "--data_split_dirs",
        type=str,
        default=None,
        help="train.txt、val.txt 所在目录；未指定时默认为 <data_dir>/data_split_dirs（而 data_dir 默认来自 dataset_root/dataset_name）",
    )
    parser.add_argument(
        "--test_data_split_dirs",
        nargs="*",
        default=None,
        help="含 test.txt 的目录，可多个（同 HTC）；未指定时默认为 <data_dir>/test_data_split_dirs；"
        "test.py 在未显式传本组参数时按 dataset_root/dataset_name 自动解析，不沿用 checkpoint 内路径",
    )
    parser.add_argument(
        "--gbh_data_dir",
        default=None,
        type=str,
        help="[别名] 同 --data_dir，若设置则覆盖未单独指定的 data_dir 语义（仍优先使用显式 --data_dir）",
    )
    parser.add_argument(
        "--gbh_split_dir",
        default=None,
        type=str,
        help="[别名] 同 --data_split_dirs",
    )
    parser.add_argument(
        "--gbh_test_split_dir",
        default=None,
        type=str,
        help="[别名] 单个 test 划分目录，同 test_data_split_dirs 仅一项时",
    )
    parser.add_argument(
        "--gbh_normalize_rgb",
        action="store_true",
        help="若存在 image_stats.pickle，对 RGB 做 mean/std 归一化（与 HTC-DC GBHDataset 一致）",
    )
    parser.add_argument(
        "--gbh_auto_stats",
        action="store_true",
        help="缺少 image_stats.pickle / ndsm_stats.pickle 时按 train+val 列表在线计算（较慢，首次运行）",
    )
    parser.add_argument(
        "--gbh_apply_dataset_inform",
        action="store_true",
        help="若存在 dataset_inform.pt 且 max_height_eval 为 0（自动高度），用 recommended_max_height_eval 覆盖；非 0 时不生效",
    )
    parser.add_argument(
        "--gbh_require_blg",
        action="store_true",
        help="gbh_txt 时要求每条样本均存在 mask/<stem>_BLG.tif，否则启动即报错（DFC2019 等建议开启）",
    )
    return parser


def _normalize_test_split_list(args) -> List[str]:
    """与 HTC 的 test_data_split_dirs 列表一致；返回若干目录路径。"""
    raw = getattr(args, "test_data_split_dirs", None)
    legacy = getattr(args, "gbh_test_split_dir", None)
    if raw is not None and len(raw) > 0:
        return [os.path.abspath(os.path.expanduser(p)) for p in raw]
    if legacy:
        return [os.path.abspath(os.path.expanduser(legacy))]
    return []


def resolve_htc_style_paths(args):
    """与 HTC-DC VHGconfig 一致：data_dir、data_split_dirs、test_data_split_dirs（列表首项用于 test.txt）。"""
    return _resolve_gbh_dirs(args)


def _resolve_gbh_dirs(args):
    # data_dir：显式 --data_dir 优先，其次别名 gbh_data_dir，再回退 dataset_root/dataset_name
    explicit = getattr(args, "data_dir", None)
    alias = getattr(args, "gbh_data_dir", None)
    fallback = os.path.join(args.dataset_root, args.dataset_name)
    workspace_fallback = str(_WORKSPACE_ROOT / "dataset" / args.dataset_name)
    data_dir = _first_existing_dir(
        [explicit or alias, fallback, workspace_fallback, _DEFAULT_HTC_DATA_DIR],
        label="data_dir",
    )

    split_dir = getattr(args, "data_split_dirs", None) or getattr(args, "gbh_split_dir", None)
    if not split_dir:
        split_dir = os.path.join(data_dir, "data_split_dirs")
    split_dir = _first_existing_dir(
        [split_dir, os.path.join(data_dir, "data_split_dirs")],
        label="data_split_dirs",
    )

    test_list = _normalize_test_split_list(args)
    if test_list:
        test_split_dir = _first_existing_dir(test_list, label="test_data_split_dirs")
    else:
        test_split_dir = _first_existing_dir(
            [
                os.path.join(data_dir, "test_data_split_dirs"),
                os.path.join(fallback, "test_data_split_dirs"),
                os.path.join(workspace_fallback, "test_data_split_dirs"),
            ],
            label="test_data_split_dirs",
        )
    return data_dir, split_dir, test_split_dir, test_list


def snapshot_max_height_user_intent(args) -> None:
    """
    在修改 args.max_height_eval 之前调用（可多次调用，仅首次生效）。
    仅当用户在命令行将 max_height_eval 设为 **0** 时，标记为「允许从数据集自动读取最大高度」。
    """
    if getattr(args, "_max_height_eval_user_intent_captured", False):
        return
    mh = getattr(args, "max_height_eval", None)
    try:
        v = int(mh)
    except (TypeError, ValueError):
        v = -1
    setattr(args, "_user_wants_auto_max_height", v == 0)
    setattr(args, "_max_height_eval_user_intent_captured", True)


def read_max_height_from_ndsm_stats_pickle(data_dir: str) -> float:
    """
    读取 HTC 风格 ndsm_stats.pickle：torch.save([mean, std, minh, maxh, count], ...)。
    返回米制最大高度 maxh（索引 3）。
    """
    ndsm_pk = os.path.join(data_dir, "ndsm_stats.pickle")
    if not os.path.isfile(ndsm_pk):
        raise FileNotFoundError(
            "未找到 ndsm_stats.pickle: {}（max_height_eval=0 时需该文件或先 gbh_txt + --gbh_auto_stats 生成）".format(
                ndsm_pk
            )
        )
    try:
        payload = torch.load(ndsm_pk, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ndsm_pk, map_location="cpu")
    if not isinstance(payload, (list, tuple)) or len(payload) < 4:
        raise RuntimeError(
            "ndsm_stats.pickle 格式异常（期望至少 4 项 [mean, std, minh, maxh, ...]）: {}".format(
                ndsm_pk
            )
        )
    maxh = payload[3]
    if torch.is_tensor(maxh):
        maxh = float(maxh.detach().cpu().item())
    else:
        maxh = float(maxh)
    return maxh


def maybe_apply_ndsm_pickle_max_height(args) -> None:
    """
    仅当用户将 max_height_eval 设为 **0** 时：从数据集目录下 ndsm_stats.pickle 读取最大高度并写回 args.max_height_eval。
    非 0 时不做任何读取（保持用户指定或 checkpoint 注入的值）。
    gbh_txt 下若尚无 pickle，会先尝试 ensure_htc_pickle_stats（与训练时一致，需 train/val.txt）。
    """
    snapshot_max_height_user_intent(args)
    if not getattr(args, "_user_wants_auto_max_height", False):
        return

    layout = getattr(args, "dataset_layout", "folder")
    if layout == "gbh_txt":
        from utils.gbh_split_dataset import ensure_htc_pickle_stats

        data_dir, split_dir, _, _ = _resolve_gbh_dirs(args)
        train_txt = os.path.join(split_dir, "train.txt")
        val_txt = os.path.join(split_dir, "val.txt")
        if not os.path.isfile(train_txt) or not os.path.isfile(val_txt):
            raise FileNotFoundError(
                "max_height_eval=0 需要 gbh_txt 下的 {} 与 {} 以定位/生成 ndsm_stats.pickle".format(
                    train_txt,
                    val_txt,
                )
            )
        ensure_htc_pickle_stats(
            data_dir,
            train_txt,
            val_txt,
            auto_compute=bool(getattr(args, "gbh_auto_stats", False)),
        )
    else:
        data_dir = os.path.abspath(
            os.path.expanduser(os.path.join(args.dataset_root, args.dataset_name))
        )

    maxh = read_max_height_from_ndsm_stats_pickle(data_dir)
    if not (maxh > 0) or maxh != maxh:
        raise RuntimeError(
            "ndsm_stats.pickle 中 max 高度无效: {}（data_dir={}）".format(maxh, data_dir)
        )
    args.max_height_eval = int(round(maxh))
    print(
        "[dataset] max_height_eval=0 -> 已从 ndsm_stats.pickle 读取最大高度: {} m（data_dir={}）".format(
            args.max_height_eval,
            data_dir,
        ),
        flush=True,
    )


def maybe_apply_dataset_inform_max_height(args) -> None:
    """与 ndsm_stats 一致：仅当 max_height_eval 在用户侧为 0（自动高度）时可用 dataset_inform 覆盖。"""
    snapshot_max_height_user_intent(args)
    if not getattr(args, "gbh_apply_dataset_inform", False):
        return
    if not getattr(args, "_user_wants_auto_max_height", False):
        return
    data_dir, _, _, _ = _resolve_gbh_dirs(args)
    pt_path = os.path.join(data_dir, "dataset_inform.pt")
    if not os.path.isfile(pt_path):
        return
    try:
        inform = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        inform = torch.load(pt_path, map_location="cpu")
    rec = inform.get("recommended_max_height_eval")
    if rec is not None and float(rec) > 0:
        args.max_height_eval = int(round(float(rec)))
        print(
            "[gbh] dataset_inform.pt -> max_height_eval={}".format(args.max_height_eval),
            flush=True,
        )


def build_train_val_datasets(args):
    layout = getattr(args, "dataset_layout", "folder")
    if layout != "gbh_txt":
        train_dir = args.train_dir or os.path.join(args.dataset_root, args.dataset_name, "train")
        val_dir = args.val_dir or os.path.join(args.dataset_root, args.dataset_name, "validation")
        return LocalDataset(train_dir, args=args), LocalDataset(val_dir, args=args)

    from utils.gbh_split_dataset import GbhTxtSplitDataset, ensure_htc_pickle_stats

    maybe_apply_dataset_inform_max_height(args)
    data_dir, split_dir, _, _ = _resolve_gbh_dirs(args)
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")
    if not os.path.isfile(train_txt):
        raise FileNotFoundError("gbh_txt 需要 {}".format(train_txt))
    if not os.path.isfile(val_txt):
        raise FileNotFoundError("gbh_txt 需要 {}".format(val_txt))
    ensure_htc_pickle_stats(
        data_dir,
        train_txt,
        val_txt,
        auto_compute=bool(getattr(args, "gbh_auto_stats", False)),
    )
    norm = bool(getattr(args, "gbh_normalize_rgb", False))
    train_data = GbhTxtSplitDataset(data_dir, train_txt, args=args, normalize_rgb=norm)
    val_data = GbhTxtSplitDataset(data_dir, val_txt, args=args, normalize_rgb=norm)
    return train_data, val_data


def apply_test_dir_arg_for_gbh_txt(args) -> None:
    """
    若 --test_dir 指向 test.txt 文件路径，或指向已含 test.txt 的目录，则自动切换为 gbh_txt 布局：
    - data_dir = test_data_split_dirs 的父目录（例如 .../htcdc_dfc23）
    - test_data_split_dirs = 含 test.txt 的目录（例如 .../htcdc_dfc23/test_data_split_dirs）

    若 --test_dir 为普通 folder 布局（无 test.txt），则不修改，仍由 build_test_dataset 按目录加载。
    """
    td = getattr(args, "test_dir", None)
    if not td:
        setattr(args, "_test_dir_gbh_resolved", False)
        return
    p = os.path.abspath(os.path.expanduser(str(td)))
    test_txt_path = None
    if os.path.isfile(p) and p.lower().endswith(".txt"):
        test_txt_path = p
    elif os.path.isdir(p):
        cand = os.path.join(p, "test.txt")
        if os.path.isfile(cand):
            test_txt_path = cand
    if test_txt_path is None:
        setattr(args, "_test_dir_gbh_resolved", False)
        return
    test_split_dir = os.path.dirname(test_txt_path)
    data_dir = os.path.dirname(test_split_dir)
    args.dataset_layout = "gbh_txt"
    args.data_dir = data_dir
    args.test_data_split_dirs = [test_split_dir]
    setattr(args, "_test_dir_gbh_resolved", True)
    print(
        "[dataset] --test_dir 指向 test 划分：layout=gbh_txt，data_dir={}，test_data_split_dirs={}".format(
            data_dir,
            test_split_dir,
        ),
        flush=True,
    )


def resolve_test_data_path(args) -> str:
    """用于日志：folder 为 test 目录；gbh_txt 为 test.txt 路径。"""
    if getattr(args, "dataset_layout", "folder") == "gbh_txt":
        _, _, test_split_dir, test_list = _resolve_gbh_dirs(args)
        if test_list and len(test_list) > 1:
            return os.path.join(test_split_dir, "test.txt") + " (test_data_split_dirs[0]，共{}项)".format(
                len(test_list)
            )
        return os.path.join(test_split_dir, "test.txt")
    return getattr(args, "test_dir", None) or os.path.join(
        args.dataset_root, args.dataset_name, "test"
    )


def build_test_dataset(args):
    layout = getattr(args, "dataset_layout", "folder")
    if layout != "gbh_txt":
        test_dir = getattr(args, "test_dir", None) or os.path.join(
            args.dataset_root, args.dataset_name, "test"
        )
        return LocalDataset(test_dir, args=args)

    from utils.gbh_split_dataset import GbhTxtSplitDataset, ensure_htc_pickle_stats

    maybe_apply_dataset_inform_max_height(args)
    data_dir, split_dir, test_split_dir, _ = _resolve_gbh_dirs(args)
    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")
    test_txt = os.path.join(test_split_dir, "test.txt")
    if not os.path.isfile(test_txt):
        raise FileNotFoundError("gbh_txt 测试需要 {}".format(test_txt))
    ensure_htc_pickle_stats(
        data_dir,
        train_txt,
        val_txt,
        auto_compute=bool(getattr(args, "gbh_auto_stats", False)),
    )
    norm = bool(getattr(args, "gbh_normalize_rgb", False))
    return GbhTxtSplitDataset(data_dir, test_txt, args=args, normalize_rgb=norm)
