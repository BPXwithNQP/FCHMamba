"""
GroupMamba（classification/groupmamba.py）封装为 FusedHE 第二支路（原 mit 位），输出四级特征并 1×1 对齐到 MiT 通道 [64,128,320,512]。

集成方式与 efficient_vmamba_backbone 一致：将 GroupMamba 源码目录加入 sys.path；需已编译 selective_scan CUDA 扩展（见 GroupMamba-main/kernels/selective_scan/），并建议与 utils.conda_glibcxx 配合。

预训练：默认从 configs 的 weights_path 解析（如 groupmamba_base_ema.pth，EMA 导出的纯 state_dict）。

说明：提供的 groupmamba_base_ema.pth 对应 GroupMamba-base，而非 EfficientVMamba；架构以 GroupMamba-main 为准。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _conda_env_lib_dirs() -> list[str]:
    out: list[str] = []
    p = os.environ.get("CONDA_PREFIX")
    if p:
        lp = os.path.join(p, "lib")
        if os.path.isdir(lp):
            out.append(lp)
    exe = getattr(sys, "executable", None) or ""
    if exe and os.path.isfile(exe):
        bindir = os.path.dirname(os.path.abspath(exe))
        if os.path.basename(bindir) == "bin":
            cand = os.path.join(os.path.dirname(bindir), "lib")
            if os.path.isdir(cand) and cand not in out:
                out.append(cand)
    return out


def _prepend_conda_lib_for_glibcxx() -> None:
    import ctypes

    lib_dirs = _conda_env_lib_dirs()
    if not lib_dirs:
        return
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [x for x in ld.split(os.pathsep) if x] if ld else []
    for d in reversed(lib_dirs):
        if d not in parts:
            parts.insert(0, d)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

    for d in lib_dirs:
        for soname in ("libstdc++.so.6", "libstdc++.so"):
            path = os.path.join(d, soname)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_prepend_conda_lib_for_glibcxx()

import torch
import torch.nn as nn

from models.Blocks.RGHF import RGHF


def _groupmamba_models_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "主干网络内容" / "GroupMamba-main" / "classification" / "models"


def _load_groupmamba_module():
    d = _groupmamba_models_dir()
    if not d.is_dir():
        raise FileNotFoundError("未找到 GroupMamba 源码目录: {}".format(d))
    p = str(d)
    if p not in sys.path:
        sys.path.insert(0, p)
    import groupmamba as gm

    return gm


_GM = None


def _get_gm():
    global _GM
    if _GM is None:
        _GM = _load_groupmamba_module()
    return _GM


# MiT-B4 对齐通道（与 mit / EfficientVMamba 融合一致）
_TARGET_CH = [64, 128, 320, 512]

_VARIANT_SRC_CH = {
    "base": [96, 192, 424, 512],
    "small": [64, 128, 348, 512],
    "tiny": [64, 128, 348, 448],
}


def _build_groupmamba(variant: str):
    gm = _get_gm()
    v = (variant or "base").lower()
    if v == "base":
        return gm.groupmamba_base(pretrained=False)
    if v == "small":
        return gm.groupmamba_small(pretrained=False)
    if v == "tiny":
        return gm.groupmamba_tiny(pretrained=False)
    raise ValueError("未知 GroupMamba variant: {!r}，可选: base, small, tiny".format(variant))


def _forward_multiscale(
    backbone: nn.Module,
    x: torch.Tensor,
    rghf_modules: Optional[nn.ModuleList] = None,
):
    """与分类 forward_features 中各 stage 输出一致：每 stage 末 norm + NCHW，共 4 级。

    若提供 rghf_modules，则在每个 stage 的 norm→NCHW 之后、写入 feats 之前
    以残差形式注入 RGHF 高频分支；非末 stage 的增强特征继续送入下一级 patch_embed。
    """
    B = x.shape[0]
    feats: list[torch.Tensor] = []
    use_rghf = rghf_modules is not None and len(rghf_modules) > 0
    for i in range(backbone.num_stages):
        patch_embed = getattr(backbone, "patch_embed{}".format(i + 1))
        block = getattr(backbone, "block{}".format(i + 1))
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        norm = getattr(backbone, "norm{}".format(i + 1))
        x = norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if use_rghf:
            x = rghf_modules[i](x)
        feats.append(x)
    return tuple(feats)


class GroupMambaFusedBackbone(nn.Module):
    """与 FusedEncoder 中 self.mit 接口一致：forward(x) -> (x1,x2,x3,x4)，各为 NCHW。"""

    def __init__(self, cfg: dict, args):
        super().__init__()
        pre = bool(cfg.get("pretrained", True))
        wp = cfg.get("weights_path")
        variant = cfg.get("variant", "base")
        self._variant = variant

        self.backbone = _build_groupmamba(variant)
        src_ch = list(cfg.get("stage_channels")) if cfg.get("stage_channels") else _VARIANT_SRC_CH.get(
            (variant or "base").lower(), _VARIANT_SRC_CH["base"]
        )

        self.native_mode = bool(cfg.get("native_mode", False)) or bool(
            getattr(args, "native_groupmamba_encoder", False)
        )
        if self.native_mode:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(4)])
            print(
                "[Encoder/GroupMamba] native_mode=True：输出原生 stage 通道 {}（不对齐 MiT）".format(
                    src_ch
                ),
                flush=True,
            )
        else:
            adapters = []
            for src, tgt in zip(src_ch, _TARGET_CH):
                if src == tgt:
                    adapters.append(nn.Identity())
                else:
                    adapters.append(nn.Conv2d(src, tgt, kernel_size=1, bias=False))
            self.adapters = nn.ModuleList(adapters)

        use_rghf = bool(cfg.get("use_RGHF", False)) or bool(getattr(args, "use_RGHF", False))
        self.use_RGHF = use_rghf
        if use_rghf:
            self.rghf = nn.ModuleList(
                [RGHF(channels=src_ch[i], stage_idx=i + 1) for i in range(4)]
            )
            print("[Encoder/GroupMamba] RGHF 已启用：每 stage 末注入门控高频残差", flush=True)
        else:
            self.rghf = None

        if pre and wp and os.path.isfile(wp):
            self._load_pretrained(wp)
        elif pre and wp:
            print(
                "[Encoder/GroupMamba] 未找到权重文件: {}".format(wp),
                flush=True,
            )

    def _load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        if not isinstance(sd, dict):
            return
        sd = {k.replace("module.", "").replace("module.module.", ""): v for k, v in sd.items()}
        # 分类头不参与融合主干时可忽略
        sd = {
            k: v
            for k, v in sd.items()
            if not (k.startswith("head.") or k.startswith("dist_head."))
        }
        info = self.backbone.load_state_dict(sd, strict=False)
        print(
            "[Encoder/GroupMamba] 预训练加载 variant={} missing={} unexpected={}".format(
                self._variant, len(info.missing_keys), len(info.unexpected_keys)
            ),
            flush=True,
        )

    def forward(self, x: torch.Tensor):
        outs = _forward_multiscale(
            self.backbone,
            x,
            rghf_modules=self.rghf,
        )
        if len(outs) != 4:
            raise RuntimeError("GroupMamba 应输出 4 个 stage 特征")
        return tuple(self.adapters[i](outs[i]) for i in range(4))
