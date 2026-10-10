"""
EfficientVMamba（Backbone_EfficientVSSM）封装为 FusedHE 第二支路（原 mit 位），输出四级特征并 1×1 对齐到 MiT 通道 [64,128,320,512]。

依赖：需能 import EfficientVMamba-main/classification 下 vmamba_efficient，并编译 selective_scan CUDA 扩展（见该仓库 selective_scan/）。

权重：默认从 configs 与 backbone_content_dir 解析（如 efficient_vmamba_base.ckpt）。
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _conda_env_lib_dirs() -> list[str]:
    """conda 环境的 lib 目录：CONDA_PREFIX/lib，或由 sys.executable 推断（无 CONDA_PREFIX 时也可用）。"""
    out: list[str] = []
    p = os.environ.get("CONDA_PREFIX")
    if p:
        lp = os.path.join(p, "lib")
        if os.path.isdir(lp):
            out.append(lp)
    exe = getattr(sys, "executable", None) or ""
    if exe and os.path.isfile(exe):
        # .../envs/<name>/bin/python -> .../envs/<name>/lib
        bindir = os.path.dirname(os.path.abspath(exe))
        if os.path.basename(bindir) == "bin":
            cand = os.path.join(os.path.dirname(bindir), "lib")
            if os.path.isdir(cand) and cand not in out:
                out.append(cand)
    return out


def _prepend_conda_lib_for_glibcxx() -> None:
    """
    selective_scan_cuda_core.so 依赖较新 GLIBCXX；系统 /lib64/libstdc++.so.6 过旧时会报错。
    1) 将 conda env 的 lib 置于 LD_LIBRARY_PATH 最前
    2) ctypes 预加载 conda 的 libstdc++.so.6（RTLD_GLOBAL），避免动态链接仍先解析到 /lib64
    """
    import ctypes

    lib_dirs = _conda_env_lib_dirs()
    if not lib_dirs:
        return
    primary = lib_dirs[0]
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


def _stub_fvcore_if_needed() -> None:
    if "fvcore.nn" in sys.modules:
        return
    nnm = types.ModuleType("fvcore.nn")

    class _DummyFlop:
        def __init__(self, *a, **k):
            pass

    nnm.FlopCountAnalysis = _DummyFlop
    nnm.flop_count_str = lambda *a, **k: ""
    nnm.flop_count = lambda *a, **k: {}
    nnm.parameter_count = lambda *a, **k: {}
    sys.modules["fvcore"] = types.ModuleType("fvcore")
    sys.modules["fvcore.nn"] = nnm


def _load_vmamba_efficient_module():
    """从 EfficientVMamba 源码目录加载 vmamba_efficient（避免与项目 models 包名冲突）。"""
    _stub_fvcore_if_needed()
    root = Path(__file__).resolve().parent.parent
    cls_root = root / "主干网络内容" / "EfficientVMamba-main" / "classification"
    if not cls_root.is_dir():
        raise FileNotFoundError(
            "未找到 EfficientVMamba 源码目录: {}".format(cls_root)
        )
    p = str(cls_root)
    if p not in sys.path:
        sys.path.insert(0, p)

    import importlib.util

    path = cls_root / "models" / "vmamba_efficient.py"
    name = "fusedhe_evm_vmamba_efficient"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 {}".format(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_EVM_MOD = None


def _get_backbone_class():
    global _EVM_MOD
    if _EVM_MOD is None:
        _EVM_MOD = _load_vmamba_efficient_module()
    return _EVM_MOD.Backbone_EfficientVSSM


# MiT-B4 对齐通道（与 mit / MambaVision 默认融合一致）
_TARGET_CH = [64, 128, 320, 512]
# EfficientVSSM-base: dims 96 -> [96,192,384,768]
_SRC_CH = [96, 192, 384, 768]


class EfficientVMambaFusedBackbone(nn.Module):
    """
    与 FusedEncoder 中 self.mit 接口一致：forward(x) -> (x1,x2,x3,x4)，各为 NCHW。
    """

    def __init__(self, cfg: dict, args):
        super().__init__()
        pre = bool(cfg.get("pretrained", True))
        wp = cfg.get("weights_path")
        depths = cfg.get("depths", [2, 2, 9, 2])
        embed_dim = cfg.get("embed_dim", 96)
        downsample_version = cfg.get("downsample_version", "v1")
        mlp_ratio = float(cfg.get("mlp_ratio", 0.0))
        d_state = int(cfg.get("d_state", 16))
        drop_path = float(cfg.get("drop_path_rate", 0.2))
        step_size = int(cfg.get("step_size", 2))

        B = _get_backbone_class()
        self.backbone = B(
            out_indices=(0, 1, 2, 3),
            pretrained=None,
            depths=list(depths),
            dims=int(embed_dim),
            num_classes=1000,
            drop_path_rate=drop_path,
            downsample_version=downsample_version,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            dt_rank="auto",
            ssm_ratio=float(cfg.get("ssm_ratio", 2.0)),
            step_size=step_size,
        )

        adapters = []
        for src, tgt in zip(_SRC_CH, _TARGET_CH):
            if src == tgt:
                adapters.append(nn.Identity())
            else:
                adapters.append(nn.Conv2d(src, tgt, kernel_size=1, bias=False))
        self.adapters = nn.ModuleList(adapters)

        if pre and wp and os.path.isfile(wp):
            self._load_pretrained(wp)
        elif pre and wp:
            print(
                "[Encoder/EfficientVMamba] 未找到权重文件: {}".format(wp),
                flush=True,
            )

    def _load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(sd, dict):
            return
        sd = {k.replace("module.module.", ""): v for k, v in sd.items()}
        info = self.backbone.load_state_dict(sd, strict=False)
        print(
            "[Encoder/EfficientVMamba] 预训练加载 missing={} unexpected={}".format(
                len(info.missing_keys), len(info.unexpected_keys)
            ),
            flush=True,
        )

    def forward(self, x: torch.Tensor):
        outs = self.backbone(x)
        if not isinstance(outs, (list, tuple)) or len(outs) != 4:
            raise RuntimeError("Backbone_EfficientVSSM 应返回 4 个 stage 特征")
        return tuple(self.adapters[i](outs[i]) for i in range(4))
