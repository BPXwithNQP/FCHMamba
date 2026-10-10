"""
在 import torch / numpy 之前优先绑定 conda 环境的 libstdc++（GLIBCXX_3.4.29 等），
避免 selective_scan_cuda_core 等扩展仍解析到系统 /lib64 的旧库。

用法：在入口脚本最前面（仅晚于 import os）执行：
    import utils.conda_glibcxx  # noqa: F401
或：
    from utils.conda_glibcxx import ensure_conda_libstdcxx_first
    ensure_conda_libstdcxx_first()
"""
from __future__ import annotations

import ctypes
import os
import sys


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


def ensure_conda_libstdcxx_first() -> None:
    """幂等；必须在 import torch 之前调用。"""
    if getattr(ensure_conda_libstdcxx_first, "_done", False):
        return

    lib_dirs = _conda_env_lib_dirs()
    if not lib_dirs:
        ensure_conda_libstdcxx_first._done = True
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

    ensure_conda_libstdcxx_first._done = True


# 被「import utils.conda_glibcxx」时立即执行
ensure_conda_libstdcxx_first()
