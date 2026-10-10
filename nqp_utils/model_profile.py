"""参数量与稠密前向 GFLOPs 估计（thop / torch.profiler）。"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_model_gflops_note(prof: dict) -> str:
    return str(prof.get("gflops_backend", "unknown"))


def profile_model(
    model: nn.Module,
    height: int,
    width: int,
    inner: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """返回 {"params", "gflops", "gflops_backend"}；失败时 gflops 为 nan。"""
    target = inner if inner is not None else model
    params = count_parameters(model)
    out: dict[str, Any] = {
        "params": int(params),
        "gflops": float("nan"),
        "gflops_backend": "none",
    }

    if device is None:
        # 默认勿用 cuda:0：若外层已 model.to(cuda:N)，inner 与数据在同一卡，此处再 .to("cuda") 会把子模块挪到 0 号卡，训练时出现 device mismatch。
        try:
            device = next(target.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    was_training = target.training
    target.eval()
    dummy = torch.zeros(1, 3, int(height), int(width), device=device, dtype=torch.float32)

    # 1) thop
    try:
        from thop import profile

        m = target
        if device.type == "cuda":
            m = m.to(device)
        with torch.no_grad():
            macs, _ = profile(m, inputs=(dummy,), verbose=False)
        out["gflops"] = float(macs) / 1e9 * 2.0
        out["gflops_backend"] = "thop"
        if was_training:
            target.train()
        return out
    except Exception:
        pass

    # 2) torch.profiler（with_flops，需 PyTorch 支持该统计）
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        m = target
        if device.type == "cuda":
            m = m.to(device)
        x = dummy.to(next(m.parameters()).device)
        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = m(x)
        flops = sum(getattr(e, "flops", 0) or 0 for e in prof.events())
        if flops > 0:
            out["gflops"] = float(flops) / 1e9
            out["gflops_backend"] = "torch.profiler"
            if was_training:
                target.train()
            return out
    except Exception as e:
        warnings.warn(f"nqp_utils.profile_model: torch.profiler 失败: {e}", stacklevel=2)

    if was_training:
        target.train()
    return out
