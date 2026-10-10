"""
主干 yaml 中 freeze 字段：训练前 N 个 epoch 冻结「已加载预训练权重」的主干参数。

GroupMamba / FDGM 约定：
  - 冻结：backbone 的 patch_embed / blocks / norm（来自 groupmamba_base_ema.pth 等）
  - 不冻结：adapters、RGHF、FDGM 的 LocalRefiner / Laplace / 高频分支 / 融合门控等新增模块

CLI --freeze / warm_start 自动冻结：
  - warm_start 时对比 ckpt 与当前 args，找出「新增模块」
  - 前 N epoch 仅训练新增模块参数，其余冻结；N+1 起解冻全网络
  - 未显式传 --freeze 且存在新增模块时，默认 N=5
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

# warm_start 未显式 --freeze 时的默认冻结 epoch 数
_DEFAULT_WARM_START_FREEZE_EPOCHS = 5

# 已打印过解冻日志的 (encoder_key, freeze_epochs)，避免每 epoch 刷屏
_UNFREEZE_LOGGED: set = set()
_NETWORK_UNFREEZE_LOGGED: set = set()


def _truthy_flag(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def _argv_explicit_longopt_dests(argv=None) -> set:
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


def detect_modules_in_state(state) -> Dict[str, bool]:
    """从 state_dict 键名判断 ckpt 已含哪些可选模块。"""
    keys = tuple(state.keys())
    return {
        "rghf": any(k.startswith("encoder.rghf.") for k in keys),
        "dmsn": any(k.startswith("decoder.dmsn.") for k in keys),
        "gsfb": any(k.startswith("decoder.block1.align_d") for k in keys),
        "fdsf": any(k.startswith("decoder.block1.gamma_d") for k in keys),
        "condition": any(k.startswith("condition_heads.") for k in keys),
        "depth_head": any(k.startswith("depth_head.") for k in keys),
    }


def infer_new_module_trainable_prefixes(args, ckpt_modules: Dict[str, bool]) -> List[str]:
    """相对 ckpt，当前训练配置中「新增」模块对应的 param 前缀（仅这些在冻结期可训）。"""
    prefixes: List[str] = []
    if _truthy_flag(getattr(args, "use_RGHF", 0)) and not ckpt_modules.get("rghf", False):
        prefixes.append("encoder.rghf.")
    if _truthy_flag(getattr(args, "use_DMSN", 0)) and not ckpt_modules.get("dmsn", False):
        prefixes.append("decoder.dmsn.")
    use_gsfb = int(getattr(args, "use_GSFB", 0) or 0) == 1
    has_skip_fusion = ckpt_modules.get("gsfb", False) or ckpt_modules.get("fdsf", False)
    if use_gsfb and not has_skip_fusion:
        for p in ("decoder.block1.", "decoder.block2.", "decoder.block3."):
            if p not in prefixes:
                prefixes.append(p)
    if bool(getattr(args, "use_condition", False)) and not ckpt_modules.get("condition", False):
        prefixes.append("condition_heads.")
    return prefixes


def _prefixes_to_labels(prefixes: List[str]) -> List[str]:
    labels = []
    for p in prefixes:
        if p.startswith("encoder.rghf."):
            labels.append("RGHF")
        elif p.startswith("decoder.dmsn."):
            labels.append("DMSN")
        elif p.startswith("decoder.block"):
            labels.append("GSFB")
        elif p.startswith("condition_heads."):
            labels.append("Condition")
        else:
            labels.append(p)
    seen = []
    for lb in labels:
        if lb not in seen:
            seen.append(lb)
    return seen


def resolve_network_freeze_epochs(args, new_prefixes: List[str], *, argv=None) -> int:
    """解析冻结 epoch 数：显式 --freeze 优先；否则 warm_start 有新增模块时默认 5。"""
    explicit = "freeze" in _argv_explicit_longopt_dests(argv)
    try:
        raw = int(getattr(args, "freeze", 0) or 0)
    except (TypeError, ValueError):
        raw = 0
    if explicit:
        return max(0, raw)
    warm = bool(getattr(args, "warm_start", "") or getattr(args, "_warm_start_configured", False))
    if warm and new_prefixes:
        return _DEFAULT_WARM_START_FREEZE_EPOCHS
    return 0


def configure_warm_start_network_freeze(
    args,
    state,
    *,
    verbose: bool = True,
) -> None:
    """warm_start 加载权重后：对比 ckpt 与 CLI，自动设置 freeze epoch 与可训练前缀。"""
    ckpt_mod = detect_modules_in_state(state)
    prefixes = infer_new_module_trainable_prefixes(args, ckpt_mod)
    setattr(args, "_warm_start_ckpt_modules", ckpt_mod)
    setattr(args, "_freeze_trainable_prefixes", prefixes)

    n = resolve_network_freeze_epochs(args, prefixes)
    if n > 0 and not prefixes:
        if verbose:
            print(
                "[warm_start] 未检测到相对 ckpt 的新增模块，忽略 freeze（无 allowlist）",
                flush=True,
            )
        n = 0
    setattr(args, "freeze", n)

    if not verbose:
        return
    labels = _prefixes_to_labels(prefixes)
    if prefixes and n > 0:
        print(
            "[warm_start] 相对 ckpt 新增模块: {} | 冻结 {} epoch，仅训练: {}".format(
                ", ".join(labels) if labels else "（无）",
                n,
                prefixes,
            ),
            flush=True,
        )
    elif prefixes and n == 0:
        print(
            "[warm_start] 检测到新增模块 {}，但 --freeze=0，全网络同时训练".format(
                ", ".join(labels)
            ),
            flush=True,
        )
    else:
        print("[warm_start] ckpt 已含当前启用模块，不启用阶段性冻结", flush=True)


def _freeze_epochs_from_cfg(cfg: Optional[Dict[str, Any]]) -> int:
    if not isinstance(cfg, dict):
        return 0
    v = cfg.get("freeze", 0)
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def collect_groupmamba_freeze_plan(args) -> List[Tuple[str, str, int]]:
    """返回 [(encoder_key, param_prefix, freeze_epochs), ...]。

    encoder_key 仅用于日志；param_prefix 为 state_dict 名前缀（如 encoder.mit.）。
    """
    enc_opt = getattr(args, "enc_opt", "fused")
    use_fdgm = bool(int(getattr(args, "use_FDGM", 0) or 0))
    plan: List[Tuple[str, str, int]] = []

    if use_fdgm:
        cfg = getattr(args, "encoder1_cfg", None) or {}
        epochs = _freeze_epochs_from_cfg(cfg)
        if epochs > 0:
            plan.append(("FDGM/GroupMamba", "encoder.", epochs))
        return plan

    if enc_opt == "fused":
        e2 = getattr(args, "encoder2_type", "")
        cfg2 = getattr(args, "encoder2_cfg", None) or {}
        if e2 == "GroupMamba":
            epochs = _freeze_epochs_from_cfg(cfg2)
            if epochs > 0:
                plan.append(("encoder2/GroupMamba", "encoder.mit.", epochs))
        e1 = getattr(args, "encoder1_type", "")
        cfg1 = getattr(args, "encoder1_cfg", None) or {}
        if e1 == "GroupMamba":
            epochs = _freeze_epochs_from_cfg(cfg1)
            if epochs > 0:
                plan.append(("encoder1/GroupMamba", "encoder.resnet.", epochs))
    elif enc_opt in ("mit", "conv"):
        e1 = getattr(args, "encoder1_type", "")
        cfg1 = getattr(args, "encoder1_cfg", None) or {}
        if e1 == "GroupMamba":
            epochs = _freeze_epochs_from_cfg(cfg1)
            if epochs > 0:
                plan.append(("encoder/GroupMamba", "encoder.", epochs))

    return plan


def is_groupmamba_pretrained_param(name: str, prefix: str) -> bool:
    """判断参数名是否属于 GroupMamba 预训练主干（patch_embed / blocks / norm）。"""
    if not name.startswith(prefix):
        return False
    rel = name[len(prefix):]

    # GroupMambaFusedBackbone：encoder[.mit|.resnet].backbone.patch_embed* / block* / norm*
    if rel.startswith("backbone."):
        parts = rel.split(".")
        if len(parts) >= 2 and parts[0] == "backbone":
            sub = parts[1]
            if sub.startswith("patch_embed") or sub.startswith("block") or sub.startswith("norm"):
                return True
        return False

    # FDGMEncoder：encoder.stages.{i}.patch_embed / blocks / norm
    if rel.startswith("stages."):
        for token in (".patch_embed.", ".blocks.", ".norm."):
            if token in rel:
                return True
        return False

    return False


def network_freeze_epochs(args) -> int:
    try:
        n = int(getattr(args, "freeze", 0) or 0)
    except (TypeError, ValueError):
        n = 0
    return max(0, n)


def network_freeze_active(args, epoch: int) -> bool:
    n = network_freeze_epochs(args)
    return n > 0 and int(epoch) <= n


def default_network_freeze_trainable_prefixes(args) -> List[str]:
    """全网络冻结期间仍允许训练的模块前缀（warm_start 自动推断或手动指定）。"""
    custom = getattr(args, "_freeze_trainable_prefixes", None)
    if custom is not None:
        return list(custom)
    prefixes: List[str] = []
    if bool(getattr(args, "use_condition", False)):
        prefixes.append("condition_heads.")
    extra = getattr(args, "freeze_trainable_prefixes", None)
    if extra:
        for p in extra:
            p = str(p).strip()
            if p and p not in prefixes:
                prefixes.append(p)
    return prefixes


def apply_network_freeze(
    model: nn.Module,
    args,
    epoch: int,
    *,
    verbose: bool = True,
) -> None:
    """CLI --freeze N：前 N 个 epoch 冻结除 allowlist 外全部参数（warm_start 自动推断 allowlist）。"""
    n = network_freeze_epochs(args)
    if n <= 0:
        return

    prefixes = default_network_freeze_trainable_prefixes(args)
    in_freeze = int(epoch) <= n
    frozen_n = 0
    train_n = 0
    for name, param in model.named_parameters():
        if in_freeze:
            trainable = any(name.startswith(p) for p in prefixes)
            param.requires_grad = trainable
            if trainable:
                train_n += param.numel()
            else:
                frozen_n += param.numel()
        else:
            param.requires_grad = True
            train_n += param.numel()

    if verbose and in_freeze:
        print(
            "[freeze] 全网络冻结 epoch {}/{} | frozen={} trainable={} allowlist={}".format(
                epoch, n, frozen_n, train_n, prefixes or "[]"
            ),
            flush=True,
        )
    elif epoch == n + 1:
        log_key = n
        if log_key not in _NETWORK_UNFREEZE_LOGGED:
            _NETWORK_UNFREEZE_LOGGED.add(log_key)
            print(
                "[freeze] epoch={} > {}：已解冻全网络（trainable={}）".format(
                    epoch, n, train_n
                ),
                flush=True,
            )


def _grfa_blocks_encoder_freeze(args) -> bool:
    phase = str(getattr(args, "grfa_train_phase", "none")).lower()
    if phase == "residual":
        return True
    if phase == "finetune" and not bool(getattr(args, "grfa_finetune_backbone", False)):
        return True
    return False


def apply_groupmamba_pretrained_freeze(
    model: nn.Module,
    args,
    epoch: int,
    *,
    verbose: bool = True,
) -> None:
    """按 yaml freeze 与当前 epoch 设置 GroupMamba 预训练部分的 requires_grad。"""
    if network_freeze_active(args, epoch):
        return
    if _grfa_blocks_encoder_freeze(args):
        return

    plan = getattr(args, "_groupmamba_freeze_plan", None)
    if plan is None:
        plan = collect_groupmamba_freeze_plan(args)
        setattr(args, "_groupmamba_freeze_plan", plan)

    if not plan:
        return

    frozen_n = 0
    trainable_new_n = 0
    for name, param in model.named_parameters():
        for enc_key, prefix, freeze_epochs in plan:
            if not name.startswith(prefix):
                continue
            if is_groupmamba_pretrained_param(name, prefix):
                should_freeze = epoch <= freeze_epochs
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_n += param.numel()
            elif epoch <= freeze_epochs:
                if not param.requires_grad:
                    param.requires_grad = True
                trainable_new_n += param.numel()
            break

    if verbose and plan:
        parts = ["{}:{}ep".format(k, e) for k, _, e in plan]
        print(
            "[freeze] GroupMamba 预训练主干冻结计划 {} | "
            "frozen_pretrained={} trainable_new_under_encoder={}".format(
                ", ".join(parts), frozen_n, trainable_new_n
            ),
            flush=True,
        )

    for enc_key, _prefix, freeze_epochs in plan:
        log_key = (enc_key, freeze_epochs)
        if epoch == freeze_epochs + 1 and log_key not in _UNFREEZE_LOGGED:
            _UNFREEZE_LOGGED.add(log_key)
            if verbose:
                print(
                    "[freeze] epoch={} > {}：已解冻 {} 的 GroupMamba 预训练主干".format(
                        epoch, freeze_epochs, enc_key
                    ),
                    flush=True,
                )
