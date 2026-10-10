"""
主干网络配置：从 configs/*.yaml 加载，并与旧版 use_mv / use_conv / no_reshape 兼容。
约定：
  - encoder1 默认 res，encoder2 默认 mit；Fused 中 encoder1 → self.resnet，encoder2 → self.mit（与旧键名一致）。
  - 单编码器模式（enc_opt 为 mit 或 conv）仅使用 encoder1。
  - 预训练权重路径：yaml 中 weights_path 可为相对路径，优先在 项目/主干网络内容/预训练权重 查找。
  - 每个 yaml 可显式设置 backbone_lr；优化器里主干学习率优先取该值，与命令行 --lr（decoder 等）解耦，未写时回退到 --lr_backbone。
  - 旧版 checkpoint（legacy）：仅从 yaml + backbone_content_dir 解析权重，不再使用 mit_ckpt_path / mv_ckpt_path / conv_ckpt_path。
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "configs"
_BACKBONE_DIRNAME = "主干网络内容"
_PRETRAIN_WEIGHTS_SUBDIR = "预训练权重"


def _default_backbone_content_dir() -> Path:
    env = os.environ.get("FUSEDHE_BACKBONE_CONTENT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_PROJECT_ROOT / _BACKBONE_DIRNAME).resolve()


def _default_pretrained_weights_dir() -> Path:
    env = os.environ.get("FUSEDHE_PRETRAIN_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_default_backbone_content_dir() / _PRETRAIN_WEIGHTS_SUBDIR).resolve()


_PRESET_FILES = {
    "res": "res.yaml",
    "res50": "res50.yaml",
    "res101": "res.yaml",
    "mit": "mit.yaml",
    "convnext": "convnext.yaml",
    "repvit": "repvit.yaml",
    "MambaVision": "MambaVision.yaml",
    "EfficientVMamba": "EfficientVMamba.yaml",
    "GroupMamba": "GroupMamba.yaml",
}


def _is_truthy(v: Any) -> bool:
    """统一解析 0/1/true/false/yes/no 等开关值。"""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def project_root() -> Path:
    return _PROJECT_ROOT


def configs_dir() -> Path:
    return _CONFIG_DIR


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        if p is None:
            continue
        try:
            rp = Path(p).expanduser().resolve()
        except (OSError, RuntimeError):
            rp = Path(p).expanduser()
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        out.append(rp)
    return out


def _resolve_configs_dir(configs_dir: Optional[str] = None) -> Path:
    """configs 目录不存在时，回退到当前工程下的 configs/。"""
    candidates = _dedupe_paths(
        [
            Path(configs_dir) if configs_dir else None,
            _CONFIG_DIR,
            _PROJECT_ROOT / "configs",
            Path.cwd() / "configs",
            Path.cwd() / "rep" / "configs",
        ]
    )
    sentinel = "mit.yaml"
    for cand in candidates:
        if cand.is_dir() and (cand / sentinel).is_file():
            if configs_dir and str(cand) != str(Path(configs_dir).expanduser()):
                print(
                    "[encoder] configs_dir 不可用 ({})，回退: {}".format(
                        configs_dir, cand
                    ),
                    flush=True,
                )
            return cand
    return _CONFIG_DIR


def _resolve_backbone_content_dir(content_dir: Optional[str] = None) -> Path:
    """预训练权重目录不存在时，回退到当前工程默认目录。"""
    default_pretrain = _default_pretrained_weights_dir()
    default_content = _default_backbone_content_dir()
    candidates = _dedupe_paths(
        [
            Path(content_dir) if content_dir else None,
            default_pretrain,
            default_content,
            _PROJECT_ROOT / _PRETRAIN_WEIGHTS_SUBDIR,
            _PROJECT_ROOT / _BACKBONE_DIRNAME / _PRETRAIN_WEIGHTS_SUBDIR,
            Path.cwd() / _BACKBONE_DIRNAME / _PRETRAIN_WEIGHTS_SUBDIR,
        ]
    )
    for cand in candidates:
        if cand.is_dir():
            if content_dir and str(cand) != str(Path(content_dir).expanduser()):
                print(
                    "[encoder] backbone_content_dir 不可用 ({})，回退: {}".format(
                        content_dir, cand
                    ),
                    flush=True,
                )
            return cand
    return default_pretrain


def normalize_encoder_name(name: Optional[str]) -> str:
    if not name or not str(name).strip():
        return ""
    s = str(name).strip()
    low = s.lower()
    if low in ("mambavision", "mv"):
        return "MambaVision"
    if low in ("efficientvmamba", "efficient_vmamba", "evm", "e_vmamba"):
        return "EfficientVMamba"
    if low in ("groupmamba", "group_mamba", "gm"):
        return "GroupMamba"
    if low in ("res50", "r50"):
        return "res50"
    if low in ("res101", "r101"):
        return "res101"
    if low in ("res", "resnet", "r"):
        return "res"
    if low in ("mit",):
        return "mit"
    if low in ("convnext", "cn", "conv_next"):
        return "convnext"
    if low in ("repvit", "rv", "rep_vit"):
        return "repvit"
    raise ValueError(
        "未知主干类型: {!r}，可选: res, mit, convnext, repvit, MambaVision, EfficientVMamba, GroupMamba".format(
            name
        )
    )


def _load_yaml_file(filename: str, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    search_roots = _dedupe_paths(
        [
            base_dir,
            _CONFIG_DIR,
            _PROJECT_ROOT / "configs",
            Path.cwd() / "configs",
            Path.cwd() / "rep" / "configs",
        ]
    )
    tried: List[Path] = []
    for root in search_roots:
        path = root / filename
        tried.append(path)
        if path.is_file():
            if base_dir is not None and root != Path(base_dir).expanduser().resolve():
                print(
                    "[encoder] 配置文件 {} 未在 {} 找到，回退: {}".format(
                        filename, base_dir, path
                    ),
                    flush=True,
                )
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        "缺少配置文件: {}（已尝试: {}）".format(
            filename, ", ".join(str(p) for p in tried)
        )
    )


def load_preset(encoder_key: str, configs_base: Optional[Path] = None) -> Dict[str, Any]:
    """encoder_key: res | mit | convnext | MambaVision | EfficientVMamba | GroupMamba"""
    k = normalize_encoder_name(encoder_key)
    fn = _PRESET_FILES.get(k)
    if fn is None:
        raise KeyError("未知主干: {}".format(encoder_key))
    raw = _load_yaml_file(fn, base_dir=configs_base)
    return copy.deepcopy(raw)


def _resolve_path(
    p: Optional[str],
    *,
    backbone_content_dir: Optional[Path] = None,
) -> Optional[str]:
    """依次尝试：绝对路径、backbone_content_dir、项目根、当前工作目录。"""
    if p is None or p == "":
        return None
    path = Path(p)
    if path.is_absolute() and path.is_file():
        return str(path.resolve())
    roots = []
    pretrain_dir = _default_pretrained_weights_dir()
    if pretrain_dir.is_dir():
        roots.append(pretrain_dir)
    if backbone_content_dir is not None:
        roots.append(Path(backbone_content_dir))
    roots.append(_default_backbone_content_dir())
    roots.append(_PROJECT_ROOT)
    for root in roots:
        cand = (root / p).resolve()
        if cand.is_file():
            return str(cand)
    if path.is_file():
        return str(path.resolve())
    # 返回最可能路径便于报错信息
    return str((roots[0] / p).resolve()) if roots else str(path)


def _merge_legacy_into_cfg(
    enc: str, cfg: Dict[str, Any], args: argparse.Namespace, legacy: bool
) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if enc == "convnext":
        uc = getattr(args, "use_conv", 0)
        if uc == 1:
            cfg["native_mode"] = True
        elif uc == 2:
            cfg["native_mode"] = False
    elif enc == "MambaVision" and legacy:
        if getattr(args, "no_reshape", False):
            cfg["native_mode"] = True

    if not legacy:
        # 显式指定 encoder 时仍允许命令行覆盖权重（新训练流程）
        if enc == "mit" and getattr(args, "mit_ckpt_path", None):
            cfg.setdefault("weights_path", getattr(args, "mit_ckpt_path"))
        if enc == "convnext" and getattr(args, "conv_ckpt_path", None):
            cfg.setdefault("weights_path", getattr(args, "conv_ckpt_path"))
        if enc == "repvit" and getattr(args, "repvit_ckpt_path", None):
            cfg.setdefault("weights_path", getattr(args, "repvit_ckpt_path"))
        if enc == "MambaVision":
            if getattr(args, "mv_ckpt_path", None):
                cfg.setdefault("weights_path", getattr(args, "mv_ckpt_path"))
            if getattr(args, "mv_variant", None):
                cfg.setdefault("variant", getattr(args, "mv_variant"))
        return cfg

    # legacy：结构仍由 use_mv / no_reshape 推断；权重路径只来自 yaml，由 backbone_content_dir 解析
    return cfg


def resolve_encoder_choice(args: argparse.Namespace) -> Tuple[str, str, bool]:
    """
    解析 encoder1 / encoder2；若均未指定则 legacy=True，由 use_conv/use_mv 推断。
    """
    e1 = (getattr(args, "encoder1", None) or "").strip()
    e2 = (getattr(args, "encoder2", None) or "").strip()

    legacy = not e1 and not e2
    if legacy:
        if getattr(args, "use_conv", 0) in (1, 2):
            e1 = "convnext"
        else:
            e1 = "res"
        if getattr(args, "use_mv", False):
            e2 = "MambaVision"
        else:
            e2 = "mit"
    else:
        if not e1:
            e1 = "res"
        if not e2:
            e2 = "mit"

    e1n = normalize_encoder_name(e1)
    e2n = normalize_encoder_name(e2)
    return e1n, e2n, legacy


def resolve_encoder_args(args: argparse.Namespace) -> None:
    """
    填充 args.encoder1_type / encoder2_type、encoder1_cfg / encoder2_cfg，
    并在 enc_opt=fused 时调整 fusion/decoder 通道（与旧 apply_* 一致）。
    """
    if getattr(args, "_encoder_resolved", False):
        return

    # FDGM 单分支编码器：忽略 enc_opt / encoder1 / encoder2，
    # 强制单分支 MitDecoder 通道布局，主干固定为 GroupMamba（复用其 yaml 的权重/variant/通道）。
    if _is_truthy(getattr(args, "use_FDGM", 0)):
        args.enc_opt = "mit"
        args.encoder1 = "GroupMamba"
        args.encoder2 = ""

    e1, e2, legacy = resolve_encoder_choice(args)
    args.encoder1_type = e1
    args.encoder2_type = e2

    _base = _resolve_configs_dir(getattr(args, "configs_dir", None))
    content_dir = _resolve_backbone_content_dir(
        getattr(args, "backbone_content_dir", None)
    )
    args.configs_dir = str(_base)
    args.backbone_content_dir = str(content_dir)

    cfg1 = load_preset(e1, configs_base=_base)
    cfg2 = load_preset(e2, configs_base=_base)
    cfg1 = _merge_legacy_into_cfg(e1, cfg1, args, legacy)
    cfg2 = _merge_legacy_into_cfg(e2, cfg2, args, legacy)

    cfg1["weights_path"] = _resolve_path(cfg1.get("weights_path"), backbone_content_dir=content_dir)
    cfg2["weights_path"] = _resolve_path(cfg2.get("weights_path"), backbone_content_dir=content_dir)

    args.encoder1_cfg = cfg1
    args.encoder2_cfg = cfg2

    enc_opt = getattr(args, "enc_opt", "fused")
    if enc_opt == "fused":
        _apply_fusion_channel_layout(args, e1, cfg1, e2, cfg2)
    elif enc_opt == "mit":
        _apply_single_mit_layout(args, e1, cfg1)
    elif enc_opt == "conv":
        _apply_single_conv_layout(args, e1, cfg1)

    args._encoder_resolved = True


def backbone_lr_from_cfg(cfg: Optional[Dict[str, Any]], args: argparse.Namespace) -> float:
    """
    从对应主干的 yaml 读取 backbone_lr；未设置时与旧行为一致，使用 --lr_backbone（不再与 --lr 绑成同一数值，除非用户如此设置）。
    """
    if isinstance(cfg, dict):
        v = cfg.get("backbone_lr", None)
        if v is not None and v != "":
            return float(v)
    return float(getattr(args, "lr_backbone", 1e-5))


def build_optimizer_param_groups(model: Any, args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    为 Adam 构造 param_groups。fused：encoder1/encoder2 分别使用 configs 中各自 backbone_lr；
    单支路 mit/conv：整段 encoder 使用 encoder1 对应 yaml 的 backbone_lr；其余用 --lr。
    """
    enc_opt = getattr(args, "enc_opt", "fused")
    lr_dec = float(getattr(args, "lr", 1e-5))
    cfg1 = getattr(args, "encoder1_cfg", None) or {}
    cfg2 = getattr(args, "encoder2_cfg", None) or {}
    lr_b1 = backbone_lr_from_cfg(cfg1, args)
    lr_b2 = backbone_lr_from_cfg(cfg2, args)

    if enc_opt == "fused":
        p_resnet: List[Any] = []
        p_mit: List[Any] = []
        p_other: List[Any] = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("encoder.resnet"):
                p_resnet.append(p)
            elif n.startswith("encoder.mit"):
                p_mit.append(p)
            else:
                p_other.append(p)
        out: List[Dict[str, Any]] = []
        if p_resnet:
            out.append({"params": p_resnet, "lr": lr_b1})
        if p_mit:
            out.append({"params": p_mit, "lr": lr_b2})
        if p_other:
            out.append({"params": p_other, "lr": lr_dec})
        return out if out else [{"params": list(model.parameters()), "lr": lr_dec}]

    if enc_opt in ("mit", "conv"):
        p_enc: List[Any] = []
        p_other2: List[Any] = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("encoder."):
                p_enc.append(p)
            else:
                p_other2.append(p)
        out2: List[Dict[str, Any]] = []
        if p_enc:
            out2.append({"params": p_enc, "lr": lr_b1})
        if p_other2:
            out2.append({"params": p_other2, "lr": lr_dec})
        return out2 if out2 else [{"params": list(model.parameters()), "lr": lr_dec}]

    backbone_names = set(getattr(args, "lr_backbone_names", ("encoder.mit",)))
    backbone_params2: List[Any] = []
    other_params2: List[Any] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(bn) for bn in backbone_names):
            backbone_params2.append(p)
        else:
            other_params2.append(p)
    out3: List[Dict[str, Any]] = []
    if backbone_params2:
        out3.append(
            {
                "params": backbone_params2,
                "lr": float(getattr(args, "lr_backbone", lr_dec)),
            }
        )
    if other_params2:
        out3.append({"params": other_params2, "lr": lr_dec})
    return out3 if out3 else [{"params": list(model.parameters()), "lr": lr_dec}]


def _apply_fusion_channel_layout(
    args: argparse.Namespace,
    e1: str,
    cfg1: Dict[str, Any],
    e2: str,
    cfg2: Dict[str, Any],
) -> None:
    """双支路时的通道配置（仅 fused）。"""
    # MambaVision 原生（encoder2）与旧 no_reshape + tiny/base 一致
    if e2 == "MambaVision" and cfg2.get("native_mode"):
        variant = cfg2.get("variant", "tiny_ade20k")
        if variant == "base_1k":
            args.out_channel_ffb = [1024, 512, 256, 128]
            args.in_channel_decoderb = [2048, 1024, 512]
            args.out_channel_decoderb = [1024, 512, 256]
        else:
            args.out_channel_ffb = [640, 320, 160, 80]
            args.in_channel_decoderb = [1280, 640, 320]
            args.out_channel_decoderb = [640, 320, 160]

    # ConvNeXt 原生在 encoder1：与旧 use_conv=1 一致
    if e1 == "convnext" and cfg1.get("native_mode"):
        args.in_channel_ffb = [640, 320, 160, 80]

    # RepViT 原生在 encoder1：按 variant 设置 FAF/FeatureFusion 的 conv 侧输入通道
    if e1 == "repvit" and cfg1.get("native_mode"):
        stage_ch = cfg1.get("stage_channels") or [64, 128, 256, 512]
        args.in_channel_ffb = list(reversed(stage_ch))


def _apply_single_mit_layout(args: argparse.Namespace, e1: str, cfg: Dict[str, Any]) -> None:
    if e1 == "MambaVision" and cfg.get("native_mode"):
        variant = cfg.get("variant", "tiny_ade20k")
        if variant == "base_1k":
            args.out_channel_ffb = [1024, 512, 256, 128]
            args.in_channel_decoderb = [2048, 1024, 512]
            args.out_channel_decoderb = [1024, 512, 256]
        else:
            args.out_channel_ffb = [640, 320, 160, 80]
            args.in_channel_decoderb = [1280, 640, 320]
            args.out_channel_decoderb = [640, 320, 160]
    if e1 == "GroupMamba" and cfg.get("native_mode"):
        stage_ch = cfg.get("stage_channels") or [96, 192, 424, 512]
        args.out_channel_ffb = list(reversed(stage_ch))
        use_sdc_layout = bool(getattr(args, "native_proj_ff", False)) or (
            bool(getattr(args, "native_align_ff", False))
            and getattr(args, "single_decoder_channels", None)
        )
        if use_sdc_layout:
            args.single_decoder_channels = getattr(args, "single_decoder_channels", None) or [
                1024,
                640,
                256,
                128,
            ]
            args.in_channel_decoderb = [1024, 640, 256]
            args.out_channel_decoderb = [640, 256, 128]


def _conv_decoder_channels_from_ffb(
    in_channel_ffb: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """
    ConvDecoder：ConvDecoderBlock 将各级 ffb 输入减半；DecoderBlock 主路输入即为 ff 输出。
    例 RepViT native [512,256,128,64] → decoder [256,128,64] / [128,64,32]。
    """
    ff_out = [int(c) // 2 for c in in_channel_ffb]
    decoder_in = ff_out[:3]
    decoder_out = [int(c) // 2 for c in decoder_in]
    return decoder_in, decoder_out


def _apply_single_conv_layout(args: argparse.Namespace, e1: str, cfg: Dict[str, Any]) -> None:
    # ResNet / RepViT(channel_align→ResNet 通道) 共用 ConvDecoder 默认布局
    if e1 in ("res", "res50", "res101"):
        args.in_channel_ffb = [2048, 1024, 512, 256]
        args.in_channel_decoderb = [1024, 512, 256]
        args.out_channel_decoderb = [512, 256, 128]
        return
    if e1 == "repvit" and not cfg.get("native_mode"):
        args.in_channel_ffb = [2048, 1024, 512, 256]
        args.in_channel_decoderb = [1024, 512, 256]
        args.out_channel_decoderb = [512, 256, 128]
        return
    if e1 == "convnext" and cfg.get("native_mode"):
        args.in_channel_ffb = [640, 320, 160, 80]
        dec_in, dec_out = _conv_decoder_channels_from_ffb(args.in_channel_ffb)
        args.in_channel_decoderb = dec_in
        args.out_channel_decoderb = dec_out
    if e1 == "repvit" and cfg.get("native_mode"):
        stage_ch = cfg.get("stage_channels") or [64, 128, 256, 512]
        args.in_channel_ffb = list(reversed(stage_ch))
        dec_in, dec_out = _conv_decoder_channels_from_ffb(args.in_channel_ffb)
        args.in_channel_decoderb = dec_in
        args.out_channel_decoderb = dec_out


def add_encoder_args(parser: argparse.ArgumentParser) -> None:
    _pretrain_dir = _default_pretrained_weights_dir()
    parser.add_argument(
        "--encoder1",
        default="",
        type=str,
        help="第一支路主干：res | mit | convnext | repvit | MambaVision | EfficientVMamba | GroupMamba；空=res（旧版由 use_conv 推断）",
    )
    parser.add_argument(
        "--encoder2",
        default="",
        type=str,
        help="第二支路主干（仅 enc_opt=fused）：空=mit（旧版由 use_mv 推断）；可选 EfficientVMamba / GroupMamba",
    )
    parser.add_argument(
        "--configs_dir",
        default=str(_CONFIG_DIR),
        type=str,
        help="主干 yaml 配置目录，默认项目下 configs/",
    )
    parser.add_argument(
        "--backbone_content_dir",
        default=str(_pretrain_dir),
        type=str,
        help="主干预训练权重目录；yaml 中 weights_path 为相对路径时优先在此目录查找；"
        "默认 项目/主干网络内容/预训练权重；可用 FUSEDHE_PRETRAIN_WEIGHTS_DIR 或 FUSEDHE_BACKBONE_CONTENT_DIR 覆盖",
    )
    # FAFBlock（Frequency-Aware Fusion Block）开关 —— 支持 0/1/true/false
    _bool01 = lambda x: str(x).lower() not in ("0", "false", "no", "off", "")
    parser.add_argument(
        "--use_FAF",
        default=False,
        type=_bool01,
        help="开启 FAFBlock 替换 FeatureFusion（仅 enc_opt=fused 生效）；接受 0/1/true/false",
    )
    parser.add_argument(
        "--faf_smooth_kernel",
        default=1,
        type=int,
        help="FAFBlock 高频提取时 AvgPool 的核大小（奇数），浅层可适当放大至 5",
    )
    parser.add_argument(
        "--faf_lambda_res",
        default=0.5,
        type=float,
        help="FAFBlock 语义残差项 lambda（默认固定 0.5）",
    )
    parser.add_argument(
        "--faf_learnable_coef",
        default=True,
        type=_bool01,
        help="FAFBlock 的 alpha/beta/lambda 是否可学习（默认固定为设计文档中的层级先验值）",
    )
    # nFAF（New Frequency-Aware Fusion）—— FAFBlock 改进版，优先级高于 use_FAF。
    parser.add_argument(
        "--use_nFAF",
        default=False,
        type=_bool01,
        help="开启 nFAF 替换 FeatureFusion/FAFBlock（仅 enc_opt=fused 生效，优先级高于 use_FAF）；"
        " 接受 0/1/true/false。模块参数自动读取 models/Blocks/block.yaml 的 nFAF 段",
    )
    # GRFA（Gated Residual Frequency Augmentation）—— FeatureFusion 同构 base + 门控高频残差，优先级最高。
    parser.add_argument(
        "--use_GRFA",
        default=False,
        type=_bool01,
        help="开启 GRFA 替换 FeatureFusion/FAFBlock/nFAF/liteFAF/RFAF（仅 enc_opt=fused 生效，"
        "优先级最高）；接受 0/1/true/false。模块参数自动读取 models/Blocks/block.yaml 的 GRFA 段",
    )
    # RGHF（Residual Gated High-Frequency）—— 单主干 GroupMamba 各 stage 门控高频残差。
    parser.add_argument(
        "--use_RGHF",
        default=False,
        type=_bool01,
        help="开启 RGHF：在 GroupMamba 编码器每个 stage 末注入门控高频残差（仅 encoder1/encoder2=GroupMamba 生效）；"
        "接受 0/1/true/false。模块参数自动读取 models/Blocks/block.yaml 的 RGHF 段",
    )
    # FDGM（Frequency-Decoupled GroupMamba / FD-RGHF 编码器）—— 单分支编码器，最高优先级。
    parser.add_argument(
        "--use_FDGM",
        default=0,
        type=lambda x: 0 if str(x).strip().lower() in ("", "0", "false", "no", "off") else 1,
        help="FDGM：0=关闭；1/true=以 FD-RGHF 编码器启动。开启后忽略 enc_opt/encoder1/encoder2，"
        "强制走 GroupMamba 单分支 + 频域解耦高频增强 + MitDecoder；主干 variant/权重复用 "
        "configs/GroupMamba.yaml，频域超参读 models/Blocks/block.yaml 的 FDGM 段。",
    )
    # liteFAF（Lite Frequency-Aware Fusion）—— 显式分频 + 带内残差注入。
    parser.add_argument(
        "--use_liteFAF",
        default=False,
        type=_bool01,
        help="开启 liteFAF 替换 FeatureFusion/FAFBlock/nFAF（仅 enc_opt=fused 生效，"
        "优先级低于 use_GRFA、高于 use_nFAF/use_FAF）；接受 0/1/true/false。"
        "模块参数自动读取 models/Blocks/block.yaml 的 liteFAF 段",
    )
    # RFAF（Residual Frequency-Aware Fusion）—— FeatureFusion + DWT 频域残差补偿。
    parser.add_argument(
        "--use_RFAF",
        default=False,
        type=_bool01,
        help="开启 RFAF 替换 FeatureFusion/FAFBlock/nFAF（仅 enc_opt=fused 生效，"
        "优先级低于 use_liteFAF、高于 use_nFAF/use_FAF）；接受 0/1/true/false。"
        "模块参数自动读取 models/Blocks/block.yaml 的 RFAF 段",
    )
    # GSFB（Gated Skip Fusion Block）开关 —— 仅接受 0/1，不传 = 默认 0 关闭
    parser.add_argument(
        "--use_GSFB",
        default=0,
        type=int,
        choices=[0, 1],
        help="开启 GSFB 替换 decoder 跳跃 block1–3（空间门控 skip 注入）；mit/conv/fused 均支持；0=关闭，1=启用",
    )
    # FDSF（Frequency-Decoupled Skip Fusion）—— GSFB 升级版（spatial + channel gate + refine）
    parser.add_argument(
        "--use_FDSF",
        default=False,
        type=_bool01,
        help="开启 FDSF 替换 decoder 跳跃 block1–3；与 use_GSFB 同坑位但 FDSF 优先，mit/conv/fused 均支持。"
        " 模块参数自动读取 models/Blocks/block.yaml 的 FDSF 段",
    )
    parser.add_argument(
        "--use_DMSN",
        default=0,
        type=_bool01,
        help="DMSN：0=关闭；1/true=启用 Directional Mamba Semantic Neck。"
        " 在 decoder ff1(stage4) 输出后、第一次 upsample 前生效（mit/conv/fused 均支持），"
        "参数读取 block.yaml.DMSN。",
    )
    def _parse_use_dpld(raw: Any) -> int:
        """0=关；1/true=读缓存或 ndsm_stats.pickle；2=每次启动遍历 train+val 的 nDSM 在线 quantile（不读不写 bin 缓存）。"""
        if raw is None:
            return 0
        s = str(raw).strip().lower()
        if s in ("", "0", "false", "no", "off"):
            return 0
        if s in ("1", "true", "yes", "on"):
            return 1
        if s == "2":
            return 2
        try:
            v = int(raw)
        except (TypeError, ValueError) as e:
            raise argparse.ArgumentTypeError("use_DPLD 无法解析: {!r}".format(raw)) from e
        if v in (0, 1, 2):
            return v
        raise argparse.ArgumentTypeError("use_DPLD 仅允许 0 / 1(true) / 2，收到: {!r}".format(raw))

    # DPLD（Dataset-Prior Local Distribution Head）—— 替代 use_condition 的新高度头
    parser.add_argument(
        "--use_DPLD",
        default=0,
        type=_parse_use_dpld,
        help="DPLD：0=关闭；1/true=读 dpld bin 缓存或 ndsm_stats.pickle 构建 quantile；"
        "2=每次启动遍历 train.txt+val.txt 所列 nDSM 在线重建 quantile edges（跳过 pickle/缓存读写）。",
    )
    # 以下三个允许在命令行/yaml 显式覆盖 block.yaml 的同名 DPLD 字段；未传时由 block.yaml 决定
    parser.add_argument(
        "--dpld_n_bins", default=None, type=int,
        help="DPLD bin 总数；默认走 block.yaml.DPLD.n_bins",
    )
    parser.add_argument(
        "--dpld_ground_bin_max_m", default=None, type=float,
        help="DPLD 地面 bin 上界（米）；默认走 block.yaml.DPLD.ground_bin_max_m",
    )
    parser.add_argument(
        "--dpld_fallback", default=None, type=str, choices=[None, "uniform", "none"],
        help="未找到 ndsm_stats.pickle 时的回退：uniform 或 none。默认走 block.yaml",
    )

    # TBWH（Truncated Bin Weighted Height Head）—— 无先验均匀分区高度头
    # 与 DPLD/use_condition 互斥；n_bins 与 max_height 按 dataset_name 自动从 block.yaml 读取
    parser.add_argument(
        "--use_TBWH",
        default=0,
        type=lambda x: 0 if str(x).strip().lower() in ("", "0", "false", "no", "off") else 1,
        help="TBWH：0=关闭；1/true=启用均匀等宽 bin 高度头（n_bins/max_height 由 dataset_name 自动查 block.yaml.TBWH）。",
    )
    parser.add_argument(
        "--use_FHRM",
        default=0,
        type=int,
        choices=[0, 1],
        help="FHRM：0=关闭；1=启用 Frequency-aware Height Refinement（参数见 models/Blocks/block.yaml FHRM；"
        "仅与原版 DepthHead 组合，use_HL/DPLD/TBWH/use_condition 时自动跳过）。",
    )

    # 旧版参数：保留以兼容 checkpoint 与脚本，不在帮助中展示
    parser.add_argument("--use_mv", default=0, type=lambda x: str(x).lower() not in ("0", "false", "no"), help=argparse.SUPPRESS)
    parser.add_argument("--no_reshape", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mv_variant", default="tiny_ade20k", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--mv_ckpt_path", default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mit_ckpt_path",
        default=str(_default_pretrained_weights_dir() / "mit_b4.pth"),
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--use_conv", default=0, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--conv_ckpt_path", default=None, type=str, help=argparse.SUPPRESS)
