"""
RepViT backbone for FusedHE encoder1（替代 ResNet / ConvNeXt 局部纹理支路）。

默认 RepViT-M1.5：~14.0M params，四级输出通道 [64, 128, 256, 512]，
512×512 输入时空间尺寸 (128, 64, 32, 16)，与 ResNet-101 / ConvNeXt-V2 Nano 一致。

实现基于 THU-MIG/RepViT detection 主干（features.* 键名），可直接加载官方 ImageNet distill 权重：
  https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m1_5_distill_450e.pth

两种模式（与 ConvNeXtV2Backbone 一致）：
  no_reshape=True  (native_mode=true):  原生通道
  no_reshape=False (native_mode=false): 1×1 对齐到 ResNet [256, 512, 1024, 2048]
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from timm.layers import SqueezeExcite


def _make_divisible(v: int, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        a: int,
        b: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1,
    ):
        super().__init__()
        self.add_module(
            "c",
            nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class Residual(nn.Module):
    def __init__(self, m: nn.Module, drop: float = 0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device
            ).ge_(self.drop).div(1 - self.drop).detach()
        return x + self.m(x)


class RepVGGDW(nn.Module):
    def __init__(self, ed: int) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.bn = nn.BatchNorm2d(ed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x) + self.conv1(x) + x)


class RepViTBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        hidden_dim: int,
        oup: int,
        kernel_size: int,
        stride: int,
        use_se: int,
        use_hs: int,
    ):
        super().__init__()
        assert stride in (1, 2)
        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=inp,
                ),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                )
            )
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


class _RepViTFeatures(nn.Module):
    """官方 RepViT 特征提取器（无分类头）。"""

    def __init__(self, cfgs: Sequence[Sequence[int]], out_indices: Sequence[int]):
        super().__init__()
        self.cfgs = list(cfgs)
        self.out_indices = list(out_indices)

        input_channel = self.cfgs[0][2]
        patch_embed = nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers: List[nn.Module] = [patch_embed]
        input_ch = input_channel
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_ch * t, 8)
            layers.append(
                RepViTBlock(input_ch, exp_size, output_channel, k, s, use_se, use_hs)
            )
            input_ch = output_channel
        self.features = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) != 4:
            raise RuntimeError(
                "RepViT 应输出 4 级特征，实际 {} 级，out_indices={}".format(
                    len(outs), self.out_indices
                )
            )
        return tuple(outs)


def _cfgs_m1_1() -> List[List[int]]:
    return [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ]


def _cfgs_m1_5() -> List[List[int]]:
    return [
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 1, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 64, 0, 0, 1],
        [3, 2, 128, 0, 0, 2],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 1, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 128, 0, 0, 1],
        [3, 2, 256, 0, 1, 2],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 1, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 256, 0, 1, 1],
        [3, 2, 512, 0, 1, 2],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
        [3, 2, 512, 1, 1, 1],
        [3, 2, 512, 0, 1, 1],
    ]


def _cfgs_m2_3() -> List[List[int]]:
    return [
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 1, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 80, 0, 0, 1],
        [3, 2, 160, 0, 0, 2],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 1, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 160, 0, 0, 1],
        [3, 2, 320, 0, 1, 2],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 1, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 320, 0, 1, 1],
        [3, 2, 640, 0, 1, 2],
        [3, 2, 640, 1, 1, 1],
        [3, 2, 640, 0, 1, 1],
    ]


_VARIANT_SPECS: Dict[str, Dict] = {
    "m1_1": {
        "cfgs": _cfgs_m1_1,
        "out_indices": [2, 6, 20, 24],
        "stage_channels": [64, 128, 256, 512],
        "timm_name": "repvit_m1_1",
    },
    "m1_5": {
        "cfgs": _cfgs_m1_5,
        "out_indices": [4, 10, 36, 42],
        "stage_channels": [64, 128, 256, 512],
        "timm_name": "repvit_m1_5",
    },
    "m2_3": {
        "cfgs": _cfgs_m2_3,
        "out_indices": [6, 14, 50, 54],
        "stage_channels": [80, 160, 320, 640],
        "timm_name": "repvit_m2_3",
    },
}

RESNET_DIMS = [256, 512, 1024, 2048]


def _normalize_variant(variant: str) -> str:
    v = (variant or "m1_5").strip().lower().replace("-", "_")
    aliases = {
        "m1.5": "m1_5",
        "m1.1": "m1_1",
        "m2.3": "m2_3",
        "repvit_m1_5": "m1_5",
        "repvit_m1_1": "m1_1",
        "repvit_m2_3": "m2_3",
    }
    v = aliases.get(v, v)
    if v not in _VARIANT_SPECS:
        raise ValueError(
            "未知 RepViT variant: {!r}，可选: {}".format(variant, list(_VARIANT_SPECS))
        )
    return v


class RepViTBackbone(nn.Module):
    """
    RepViT 四级特征主干，接口与 ConvNeXtV2Backbone 一致。

    Args:
        variant: m1_1 | m1_5 | m2_3
        pretrained: 是否加载 ImageNet 预训练
        ckpt_path: 权重路径；None 时由 yaml weights_path 解析
        no_reshape: True → 原生通道；False → 1×1 对齐 ResNet 通道
    """

    def __init__(
        self,
        variant: str = "m1_5",
        pretrained: bool = True,
        ckpt_path: str | None = None,
        no_reshape: bool = True,
    ):
        super().__init__()
        self.variant = _normalize_variant(variant)
        spec = _VARIANT_SPECS[self.variant]
        self.no_reshape = bool(no_reshape)
        self.native_dims = list(spec["stage_channels"])

        self.backbone = _RepViTFeatures(
            cfgs=spec["cfgs"](),
            out_indices=spec["out_indices"],
        )

        if self.no_reshape:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(4)])
        else:
            self.adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(nd, rd, 1, bias=False),
                        nn.BatchNorm2d(rd),
                    )
                    for nd, rd in zip(self.native_dims, RESNET_DIMS)
                ]
            )

        if pretrained:
            self._load_pretrained(ckpt_path)

    def _load_pretrained(self, ckpt_path: str | None) -> None:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            print(
                "[RepViTBackbone] 未找到预训练权重: {}".format(ckpt_path),
                flush=True,
            )
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            sd = ckpt
        if not isinstance(sd, dict):
            print("[RepViTBackbone] 权重格式无法解析: {}".format(ckpt_path), flush=True)
            return

        cleaned = {}
        for k, v in sd.items():
            key = k.replace("module.", "").replace("module.module.", "")
            if key.startswith("backbone."):
                key = key[len("backbone.") :]
            if key.startswith(("classifier.", "head.", "head_dist.")):
                continue
            if not key.startswith("features."):
                continue
            cleaned[key] = v

        info = self.backbone.load_state_dict(cleaned, strict=False)
        matched = len(cleaned) - len(info.unexpected_keys)
        print(
            "[RepViTBackbone] 加载预训练 variant={} file={} matched≈{} missing={} unexpected={}".format(
                self.variant,
                os.path.basename(ckpt_path),
                matched,
                len(info.missing_keys),
                len(info.unexpected_keys),
            ),
            flush=True,
        )
        if info.missing_keys:
            print("  missing (前5): {}".format(info.missing_keys[:5]), flush=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = self.backbone(x)
        return tuple(self.adapters[i](outs[i]) for i in range(4))
