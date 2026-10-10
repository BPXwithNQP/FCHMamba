"""
ConvNeXt V2 Nano Backbone for FusedHE, replacing ResNet101.

Nano config: depths=[2,2,8,2], dims=[80,160,320,640], ~15.6M params
Pretrained: convnextv2_nano_1k_224_ema.pt (ImageNet-1K fine-tuned, 224x224)

ConvNeXt V2 is fully convolutional (no position embedding), so 224-pretrained
weights work directly at 512x512 without any interpolation or adaptation.

Two modes:
  no_reshape=True  (use_conv=1): native channels [80,160,320,640]
                                  downstream FeatureFusion adapts via conv layers
  no_reshape=False (use_conv=2): 1x1 conv adapters → ResNet channels [256,512,1024,2048]
                                  in_channel_ffb stays [2048,1024,512,256]

Spatial sizes at 512 input: (128, 64, 32, 16) — identical to ResNet101.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from timm.models.layers import trunc_normal_, DropPath


# ── Self-contained components from official ConvNeXt V2 (避免导入 MinkowskiEngine) ──

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first data formats."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    """Global Response Normalization layer."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXt V2 Block: depthwise conv → LayerNorm → pointwise → GELU → GRN → pointwise."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N,C,H,W) → (N,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N,H,W,C) → (N,C,H,W)
        return shortcut + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2 feature extractor (no classification head)."""
    def __init__(self, in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            ))
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ── Backbone wrapper ──

NANO_DEPTHS = [2, 2, 8, 2]
NANO_DIMS = [80, 160, 320, 640]
RESNET_DIMS = [256, 512, 1024, 2048]


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXt V2 Nano backbone, returns 4-stage hierarchical features.

    Output order: (stage0, stage1, stage2, stage3) — same as ResNet (layer1..layer4).
      Spatial sizes at 512 input: (128, 64, 32, 16).
      Native channels: [80, 160, 320, 640].

    Args:
        pretrained: load ImageNet-1K fine-tuned weights
        ckpt_path:  path to .pt file; None → auto-detect project root
        no_reshape: True  → native channels (use_conv=1)
                    False → 1x1 adapters to [256,512,1024,2048] (use_conv=2)
    """
    def __init__(self, pretrained=True, ckpt_path=None, no_reshape=False):
        super().__init__()
        self.no_reshape = no_reshape
        self.native_dims = list(NANO_DIMS)

        self.backbone = ConvNeXtV2(
            in_chans=3, depths=NANO_DEPTHS, dims=NANO_DIMS, drop_path_rate=0.,
        )

        if no_reshape:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(4)])
        else:
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(nd, rd, 1, bias=False),
                    nn.BatchNorm2d(rd),
                ) for nd, rd in zip(NANO_DIMS, RESNET_DIMS)
            ])

        if pretrained:
            self._load_pretrained(ckpt_path)

    def _load_pretrained(self, ckpt_path):
        if ckpt_path is None:
            ckpt_path = str(
                Path(__file__).resolve().parent.parent / 'convnextv2_nano_1k_224_ema.pt')
        if not os.path.isfile(ckpt_path):
            print(f"[ConvNeXtV2Backbone] 未找到预训练权重: {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt.get('model', ckpt)
        sd = {k: v for k, v in sd.items() if not k.startswith(('head.', 'norm.'))}
        info = self.backbone.load_state_dict(sd, strict=False)
        print(f"[ConvNeXtV2Backbone] 加载预训练权重: {os.path.basename(ckpt_path)}")
        if info.missing_keys:
            print(f"  missing: {info.missing_keys[:5]}...")
        if info.unexpected_keys:
            print(f"  unexpected: {info.unexpected_keys[:5]}...")

    def forward(self, x):
        """
        Returns:
            tuple of 4 tensors, ordered (stage0, stage1, stage2, stage3).
            Native channels [80,160,320,640] or adapted [256,512,1024,2048].
        """
        outs = []
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            outs.append(x)
        return tuple(self.adapters[i](outs[i]) for i in range(4))
