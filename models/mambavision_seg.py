"""
MambaVision-Tiny Backbone for FusedHE  (ADE20K 512×512 segmentation pretrained)

Tiny 配置: dim=80, in_dim=32, depths=[1,3,8,4], 输出通道 [80, 160, 320, 640]
通道适配 → MiT 对齐 [64, 128, 320, 512]:
  Stage 1: 80  → 64  (1×1 conv)
  Stage 2: 160 → 128 (1×1 conv)
  Stage 3: 320 → 320 (恒等映射，完美匹配)
  Stage 4: 640 → 512 (1×1 conv, 仅 1.25× 压缩)

相比 MambaVision-B-1K 224px 分类预训练的优势:
  - 512×512 原生分辨率, 无需位置编码适配
  - ADE20K 语义分割预训练, 特征更适合密集预测
  - Stage 3 通道完美匹配, Stage 4 压缩大幅降低
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import types
from pathlib import Path

if not hasattr(F, 'scaled_dot_product_attention'):
    def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                      is_causal=False, scale=None, **kwargs):
        scale = scale or (q.size(-1) ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        if is_causal and attn.size(-2) == attn.size(-1):
            causal_mask = torch.triu(
                torch.ones(attn.shape[-2], attn.shape[-1],
                           device=attn.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return attn @ v
    F.scaled_dot_product_attention = _scaled_dot_product_attention

try:
    from mambavision.models.mamba_vision import MambaVision, window_partition, window_reverse
    _HAS_MAMBAVISION = True
except ImportError:
    _HAS_MAMBAVISION = False


def _load_ade20k_checkpoint(model, path, strict=False):
    """加载 MMSeg ADE20K checkpoint, 自动处理 mmengine 依赖和 backbone. 前缀"""

    # mmengine mock (checkpoint 由 MMEngine 保存, unpickle 时需要对应的类)
    class _AnyMeta(type):
        def __getattr__(cls, name):
            return cls

    class _Any(metaclass=_AnyMeta):
        def __init__(self, *a, **kw):
            if a and isinstance(a[0], dict):
                self.__dict__.update(a[0])
            for k, v in kw.items():
                setattr(self, k, v)
        def __getattr__(self, name):
            return _Any()
        def __call__(self, *a, **kw):
            return _Any()
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
        def __reduce__(self):
            return (_Any, ())

    class _MockFinder:
        def find_module(self, fullname, path=None):
            return self if fullname.startswith('mmengine') else None
        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            class Mod(types.ModuleType):
                def __getattr__(self, name):
                    return _Any
            m = Mod(fullname)
            m.__path__ = [fullname]
            m.__loader__ = self
            sys.modules[fullname] = m
            return m

    finder = _MockFinder()
    sys.meta_path.insert(0, finder)
    try:
        ckpt = torch.load(path, map_location='cpu')
    finally:
        sys.meta_path.remove(finder)

    sd = ckpt.get('state_dict', ckpt) if hasattr(ckpt, 'get') else ckpt

    backbone_sd = {}
    for k, v in sd.items():
        if not hasattr(v, 'shape'):
            continue
        if k.startswith('backbone.'):
            backbone_sd[k[len('backbone.'):]] = v

    info = model.load_state_dict(backbone_sd, strict=strict)
    n_loaded = len(backbone_sd) - len(info.unexpected_keys)
    print(f"[MambaVision-Tiny-ADE20K] 加载 {n_loaded}/{len(backbone_sd)} 个 backbone 权重 "
          f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)})")
    if info.missing_keys:
        print(f"  missing (前10): {info.missing_keys[:10]}")
    return ckpt


class MambaVisionTinyBackbone(nn.Module):
    """
    MambaVision-Tiny 主干 (ADE20K 512×512 分割预训练)
    默认 4 stage 经 1×1 与 MiT 对齐 [64,128,320,512]；no_reshape=True 时保持原生 [80,160,320,640]。
    """
    # Tiny 配置
    DIM = 80
    IN_DIM = 32
    DEPTHS = [1, 3, 8, 4]
    NUM_HEADS = [2, 4, 8, 16]
    WINDOW_SIZE = [8, 8, 64, 32]
    OUT_CHANNELS = [80, 160, 320, 640]
    TARGET_CHANNELS = [64, 128, 320, 512]

    def __init__(self, pretrained=True, ckpt_path=None, no_reshape=False):
        super().__init__()
        self.no_reshape = no_reshape
        if not _HAS_MAMBAVISION:
            raise ImportError(
                "请先安装 mambavision 及依赖:\n"
                "  pip install mambavision mamba-ssm einops"
            )

        self.backbone = MambaVision(
            dim=self.DIM,
            in_dim=self.IN_DIM,
            depths=self.DEPTHS,
            num_heads=self.NUM_HEADS,
            window_size=self.WINDOW_SIZE,
            mlp_ratio=4,
            drop_path_rate=0.3,
            layer_scale=None,
            layer_scale_conv=None,
            in_chans=3,
            num_classes=0,
        )

        # ADE20K checkpoint 包含 outnorm 层, 添加到模型中以复用预训练权重
        self.outnorms = nn.ModuleList([
            nn.LayerNorm(ch) for ch in self.OUT_CHANNELS
        ])

        # 通道适配: [80,160,320,640] → [64,128,320,512]；no_reshape 时全部 Identity
        if no_reshape:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(4)])
        else:
            adapters = []
            for src, tgt in zip(self.OUT_CHANNELS, self.TARGET_CHANNELS):
                if src == tgt:
                    adapters.append(nn.Identity())
                else:
                    adapters.append(nn.Conv2d(src, tgt, 1))
            self.adapters = nn.ModuleList(adapters)

        if pretrained:
            path = ckpt_path or self._default_ckpt_path()
            if os.path.isfile(path):
                self._load_pretrained(path)
            else:
                print(f"[MambaVision-Tiny] 未找到权重 {path}")

    def _default_ckpt_path(self):
        return str(Path(__file__).resolve().parent.parent
                   / "mamba_vision_160k_ade20k-512x512_tiny.pth")

    def _load_pretrained(self, path):
        """加载 ADE20K checkpoint 到 backbone + outnorms"""

        class _AnyMeta(type):
            def __getattr__(cls, name):
                return cls
        class _Any(metaclass=_AnyMeta):
            def __init__(self, *a, **kw):
                if a and isinstance(a[0], dict):
                    self.__dict__.update(a[0])
                for k, v in kw.items():
                    setattr(self, k, v)
            def __getattr__(self, name): return _Any()
            def __call__(self, *a, **kw): return _Any()
            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
            def __reduce__(self): return (_Any, ())
        class _MockFinder:
            def find_module(self, fullname, path=None):
                return self if fullname.startswith('mmengine') else None
            def load_module(self, fullname):
                if fullname in sys.modules:
                    return sys.modules[fullname]
                class Mod(types.ModuleType):
                    def __getattr__(self, name): return _Any
                m = Mod(fullname); m.__path__ = [fullname]; m.__loader__ = self
                sys.modules[fullname] = m; return m

        finder = _MockFinder()
        injected_modules = [k for k in sys.modules if k.startswith('mmengine')]
        sys.meta_path.insert(0, finder)
        try:
            ckpt = torch.load(path, map_location='cpu')
        finally:
            sys.meta_path.remove(finder)
            for k in list(sys.modules):
                if k.startswith('mmengine') and k not in injected_modules:
                    del sys.modules[k]

        sd = ckpt.get('state_dict', ckpt) if hasattr(ckpt, 'get') else ckpt

        backbone_sd = {}
        outnorm_sd = {}
        for k, v in sd.items():
            if not hasattr(v, 'shape'):
                continue
            if not k.startswith('backbone.'):
                continue
            clean_k = k[len('backbone.'):]
            # outnormX → self.outnorms.X
            if clean_k.startswith('outnorm'):
                idx = clean_k[len('outnorm')]
                rest = clean_k[len('outnormX'):]
                new_k = f'{idx}{rest}'
                outnorm_sd[new_k] = v
            else:
                backbone_sd[clean_k] = v

        info_bb = self.backbone.load_state_dict(backbone_sd, strict=False)
        n_bb = len(backbone_sd) - len(info_bb.unexpected_keys)

        # outnorm: outnorm_sd keys like "0.weight", "0.bias", ...
        info_on = self.outnorms.load_state_dict(outnorm_sd, strict=False)
        n_on = len(outnorm_sd) - len(info_on.unexpected_keys)

        total = n_bb + n_on
        print(f"[MambaVision-Tiny-ADE20K] 加载 backbone={n_bb}, outnorm={n_on} 个权重 "
              f"(bb_missing={len(info_bb.missing_keys)}, bb_unexpected={len(info_bb.unexpected_keys)})")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — 输入图像, H=W=512
        Returns:
            (x1, x2, x3, x4): 默认 [64,128,320,512]；no_reshape 时为 [80,160,320,640]。
            空间尺寸 (H/4, W/4), (H/8, W/8), (H/16, W/16), (H/32, W/32)
        """
        x = self.backbone.patch_embed(x)
        outs = []
        for level in self.backbone.levels:
            _, _, H, W = x.shape
            if level.transformer_block:
                pad_r = (level.window_size - W % level.window_size) % level.window_size
                pad_b = (level.window_size - H % level.window_size) % level.window_size
                if pad_r > 0 or pad_b > 0:
                    x = F.pad(x, (0, pad_r, 0, pad_b))
                    _, _, Hp, Wp = x.shape
                else:
                    Hp, Wp = H, W
                x = window_partition(x, level.window_size)
            for blk in level.blocks:
                x = blk(x)
            if level.transformer_block:
                x = window_reverse(x, level.window_size, Hp, Wp)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W].contiguous()
            outs.append(x)
            if level.downsample is not None:
                x = level.downsample(x)

        result = []
        for i, feat in enumerate(outs):
            # outnorm: (B, C, H, W) → permute → LN → permute back
            B, C, fH, fW = feat.shape
            feat_ln = self.outnorms[i](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            result.append(self.adapters[i](feat_ln))
        return tuple(result)
