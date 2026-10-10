"""
MambaVision-B Backbone for FusedHE
参照官方 NVlabs/MambaVision 结构，4 级联 stage。
默认经 1×1 与 MiT 对齐 [64,128,320,512]；no_reshape=True 时保持原生 [128,256,512,1024]。
预训练权重：MambaVision-B-1K 224x224
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

# PyTorch 1.x 兼容：2.0 才提供 F.scaled_dot_product_attention，1.13 需手动实现
if not hasattr(F, 'scaled_dot_product_attention'):
    def _scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        scale = scale or (q.size(-1) ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        if is_causal and attn.size(-2) == attn.size(-1):
            causal_mask = torch.triu(torch.ones(attn.shape[-2], attn.shape[-1], device=attn.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
        return attn @ v
    F.scaled_dot_product_attention = _scaled_dot_product_attention

# 尝试导入 mambavision，若未安装会提示
try:
    from mambavision.models.mamba_vision import MambaVision, window_partition, window_reverse
    _HAS_MAMBAVISION = True
except ImportError:
    _HAS_MAMBAVISION = False

# 预训练权重 URL (224)
MAMBAVISION_B_1K_URL = "https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar"


def _load_checkpoint(model, filename, map_location='cpu', strict=False):
    """加载 checkpoint，兼容 state_dict / model / encoder 等 key 前缀"""
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint 格式异常: {filename}")
    state_dict = checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint
    if not state_dict:
        raise RuntimeError(f"Checkpoint 中无有效 state_dict: {filename}")
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if state_dict and list(state_dict.keys())[0].startswith('encoder.'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


class MambaVisionBackbone(nn.Module):
    """
    MambaVision-B 主干，返回 4 个 stage 特征。
    默认通道经 1×1 与 MiT 对齐 [64, 128, 320, 512]；no_reshape=True 时保持原生 [128, 256, 512, 1024]。
    """
    def __init__(self, pretrained=True, ckpt_path=None, no_reshape=False):
        super().__init__()
        self.no_reshape = no_reshape
        if not _HAS_MAMBAVISION:
            raise ImportError(
                "请先安装 mambavision 及依赖:\n"
                "  pip install mambavision mamba-ssm einops"
            )

        # MambaVision-B 配置，window_size 适配 512 输入（官方 224 为 [8,8,14,7]）
        self.backbone = MambaVision(
            dim=128,
            in_dim=64,
            depths=[3, 3, 10, 5],
            num_heads=[2, 4, 8, 16],
            window_size=[8, 8, 32, 16],
            mlp_ratio=4,
            drop_path_rate=0.3,
            layer_scale=1e-5,
            layer_scale_conv=None,
            in_chans=3,
            num_classes=0,
        )

        # 通道适配：原生 [128,256,512,1024] → MiT [64,128,320,512]；no_reshape 时用 Identity
        if no_reshape:
            self.adapters = nn.ModuleList([nn.Identity() for _ in range(4)])
        else:
            self.adapters = nn.ModuleList([
                nn.Conv2d(128, 64, 1),
                nn.Conv2d(256, 128, 1),
                nn.Conv2d(512, 320, 1),
                nn.Conv2d(1024, 512, 1),
            ])

        if pretrained:
            path = ckpt_path or self._default_ckpt_path()
            if os.path.isfile(path):
                _load_checkpoint(self.backbone, path, strict=False)
            else:
                print(f"[MambaVisionBackbone] 未找到权重 {path}，请运行 download_mambavision_weights.py 下载")

    def _default_ckpt_path(self):
        return str(Path(__file__).resolve().parent.parent / "weights" / "mambavision_base_1k.pth.tar")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (x1, x2, x3, x4)：默认通道 [64,128,320,512]；no_reshape 时为 [128,256,512,1024]。
            空间尺寸 (H/4,W/4), (H/8,W/8), (H/16,W/16), (H/32,W/32)
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
        return tuple(self.adapters[i](outs[i]) for i in range(4))


def download_mambavision_b_1k(save_dir=None):
    """下载 MambaVision-B-1K 224 预训练权重"""
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent.parent / "weights"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "mambavision_base_1k.pth.tar"
    if save_path.exists():
        print(f"权重已存在: {save_path}")
        return str(save_path)
    print(f"正在下载 MambaVision-B-1K 到 {save_path} ...")
    torch.hub.download_url_to_file(MAMBAVISION_B_1K_URL, str(save_path))
    print("下载完成.")
    return str(save_path)
