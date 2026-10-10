"""
FDGM —— Frequency-Decoupled GroupMamba（FD-RGHF 编码器）。

设计来源：models/encoder.md。在保留 GroupMamba 预训练权重热启动能力的前提下，
对每个 stage 的输入显式做「频域解耦 + 高频残差增强」：

    x
    └─ LocalRefiner:  x_loc = x + β · Local(x)
    └─ Laplace 解耦:  high = x_loc - LP(x_loc);  low = x_loc - ρ · high
    ├─ 低频分支:      y_low  = GroupMambaStage(low)      （原 stage，会下采样/改通道）
    ├─ 高频分支:      y_high = high + γ_h · gate · Refine(high)   （RGHF 边界增强）
    └─ 门控残差融合:  y = y_low + γ_f · Gate(y_low, P(y_high)) · P(y_high)

其中 P 负责把高频分支（输入分辨率/通道）对齐到 stage 输出（自适应下采样 + 1×1）。

热启动：β、ρ、γ_h、γ_f 全部 0 初始化 → x_loc=x、low=x_loc、y=y_low，
即输出严格等于原始 GroupMamba 四级特征，可安全加载预训练 GroupMamba checkpoint。

主干 variant / weights_path / pretrained / stage_channels 复用 configs/GroupMamba.yaml
（即 args.encoder1_cfg）；FD-RGHF 分支的频域超参数读 models/Blocks/block.yaml 的 FDGM 段。
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 GroupMamba 主干构建 / 通道常量；导入即完成 GLIBCXX / selective_scan 的 LD 预处理。
from models.groupmamba_backbone import (
    _TARGET_CH,
    _VARIANT_SRC_CH,
    _build_groupmamba,
)
from models.Blocks._config import load_block_cfg


def _to_odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


class LocalRefiner(nn.Module):
    """轻量局部细化：DWConv k×k + BN + GELU + PWConv 1×1 + BN（dim -> dim）。

    以残差形式补充局部高频/边界视野，浅层用大核、深层用小核。
    """

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        kernel = _to_odd(kernel)
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _BandRefine(nn.Module):
    """高频带细化：DWConv 3×3 + BN + GELU + PWConv 1×1 + BN（dim -> dim）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FDGMStage(nn.Module):
    """单个 GroupMamba stage 的 FD-RGHF 封装。

    复用原 patch_embed / blocks / norm（低频主干，承载预训练权重），
    叠加 LocalRefiner、Laplace 解耦、高频 RGHF 分支与门控残差融合（新增、零初始化）。
    """

    def __init__(
        self,
        backbone: nn.Module,
        stage_idx0: int,
        in_ch: int,
        out_ch: int,
        cfg: dict,
    ):
        super().__init__()
        self.stage_idx0 = int(stage_idx0)       # 0-based
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        # stage1 是 Stem（stride 4），其余 DownSamples（stride 2）
        self.stride = 4 if self.stage_idx0 == 0 else 2

        # ---- 低频主干（原 GroupMamba stage 组件，热启动权重落在这里）----
        self.patch_embed = getattr(backbone, "patch_embed{}".format(self.stage_idx0 + 1))
        self.blocks = getattr(backbone, "block{}".format(self.stage_idx0 + 1))
        self.norm = getattr(backbone, "norm{}".format(self.stage_idx0 + 1))

        # ---- 1. LocalRefiner ----
        local_kernels = cfg.get("local_kernels", [7, 7, 5, 3])
        k = local_kernels[self.stage_idx0] if self.stage_idx0 < len(local_kernels) else local_kernels[-1]
        self.local = LocalRefiner(in_ch, kernel=int(k))
        beta_init = float(cfg.get("beta_init", 0.0))
        self.beta = nn.Parameter(torch.full((1,), beta_init))

        # ---- 2. Laplace 频域解耦 ----
        lp_kernel = _to_odd(int(cfg.get("lp_kernel", 5)))
        self.lp_kernel = lp_kernel
        self.lp_pad = lp_kernel // 2
        rho_init = float(cfg.get("rho_init", 0.0))
        self.rho = nn.Parameter(torch.full((1,), rho_init))
        self.rho_max = float(cfg.get("rho_max", 1.0))

        # ---- 3. 高频 RGHF 边界增强分支（作用于输入通道 in_ch）----
        self.band_refine = _BandRefine(in_ch)
        hg_red = int(cfg.get("high_gate_reduction", 4))
        hg_k = _to_odd(int(cfg.get("high_gate_kernel", 3)))
        hg_hidden = max(in_ch // max(1, hg_red), 16)
        self.high_gate = nn.Sequential(
            nn.Conv2d(2 * in_ch, hg_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hg_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hg_hidden, 1, kernel_size=hg_k, padding=hg_k // 2, bias=True),
            nn.Sigmoid(),
        )
        gamma_h_init = float(cfg.get("gamma_h_init", 0.0))
        self.gamma_h = nn.Parameter(torch.full((1, in_ch, 1, 1), gamma_h_init))

        # ---- 4. 门控残差融合（对齐到 stage 输出分辨率/通道 out_ch）----
        # Proj：把高频分支（in_ch、输入分辨率）下采样并对齐到 out_ch。
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        fg_red = int(cfg.get("fuse_gate_reduction", 4))
        fg_k = _to_odd(int(cfg.get("fuse_gate_kernel", 3)))
        fg_hidden = max(out_ch // max(1, fg_red), 16)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(2 * out_ch, fg_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(fg_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(fg_hidden, 1, kernel_size=fg_k, padding=fg_k // 2, bias=True),
            nn.Sigmoid(),
        )
        gamma_f_init = float(cfg.get("gamma_f_init", 0.0))
        self.gamma_f = nn.Parameter(torch.full((1, out_ch, 1, 1), gamma_f_init))

    def _low_branch(self, low: torch.Tensor) -> torch.Tensor:
        """原 GroupMamba stage：patch_embed → blocks → norm → NCHW。"""
        B = low.shape[0]
        x, H, W = self.patch_embed(low)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def _high_frequency(self, x_loc: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x_loc, kernel_size=self.lp_kernel, stride=1, padding=self.lp_pad)
        return x_loc - low

    def forward(self, x: torch.Tensor, capture: Optional[dict] = None) -> torch.Tensor:
        # 1. LocalRefiner 残差注入
        x_loc = x + self.beta * self.local(x)

        # 2. Laplace 解耦
        high = self._high_frequency(x_loc)
        rho = self.rho.clamp(0.0, self.rho_max)
        low = x_loc - rho * high

        # 3. 低频主干
        y_low = self._low_branch(low)

        # 4. 高频 RGHF 分支
        edge_residual = self.band_refine(high)
        edge_gate = self.high_gate(torch.cat((high, edge_residual), dim=1))
        y_high = high + self.gamma_h * (edge_gate * edge_residual)

        # 5. 门控残差融合（先把高频对齐到低频输出分辨率/通道）
        Ho, Wo = y_low.shape[-2], y_low.shape[-1]
        if y_high.shape[-2] != Ho or y_high.shape[-1] != Wo:
            y_high = F.adaptive_avg_pool2d(y_high, (Ho, Wo))
        p = self.proj(y_high)
        fuse_gate = self.fuse_gate(torch.cat((y_low, p), dim=1))
        y = y_low + self.gamma_f * (fuse_gate * p)

        if capture is not None:
            capture["edge_gate"] = edge_gate.detach().cpu()
            capture["y_low"] = y_low.detach().cpu()
            capture["fdgm_out"] = y.detach().cpu()
        return y


class FDGMEncoder(nn.Module):
    """FD-RGHF 编码器：单分支，forward(x) -> (f1,f2,f3,f4)，各级对齐到 MiT [64,128,320,512]。

    与 GroupMambaFusedBackbone 一致地把四级特征 1×1 对齐到 _TARGET_CH，便于直接接 MitDecoder。
    """

    def __init__(self, args):
        super().__init__()
        cfg_bb = getattr(args, "encoder1_cfg", None) or {}
        fdgm_cfg = load_block_cfg("FDGM")

        # 主干超参：block.yaml.FDGM 可覆盖，否则取 GroupMamba.yaml（encoder1_cfg）
        variant = fdgm_cfg.get("variant") or cfg_bb.get("variant", "base")
        self._variant = variant
        pretrained = fdgm_cfg.get("pretrained")
        if pretrained is None:
            pretrained = cfg_bb.get("pretrained", True)
        pretrained = bool(pretrained)
        weights_path = fdgm_cfg.get("weights_path") or cfg_bb.get("weights_path")

        backbone = _build_groupmamba(variant)

        src_ch = list(cfg_bb.get("stage_channels")) if cfg_bb.get("stage_channels") else list(
            _VARIANT_SRC_CH.get((variant or "base").lower(), _VARIANT_SRC_CH["base"])
        )
        # 各 stage 输入通道：stage1 = 3（原图），其余 = 上一级输出通道
        in_chs = [3] + src_ch[:-1]
        out_chs = src_ch

        self.stages = nn.ModuleList(
            [
                FDGMStage(backbone, i, in_chs[i], out_chs[i], fdgm_cfg)
                for i in range(backbone.num_stages)
            ]
        )
        self.num_stages = backbone.num_stages

        adapters = []
        for src, tgt in zip(src_ch, _TARGET_CH):
            if src == tgt:
                adapters.append(nn.Identity())
            else:
                adapters.append(nn.Conv2d(src, tgt, kernel_size=1, bias=False))
        self.adapters = nn.ModuleList(adapters)

        self._print_summary(in_chs, out_chs, fdgm_cfg)

        if pretrained and weights_path and os.path.isfile(weights_path):
            self._load_pretrained(weights_path)
        elif pretrained and weights_path:
            print("[Encoder/FDGM] 未找到权重文件: {}".format(weights_path), flush=True)
        elif pretrained:
            print("[Encoder/FDGM] 未提供 weights_path，跳过预训练加载", flush=True)

    def _print_summary(self, in_chs, out_chs, fdgm_cfg) -> None:
        print(
            "[Encoder/FDGM] variant={} stages={} local_kernels={} lp_kernel={} "
            "beta/rho/gamma_h/gamma_f init={}/{}/{}/{}".format(
                self._variant,
                self.num_stages,
                fdgm_cfg.get("local_kernels", [7, 7, 5, 3]),
                fdgm_cfg.get("lp_kernel", 5),
                fdgm_cfg.get("beta_init", 0.0),
                fdgm_cfg.get("rho_init", 0.0),
                fdgm_cfg.get("gamma_h_init", 0.0),
                fdgm_cfg.get("gamma_f_init", 0.0),
            ),
            flush=True,
        )
        print(
            "[Encoder/FDGM] in_ch={} out_ch={} -> target {}".format(in_chs, out_chs, _TARGET_CH),
            flush=True,
        )

    def _load_pretrained(self, path: str) -> None:
        """加载 GroupMamba checkpoint。键名从 backbone.* 映射到 stages.{i}.*。"""
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        if not isinstance(sd, dict):
            return
        sd = {k.replace("module.", "").replace("module.module.", ""): v for k, v in sd.items()}

        remapped: dict = {}
        for i in range(self.num_stages):
            pe = "patch_embed{}.".format(i + 1)
            bl = "block{}.".format(i + 1)
            nm = "norm{}.".format(i + 1)
            for k, v in sd.items():
                if k.startswith(pe):
                    remapped["stages.{}.patch_embed.".format(i) + k[len(pe):]] = v
                elif k.startswith(bl):
                    remapped["stages.{}.blocks.".format(i) + k[len(bl):]] = v
                elif k.startswith(nm):
                    remapped["stages.{}.norm.".format(i) + k[len(nm):]] = v

        info = self.load_state_dict(remapped, strict=False)
        # 仅统计低频主干（stages.*.patch_embed/blocks/norm）相关的 missing，便于核对热启动
        backbone_missing = [
            k for k in info.missing_keys
            if any(s in k for s in (".patch_embed.", ".blocks.", ".norm."))
        ]
        print(
            "[Encoder/FDGM] 预训练加载 variant={} 映射键={} backbone_missing={} unexpected={}".format(
                self._variant, len(remapped), len(backbone_missing), len(info.unexpected_keys)
            ),
            flush=True,
        )

    def forward(self, x: torch.Tensor, fdgm_capture: Optional[Dict[str, dict]] = None):
        feats = []
        cur = x
        for i in range(self.num_stages):
            cap = fdgm_capture["stage{}".format(i + 1)] if fdgm_capture is not None else None
            cur = self.stages[i](cur, capture=cap)
            feats.append(cur)
        if len(feats) != 4:
            raise RuntimeError("FDGM 应输出 4 个 stage 特征")
        return tuple(self.adapters[i](feats[i]) for i in range(4))
