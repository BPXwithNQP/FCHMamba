"""
HLAttention —— 双分支主干 / Unet 解码器之后的「整尾部」模块。

启用方式
--------
通过 main--.py 的 `--use_HL` 开关激活。打开后，`FHE` 不再使用原有的：
  - condition_heads / depth_head（高度头）
  - segment_head（分割头，且原版末端有 `nn.ReLU()`，与 `BCEWithLogitsLoss` 不兼容）
而是把 `dec_final` 整张特征图丢给本文件中的 `HLTail`，由它一次性产出
  pred_d (米制归一) / pred_s (logits) / 条件头中间量

整体流程
--------
1. **HL 解耦**  L = depthwise box-average(F)，H = F - L。
   语义上对应 tt.py / generate_borndery_structure.py 里 `avg(input)` 当低频、
   `input - avg(input)` 当高频的做法，但作用对象是 `dec_final` 这张特征图。

2. **分割支路**：
       A_edge   = sigmoid(Conv(H))           # 高频 → 边界注意力
       A_region = sigmoid(Conv(L) / τ)       # 低频 → 区域门控（限制 A_edge 只在合理区域生效）
       A        = A_edge * A_region
       F_seg    = F + α · A · Proj(H)        # α 用 tanh 限幅，初值 0：训练初期等价不动

   分割头与原 SegmentHead 等价，但去掉末端 `nn.ReLU()` 以匹配 `BCEWithLogitsLoss`。

3. **高度支路（先验注入）**：
       p_subject = sigmoid(seg_logits)       # 主体（建筑/树木）概率，∈[0,1]
       F_h       = Conv1x1( cat[F, p_subject] )
   `F_h` 再喂给 ConditionalCascadeHeads（use_condition=True）或 DepthHead。

4. **地表软钳**（默认开启，可由 `--use_ground_cap` 关掉）：
       cap = ground_cap_m / max_height           # 归一化高度
       h_cap = pred_d.clamp(max=cap)
       pred_d_final = p_subject · pred_d + (1 - p_subject) · h_cap
   即「分割认为是地表」的像素显式不准超过 ground_cap_m，建筑/树木区域不限制。

实现注意
--------
- `_DepthwiseBoxAvg` 用 buffer 存核（不可训练），与 tt.py / generate_borndery_structure.py
  口径一致；改 kernel 走 `--hl_avg_kernel`。
- 高度头有意保留两份内部实现（`_ConditionalCascadeHeads` / `_DepthHead`），不直接 import
  `models.model` 以避免循环依赖。
- 只读出 `pred_d_raw` 给 criterion 做 ground-cap 损失；最终下游使用的 `pred_d` 是钳位后版本。
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#   高低频解耦 —— depthwise box average
# ============================================================================
class _DepthwiseBoxAvg(nn.Module):
    """通道独立的 box-average（uniform_filter 的 GPU 版本）。"""

    def __init__(self, channels: int, kernel_size: int = 11):
        super().__init__()
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.k = k
        self.channels = channels
        weight = torch.full((channels, 1, k, k), 1.0 / float(k * k))
        # buffer 而非 Parameter：不参与反向传播
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.k // 2, groups=self.channels)


class HLDecouple(nn.Module):
    """L = boxavg(F), H = F - L."""

    def __init__(self, channels: int, kernel_size: int = 11):
        super().__init__()
        self.avg = _DepthwiseBoxAvg(channels, kernel_size)

    def forward(self, feat: torch.Tensor):
        low = self.avg(feat)
        high = feat - low
        return high, low


# ============================================================================
#   注意力 / 门控
# ============================================================================
class _BoundaryAttention(nn.Module):
    """高频 → A_edge (B,1,H,W) 与 H_proj (B,C,H,W)。"""

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 8)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(True),
            nn.Conv2d(mid, 1, kernel_size=1),
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, h: torch.Tensor):
        a_edge = torch.sigmoid(self.attn(h))
        h_proj = self.proj(h)
        return a_edge, h_proj


class _RegionGate(nn.Module):
    """低频 → A_region (B,1,H,W)。τ 可学习，通过 log_tau 参数化保持 >0。"""

    def __init__(self, channels: int, init_temp: float = 1.0):
        super().__init__()
        mid = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(True),
            nn.Conv2d(mid, 1, kernel_size=1),
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(max(float(init_temp), 1e-3))))

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        logits = self.gate(low)
        tau = self.log_tau.exp().clamp_min(1e-3)
        return torch.sigmoid(logits / tau)


# ============================================================================
#   分割头（去 ReLU 末端）
# ============================================================================
class _SegmentHead(nn.Module):
    """与 model.SegmentHead 同构，但末端 **没有** ReLU，直接输出 logits。"""

    def __init__(self, channels_out: int, num_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(channels_out, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ============================================================================
#   高度头
# ============================================================================
class _DepthHead(nn.Module):
    """与 model.DepthHead 同构。"""

    def __init__(self, channels_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _ConditionalCascadeHeads(nn.Module):
    """与 model.ConditionalCascadeHeads 同构（独立一份避免循环 import）。"""

    def __init__(
        self,
        channels_out: int,
        n_bins: int = 10,
        bin_len: float = 6.0,
        max_height: float = 185.0,
        condition_k: int = 0,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.bin_len = float(bin_len)
        self.max_height = max_height
        self.condition_k = int(condition_k)
        centers = [(i + 0.5) * self.bin_len for i in range(n_bins)]
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.bin_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, n_bins, 1),
        )
        self.res_head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, 1, 1),
        )

    def forward(self, x: torch.Tensor, adaptive_centers: Optional[torch.Tensor] = None):
        bin_logits = self.bin_head(x)
        bin_prob = F.softmax(bin_logits, dim=1)
        residual = self.res_head(x)
        if adaptive_centers is not None:
            centers = adaptive_centers.unsqueeze(-1).unsqueeze(-1)
        else:
            centers = self.centers.view(1, -1, 1, 1)
        prob_h = bin_prob
        k = self.condition_k
        if k > 0:
            k = min(k, int(bin_prob.size(1)))
            topv, topi = torch.topk(bin_prob, k, dim=1)
            prob_h = torch.zeros_like(bin_prob)
            prob_h.scatter_(1, topi, topv)
            prob_h = prob_h / prob_h.sum(dim=1, keepdim=True).clamp(min=1e-8)
        h_meters = (prob_h * (centers + residual)).sum(dim=1, keepdim=True)
        pred_d = (h_meters / self.max_height).clamp(0.0, 1.0)
        return pred_d, bin_logits, bin_prob, residual


# ============================================================================
#   主模块
# ============================================================================
class HLTail(nn.Module):
    """
    HL 整尾部模块。

    Args
    ----
    args : Namespace
        需读取以下字段（缺省值见 main--.py）：
            num_classes        : int   分割通道数（默认 1）
            use_condition      : bool  使用条件高度分区
            bin_num, bin_len   : 条件分区参数
            condition_k        : int  高度加权只用每像素 top-k 概率（0=全档位）
            max_height_eval    : float  反归一化最大高度（米）
            hl_avg_kernel      : int    HL 解耦盒平均核（特征图尺度）
            hl_alpha_init      : float  α 初值（tanh 之前），默认 0
            hl_region_temp_init: float  区域门控温度初值
            use_ground_cap     : bool   是否对地表像素软钳
            ground_cap_m       : float  地表上限（米）
    channels_out : int
        `dec_final` 的通道数（即 args.out_decoder）。
    """

    def __init__(self, args, channels_out: int, dpld_head: Optional[nn.Module] = None):
        super().__init__()
        self.use_condition = bool(getattr(args, "use_condition", False))
        self.use_DPLD = bool(getattr(args, "use_DPLD", False)) and dpld_head is not None
        self.num_classes = int(getattr(args, "num_classes", 1))
        self.max_height = float(getattr(args, "max_height_eval", 185.0))

        # ---- HL 解耦 ----
        kernel = int(getattr(args, "hl_avg_kernel", 11))
        self.decouple = HLDecouple(channels_out, kernel_size=kernel)

        # ---- 注意力 / 门控 ----
        self.boundary_attn = _BoundaryAttention(channels_out)
        self.region_gate = _RegionGate(
            channels_out, init_temp=float(getattr(args, "hl_region_temp_init", 1.0))
        )

        # α 用 tanh 限幅（[-1, 1]），初值 0：训练第 0 个 step 时 F_seg = F，不破坏现有功能
        alpha_init = float(getattr(args, "hl_alpha_init", 0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # ---- 分割头 ----
        self.segment_head = _SegmentHead(channels_out, self.num_classes)

        # ---- 高度先验注入：cat[F, p_subject] -> 1x1 conv -> channels_out ----
        self.height_input_proj = nn.Conv2d(
            channels_out + self.num_classes, channels_out, kernel_size=1
        )

        # ---- 高度头 ----
        # 优先级：DPLD > use_condition > DepthHead。DPLD 由外部注入，避免 HLTail 重复构建 bin edges。
        if self.use_DPLD:
            self.dpld_head = dpld_head
            self.condition_heads = None
            self.depth_head = None
        elif self.use_condition:
            self.dpld_head = None
            n_bins = int(getattr(args, "bin_num", 10))
            bin_len = float(getattr(args, "bin_len", 6.0))
            _ck = int(getattr(args, "condition_k", 0))
            self.condition_heads = _ConditionalCascadeHeads(
                channels_out,
                n_bins=n_bins,
                bin_len=bin_len,
                max_height=self.max_height,
                condition_k=_ck,
            )
            self.depth_head = None
        else:
            self.dpld_head = None
            self.condition_heads = None
            self.depth_head = _DepthHead(channels_out)

        # ---- 地表软钳 ----
        self.use_ground_cap = bool(getattr(args, "use_ground_cap", True))
        self.ground_cap_m = float(getattr(args, "ground_cap_m", 1.0))

    # ------------------------------------------------------------------
    def _alpha(self) -> torch.Tensor:
        """tanh 限幅，初始 0；自动学习。"""
        return torch.tanh(self.alpha_raw)

    # ------------------------------------------------------------------
    def forward(
        self,
        dec_final: torch.Tensor,
        adaptive_centers: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # ---- 1) 解耦 ----
        h_feat, l_feat = self.decouple(dec_final)

        # ---- 2) 分割支路 ----
        a_edge, h_proj = self.boundary_attn(h_feat)
        a_region = self.region_gate(l_feat)
        a_refined = a_edge * a_region                                    # (B,1,H,W)
        f_seg = dec_final + self._alpha() * a_refined * h_proj           # (B,C,H,W)

        seg_logits = self.segment_head(f_seg)                            # (B,num_classes,H,W)
        # p_subject 用 detach 切断梯度：避免高度损失把 seg 头拉偏（seg 自有 BCE 监督）
        p_subject_for_height = torch.sigmoid(seg_logits).detach()        # (B,num_classes,H,W)

        # ---- 3) 高度支路：先验注入 ----
        f_h = self.height_input_proj(torch.cat([dec_final, p_subject_for_height], dim=1))

        if self.use_DPLD and self.dpld_head is not None:
            pred_d_raw, bin_logits, bin_prob = self.dpld_head(f_h)
            data: Dict[str, torch.Tensor] = {
                "pred_d": pred_d_raw,
                "pred_s": seg_logits,
                "bin_logits": bin_logits,
                "bin_prob": bin_prob,
                "bin_edges_m": self.dpld_head.bin_edges_m,
                "bin_centers_m": self.dpld_head.bin_centers_m,
            }
        elif self.use_condition and self.condition_heads is not None:
            pred_d_raw, bin_logits, bin_prob, residual = self.condition_heads(
                f_h, adaptive_centers
            )
            data = {
                "pred_d": pred_d_raw,
                "pred_s": seg_logits,
                "bin_logits": bin_logits,
                "bin_prob": bin_prob,
                "residual": residual,
            }
        else:
            pred_d_raw = self.depth_head(f_h)                            # (B,1,H,W)
            data = {"pred_d": pred_d_raw, "pred_s": seg_logits}

        # ---- 4) 地表软钳：(1-p_subject) 区域 h ≤ ground_cap_m ----
        if self.use_ground_cap:
            cap_norm = self.ground_cap_m / max(self.max_height, 1e-6)    # 归一化高度上限
            # num_classes>1 时取均值作为「主体概率」；默认 num_classes=1 时直接用
            if p_subject_for_height.size(1) == 1:
                p_main = p_subject_for_height
            else:
                p_main = p_subject_for_height.mean(dim=1, keepdim=True)
            h_capped = pred_d_raw.clamp(max=cap_norm)
            pred_d_final = p_main * pred_d_raw + (1.0 - p_main) * h_capped
            pred_d_final = pred_d_final.clamp(0.0, 1.0)
            data["pred_d"] = pred_d_final
            data["pred_d_raw"] = pred_d_raw                              # 给 ground-cap loss / 调试

        # ---- 5) 监控量：HL alpha 当前值（标量，不参与梯度，便于日志） ----
        data["hl_alpha"] = self._alpha().detach()

        return data
