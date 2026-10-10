"""DMSN: Directional Mamba Semantic Neck.

放在 FusedDecoder 的 stage4 融合特征之后、第一次 upsample 之前，只校准最深层语义特征。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Blocks._config import load_block_cfg


class DropPath(nn.Module):
    """按样本随机丢弃残差分支；只在训练阶段生效。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class LightweightSelectiveScanBlock(nn.Module):
    """轻量 1D Mamba/Selective-Scan 单元，输入输出均为 (B, L, C)。

    这里不用普通 gated conv 兜底，而是显式维护 diagonal SSM state，并按序列长度做
    selective scan recurrence。stage4 的 H/W 较小，纯 PyTorch scan 的开销可控。
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: float = 2.0,
        conv_kernel: int = 3,
    ):
        super().__init__()
        inner_dim = max(int(dim * expand), dim)
        self.inner_dim = inner_dim
        self.d_state = int(max(1, d_state))

        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=True)
        self.dwconv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inner_dim,
            bias=True,
        )
        self.dt_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        self.b_proj = nn.Linear(inner_dim, inner_dim * self.d_state, bias=False)
        self.c_proj = nn.Linear(inner_dim, inner_dim * self.d_state, bias=False)

        self.A_log = nn.Parameter(torch.empty(inner_dim, self.d_state))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.b_proj.weight)
        nn.init.xavier_uniform_(self.c_proj.weight)
        nn.init.constant_(self.dt_proj.bias, -2.0)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        with torch.no_grad():
            states = torch.arange(1, self.d_state + 1, dtype=torch.float32)
            self.A_log.copy_(torch.log(states).unsqueeze(0).repeat(self.inner_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)

        dt = F.softplus(self.dt_proj(u)).clamp(max=10.0)
        b_param = self.b_proj(u).view(bsz, length, self.inner_dim, self.d_state)
        c_param = self.c_proj(u).view(bsz, length, self.inner_dim, self.d_state)

        a = -torch.exp(self.A_log).to(dtype=u.dtype, device=u.device)
        state = u.new_zeros(bsz, self.inner_dim, self.d_state)
        ys = []
        for i in range(length):
            dt_i = dt[:, i].unsqueeze(-1)
            u_i = u[:, i].unsqueeze(-1)
            state = torch.exp(dt_i * a) * state + dt_i * b_param[:, i] * u_i
            y_i = (state * c_param[:, i]).sum(dim=-1) + self.D * u[:, i]
            ys.append(y_i)

        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class DirectionalMambaUnit(nn.Module):
    """单方向 Mamba 状态建模单元。输入/输出: (B, L, C)。"""

    def __init__(
        self,
        dim: int,
        mamba_block: Optional[Callable[[int], nn.Module]] = None,
        d_state: int = 16,
        expand: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if mamba_block is None:
            self.mamba = LightweightSelectiveScanBlock(dim, d_state=d_state, expand=expand)
        else:
            self.mamba = mamba_block(dim)
        self.out_drop = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        y = self.out_drop(self.mamba(x_norm))
        g = self.gate(x_norm)
        return residual + g * y


class DMSN(nn.Module):
    """
    Directional Mamba Semantic Neck.
    放在 stage4 ff1 输出之后、第一次 upsample 之前（MitDecoder / ConvDecoder / FusedDecoder）。
    """
    def __init__(
        self,
        in_channels: int,
        mamba_block: Optional[Callable[[int], nn.Module]] = None,
        reduction: Optional[int] = None,
        gamma_init: Optional[float] = None,
        share_direction_mamba: Optional[bool] = None,
        min_hidden: Optional[int] = None,
        d_state: Optional[int] = None,
        expand: Optional[float] = None,
        dropout: Optional[float] = None,
        drop_path: Optional[float] = None,
    ):
        super().__init__()
        cfg = load_block_cfg("DMSN")

        reduction = int(cfg.get("reduction", 4) if reduction is None else reduction)
        gamma_init = float(cfg.get("gamma_init", 0.05) if gamma_init is None else gamma_init)
        share_direction_mamba = bool(
            cfg.get("share_direction_mamba", True)
            if share_direction_mamba is None
            else share_direction_mamba
        )
        min_hidden = int(cfg.get("min_hidden", 64) if min_hidden is None else min_hidden)
        d_state = int(cfg.get("d_state", 16) if d_state is None else d_state)
        expand = float(cfg.get("expand", 2.0) if expand is None else expand)
        dropout = float(cfg.get("dropout", 0.05) if dropout is None else dropout)
        drop_path = float(cfg.get("drop_path", 0.05) if drop_path is None else drop_path)

        hidden = max(in_channels // max(1, reduction), min_hidden)
        self.hidden = hidden
        self.share_direction_mamba = share_direction_mamba

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.local_mix = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        if share_direction_mamba:
            self.mamba_h = DirectionalMambaUnit(
                hidden,
                mamba_block,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
            self.mamba_v = DirectionalMambaUnit(
                hidden,
                mamba_block,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
        else:
            self.mamba_lr = DirectionalMambaUnit(
                hidden,
                mamba_block,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
            self.mamba_rl = DirectionalMambaUnit(
                hidden,
                mamba_block,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
            self.mamba_tb = DirectionalMambaUnit(
                hidden,
                mamba_block,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
            self.mamba_bt = DirectionalMambaUnit(
                hidden,
                mamba_block,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )

        self.dir_gate = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )

        self.up = nn.Sequential(
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.drop_path = DropPath(drop_path)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, f4: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f4.shape

        x = self.down(f4)
        x = x + self.local_mix(x)

        # -------------------------
        # Horizontal sequence
        # -------------------------
        x_h = x.permute(0, 2, 3, 1).reshape(b * h, w, self.hidden)

        if self.share_direction_mamba:
            y_lr = self.mamba_h(x_h)
            y_rl = self.mamba_h(torch.flip(x_h, dims=[1]))
        else:
            y_lr = self.mamba_lr(x_h)
            y_rl = self.mamba_rl(torch.flip(x_h, dims=[1]))

        y_rl = torch.flip(y_rl, dims=[1])
        y_h = y_lr + y_rl
        y_h = y_h.reshape(b, h, w, self.hidden).permute(0, 3, 1, 2)

        # -------------------------
        # Vertical sequence
        # -------------------------
        x_v = x.permute(0, 3, 2, 1).reshape(b * w, h, self.hidden)

        if self.share_direction_mamba:
            y_tb = self.mamba_v(x_v)
            y_bt = self.mamba_v(torch.flip(x_v, dims=[1]))
        else:
            y_tb = self.mamba_tb(x_v)
            y_bt = self.mamba_bt(torch.flip(x_v, dims=[1]))

        y_bt = torch.flip(y_bt, dims=[1])
        y_v = y_tb + y_bt
        y_v = y_v.reshape(b, w, h, self.hidden).permute(0, 3, 2, 1)

        # -------------------------
        # Direction-aware fusion
        # -------------------------
        gate = self.dir_gate(torch.cat([y_h, y_v], dim=1))
        y = gate * y_h + (1.0 - gate) * y_v

        y = self.drop_path(self.up(y))

        return f4 + self.gamma * y