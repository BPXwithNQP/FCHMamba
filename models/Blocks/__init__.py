"""Custom fusion blocks for FusedHE (FAFBlock / GSFB / FDSF / DPLD 等)。

`block.yaml` 与本目录同级，存放 FDSF / DPLD 等模块的默认参数。
模块构造时显式传入的参数优先，未传入的部分由 yaml 兜底。
"""
from models.Blocks._config import load_block_cfg, merge_block_kwargs  # noqa: F401
from models.Blocks.FAFBlock import FAFBlock  # noqa: F401
from models.Blocks.GSFB import GSFB  # noqa: F401
from models.Blocks.FDSF import FDSF  # noqa: F401
from models.Blocks.DMSN import DMSN  # noqa: F401
from models.Blocks.DPLD import (  # noqa: F401
    DPLDHead,
    compute_or_load_dpld_bin_edges,
    compute_or_load_dpld_bin_stats,
)

__all__ = [
    "FAFBlock",
    "GSFB",
    "FDSF",
    "DMSN",
    "DPLDHead",
    "compute_or_load_dpld_bin_edges",
    "compute_or_load_dpld_bin_stats",
    "load_block_cfg",
    "merge_block_kwargs",
]
