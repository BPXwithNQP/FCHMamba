"""block.yaml 的缓存读取工具。

切出为独立文件，避免和 `Blocks/__init__.py` 的子模块导入产生隐性循环。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

_BLOCK_DIR = os.path.dirname(os.path.abspath(__file__))
_BLOCK_YAML = os.path.join(_BLOCK_DIR, "block.yaml")
_CACHED_CFG: Optional[Dict[str, Any]] = None


def _load_yaml() -> Dict[str, Any]:
    global _CACHED_CFG
    if _CACHED_CFG is not None:
        return _CACHED_CFG
    if not os.path.isfile(_BLOCK_YAML):
        _CACHED_CFG = {}
        return _CACHED_CFG
    with open(_BLOCK_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raw = {}
    _CACHED_CFG = raw
    return _CACHED_CFG


def load_block_cfg(section: Optional[str] = None) -> Dict[str, Any]:
    """返回 block.yaml 中指定 section 的 dict；section=None 时返回顶层全量。"""
    cfg = _load_yaml()
    if section is None:
        return dict(cfg)
    sub = cfg.get(section, {})
    if not isinstance(sub, dict):
        return {}
    return dict(sub)


def merge_block_kwargs(section: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """以 yaml 默认值为底，override 中非 None 的字段覆盖。"""
    base = load_block_cfg(section)
    if not override:
        return base
    for k, v in override.items():
        if v is not None:
            base[k] = v
    return base


__all__ = ["load_block_cfg", "merge_block_kwargs"]
