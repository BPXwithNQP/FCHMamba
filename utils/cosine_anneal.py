"""无重启的余弦退火：T_max 个 step 内从各 param_group 的 base_lr 余弦降至 eta_min，之后恒为 eta_min。"""
from __future__ import annotations

import math
from typing import List


class CosineAnnealingHoldEnd:
    """
    与 PyTorch 的 ``CosineAnnealingLR`` 不同：在达到 ``T_max`` 次 :meth:`step` 后学习率**保持**为 ``eta_min``，
    不会进入下一个余弦上升半周（无重启、无循环）。

    每次 :meth:`step` 对应「一个 epoch 结束」时调用一次。
    第 t 次 ``step``（t=1..T_max）时：
    ``lr = eta_min + (base - eta_min) * (1 + cos(pi * t / T_max)) / 2``。
    当 t > T_max 时，``lr = eta_min``。
    """

    def __init__(self, optimizer, T_max: int, eta_min: float, last_step: int = 0):
        self.optimizer = optimizer
        self.T_max = max(int(T_max), 1)
        self.eta_min = float(eta_min)
        self._base_lrs: List[float] = [float(g["lr"]) for g in optimizer.param_groups]
        self._step = int(last_step)
        if self._step < 0:
            self._step = 0
        # 若从 checkpoint 恢复且 last_step>0，立即同步一次 lr
        if self._step > 0:
            self._set_lrs()

    def _set_lrs(self) -> None:
        t = self._step
        for i, g in enumerate(self.optimizer.param_groups):
            base = self._base_lrs[i]
            if t > self.T_max:
                g["lr"] = self.eta_min
            else:
                g["lr"] = self.eta_min + (base - self.eta_min) * 0.5 * (
                    1.0 + math.cos(math.pi * t / self.T_max)
                )

    def step(self) -> None:
        self._step += 1
        self._set_lrs()

    @property
    def last_step(self) -> int:
        return self._step

    def state_dict(self) -> dict:
        return {
            "T_max": self.T_max,
            "eta_min": self.eta_min,
            "base_lrs": list(self._base_lrs),
            "step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.T_max = int(state.get("T_max", self.T_max))
        self.eta_min = float(state.get("eta_min", self.eta_min))
        self._base_lrs = [float(x) for x in state.get("base_lrs", self._base_lrs)]
        self._step = int(state.get("step", 0))
        self._set_lrs()
