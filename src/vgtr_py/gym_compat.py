"""Gymnasium 兼容层。

真实依赖已在 ``pyproject.toml`` 中声明。本模块仅在 gymnasium 不可用时
（如沙箱测试环境）提供最小化的 Mock 实现，保证本地测试和类型检查可正常进行。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - exercised when gymnasium is available
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised in the sandbox

    class _Env:
        """最小化 Env Mock，用于无 gymnasium 时的类型占位。"""

        metadata: dict[str, object] = {}

        def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
            raise NotImplementedError

        def step(self, action: np.ndarray):
            raise NotImplementedError

    @dataclass(slots=True)
    class _Box:
        """最小化 Box Mock，用于无 gymnasium 时的类型占位。"""

        low: np.ndarray
        high: np.ndarray
        shape: tuple[int, ...]
        dtype: np.dtype

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x: object) -> bool:
            array = np.asarray(x, dtype=self.dtype)
            return array.shape == self.shape

    class _SpacesModule:
        Box = _Box

    class _GymModule:
        Env = _Env

    gym = _GymModule()
    spaces = _SpacesModule()
