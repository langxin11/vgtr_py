"""配置与系统参数模块。

包含了用于物理仿真与模拟的各项配置类（如 SimulationConfig 等），提供了对超参的校验和配置存取逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec


class _ConfigFile(msgspec.Struct, kw_only=True):
    """用于解析配置文件的内部数据结构。

    Attributes:
        k: 弹簧刚度系数。
        h: 模拟时间步长。
        dampingRatio: 阻尼比。
        contractionInterval: 每次收缩级别的收缩量间隔。
        contractionLevels: 收缩级别总数。
        contractionPercentRate: 每次迭代的收缩百分比变化率。
        gravityFactor: 重力因子。
        gravity: 是否启用重力 (1 表示启用，0 表示禁用)。
        defaultMinLength: 默认最小边长。
        frictionFactor: 摩擦力因子。
        directionalFriction: 是否启用方向性摩擦 (1 表示启用，0 表示禁用)。
        numStepsActionMultiplier: 动作步数乘数。
        defaultNumActions: 默认动作数量。
        defaultNumChannels: 默认通道数量。
        angleThreshold: 角度阈值。
        angleCheckFrequencyMultiplier: 角度检查频率乘数。
    """

    k: float = 200000.0
    h: float = 0.001
    dampingRatio: float = 0.999
    contractionInterval: float = 0.1
    contractionLevels: int = 4
    contractionPercentRate: float = 1e-3
    gravityFactor: float = 180.0
    gravity: int = 1
    defaultMinLength: float = 1.2
    frictionFactor: float = 0.8
    directionalFriction: int = 0
    numStepsActionMultiplier: float = 2.0
    defaultNumActions: int = 1
    defaultNumChannels: int = 5
    angleThreshold: float = 1.5708
    angleCheckFrequencyMultiplier: float = 0.05


@dataclass(slots=True)
class SimulationConfig:
    """仿真环境的配置参数。

    Attributes:
        k: 弹簧刚度系数。
        h: 模拟时间步长。
        damping_ratio: 阻尼比，用于能量衰减。
        contraction_interval: 每次收缩级别的收缩量间隔。
        contraction_levels: 允许的收缩级别总数。
        contraction_percent_rate: 每次迭代的收缩百分比变化率。
        gravity_factor: 重力影响因子。
        gravity: 是否启用重力。
        default_min_length: 默认最小边长。
        friction_factor: 摩擦力因子。
        directional_friction: 是否启用方向性摩擦。
        num_steps_action_multiplier: 动作步数乘数。
        default_num_actions: 默认动作数量。
        default_num_channels: 默认通道数量。
        angle_threshold: 角度阈值，用于特定碰撞或弯曲检测。
        angle_check_frequency_multiplier: 角度检查频率的乘数。
    """

    k: float = 200000.0
    h: float = 0.001
    damping_ratio: float = 0.999
    contraction_interval: float = 0.1
    contraction_levels: int = 4
    contraction_percent_rate: float = 1e-3
    gravity_factor: float = 180.0
    gravity: bool = True
    default_min_length: float = 1.2
    friction_factor: float = 0.8
    directional_friction: bool = False
    num_steps_action_multiplier: float = 2.0
    default_num_actions: int = 1
    default_num_channels: int = 5
    angle_threshold: float = 1.5708
    angle_check_frequency_multiplier: float = 0.05

    @property
    def max_max_contraction(self) -> float:
        """返回系统允许的最大收缩比例。

        Returns:
            最大收缩比例（保留两位小数）。
        """
        return round(self.contraction_interval * (self.contraction_levels - 1), 2)

    @property
    def default_max_length(self) -> float:
        """根据默认最小边长和收缩比例推导默认最大边长。

        Returns:
            默认最大边长。
        """
        return self.default_min_length / (1 - self.max_max_contraction)

    @property
    def num_steps_action(self) -> float:
        """返回一个动作周期对应的仿真步数。

        Returns:
            动作周期步数。
        """
        return self.num_steps_action_multiplier / self.h

    @property
    def angle_check_frequency(self) -> float:
        """返回角度检查间隔步数。

        Returns:
            两次角度检查之间的仿真步数。
        """
        return self.num_steps_action * self.angle_check_frequency_multiplier

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SimulationConfig:
        """从 JSON 字典构建 ``SimulationConfig``。

        Args:
            raw: 配置字典。

        Returns:
            ``SimulationConfig`` 实例。
        """
        file_model = msgspec.convert(raw, type=_ConfigFile)
        return cls(
            k=file_model.k,
            h=file_model.h,
            damping_ratio=file_model.dampingRatio,
            contraction_interval=file_model.contractionInterval,
            contraction_levels=file_model.contractionLevels,
            contraction_percent_rate=file_model.contractionPercentRate,
            gravity_factor=file_model.gravityFactor,
            gravity=bool(file_model.gravity),
            default_min_length=file_model.defaultMinLength,
            friction_factor=file_model.frictionFactor,
            directional_friction=bool(file_model.directionalFriction),
            num_steps_action_multiplier=file_model.numStepsActionMultiplier,
            default_num_actions=file_model.defaultNumActions,
            default_num_channels=file_model.defaultNumChannels,
            angle_threshold=file_model.angleThreshold,
            angle_check_frequency_multiplier=file_model.angleCheckFrequencyMultiplier,
        )


def default_config() -> SimulationConfig:
    """返回默认仿真配置。

    Returns:
        默认 ``SimulationConfig`` 实例。
    """
    return SimulationConfig()


def load_config(path: str | Path) -> SimulationConfig:
    """从 JSON 文件加载仿真配置。

    Args:
        path: 配置文件路径。

    Returns:
        由此配置文件内容初始化的 ``SimulationConfig`` 实例。

    Raises:
        TypeError: 当 JSON 内容无法解析为对象/字典时抛出。
    """
    raw = msgspec.json.decode(Path(path).read_bytes())
    if not isinstance(raw, dict):
        raise TypeError("config file must decode to a JSON object")
    return SimulationConfig.from_mapping(raw)
