"""配置与系统参数模块。

统一提供：
- 纯 Python 配置入口（RobotConfig / SimulationConfig / AnchorConfig / RodGroupConfig）
- JSON 仿真参数覆盖读取（load_config）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import msgspec


@dataclass(slots=True)
class SimulationConfig:
    """仿真配置。"""

    # 弹簧刚度系数，控制杆组长度误差转化为轴向力的强度。
    k: float = 20000.0
    # 仿真时间步长（秒），越小越稳定但计算开销更高。
    h: float = 0.001
    # 速度阻尼因子，每步对速度进行衰减以抑制振荡。
    damping_ratio: float = 0.95
    # 收缩分档间隔，用于推导最大收缩比例。
    contraction_interval: float = 0.1
    # 收缩分档数量，用于推导最大收缩比例。
    contraction_levels: int = 4
    # 控制组目标值向当前值逼近的每步最大变化率。
    contraction_percent_rate: float = 1e-3
    # 重力加速度缩放因子。
    gravity_factor: float = 180.0
    # 是否启用重力。
    gravity: bool = True
    # 默认最小长度基准（用于拓扑编辑时新增节点偏移和长度推导基线）。
    default_min_length: float = 1.2
    # 地面摩擦速度衰减系数。
    friction_factor: float = 0.8
    # 动作周期倍率，num_steps_action = num_steps_action_multiplier / h。
    num_steps_action_multiplier: float = 2.0

    @property
    def max_max_contraction(self) -> float:
        """系统允许的最大收缩比例。"""
        return round(self.contraction_interval * (self.contraction_levels - 1), 2)

    @property
    def default_max_length(self) -> float:
        """由最小长度和最大收缩比例推导默认最大长度。"""
        return self.default_min_length / (1 - self.max_max_contraction)

    @property
    def num_steps_action(self) -> float:
        """单个动作周期对应的仿真步数。"""
        return self.num_steps_action_multiplier / self.h


@dataclass(slots=True)
class AnchorConfig:
    """锚点（节点球）配置。"""

    # 锚点球体半径 (m)，用于可视化和碰撞体尺寸。
    radius: float = 0.04
    # 锚点质量 (kg)，影响动力学积分结果。
    mass: float = 0.25


@dataclass(slots=True)
class RodGroupConfig:
    """杆组显示与长度约束配置。"""

    # 左杆/右杆圆柱半径 (m)。
    rod_radius: float = 0.0125
    # 左右杆默认颜色（RGB，0-255），用于可视化材质。
    rod_color_rgb: list[int] = field(default_factory=lambda: [220, 220, 220])
    # 套筒默认颜色（RGB，0-255），用于可视化材质（炭黑）。
    sleeve_color_rgb: list[int] = field(default_factory=lambda: [30, 30, 30])
    # 套筒半径 (m)，用于套筒圆柱的可视化半径。
    sleeve_radius: float = 0.020
    # 套筒显示半长度占比，按当前两锚点距离的比例计算套筒半长度。
    sleeve_display_half_length_ratio: float = 0.3
    # 杆端修剪量 (m)，避免杆端和锚点球/套筒几何直接重叠。
    rod_trim: float = 0.01
    # 初始姿态约定：每侧杆的内端点位于套筒中心。
    # 后续两点距离变化通过“左右杆相对套筒滑动”表达，而不是三段比例缩放。
    initial_inner_end_at_center: bool = True
    # 长度裕量 delta，满足 L_max = L_min + length_delta。
    length_delta: float = 0.1


@dataclass(slots=True)
class RobotConfig:
    """机器人默认配置集合。"""

    # 仿真参数。
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    # 锚点几何与质量参数。
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    # 杆组几何与长度约束参数。
    rod_group: RodGroupConfig = field(default_factory=RodGroupConfig)


def default_robot_config() -> RobotConfig:
    """返回默认机器人配置。"""
    return RobotConfig()


def default_simulation_config() -> SimulationConfig:
    """返回默认仿真配置。"""
    return default_robot_config().simulation

_DEFAULT_SIM = default_simulation_config()


class _ConfigFile(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
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
        numStepsActionMultiplier: 动作步数乘数。
    """

    k: float = _DEFAULT_SIM.k
    h: float = _DEFAULT_SIM.h
    dampingRatio: float = _DEFAULT_SIM.damping_ratio
    contractionInterval: float = _DEFAULT_SIM.contraction_interval
    contractionLevels: int = _DEFAULT_SIM.contraction_levels
    contractionPercentRate: float = _DEFAULT_SIM.contraction_percent_rate
    gravityFactor: float = _DEFAULT_SIM.gravity_factor
    gravity: int = int(_DEFAULT_SIM.gravity)
    defaultMinLength: float = _DEFAULT_SIM.default_min_length
    frictionFactor: float = _DEFAULT_SIM.friction_factor
    numStepsActionMultiplier: float = _DEFAULT_SIM.num_steps_action_multiplier


def default_config() -> SimulationConfig:
    """返回默认仿真配置。

    Returns:
        默认 ``SimulationConfig`` 实例。
    """
    return default_simulation_config()


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
    file_model = msgspec.convert(raw, type=_ConfigFile)
    return SimulationConfig(
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
        num_steps_action_multiplier=file_model.numStepsActionMultiplier,
    )
