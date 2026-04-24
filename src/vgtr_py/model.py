"""编译后的只读仿真模型。

VGTRModel 将 Workspace 中的拓扑、物理参数与控制配置预计算为静态只读结构，
供 Simulator 在步进中高频读取，避免步进内反复访问工作区的可变状态。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import RobotConfig, SimulationConfig
from .workspace import Workspace

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
UInt8Array = NDArray[np.uint8]


@dataclass(slots=True)
class VGTRModel:
    """供 Simulator 与环境使用的只读编译模型。

    Attributes:
        config: 仿真全局参数（步长、刚度、阻尼等）。
        robot_config: 机器人几何/物理配置（锚点半径、杆半径等）。
        anchor_ids: 锚点标识符列表。
        anchor_count: 锚点数量。
        anchor_rest_pos: 锚点静止位置，shape (N, 3)。
        anchor_fixed: 锚点是否固定，shape (N,)。
        anchor_mass: 锚点质量，shape (N,)。
        anchor_radius: 锚点半径，shape (N,)。
        anchor_projection_target: 是否作为投影目标锚点，shape (N,)。
        projection_anchor_indices: 投影目标锚点索引数组。
        rod_group_ids: 杆组标识符列表。
        rod_count: 杆组数量。
        rod_anchors: 杆组两端锚点索引，shape (R, 2)。
        rod_type: 杆组类型（0=被动, 1=主动, 2=柔性），shape (R,)。
        rod_enabled: 杆组是否启用，shape (R,)。
        rod_control_group: 杆组所属控制组索引，shape (R,)。
        rod_rest_length: 杆组静止长度，shape (R,)。
        rod_length_limits: 杆组行程限制 [min, max]，shape (R, 2)。
        rod_force_limits: 杆组出力限制 [min, max]，shape (R, 2)。
        rod_group_mass: 杆组质量，shape (R,)。
        rod_radius: 杆组半径，shape (R,)。
        rod_sleeve_half: 杆组套筒显示参数，shape (R, 3)。
        control_group_ids: 控制组标识符列表。
        control_group_count: 控制组数量。
        control_group_enabled: 控制组是否启用，shape (C,)。
        control_group_default_target: 控制组默认目标值，shape (C,)。
        control_group_colors: 控制组显示颜色，shape (C, 3)。
        script: 动作脚本矩阵，shape (C, num_actions)。
        num_actions: 脚本动作数。
    """

    config: SimulationConfig
    robot_config: RobotConfig
    anchor_ids: list[str]
    anchor_count: int
    anchor_rest_pos: FloatArray
    anchor_fixed: BoolArray
    anchor_mass: FloatArray
    anchor_radius: FloatArray
    anchor_projection_target: BoolArray
    projection_anchor_indices: IntArray
    rod_group_ids: list[str]
    rod_count: int
    rod_anchors: IntArray
    rod_type: IntArray
    rod_enabled: BoolArray
    rod_control_group: IntArray
    rod_rest_length: FloatArray
    rod_length_limits: FloatArray
    rod_force_limits: FloatArray
    rod_group_mass: FloatArray
    rod_radius: FloatArray
    rod_sleeve_half: FloatArray
    control_group_ids: list[str]
    control_group_count: int
    control_group_enabled: BoolArray
    control_group_default_target: FloatArray
    control_group_colors: UInt8Array
    script: FloatArray
    num_actions: int


def compile_workspace(workspace: Workspace) -> VGTRModel:
    """将编辑态工作区编译为只读运行时模型。

    Args:
        workspace: 当前工作区。

    Returns:
        编译后的 VGTRModel 实例。
    """
    topology = workspace.topology
    script = workspace.script
    projection_anchor_indices = np.flatnonzero(topology.anchor_projection_target).astype(np.int32)
    return VGTRModel(
        config=workspace.config,
        robot_config=workspace.robot_config,
        anchor_ids=topology.anchor_ids.copy(),
        anchor_count=int(topology.anchor_pos.shape[0]),
        anchor_rest_pos=topology.anchor_pos.copy(),
        anchor_fixed=topology.anchor_fixed.copy(),
        anchor_mass=topology.anchor_mass.copy(),
        anchor_radius=topology.anchor_radius.copy(),
        anchor_projection_target=topology.anchor_projection_target.copy(),
        projection_anchor_indices=projection_anchor_indices,
        rod_group_ids=topology.rod_group_ids.copy(),
        rod_count=int(topology.rod_anchors.shape[0]),
        rod_anchors=topology.rod_anchors.copy(),
        rod_type=topology.rod_type.copy(),
        rod_enabled=topology.rod_enabled.copy(),
        rod_control_group=topology.rod_control_group.copy(),
        rod_rest_length=topology.rod_rest_length.copy(),
        rod_length_limits=topology.rod_length_limits.copy(),
        rod_force_limits=topology.rod_force_limits.copy(),
        rod_group_mass=topology.rod_group_mass.copy(),
        rod_radius=topology.rod_radius.copy(),
        rod_sleeve_half=topology.rod_sleeve_half.copy(),
        control_group_ids=script.control_group_ids.copy(),
        control_group_count=int(script.num_channels),
        control_group_enabled=script.control_group_enabled.copy(),
        control_group_default_target=script.control_group_default_target.copy(),
        control_group_colors=script.control_group_colors.copy(),
        script=script.script.copy(),
        num_actions=int(script.num_actions),
    )
