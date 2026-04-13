"""
工作区及物理数据结构定义。

本模块定义了描述网格仿真全状态的核心容器类 Workspace，
以及拓扑、物理、控制脚本、界面等相关状态的数据结构。
主要用于变几何桁架机器人/软体机器人仿真，
支持状态快照、恢复、导入导出等功能。

主要内容：
1. TopologyState：描述机器人结构与连接关系。
2. PhysicsState：描述仿真运行时的物理量。
3. ScriptState：描述控制脚本与分组颜色等。
4. UiState：描述界面交互相关状态。
5. Workspace：核心容器，聚合上述所有状态。
6. 各类辅助函数：用于数组初始化、数据转换等。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import RobotConfig, SimulationConfig, default_robot_config
from .schema import (
    ControlGroupFile,
    RodGroupFile,
    SiteFile,
    WorkspaceFile,
    WorkspaceFileData,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
StatusArray = NDArray[np.int8]
UInt8Array = NDArray[np.uint8]

# 预定义的默认颜色调色板，用于控制组的默认颜色分配。
_DEFAULT_COLOR_PALETTE = np.asarray(
    [
        [26, 77, 153],
        [128, 102, 26],
        [128, 26, 128],
        [26, 128, 51],
        [153, 51, 102],
        [51, 153, 102],
        [102, 153, 51],
        [153, 153, 51],
        [51, 153, 153],
    ],
    dtype=np.uint8,
)


def _empty_bools() -> BoolArray:
    """
    创建一个空的布尔数组。

    Returns:
        BoolArray: 长度为0的np.bool_类型数组。
    """
    return np.zeros(0, dtype=np.bool_)

def _empty_colors() -> UInt8Array:
    """
    创建一个空的颜色数组，形状为(0, 3)。

    Returns:
        UInt8Array: 形状为(0, 3)的np.uint8类型数组。
    """
    return np.zeros((0, 3), dtype=np.uint8)

def _empty_float3() -> FloatArray:
    """
    创建一个空的三维浮点数组，形状为(0, 3)。

    Returns:
        FloatArray: 形状为(0, 3)的np.float64类型数组。
    """
    return np.zeros((0, 3), dtype=np.float64)


@dataclass(slots=True)
class TopologyState:
    """
    变几何桁架机器人的拓扑状态。
    描述机器人结构、锚点、杆组及其连接关系和物理属性。
    """
    anchor_ids: list[str]               # 锚点ID列表，顺序与anchor_pos等数组索引一致
    anchor_pos: FloatArray              # 锚点三维坐标，形状 (anchor_count, 3)
    anchor_fixed: BoolArray             # 锚点是否固定，形状 (anchor_count,)
    anchor_mass: FloatArray             # 锚点质量，形状 (anchor_count,)
    anchor_radius: FloatArray           # 锚点半径，形状 (anchor_count,)
    rod_group_ids: list[str]            # 杆组ID列表，顺序与rod_anchors等数组索引一致
    rod_anchors: IntArray               # 每根杆连接的两个锚点索引，形状 (rod_count, 2)
    rod_rest_length: FloatArray         # 杆初始（静止）长度，形状 (rod_count,)
    rod_min_length: FloatArray          # 杆最小允许长度，形状 (rod_count,)
    rod_control_group: IntArray         # 杆所属控制组索引，形状 (rod_count,)
    rod_enabled: BoolArray              # 杆是否启用，形状 (rod_count,)
    rod_actuated: BoolArray             # 杆是否可驱动，形状 (rod_count,)
    rod_group_mass: FloatArray          # 杆组质量，形状 (rod_count,)
    rod_radius: FloatArray              # 杆半径，形状 (rod_count,)
    rod_sleeve_half: FloatArray         # 杆套筒半长，形状 (rod_count, 3)

@dataclass(slots=True)
class PhysicsState:
    """
    锚点级物理与运行时状态。
    保存仿真过程中所有动态物理量，便于状态更新、快照和恢复。
    """
    v0: FloatArray          # 锚点初始位置，形状 (anchor_count, 3)，用于重置/回放
    velocities: FloatArray  # 锚点速度，形状 (anchor_count, 3)
    forces: FloatArray      # 锚点受力，形状 (anchor_count, 3)
    lengths: FloatArray     # 各杆当前长度，形状 (rod_count,)
    control_group_target: FloatArray  # 控制组目标值，形状 (num_control_groups,)
    control_group_value: FloatArray   # 控制组当前值，形状 (num_control_groups,)
    num_steps: int = 0                # 仿真步数计数器
    i_action: int = 0                 # 当前动作索引（用于脚本）
    i_action_prev: int = 0            # 上一步动作索引
    record_frames: bool = False       # 是否记录仿真帧
    frames: list[list[list[float]]] | None = None  # 仿真帧序列（用于回放/导出）

@dataclass(slots=True)
class ScriptState:
    """
    控制组脚本与颜色定义。
    保存控制脚本、分组ID、颜色、使能状态等。
    """
    script: FloatArray  # 控制脚本数组，形状 (num_channels, num_actions)
    num_channels: int   # 控制通道数（控制组数）
    num_actions: int    # 动作步数
    control_group_ids: list[str] = field(default_factory=list)  # 控制组ID列表
    control_group_colors: UInt8Array = field(default_factory=_empty_colors)  # 控制组颜色 (num_channels, 3)
    control_group_enabled: BoolArray = field(default_factory=_empty_bools)  # 控制组使能状态 (num_channels,)

@dataclass(slots=True)
class UiState:
    """
    用户界面交互状态。
    保存与UI相关的选择、编辑、显示等状态。
    """
    anchor_status: StatusArray      # 锚点UI状态（选中/高亮等），形状 (anchor_count,)
    rod_group_status: StatusArray   # 杆组UI状态，形状 (rod_count,)
    face_status: StatusArray        # 面片UI状态，形状 (face_count,)
    editing: bool = False           # 是否处于编辑模式
    moving_anchor: bool = False     # 是否正在移动锚点
    moving_body: bool = False       # 是否正在移动整体
    show_control_group: bool = False # 是否显示控制组
    simulate: bool = True           # 是否处于仿真模式
    record: bool = False            # 是否录制仿真

@dataclass(slots=True)
class WorkspaceSnapshot:
    """
    工作区快照。
    用于保存/恢复仿真全状态，支持撤销、回放等功能。
    """
    topology: TopologyState
    physics: PhysicsState
    script: ScriptState
    ui: UiState

@dataclass(slots=True)
class Workspace:
    """
    VGTR 工作区核心容器。
    聚合仿真所需的所有状态，支持快照、恢复、导入导出等操作。
    """
    config: SimulationConfig         # 仿真全局配置
    robot_config: RobotConfig        # 机器人结构/参数配置
    topology: TopologyState          # 拓扑结构状态
    physics: PhysicsState            # 物理仿真状态
    script: ScriptState              # 控制脚本与分组状态
    ui: UiState                      # UI交互状态
    storage_format: str = "vgtr"     # 存储格式标识

    @classmethod
    def from_file_data(
        cls,
        workspace_file: WorkspaceFileData,
        config: SimulationConfig,
        robot_config: RobotConfig | None = None,
    ) -> Workspace:
        """
        由WorkspaceFileData对象创建Workspace实例。

        Args:
            workspace_file (WorkspaceFileData): 工作区文件数据对象。
            config (SimulationConfig): 仿真配置。
            robot_config (RobotConfig, optional): 机器人配置。
        Returns:
            Workspace: 工作区实例。
        """
        return cls.from_workspace_file(workspace_file, config, robot_config=robot_config)

    @classmethod
    def from_workspace_file(
        cls,
        workspace_file: WorkspaceFile,
        config: SimulationConfig,
        robot_config: RobotConfig | None = None,
    ) -> Workspace:
        """
        由WorkspaceFile对象创建Workspace实例。

        Args:
            workspace_file (WorkspaceFile): 工作区文件对象。
            config (SimulationConfig): 仿真配置。
            robot_config (RobotConfig, optional): 机器人配置。
        Returns:
            Workspace: 工作区实例。
        """
        robot = robot_config or default_robot_config()
        anchor_ids = list(workspace_file.sites.keys())
        anchor_count = len(anchor_ids)
        anchor_index = {anchor_id: i for i, anchor_id in enumerate(anchor_ids)}

        anchor_pos = np.asarray(
            [workspace_file.sites[anchor_id].pos for anchor_id in anchor_ids],
            dtype=np.float64,
        )
        anchor_fixed = np.zeros(anchor_count, dtype=np.bool_)
        anchor_mass = np.full(anchor_count, float(robot.anchor.mass), dtype=np.float64)
        anchor_radius = np.full(anchor_count, float(robot.anchor.radius), dtype=np.float64)

        control_groups = _normalized_control_groups(workspace_file)
        control_group_ids = [group.name for group in control_groups]
        control_group_lookup = {group_name: i for i, group_name in enumerate(control_group_ids)}

        rod_group_ids: list[str] = []
        rod_anchors = np.zeros((len(workspace_file.rod_groups), 2), dtype=np.int32)
        rod_rest_length = np.zeros(len(workspace_file.rod_groups), dtype=np.float64)
        rod_min_length = np.zeros(len(workspace_file.rod_groups), dtype=np.float64)
        rod_control_group = np.zeros(len(workspace_file.rod_groups), dtype=np.int32)
        rod_enabled = np.ones(len(workspace_file.rod_groups), dtype=np.bool_)
        rod_actuated = np.ones(len(workspace_file.rod_groups), dtype=np.bool_)
        rod_group_mass = np.ones(len(workspace_file.rod_groups), dtype=np.float64)
        rod_radius = np.full(
            len(workspace_file.rod_groups),
            float(robot.rod_group.rod_radius),
            dtype=np.float64,
        )
        default_sleeve = np.asarray(
            [
                float(robot.rod_group.sleeve_radius),
                float(robot.rod_group.sleeve_radius),
                float(robot.rod_group.sleeve_display_half_length_ratio),
            ],
            dtype=np.float64,
        )
        rod_sleeve_half = np.tile(
            default_sleeve,
            (len(workspace_file.rod_groups), 1),
        )

        for i, rod_group in enumerate(workspace_file.rod_groups):
            rod_group_ids.append(rod_group.name)
            rod_anchors[i] = [anchor_index[rod_group.site1], anchor_index[rod_group.site2]]

            anchor_distance = float(
                np.linalg.norm(anchor_pos[rod_anchors[i, 1]] - anchor_pos[rod_anchors[i, 0]])
            )
            min_length = anchor_distance
            rest_length = min_length + float(robot.rod_group.length_delta)

            rod_rest_length[i] = rest_length
            rod_min_length[i] = max(min_length, 1e-6)
            control_name = rod_group.control_group or rod_group.name
            rod_control_group[i] = control_group_lookup[control_name]
            rod_enabled[i] = True
            rod_actuated[i] = True
            rod_group_mass[i] = 1.0

            # 套筒参数：仅使用新字段。
            sleeve_radius = float(robot.rod_group.sleeve_radius)
            sleeve_ratio = float(robot.rod_group.sleeve_display_half_length_ratio)
            if rod_group.sleeve_radius is not None:
                sleeve_radius = float(rod_group.sleeve_radius)
            if rod_group.sleeve_display_half_length_ratio is not None:
                sleeve_ratio = float(rod_group.sleeve_display_half_length_ratio)
            rod_sleeve_half[i] = np.asarray(
                [
                    max(sleeve_radius, 1e-6),
                    max(sleeve_radius, 1e-6),
                    float(np.clip(sleeve_ratio, 0.0, 0.5)),
                ],
                dtype=np.float64,
            )

        num_channels = len(control_groups)
        num_actions = workspace_file.num_actions or _infer_num_actions(workspace_file.script)
        script = _script_array(
            workspace_file.script,
            num_channels=num_channels,
            num_actions=num_actions,
        )
        control_group_target = np.asarray(
            [float(group.default_target or 0.0) for group in control_groups],
            dtype=np.float64,
        )
        control_group_value = control_group_target.copy()
        control_group_colors = np.asarray(
            [_color_to_uint8(group.color, index=i) for i, group in enumerate(control_groups)],
            dtype=np.uint8,
        )
        control_group_enabled = np.asarray(
            [bool(group.enabled) for group in control_groups],
            dtype=np.bool_,
        )

        topology = TopologyState(
            anchor_ids=anchor_ids,
            anchor_pos=anchor_pos,
            anchor_fixed=anchor_fixed,
            anchor_mass=anchor_mass,
            anchor_radius=anchor_radius,
            rod_group_ids=rod_group_ids,
            rod_anchors=rod_anchors,
            rod_rest_length=rod_rest_length,
            rod_min_length=rod_min_length,
            rod_control_group=rod_control_group,
            rod_enabled=rod_enabled,
            rod_actuated=rod_actuated,
            rod_group_mass=rod_group_mass,
            rod_radius=rod_radius,
            rod_sleeve_half=rod_sleeve_half,
        )
        physics = PhysicsState(
            v0=_record_v0(anchor_pos),
            velocities=np.zeros_like(anchor_pos),
            forces=np.zeros_like(anchor_pos),
            lengths=_edge_lengths(anchor_pos, rod_anchors),
            control_group_target=control_group_target,
            control_group_value=control_group_value,
            frames=deepcopy(workspace_file.anchor_frames),
        )
        ui = UiState(
            anchor_status=np.zeros(anchor_count, dtype=np.int8),
            rod_group_status=np.zeros(rod_anchors.shape[0], dtype=np.int8),
            face_status=np.zeros(0, dtype=np.int8),
            show_control_group=True,
        )
        script_state = ScriptState(
            script=script,
            num_channels=num_channels,
            num_actions=num_actions,
            control_group_ids=control_group_ids,
            control_group_colors=control_group_colors,
            control_group_enabled=control_group_enabled,
        )
        return cls(
            config=config,
            robot_config=robot,
            topology=topology,
            physics=physics,
            script=script_state,
            ui=ui,
            storage_format="vgtr",
        )

    def snapshot(self) -> WorkspaceSnapshot:
        """
        获取当前工作区的完整快照。

        Returns:
            WorkspaceSnapshot: 包含所有状态的深拷贝。
        """
        return WorkspaceSnapshot(
            topology=TopologyState(
                anchor_ids=self.topology.anchor_ids.copy(),
                anchor_pos=self.topology.anchor_pos.copy(),
                anchor_fixed=self.topology.anchor_fixed.copy(),
                anchor_mass=self.topology.anchor_mass.copy(),
                anchor_radius=self.topology.anchor_radius.copy(),
                rod_group_ids=self.topology.rod_group_ids.copy(),
                rod_anchors=self.topology.rod_anchors.copy(),
                rod_rest_length=self.topology.rod_rest_length.copy(),
                rod_min_length=self.topology.rod_min_length.copy(),
                rod_control_group=self.topology.rod_control_group.copy(),
                rod_enabled=self.topology.rod_enabled.copy(),
                rod_actuated=self.topology.rod_actuated.copy(),
                rod_group_mass=self.topology.rod_group_mass.copy(),
                rod_radius=self.topology.rod_radius.copy(),
                rod_sleeve_half=self.topology.rod_sleeve_half.copy(),
            ),
            physics=PhysicsState(
                v0=self.physics.v0.copy(),
                velocities=self.physics.velocities.copy(),
                forces=self.physics.forces.copy(),
                lengths=self.physics.lengths.copy(),
                control_group_target=self.physics.control_group_target.copy(),
                control_group_value=self.physics.control_group_value.copy(),
                num_steps=self.physics.num_steps,
                i_action=self.physics.i_action,
                i_action_prev=self.physics.i_action_prev,
                record_frames=self.physics.record_frames,
                frames=deepcopy(self.physics.frames),
            ),
            script=ScriptState(
                script=self.script.script.copy(),
                num_channels=self.script.num_channels,
                num_actions=self.script.num_actions,
                control_group_ids=self.script.control_group_ids.copy(),
                control_group_colors=self.script.control_group_colors.copy(),
                control_group_enabled=self.script.control_group_enabled.copy(),
            ),
            ui=UiState(
                anchor_status=self.ui.anchor_status.copy(),
                rod_group_status=self.ui.rod_group_status.copy(),
                face_status=self.ui.face_status.copy(),
                editing=self.ui.editing,
                moving_anchor=self.ui.moving_anchor,
                moving_body=self.ui.moving_body,
                show_control_group=self.ui.show_control_group,
                simulate=self.ui.simulate,
                record=self.ui.record,
            ),
        )

    def restore(self, snapshot: WorkspaceSnapshot) -> None:
        """
        用快照恢复工作区所有状态。

        Args:
            snapshot (WorkspaceSnapshot): 快照对象。
        """
        self.topology = snapshot.topology
        self.physics = snapshot.physics
        self.script = snapshot.script
        self.ui = snapshot.ui

    def reset_runtime(self) -> None:
        """
        重置仿真运行时状态（速度、受力、长度、步数等），不影响结构。
        """
        self.physics.velocities.fill(0.0)
        self.physics.forces.fill(0.0)
        self.physics.lengths = _edge_lengths(self.topology.anchor_pos, self.topology.rod_anchors)
        self.physics.v0 = _record_v0(self.topology.anchor_pos)
        self.physics.num_steps = 0
        self.physics.i_action = 0
        self.physics.i_action_prev = 0
        self.physics.control_group_target = np.zeros(self.script.num_channels, dtype=np.float64)
        self.physics.control_group_value = np.zeros(self.script.num_channels, dtype=np.float64)

    def restore_initial_state(self) -> None:
        """
        恢复锚点到初始位置，并重置速度、受力、长度等。
        """
        self.topology.anchor_pos = self.physics.v0.copy()
        self.physics.velocities = np.zeros_like(self.topology.anchor_pos)
        self.physics.forces = np.zeros_like(self.topology.anchor_pos)
        self.physics.lengths = _edge_lengths(self.topology.anchor_pos, self.topology.rod_anchors)
        self.physics.num_steps = 0
        self.physics.i_action = 0
        self.physics.i_action_prev = 0
        self.physics.control_group_target = np.zeros(self.script.num_channels, dtype=np.float64)
        self.physics.control_group_value = np.zeros(self.script.num_channels, dtype=np.float64)

    def to_workspace_file(self) -> WorkspaceFile:
        """
        导出当前工作区为WorkspaceFile对象（用于保存/导出）。

        Returns:
            WorkspaceFile: 导出的工作区文件对象。
        """
        sites = {
            anchor_id: SiteFile(
                pos=self.topology.anchor_pos[i].tolist(),
                radius=float(self.topology.anchor_radius[i]),
                fixed=bool(self.topology.anchor_fixed[i]),
                mass=float(self.topology.anchor_mass[i]),
            )
            for i, anchor_id in enumerate(self.topology.anchor_ids)
        }
        rod_groups = []
        for i, rod_group_id in enumerate(self.topology.rod_group_ids):
            control_group = None
            if self.script.control_group_ids and i < self.topology.rod_control_group.shape[0]:
                control_group = self.script.control_group_ids[
                    int(self.topology.rod_control_group[i])
                ]
            rod_groups.append(
                RodGroupFile(
                    name=rod_group_id,
                    site1=self.topology.anchor_ids[int(self.topology.rod_anchors[i, 0])],
                    site2=self.topology.anchor_ids[int(self.topology.rod_anchors[i, 1])],
                    actuated=bool(self.topology.rod_actuated[i]),
                    enabled=bool(self.topology.rod_enabled[i]),
                    control_group=control_group,
                    group_mass=float(self.topology.rod_group_mass[i]),
                    rod_radius=float(self.topology.rod_radius[i]),
                    sleeve_radius=float(self.topology.rod_sleeve_half[i, 0]),
                    sleeve_display_half_length_ratio=float(self.topology.rod_sleeve_half[i, 2]),
                    rest_length=float(self.topology.rod_rest_length[i]),
                    min_length=float(self.topology.rod_min_length[i]),
                )
            )
        control_groups = [
            ControlGroupFile(
                name=(
                    self.script.control_group_ids[i]
                    if self.script.control_group_ids
                    else f"control_{i}"
                ),
                color=(self.script.control_group_colors[i].astype(np.float64) / 255.0).tolist()
                if self.script.control_group_colors.shape[0] > i
                else (
                    _DEFAULT_COLOR_PALETTE[i % len(_DEFAULT_COLOR_PALETTE)].astype(np.float64)
                    / 255.0
                ).tolist(),
                default_target=float(self.physics.control_group_target[i])
                if self.physics.control_group_target.shape[0] > i
                else 0.0,
                enabled=bool(self.script.control_group_enabled[i])
                if self.script.control_group_enabled.shape[0] > i
                else True,
            )
            for i in range(self.script.num_channels)
        ]
        return WorkspaceFile(
            sites=sites,
            rod_groups=rod_groups,
            control_groups=control_groups,
            script=self.script.script.tolist(),
            num_actions=self.script.num_actions,
            anchor_frames=deepcopy(self.physics.frames or []),
        )


def _float_array(values: list[float], *, size: int, fill: float) -> FloatArray:
    """
    构造指定长度的浮点数组。

    Args:
        values (list[float]): 初始值列表。
        size (int): 目标长度。
        fill (float): 填充值。
    Returns:
        FloatArray: np.float64类型数组。
    """
    array = np.full(size, fill, dtype=np.float64)
    if values:
        value_array = np.asarray(values, dtype=np.float64)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _int_array(values: list[int], *, size: int, fill: int) -> IntArray:
    """
    构造指定长度的整型数组。

    Args:
        values (list[int]): 初始值列表。
        size (int): 目标长度。
        fill (int): 填充值。
    Returns:
        IntArray: np.int32类型数组。
    """
    array = np.full(size, fill, dtype=np.int32)
    if values:
        value_array = np.asarray(values, dtype=np.int32)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _bool_array(values: list[bool | int], *, size: int, fill: bool) -> BoolArray:
    """
    构造指定长度的布尔数组。

    Args:
        values (list[bool | int]): 初始值列表。
        size (int): 目标长度。
        fill (bool): 填充值。
    Returns:
        BoolArray: np.bool_类型数组。
    """
    array = np.full(size, fill, dtype=np.bool_)
    if values:
        value_array = np.asarray(values, dtype=np.bool_)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _status_array(values: list[int], *, size: int) -> StatusArray:
    """
    构造指定长度的状态数组（int8）。

    Args:
        values (list[int]): 初始值列表。
        size (int): 目标长度。
    Returns:
        StatusArray: np.int8类型数组。
    """
    array = np.zeros(size, dtype=np.int8)
    if values and size:
        value_array = np.asarray(values, dtype=np.int8)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _script_array(
    values: list[list[bool | int | float]],
    *,
    num_channels: int,
    num_actions: int,
) -> FloatArray:
    """
    构造控制脚本二维数组。

    Args:
        values (list[list[bool | int | float]]): 原始二维列表。
        num_channels (int): 通道数。
        num_actions (int): 动作步数。
    Returns:
        FloatArray: np.float64类型二维数组。
    """
    array = np.zeros((num_channels, num_actions), dtype=np.float64)
    if not values:
        return array
    for channel_index, row in enumerate(values[:num_channels]):
        row_array = np.asarray(row, dtype=np.float64)
        array[channel_index, : min(num_actions, row_array.shape[0])] = row_array[:num_actions]
    return array


def _edge_lengths(vertices: FloatArray, edges: IntArray) -> FloatArray:
    """
    计算所有边（杆）的长度。

    Args:
        vertices (FloatArray): 顶点坐标数组。
        edges (IntArray): 边的索引数组。
    Returns:
        FloatArray: 每条边的长度数组。
    """
    if edges.size == 0:
        return np.zeros(0, dtype=np.float64)
    edge_vec = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    return np.linalg.norm(edge_vec, axis=1)


def _record_v0(vertices: FloatArray) -> FloatArray:
    """
    记录初始位置（z轴归零并上移），用于仿真重置。

    Args:
        vertices (FloatArray): 顶点坐标数组。
    Returns:
        FloatArray: 处理后的初始位置数组。
    """
    if vertices.size == 0:
        return vertices.copy()
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    z_offset = -bbox_min[2]
    v0 = vertices.copy()
    v0[:, 2] += z_offset
    if bbox_max[2] != bbox_min[2]:
        v0[:, 2] += bbox_max[2] - bbox_min[2]
    return v0


def _infer_num_actions(script: list[list[bool | int | float]]) -> int:
    """
    推断控制脚本的动作步数。

    Args:
        script (list[list[bool | int | float]]): 控制脚本二维列表。
    Returns:
        int: 动作步数。
    """
    if not script:
        return 1
    return max(len(row) for row in script) or 1


def _float3(values: list[float], *, fill: float) -> FloatArray:
    """
    构造三维浮点数组。

    Args:
        values (list[float]): 初始值列表。
        fill (float): 填充值。
    Returns:
        FloatArray: np.float64类型长度为3的数组。
    """
    array = np.full(3, fill, dtype=np.float64)
    if values:
        value_array = np.asarray(values, dtype=np.float64)
        array[: min(3, value_array.shape[0])] = value_array[:3]
    return array


def _normalized_control_groups(workspace_file: WorkspaceFile) -> list[ControlGroupFile]:
    """
    归一化控制组列表，确保所有杆组均有对应控制组。

    Args:
        workspace_file (WorkspaceFile): 工作区文件对象。
    Returns:
        list[ControlGroupFile]: ControlGroupFile对象列表。
    """
    if workspace_file.control_groups:
        groups = list(workspace_file.control_groups)
    else:
        groups = []

    existing = {group.name for group in groups}
    for rod_group in workspace_file.rod_groups:
        group_name = rod_group.control_group or rod_group.name
        if group_name not in existing:
            groups.append(ControlGroupFile(name=group_name))
            existing.add(group_name)
    return groups


def _default_control_colors(count: int) -> UInt8Array:
    """
    生成指定数量的默认控制组颜色。

    Args:
        count (int): 颜色数量。
    Returns:
        UInt8Array: np.uint8类型颜色数组。
    """
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(
        [_DEFAULT_COLOR_PALETTE[i % len(_DEFAULT_COLOR_PALETTE)] for i in range(count)],
        dtype=np.uint8,
    )


def _color_to_uint8(values: list[float], *, index: int) -> np.ndarray:
    """
    将颜色值（0~1或0~255）转换为uint8 RGB数组。

    Args:
        values (list[float]): 原始颜色。
        index (int): 默认调色板索引。
    Returns:
        np.ndarray: np.uint8类型RGB数组。
    """
    if not values:
        return _DEFAULT_COLOR_PALETTE[index % len(_DEFAULT_COLOR_PALETTE)].copy()
    array = np.asarray(values, dtype=np.float64)
    if array.size >= 3 and np.max(array[:3]) <= 1.0:
        rgb = np.clip(np.round(array[:3] * 255.0), 0, 255)
    else:
        rgb = np.clip(np.round(array[:3]), 0, 255)
    return rgb.astype(np.uint8)

