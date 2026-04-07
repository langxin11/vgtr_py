"""工作区及物理数据结构定义。

定义了描述网格仿真全状态的核心容器类（`Workspace`），包含拓扑、驱动器及运动配置实例等关键引用。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .schema import (
    ControlGroupFile,
    LegacyWorkspaceFile,
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
    return np.zeros(0, dtype=np.bool_)


def _empty_colors() -> UInt8Array:
    return np.zeros((0, 3), dtype=np.uint8)


def _empty_float3() -> FloatArray:
    return np.zeros((0, 3), dtype=np.float64)


@dataclass(slots=True)
class TopologyState:
    """变几何桁架机器人的拓扑状态。"""

    anchor_ids: list[str]
    anchor_pos: FloatArray
    anchor_fixed: BoolArray
    anchor_mass: FloatArray
    anchor_radius: FloatArray
    rod_group_ids: list[str]
    rod_anchors: IntArray
    rod_rest_length: FloatArray
    rod_min_length: FloatArray
    rod_control_group: IntArray
    rod_enabled: BoolArray
    rod_actuated: BoolArray
    rod_group_mass: FloatArray
    rod_radius: FloatArray
    rod_sleeve_half: FloatArray

    @property
    def vertices(self) -> FloatArray:
        return self.anchor_pos

    @vertices.setter
    def vertices(self, value: FloatArray) -> None:
        self.anchor_pos = np.asarray(value, dtype=np.float64)

    @property
    def edges(self) -> IntArray:
        return self.rod_anchors

    @edges.setter
    def edges(self, value: IntArray) -> None:
        self.rod_anchors = np.asarray(value, dtype=np.int32)

    @property
    def fixed_vs(self) -> BoolArray:
        return self.anchor_fixed

    @fixed_vs.setter
    def fixed_vs(self, value: BoolArray) -> None:
        self.anchor_fixed = np.asarray(value, dtype=np.bool_)

    @property
    def l_max(self) -> FloatArray:
        return self.rod_rest_length

    @l_max.setter
    def l_max(self, value: FloatArray) -> None:
        previous = self.max_contraction.copy()
        self.rod_rest_length = np.asarray(value, dtype=np.float64)
        self.max_contraction = previous

    @property
    def max_contraction(self) -> FloatArray:
        if self.rod_rest_length.size == 0:
            return np.zeros(0, dtype=np.float64)
        safe_rest = np.where(self.rod_rest_length > 1e-12, self.rod_rest_length, 1.0)
        contraction = 1.0 - (self.rod_min_length / safe_rest)
        return np.round(np.clip(contraction, 0.0, 1.0), decimals=12)

    @max_contraction.setter
    def max_contraction(self, value: FloatArray) -> None:
        contraction = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
        resized = np.zeros(self.rod_rest_length.shape[0], dtype=np.float64)
        resized[: min(resized.shape[0], contraction.shape[0])] = contraction[: resized.shape[0]]
        self.rod_min_length = self.rod_rest_length * (1.0 - resized)

    @property
    def edge_channel(self) -> IntArray:
        return self.rod_control_group

    @edge_channel.setter
    def edge_channel(self, value: IntArray) -> None:
        self.rod_control_group = np.asarray(value, dtype=np.int32)

    @property
    def edge_active(self) -> BoolArray:
        return self.rod_enabled

    @edge_active.setter
    def edge_active(self, value: BoolArray) -> None:
        self.rod_enabled = np.asarray(value, dtype=np.bool_)


@dataclass(slots=True)
class PhysicsState:
    """锚点级物理与运行时状态。"""

    v0: FloatArray
    velocities: FloatArray
    forces: FloatArray
    lengths: FloatArray
    control_group_target: FloatArray
    control_group_value: FloatArray
    num_steps: int = 0
    i_action: int = 0
    i_action_prev: int = 0
    record_frames: bool = False
    frames: list[list[list[float]]] | None = None

    @property
    def inflate_channel(self) -> BoolArray:
        return self.control_group_target >= 0.5

    @inflate_channel.setter
    def inflate_channel(self, value: BoolArray) -> None:
        self.control_group_target = np.asarray(value, dtype=np.float64)

    @property
    def contraction_percent(self) -> FloatArray:
        return 1.0 - self.control_group_value

    @contraction_percent.setter
    def contraction_percent(self, value: FloatArray) -> None:
        self.control_group_value = 1.0 - np.asarray(value, dtype=np.float64)


@dataclass(slots=True)
class ScriptState:
    """控制组脚本与颜色定义。"""

    script: FloatArray
    num_channels: int
    num_actions: int
    control_group_ids: list[str] = field(default_factory=list)
    control_group_colors: UInt8Array = field(default_factory=_empty_colors)
    control_group_enabled: BoolArray = field(default_factory=_empty_bools)

    @property
    def num_control_groups(self) -> int:
        return self.num_channels


@dataclass(slots=True)
class UiState:
    """用户界面交互状态。"""

    anchor_status: StatusArray
    rod_group_status: StatusArray
    face_status: StatusArray
    editing: bool = False
    moving_anchor: bool = False
    moving_body: bool = False
    show_control_group: bool = False
    simulate: bool = True
    record: bool = False

    @property
    def vertex_status(self) -> StatusArray:
        return self.anchor_status

    @vertex_status.setter
    def vertex_status(self, value: StatusArray) -> None:
        self.anchor_status = np.asarray(value, dtype=np.int8)

    @property
    def edge_status(self) -> StatusArray:
        return self.rod_group_status

    @edge_status.setter
    def edge_status(self, value: StatusArray) -> None:
        self.rod_group_status = np.asarray(value, dtype=np.int8)

    @property
    def moving_joint(self) -> bool:
        return self.moving_anchor

    @moving_joint.setter
    def moving_joint(self, value: bool) -> None:
        self.moving_anchor = bool(value)

    @property
    def show_channel(self) -> bool:
        return self.show_control_group

    @show_channel.setter
    def show_channel(self, value: bool) -> None:
        self.show_control_group = bool(value)


@dataclass(slots=True)
class WorkspaceSnapshot:
    """工作区快照。"""

    topology: TopologyState
    physics: PhysicsState
    script: ScriptState
    ui: UiState


@dataclass(slots=True)
class Workspace:
    """统一的 VGTR / PneuMesh 过渡工作区。"""

    config: SimulationConfig
    topology: TopologyState
    physics: PhysicsState
    script: ScriptState
    ui: UiState
    storage_format: str = "legacy"

    @classmethod
    def from_file_data(
        cls,
        workspace_file: WorkspaceFileData,
        config: SimulationConfig,
    ) -> Workspace:
        if isinstance(workspace_file, WorkspaceFile):
            return cls.from_workspace_file(workspace_file, config)
        return cls.from_legacy_file(workspace_file, config)

    @classmethod
    def from_legacy_file(
        cls,
        workspace_file: LegacyWorkspaceFile,
        config: SimulationConfig,
    ) -> Workspace:
        vertices = np.asarray(workspace_file.v, dtype=np.float64)
        edges = np.asarray(workspace_file.e, dtype=np.int32)

        num_vertices = vertices.shape[0]
        num_edges = edges.shape[0]
        num_channels = workspace_file.numChannels or config.default_num_channels
        num_actions = workspace_file.numActions or config.default_num_actions

        fixed_vs = _bool_array(workspace_file.fixedVs, size=num_vertices, fill=False)
        l_max = _float_array(workspace_file.lMax, size=num_edges, fill=config.default_max_length)
        max_contraction = _float_array(
            workspace_file.maxContraction,
            size=num_edges,
            fill=config.max_max_contraction,
        )
        rod_min_length = l_max * (1.0 - max_contraction)
        edge_channel = _int_array(workspace_file.edgeChannel, size=num_edges, fill=0)
        edge_active = _bool_array(workspace_file.edgeActive, size=num_edges, fill=True)
        script = _script_array(
            workspace_file.script,
            num_channels=num_channels,
            num_actions=num_actions,
        )

        lengths = _edge_lengths(vertices, edges)
        velocities = np.zeros_like(vertices)
        forces = np.zeros_like(vertices)
        v0 = (
            np.asarray(workspace_file.v0, dtype=np.float64)
            if workspace_file.v0
            else _record_v0(vertices)
        )
        if v0.shape != vertices.shape:
            v0 = _record_v0(vertices)

        legacy_inflate = _bool_array(workspace_file.inflateChannel, size=num_channels, fill=False)
        legacy_contraction = _float_array(
            workspace_file.contractionPercent,
            size=num_channels,
            fill=1.0,
        )
        control_group_target = legacy_inflate.astype(np.float64)
        control_group_value = np.clip(1.0 - legacy_contraction, 0.0, 1.0)

        vertex_status = _status_array(workspace_file.vStatus, size=num_vertices)
        edge_status = _status_array(workspace_file.eStatus, size=num_edges)
        face_status = _status_array(workspace_file.fStatus, size=0)

        topology = TopologyState(
            anchor_ids=[f"s{i + 1}" for i in range(num_vertices)],
            anchor_pos=vertices,
            anchor_fixed=fixed_vs,
            anchor_mass=np.ones(num_vertices, dtype=np.float64),
            anchor_radius=np.full(num_vertices, 0.06, dtype=np.float64),
            rod_group_ids=[f"g{i}" for i in range(num_edges)],
            rod_anchors=edges,
            rod_rest_length=l_max,
            rod_min_length=rod_min_length,
            rod_control_group=edge_channel,
            rod_enabled=edge_active,
            rod_actuated=edge_active.copy(),
            rod_group_mass=np.ones(num_edges, dtype=np.float64),
            rod_radius=np.full(num_edges, 0.02, dtype=np.float64),
            rod_sleeve_half=np.zeros((num_edges, 3), dtype=np.float64),
        )
        physics = PhysicsState(
            v0=v0,
            velocities=velocities,
            forces=forces,
            lengths=lengths,
            control_group_target=control_group_target,
            control_group_value=control_group_value,
            frames=deepcopy(workspace_file.v_frames),
        )
        script_state = ScriptState(
            script=script,
            num_channels=num_channels,
            num_actions=num_actions,
            control_group_ids=[f"control_{i}" for i in range(num_channels)],
            control_group_colors=_default_control_colors(num_channels),
            control_group_enabled=np.ones(num_channels, dtype=np.bool_),
        )
        ui = UiState(
            anchor_status=vertex_status,
            rod_group_status=edge_status,
            face_status=face_status,
            show_control_group=bool(workspace_file.edgeChannel),
        )
        return cls(
            config=config,
            topology=topology,
            physics=physics,
            script=script_state,
            ui=ui,
            storage_format="legacy",
        )

    @classmethod
    def from_workspace_file(
        cls,
        workspace_file: WorkspaceFile,
        config: SimulationConfig,
    ) -> Workspace:
        anchor_ids = list(workspace_file.sites.keys())
        anchor_count = len(anchor_ids)
        anchor_index = {anchor_id: i for i, anchor_id in enumerate(anchor_ids)}

        anchor_pos = np.asarray(
            [workspace_file.sites[anchor_id].pos for anchor_id in anchor_ids],
            dtype=np.float64,
        )
        anchor_fixed = np.asarray(
            [bool(workspace_file.sites[anchor_id].fixed or False) for anchor_id in anchor_ids],
            dtype=np.bool_,
        )
        anchor_mass = np.asarray(
            [float(workspace_file.sites[anchor_id].mass or 1.0) for anchor_id in anchor_ids],
            dtype=np.float64,
        )
        anchor_radius = np.asarray(
            [float(workspace_file.sites[anchor_id].radius or 0.06) for anchor_id in anchor_ids],
            dtype=np.float64,
        )

        control_groups = _normalized_control_groups(workspace_file)
        control_group_ids = [group.name for group in control_groups]
        control_group_lookup = {group_name: i for i, group_name in enumerate(control_group_ids)}

        rod_group_ids: list[str] = []
        rod_anchors = np.zeros((len(workspace_file.rod_groups), 2), dtype=np.int32)
        rod_rest_length = np.zeros(len(workspace_file.rod_groups), dtype=np.float64)
        rod_min_length = np.zeros(len(workspace_file.rod_groups), dtype=np.float64)
        rod_control_group = np.zeros(len(workspace_file.rod_groups), dtype=np.int32)
        rod_enabled = np.ones(len(workspace_file.rod_groups), dtype=np.bool_)
        rod_actuated = np.zeros(len(workspace_file.rod_groups), dtype=np.bool_)
        rod_group_mass = np.ones(len(workspace_file.rod_groups), dtype=np.float64)
        rod_radius = np.full(len(workspace_file.rod_groups), 0.02, dtype=np.float64)
        rod_sleeve_half = np.zeros((len(workspace_file.rod_groups), 3), dtype=np.float64)

        for i, rod_group in enumerate(workspace_file.rod_groups):
            rod_group_ids.append(rod_group.name)
            rod_anchors[i] = [anchor_index[rod_group.site1], anchor_index[rod_group.site2]]

            anchor_distance = float(
                np.linalg.norm(anchor_pos[rod_anchors[i, 1]] - anchor_pos[rod_anchors[i, 0]])
            )
            rest_length = float(rod_group.rest_length or anchor_distance)
            if rod_group.min_length is not None:
                min_length = float(rod_group.min_length)
            elif rod_group.max_contraction is not None:
                min_length = rest_length * (1.0 - float(rod_group.max_contraction))
            else:
                min_length = rest_length * (1.0 - config.max_max_contraction)

            rod_rest_length[i] = rest_length
            rod_min_length[i] = max(min_length, 1e-6)
            control_name = rod_group.control_group or rod_group.name
            rod_control_group[i] = control_group_lookup[control_name]
            rod_enabled[i] = bool(rod_group.enabled)
            rod_actuated[i] = bool(rod_group.actuated)
            rod_group_mass[i] = float(rod_group.group_mass or 1.0)
            rod_radius[i] = float(rod_group.rod_radius or 0.02)
            rod_sleeve_half[i] = _float3(rod_group.sleeve_half, fill=0.0)

        num_channels = len(control_groups)
        num_actions = workspace_file.numActions or _infer_num_actions(workspace_file.script)
        script = _script_array(workspace_file.script, num_channels=num_channels, num_actions=num_actions)
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
            topology=topology,
            physics=physics,
            script=script_state,
            ui=ui,
            storage_format="vgtr",
        )

    def snapshot(self) -> WorkspaceSnapshot:
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
        self.topology = snapshot.topology
        self.physics = snapshot.physics
        self.script = snapshot.script
        self.ui = snapshot.ui

    def reset_runtime(self) -> None:
        self.physics.velocities.fill(0.0)
        self.physics.forces.fill(0.0)
        self.physics.lengths = _edge_lengths(self.topology.anchor_pos, self.topology.rod_anchors)
        self.physics.v0 = _record_v0(self.topology.anchor_pos)
        self.physics.num_steps = 0
        self.physics.i_action = 0
        self.physics.i_action_prev = 0
        self.physics.control_group_target = np.zeros(self.script.num_channels, dtype=np.float64)
        self.physics.control_group_value = np.zeros(self.script.num_channels, dtype=np.float64)

    def to_legacy_file(self) -> LegacyWorkspaceFile:
        return LegacyWorkspaceFile(
            v=self.topology.anchor_pos.tolist(),
            e=self.topology.rod_anchors.tolist(),
            v0=self.physics.v0.tolist(),
            fixedVs=self.topology.anchor_fixed.tolist(),
            lMax=self.topology.rod_rest_length.tolist(),
            edgeChannel=self.topology.rod_control_group.tolist(),
            edgeActive=self.topology.rod_enabled.tolist(),
            script=(self.script.script >= 0.5).astype(bool).tolist(),
            maxContraction=self.topology.max_contraction.tolist(),
            vStatus=self.ui.anchor_status.astype(int).tolist(),
            eStatus=self.ui.rod_group_status.astype(int).tolist(),
            fStatus=self.ui.face_status.astype(int).tolist(),
            numChannels=self.script.num_channels,
            numActions=self.script.num_actions,
            inflateChannel=self.physics.inflate_channel.astype(bool).tolist(),
            contractionPercent=self.physics.contraction_percent.tolist(),
            v_frames=deepcopy(self.physics.frames or []),
        )

    def to_workspace_file(self) -> WorkspaceFile:
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
                control_group = self.script.control_group_ids[int(self.topology.rod_control_group[i])]
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
                    sleeve_half=self.topology.rod_sleeve_half[i].tolist(),
                    rest_length=float(self.topology.rod_rest_length[i]),
                    min_length=float(self.topology.rod_min_length[i]),
                )
            )
        control_groups = [
            ControlGroupFile(
                name=self.script.control_group_ids[i] if self.script.control_group_ids else f"control_{i}",
                color=(self.script.control_group_colors[i].astype(np.float64) / 255.0).tolist()
                if self.script.control_group_colors.shape[0] > i
                else (_DEFAULT_COLOR_PALETTE[i % len(_DEFAULT_COLOR_PALETTE)].astype(np.float64) / 255.0).tolist(),
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
            numActions=self.script.num_actions,
            anchor_frames=deepcopy(self.physics.frames or []),
        )


def _float_array(values: list[float], *, size: int, fill: float) -> FloatArray:
    array = np.full(size, fill, dtype=np.float64)
    if values:
        value_array = np.asarray(values, dtype=np.float64)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _int_array(values: list[int], *, size: int, fill: int) -> IntArray:
    array = np.full(size, fill, dtype=np.int32)
    if values:
        value_array = np.asarray(values, dtype=np.int32)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _bool_array(values: list[bool | int], *, size: int, fill: bool) -> BoolArray:
    array = np.full(size, fill, dtype=np.bool_)
    if values:
        value_array = np.asarray(values, dtype=np.bool_)
        array[: min(size, value_array.shape[0])] = value_array[:size]
    return array


def _status_array(values: list[int], *, size: int) -> StatusArray:
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
    array = np.zeros((num_channels, num_actions), dtype=np.float64)
    if not values:
        return array
    for channel_index, row in enumerate(values[:num_channels]):
        row_array = np.asarray(row, dtype=np.float64)
        array[channel_index, : min(num_actions, row_array.shape[0])] = row_array[:num_actions]
    return array


def _edge_lengths(vertices: FloatArray, edges: IntArray) -> FloatArray:
    if edges.size == 0:
        return np.zeros(0, dtype=np.float64)
    edge_vec = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    return np.linalg.norm(edge_vec, axis=1)


def _record_v0(vertices: FloatArray) -> FloatArray:
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
    if not script:
        return 1
    return max(len(row) for row in script) or 1


def _float3(values: list[float], *, fill: float) -> FloatArray:
    array = np.full(3, fill, dtype=np.float64)
    if values:
        value_array = np.asarray(values, dtype=np.float64)
        array[: min(3, value_array.shape[0])] = value_array[:3]
    return array


def _normalized_control_groups(workspace_file: WorkspaceFile) -> list[ControlGroupFile]:
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
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(
        [_DEFAULT_COLOR_PALETTE[i % len(_DEFAULT_COLOR_PALETTE)] for i in range(count)],
        dtype=np.uint8,
    )


def _color_to_uint8(values: list[float], *, index: int) -> np.ndarray:
    if not values:
        return _DEFAULT_COLOR_PALETTE[index % len(_DEFAULT_COLOR_PALETTE)].copy()
    array = np.asarray(values, dtype=np.float64)
    if array.size >= 3 and np.max(array[:3]) <= 1.0:
        rgb = np.clip(np.round(array[:3] * 255.0), 0, 255)
    else:
        rgb = np.clip(np.round(array[:3]), 0, 255)
    return rgb.astype(np.uint8)
