"""三维图形渲染封装。

负责统管从内部物理拓扑至 Viser 前端三维可视化的转换映射。渲染循环、图元绘制与主题视觉表达皆于此类维护。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import viser

from .workspace import Workspace

ANCHOR_RADIUS = 0.06
ROD_RADIUS = 0.01
ROD_PICK_RADIUS = 0.025
CHANNEL_COLORS = np.asarray(
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
WHITE = np.array([153, 153, 153], dtype=np.uint8)
BLACK = np.array([13, 13, 13], dtype=np.uint8)
RED = np.array([230, 0, 0], dtype=np.uint8)
ORANGE = np.array([230, 13, 0], dtype=np.uint8)
CYAN = np.array([51, 153, 153], dtype=np.uint8)


@dataclass
class RodVisualGeometry:
    """单杆套筒相对活塞渲染的局部几何参数。

    Attributes:
        sleeve_radius: 套筒半径。
        sleeve_height: 套筒高度。
        rod_radius: 活塞杆半径。
        rod_height: 活塞杆高度。
        rod_l_position: 左端活塞杆局部位置。
        rod_r_position: 右端活塞杆局部位置。
    """

    sleeve_radius: float
    sleeve_height: float
    rod_radius: float
    rod_height: float
    rod_l_position: tuple[float, float, float]
    rod_r_position: tuple[float, float, float]


@dataclass
class SceneRenderer:
    """负责将工作区状态渲染到 Viser 场景。

    Attributes:
        server: Viser 服务器实例。
        on_anchor_click: 锚点点击回调。
        on_rod_group_click: 杆组点击回调。
        anchor_handles: 锚点图元句柄映射。
        sleeve_handles: 杆组套筒图元句柄映射。
        rod_l_handles: 杆组左端活塞图元句柄映射。
        rod_r_handles: 杆组右端活塞图元句柄映射。
        rod_hitbox_handles: 杆组拾取图元句柄映射。
        transform_handle: 锚点拖拽控制器句柄。
        selected_drag_index: 当前拖拽的锚点索引。
    """

    server: viser.ViserServer
    on_anchor_click: Callable[[int], None] | None = None
    on_rod_group_click: Callable[[int], None] | None = None
    anchor_handles: dict[int, viser.IcosphereHandle] = field(default_factory=dict)
    
    # 三段式渲染句柄
    sleeve_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    rod_l_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    rod_r_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    
    rod_hitbox_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    batch_anchor_handles: dict[tuple[int, int], viser.IcosphereHandle] = field(default_factory=dict)
    batch_sleeve_handles: dict[tuple[int, int], viser.CylinderHandle] = field(default_factory=dict)
    batch_rod_l_handles: dict[tuple[int, int], viser.CylinderHandle] = field(default_factory=dict)
    batch_rod_r_handles: dict[tuple[int, int], viser.CylinderHandle] = field(default_factory=dict)
    transform_handle: viser.TransformControlsHandle | None = None
    selected_drag_index: int | None = None

    def setup_scene(self) -> None:
        """初始化基础场景（地面网格等）。"""
        self.server.scene.add_grid(
            "/world/grid",
            width=30,
            height=30,
            cell_size=1.0,
            section_size=5.0,
            plane="xy",
        )

    def render(self, workspace: Workspace, *, anchor_pos: np.ndarray | None = None) -> None:
        """刷新杆组、锚点和变换控件状态。

        Args:
            workspace: 当前工作区。
        """
        self._clear_batch_handles()
        positions = workspace.topology.anchor_pos if anchor_pos is None else anchor_pos
        self._render_rod_groups(workspace, positions)
        self._render_anchors(workspace, positions)
        self._sync_transform_control(workspace)

    def render_batch(
        self,
        workspace: Workspace,
        *,
        batch_anchor_pos: np.ndarray,
        env_origins: np.ndarray | None = None,
        selected_env: int = 0,
        show_only_selected: bool = False,
        spacing: float = 3.0,
    ) -> None:
        """在同一 Viser 场景中渲染多个同构仿真实例。

        这是面向 ``VectorVGTREnv`` 的过渡版多实例渲染：每个 env 暂时创建独立
        primitive handle，后续可升级为 Viser batched mesh handle。

        Args:
            workspace: 当前工作区。
            batch_anchor_pos: 批量锚点位置，shape ``(num_envs, anchor_count, 3)``。
            env_origins: 可选环境平铺偏移，shape ``(num_envs, 3)``。
            selected_env: 当前选中的环境索引。
            show_only_selected: 是否只显示选中环境。
            spacing: 自动生成平铺偏移时的环境间距。
        """
        self._clear_single_handles()
        positions = _as_batch_anchor_pos(batch_anchor_pos, workspace.topology.anchor_pos.shape[0])
        num_envs = int(positions.shape[0])
        if not 0 <= selected_env < num_envs:
            raise ValueError(f"selected_env must be in [0, {num_envs - 1}], got {selected_env}")
        origins = (
            _grid_env_origins(num_envs, spacing=spacing)
            if env_origins is None
            else _as_env_origins(env_origins, num_envs)
        )
        displayed_positions = positions + origins[:, None, :]
        visible_envs = {selected_env} if show_only_selected else set(range(num_envs))

        self._render_batch_rod_visuals(workspace, displayed_positions, visible_envs)
        self._render_batch_anchors(workspace, displayed_positions, visible_envs)
        if self.transform_handle is not None:
            self.transform_handle.visible = False
        self.selected_drag_index = None

    def _render_rod_groups(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
        """渲染并更新全部杆组。

        Args:
            workspace: 当前工作区。
        """
        topology = workspace.topology
        if topology.rod_anchors.size == 0:
            # 清理所有三段式句柄
            for dic in [self.sleeve_handles, self.rod_l_handles, self.rod_r_handles, self.rod_hitbox_handles]:
                for handle in dic.values():
                    handle.remove()
                dic.clear()
            return

        self._render_rod_visuals(workspace, anchor_pos)
        self._render_rod_hitboxes(workspace, anchor_pos)

    def _render_rod_visuals(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
        topology = workspace.topology
        existing = set(self.sleeve_handles)
        current = set(range(topology.rod_anchors.shape[0]))

        # 移除已废弃的句柄
        for index in existing - current:
            self.sleeve_handles.pop(index).remove()
            self.rod_l_handles.pop(index).remove()
            self.rod_r_handles.pop(index).remove()

        # 更新或创建新句柄
        for index in current:
            # 1. 基础几何计算：套筒是父 frame，左右杆在套筒局部 Z 轴上滑动。
            pos_a = anchor_pos[topology.rod_anchors[index, 0]]
            pos_b = anchor_pos[topology.rod_anchors[index, 1]]
            midpoint = (pos_a + pos_b) / 2.0
            direction = pos_b - pos_a
            length = float(np.linalg.norm(direction))
            wxyz = tuple(float(x) for x in _quaternion_from_z_axis(direction))
            color = tuple(int(x) for x in _rod_color(workspace, index))
            geometry = _rod_visual_geometry(workspace, index, current_length=length)
            
            # 2. 渲染 Sleeve (中段套筒)
            if index not in self.sleeve_handles:
                self.sleeve_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/sleeve",
                    radius=geometry.sleeve_radius,
                    height=geometry.sleeve_height,
                    color=color,
                    position=tuple(float(x) for x in midpoint),
                    wxyz=wxyz,
                )
            else:
                h = self.sleeve_handles[index]
                h.position = tuple(float(x) for x in midpoint)
                h.wxyz = wxyz
                h.height = geometry.sleeve_height
                h.radius = geometry.sleeve_radius
                h.color = color

            # 3. 渲染 Rod Left：作为 sleeve 的子节点，局部 -Z 方向滑动。
            if index not in self.rod_l_handles:
                self.rod_l_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/sleeve/rod_l",
                    radius=geometry.rod_radius,
                    height=geometry.rod_height,
                    color=color,
                    position=geometry.rod_l_position,
                )
            else:
                h = self.rod_l_handles[index]
                h.position = geometry.rod_l_position
                h.wxyz = (1.0, 0.0, 0.0, 0.0)
                h.height = geometry.rod_height
                h.radius = geometry.rod_radius
                h.color = color

            # 4. 渲染 Rod Right：作为 sleeve 的子节点，局部 +Z 方向滑动。
            if index not in self.rod_r_handles:
                self.rod_r_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/sleeve/rod_r",
                    radius=geometry.rod_radius,
                    height=geometry.rod_height,
                    color=color,
                    position=geometry.rod_r_position,
                )
            else:
                h = self.rod_r_handles[index]
                h.position = geometry.rod_r_position
                h.wxyz = (1.0, 0.0, 0.0, 0.0)
                h.height = geometry.rod_height
                h.radius = geometry.rod_radius
                h.color = color

    def _render_batch_rod_visuals(
        self,
        workspace: Workspace,
        batch_anchor_pos: np.ndarray,
        visible_envs: set[int],
    ) -> None:
        topology = workspace.topology
        num_envs = int(batch_anchor_pos.shape[0])
        current = {
            (env_index, rod_index)
            for env_index in range(num_envs)
            for rod_index in range(topology.rod_anchors.shape[0])
        }

        for handles in [self.batch_sleeve_handles, self.batch_rod_l_handles, self.batch_rod_r_handles]:
            for key in set(handles) - current:
                handles.pop(key).remove()

        for env_index in range(num_envs):
            visible = env_index in visible_envs
            for rod_index in range(topology.rod_anchors.shape[0]):
                key = (env_index, rod_index)
                pos_a = batch_anchor_pos[env_index, topology.rod_anchors[rod_index, 0]]
                pos_b = batch_anchor_pos[env_index, topology.rod_anchors[rod_index, 1]]
                midpoint = (pos_a + pos_b) / 2.0
                direction = pos_b - pos_a
                length = float(np.linalg.norm(direction))
                wxyz = tuple(float(x) for x in _quaternion_from_z_axis(direction))
                color = tuple(int(x) for x in _rod_color(workspace, rod_index))
                geometry = _rod_visual_geometry(workspace, rod_index, current_length=length)
                prefix = f"/world/envs/{env_index}/rod_vis/{rod_index}/sleeve"

                if key not in self.batch_sleeve_handles:
                    self.batch_sleeve_handles[key] = self.server.scene.add_cylinder(
                        prefix,
                        radius=geometry.sleeve_radius,
                        height=geometry.sleeve_height,
                        color=color,
                        position=tuple(float(x) for x in midpoint),
                        wxyz=wxyz,
                        visible=visible,
                    )
                else:
                    h = self.batch_sleeve_handles[key]
                    h.position = tuple(float(x) for x in midpoint)
                    h.wxyz = wxyz
                    h.height = geometry.sleeve_height
                    h.radius = geometry.sleeve_radius
                    h.color = color
                    h.visible = visible

                if key not in self.batch_rod_l_handles:
                    self.batch_rod_l_handles[key] = self.server.scene.add_cylinder(
                        f"{prefix}/rod_l",
                        radius=geometry.rod_radius,
                        height=geometry.rod_height,
                        color=color,
                        position=geometry.rod_l_position,
                        visible=visible,
                    )
                else:
                    h = self.batch_rod_l_handles[key]
                    h.position = geometry.rod_l_position
                    h.wxyz = (1.0, 0.0, 0.0, 0.0)
                    h.height = geometry.rod_height
                    h.radius = geometry.rod_radius
                    h.color = color
                    h.visible = visible

                if key not in self.batch_rod_r_handles:
                    self.batch_rod_r_handles[key] = self.server.scene.add_cylinder(
                        f"{prefix}/rod_r",
                        radius=geometry.rod_radius,
                        height=geometry.rod_height,
                        color=color,
                        position=geometry.rod_r_position,
                        visible=visible,
                    )
                else:
                    h = self.batch_rod_r_handles[key]
                    h.position = geometry.rod_r_position
                    h.wxyz = (1.0, 0.0, 0.0, 0.0)
                    h.height = geometry.rod_height
                    h.radius = geometry.rod_radius
                    h.color = color
                    h.visible = visible

    def _render_anchors(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
        """渲染锚点为主体球形。

        Args:
            workspace: 当前工作区。
        """
        topology = workspace.topology
        existing = set(self.anchor_handles)
        current = set(range(topology.anchor_pos.shape[0]))

        for index in existing - current:
            self.anchor_handles.pop(index).remove()

        for index in current:
            position = tuple(float(x) for x in anchor_pos[index])
            color = tuple(int(x) for x in _anchor_color(workspace, index))
            if index not in self.anchor_handles:
                handle = self.server.scene.add_icosphere(
                    f"/world/anchors/v_{index}",
                    radius=ANCHOR_RADIUS,
                    color=color,
                    position=position,
                )
                if self.on_anchor_click is not None:
                    handle.on_click(
                        lambda _event, anchor_index=index: self.on_anchor_click(anchor_index)
                    )
                self.anchor_handles[index] = handle
            else:
                handle = self.anchor_handles[index]
                handle.position = position
                handle.color = color

    def _render_batch_anchors(
        self,
        workspace: Workspace,
        batch_anchor_pos: np.ndarray,
        visible_envs: set[int],
    ) -> None:
        topology = workspace.topology
        num_envs = int(batch_anchor_pos.shape[0])
        current = {
            (env_index, anchor_index)
            for env_index in range(num_envs)
            for anchor_index in range(topology.anchor_pos.shape[0])
        }

        for key in set(self.batch_anchor_handles) - current:
            self.batch_anchor_handles.pop(key).remove()

        for env_index in range(num_envs):
            visible = env_index in visible_envs
            for anchor_index in range(topology.anchor_pos.shape[0]):
                key = (env_index, anchor_index)
                position = tuple(float(x) for x in batch_anchor_pos[env_index, anchor_index])
                color = tuple(int(x) for x in _anchor_color(workspace, anchor_index))
                if key not in self.batch_anchor_handles:
                    self.batch_anchor_handles[key] = self.server.scene.add_icosphere(
                        f"/world/envs/{env_index}/anchors/v_{anchor_index}",
                        radius=ANCHOR_RADIUS,
                        color=color,
                        position=position,
                        visible=visible,
                    )
                else:
                    handle = self.batch_anchor_handles[key]
                    handle.position = position
                    handle.color = color
                    handle.visible = visible

    def _render_rod_hitboxes(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
        """同步杆组拾取代理，提升点击稳定性。

        Args:
            workspace: 当前工作区。
        """
        topology = workspace.topology
        if not workspace.ui.editing or self.on_rod_group_click is None:
            for handle in self.rod_hitbox_handles.values():
                handle.remove()
            self.rod_hitbox_handles.clear()
            return

        existing = set(self.rod_hitbox_handles)
        current = set(range(topology.rod_anchors.shape[0]))

        for index in existing - current:
            self.rod_hitbox_handles.pop(index).remove()

        for index in current:
            start = anchor_pos[topology.rod_anchors[index, 0]]
            end = anchor_pos[topology.rod_anchors[index, 1]]
            midpoint = tuple(float(x) for x in (start + end) / 2.0)
            direction = end - start
            length = float(np.linalg.norm(direction))
            wxyz = tuple(float(x) for x in _quaternion_from_z_axis(direction))
            color = tuple(int(x) for x in _rod_color(workspace, index))

            if index not in self.rod_hitbox_handles:
                handle = self.server.scene.add_cylinder(
                    f"/world/rod_hits/e_{index}",
                    radius=ROD_PICK_RADIUS,
                    height=max(length, 1e-4),
                    color=color,
                    opacity=0.0,
                    cast_shadow=False,
                    receive_shadow=False,
                    position=midpoint,
                    wxyz=wxyz,
                )
                if self.on_rod_group_click is not None:
                    handle.on_click(lambda _event, rod_index=index: self.on_rod_group_click(rod_index))
                self.rod_hitbox_handles[index] = handle
            else:
                handle = self.rod_hitbox_handles[index]
                handle.position = midpoint
                handle.wxyz = wxyz
                handle.height = max(length, 1e-4)
                handle.color = color

    def _sync_transform_control(self, workspace: Workspace) -> None:
        """根据当前编辑状态显示或隐藏锚点变换控件。

        Args:
            workspace: 当前工作区。
        """
        selected = np.flatnonzero(workspace.ui.anchor_status == 2)
        if not workspace.ui.editing or not workspace.ui.moving_anchor or selected.size != 1:
            if self.transform_handle is not None:
                self.transform_handle.visible = False
            self.selected_drag_index = None
            return

        anchor_index = int(selected[0])
        position = tuple(float(x) for x in workspace.topology.anchor_pos[anchor_index])

        if self.transform_handle is None:
            self.transform_handle = self.server.scene.add_transform_controls(
                "/world/transform",
                scale=0.6,
                disable_rotations=True,
            )
        self.transform_handle.visible = True
        self.transform_handle.position = position
        self.selected_drag_index = anchor_index

    def install_transform_callback(
        self,
        callback: Callable[[int, np.ndarray], None],
        *,
        drag_start: Callable[[], None] | None = None,
        drag_end: Callable[[], None] | None = None,
    ) -> None:
        """注册变换控件的更新与拖拽回调。

        Args:
            callback: 变换更新回调，参数为锚点索引和目标位置。
            drag_start: 拖拽开始回调。
            drag_end: 拖拽结束回调。
        """
        if self.transform_handle is None:
            self.transform_handle = self.server.scene.add_transform_controls(
                "/world/transform",
                scale=0.6,
                disable_rotations=True,
                visible=False,
            )

        @self.transform_handle.on_update
        def _on_update(event: viser.TransformControlsEvent) -> None:
            if self.selected_drag_index is None:
                return
            callback(self.selected_drag_index, np.asarray(event.target.position, dtype=np.float64))

        if drag_start is not None:

            @self.transform_handle.on_drag_start
            def _on_drag_start(_event: viser.TransformControlsEvent) -> None:
                drag_start()

        if drag_end is not None:

            @self.transform_handle.on_drag_end
            def _on_drag_end(_event: viser.TransformControlsEvent) -> None:
                drag_end()

    def _clear_single_handles(self) -> None:
        """移除单实例渲染句柄。"""
        for handles in [
            self.anchor_handles,
            self.sleeve_handles,
            self.rod_l_handles,
            self.rod_r_handles,
            self.rod_hitbox_handles,
        ]:
            for handle in handles.values():
                handle.remove()
            handles.clear()

    def _clear_batch_handles(self) -> None:
        """移除批量渲染句柄。"""
        for handles in [
            self.batch_anchor_handles,
            self.batch_sleeve_handles,
            self.batch_rod_l_handles,
            self.batch_rod_r_handles,
        ]:
            for handle in handles.values():
                handle.remove()
            handles.clear()


def _anchor_color(workspace: Workspace, index: int) -> np.ndarray:
    """按锚点状态返回显示颜色。

    Args:
        workspace: 当前工作区。
        index: 锚点索引。

    Returns:
        RGB 颜色数组。
    """
    status = int(workspace.ui.anchor_status[index])
    if status == 2:
        return RED
    if status == 1:
        return ORANGE
    if workspace.topology.anchor_fixed[index]:
        return CYAN
    return WHITE


def _as_batch_anchor_pos(batch_anchor_pos: np.ndarray, anchor_count: int) -> np.ndarray:
    """校验并返回批量锚点位置数组。"""
    positions = np.asarray(batch_anchor_pos, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (anchor_count, 3):
        raise ValueError(
            "batch_anchor_pos must have shape "
            f"(num_envs, {anchor_count}, 3), got {positions.shape}"
        )
    if positions.shape[0] <= 0:
        raise ValueError("batch_anchor_pos must contain at least one environment")
    return positions


def _as_env_origins(env_origins: np.ndarray, num_envs: int) -> np.ndarray:
    """校验并返回环境平铺偏移。"""
    origins = np.asarray(env_origins, dtype=np.float64)
    if origins.shape != (num_envs, 3):
        raise ValueError(f"env_origins must have shape ({num_envs}, 3), got {origins.shape}")
    return origins


def _grid_env_origins(num_envs: int, *, spacing: float = 3.0) -> np.ndarray:
    """按二维网格生成环境平铺偏移。"""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    cols = int(np.ceil(np.sqrt(num_envs)))
    origins = np.zeros((num_envs, 3), dtype=np.float64)
    for env_index in range(num_envs):
        row, col = divmod(env_index, cols)
        origins[env_index, 0] = float(col) * spacing
        origins[env_index, 1] = float(row) * spacing
    return origins


def _rod_color(workspace: Workspace, index: int) -> np.ndarray:
    """按杆组状态和控制组配置返回显示颜色。

    Args:
        workspace: 当前工作区。
        index: 杆组索引。

    Returns:
        RGB 颜色数组。
    """
    status = int(workspace.ui.rod_group_status[index])
    if status == 2:
        return RED
    if status == 1:
        return ORANGE
    if workspace.ui.show_control_group:
        colors = workspace.script.control_group_colors
        control_group = int(workspace.topology.rod_control_group[index])
        if colors.shape[0] > control_group:
            return colors[control_group]
        return CHANNEL_COLORS[control_group % len(CHANNEL_COLORS)]
    return BLACK if workspace.topology.rod_enabled[index] else WHITE


def _rod_visual_geometry(
    workspace: Workspace,
    index: int,
    *,
    current_length: float,
) -> RodVisualGeometry:
    """计算单杆组的套筒局部活塞几何。

    套筒始终保持在两端锚点中间。最小执行器长度 ``L_min`` 为视觉收拢基准：
    当杆长为 ``L_min`` 时，每根活塞杆从锚点延伸至套筒中心。
    若锚点间距相对 ``L_min`` 增加 ``dx``，则每根活塞杆中心相对套筒平移 ``dx / 2``。

    Args:
        workspace: 当前工作区。
        index: 杆组索引。
        current_length: 当前杆长。

    Returns:
        该杆组的局部渲染几何参数。
    """
    topology = workspace.topology
    robot = workspace.robot_config

    base_length = float(topology.rod_min_length[index])
    if topology.rod_length_limits.shape[0] > index:
        base_length = float(topology.rod_length_limits[index, 0])
    if base_length <= 1e-9:
        base_length = max(float(topology.rod_rest_length[index]), float(current_length), 1e-9)

    rod_radius = (
        float(topology.rod_radius[index])
        if topology.rod_radius.shape[0] > index
        else float(robot.rod_group.rod_radius)
    )
    sleeve_radius = float(robot.rod_group.sleeve_radius)
    sleeve_ratio = float(robot.rod_group.sleeve_display_half_length_ratio)
    if topology.rod_sleeve_half.shape[0] > index:
        sleeve_radius = float(topology.rod_sleeve_half[index, 0])
        sleeve_ratio = float(topology.rod_sleeve_half[index, 2])

    sleeve_ratio = float(np.clip(sleeve_ratio, 0.0, 0.5))
    sleeve_height = max(base_length * sleeve_ratio * 2.0, 1e-4)
    rod_height = max(base_length * 0.5, 1e-4)

    slide = (float(current_length) - base_length) * 0.5
    center_offset = base_length * 0.25 + slide

    return RodVisualGeometry(
        sleeve_radius=max(sleeve_radius, 1e-6),
        sleeve_height=sleeve_height,
        rod_radius=max(rod_radius, 1e-6),
        rod_height=rod_height,
        rod_l_position=(0.0, 0.0, -center_offset),
        rod_r_position=(0.0, 0.0, center_offset),
    )


def _quaternion_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """返回将局部 +Z 轴旋转到目标方向的四元数，格式为 wxyz。"""
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    target = direction / norm
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot >= 1.0 - 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot <= -1.0 + 1e-8:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    axis = np.cross(source, target)
    axis /= np.linalg.norm(axis)
    half_angle = np.arccos(dot) / 2.0
    sin_half_angle = np.sin(half_angle)
    return np.asarray(
        [
            np.cos(half_angle),
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle,
        ],
        dtype=np.float64,
    )
