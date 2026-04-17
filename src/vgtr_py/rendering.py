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

    def render(self, workspace: Workspace) -> None:
        """刷新杆组、锚点和变换控件状态。

        Args:
            workspace: 当前工作区。
        """
        self._render_rod_groups(workspace)
        self._render_anchors(workspace)
        self._sync_transform_control(workspace)

    def _render_rod_groups(self, workspace: Workspace) -> None:
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

        self._render_rod_visuals(workspace)
        self._render_rod_hitboxes(workspace)

    def _render_rod_visuals(self, workspace: Workspace) -> None:
        topology = workspace.topology
        robot = workspace.robot_config
        existing = set(self.sleeve_handles)
        current = set(range(topology.rod_anchors.shape[0]))

        # 移除已废弃的句柄
        for index in existing - current:
            self.sleeve_handles.pop(index).remove()
            self.rod_l_handles.pop(index).remove()
            self.rod_r_handles.pop(index).remove()

        # 更新或创建新句柄
        for index in current:
            # 1. 基础几何计算
            pos_a = topology.anchor_pos[topology.rod_anchors[index, 0]]
            pos_b = topology.anchor_pos[topology.rod_anchors[index, 1]]
            midpoint = (pos_a + pos_b) / 2.0
            direction = pos_b - pos_a
            length = float(np.linalg.norm(direction))
            wxyz = tuple(float(x) for x in _quaternion_from_z_axis(direction))
            color = tuple(int(x) for x in _rod_color(workspace, index))
            
            # 套筒长度：取初始/当前长度的 30% 或配置值
            sleeve_ratio = robot.rod_group.sleeve_display_half_length_ratio
            sleeve_len = length * sleeve_ratio * 2.0
            
            # 2. 渲染 Sleeve (中段套筒)
            if index not in self.sleeve_handles:
                self.sleeve_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/sleeve",
                    radius=ROD_RADIUS * 1.8, # 套筒略粗
                    height=max(sleeve_len, 1e-4),
                    color=color,
                    position=tuple(float(x) for x in midpoint),
                    wxyz=wxyz,
                )
            else:
                h = self.sleeve_handles[index]
                h.position = tuple(float(x) for x in midpoint)
                h.wxyz = wxyz
                h.height = max(sleeve_len, 1e-4)
                h.color = color

            # 3. 渲染 Rod Left (左杆: A -> Mid)
            len_l = length / 2.0
            pos_l = (pos_a + midpoint) / 2.0
            if index not in self.rod_l_handles:
                self.rod_l_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/rod_l",
                    radius=ROD_RADIUS,
                    height=max(len_l, 1e-4),
                    color=color,
                    position=tuple(float(x) for x in pos_l),
                    wxyz=wxyz,
                )
            else:
                h = self.rod_l_handles[index]
                h.position = tuple(float(x) for x in pos_l)
                h.wxyz = wxyz
                h.height = max(len_l, 1e-4)
                h.color = color

            # 4. 渲染 Rod Right (右杆: B -> Mid)
            len_r = length / 2.0
            pos_r = (pos_b + midpoint) / 2.0
            if index not in self.rod_r_handles:
                self.rod_r_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/rod_r",
                    radius=ROD_RADIUS,
                    height=max(len_r, 1e-4),
                    color=color,
                    position=tuple(float(x) for x in pos_r),
                    wxyz=wxyz,
                )
            else:
                h = self.rod_r_handles[index]
                h.position = tuple(float(x) for x in pos_r)
                h.wxyz = wxyz
                h.height = max(len_r, 1e-4)
                h.color = color

    def _render_anchors(self, workspace: Workspace) -> None:
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
            position = tuple(float(x) for x in topology.anchor_pos[index])
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

    def _render_rod_hitboxes(self, workspace: Workspace) -> None:
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
            start = topology.anchor_pos[topology.rod_anchors[index, 0]]
            end = topology.anchor_pos[topology.rod_anchors[index, 1]]
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
