"""三维图形渲染封装。

负责统管从内部物理拓扑至 Viser 前端三维可视化的转换映射。渲染循环、图元绘制与主题视觉表达皆于此类维护。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import viser

from .workspace import Workspace

VERTEX_RADIUS = 0.06
EDGE_RADIUS = 0.01
EDGE_PICK_RADIUS = 0.025
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
        on_vertex_click: 顶点点击回调。
        on_edge_click: 边点击回调。
        vertex_handles: 顶点图元句柄映射。
        edge_visual_handles: 可见杆组图元句柄映射。
        edge_handles: 边拾取图元句柄映射。
        transform_handle: 顶点拖拽控制器句柄。
        selected_drag_index: 当前拖拽的顶点索引。
    """

    server: viser.ViserServer
    on_vertex_click: Callable[[int], None] | None = None
    on_edge_click: Callable[[int], None] | None = None
    vertex_handles: dict[int, viser.IcosphereHandle] = field(default_factory=dict)
    edge_visual_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    edge_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
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
        """刷新边、顶点和变换控件状态。

        Args:
            workspace: 当前工作区。
        """
        self._render_edges(workspace)
        self._render_vertices(workspace)
        self._sync_transform_control(workspace)

    def _render_edges(self, workspace: Workspace) -> None:
        """渲染并更新全部边线。

        Args:
            workspace: 当前工作区。
        """
        topology = workspace.topology
        if topology.edges.size == 0:
            for handle in self.edge_visual_handles.values():
                handle.remove()
            self.edge_visual_handles.clear()
            for handle in self.edge_handles.values():
                handle.remove()
            self.edge_handles.clear()
            return

        self._render_edge_visuals(workspace)
        self._render_edge_hitboxes(workspace)

    def _render_edge_visuals(self, workspace: Workspace) -> None:
        topology = workspace.topology
        existing = set(self.edge_visual_handles)
        current = set(range(topology.edges.shape[0]))

        for index in existing - current:
            self.edge_visual_handles.pop(index).remove()

        for index in current:
            start = topology.vertices[topology.edges[index, 0]]
            end = topology.vertices[topology.edges[index, 1]]
            midpoint = tuple(float(x) for x in (start + end) / 2.0)
            direction = end - start
            length = float(np.linalg.norm(direction))
            wxyz = tuple(float(x) for x in _quaternion_from_z_axis(direction))
            color = tuple(int(x) for x in _edge_color(workspace, index))

            if index not in self.edge_visual_handles:
                handle = self.server.scene.add_cylinder(
                    f"/world/edge_visuals/e_{index}",
                    radius=EDGE_RADIUS,
                    height=max(length, 1e-4),
                    color=color,
                    cast_shadow=False,
                    receive_shadow=False,
                    position=midpoint,
                    wxyz=wxyz,
                )
                self.edge_visual_handles[index] = handle
            else:
                handle = self.edge_visual_handles[index]
                handle.position = midpoint
                handle.wxyz = wxyz
                handle.height = max(length, 1e-4)
                handle.color = color

    def _render_vertices(self, workspace: Workspace) -> None:
        """渲染顶点为主体球形。

        增添新的点，修改现有节点位置，并清空不再存活的节点。为每一个存在的点附加点击事件处理逻辑。

        Args:
            workspace: 当前工作区。
        """
        topology = workspace.topology
        existing = set(self.vertex_handles)
        current = set(range(topology.vertices.shape[0]))

        for index in existing - current:
            self.vertex_handles.pop(index).remove()

        for index in current:
            position = tuple(float(x) for x in topology.vertices[index])
            color = tuple(int(x) for x in _vertex_color(workspace, index))
            if index not in self.vertex_handles:
                handle = self.server.scene.add_icosphere(
                    f"/world/vertices/v_{index}",
                    radius=VERTEX_RADIUS,
                    color=color,
                    position=position,
                )
                if self.on_vertex_click is not None:
                    handle.on_click(
                        lambda _event, vertex_index=index: self.on_vertex_click(vertex_index)
                    )
                self.vertex_handles[index] = handle
            else:
                handle = self.vertex_handles[index]
                handle.position = position
                handle.color = color

    def _render_edge_hitboxes(self, workspace: Workspace) -> None:
        """同步边拾取代理，提升边点击稳定性。

        Args:
            workspace: 当前工作区。
        """
        topology = workspace.topology
        if not workspace.ui.editing or self.on_edge_click is None:
            for handle in self.edge_handles.values():
                handle.remove()
            self.edge_handles.clear()
            return

        existing = set(self.edge_handles)
        current = set(range(topology.edges.shape[0]))

        for index in existing - current:
            self.edge_handles.pop(index).remove()

        for index in current:
            start = topology.vertices[topology.edges[index, 0]]
            end = topology.vertices[topology.edges[index, 1]]
            midpoint = tuple(float(x) for x in (start + end) / 2.0)
            direction = end - start
            length = float(np.linalg.norm(direction))
            wxyz = tuple(float(x) for x in _quaternion_from_z_axis(direction))
            color = tuple(int(x) for x in _edge_color(workspace, index))

            if index not in self.edge_handles:
                handle = self.server.scene.add_cylinder(
                    f"/world/edge_hits/e_{index}",
                    radius=EDGE_PICK_RADIUS,
                    height=max(length, 1e-4),
                    color=color,
                    opacity=0.0,
                    cast_shadow=False,
                    receive_shadow=False,
                    position=midpoint,
                    wxyz=wxyz,
                )
                if self.on_edge_click is not None:
                    handle.on_click(lambda _event, edge_index=index: self.on_edge_click(edge_index))
                self.edge_handles[index] = handle
            else:
                handle = self.edge_handles[index]
                handle.position = midpoint
                handle.wxyz = wxyz
                handle.height = max(length, 1e-4)
                handle.color = color

    def _sync_transform_control(self, workspace: Workspace) -> None:
        """根据当前编辑状态显示或隐藏顶点变换控件。

        Args:
            workspace: 当前工作区。
        """
        selected = np.flatnonzero(workspace.ui.vertex_status == 2)
        if not workspace.ui.editing or not workspace.ui.moving_joint or selected.size != 1:
            if self.transform_handle is not None:
                self.transform_handle.visible = False
            self.selected_drag_index = None
            return

        vertex_index = int(selected[0])
        position = tuple(float(x) for x in workspace.topology.vertices[vertex_index])

        if self.transform_handle is None:
            self.transform_handle = self.server.scene.add_transform_controls(
                "/world/transform",
                scale=0.6,
                disable_rotations=True,
            )
        self.transform_handle.visible = True
        self.transform_handle.position = position
        self.selected_drag_index = vertex_index

    def install_transform_callback(
        self,
        callback: Callable[[int, np.ndarray], None],
        *,
        drag_start: Callable[[], None] | None = None,
        drag_end: Callable[[], None] | None = None,
    ) -> None:
        """注册变换控件的更新与拖拽回调。

        Args:
            callback: 变换更新回调，参数为顶点索引和目标位置。
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


def _vertex_color(workspace: Workspace, index: int) -> np.ndarray:
    """按顶点状态返回显示颜色。

    Args:
        workspace: 当前工作区。
        index: 顶点索引。

    Returns:
        RGB 颜色数组。
    """
    status = int(workspace.ui.vertex_status[index])
    if status == 2:
        return RED
    if status == 1:
        return ORANGE
    if workspace.topology.fixed_vs[index]:
        return CYAN
    return WHITE


def _edge_color(workspace: Workspace, index: int) -> np.ndarray:
    """按边状态和控制组配置返回显示颜色。

    Args:
        workspace: 当前工作区。
        index: 边索引。

    Returns:
        RGB 颜色数组。
    """
    status = int(workspace.ui.edge_status[index])
    if status == 2:
        return RED
    if status == 1:
        return ORANGE
    if workspace.ui.show_channel:
        colors = workspace.script.control_group_colors
        control_group = int(workspace.topology.edge_channel[index])
        if colors.shape[0] > control_group:
            return colors[control_group]
        return CHANNEL_COLORS[control_group % len(CHANNEL_COLORS)]
    return BLACK if workspace.topology.edge_active[index] else WHITE


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
