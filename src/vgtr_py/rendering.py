"""三维图形渲染封装。

负责统管从内部物理拓扑至 Viser 前端三维可视化的转换映射，包括单实例渲染与
基于 batched mesh 的批量环境渲染。
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

# ---------------------------------------------------------------------------
# Mesh 生成 — 全局缓存，只算一次
# ---------------------------------------------------------------------------

_CACHED_SPHERE: tuple[np.ndarray, np.ndarray] | None = None
_CACHED_CYLINDER: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _unit_icosphere_mesh(subdiv: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """生成单位半径二十面体球 mesh。

    Returns:
        ``(vertices, faces)``，vertices 形状 ``(V, 3)``，faces 形状 ``(F, 3)``
        均为 float64/int32。
    """
    global _CACHED_SPHERE
    if _CACHED_SPHERE is not None:
        return _CACHED_SPHERE

    # 初始正二十面体（边长 2 的内接球）
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1.0,  phi,  0.0],
        [ 1.0,  phi,  0.0],
        [-1.0, -phi,  0.0],
        [ 1.0, -phi,  0.0],
        [ 0.0, -1.0,  phi],
        [ 0.0,  1.0,  phi],
        [ 0.0, -1.0, -phi],
        [ 0.0,  1.0, -phi],
        [ phi,  0.0, -1.0],
        [ phi,  0.0,  1.0],
        [-phi,  0.0, -1.0],
        [-phi,  0.0,  1.0],
    ], dtype=np.float64)
    # 归一化到单位球
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    # Loop subdivision
    edge_map: dict[tuple[int, int], int] = {}
    for _ in range(subdiv):
        new_faces = []
        for tri in faces:
            mid = []
            for k in range(3):
                i0, i1 = int(tri[k]), int(tri[(k + 1) % 3])
                key = (min(i0, i1), max(i0, i1))
                if key not in edge_map:
                    mp = (verts[i0] + verts[i1]) / 2.0
                    mp /= np.linalg.norm(mp)
                    edge_map[key] = verts.shape[0]
                    verts = np.vstack([verts, mp])
                mid.append(edge_map[key])
            new_faces.extend([
                [tri[0], mid[0], mid[2]],
                [tri[1], mid[1], mid[0]],
                [tri[2], mid[2], mid[1]],
                [mid[0], mid[1], mid[2]],
            ])
        faces = np.array(new_faces, dtype=np.int32)
        edge_map.clear()

    _CACHED_SPHERE = (verts, faces)
    return _CACHED_SPHERE


def _unit_cylinder_mesh(segments: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """生成单位半径、单位高度的封闭圆柱 mesh。

    Cylinder 沿 Z 轴，中心在原点，Z 范围 [-0.5, 0.5]。

    Returns:
        ``(vertices, faces)``，vertices 形状 ``(V, 3)``，faces 形状 ``(F, 3)``。
    """
    global _CACHED_CYLINDER
    if segments in _CACHED_CYLINDER:
        return _CACHED_CYLINDER[segments]

    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False, dtype=np.float64)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    bottom = np.column_stack([cos_a, sin_a, np.full(segments, -0.5, dtype=np.float64)])
    top = np.column_stack([cos_a, sin_a, np.full(segments, 0.5, dtype=np.float64)])
    centers = np.asarray([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]], dtype=np.float64)
    verts = np.vstack([bottom, top, centers])  # 0..S-1: bottom, S..2S-1: top
    bottom_center = 2 * segments
    top_center = bottom_center + 1

    faces_list = []
    for i in range(segments):
        j = (i + 1) % segments
        b_i, b_j = i, j
        t_i, t_j = i + segments, j + segments
        faces_list.append([b_i, b_j, t_j])
        faces_list.append([b_i, t_j, t_i])
        faces_list.append([bottom_center, b_j, b_i])
        faces_list.append([top_center, t_i, t_j])

    _CACHED_CYLINDER[segments] = (verts, np.array(faces_list, dtype=np.int32))
    return _CACHED_CYLINDER[segments]


# ---------------------------------------------------------------------------
# 四元数工具（向量化）
# ---------------------------------------------------------------------------

def _batch_quaternion_from_z_axis(directions: np.ndarray) -> np.ndarray:
    """将一组 Z 轴方向向量转为四元数 (wxyz)，shape ``(N, 3) → (N, 4)``。"""
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    target = np.where(norms > 1e-12, directions / norms, [0.0, 0.0, 1.0])
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dots = np.clip(np.sum(target * source[None, :], axis=1), -1.0, 1.0)

    # 接近 +Z → 恒等四元数
    near_pos = dots >= 1.0 - 1e-8
    # 接近 -Z → 绕 X 轴 180°
    near_neg = dots <= -1.0 + 1e-8

    axes = np.cross(source[None, :], target)
    axis_norms = np.linalg.norm(axes, axis=1)
    valid_axis = axis_norms > 1e-12
    safe_norms = np.where(valid_axis, axis_norms, 1.0)
    axes = axes / safe_norms[:, None]
    axes[~valid_axis] = 0.0

    half_angles = np.arccos(dots) / 2.0
    sin_half = np.sin(half_angles)
    cos_half = np.cos(half_angles)

    result = np.zeros((directions.shape[0], 4), dtype=np.float64)
    result[:, 0] = cos_half
    result[:, 1:] = axes * sin_half[:, None]

    result[near_pos] = [1.0, 0.0, 0.0, 0.0]
    result[near_neg] = [0.0, 1.0, 0.0, 0.0]
    return result


def _batch_rotate_vectors(quats: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """通过四元数 ``quats (N, 4) wxyz`` 旋转 ``vectors (N, 3)``。

    公式: v' = v + 2*w*(q × v) + 2*(q × (q × v))
    """
    w = quats[:, 0:1]
    q = quats[:, 1:4]
    cross1 = np.cross(q, vectors)
    cross2 = np.cross(q, cross1)
    return vectors + 2.0 * w * cross1 + 2.0 * cross2


# ---------------------------------------------------------------------------
# SceneRenderer
# ---------------------------------------------------------------------------

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
class RodTransformState:
    """单根杆件的渲染 transform 快照。"""

    midpoint: np.ndarray
    wxyz: np.ndarray
    length: float
    geometry: RodVisualGeometry
    color: np.ndarray
    rod_l_position: np.ndarray
    rod_r_position: np.ndarray


@dataclass
class BatchRenderState:
    """批量渲染的纯数据快照，backend 只消费该状态。"""

    batch_qpos: np.ndarray
    displayed_qpos: np.ndarray
    origins: np.ndarray
    selected_env: int
    show_only_selected: bool
    track_selected: bool
    anchor_positions: np.ndarray
    anchor_wxyzs: np.ndarray
    anchor_colors: np.ndarray
    anchor_opacities: np.ndarray | None
    rod_transforms: dict[int, RodTransformState]

    @property
    def num_envs(self) -> int:
        return int(self.batch_qpos.shape[0])

    @property
    def anchor_count(self) -> int:
        return int(self.batch_qpos.shape[1])

    @property
    def rod_count(self) -> int:
        return len(self.rod_transforms)


@dataclass
class _BatchedRodHandles:
    """单个 rod 的 batched mesh 三件套。"""

    sleeve: viser.BatchedMeshHandle
    rod_l: viser.BatchedMeshHandle
    rod_r: viser.BatchedMeshHandle


@dataclass
class BatchHandleRegistry:
    """管理 batched mesh handles 的生命周期。"""

    anchor_mesh: viser.BatchedMeshHandle | None = None
    rod_handles: dict[int, _BatchedRodHandles] = field(default_factory=dict)
    num_envs: int = 0
    anchor_count: int = 0
    rod_count: int = 0

    def matches(self, num_envs: int, anchor_count: int, rod_count: int) -> bool:
        return (
            self.anchor_mesh is not None
            and self.num_envs == num_envs
            and self.anchor_count == anchor_count
            and self.rod_count == rod_count
        )

    def clear(self) -> None:
        """移除所有 batched mesh handles。"""
        if self.anchor_mesh is not None:
            self.anchor_mesh.remove()
            self.anchor_mesh = None
        for handles in self.rod_handles.values():
            handles.sleeve.remove()
            handles.rod_l.remove()
            handles.rod_r.remove()
        self.rod_handles.clear()
        self.num_envs = 0
        self.anchor_count = 0
        self.rod_count = 0

    def ensure(self, server: viser.ViserServer, state: BatchRenderState) -> None:
        """按需创建或复用 batched mesh handles。"""
        if self.matches(state.num_envs, state.anchor_count, state.rod_count):
            return

        self.clear()
        self.num_envs = state.num_envs
        self.anchor_count = state.anchor_count
        self.rod_count = state.rod_count

        sphere_v, sphere_f = _unit_icosphere_mesh()
        cyl_v, cyl_f = _unit_cylinder_mesh()
        anchor_instances = state.num_envs * state.anchor_count

        self.anchor_mesh = server.scene.add_batched_meshes_simple(
            "/world/batch/anchors",
            vertices=sphere_v,
            faces=sphere_f,
            batched_positions=np.zeros((anchor_instances, 3), dtype=np.float64),
            batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (anchor_instances, 1)),
            batched_colors=np.tile(WHITE.astype(np.uint8), (anchor_instances, 1)),
            batched_scales=np.full(anchor_instances, ANCHOR_RADIUS, dtype=np.float64),
            batched_opacities=None,
            material="standard",
        )

        for rod_index, transform in state.rod_transforms.items():
            color = np.asarray(transform.color, dtype=np.uint8)
            geometry = transform.geometry
            sleeve_scales = np.tile(
                [geometry.sleeve_radius, geometry.sleeve_radius, geometry.sleeve_height],
                (state.num_envs, 1),
            )
            rod_scales = np.tile(
                [geometry.rod_radius, geometry.rod_radius, geometry.rod_height],
                (state.num_envs, 1),
            )
            sleeve = server.scene.add_batched_meshes_simple(
                f"/world/batch/rod_vis/{rod_index}/sleeve",
                vertices=cyl_v,
                faces=cyl_f,
                batched_positions=np.zeros((state.num_envs, 3), dtype=np.float64),
                batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (state.num_envs, 1)),
                batched_colors=np.tile(color, (state.num_envs, 1)),
                batched_scales=sleeve_scales,
                batched_opacities=None,
                material="standard",
            )
            rod_l = server.scene.add_batched_meshes_simple(
                f"/world/batch/rod_vis/{rod_index}/rod_l",
                vertices=cyl_v,
                faces=cyl_f,
                batched_positions=np.zeros((state.num_envs, 3), dtype=np.float64),
                batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (state.num_envs, 1)),
                batched_colors=np.tile(color, (state.num_envs, 1)),
                batched_scales=rod_scales,
                batched_opacities=None,
                material="standard",
            )
            rod_r = server.scene.add_batched_meshes_simple(
                f"/world/batch/rod_vis/{rod_index}/rod_r",
                vertices=cyl_v,
                faces=cyl_f,
                batched_positions=np.zeros((state.num_envs, 3), dtype=np.float64),
                batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (state.num_envs, 1)),
                batched_colors=np.tile(color, (state.num_envs, 1)),
                batched_scales=rod_scales,
                batched_opacities=None,
                material="standard",
            )
            self.rod_handles[rod_index] = _BatchedRodHandles(
                sleeve=sleeve,
                rod_l=rod_l,
                rod_r=rod_r,
            )

    def update(self, server: viser.ViserServer, state: BatchRenderState) -> None:
        """把 BatchRenderState 写入 batched mesh handles。"""
        if self.anchor_mesh is None:
            raise RuntimeError("batched handles must be ensured before update")
        rod_opacities = _env_opacity_for_rods(state)

        with server.atomic():
            self.anchor_mesh.batched_positions = state.anchor_positions
            self.anchor_mesh.batched_wxyzs = state.anchor_wxyzs
            self.anchor_mesh.batched_colors = state.anchor_colors
            self.anchor_mesh.batched_opacities = state.anchor_opacities

            for rod_index, transform in state.rod_transforms.items():
                handles = self.rod_handles[rod_index]
                handles.sleeve.batched_positions = transform.midpoint
                handles.sleeve.batched_wxyzs = transform.wxyz
                handles.sleeve.batched_opacities = rod_opacities

                handles.rod_l.batched_positions = transform.rod_l_position
                handles.rod_l.batched_wxyzs = transform.wxyz
                handles.rod_l.batched_opacities = rod_opacities

                handles.rod_r.batched_positions = transform.rod_r_position
                handles.rod_r.batched_wxyzs = transform.wxyz
                handles.rod_r.batched_opacities = rod_opacities

            server.flush()



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

    # 单实例三段式渲染句柄
    sleeve_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    rod_l_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)
    rod_r_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)

    rod_hitbox_handles: dict[int, viser.CylinderHandle] = field(default_factory=dict)

    # Batched mesh 渲染
    batch_registry: BatchHandleRegistry = field(default_factory=BatchHandleRegistry)

    transform_handle: viser.TransformControlsHandle | None = None
    selected_drag_index: int | None = None

    # ------------------------------------------------------------------
    # 场景初始化
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 单实例渲染
    # ------------------------------------------------------------------

    def render(self, workspace: Workspace, *, anchor_pos: np.ndarray | None = None) -> None:
        """刷新杆组、锚点和变换控件状态。"""
        self._clear_batch_handles()
        positions = workspace.topology.anchor_pos if anchor_pos is None else anchor_pos
        self._render_rod_groups(workspace, positions)
        self._render_anchors(workspace, positions)
        self._sync_transform_control(workspace)

    # ------------------------------------------------------------------
    # Batched mesh 渲染
    # ------------------------------------------------------------------

    def render_batch(
        self,
        workspace: Workspace,
        *,
        batch_qpos: np.ndarray,
        selected_env: int = 0,
        show_only_selected: bool = False,
        spacing: float = 3.0,
        track_selected: bool = True,
    ) -> None:
        """使用 batched mesh 渲染多个同构仿真实例。

        Args:
            workspace: 当前工作区。
            batch_qpos: 批量锚点位置，shape ``(num_envs, anchor_count, 3)``。
            selected_env: 当前选中的环境索引。
            show_only_selected: 是否只显示选中环境。
            spacing: 环境平铺间距。
            track_selected: 是否将选中环境居中于视野。
        """
        self._clear_single_handles()
        state = _compute_batch_render_state(
            workspace,
            batch_qpos=batch_qpos,
            selected_env=selected_env,
            show_only_selected=show_only_selected,
            spacing=spacing,
            track_selected=track_selected,
        )
        self.batch_registry.ensure(self.server, state)
        self.batch_registry.update(self.server, state)
        if self.transform_handle is not None:
            self.transform_handle.visible = False
        self.selected_drag_index = None

    # ------------------------------------------------------------------
    # 单实例 — 杆组渲染
    # ------------------------------------------------------------------

    def _render_rod_groups(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
        topology = workspace.topology
        if topology.rod_anchors.size == 0:
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

        for index in existing - current:
            self.sleeve_handles.pop(index).remove()
            self.rod_l_handles.pop(index).remove()
            self.rod_r_handles.pop(index).remove()

        for index in current:
            transform = _compute_single_rod_transform(workspace, anchor_pos, index)
            midpoint = tuple(float(x) for x in transform.midpoint)
            wxyz = tuple(float(x) for x in transform.wxyz)
            color = tuple(int(x) for x in transform.color)
            geometry = transform.geometry

            if index not in self.sleeve_handles:
                self.sleeve_handles[index] = self.server.scene.add_cylinder(
                    f"/world/rod_vis/{index}/sleeve",
                    radius=geometry.sleeve_radius,
                    height=geometry.sleeve_height,
                    color=color,
                    position=midpoint,
                    wxyz=wxyz,
                )
            else:
                h = self.sleeve_handles[index]
                h.position = midpoint
                h.wxyz = wxyz
                h.height = geometry.sleeve_height
                h.radius = geometry.sleeve_radius
                h.color = color

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

    def _render_anchors(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
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

    def _render_rod_hitboxes(self, workspace: Workspace, anchor_pos: np.ndarray) -> None:
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
        """移除所有 batched mesh handles。"""
        self.batch_registry.clear()


# ---------------------------------------------------------------------------
# 颜色工具
# ---------------------------------------------------------------------------

def _anchor_color(workspace: Workspace, index: int) -> np.ndarray:
    status = int(workspace.ui.anchor_status[index])
    if status == 2:
        return RED
    if status == 1:
        return ORANGE
    if workspace.topology.anchor_fixed[index]:
        return CYAN
    return WHITE


def _rod_color(workspace: Workspace, index: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# 批量数据校验 / 布局
# ---------------------------------------------------------------------------

def _as_batch_anchor_pos(batch_anchor_pos: np.ndarray, anchor_count: int) -> np.ndarray:
    positions = np.asarray(batch_anchor_pos, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (anchor_count, 3):
        raise ValueError(
            "batch_anchor_pos must have shape "
            f"(num_envs, {anchor_count}, 3), got {positions.shape}"
        )
    if positions.shape[0] <= 0:
        raise ValueError("batch_anchor_pos must contain at least one environment")
    return positions


def _grid_env_origins(num_envs: int, *, spacing: float = 3.0) -> np.ndarray:
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    cols = int(np.ceil(np.sqrt(num_envs)))
    origins = np.zeros((num_envs, 3), dtype=np.float64)
    for env_index in range(num_envs):
        row, col = divmod(env_index, cols)
        origins[env_index, 0] = float(col) * spacing
        origins[env_index, 1] = float(row) * spacing
    return origins


def _env_opacity_for_rods(state: BatchRenderState) -> np.ndarray | None:
    """返回每个 env 的 rod opacity。"""
    if not state.show_only_selected or state.num_envs <= 1:
        return None
    opacities = np.zeros(state.num_envs, dtype=np.float64)
    opacities[state.selected_env] = 1.0
    return opacities


# ---------------------------------------------------------------------------
# 几何计算（单实例 + batch 共用）
# ---------------------------------------------------------------------------

def _compute_single_rod_transform(
    workspace: Workspace,
    anchor_pos: np.ndarray,
    rod_index: int,
) -> RodTransformState:
    """计算单根杆在给定位形下的公共 transform。"""
    topology = workspace.topology
    pos_a = anchor_pos[topology.rod_anchors[rod_index, 0]]
    pos_b = anchor_pos[topology.rod_anchors[rod_index, 1]]
    midpoint = (pos_a + pos_b) / 2.0
    direction = pos_b - pos_a
    length = float(np.linalg.norm(direction))
    wxyz = _quaternion_from_z_axis(direction)
    geometry = _rod_visual_geometry(workspace, rod_index, current_length=length)
    color = _rod_color(workspace, rod_index)
    return RodTransformState(
        midpoint=np.asarray(midpoint, dtype=np.float64),
        wxyz=np.asarray(wxyz, dtype=np.float64),
        length=length,
        geometry=geometry,
        color=np.asarray(color, dtype=np.uint8),
        rod_l_position=midpoint
        + _batch_rotate_vectors(
            wxyz.reshape(1, 4),
            np.asarray(geometry.rod_l_position, dtype=np.float64).reshape(1, 3),
        )[0],
        rod_r_position=midpoint
        + _batch_rotate_vectors(
            wxyz.reshape(1, 4),
            np.asarray(geometry.rod_r_position, dtype=np.float64).reshape(1, 3),
        )[0],
    )


def _compute_batch_render_state(
    workspace: Workspace,
    *,
    batch_qpos: np.ndarray,
    selected_env: int,
    show_only_selected: bool,
    spacing: float,
    track_selected: bool,
) -> BatchRenderState:
    """计算批量渲染状态，backend 不再包含 transform 逻辑。"""
    qpos = _as_batch_anchor_pos(batch_qpos, workspace.topology.anchor_pos.shape[0])
    num_envs = int(qpos.shape[0])
    anchor_count = int(qpos.shape[1])
    if not 0 <= selected_env < num_envs:
        raise ValueError(f"selected_env must be in [0, {num_envs - 1}], got {selected_env}")

    origins = _grid_env_origins(num_envs, spacing=spacing)
    displayed = qpos + origins[:, None, :]
    if track_selected:
        displayed = displayed - displayed[selected_env].mean(axis=0)

    anchor_positions = displayed.reshape(-1, 3)
    anchor_wxyzs = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs * anchor_count, 1))
    anchor_colors = np.empty((num_envs * anchor_count, 3), dtype=np.uint8)
    for env_index in range(num_envs):
        for anchor_index in range(anchor_count):
            flat_index = env_index * anchor_count + anchor_index
            anchor_colors[flat_index] = _anchor_color(workspace, anchor_index)

    if show_only_selected and num_envs > 1:
        env_opacity = np.zeros(num_envs, dtype=np.float64)
        env_opacity[selected_env] = 1.0
        anchor_opacities: np.ndarray | None = np.repeat(env_opacity, anchor_count)
    else:
        anchor_opacities = None

    rod_transforms: dict[int, RodTransformState] = {}
    topology = workspace.topology
    for rod_index in range(topology.rod_anchors.shape[0]):
        anchor_a, anchor_b = topology.rod_anchors[rod_index]
        start = displayed[:, anchor_a, :]
        end = displayed[:, anchor_b, :]
        midpoint = (start + end) / 2.0
        direction = end - start
        lengths = np.linalg.norm(direction, axis=1)
        wxyz = _batch_quaternion_from_z_axis(direction)
        geometry = _rod_visual_geometry(
            workspace,
            rod_index,
            current_length=float(lengths[0]) if lengths.size else 0.0,
        )
        base_length = _rod_visual_base_length(workspace, rod_index, current_length=float(lengths[0]))
        slide = (lengths - base_length) * 0.5
        center_offset = base_length * 0.25 + slide
        local_l = np.zeros((num_envs, 3), dtype=np.float64)
        local_l[:, 2] = -center_offset
        local_r = np.zeros((num_envs, 3), dtype=np.float64)
        local_r[:, 2] = center_offset
        rod_transforms[rod_index] = RodTransformState(
            midpoint=midpoint,
            wxyz=wxyz,
            length=float(lengths[0]) if lengths.size else 0.0,
            geometry=geometry,
            color=_rod_color(workspace, rod_index),
            rod_l_position=midpoint + _batch_rotate_vectors(wxyz, local_l),
            rod_r_position=midpoint + _batch_rotate_vectors(wxyz, local_r),
        )

    return BatchRenderState(
        batch_qpos=qpos,
        displayed_qpos=displayed,
        origins=origins,
        selected_env=selected_env,
        show_only_selected=show_only_selected,
        track_selected=track_selected,
        anchor_positions=anchor_positions,
        anchor_wxyzs=anchor_wxyzs,
        anchor_colors=anchor_colors,
        anchor_opacities=anchor_opacities,
        rod_transforms=rod_transforms,
    )


def _rod_visual_base_length(
    workspace: Workspace,
    index: int,
    *,
    current_length: float,
) -> float:
    """返回三段式视觉的收拢基准长度。"""
    topology = workspace.topology
    base_length = float(topology.rod_min_length[index])
    if topology.rod_length_limits.shape[0] > index:
        base_length = float(topology.rod_length_limits[index, 0])
    if base_length <= 1e-9:
        base_length = max(float(topology.rod_rest_length[index]), float(current_length), 1e-9)
    return base_length

def _rod_visual_geometry(
    workspace: Workspace,
    index: int,
    *,
    current_length: float,
) -> RodVisualGeometry:
    topology = workspace.topology
    robot = workspace.robot_config

    base_length = _rod_visual_base_length(workspace, index, current_length=current_length)

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
    """返回将局部 +Z 轴旋转到目标方向的四元数，格式为 wxyz（单向量版）。"""
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
