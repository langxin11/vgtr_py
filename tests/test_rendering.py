"""渲染与运动学测试。

覆盖 SceneRenderer 场景构建、杆组视觉几何计算、四元数推导及 batched mesh 渲染。
"""

import numpy as np

from vgtr_py.config import default_config
from vgtr_py.rendering import (
    SceneRenderer,
    _batch_quaternion_from_z_axis,
    _batch_rotate_vectors,
    _compute_batch_render_state,
    _compute_single_rod_transform,
    _grid_env_origins,
    _quaternion_from_z_axis,
    _rod_visual_geometry,
    _unit_cylinder_mesh,
    _unit_icosphere_mesh,
)
from vgtr_py.schema import RodGroupFile, SiteFile, WorkspaceFile
from vgtr_py.workspace import Workspace


class DummyHandle:
    def __init__(self, **kwargs: object) -> None:
        self.removed = False
        self.visible = bool(kwargs.get("visible", True))
        for key, value in kwargs.items():
            setattr(self, key, value)

    def remove(self) -> None:
        self.removed = True

    def on_click(self, _callback: object) -> None:
        pass


class DummyBatchedMeshHandle(DummyHandle):
    """模拟 viser.BatchedMeshHandle 的可设置属性语义。"""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._batched_positions: np.ndarray = np.zeros((0, 3))
        self._batched_wxyzs: np.ndarray = np.zeros((0, 4))
        self._batched_colors: np.ndarray = np.zeros((0, 3), dtype=np.uint8)
        self.batched_scales = getattr(self, "batched_scales", None)
        self._batched_opacities: np.ndarray | None = None

    @property
    def batched_positions(self) -> np.ndarray:
        return self._batched_positions

    @batched_positions.setter
    def batched_positions(self, value: np.ndarray) -> None:
        self._batched_positions = np.asarray(value)

    @property
    def batched_wxyzs(self) -> np.ndarray:
        return self._batched_wxyzs

    @batched_wxyzs.setter
    def batched_wxyzs(self, value: np.ndarray) -> None:
        self._batched_wxyzs = np.asarray(value)

    @property
    def batched_colors(self) -> np.ndarray:
        return self._batched_colors

    @batched_colors.setter
    def batched_colors(self, value: np.ndarray) -> None:
        self._batched_colors = np.asarray(value)

    @property
    def batched_opacities(self) -> np.ndarray | None:
        return self._batched_opacities

    @batched_opacities.setter
    def batched_opacities(self, value: np.ndarray | None) -> None:
        self._batched_opacities = value if value is None else np.asarray(value)


class DummyScene:
    def __init__(self) -> None:
        self.cylinders: list[tuple[str, dict[str, object]]] = []
        self.icospheres: list[tuple[str, dict[str, object]]] = []
        self.batched_meshes: list[tuple[str, dict[str, object]]] = []

    def add_cylinder(self, name: str, **kwargs: object) -> DummyHandle:
        self.cylinders.append((name, kwargs))
        return DummyHandle(**kwargs)

    def add_icosphere(self, name: str, **kwargs: object) -> DummyHandle:
        self.icospheres.append((name, kwargs))
        return DummyHandle(**kwargs)

    def add_batched_meshes_simple(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        batched_positions: np.ndarray,
        batched_wxyzs: np.ndarray,
        *,
        batched_colors: np.ndarray | None = None,
        batched_opacities: np.ndarray | None = None,
        batched_scales: np.ndarray | tuple | None = None,
        material: str = "standard",
    ) -> DummyBatchedMeshHandle:
        self.batched_meshes.append((name, {"material": material}))
        return DummyBatchedMeshHandle(
            batched_positions=batched_positions,
            batched_wxyzs=batched_wxyzs,
            batched_colors=batched_colors,
            batched_scales=None if batched_scales is None else np.asarray(batched_scales),
            batched_opacities=batched_opacities,
        )

    def add_grid(self, *args: object, **kwargs: object) -> None:
        pass

    def add_transform_controls(self, *args: object, **kwargs: object) -> None:
        pass


class DummyServer:
    def __init__(self) -> None:
        self.scene = DummyScene()

    def atomic(self) -> "_DummyAtomic":
        return _DummyAtomic()

    def flush(self) -> None:
        pass


class _DummyAtomic:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


def make_render_workspace() -> Workspace:
    workspace = Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0]),
                "s2": SiteFile(pos=[2.0, 0.0, 0.0]),
            },
            rod_groups=[
                RodGroupFile(
                    name="g12",
                    site1="s1",
                    site2="s2",
                    rod_radius=0.03,
                    sleeve_radius=0.05,
                    sleeve_display_half_length_ratio=0.2,
                    length_limits=[1.6, 2.4],
                )
            ],
        ),
        default_config(),
    )
    workspace.topology.rod_rest_length[0] = 2.0
    return workspace


def rotate_with_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    q_vec = np.asarray([x, y, z], dtype=np.float64)
    uv = np.cross(q_vec, vector)
    uuv = np.cross(q_vec, uv)
    return vector + 2.0 * (w * uv + uuv)


# ---------------------------------------------------------------------------
# 四元数测试
# ---------------------------------------------------------------------------


def test_quaternion_from_z_axis_keeps_z_direction() -> None:
    quaternion = _quaternion_from_z_axis(np.asarray([0.0, 0.0, 2.0], dtype=np.float64))
    rotated = rotate_with_quaternion(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), quaternion)
    np.testing.assert_allclose(rotated, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def test_quaternion_from_z_axis_rotates_to_x_direction() -> None:
    quaternion = _quaternion_from_z_axis(np.asarray([3.0, 0.0, 0.0], dtype=np.float64))
    rotated = rotate_with_quaternion(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), quaternion)
    np.testing.assert_allclose(rotated, np.asarray([1.0, 0.0, 0.0], dtype=np.float64), atol=1e-7)


def test_batch_quaternion_from_z_axis_returns_correct_shape() -> None:
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    quats = _batch_quaternion_from_z_axis(dirs)
    assert quats.shape == (3, 4)


def test_batch_quaternion_near_z_is_identity() -> None:
    quats = _batch_quaternion_from_z_axis(np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
    np.testing.assert_allclose(quats[0], [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_batch_quaternion_negative_z_is_180_around_x() -> None:
    quats = _batch_quaternion_from_z_axis(np.array([[0.0, 0.0, -1.0]], dtype=np.float64))
    np.testing.assert_allclose(quats[0], [0.0, 1.0, 0.0, 0.0], atol=1e-7)


def test_batch_rotate_vectors_z_to_x() -> None:
    q = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]], dtype=np.float64)
    v = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    v_rot = _batch_rotate_vectors(q, v)
    np.testing.assert_allclose(v_rot[0], [0.0, -1.0, 0.0], atol=1e-7)


# ---------------------------------------------------------------------------
# Rod visual geometry
# ---------------------------------------------------------------------------


def test_rod_visual_geometry_slides_rods_half_of_length_delta() -> None:
    workspace = make_render_workspace()

    min_geometry = _rod_visual_geometry(workspace, 0, current_length=1.6)
    initial_geometry = _rod_visual_geometry(workspace, 0, current_length=2.0)
    extended_geometry = _rod_visual_geometry(workspace, 0, current_length=2.4)

    np.testing.assert_allclose(min_geometry.rod_l_position, (0.0, 0.0, -0.4))
    np.testing.assert_allclose(min_geometry.rod_r_position, (0.0, 0.0, 0.4))
    np.testing.assert_allclose(initial_geometry.rod_l_position, (0.0, 0.0, -0.6))
    np.testing.assert_allclose(initial_geometry.rod_r_position, (0.0, 0.0, 0.6))
    np.testing.assert_allclose(extended_geometry.rod_l_position, (0.0, 0.0, -0.8))
    np.testing.assert_allclose(extended_geometry.rod_r_position, (0.0, 0.0, 0.8))
    assert extended_geometry.sleeve_height == min_geometry.sleeve_height
    assert extended_geometry.rod_height == min_geometry.rod_height


def test_rod_visuals_create_rods_as_sleeve_children() -> None:
    workspace = make_render_workspace()
    renderer = SceneRenderer(server=DummyServer())  # type: ignore[arg-type]

    renderer._render_rod_visuals(workspace, workspace.topology.anchor_pos)

    names = [name for name, _kwargs in renderer.server.scene.cylinders]
    assert names == [
        "/world/rod_vis/0/sleeve",
        "/world/rod_vis/0/sleeve/rod_l",
        "/world/rod_vis/0/sleeve/rod_r",
    ]


# ---------------------------------------------------------------------------
# 网格工具
# ---------------------------------------------------------------------------


def test_grid_env_origins_tiles_envs_on_xy_grid() -> None:
    origins = _grid_env_origins(4, spacing=2.5)

    np.testing.assert_allclose(
        origins,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.0, 2.5, 0.0],
                [2.5, 2.5, 0.0],
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Mesh 生成
# ---------------------------------------------------------------------------


def test_unit_icosphere_mesh_shape() -> None:
    v, f = _unit_icosphere_mesh(subdiv=2)
    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert f.shape[0] > 0
    assert v.shape[0] > 0


def test_unit_cylinder_mesh_shape() -> None:
    v, f = _unit_cylinder_mesh(segments=16)
    assert v.shape[1] == 3
    assert f.shape[1] == 3
    assert v.shape == (34, 3)
    assert f.shape == (64, 3)
    np.testing.assert_allclose(v[-2:], [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])


def test_unit_cylinder_mesh_is_watertight() -> None:
    _v, f = _unit_cylinder_mesh(segments=16)
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in f:
        for i in range(3):
            edge = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    assert edge_counts
    assert all(count == 2 for count in edge_counts.values())


def test_mesh_caching() -> None:
    v1, f1 = _unit_icosphere_mesh()
    v2, f2 = _unit_icosphere_mesh()
    assert v1 is v2
    assert f1 is f2


# ---------------------------------------------------------------------------
# Render state / transform kernel
# ---------------------------------------------------------------------------


def test_batch_render_state_uses_env_major_anchor_order() -> None:
    workspace = make_render_workspace()
    workspace.ui.anchor_status[1] = 2
    qpos = np.stack(
        [
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos + [3.0, 0.0, 0.0],
        ]
    )

    state = _compute_batch_render_state(
        workspace,
        batch_qpos=qpos,
        selected_env=1,
        show_only_selected=True,
        spacing=3.0,
        track_selected=False,
    )

    np.testing.assert_allclose(state.anchor_positions, state.displayed_qpos.reshape(-1, 3))
    np.testing.assert_array_equal(
        state.anchor_colors,
        np.asarray(
            [
                [153, 153, 153],
                [230, 0, 0],
                [153, 153, 153],
                [230, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_allclose(state.anchor_opacities, [0.0, 0.0, 1.0, 1.0])


def test_single_and_batch_share_rod_transform_kernel() -> None:
    workspace = make_render_workspace()
    qpos = np.stack(
        [workspace.topology.anchor_pos, workspace.topology.anchor_pos + [3.0, 0.0, 0.0]]
    )

    single = _compute_single_rod_transform(workspace, workspace.topology.anchor_pos, 0)
    batch = _compute_batch_render_state(
        workspace,
        batch_qpos=qpos,
        selected_env=0,
        show_only_selected=False,
        spacing=3.0,
        track_selected=False,
    ).rod_transforms[0]

    np.testing.assert_allclose(batch.midpoint[0], single.midpoint)
    np.testing.assert_allclose(batch.wxyz[0], single.wxyz)
    np.testing.assert_allclose(batch.rod_l_position[0], single.rod_l_position, atol=1e-12)
    np.testing.assert_allclose(batch.rod_r_position[0], single.rod_r_position, atol=1e-12)


def test_batch_rod_visuals_use_min_length_baseline() -> None:
    workspace = make_render_workspace()
    qpos = np.stack(
        [
            np.asarray([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], dtype=np.float64),
        ]
    )

    state = _compute_batch_render_state(
        workspace,
        batch_qpos=qpos,
        selected_env=0,
        show_only_selected=False,
        spacing=3.0,
        track_selected=False,
    )
    geometry = _rod_visual_geometry(workspace, 0, current_length=1.6)

    np.testing.assert_allclose(
        state.rod_transforms[0].rod_l_position[0],
        state.rod_transforms[0].midpoint[0] + [-0.4, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(geometry.rod_l_position, (0.0, 0.0, -0.4))


# ---------------------------------------------------------------------------
# Batched mesh 渲染
# ---------------------------------------------------------------------------


def test_render_batch_creates_batched_mesh_handles() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    qpos = np.stack(
        [
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos + [3.0, 0.0, 0.0],
        ]
    )

    renderer.render_batch(workspace, batch_qpos=qpos, spacing=3.0)

    # 验证创建了 batched mesh handles（不是 cylinder 原语）
    names = [name for name, _kwargs in server.scene.batched_meshes]
    assert names == [
        "/world/batch/anchors",
        "/world/batch/rod_vis/0/sleeve",
        "/world/batch/rod_vis/0/rod_l",
        "/world/batch/rod_vis/0/rod_r",
    ]

    # anchor handle batch_count == num_envs * anchor_count
    assert renderer.batch_registry.anchor_mesh is not None
    assert renderer.batch_registry.anchor_mesh.batched_positions.shape == (4, 3)
    assert renderer.batch_registry.anchor_mesh.batched_scales is not None
    assert renderer.batch_registry.anchor_mesh.batched_scales.shape == (4,)

    # rod handles 各 num_envs 个实例
    rh = renderer.batch_registry.rod_handles[0]
    assert rh.sleeve.batched_positions.shape == (2, 3)
    assert rh.sleeve.batched_scales is not None
    assert rh.sleeve.batched_scales.shape == (2, 3)


def test_render_batch_selected_env_opacity() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    qpos = np.stack(
        [
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos + [3.0, 0.0, 0.0],
        ]
    )

    renderer.render_batch(
        workspace,
        batch_qpos=qpos,
        selected_env=0,
        show_only_selected=True,
    )

    # 验证 opacity 数组只开启 selected env
    rh = renderer.batch_registry.rod_handles[0]
    opac = rh.sleeve.batched_opacities
    assert opac is not None
    assert opac[0] == 1.0
    assert opac[1] == 0.0


def test_render_batch_hide_others_off_restores_full_visibility() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    qpos = np.stack(
        [
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos + [3.0, 0.0, 0.0],
        ]
    )

    renderer.render_batch(workspace, batch_qpos=qpos, show_only_selected=False)

    rh = renderer.batch_registry.rod_handles[0]
    assert rh.sleeve.batched_opacities is None  # None = 全部可见


def test_render_batch_selected_env_out_of_bounds_raises() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    qpos = np.stack([workspace.topology.anchor_pos, workspace.topology.anchor_pos])

    import pytest

    with pytest.raises(ValueError):
        renderer.render_batch(workspace, batch_qpos=qpos, selected_env=5)


def test_render_batch_clears_single_handles() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    # 先走单实例渲染创建 cylinder handles
    renderer.render(workspace)
    assert len(server.scene.cylinders) > 0

    # 切换到 batch → 单实例句柄应当被清理
    qpos = np.stack([workspace.topology.anchor_pos, workspace.topology.anchor_pos])
    renderer.render_batch(workspace, batch_qpos=qpos)
    for h in list(renderer.anchor_handles.values()) + list(renderer.sleeve_handles.values()):
        assert h.removed


def test_render_batch_handles_reused_on_same_dimensions() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    qpos = np.stack([workspace.topology.anchor_pos, workspace.topology.anchor_pos])

    renderer.render_batch(workspace, batch_qpos=qpos)
    anchor_h1 = renderer.batch_registry.anchor_mesh
    rod_h1 = renderer.batch_registry.rod_handles[0].sleeve

    # 同尺寸再次调用 → 复用 handle
    renderer.render_batch(workspace, batch_qpos=qpos + 0.1)
    assert renderer.batch_registry.anchor_mesh is anchor_h1
    assert renderer.batch_registry.rod_handles[0].sleeve is rod_h1


def test_render_batch_recreates_handles_when_dimensions_change() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    qpos_2 = np.stack([workspace.topology.anchor_pos, workspace.topology.anchor_pos])
    qpos_3 = np.stack(
        [
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos,
        ]
    )

    renderer.render_batch(workspace, batch_qpos=qpos_2)
    anchor_h1 = renderer.batch_registry.anchor_mesh

    renderer.render_batch(workspace, batch_qpos=qpos_3)

    assert anchor_h1 is not None
    assert anchor_h1.removed is True
    assert renderer.batch_registry.anchor_mesh is not anchor_h1
    assert renderer.batch_registry.num_envs == 3


def test_single_render_clears_batch_handles() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]

    qpos = np.stack([workspace.topology.anchor_pos, workspace.topology.anchor_pos])
    renderer.render_batch(workspace, batch_qpos=qpos)
    assert renderer.batch_registry.anchor_mesh is not None

    # 切换回单实例 → batch handles 清理
    renderer.render(workspace)
    assert renderer.batch_registry.anchor_mesh is None
    assert len(renderer.batch_registry.rod_handles) == 0
