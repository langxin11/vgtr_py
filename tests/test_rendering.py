"""渲染与运动学测试。

覆盖 SceneRenderer 场景构建、杆组视觉几何计算及四元数推导。
"""

import numpy as np

from vgtr_py.config import default_config
from vgtr_py.rendering import (
    SceneRenderer,
    _grid_env_origins,
    _quaternion_from_z_axis,
    _rod_visual_geometry,
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


class DummyScene:
    def __init__(self) -> None:
        self.cylinders: list[tuple[str, dict[str, object]]] = []
        self.icospheres: list[tuple[str, dict[str, object]]] = []

    def add_cylinder(self, name: str, **kwargs: object) -> DummyHandle:
        self.cylinders.append((name, kwargs))
        return DummyHandle(**kwargs)

    def add_icosphere(self, name: str, **kwargs: object) -> DummyHandle:
        self.icospheres.append((name, kwargs))
        return DummyHandle(**kwargs)


class DummyServer:
    def __init__(self) -> None:
        self.scene = DummyScene()


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
    """
    根据给定的四元数旋转一个三维向量。

    参数:
        vector: 需要旋转的原始三维向量（长度为3的 numpy 数组）
        quaternion: 表示旋转的四元数 [w, x, y, z]（长度为4的 numpy 数组）

    返回:
        旋转后的三维向量
    """
    w, x, y, z = quaternion
    q_vec = np.asarray([x, y, z], dtype=np.float64)
    uv = np.cross(q_vec, vector)
    uuv = np.cross(q_vec, uv)
    return vector + 2.0 * (w * uv + uuv)


def test_quaternion_from_z_axis_keeps_z_direction() -> None:
    """
    测试：如果目标方向已经是 Z 轴正向，则生成的四元数不应对 Z 轴向量产生任何旋转效果。
    """
    # Arrange: 构造目标向量 (0, 0, 2)
    quaternion = _quaternion_from_z_axis(np.asarray([0.0, 0.0, 2.0], dtype=np.float64))

    # Act: 尝试用该四元数旋转标准的 Z 轴单位向量 (0, 0, 1)
    rotated = rotate_with_quaternion(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), quaternion)

    # Assert: 旋转后的向量必须保持为 (0, 0, 1)
    np.testing.assert_allclose(rotated, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def test_quaternion_from_z_axis_rotates_to_x_direction() -> None:
    """
    测试：如果目标方向是 X 轴方向，则生成的四元数应该将 Z 轴基础向量正确旋转到 X 轴方向。
    """
    # Arrange: 构造指向 X 轴的目标向量 (3, 0, 0)
    quaternion = _quaternion_from_z_axis(np.asarray([3.0, 0.0, 0.0], dtype=np.float64))

    # Act: 尝试用该四元数旋转标准的 Z 轴单位向量 (0, 0, 1)
    rotated = rotate_with_quaternion(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), quaternion)

    # Assert: 旋转后的向量必须是指向 X 轴的单位向量 (1, 0, 0)，考虑到计算误差设置了 atol
    np.testing.assert_allclose(rotated, np.asarray([1.0, 0.0, 0.0], dtype=np.float64), atol=1e-7)


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
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]

    renderer._render_rod_visuals(workspace, workspace.topology.anchor_pos)

    names = [name for name, _kwargs in server.scene.cylinders]
    assert names == [
        "/world/rod_vis/0/sleeve",
        "/world/rod_vis/0/sleeve/rod_l",
        "/world/rod_vis/0/sleeve/rod_r",
    ]


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


def test_render_batch_creates_env_prefixed_three_part_rods_and_anchors() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    batch_anchor_pos = np.stack(
        [
            workspace.topology.anchor_pos,
            workspace.topology.anchor_pos + np.asarray([0.0, 0.0, 0.5]),
        ]
    )

    renderer.render_batch(
        workspace,
        batch_anchor_pos=batch_anchor_pos,
        env_origins=np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )

    cylinder_names = [name for name, _kwargs in server.scene.cylinders]
    assert cylinder_names == [
        "/world/envs/0/rod_vis/0/sleeve",
        "/world/envs/0/rod_vis/0/sleeve/rod_l",
        "/world/envs/0/rod_vis/0/sleeve/rod_r",
        "/world/envs/1/rod_vis/0/sleeve",
        "/world/envs/1/rod_vis/0/sleeve/rod_l",
        "/world/envs/1/rod_vis/0/sleeve/rod_r",
    ]
    icosphere_names = [name for name, _kwargs in server.scene.icospheres]
    assert icosphere_names == [
        "/world/envs/0/anchors/v_0",
        "/world/envs/0/anchors/v_1",
        "/world/envs/1/anchors/v_0",
        "/world/envs/1/anchors/v_1",
    ]
    np.testing.assert_allclose(
        renderer.batch_anchor_handles[(1, 0)].position,
        (3.0, 0.0, 0.5),
    )


def test_render_batch_can_hide_non_selected_envs() -> None:
    workspace = make_render_workspace()
    server = DummyServer()
    renderer = SceneRenderer(server=server)  # type: ignore[arg-type]
    batch_anchor_pos = np.stack([workspace.topology.anchor_pos, workspace.topology.anchor_pos])

    renderer.render_batch(
        workspace,
        batch_anchor_pos=batch_anchor_pos,
        selected_env=1,
        show_only_selected=True,
    )

    assert renderer.batch_anchor_handles[(0, 0)].visible is False
    assert renderer.batch_anchor_handles[(1, 0)].visible is True
    assert renderer.batch_sleeve_handles[(0, 0)].visible is False
    assert renderer.batch_sleeve_handles[(1, 0)].visible is True
