from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vgtr_py.config import default_config
from vgtr_py.schema import LegacyWorkspaceFile
from vgtr_py.ui import VgtrUiApp
from vgtr_py.workspace import Workspace


def make_workspace() -> Workspace:
    """
    辅助函数：快速构造一个供内存操作测试用的 3顶点 2条边 拓扑的工作区。
    """
    return Workspace.from_legacy_file(
        LegacyWorkspaceFile(
            v=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            e=[
                [0, 1],
                [1, 2],
            ],
        ),
        default_config(),
    )


def test_edge_click_toggle_mode_updates_edge_selection() -> None:
    """
    测试：在 UI 的 Toggle（切换/多选）模式下，点击同一条边会翻转该边的选中状态
    （选中 -> 取消选中，反之亦然），而不影响系统里的其他选择状态。
    """
    # Arrange: 准备应用环境，并强行将选择模式设置为 toggle 模式
    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._select_mode = SimpleNamespace(value="toggle")
    app.refresh = lambda: None  # 禁用真实渲染刷新

    # Act 1: 第一次点击索引为 1 的边 (即第二条边 [1-2])
    app._on_edge_click(1)

    # Assert 1: 验证仅该边状态变为选中 (状态码2)
    np.testing.assert_array_equal(workspace.ui.edge_status, np.asarray([0, 2], dtype=np.int8))

    # Act 2: 第二次点击相同的这根边
    app._on_edge_click(1)

    # Assert 2: 验证选中状态重置，一切归零
    np.testing.assert_array_equal(workspace.ui.edge_status, np.asarray([0, 0], dtype=np.int8))


def test_edge_click_replace_mode_clears_vertex_selection() -> None:
    """
    测试：在 UI 的 Replace（单选替换）模式下，点击某条边除了能选中它，还要能够
    干净地清空这之前用户的其他选择（例如清空那些已经被选中的顶点）。
    """
    # Arrange: 工作区中，手动让某些顶点预先处于被选中状态，系统配置为 replace 模式
    workspace = make_workspace()
    workspace.ui.vertex_status[:] = np.asarray([2, 0, 0], dtype=np.int8)
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._select_mode = SimpleNamespace(value="replace")
    app.refresh = lambda: None

    # Act: 点击了索引为 0 的边 ([0-1])
    app._on_edge_click(0)

    # Assert: 原先被选中的顶点数组现在必须是一干二净 (全部归零)
    np.testing.assert_array_equal(workspace.ui.vertex_status, np.asarray([0, 0, 0], dtype=np.int8))
    # 并且边 0 现在是全场唯一被选定的对象 (状态码为2)
    np.testing.assert_array_equal(workspace.ui.edge_status, np.asarray([2, 0], dtype=np.int8))


def test_load_workspace_updates_state_and_controls(tmp_path: Path) -> None:
    """
    测试：当 UI 组件触发加载新工程（_load_workspace）时，不仅要覆写后端的底层 Workspace 数据，
    还必须要顺藤摸瓜同步更新所有上层 GUI Controls 面板（Checkbox, Text 等）上的状态显示。
    """
    # Arrange: 写了一套崭新的包含2个点的配置文件用于加载
    config_path = tmp_path / "config.json"
    example_path = tmp_path / "example.json"
    config_path.write_text('{"directionalFriction": 1}', encoding="utf-8")
    example_path.write_text(
        '{"v": [[0, 0, 0], [1, 0, 0]], "e": [[0, 1]], "edgeChannel": [2]}',
        encoding="utf-8",
    )

    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())

    # 伪造大量绑定在前端界面上的可交互 DOM 组件 (mock UI objects)
    app._simulate_checkbox = SimpleNamespace(value=None)
    app._editing_checkbox = SimpleNamespace(value=None)
    app._move_joint_checkbox = SimpleNamespace(value=None)
    app._show_channel_checkbox = SimpleNamespace(value=None)
    app._config_path_text = SimpleNamespace(value="")
    app._example_path_text = SimpleNamespace(value="")
    app._save_path_text = SimpleNamespace(value="")
    app.refresh = lambda: None

    # Act: 发起 UI 件的载入底层工作区的动作
    app._load_workspace(config_path=config_path, example_path=example_path)

    # Assert: 验证双向绑定与状态同步完全生效
    assert app.current_config_path == config_path
    assert app.current_example_path == example_path
    assert app.workspace.topology.vertices.shape == (2, 3)  # 后端点从原本3变成了由文件接管的2个点

    np.testing.assert_array_equal(
        app.workspace.topology.edge_channel, np.asarray([2], dtype=np.int32)
    )
    # UI 面板视图（ViewModel层）也必须跟随底层数据保持一致
    assert app.workspace.ui.show_channel is True
    assert app._show_channel_checkbox.value is True
    assert app._config_path_text.value == str(config_path)
    assert app._example_path_text.value == str(example_path)
    assert app._save_path_text.value == str(example_path)


def test_sync_gui_updates_edit_folder_visibility() -> None:
    """
    测试：UI 同步引擎（_sync_gui_from_workspace）能否根据核心状态里的开关，
    动态地展开或隐藏特定区域的控制面版（即测试 编辑状态 对应的工具文件夹可见性）。
    """
    # Arrange: 强行干预底层核心库状态，将其设置为处于 "编辑态" (editing = True)
    workspace = make_workspace()
    workspace.ui.editing = True
    workspace.ui.moving_joint = True
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())

    app._editing_checkbox = SimpleNamespace(value=None)
    app._move_joint_checkbox = SimpleNamespace(value=None)
    app._simulate_checkbox = SimpleNamespace(value=None)
    app._show_channel_checkbox = SimpleNamespace(value=None)
    app._edit_tools_folder = SimpleNamespace(visible=False)  # 初始预设工具栏被折叠/不可见

    # Act: 模拟每帧对前端界面发起的全量同步刷新
    app._sync_gui_from_workspace()

    # Assert: 因核心为真，相关复选框全都需要被同步勾上，且关键工具菜单文件夹必须暴露出来
    assert app._editing_checkbox.value is True
    assert app._move_joint_checkbox.value is True
    assert app._edit_tools_folder.visible is True
