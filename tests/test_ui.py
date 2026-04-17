from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vgtr_py.config import default_config
from vgtr_py.schema import WorkspaceFile, SiteFile, RodGroupFile
from vgtr_py.ui import VgtrUiApp
from vgtr_py.workspace import Workspace


def make_workspace() -> Workspace:
    """
    辅助函数：快速构造一个供内存操作测试用的 3锚点 2杆组 拓扑的工作区。
    """
    return Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0]),
                "s2": SiteFile(pos=[1.0, 0.0, 0.0]),
                "s3": SiteFile(pos=[2.0, 0.0, 0.0]),
            },
            rod_groups=[
                RodGroupFile(name="g1", site1="s1", site2="s2"),
                RodGroupFile(name="g2", site1="s2", site2="s3"),
            ],
        ),
        default_config(),
    )


def test_rod_group_click_toggle_mode_updates_selection() -> None:
    """
    测试：在 UI 的 Toggle 模式下，点击杆组会翻转其选中状态。
    """
    # Arrange
    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._select_mode = SimpleNamespace(value="toggle")
    app._file_status_markdown = SimpleNamespace(content="") # Fix mock
    app.refresh = lambda: None

    # Act 1: 第一次点击索引为 1 的杆组
    app._on_rod_group_click(1)

    # Assert 1
    np.testing.assert_array_equal(workspace.ui.rod_group_status, np.asarray([0, 2], dtype=np.int8))

    # Act 2: 第二次点击
    app._on_rod_group_click(1)

    # Assert 2
    np.testing.assert_array_equal(workspace.ui.rod_group_status, np.asarray([0, 0], dtype=np.int8))


def test_rod_group_click_replace_mode_clears_anchor_selection() -> None:
    """
    测试：在 UI 的 Replace 模式下，点击杆组会清空之前的锚点选择。
    """
    # Arrange
    workspace = make_workspace()
    workspace.ui.anchor_status[:] = np.asarray([2, 0, 0], dtype=np.int8)
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._select_mode = SimpleNamespace(value="replace")
    app._file_status_markdown = SimpleNamespace(content="") # Fix mock
    app.refresh = lambda: None

    # Act
    app._on_rod_group_click(0)

    # Assert
    np.testing.assert_array_equal(workspace.ui.anchor_status, np.asarray([0, 0, 0], dtype=np.int8))
    np.testing.assert_array_equal(workspace.ui.rod_group_status, np.asarray([2, 0], dtype=np.int8))


def test_load_workspace_updates_state_and_controls(tmp_path: Path) -> None:
    """
    测试：加载新工程时同步更新 GUI 控件。
    """
    # Arrange
    example_path = tmp_path / "example.json"
    example_path.write_text(
        '{"sites": {"s1": {"pos": [0,0,0]}, "s2": {"pos": [1,0,0]}}, "rod_groups": [{"name": "g1", "site1": "s1", "site2": "s2"}]}',
        encoding="utf-8",
    )

    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())

    # Mock UI components
    app._simulate_checkbox = SimpleNamespace(value=None)
    app._editing_checkbox = SimpleNamespace(value=None)
    app._move_anchor_checkbox = SimpleNamespace(value=None)
    app._show_control_group_checkbox = SimpleNamespace(value=None)
    app._example_dropdown = SimpleNamespace(value="", options=["example.json"])
    app._save_path_text = SimpleNamespace(value="")
    app._file_status_markdown = SimpleNamespace(content="")
    app._k_slider = SimpleNamespace(value=0.0)
    app._damping_slider = SimpleNamespace(value=0.0)
    app._friction_slider = SimpleNamespace(value=0.0)
    app._ground_k_slider = SimpleNamespace(value=0.0)
    app._ground_d_slider = SimpleNamespace(value=0.0)
    app._gravity_toggle = SimpleNamespace(value=None)
    app._scripting_toggle = SimpleNamespace(value=None)
    app._refresh_actuation_ui = lambda: None
    app.refresh = lambda: None

    # Act
    app._load_workspace(config_path=None, example_path=example_path)

    # Assert
    assert app.current_example_path == example_path
    assert app.workspace.topology.anchor_pos.shape == (2, 3)
    assert app._example_dropdown.value == "example.json"
    assert app._save_path_text.value == str(example_path)


def test_sync_gui_updates_edit_folder_visibility() -> None:
    """
    测试：UI 同步逻辑能否正确处理文件夹可见性。
    """
    # Arrange
    workspace = make_workspace()
    workspace.ui.editing = True
    workspace.ui.moving_anchor = True
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())

    app._editing_checkbox = SimpleNamespace(value=None)
    app._move_anchor_checkbox = SimpleNamespace(value=None)
    app._simulate_checkbox = SimpleNamespace(value=None)
    app._show_control_group_checkbox = SimpleNamespace(value=None)
    app._edit_tools_folder = SimpleNamespace(visible=False)
    
    # 物理参数句柄
    app._k_slider = SimpleNamespace(value=0.0)
    app._damping_slider = SimpleNamespace(value=0.0)
    app._friction_slider = SimpleNamespace(value=0.0)
    app._ground_k_slider = SimpleNamespace(value=0.0)
    app._ground_d_slider = SimpleNamespace(value=0.0)
    app._gravity_toggle = SimpleNamespace(value=None)

    # Act
    app._sync_gui_from_workspace()

    # Assert
    assert app._editing_checkbox.value is True
    assert app._move_anchor_checkbox.value is True
    assert app._edit_tools_folder.visible is True
