from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vgtr_py.config import default_config
from vgtr_py.model import compile_workspace
from vgtr_py.runtime import RuntimeSession, make_state
from vgtr_py.schema import RodGroupFile, SiteFile, WorkspaceFile
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
                RodGroupFile(name="g1", site1="s1", site2="s2", actuated=True, rod_type="active"),
                RodGroupFile(name="g2", site1="s2", site2="s3", actuated=True, rod_type="active"),
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
    app._file_status_markdown = SimpleNamespace(content="")  # Fix mock
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
    app._file_status_markdown = SimpleNamespace(content="")  # Fix mock
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


def test_update_status_includes_runtime_summary() -> None:
    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._status_markdown = SimpleNamespace(content="")
    workspace.ui.simulate = True
    workspace.ui.anchor_status[:] = np.asarray([2, 2, 2], dtype=np.int8)
    workspace.ui.rod_group_status[:] = np.asarray([2, 2], dtype=np.int8)
    assert app.session is not None
    app.session.state.step_count[0] = 12
    app.session.state.ctrl_target[0] = np.asarray([0.1, 0.2], dtype=np.float64)

    app._update_status()

    assert "running" in app._status_markdown.content
    assert "Step:** 12" in app._status_markdown.content
    assert "Ctrl Target" in app._status_markdown.content


def test_refresh_actuation_ui_per_rod_mode_builds_selected_rod_sliders() -> None:
    class DummyFolder:
        visible = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummySlider:
        def __init__(self, value: float) -> None:
            self.value = value
            self._cb = None

        def on_update(self, cb):
            self._cb = cb
            return cb

        def remove(self) -> None:
            return None

    created_labels: list[str] = []

    def add_slider(label: str, **kwargs):
        created_labels.append(label)
        return DummySlider(kwargs["initial_value"])

    workspace = make_workspace()
    workspace.ui.rod_group_status[:] = np.asarray([2, 0], dtype=np.int8)
    app = VgtrUiApp(
        server=SimpleNamespace(gui=SimpleNamespace(add_slider=add_slider)),
        workspace=workspace,
        renderer=SimpleNamespace(),
    )
    app.model = compile_workspace(workspace)
    app.session = RuntimeSession(app.model, make_state(app.model), control_mode="direct")
    app._actuation_folder = DummyFolder()
    app._actuation_status_markdown = SimpleNamespace(content="")
    app._actuation_mode_dropdown = SimpleNamespace(value="per-rod")

    app._refresh_actuation_ui()

    assert created_labels == ["Rod 0", "Rod 1"]


def test_set_batch_mode_disables_editing_controls() -> None:
    workspace = make_workspace()
    workspace.ui.editing = True
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._num_envs_input = SimpleNamespace(value=3)
    app._env_select_slider = SimpleNamespace(max=0.0)
    app._editing_checkbox = SimpleNamespace(value=True, disabled=False)
    app._move_anchor_checkbox = SimpleNamespace(value=True, disabled=False)
    app._edit_tools_folder = SimpleNamespace(visible=True)
    app.refresh = lambda: None

    app._set_batch_mode(True)

    assert app._batch_mode is True
    assert workspace.ui.editing is False
    assert app._batch_session is not None
    assert app._editing_checkbox.value is False
    assert app._editing_checkbox.disabled is True
    assert app._move_anchor_checkbox.value is False
    assert app._move_anchor_checkbox.disabled is True
    assert app._edit_tools_folder.visible is False


def test_set_batch_mode_false_restores_editing_controls() -> None:
    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._batch_mode = True
    app._batch_session = RuntimeSession(
        app.model,
        make_state(app.model, num_envs=1),
        control_mode="direct",
    )
    app._batch_action = np.zeros((1, 1))
    app._editing_checkbox = SimpleNamespace(value=False, disabled=True)
    app._move_anchor_checkbox = SimpleNamespace(value=False, disabled=True)
    app.renderer = SimpleNamespace(_clear_batch_handles=lambda: None)
    app.refresh = lambda: None

    app._set_batch_mode(False)

    assert app._batch_mode is False
    assert app._batch_session is None
    assert app._batch_action is None
    assert app._editing_checkbox.disabled is False
    assert app._move_anchor_checkbox.disabled is False


def test_rebuild_batch_runtime_updates_select_max() -> None:
    workspace = make_workspace()
    app = VgtrUiApp(server=SimpleNamespace(), workspace=workspace, renderer=SimpleNamespace())
    app._env_select_slider = SimpleNamespace(max=0.0)

    app._rebuild_batch_runtime(5)

    assert app._batch_session is not None
    assert app._batch_session.num_envs == 5
    assert app._env_select_slider.max == 4.0
