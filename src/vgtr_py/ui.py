"""前端用户界面总成。

基于 Viser 构建了带菜单栏、设置面板等交互控件以及快捷键绑定的 WebUI 主实例。同时维持逻辑与渲染的双线事件循环。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import viser

from .commands import (
    add_joint_from_selection,
    add_joint_from_vertex,
    center_model_command,
    clear_workspace_history,
    clear_workspace_selection,
    complete_drag_edit,
    connect_selected_vertices,
    fix_selected,
    load_workspace_from_paths,
    redo,
    remove_selected_edges_command,
    remove_selected_vertices_command,
    save_workspace_to_path,
    select_edge_by_mode,
    select_vertex_by_mode,
    set_selected_vertex_position,
    undo,
)
from .engine import step
from .history import WorkspaceHistory
from .rendering import SceneRenderer
from .workspace import Workspace, WorkspaceSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass
class VgtrUiApp:
    """VGTR 编辑与仿真界面应用入口类。

    该类通过集成 ViserServer 和后端 Workspace，
    提供图形界面操作、物理渲染控制和多线程步骤循环的胶水逻辑。

    Attributes:
        server: Viser 服务器实例，用于生成浏览器三维页面与通讯。
        workspace: 用于存储所有状态的后端工作区数据模型。
        renderer: 将工作区数据绘制到 Viser 场景的渲染器。
        render_hz: 引擎/渲染独立于绘图逻辑运行的每秒帧率(Hz)。
    """

    server: viser.ViserServer
    workspace: Workspace
    renderer: SceneRenderer
    render_hz: float = 60.0
    simulation_steps_per_second: float = 1000.0
    current_config_path: Path | None = None
    current_example_path: Path | None = None
    history: WorkspaceHistory = field(default_factory=WorkspaceHistory)

    _running: bool = False
    _lock: threading.RLock = threading.RLock()
    _simulation_thread: threading.Thread | None = None
    _simulate_checkbox: viser.GuiCheckboxHandle | None = None
    _editing_checkbox: viser.GuiCheckboxHandle | None = None
    _move_joint_checkbox: viser.GuiCheckboxHandle | None = None
    _show_channel_checkbox: viser.GuiCheckboxHandle | None = None
    _step_number: viser.GuiNumberHandle[int] | None = None
    _status_markdown: viser.GuiMarkdownHandle | None = None
    _config_path_text: viser.GuiTextHandle | None = None
    _example_path_text: viser.GuiTextHandle | None = None
    _save_path_text: viser.GuiTextHandle | None = None
    _file_status_markdown: viser.GuiMarkdownHandle | None = None
    _edit_tools_folder: viser.GuiFolderHandle | None = None
    _drag_snapshot: WorkspaceSnapshot | None = None
    _is_transform_dragging: bool = False
    _simulation_step_accumulator: float = 0.0
    _last_simulation_tick_time: float | None = None

    def start(self) -> None:
        """初始化场景和 GUI，并启动后台仿真循环。"""
        self.renderer.setup_scene()
        self.renderer.install_transform_callback(
            self._on_transform_update,
            drag_start=self._on_transform_drag_start,
            drag_end=self._on_transform_drag_end,
        )
        self._build_gui()
        self.refresh()
        self._running = True
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()

    def refresh(self) -> None:
        """立即刷新渲染和状态栏。"""
        with self._lock:
            self.renderer.render(self.workspace)
            self._update_status()

    def stop(self) -> None:
        """停止后台仿真循环。"""
        self._running = False

    def _set_editing_state(
        self,
        editing: bool,
        *,
        restore_pose: bool = False,
        preserve_move_state: bool = False,
    ) -> None:
        self.workspace.ui.editing = editing
        if editing and restore_pose:
            self.workspace.restore_initial_state()
        if editing and getattr(self, "_select_mode", None) is not None:
            self._select_mode.value = "toggle"
        if not preserve_move_state:
            self.workspace.ui.moving_joint = False
            self.workspace.ui.moving_body = False
            if self._move_joint_checkbox is not None:
                self._move_joint_checkbox.value = False
        self._is_transform_dragging = False
        self._drag_snapshot = None
        if self._edit_tools_folder is not None:
            self._edit_tools_folder.visible = editing

    def _build_gui(self) -> None:
        """初始化 Viser 后台与前端控制面板，包含各类复选框、按钮和操作选项组与它们的回调函数。

        负责“工作区”、“网格选择”、“拓扑建立”以及“步进操作”UI的定义及交互挂载。
        """
        with self.server.gui.add_folder("Workspace"):
            config_text = self.server.gui.add_text(
                "Config Path",
                str(self.current_config_path or ""),
            )
            example_text = self.server.gui.add_text(
                "Example Path",
                str(self.current_example_path or ""),
            )
            load_button = self.server.gui.add_button("Load Workspace")
            save_text = self.server.gui.add_text(
                "Save Path",
                str(self.current_example_path or ""),
            )
            save_button = self.server.gui.add_button("Save Workspace")
            undo_button = self.server.gui.add_button("Undo")
            redo_button = self.server.gui.add_button("Redo")
            self._file_status_markdown = self.server.gui.add_markdown("Ready.")
            simulate = self.server.gui.add_checkbox("Simulate", self.workspace.ui.simulate)
            editing = self.server.gui.add_checkbox("Editing", self.workspace.ui.editing)
            show_channel = self.server.gui.add_checkbox(
                "Show Control Group", self.workspace.ui.show_channel
            )
            simulation_rate = self.server.gui.add_number(
                "Simulation Steps / Second",
                int(self.simulation_steps_per_second),
                min=1,
                max=5000,
                step=50,
            )
            render_rate = self.server.gui.add_dropdown(
                "Render Hz",
                ("30", "60"),
                initial_value=str(int(self.render_hz)),
            )
            step_count = self.server.gui.add_number("Manual Step Count", 8, min=1, max=200, step=1)

        edit_tools_folder = self.server.gui.add_folder(
            "Editing Tools", visible=self.workspace.ui.editing
        )
        with edit_tools_folder:
            move_joint = self.server.gui.add_checkbox("Move Anchor", self.workspace.ui.moving_joint)
            with self.server.gui.add_folder("Selection"):
                clear_button = self.server.gui.add_button("Clear Selection")
                fix_button = self.server.gui.add_button("Fix Selected")
                unfix_button = self.server.gui.add_button("Unfix Selected")

            with self.server.gui.add_folder("Topology"):
                add_joint_button = self.server.gui.add_button("Add Anchor")
                connect_button = self.server.gui.add_button("Add Rod Group")
                remove_edge_button = self.server.gui.add_button("Remove Selected Rod Groups")
                remove_vertex_button = self.server.gui.add_button("Remove Selected Anchors")
                center_button = self.server.gui.add_button("Center Model")
                select_mode = self.server.gui.add_dropdown(
                    "Click Mode",
                    ("replace", "toggle", "add-child"),
                    initial_value="toggle",
                )

        @load_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                try:
                    self._load_workspace(
                        config_path=Path(config_text.value).expanduser(),
                        example_path=Path(example_text.value).expanduser(),
                    )
                except Exception as exc:
                    self._set_file_status(f"Load failed: `{exc}`", log_terminal=True)
                else:
                    self._set_file_status(f"Loaded `{self.current_example_path}`", log_terminal=True)

        @save_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                try:
                    self._save_workspace(Path(save_text.value).expanduser())
                except Exception as exc:
                    self._set_file_status(f"Save failed: `{exc}`", log_terminal=True)
                else:
                    self._set_file_status(f"Saved `{save_text.value}`", log_terminal=True)

        @undo_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                self._undo()

        @redo_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                self._redo()

        @simulate.on_update
        def _(_event: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            with self._lock:
                self.workspace.ui.simulate = simulate.value
                self.refresh()

        @editing.on_update
        def _(_event: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            with self._lock:
                self._set_editing_state(editing.value, restore_pose=editing.value)
                self.refresh()

        @move_joint.on_update
        def _(_event: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            with self._lock:
                self.workspace.ui.moving_joint = move_joint.value and self.workspace.ui.editing
                if move_joint.value and not self.workspace.ui.editing:
                    self._set_editing_state(
                        True,
                        restore_pose=True,
                        preserve_move_state=True,
                    )
                    self.workspace.ui.moving_joint = True
                    editing.value = True
                self.refresh()

        @show_channel.on_update
        def _(_event: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
            with self._lock:
                self.workspace.ui.show_channel = show_channel.value
                self.refresh()

        @simulation_rate.on_update
        def _(_event: viser.GuiEvent[viser.GuiNumberHandle[int]]) -> None:
            with self._lock:
                self.simulation_steps_per_second = float(simulation_rate.value)
                self._simulation_step_accumulator = 0.0
                self._last_simulation_tick_time = None
                self.refresh()

        @render_rate.on_update
        def _(_event: viser.GuiEvent[viser.GuiDropdownHandle[str]]) -> None:
            with self._lock:
                self.render_hz = float(render_rate.value)

        @clear_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                clear_workspace_selection(self.workspace)
                self._set_file_status("Selection cleared.", log_terminal=True)
                self.refresh()

        @fix_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                fix_selected(self.workspace, self.history, True)
                self.refresh()

        @unfix_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                fix_selected(self.workspace, self.history, False)
                self.refresh()

        @add_joint_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                selected = np.flatnonzero(self.workspace.ui.vertex_status == 2).tolist()
                before_edges = int(self.workspace.topology.edges.shape[0])
                changed = add_joint_from_selection(self.workspace, self.history)
                if changed:
                    new_index = int(self.workspace.topology.vertices.shape[0] - 1)
                    auto_connected = int(self.workspace.topology.edges.shape[0]) > before_edges
                    if len(selected) == 1:
                        self._set_file_status(
                            f"Added anchor {new_index} from selected anchor {selected[0]}."
                            + (" Auto-connected with a new rod group." if auto_connected else ""),
                            log_terminal=True,
                        )
                    else:
                        self._set_file_status(
                            f"Added anchor {new_index} from centroid of selection {selected}."
                            ,
                            log_terminal=True,
                        )
                else:
                    self._set_file_status("Add Anchor did not change the topology.", log_terminal=True)
                self.refresh()

        @connect_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                selected_count = int(np.flatnonzero(self.workspace.ui.vertex_status == 2).size)
                edge_count_before = int(self.workspace.topology.edges.shape[0])
                changed = connect_selected_vertices(self.workspace, self.history)
                edge_count_after = int(self.workspace.topology.edges.shape[0])
                added_count = edge_count_after - edge_count_before
                if not changed:
                    if selected_count < 2:
                        self._set_file_status(
                            "Add Rod Group needs at least 2 selected anchors. "
                            "Use Click Mode = `toggle` to multi-select.",
                            log_terminal=True,
                        )
                    else:
                        self._set_file_status(
                            "Selected anchors are already fully connected.",
                            log_terminal=True,
                        )
                else:
                    self._set_file_status(f"Added {added_count} rod group(s).", log_terminal=True)
                self.refresh()

        @remove_vertex_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                remove_selected_vertices_command(self.workspace, self.history)
                self.refresh()

        @remove_edge_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                remove_selected_edges_command(self.workspace, self.history)
                self.refresh()

        @center_button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            with self._lock:
                center_model_command(self.workspace, self.history)
                self.refresh()

        self._select_mode = select_mode

        with self.server.gui.add_folder("Playback"):
            step_once = self.server.gui.add_button("Step Once")
            reset_runtime = self.server.gui.add_button("Reset Runtime")
            self._status_markdown = self.server.gui.add_markdown("status")

            @step_once.on_click
            def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
                with self._lock:
                    step(self.workspace, n=int(step_count.value))
                    self.refresh()

            @reset_runtime.on_click
            def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
                with self._lock:
                    self.workspace.reset_runtime()
                    self.refresh()

        self._config_path_text = config_text
        self._example_path_text = example_text
        self._save_path_text = save_text
        self._edit_tools_folder = edit_tools_folder
        self._simulate_checkbox = simulate
        self._editing_checkbox = editing
        self._move_joint_checkbox = move_joint
        self._show_channel_checkbox = show_channel
        self._step_number = step_count
        self.renderer.on_vertex_click = self._on_vertex_click
        self.renderer.on_edge_click = self._on_edge_click

    def _on_vertex_click(self, index: int) -> None:
        """处理顶点点击事件。

        Args:
            index: 被点击的顶点索引。
        """
        with self._lock:
            mode = self._select_mode.value
            if mode == "add-child":
                before_edges = int(self.workspace.topology.edges.shape[0])
                added = add_joint_from_vertex(self.workspace, self.history, index)
                if added:
                    new_index = int(self.workspace.topology.vertices.shape[0] - 1)
                    auto_connected = int(self.workspace.topology.edges.shape[0]) > before_edges
                    self._set_file_status(
                        f"Added anchor {new_index} from anchor {index}."
                        + (" Auto-connected with a new rod group." if auto_connected else ""),
                        log_terminal=True,
                    )
                self.refresh()
                return
            select_vertex_by_mode(self.workspace, index=index, mode=mode)
            selected = np.flatnonzero(self.workspace.ui.vertex_status == 2).tolist()
            self._set_file_status(f"Selected anchors: {selected} (mode: {mode})")
            self.refresh()

    def _on_transform_update(self, index: int, position: np.ndarray) -> None:
        """接收变换控制器发出的位置变动事件并写入底层的绝对坐标中去。

        Args:
            index: 顶点索引。
            position: 目标坐标。
        """
        with self._lock:
            set_selected_vertex_position(self.workspace, index, position)
            self.refresh()

    def _on_transform_drag_start(self) -> None:
        with self._lock:
            self._is_transform_dragging = True
            self._drag_snapshot = self.workspace.snapshot()

    def _on_transform_drag_end(self) -> None:
        with self._lock:
            self._is_transform_dragging = False
            if self._drag_snapshot is None:
                return
            complete_drag_edit(self.workspace, self.history, self._drag_snapshot)
            self._drag_snapshot = None

    def _on_edge_click(self, index: int) -> None:
        """处理边点击后的选择逻辑。"""
        with self._lock:
            mode = self._select_mode.value
            select_edge_by_mode(self.workspace, index=index, mode=mode)
            selected = np.flatnonzero(self.workspace.ui.edge_status == 2).tolist()
            self._set_file_status(f"Selected rod groups: {selected} (mode: {mode})")
            self.refresh()

    def _simulation_loop(self) -> None:
        """后台循环：按条件推进仿真并刷新场景。"""
        while self._running:
            now = time.perf_counter()
            with self._lock:
                if self._should_step_simulation():
                    if self._last_simulation_tick_time is None:
                        self._last_simulation_tick_time = now
                    elapsed = max(0.0, now - self._last_simulation_tick_time)
                    self._last_simulation_tick_time = now
                    self._simulation_step_accumulator += elapsed * self.simulation_steps_per_second
                    n_steps = int(self._simulation_step_accumulator)
                    if n_steps > 0:
                        step(self.workspace, n=n_steps)
                        self._simulation_step_accumulator -= n_steps
                    self.renderer.render(self.workspace)
                    self._update_status()
                else:
                    self._simulation_step_accumulator = 0.0
                    self._last_simulation_tick_time = None
            time.sleep(1.0 / self.render_hz)

    def _load_workspace(self, *, config_path: Path, example_path: Path) -> None:
        """按给定路径加载工作区并同步 GUI。

        Args:
            config_path: 配置文件路径。
            example_path: 示例文件路径。
        """
        self.workspace = load_workspace_from_paths(
            config_path=config_path,
            example_path=example_path,
        )
        clear_workspace_history(self.history)
        self._drag_snapshot = None
        self.current_config_path = config_path
        self.current_example_path = example_path
        self._sync_gui_from_workspace()
        if self._config_path_text is not None:
            self._config_path_text.value = str(config_path)
        if self._example_path_text is not None:
            self._example_path_text.value = str(example_path)
        if self._save_path_text is not None:
            self._save_path_text.value = str(example_path)
        self.refresh()

    def _save_workspace(self, path: Path) -> None:
        """将当前工作区保存到对应格式的`JSON`文件。

        Args:
            path: 输出文件路径。
        """
        save_workspace_to_path(self.workspace, path)

    def _sync_gui_from_workspace(self) -> None:
        """将 GUI 控件值同步为当前工作区状态。"""
        if self._simulate_checkbox is not None:
            self._simulate_checkbox.value = self.workspace.ui.simulate
        if self._editing_checkbox is not None:
            self._editing_checkbox.value = self.workspace.ui.editing
        if self._move_joint_checkbox is not None:
            self._move_joint_checkbox.value = self.workspace.ui.moving_joint
        if self._show_channel_checkbox is not None:
            self._show_channel_checkbox.value = self.workspace.ui.show_channel
        if self._edit_tools_folder is not None:
            self._edit_tools_folder.visible = self.workspace.ui.editing

    def _set_file_status(self, message: str, *, log_terminal: bool = False) -> None:
        if self._file_status_markdown is not None:
            self._file_status_markdown.content = message
        if log_terminal:
            LOGGER.info(message)

    def _undo(self) -> None:
        if not undo(self.workspace, self.history):
            self._set_file_status("Nothing to undo.", log_terminal=True)
            return
        self._drag_snapshot = None
        self._is_transform_dragging = False
        self._sync_gui_from_workspace()
        self.refresh()
        self._set_file_status("Undo applied.", log_terminal=True)

    def _redo(self) -> None:
        if not redo(self.workspace, self.history):
            self._set_file_status("Nothing to redo.", log_terminal=True)
            return
        self._drag_snapshot = None
        self._is_transform_dragging = False
        self._sync_gui_from_workspace()
        self.refresh()
        self._set_file_status("Redo applied.", log_terminal=True)

    def _should_step_simulation(self) -> bool:
        if not self.workspace.ui.simulate:
            return False
        if not self.workspace.ui.editing:
            return True
        return self._is_transform_dragging and (
            self.workspace.ui.moving_joint or self.workspace.ui.moving_body
        )

    def _update_status(self) -> None:
        """更新状态面板文本。"""
        if self._status_markdown is None:
            return
        selected_vertex_indices = np.flatnonzero(self.workspace.ui.vertex_status == 2).tolist()
        selected_edge_indices = np.flatnonzero(self.workspace.ui.edge_status == 2).tolist()
        selected_vertices = len(selected_vertex_indices)
        selected_edges = len(selected_edge_indices)
        click_mode = self._select_mode.value if hasattr(self, "_select_mode") else "unknown"
        self._status_markdown.content = (
            f"**Anchors:** {self.workspace.topology.vertices.shape[0]}  \n"
            f"**Rod Groups:** {self.workspace.topology.edges.shape[0]}  \n"
            f"**Control Groups:** {self.workspace.script.num_channels}  \n"
            f"**Steps:** {self.workspace.physics.num_steps}  \n"
            f"**Simulation Rate:** {int(self.simulation_steps_per_second)} steps/s  \n"
            f"**Render Rate:** {int(self.render_hz)} Hz  \n"
            f"**Click Mode:** {click_mode}  \n"
            f"**Selected Anchors:** {selected_vertices}  \n"
            f"**Selected Anchor Indices:** {selected_vertex_indices}  \n"
            f"**Selected Rod Groups:** {selected_edges}  \n"
            f"**Selected Rod Group Indices:** {selected_edge_indices}"
        )


def create_app(
    *,
    config_path: str | Path,
    example_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> VgtrUiApp:
    """基于配置和示例文件创建 ``VgtrUiApp``。

    Args:
        config_path: 配置文件路径。
        example_path: 示例文件路径。
        host: 服务监听地址。
        port: 服务端口号。

    Returns:
        已初始化的 ``VgtrUiApp`` 实例。
    """
    workspace = load_workspace_from_paths(
        config_path=Path(config_path),
        example_path=Path(example_path),
    )
    server = viser.ViserServer(host=host, port=port)
    renderer = SceneRenderer(server=server)
    return VgtrUiApp(
        server=server,
        workspace=workspace,
        renderer=renderer,
        current_config_path=Path(config_path),
        current_example_path=Path(example_path),
    )
