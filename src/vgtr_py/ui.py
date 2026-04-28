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
    add_joint_from_anchor,
    add_joint_from_selection,
    assign_selected_rod_groups_control_group,
    center_model_command,
    clear_workspace_history,
    clear_workspace_selection,
    complete_drag_edit,
    connect_selected_anchors,
    fix_selected,
    load_workspace_from_paths,
    redo,
    remove_selected_anchors_command,
    remove_selected_rod_groups_command,
    save_workspace_to_path,
    select_anchor_by_mode,
    select_rod_group_by_mode,
    set_selected_anchor_position,
    undo,
)
from .batch import BatchSimulator, BatchVGTRData, make_batch_data, reset_batch_data
from .data import VGTRData, make_data, reset_data
from .history import WorkspaceHistory
from .model import VGTRModel, compile_workspace
from .rendering import SceneRenderer
from .sim import Simulator, advance_script_targets
from .workspace import ROD_TYPE_ACTIVE, Workspace, WorkspaceSnapshot

LOGGER = logging.getLogger(__name__)
_MAX_CONTROL_SLIDERS = 60
_MAX_STATUS_SELECTION_ITEMS = 6


@dataclass
class VgtrUiApp:
    """VGTR 编辑与仿真界面应用入口类。

    该类通过集成 Viser Server 和后端 Workspace，
    提供图形界面操作、物理渲染控制和多线程步骤循环的胶水逻辑。
    """

    server: viser.ViserServer
    workspace: Workspace
    renderer: SceneRenderer
    model: VGTRModel | None = None
    data: VGTRData | None = None
    simulator: Simulator = field(default_factory=Simulator)
    render_hz: float = 60.0
    simulation_steps_per_second: float = 1000.0
    current_config_path: Path | None = None
    current_example_path: Path | None = None
    history: WorkspaceHistory = field(default_factory=WorkspaceHistory)

    _running: bool = False
    _lock: threading.RLock = threading.RLock()
    _simulation_thread: threading.Thread | None = None
    
    # 核心 UI 句柄缓存
    _simulate_checkbox: viser.GuiCheckboxHandle | None = None
    _editing_checkbox: viser.GuiCheckboxHandle | None = None
    _move_anchor_checkbox: viser.GuiCheckboxHandle | None = None
    _show_control_group_checkbox: viser.GuiCheckboxHandle | None = None
    _example_dropdown: viser.GuiDropdownHandle[str] | None = None
    _save_path_text: viser.GuiTextHandle | None = None
    _status_markdown: viser.GuiMarkdownHandle | None = None
    _file_status_markdown: viser.GuiMarkdownHandle | None = None
    _edit_tools_folder: viser.GuiFolderHandle | None = None
    _actuation_folder: viser.GuiFolderHandle | None = None
    _control_sliders: list[viser.GuiSliderHandle[float]] = field(default_factory=list)
    _actuation_mode_dropdown: viser.GuiDropdownHandle[str] | None = None
    _actuation_status_markdown: viser.GuiMarkdownHandle | None = None

    # 仿真参数句柄
    _k_slider: viser.GuiSliderHandle[float] | None = None
    _damping_slider: viser.GuiSliderHandle[float] | None = None
    _gravity_toggle: viser.GuiCheckboxHandle | None = None
    _friction_slider: viser.GuiSliderHandle[float] | None = None
    _ground_k_slider: viser.GuiSliderHandle[float] | None = None
    _ground_d_slider: viser.GuiSliderHandle[float] | None = None
    _scripting_toggle: viser.GuiCheckboxHandle | None = None

    _drag_snapshot: WorkspaceSnapshot | None = None
    _is_transform_dragging: bool = False
    _simulation_step_accumulator: float = 0.0
    _last_simulation_tick_time: float | None = None

    # 批量环境
    _batch_mode: bool = False
    _batch_data: BatchVGTRData | None = None
    _batch_simulator: BatchSimulator = field(default_factory=BatchSimulator)
    _batch_action: np.ndarray | None = None

    # 批量环境 UI 句柄
    _batch_mode_chk: viser.GuiCheckboxHandle | None = None
    _num_envs_input: viser.GuiNumberHandle | None = None
    _env_select_slider: viser.GuiSliderHandle[float] | None = None
    _hide_others_chk: viser.GuiCheckboxHandle | None = None
    _track_selected_chk: viser.GuiCheckboxHandle | None = None
    _spacing_slider: viser.GuiSliderHandle[float] | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = compile_workspace(self.workspace)
        if self.data is None:
            self.data = make_data(self.model)

    def start(self) -> None:
        """初始化场景和 GUI，并启动后台仿真循环。"""
        self.renderer.setup_scene()
        self.renderer.install_transform_callback(
            self._on_transform_update,
            drag_start=self._on_transform_drag_start,
            drag_end=self._on_transform_drag_end,
        )
        self._build_gui()
        self._refresh_actuation_ui() # 初次加载控制条
        self.refresh()
        self._running = True
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()

    def refresh(self) -> None:
        """立即刷新渲染和状态栏。"""
        with self._lock:
            if self._batch_mode and self._batch_data is not None:
                self.renderer.render_batch(
                    self.workspace,
                    batch_qpos=self._batch_data.qpos,
                    selected_env=int(self._env_select_slider.value) if self._env_select_slider is not None else 0,
                    show_only_selected=bool(self._hide_others_chk.value) if self._hide_others_chk is not None else False,
                    spacing=float(self._spacing_slider.value) if self._spacing_slider is not None else 3.0,
                    track_selected=bool(self._track_selected_chk.value) if self._track_selected_chk is not None else True,
                )
            else:
                self.renderer.render(self.workspace, anchor_pos=self._runtime_anchor_pos())
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
        """切换编辑模式并同步相关 UI 状态。

        Args:
            editing: 是否进入编辑模式。
            restore_pose: 进入编辑模式时是否重置运行时姿态。
            preserve_move_state: 是否保留当前的移动锚点/整体状态。
        """
        self.workspace.ui.editing = editing
        if editing and restore_pose:
            self._reset_runtime_state()
        if editing and getattr(self, "_select_mode", None) is not None:
            self._select_mode.value = "toggle"
        if not preserve_move_state:
            self.workspace.ui.moving_anchor = False
            self.workspace.ui.moving_body = False
            if self._move_anchor_checkbox is not None:
                self._move_anchor_checkbox.value = False
        self._is_transform_dragging = False
        self._drag_snapshot = None
        if self._edit_tools_folder is not None:
            self._edit_tools_folder.visible = editing

    def _set_file_status(self, msg: str, *, log_terminal: bool = False) -> None:
        """更新文件状态栏文本，并可选择同时输出到终端。

        Args:
            msg: 状态信息。
            log_terminal: 是否打印到终端。
        """
        self._file_status_markdown.content = msg
        if log_terminal:
            print(msg)

    def _runtime_anchor_pos(self) -> np.ndarray:
        """获取用于渲染的锚点坐标。

        编辑模式下返回拓扑原始位置，仿真模式下返回运行时数据位置。

        Returns:
            锚点坐标数组，形状 (anchor_count, 3)。
        """
        if self.data is None:
            return self.workspace.topology.anchor_pos
        return self.workspace.topology.anchor_pos if self.workspace.ui.editing else self.data.qpos

    def _format_index_summary(self, indices: list[int]) -> str:
        """格式化索引摘要，避免长列表淹没状态面板。"""
        if not indices:
            return "[]"
        if len(indices) <= _MAX_STATUS_SELECTION_ITEMS:
            return str(indices)
        head = indices[:_MAX_STATUS_SELECTION_ITEMS]
        return f"{head} +{len(indices) - len(head)} more"

    def _rebuild_runtime(self) -> None:
        """根据当前工作区重新编译仿真模型与数据。"""
        self.model = compile_workspace(self.workspace)
        if self._batch_mode:
            n = int(self._num_envs_input.value) if self._num_envs_input is not None else 4
            self._rebuild_batch_runtime(n)
        else:
            self.data = make_data(self.model)

    def _reset_runtime_state(self) -> None:
        """重置仿真数据到初始状态；若未初始化则先重建。"""
        if self.model is None:
            self._rebuild_runtime()
            return
        if self._batch_mode and self._batch_data is not None:
            reset_batch_data(self.model, self._batch_data)
            return
        if self.data is None:
            self._rebuild_runtime()
            return
        reset_data(self.model, self.data)

    def _run_simulation_steps(self, n_steps: int, *, scripting: bool) -> None:
        """在后台连续推进若干仿真步。

        Args:
            n_steps: 步数。
            scripting: 是否在此步进过程中推进脚本动作。
        """
        if self._batch_mode and self._batch_data is not None and self.model is not None:
            for _ in range(n_steps):
                self._batch_simulator.step(self.model, self._batch_data, self._batch_action)
            return
        if self.model is None or self.data is None or n_steps <= 0:
            return
        for _ in range(n_steps):
            if scripting:
                advance_script_targets(self.model, self.data)
            self.simulator.step(self.model, self.data)

    def _set_batch_mode(self, enabled: bool) -> None:
        """切换批量渲染模式。"""
        if enabled == self._batch_mode:
            return
        self._batch_mode = enabled
        if enabled:
            n = int(self._num_envs_input.value) if self._num_envs_input is not None else 4
            self._set_editing_state(False)
            if self._editing_checkbox is not None:
                self._editing_checkbox.value = False
                self._editing_checkbox.disabled = True
            if self._move_anchor_checkbox is not None:
                self._move_anchor_checkbox.value = False
                self._move_anchor_checkbox.disabled = True
            if self._edit_tools_folder is not None:
                self._edit_tools_folder.visible = False
            self._rebuild_batch_runtime(n)
        else:
            if self._editing_checkbox is not None:
                self._editing_checkbox.disabled = False
            if self._move_anchor_checkbox is not None:
                self._move_anchor_checkbox.disabled = False
            self._batch_data = None
            self._batch_action = None
            self.renderer._clear_batch_handles()
        self.refresh()

    def _rebuild_batch_runtime(self, num_envs: int) -> None:
        """根据当前 model 重建批量运行时。"""
        if self.model is None:
            return
        self._batch_data = make_batch_data(self.model, num_envs)
        self._batch_action = self._make_batch_action()
        if self._env_select_slider is not None:
            self._env_select_slider.max = float(max(num_envs - 1, 0))

    def _make_batch_action(self) -> np.ndarray:
        """为每个 env 生成批量动作。"""
        if self.model is None:
            return np.zeros((0,), dtype=np.float64)
        cg = self.model.control_group_count
        if self.model.projection_anchor_indices.size:
            dim = int(self.model.projection_anchor_indices.shape[0] * 3)
        elif cg:
            dim = cg
        else:
            dim = 1
        n = self._batch_data.num_envs if self._batch_data is not None else 1
        actions = np.zeros((n, dim), dtype=np.float64)
        if cg:
            actions[:, 0] = np.linspace(0.0, 1.0, n, dtype=np.float64)
        return actions

    def _build_gui(self) -> None:
        """构建优化后的参数化控制面板。"""
        
        # 稳健定位 configs 目录
        project_root = Path(__file__).resolve().parents[2]
        config_dir = project_root / "configs"
        if not config_dir.exists(): config_dir = Path("configs")
            
        example_files = sorted([f.name for f in config_dir.glob("*.json") if f.is_file()])
            
        example_dropdown = self.server.gui.add_dropdown(
            "Robot Model",
            tuple(example_files),
            initial_value=self.current_example_path.name if self.current_example_path and self.current_example_path.name in example_files else (example_files[0] if example_files else ""),
        )
        
        # 创建 Tab 标签页容器
        tab_group = self.server.gui.add_tab_group()
        
        # --- Tab 1: Editor (搭建与编辑) ---
        with tab_group.add_tab("Editor", icon="pencil"):
            # 1. 场景与模型管理
            with self.server.gui.add_folder("Scene & Model"):
                load_button = self.server.gui.add_button("Reload / Load Model")
                save_text = self.server.gui.add_text("Export Path", str(self.current_example_path or ""))
                save_button = self.server.gui.add_button("Export Workspace")
                
                with self.server.gui.add_folder("History", expand_by_default=False):
                    undo_button = self.server.gui.add_button("Undo")
                    redo_button = self.server.gui.add_button("Redo")
                
                self._file_status_markdown = self.server.gui.add_markdown("Ready.")

            # 2. 编辑模式与视图开关：固定可见，避免用户为了进入编辑模式先去 Physics tab。
            with self.server.gui.add_folder("Edit Mode"):
                editing = self.server.gui.add_checkbox("Editing Mode", self.workspace.ui.editing)
                move_anchor = self.server.gui.add_checkbox(
                    "Enable Transform Gizmo",
                    self.workspace.ui.moving_anchor,
                )
                show_control_group = self.server.gui.add_checkbox(
                    "Show CG Colors",
                    self.workspace.ui.show_control_group,
                )

            # 3. 编辑工具：只在编辑模式下显示。
            edit_tools_folder = self.server.gui.add_folder("Editing Tools", visible=self.workspace.ui.editing)
            with edit_tools_folder:
                with self.server.gui.add_folder("Selection"):
                    self.server.gui.add_markdown(
                        "Selection actions stay compact by design; large selections are summarized in the status bar."
                    )
                    clear_button = self.server.gui.add_button("Clear Selection")
                    fix_button = self.server.gui.add_button("Fix Selected")
                    unfix_button = self.server.gui.add_button("Unfix Selected")
                    with self.server.gui.add_folder("Assign Control Group"):
                        cg_index = self.server.gui.add_number("Target CG Index", 0, min=0, max=20, step=1)
                        assign_cg_button = self.server.gui.add_button("Apply to Selected Rods")

                with self.server.gui.add_folder("Topology"):
                    add_anchor_button = self.server.gui.add_button("Add Anchor")
                    connect_button = self.server.gui.add_button("Add Rod Group")
                    remove_rod_button = self.server.gui.add_button("Remove Rods")
                    remove_anchor_button = self.server.gui.add_button("Remove Anchors")
                    center_button = self.server.gui.add_button("Center Model")
                    select_mode = self.server.gui.add_dropdown("Click Mode", ("replace", "toggle", "add-child"), initial_value="toggle")

        # --- Tab 2: Physics (物理参数与步进) ---
        with tab_group.add_tab("Physics", icon="settings"):
            # 2. 实时仿真参数
            with self.server.gui.add_folder("Run Control"):
                simulate = self.server.gui.add_checkbox("Simulate", self.workspace.ui.simulate)
                simulation_rate = self.server.gui.add_number(
                    "Steps / Second", int(self.simulation_steps_per_second),
                    min=10, max=5000, step=50
                )
                render_rate = self.server.gui.add_dropdown(
                    "UI Render Hz", ("30", "60"), initial_value=str(int(self.render_hz))
                )

            with self.server.gui.add_folder("Solver Parameters"):
                k_slider = self.server.gui.add_slider(
                    "Rod Stiffness (k)", min=5000.0, max=50000.0, step=1000.0, 
                    initial_value=float(self.workspace.config.k)
                )
                damping_slider = self.server.gui.add_slider(
                    "Global Damping", min=0.5, max=1.0, step=0.01, 
                    initial_value=float(self.workspace.config.damping_ratio)
                )
                friction_slider = self.server.gui.add_slider(
                    "Coulomb Friction (mu)", min=0.0, max=2.0, step=0.05, 
                    initial_value=float(self.workspace.config.friction_factor)
                )
                
                with self.server.gui.add_folder("Ground Penalty (Spring)", expand_by_default=False):
                    ground_k_slider = self.server.gui.add_slider(
                        "Ground Stiffness", min=1e4, max=5e5, step=1e4, 
                        initial_value=float(self.workspace.config.ground_k)
                    )
                    ground_d_slider = self.server.gui.add_slider(
                        "Ground Damping", min=0.0, max=5000.0, step=50.0, 
                        initial_value=float(self.workspace.config.ground_d)
                    )
                
                gravity_toggle = self.server.gui.add_checkbox("Gravity", self.workspace.config.gravity)

            # 5. 回放与状态
            with self.server.gui.add_folder("Playback"):
                step_count = self.server.gui.add_number("Manual steps", 8, min=1, max=200, step=1)
                step_once = self.server.gui.add_button("Step Once")
                reset_runtime = self.server.gui.add_button("Reset Physics State")
                self._status_markdown = self.server.gui.add_markdown("status")

            # 批量环境
            with self.server.gui.add_folder("Environment", expand_by_default=False):
                batch_mode = self.server.gui.add_checkbox("Batch Mode", False)
                num_envs_input = self.server.gui.add_number(
                    "Num Envs", 4, min=1, max=256, step=1,
                )
                env_select = self.server.gui.add_slider(
                    "Select", 0, 3, step=1, initial_value=0,
                )
                hide_others = self.server.gui.add_checkbox("Hide others", False)
                track_selected = self.server.gui.add_checkbox("Track selected", True)
                spacing_slider = self.server.gui.add_slider(
                    "Grid Spacing", 1.0, 10.0, step=0.5, initial_value=3.0,
                )

        # --- Tab 3: Actuation (主动驱动测试) ---
        with tab_group.add_tab("Actuation", icon="device-gamepad-2"):
            # 3. 动态控制条容器 (Actuation)
            with self.server.gui.add_folder("Control Mode"):
                scripting = self.server.gui.add_checkbox("Enable Scripting", True)
                actuation_mode = self.server.gui.add_dropdown(
                    "Manual Mode",
                    ("control-groups", "per-rod"),
                    initial_value="control-groups",
                )
                self._actuation_status_markdown = self.server.gui.add_markdown("Ready.")
            self._actuation_folder = self.server.gui.add_folder("Actuation Controls")

        # --- 回调绑定 ---

        @load_button.on_click
        def _(_):
            with self._lock:
                try:
                    selected_path = config_dir / example_dropdown.value
                    self._load_workspace(config_path=self.current_config_path, example_path=selected_path)
                except Exception as exc:
                    self._set_file_status(f"Load failed: `{exc}`", log_terminal=True)
                else:
                    self._set_file_status(f"Loaded `{selected_path.name}`", log_terminal=True)

        @save_button.on_click
        def _(_):
            with self._lock:
                try:
                    self._save_workspace(Path(save_text.value).expanduser())
                except Exception as exc:
                    self._set_file_status(f"Export failed: `{exc}`", log_terminal=True)
                else:
                    self._set_file_status(f"Exported to `{save_text.value}`", log_terminal=True)

        @k_slider.on_update
        def _(_):
            self.workspace.config.k = k_slider.value
            self._rebuild_runtime()
        @damping_slider.on_update
        def _(_):
            self.workspace.config.damping_ratio = damping_slider.value
            self._rebuild_runtime()
        @friction_slider.on_update
        def _(_):
            self.workspace.config.friction_factor = friction_slider.value
            self._rebuild_runtime()
        @ground_k_slider.on_update
        def _(_):
            self.workspace.config.ground_k = ground_k_slider.value
            self._rebuild_runtime()
        @ground_d_slider.on_update
        def _(_):
            self.workspace.config.ground_d = ground_d_slider.value
            self._rebuild_runtime()
        @gravity_toggle.on_update
        def _(_):
            self.workspace.config.gravity = gravity_toggle.value
            self._rebuild_runtime()
            self.refresh()

        @simulation_rate.on_update
        def _(_): self.simulation_steps_per_second = float(simulation_rate.value)
        @render_rate.on_update
        def _(_): self.render_hz = float(render_rate.value)
        @actuation_mode.on_update
        def _(_):
            with self._lock:
                if self.data is not None and actuation_mode.value == "control-groups":
                    self.data.rod_target_override.fill(np.nan)
                self._refresh_actuation_ui()
                self.refresh()

        @undo_button.on_click
        def _(_): 
            with self._lock: self._undo()
        @redo_button.on_click
        def _(_): 
            with self._lock: self._redo()

        @simulate.on_update
        def _(_):
            with self._lock: self.workspace.ui.simulate = simulate.value; self.refresh()

        @editing.on_update
        def _(_):
            with self._lock: self._set_editing_state(editing.value, restore_pose=editing.value); self.refresh()

        @move_anchor.on_update
        def _(_):
            with self._lock:
                self.workspace.ui.moving_anchor = move_anchor.value and self.workspace.ui.editing
                if move_anchor.value and not self.workspace.ui.editing:
                    self._set_editing_state(True, restore_pose=True, preserve_move_state=True)
                    editing.value = True
                self.refresh()

        @show_control_group.on_update
        def _(_):
            with self._lock: self.workspace.ui.show_control_group = show_control_group.value; self.refresh()

        @clear_button.on_click
        def _(_):
            with self._lock:
                clear_workspace_selection(self.workspace)
                self._refresh_actuation_ui()
                self.refresh()
        @fix_button.on_click
        def _(_):
            with self._lock:
                if fix_selected(self.workspace, self.history, True):
                    self._rebuild_runtime()
                    self._set_file_status("Fixed selected anchors.", log_terminal=True)
                self.refresh()
        @unfix_button.on_click
        def _(_):
            with self._lock:
                if fix_selected(self.workspace, self.history, False):
                    self._rebuild_runtime()
                    self._set_file_status("Unfixed selected anchors.", log_terminal=True)
                self.refresh()

        @assign_cg_button.on_click
        def _(_):
            with self._lock:
                changed = assign_selected_rod_groups_control_group(self.workspace, self.history, control_group_index=int(cg_index.value))
                if changed:
                    self._rebuild_runtime()
                    self._set_file_status(f"Assigned rods to CG {int(cg_index.value)}", log_terminal=True)
                self.refresh()

        @add_anchor_button.on_click
        def _(_):
            with self._lock:
                selected = np.flatnonzero(self.workspace.ui.anchor_status == 2).tolist()
                before_rods = int(self.workspace.topology.rod_anchors.shape[0])
                changed = add_joint_from_selection(self.workspace, self.history)
                if changed:
                    self._rebuild_runtime()
                    new_index = int(self.workspace.topology.anchor_pos.shape[0] - 1)
                    auto_connected = int(self.workspace.topology.rod_anchors.shape[0]) > before_rods
                    msg = f"Added anchor {new_index}"
                    if len(selected) == 1: msg += f" from anchor {selected[0]}"
                    if auto_connected: msg += " (Auto-rod)"
                    self._set_file_status(msg, log_terminal=True)
                self.refresh()

        @connect_button.on_click
        def _(_):
            with self._lock:
                if connect_selected_anchors(self.workspace, self.history):
                    self._rebuild_runtime()
                self.refresh()

        @remove_anchor_button.on_click
        def _(_):
            with self._lock:
                if remove_selected_anchors_command(self.workspace, self.history):
                    self._rebuild_runtime()
                self.refresh()

        @remove_rod_button.on_click
        def _(_):
            with self._lock:
                if remove_selected_rod_groups_command(self.workspace, self.history):
                    self._rebuild_runtime()
                self.refresh()

        @center_button.on_click
        def _(_):
            with self._lock:
                if center_model_command(self.workspace, self.history):
                    self._rebuild_runtime()
                self.refresh()

        @step_once.on_click
        def _(_):
            with self._lock:
                self._run_simulation_steps(int(step_count.value), scripting=bool(self._scripting_toggle.value))
                self.refresh()

        @reset_runtime.on_click
        def _(_):
            with self._lock:
                self._reset_runtime_state()
                self.refresh()

        # 保存句柄
        self._example_dropdown = example_dropdown
        self._save_path_text = save_text
        self._edit_tools_folder = edit_tools_folder
        self._simulate_checkbox = simulate
        self._editing_checkbox = editing
        self._move_anchor_checkbox = move_anchor
        self._show_control_group_checkbox = show_control_group
        self._step_number = step_count
        self._k_slider = k_slider
        self._damping_slider = damping_slider
        self._gravity_toggle = gravity_toggle
        self._friction_slider = friction_slider
        self._scripting_toggle = scripting
        self._actuation_mode_dropdown = actuation_mode
        
        self.renderer.on_anchor_click = self._on_anchor_click
        self.renderer.on_rod_group_click = self._on_rod_group_click
        self._select_mode = select_mode

        # 批量环境句柄
        self._batch_mode_chk = batch_mode
        self._num_envs_input = num_envs_input
        self._env_select_slider = env_select
        self._hide_others_chk = hide_others
        self._track_selected_chk = track_selected
        self._spacing_slider = spacing_slider

        # --- 批量环境回调 ---

        @batch_mode.on_update
        def _(checked: bool) -> None:
            with self._lock:
                self._set_batch_mode(checked)

        @num_envs_input.on_update
        def _(_):
            with self._lock:
                n = int(num_envs_input.value)
                self._env_select_slider.max = float(max(n - 1, 0))  # type: ignore[union-attr]
                if self._batch_mode:
                    self._rebuild_batch_runtime(n)
                    self.refresh()

        @env_select.on_update
        def _(_):
            if self._batch_mode:
                self.refresh()

        @hide_others.on_update
        def _(_):
            if self._batch_mode:
                self.refresh()

        @track_selected.on_update
        def _(_):
            if self._batch_mode:
                self.refresh()

        @spacing_slider.on_update
        def _(_):
            if self._batch_mode:
                self.refresh()

    def _refresh_actuation_ui(self) -> None:
        """动态刷新驱动控制 UI，根据模式展示控制组或按杆件滑块。"""
        if self._actuation_folder is None:
            return
            
        # 清理旧滑块
        for s in self._control_sliders:
            s.remove()
        self._control_sliders.clear()
        if self._actuation_status_markdown is not None:
            self._actuation_status_markdown.content = "Ready."

        if self.data is None or self.model is None:
            self._actuation_folder.visible = False
            return

        mode = (
            self._actuation_mode_dropdown.value
            if self._actuation_mode_dropdown is not None
            else "control-groups"
        )

        self._actuation_folder.visible = True
        if mode == "per-rod":
            self._build_per_rod_sliders()
            return
        self._build_control_group_sliders()

    def _build_control_group_sliders(self) -> None:
        """渲染控制组滑块，并在数量过大时降级为摘要提示。"""
        if self.data is None:
            return
        num_channels = self.workspace.script.num_channels
        if num_channels == 0:
            self._actuation_folder.visible = False
            if self._actuation_status_markdown is not None:
                self._actuation_status_markdown.content = "No control groups available."
            return
        if num_channels > _MAX_CONTROL_SLIDERS:
            if self._actuation_status_markdown is not None:
                self._actuation_status_markdown.content = (
                    f"Showing summary only: {num_channels} control groups exceed slider limit "
                    f"of {_MAX_CONTROL_SLIDERS}. Switch to `per-rod` only when editing a smaller model."
                )
            return
        with self._actuation_folder:
            for i in range(num_channels):
                slider = self.server.gui.add_slider(
                    f"CG {i}",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(self.data.ctrl_target[i]),
                )

                @slider.on_update
                def _(_, idx=i, s=slider):
                    with self._lock:
                        if self.data is not None:
                            self.data.ctrl_target[idx] = s.value
                            if self.data.rod_target_override.shape[0]:
                                self.data.rod_target_override.fill(np.nan)

                self._control_sliders.append(slider)

    def _build_per_rod_sliders(self) -> None:
        """渲染按杆件直接控制的滑块。"""
        if self.data is None or self.model is None:
            return
        active_indices = np.flatnonzero(self.model.rod_type == ROD_TYPE_ACTIVE).tolist()
        if not active_indices:
            self._actuation_folder.visible = False
            if self._actuation_status_markdown is not None:
                self._actuation_status_markdown.content = "No active rods available for direct control."
            return

        slider_indices = active_indices
        if len(slider_indices) > _MAX_CONTROL_SLIDERS:
            selected = np.flatnonzero(self.workspace.ui.rod_group_status == 2).tolist()
            selected_active = [index for index in selected if index in active_indices]
            if selected_active:
                slider_indices = selected_active[:_MAX_CONTROL_SLIDERS]
                if self._actuation_status_markdown is not None:
                    self._actuation_status_markdown.content = (
                        f"Too many active rods ({len(active_indices)}). Showing selected rods only: "
                        f"{self._format_index_summary(selected_active)}."
                    )
            else:
                if self._actuation_status_markdown is not None:
                    self._actuation_status_markdown.content = (
                        f"Too many active rods ({len(active_indices)}). Select rods in the Editor tab "
                        "to expose direct sliders."
                    )
                return

        with self._actuation_folder:
            for rod_index in slider_indices:
                lo, hi = self.model.rod_length_limits[rod_index]
                current = float(
                    self.data.rod_target_override[rod_index]
                    if np.isfinite(self.data.rod_target_override[rod_index])
                    else self.data.rod_target_length[rod_index]
                )
                
                # Convert current length to 0-1 normalized value: 0=min_length(lo), 1=max_length(hi)
                current_norm = 0.0 if hi <= lo else (np.clip(current, lo, hi) - lo) / (hi - lo)
                
                slider = self.server.gui.add_slider(
                    f"Rod {rod_index}",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=float(current_norm),
                )

                @slider.on_update
                def _(_, idx=rod_index, s=slider, min_l=float(lo), max_l=float(hi)):
                    with self._lock:
                        if self.data is not None:
                            self.data.rod_target_override[idx] = min_l + s.value * (max_l - min_l)

                self._control_sliders.append(slider)

    def _on_anchor_click(self, index: int) -> None:
        """锚点图元点击回调：根据选择模式执行选中或添加子节点。

        Args:
            index: 被点击的锚点索引。
        """
        with self._lock:
            mode = self._select_mode.value
            if mode == "add-child":
                before_rods = int(self.workspace.topology.rod_anchors.shape[0])
                added = add_joint_from_anchor(self.workspace, self.history, index)
                if added:
                    self._rebuild_runtime()
                    new_index = int(self.workspace.topology.anchor_pos.shape[0] - 1)
                    auto_connected = int(self.workspace.topology.rod_anchors.shape[0]) > before_rods
                    msg = f"Added anchor {new_index} from {index}"
                    if auto_connected:
                        msg += " (Auto-rod)"
                    self._set_file_status(msg, log_terminal=True)
                self.refresh()
                return
            select_anchor_by_mode(self.workspace, index=index, mode=mode)
            selected = np.flatnonzero(self.workspace.ui.anchor_status == 2).tolist()
            self._set_file_status(f"Selected anchors: {selected}")
            self._refresh_actuation_ui()
            self.refresh()

    def _on_transform_update(self, index: int, position: np.ndarray) -> None:
        """变换控件拖拽更新回调：同步锚点位置并重建运行时模型。

        Args:
            index: 被拖拽的锚点索引。
            position: 新的三维坐标。
        """
        with self._lock:
            set_selected_anchor_position(self.workspace, index, position)
            self._rebuild_runtime()
            self.refresh()

    def _on_transform_drag_start(self) -> None:
        """变换控件拖拽开始回调：记录快照以便撤销。"""
        with self._lock:
            self._is_transform_dragging = True
            self._drag_snapshot = self.workspace.snapshot()

    def _on_transform_drag_end(self) -> None:
        """变换控件拖拽结束回调：将拖拽操作提交到历史栈。"""
        with self._lock:
            self._is_transform_dragging = False
            if self._drag_snapshot is None: return
            complete_drag_edit(self.workspace, self.history, self._drag_snapshot)
            self._drag_snapshot = None

    def _on_rod_group_click(self, index: int) -> None:
        """杆组图元点击回调：根据选择模式执行选中。

        Args:
            index: 被点击的杆组索引。
        """
        with self._lock:
            mode = self._select_mode.value
            select_rod_group_by_mode(self.workspace, index=index, mode=mode)
            selected = np.flatnonzero(self.workspace.ui.rod_group_status == 2).tolist()
            self._set_file_status(f"Selected rod groups: {selected}")
            self._refresh_actuation_ui()
            self.refresh()

    def _simulation_loop(self) -> None:
        """后台仿真主循环：按渲染帧率积累时间并推进物理步数。"""
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
                        self._run_simulation_steps(
                            n_steps,
                            scripting=bool(self._scripting_toggle.value),
                        )
                        self._simulation_step_accumulator -= n_steps

                        # 同步滑块位置（如果是脚本在驱动）
                        if (
                            self._scripting_toggle.value
                            and self.data is not None
                            and (
                                self._actuation_mode_dropdown is None
                                or self._actuation_mode_dropdown.value == "control-groups"
                            )
                        ):
                            for i, s in enumerate(self._control_sliders):
                                s.value = float(self.data.ctrl_target[i])

                    if self._batch_mode and self._batch_data is not None:
                        self.renderer.render_batch(
                            self.workspace,
                            batch_qpos=self._batch_data.qpos,
                            selected_env=int(self._env_select_slider.value) if self._env_select_slider is not None else 0,
                            show_only_selected=bool(self._hide_others_chk.value) if self._hide_others_chk is not None else False,
                            spacing=float(self._spacing_slider.value) if self._spacing_slider is not None else 3.0,
                            track_selected=bool(self._track_selected_chk.value) if self._track_selected_chk is not None else True,
                        )
                    else:
                        self.renderer.render(self.workspace, anchor_pos=self._runtime_anchor_pos())
                    self._update_status()
                else:
                    self._simulation_step_accumulator = 0.0
                    self._last_simulation_tick_time = None
            time.sleep(1.0 / self.render_hz)

    def _load_workspace(self, *, config_path: Path | None, example_path: Path) -> None:
        """加载新工作区并重置所有运行时状态与 UI。

        Args:
            config_path: 配置文件路径。
            example_path: 示例/模型文件路径。
        """
        self.workspace = load_workspace_from_paths(
            config_path=config_path,
            example_path=example_path,
        )
        self._rebuild_runtime()
        clear_workspace_history(self.history)
        self._drag_snapshot = None
        self.current_config_path = config_path
        self.current_example_path = example_path
        self._sync_gui_from_workspace()
        self._refresh_actuation_ui() # 加载新模型后刷新控制条

        if self._example_dropdown is not None:
            if example_path.name in self._example_dropdown.options:
                self._example_dropdown.value = example_path.name
        if self._save_path_text is not None:
            self._save_path_text.value = str(example_path)
        self.refresh()

    def _save_workspace(self, path: Path) -> None:
        """将当前工作区导出到指定路径。

        Args:
            path: 目标文件路径。
        """
        save_workspace_to_path(self.workspace, path)

    def _sync_gui_from_workspace(self) -> None:
        """将 GUI 控件同步为当前工作区状态。"""
        if self._simulate_checkbox is not None:
            self._simulate_checkbox.value = self.workspace.ui.simulate
        if self._editing_checkbox is not None:
            self._editing_checkbox.value = self.workspace.ui.editing
        if self._move_anchor_checkbox is not None:
            self._move_anchor_checkbox.value = self.workspace.ui.moving_anchor
        if self._show_control_group_checkbox is not None:
            self._show_control_group_checkbox.value = self.workspace.ui.show_control_group
        if self._edit_tools_folder is not None:
            self._edit_tools_folder.visible = self.workspace.ui.editing
        if self._batch_mode:
            if self._editing_checkbox is not None:
                self._editing_checkbox.value = False
                self._editing_checkbox.disabled = True
            if self._move_anchor_checkbox is not None:
                self._move_anchor_checkbox.value = False
                self._move_anchor_checkbox.disabled = True
            if self._edit_tools_folder is not None:
                self._edit_tools_folder.visible = False
        else:
            if self._editing_checkbox is not None:
                self._editing_checkbox.disabled = False
            if self._move_anchor_checkbox is not None:
                self._move_anchor_checkbox.disabled = False

        # 同步物理滑块
        if self._k_slider is not None:
            self._k_slider.value = float(self.workspace.config.k)
        if self._damping_slider is not None:
            self._damping_slider.value = float(self.workspace.config.damping_ratio)
        if self._friction_slider is not None:
            self._friction_slider.value = float(self.workspace.config.friction_factor)
        if self._ground_k_slider is not None:
            self._ground_k_slider.value = float(self.workspace.config.ground_k)
        if self._ground_d_slider is not None:
            self._ground_d_slider.value = float(self.workspace.config.ground_d)
        if self._gravity_toggle is not None:
            self._gravity_toggle.value = self.workspace.config.gravity

    def _undo(self) -> None:
        """执行撤销：恢复上一个历史快照并重建运行时模型。"""
        if not undo(self.workspace, self.history):
            self._set_file_status("Nothing to undo.", log_terminal=True)
            return
        self._rebuild_runtime()
        self._drag_snapshot = None
        self._is_transform_dragging = False
        self._sync_gui_from_workspace()
        self.refresh()
        self._set_file_status("Undo applied.", log_terminal=True)

    def _redo(self) -> None:
        """执行重做：恢复下一个历史快照并重建运行时模型。"""
        if not redo(self.workspace, self.history):
            self._set_file_status("Nothing to redo.", log_terminal=True)
            return
        self._rebuild_runtime()
        self._drag_snapshot = None
        self._is_transform_dragging = False
        self._sync_gui_from_workspace()
        self.refresh()
        self._set_file_status("Redo applied.", log_terminal=True)

    def _should_step_simulation(self) -> bool:
        """判断当前是否应当推进物理仿真。

        Returns:
            当仿真开启且非编辑模式，或在编辑模式下正在拖拽锚点/整体时为 True。
            批量模式下，simulate 为 True 即步进。
        """
        if not self.workspace.ui.simulate:
            return False
        if self._batch_mode:
            return True
        if not self.workspace.ui.editing:
            return True
        return self._is_transform_dragging and (
            self.workspace.ui.moving_anchor or self.workspace.ui.moving_body
        )

    def _update_status(self) -> None:
        """更新状态面板文本。"""
        if self._status_markdown is None: return
        n_a = self.workspace.topology.anchor_pos.shape[0]
        n_r = self.workspace.topology.rod_anchors.shape[0]
        n_c = self.workspace.script.num_channels
        sel_a = np.flatnonzero(self.workspace.ui.anchor_status == 2).tolist()
        sel_r = np.flatnonzero(self.workspace.ui.rod_group_status == 2).tolist()
        step_count = self.data.step_count if self.data is not None else 0
        sim_state = "running" if self.workspace.ui.simulate else "paused"
        effective_hz = int(self.simulation_steps_per_second) if self.workspace.ui.simulate else 0
        realtime_factor = self.simulation_steps_per_second * self.workspace.config.h
        ctrl_summary = "[]"
        if self.data is not None and self.data.ctrl_target.size:
            ctrl_values = [round(float(v), 3) for v in self.data.ctrl_target[: min(4, self.data.ctrl_target.size)]]
            ctrl_summary = f"{ctrl_values}"
            if self.data.ctrl_target.size > 4:
                ctrl_summary += f" +{self.data.ctrl_target.size - 4} more"

        self._status_markdown.content = (
            f"**State:** {sim_state} | **Step:** {step_count} | **Rate:** {effective_hz} steps/s | "
            f"**Realtime:** {realtime_factor:.2f}x  \n"
            f"**Model:** A={n_a}, R={n_r}, CG={n_c} | **Ctrl Target:** {ctrl_summary}  \n"
            f"**Selection:** A={len(sel_a)} {self._format_index_summary(sel_a)}, "
            f"R={len(sel_r)} {self._format_index_summary(sel_r)}"
        )


def create_app(
    *,
    config_path: str | Path | None,
    example_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> VgtrUiApp:
    """基于配置和示例文件创建 ``VgtrUiApp``。"""
    workspace = load_workspace_from_paths(
        config_path=Path(config_path) if config_path is not None else None,
        example_path=Path(example_path),
    )
    model = compile_workspace(workspace)
    data = make_data(model)
    server = viser.ViserServer(host=host, port=port)
    renderer = SceneRenderer(server=server)
    return VgtrUiApp(
        server=server,
        workspace=workspace,
        renderer=renderer,
        model=model,
        data=data,
        current_config_path=Path(config_path) if config_path is not None else None,
        current_example_path=Path(example_path),
    )
