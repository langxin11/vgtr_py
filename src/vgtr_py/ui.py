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
from .engine import step
from .history import WorkspaceHistory
from .rendering import SceneRenderer
from .workspace import Workspace, WorkspaceSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass
class VgtrUiApp:
    """VGTR 编辑与仿真界面应用入口类。

    该类通过集成 Viser Server 和后端 Workspace，
    提供图形界面操作、物理渲染控制和多线程步骤循环的胶水逻辑。
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
            self.workspace.ui.moving_anchor = False
            self.workspace.ui.moving_body = False
            if self._move_anchor_checkbox is not None:
                self._move_anchor_checkbox.value = False
        self._is_transform_dragging = False
        self._drag_snapshot = None
        if self._edit_tools_folder is not None:
            self._edit_tools_folder.visible = editing

    def _set_file_status(self, msg: str, *, log_terminal: bool = False) -> None:
        self._file_status_markdown.content = msg
        if log_terminal:
            print(msg)

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
        
        # 1. 场景与模型管理
        with self.server.gui.add_folder("Scene & Model"):
            load_button = self.server.gui.add_button("Reload / Load Model")
            save_text = self.server.gui.add_text("Export Path", str(self.current_example_path or ""))
            save_button = self.server.gui.add_button("Export Workspace")
            
            with self.server.gui.add_folder("History", expand_by_default=False):
                undo_button = self.server.gui.add_button("Undo")
                redo_button = self.server.gui.add_button("Redo")
            
            self._file_status_markdown = self.server.gui.add_markdown("Ready.")

        # 2. 实时仿真参数
        with self.server.gui.add_folder("Simulation Settings"):
            simulate = self.server.gui.add_checkbox("Simulate", self.workspace.ui.simulate)
            editing = self.server.gui.add_checkbox("Editing Mode", self.workspace.ui.editing)
            scripting = self.server.gui.add_checkbox("Enable Scripting", True)
            
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
            
            with self.server.gui.add_folder("Advanced Timestep"):
                simulation_rate = self.server.gui.add_number(
                    "Steps / Second", int(self.simulation_steps_per_second),
                    min=10, max=5000, step=50
                )
                render_rate = self.server.gui.add_dropdown(
                    "UI Render Hz", ("30", "60"), initial_value=str(int(self.render_hz))
                )

        # 3. 动态控制条容器 (Actuation)
        self._actuation_folder = self.server.gui.add_folder("Actuation / CG Control")

        # 4. 编辑工具
        edit_tools_folder = self.server.gui.add_folder("Editing Tools", visible=self.workspace.ui.editing)
        with edit_tools_folder:
            move_anchor = self.server.gui.add_checkbox("Enable Transform Gizmo", self.workspace.ui.moving_anchor)
            show_control_group = self.server.gui.add_checkbox("Show CG Colors", self.workspace.ui.show_control_group)
            
            with self.server.gui.add_folder("Selection"):
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

        # 5. 回放与状态
        with self.server.gui.add_folder("Playback"):
            step_count = self.server.gui.add_number("Manual steps", 8, min=1, max=200, step=1)
            step_once = self.server.gui.add_button("Step Once")
            reset_runtime = self.server.gui.add_button("Reset Physics State")
            self._status_markdown = self.server.gui.add_markdown("status")

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
        def _(_): self.workspace.config.k = k_slider.value
        @damping_slider.on_update
        def _(_): self.workspace.config.damping_ratio = damping_slider.value
        @friction_slider.on_update
        def _(_): self.workspace.config.friction_factor = friction_slider.value
        @ground_k_slider.on_update
        def _(_): self.workspace.config.ground_k = ground_k_slider.value
        @ground_d_slider.on_update
        def _(_): self.workspace.config.ground_d = ground_d_slider.value
        @gravity_toggle.on_update
        def _(_): self.workspace.config.gravity = gravity_toggle.value; self.refresh()

        @simulation_rate.on_update
        def _(_): self.simulation_steps_per_second = float(simulation_rate.value)
        @render_rate.on_update
        def _(_): self.render_hz = float(render_rate.value)

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
            with self._lock: clear_workspace_selection(self.workspace); self.refresh()
        @fix_button.on_click
        def _(_):
            with self._lock:
                if fix_selected(self.workspace, self.history, True):
                    self._set_file_status("Fixed selected anchors.", log_terminal=True)
                self.refresh()
        @unfix_button.on_click
        def _(_):
            with self._lock:
                if fix_selected(self.workspace, self.history, False):
                    self._set_file_status("Unfixed selected anchors.", log_terminal=True)
                self.refresh()

        @assign_cg_button.on_click
        def _(_):
            with self._lock:
                changed = assign_selected_rod_groups_control_group(self.workspace, self.history, control_group_index=int(cg_index.value))
                if changed: self._set_file_status(f"Assigned rods to CG {int(cg_index.value)}", log_terminal=True)
                self.refresh()

        @add_anchor_button.on_click
        def _(_):
            with self._lock:
                selected = np.flatnonzero(self.workspace.ui.anchor_status == 2).tolist()
                before_rods = int(self.workspace.topology.rod_anchors.shape[0])
                changed = add_joint_from_selection(self.workspace, self.history)
                if changed:
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
                connect_selected_anchors(self.workspace, self.history)
                self.refresh()

        @remove_anchor_button.on_click
        def _(_):
            with self._lock: remove_selected_anchors_command(self.workspace, self.history); self.refresh()

        @remove_rod_button.on_click
        def _(_):
            with self._lock: remove_selected_rod_groups_command(self.workspace, self.history); self.refresh()

        @center_button.on_click
        def _(_):
            with self._lock: center_model_command(self.workspace, self.history); self.refresh()

        @step_once.on_click
        def _(_):
            with self._lock: step(self.workspace, n=int(step_count.value)); self.refresh()

        @reset_runtime.on_click
        def _(_):
            with self._lock: self.workspace.reset_runtime(); self.refresh()

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
        
        self.renderer.on_anchor_click = self._on_anchor_click
        self.renderer.on_rod_group_click = self._on_rod_group_click
        self._select_mode = select_mode

    def _refresh_actuation_ui(self) -> None:
        """动态刷新控制组滑块。"""
        if self._actuation_folder is None:
            return
            
        # 清理旧滑块
        for s in self._control_sliders:
            s.remove()
        self._control_sliders.clear()
        
        # 获取当前控制组数量
        num_channels = self.workspace.script.num_channels
        if num_channels == 0:
            self._actuation_folder.visible = False
            return
            
        self._actuation_folder.visible = True
        with self._actuation_folder:
            for i in range(num_channels):
                # 创建滑块，闭包捕获索引 i
                slider = self.server.gui.add_slider(
                    f"CG {i}", min=0.0, max=1.0, step=0.01, 
                    initial_value=float(self.workspace.physics.control_group_target[i])
                )
                
                @slider.on_update
                def _(_, idx=i, s=slider):
                    with self._lock:
                        self.workspace.physics.control_group_target[idx] = s.value
                
                self._control_sliders.append(slider)

    def _on_anchor_click(self, index: int) -> None:
        with self._lock:
            mode = self._select_mode.value
            if mode == "add-child":
                before_rods = int(self.workspace.topology.rod_anchors.shape[0])
                added = add_joint_from_anchor(self.workspace, self.history, index)
                if added:
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
            self.refresh()

    def _on_transform_update(self, index: int, position: np.ndarray) -> None:
        with self._lock:
            set_selected_anchor_position(self.workspace, index, position)
            self.refresh()

    def _on_transform_drag_start(self) -> None:
        with self._lock:
            self._is_transform_dragging = True
            self._drag_snapshot = self.workspace.snapshot()

    def _on_transform_drag_end(self) -> None:
        with self._lock:
            self._is_transform_dragging = False
            if self._drag_snapshot is None: return
            complete_drag_edit(self.workspace, self.history, self._drag_snapshot)
            self._drag_snapshot = None

    def _on_rod_group_click(self, index: int) -> None:
        with self._lock:
            mode = self._select_mode.value
            select_rod_group_by_mode(self.workspace, index=index, mode=mode)
            selected = np.flatnonzero(self.workspace.ui.rod_group_status == 2).tolist()
            self._set_file_status(f"Selected rod groups: {selected}")
            self.refresh()

    def _simulation_loop(self) -> None:
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
                        # 仅在 Enable Scripting 开启时执行脚本
                        step(self.workspace, n=n_steps, scripting=self._scripting_toggle.value)
                        self._simulation_step_accumulator -= n_steps
                        
                        # 同步滑块位置（如果是脚本在驱动）
                        if self._scripting_toggle.value:
                            for i, s in enumerate(self._control_sliders):
                                s.value = float(self.workspace.physics.control_group_target[i])

                    self.renderer.render(self.workspace)
                    self._update_status()
                else:
                    self._simulation_step_accumulator = 0.0
                    self._last_simulation_tick_time = None
            time.sleep(1.0 / self.render_hz)

    def _load_workspace(self, *, config_path: Path | None, example_path: Path) -> None:
        self.workspace = load_workspace_from_paths(
            config_path=config_path,
            example_path=example_path,
        )
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
        
        self._status_markdown.content = (
            f"**Stat:** A={n_a}, R={n_r}, CG={n_c} | **Step:** {self.workspace.physics.num_steps}  \n"
            f"**Sel:** A={len(sel_a)} {sel_a if sel_a else ''}, R={len(sel_r)} {sel_r if sel_r else ''}"
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
    server = viser.ViserServer(host=host, port=port)
    renderer = SceneRenderer(server=server)
    return VgtrUiApp(
        server=server,
        workspace=workspace,
        renderer=renderer,
        current_config_path=Path(config_path) if config_path is not None else None,
        current_example_path=Path(example_path),
    )
