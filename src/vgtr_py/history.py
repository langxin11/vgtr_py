"""历史状态与撤销重做模块。

实现了基于工作区快照栈（Snapshot Stack）的历史记录追踪系统，确保可对操作进行回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .workspace import Workspace, WorkspaceSnapshot


@dataclass
class WorkspaceHistory:
    """基于完整快照的撤销/重做历史。"""

    max_entries: int = 100
    _undo_stack: list[WorkspaceSnapshot] = field(default_factory=list)
    _redo_stack: list[WorkspaceSnapshot] = field(default_factory=list)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def push(self, snapshot: WorkspaceSnapshot) -> None:
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self.max_entries:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, workspace: Workspace) -> bool:
        if not self._undo_stack:
            return False
        current = workspace.snapshot()
        workspace.restore(self._undo_stack.pop())
        self._redo_stack.append(current)
        return True

    def redo(self, workspace: Workspace) -> bool:
        if not self._redo_stack:
            return False
        current = workspace.snapshot()
        workspace.restore(self._redo_stack.pop())
        self._undo_stack.append(current)
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    @property
    def undo_stack_size(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_stack_size(self) -> int:
        return len(self._redo_stack)


def snapshots_equal(left: WorkspaceSnapshot, right: WorkspaceSnapshot) -> bool:
    """比较两个工作区快照是否完全一致。"""
    return (
        np.array_equal(left.topology.anchor_pos, right.topology.anchor_pos)
        and np.array_equal(left.topology.rod_anchors, right.topology.rod_anchors)
        and np.array_equal(left.topology.anchor_fixed, right.topology.anchor_fixed)
        and np.array_equal(left.topology.rod_rest_length, right.topology.rod_rest_length)
        and np.array_equal(left.topology.rod_min_length, right.topology.rod_min_length)
        and np.array_equal(left.topology.rod_control_group, right.topology.rod_control_group)
        and np.array_equal(left.topology.rod_enabled, right.topology.rod_enabled)
        and np.array_equal(left.topology.rod_actuated, right.topology.rod_actuated)
        and np.array_equal(left.topology.rod_group_mass, right.topology.rod_group_mass)
        and np.array_equal(left.topology.rod_radius, right.topology.rod_radius)
        and np.array_equal(left.topology.rod_sleeve_half, right.topology.rod_sleeve_half)
        and np.array_equal(left.physics.v0, right.physics.v0)
        and np.array_equal(left.physics.velocities, right.physics.velocities)
        and np.array_equal(left.physics.forces, right.physics.forces)
        and np.array_equal(left.physics.lengths, right.physics.lengths)
        and np.array_equal(left.physics.control_group_target, right.physics.control_group_target)
        and np.array_equal(left.physics.control_group_value, right.physics.control_group_value)
        and left.physics.num_steps == right.physics.num_steps
        and left.physics.i_action == right.physics.i_action
        and left.physics.i_action_prev == right.physics.i_action_prev
        and left.physics.record_frames == right.physics.record_frames
        and left.physics.frames == right.physics.frames
        and np.array_equal(left.script.script, right.script.script)
        and left.script.num_channels == right.script.num_channels
        and left.script.num_actions == right.script.num_actions
        and np.array_equal(left.ui.anchor_status, right.ui.anchor_status)
        and np.array_equal(left.ui.rod_group_status, right.ui.rod_group_status)
        and np.array_equal(left.ui.face_status, right.ui.face_status)
        and left.ui.editing == right.ui.editing
        and left.ui.moving_anchor == right.ui.moving_anchor
        and left.ui.moving_body == right.ui.moving_body
        and left.ui.show_control_group == right.ui.show_control_group
        and left.ui.simulate == right.ui.simulate
        and left.ui.record == right.ui.record
    )
