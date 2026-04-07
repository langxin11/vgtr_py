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


def snapshots_equal(left: WorkspaceSnapshot, right: WorkspaceSnapshot) -> bool:
    """比较两个工作区快照是否完全一致。"""
    return (
        np.array_equal(left.topology.vertices, right.topology.vertices)
        and np.array_equal(left.topology.edges, right.topology.edges)
        and np.array_equal(left.topology.fixed_vs, right.topology.fixed_vs)
        and np.array_equal(left.topology.l_max, right.topology.l_max)
        and np.array_equal(left.topology.max_contraction, right.topology.max_contraction)
        and np.array_equal(left.topology.edge_channel, right.topology.edge_channel)
        and np.array_equal(left.topology.edge_active, right.topology.edge_active)
        and np.array_equal(left.physics.v0, right.physics.v0)
        and np.array_equal(left.physics.velocities, right.physics.velocities)
        and np.array_equal(left.physics.forces, right.physics.forces)
        and np.array_equal(left.physics.lengths, right.physics.lengths)
        and np.array_equal(left.physics.inflate_channel, right.physics.inflate_channel)
        and np.array_equal(left.physics.contraction_percent, right.physics.contraction_percent)
        and left.physics.num_steps == right.physics.num_steps
        and left.physics.i_action == right.physics.i_action
        and left.physics.i_action_prev == right.physics.i_action_prev
        and left.physics.record_frames == right.physics.record_frames
        and left.physics.frames == right.physics.frames
        and np.array_equal(left.script.script, right.script.script)
        and left.script.num_channels == right.script.num_channels
        and left.script.num_actions == right.script.num_actions
        and np.array_equal(left.ui.vertex_status, right.ui.vertex_status)
        and np.array_equal(left.ui.edge_status, right.ui.edge_status)
        and np.array_equal(left.ui.face_status, right.ui.face_status)
        and left.ui.editing == right.ui.editing
        and left.ui.moving_joint == right.ui.moving_joint
        and left.ui.moving_body == right.ui.moving_body
        and left.ui.show_channel == right.ui.show_channel
        and left.ui.simulate == right.ui.simulate
        and left.ui.record == right.ui.record
    )
