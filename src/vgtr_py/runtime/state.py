"""Env-major runtime state containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _readonly_view(array: np.ndarray) -> np.ndarray:
    view = array.view()
    view.setflags(write=False)
    return view


@dataclass(slots=True)
class RuntimeStateView:
    """Read-only view of a single environment."""

    env_index: int
    qpos: FloatArray
    qvel: FloatArray
    forces: FloatArray
    ctrl: FloatArray
    ctrl_target: FloatArray
    rod_ctrl: FloatArray
    rod_ctrl_target: FloatArray
    rod_length: FloatArray
    rod_target_length: FloatArray
    rod_target_override: FloatArray
    rod_axial_force: FloatArray
    rod_strain: FloatArray
    rod_stalled: BoolArray
    contact_mask: BoolArray
    time: float
    step_count: int
    i_action: int
    i_action_prev: int


@dataclass(slots=True)
class RuntimeState:
    """Unified env-major runtime state."""

    qpos: FloatArray
    qvel: FloatArray
    forces: FloatArray
    ctrl: FloatArray
    ctrl_target: FloatArray
    rod_ctrl: FloatArray
    rod_ctrl_target: FloatArray
    rod_length: FloatArray
    rod_target_length: FloatArray
    rod_target_override: FloatArray
    rod_axial_force: FloatArray
    rod_strain: FloatArray
    rod_stalled: BoolArray
    contact_mask: BoolArray
    time: FloatArray
    step_count: IntArray
    i_action: IntArray
    i_action_prev: IntArray
    record_frames: bool = False
    frames: list[list[list[list[float]]]] | None = None

    @property
    def num_envs(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def is_batched(self) -> bool:
        return self.num_envs > 1

    def env_view(self, index: int) -> RuntimeStateView:
        if index < 0 or index >= self.num_envs:
            raise IndexError(f"env index {index} out of range for {self.num_envs} envs")
        return RuntimeStateView(
            env_index=index,
            qpos=_readonly_view(self.qpos[index]),
            qvel=_readonly_view(self.qvel[index]),
            forces=_readonly_view(self.forces[index]),
            ctrl=_readonly_view(self.ctrl[index]),
            ctrl_target=_readonly_view(self.ctrl_target[index]),
            rod_ctrl=_readonly_view(self.rod_ctrl[index]),
            rod_ctrl_target=_readonly_view(self.rod_ctrl_target[index]),
            rod_length=_readonly_view(self.rod_length[index]),
            rod_target_length=_readonly_view(self.rod_target_length[index]),
            rod_target_override=_readonly_view(self.rod_target_override[index]),
            rod_axial_force=_readonly_view(self.rod_axial_force[index]),
            rod_strain=_readonly_view(self.rod_strain[index]),
            rod_stalled=_readonly_view(self.rod_stalled[index]),
            contact_mask=_readonly_view(self.contact_mask[index]),
            time=float(self.time[index]),
            step_count=int(self.step_count[index]),
            i_action=int(self.i_action[index]),
            i_action_prev=int(self.i_action_prev[index]),
        )
