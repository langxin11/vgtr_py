"""Allocation and reset helpers for unified runtime state."""

from __future__ import annotations

import numpy as np

from ..model import VGTRModel
from .state import RuntimeState


def _batch_rod_lengths(qpos: np.ndarray, rod_anchors: np.ndarray) -> np.ndarray:
    if rod_anchors.size == 0:
        return np.zeros((qpos.shape[0], 0), dtype=np.float64)
    diff = qpos[:, rod_anchors[:, 1], :] - qpos[:, rod_anchors[:, 0], :]
    return np.linalg.norm(diff, axis=2)


def _default_rod_ctrl(model: VGTRModel) -> np.ndarray:
    active_indices = model.active_rod_indices
    defaults = np.zeros(active_indices.shape[0], dtype=np.float64)
    if defaults.size == 0 or model.control_group_count == 0:
        return defaults
    groups = model.rod_control_group[active_indices]
    valid = (groups >= 0) & (groups < model.control_group_count)
    defaults[valid] = model.control_group_default_target[groups[valid]]
    return np.clip(defaults, 0.0, 1.0)


def make_state(
    model: VGTRModel,
    num_envs: int = 1,
    *,
    seed: int | None = None,
) -> RuntimeState:
    """Allocate a runtime state with env-major arrays."""
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    state = RuntimeState(
        qpos=np.broadcast_to(model.anchor_rest_pos, (num_envs, model.anchor_count, 3)).copy(),
        qvel=np.zeros((num_envs, model.anchor_count, 3), dtype=np.float64),
        forces=np.zeros((num_envs, model.anchor_count, 3), dtype=np.float64),
        ctrl=np.broadcast_to(
            model.control_group_default_target,
            (num_envs, model.control_group_count),
        ).copy(),
        ctrl_target=np.broadcast_to(
            model.control_group_default_target,
            (num_envs, model.control_group_count),
        ).copy(),
        rod_ctrl=np.broadcast_to(
            _default_rod_ctrl(model),
            (num_envs, model.active_rod_indices.shape[0]),
        ).copy(),
        rod_ctrl_target=np.broadcast_to(
            _default_rod_ctrl(model),
            (num_envs, model.active_rod_indices.shape[0]),
        ).copy(),
        rod_length=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_target_length=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_target_override=np.full((num_envs, model.rod_count), np.nan, dtype=np.float64),
        rod_axial_force=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_strain=np.zeros((num_envs, model.rod_count), dtype=np.float64),
        rod_stalled=np.zeros((num_envs, model.rod_count), dtype=np.bool_),
        contact_mask=np.zeros((num_envs, model.anchor_count), dtype=np.bool_),
        time=np.zeros(num_envs, dtype=np.float64),
        step_count=np.zeros(num_envs, dtype=np.int64),
        i_action=np.zeros(num_envs, dtype=np.int64),
        i_action_prev=np.zeros(num_envs, dtype=np.int64),
    )
    reset_state(model, state, seed=seed)
    return state


def reset_state(model: VGTRModel, state: RuntimeState, *, seed: int | None = None) -> None:
    """Reset a runtime state to the model rest configuration."""
    del seed
    num_envs = state.num_envs
    state.qpos[...] = model.anchor_rest_pos[None, :, :]
    state.qvel.fill(0.0)
    state.forces.fill(0.0)
    state.ctrl[...] = model.control_group_default_target[None, :]
    state.ctrl_target[...] = model.control_group_default_target[None, :]
    default_rod_ctrl = _default_rod_ctrl(model)
    state.rod_ctrl[...] = default_rod_ctrl[None, :]
    state.rod_ctrl_target[...] = default_rod_ctrl[None, :]
    state.rod_length[...] = _batch_rod_lengths(state.qpos, model.rod_anchors)
    state.rod_target_length[...] = model.rod_rest_length[None, :]
    state.rod_target_override.fill(np.nan)
    state.rod_axial_force.fill(0.0)
    state.rod_strain.fill(0.0)
    state.rod_stalled.fill(False)
    state.contact_mask.fill(False)
    state.time.fill(0.0)
    state.step_count.fill(0)
    state.i_action.fill(0)
    state.i_action_prev.fill(0)
    if state.record_frames:
        state.frames = []
    elif state.frames is None:
        state.frames = None
