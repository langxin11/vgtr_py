"""Reward and info builders for runtime sessions."""

from __future__ import annotations

import numpy as np

from ..model import VGTRModel
from .state import RuntimeState


def compute_reward(
    model: VGTRModel,
    state: RuntimeState,
    action: np.ndarray | None = None,
) -> np.ndarray:
    tracking_error = np.zeros(state.num_envs, dtype=np.float64)
    if action is not None and model.projection_anchor_indices.size:
        action_array = np.asarray(action, dtype=np.float64)
        target_dim = int(model.projection_anchor_indices.shape[0] * 3)
        if action_array.ndim == 1:
            action_array = np.broadcast_to(action_array, (state.num_envs, target_dim))
        target = action_array.reshape(state.num_envs, model.projection_anchor_indices.shape[0], 3)
        current = state.qpos[:, model.projection_anchor_indices, :]
        tracking_error = np.linalg.norm(current - target, axis=2).mean(axis=1)

    overload_penalty = np.count_nonzero(state.rod_stalled, axis=1).astype(np.float64)
    passive_mask = model.rod_type == 0
    if np.any(passive_mask):
        passive_force_penalty = np.mean(np.abs(state.rod_axial_force[:, passive_mask]), axis=1)
    else:
        passive_force_penalty = np.zeros(state.num_envs, dtype=np.float64)
    if state.rod_strain.size:
        internal_stress_penalty = np.mean(np.abs(state.rod_strain), axis=1)
    else:
        internal_stress_penalty = np.zeros(state.num_envs, dtype=np.float64)
    return -(
        tracking_error
        + 0.1 * overload_penalty
        + 0.01 * passive_force_penalty
        + internal_stress_penalty
    )


def build_info(model: VGTRModel, state: RuntimeState) -> dict[str, np.ndarray]:
    tracking_error = np.zeros(state.num_envs, dtype=np.float64)
    if model.projection_anchor_indices.size:
        current = state.qpos[:, model.projection_anchor_indices, :]
        rest = model.anchor_rest_pos[model.projection_anchor_indices][None, :, :]
        tracking_error = np.linalg.norm(current - rest, axis=(1, 2))
    return {
        "rod_length": state.rod_length.copy(),
        "rod_axial_force": state.rod_axial_force.copy(),
        "rod_stalled": state.rod_stalled.copy(),
        "contact_mask": state.contact_mask.copy(),
        "tracking_error": tracking_error,
        "stalled_count": np.count_nonzero(state.rod_stalled, axis=1).astype(np.int64),
    }
