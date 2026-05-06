"""Projection helpers for projection-control sessions."""

from __future__ import annotations

import numpy as np

from ..model import VGTRModel
from .state import RuntimeState


def _reshape_projection_targets(
    model: VGTRModel,
    state: RuntimeState,
    target_positions: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_positions, dtype=np.float64)
    target_dim = int(model.projection_anchor_indices.shape[0] * 3)
    expected = (state.num_envs, target_dim)
    if target.ndim == 1:
        if target.shape[0] != target_dim:
            raise ValueError(
                f"expected projection target with shape ({target_dim},), got {target.shape}"
            )
        target = np.broadcast_to(target, expected)
    if target.shape != expected:
        raise ValueError(f"expected projection target with shape {expected}, got {target.shape}")
    return target.reshape(state.num_envs, model.projection_anchor_indices.shape[0], 3)


def project_anchor_targets(
    model: VGTRModel,
    state: RuntimeState,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Project anchor targets into env-major per-active-rod control targets."""
    active_indices = model.active_rod_indices
    if model.projection_anchor_indices.size == 0 or active_indices.size == 0:
        return np.zeros((state.num_envs, active_indices.shape[0]), dtype=np.float64)

    target = _reshape_projection_targets(model, state, target_positions)
    projected_qpos = state.qpos.copy()
    projected_qpos[:, model.projection_anchor_indices, :] = target

    rod_target = np.zeros((state.num_envs, active_indices.shape[0]), dtype=np.float64)

    for active_col, rod_index in enumerate(active_indices):
        anchor_pair = model.rod_anchors[rod_index]
        desired_length = np.linalg.norm(
            projected_qpos[:, anchor_pair[1], :] - projected_qpos[:, anchor_pair[0], :],
            axis=1,
        )
        min_length, max_length = model.rod_length_limits[rod_index]
        if max_length <= min_length + 1e-9:
            normalized = np.zeros(state.num_envs, dtype=np.float64)
        else:
            normalized = (desired_length - min_length) / (max_length - min_length)
        rod_target[:, active_col] = np.clip(normalized, 0.0, 1.0)

    return rod_target
