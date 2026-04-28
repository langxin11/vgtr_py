"""Projection helpers for projection-control sessions."""

from __future__ import annotations

import numpy as np

from ..model import VGTRModel
from ..workspace import ROD_TYPE_ACTIVE
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
    """Project anchor targets into env-major control targets."""
    if model.projection_anchor_indices.size == 0 or model.control_group_count == 0:
        return np.zeros((state.num_envs, model.control_group_count), dtype=np.float64)

    target = _reshape_projection_targets(model, state, target_positions)
    projected_qpos = state.qpos.copy()
    projected_qpos[:, model.projection_anchor_indices, :] = target

    control_target = state.ctrl_target.copy()
    contributions = np.zeros((state.num_envs, model.control_group_count), dtype=np.float64)
    counts = np.zeros(model.control_group_count, dtype=np.int32)

    for rod_index in np.flatnonzero(model.rod_type == ROD_TYPE_ACTIVE):
        anchor_pair = model.rod_anchors[rod_index]
        desired_length = np.linalg.norm(
            projected_qpos[:, anchor_pair[1], :] - projected_qpos[:, anchor_pair[0], :],
            axis=1,
        )
        min_length, max_length = model.rod_length_limits[rod_index]
        if max_length <= min_length + 1e-9:
            normalized = np.zeros(state.num_envs, dtype=np.float64)
        else:
            normalized = (max_length - desired_length) / (max_length - min_length)
        group_index = int(model.rod_control_group[rod_index])
        contributions[:, group_index] += np.clip(normalized, 0.0, 1.0)
        counts[group_index] += 1

    mask = counts > 0
    control_target[:, mask] = contributions[:, mask] / counts[mask][None, :]
    np.clip(control_target, 0.0, 1.0, out=control_target)
    return control_target
