"""Shared env-major runtime kernels."""

from __future__ import annotations

import numpy as np

from ..model import VGTRModel
from ..workspace import ROD_TYPE_ACTIVE, ROD_TYPE_ELASTIC, ROD_TYPE_PASSIVE
from .state import RuntimeState


def set_ctrl_target(model: VGTRModel, state: RuntimeState, ctrl_target: np.ndarray | None) -> None:
    """Validate and write control targets with env-major semantics."""
    if ctrl_target is None:
        return
    if model.control_group_count == 0:
        return
    target = np.asarray(ctrl_target, dtype=np.float64)
    expected = (state.num_envs, model.control_group_count)
    if target.ndim == 1:
        if target.shape[0] != model.control_group_count:
            raise ValueError(
                f"expected control target with shape ({model.control_group_count},), got {target.shape}"
            )
        target = np.broadcast_to(target, expected)
    if target.shape != expected:
        raise ValueError(f"expected control target with shape {expected}, got {target.shape}")
    state.ctrl_target[...] = np.clip(target, 0.0, 1.0)


def advance_script_targets(model: VGTRModel, state: RuntimeState) -> None:
    """Advance script targets for each environment independently."""
    if model.script.size == 0 or model.num_actions <= 0 or model.control_group_count == 0:
        return
    threshold = ((state.i_action + 1) % model.num_actions) * model.config.num_steps_action
    update_mask = state.step_count > threshold
    if not np.any(update_mask):
        return
    next_action = (
        (state.step_count[update_mask] // model.config.num_steps_action) % model.num_actions
    ).astype(np.int64, copy=False)
    state.i_action_prev[update_mask] = state.i_action[update_mask]
    state.i_action[update_mask] = next_action
    state.ctrl_target[update_mask] = np.clip(model.script[:, next_action].T, 0.0, 1.0)


def update_ctrl(model: VGTRModel, state: RuntimeState) -> None:
    if state.ctrl.size == 0:
        return
    delta = np.clip(
        state.ctrl_target - state.ctrl,
        -model.config.contraction_percent_rate,
        model.config.contraction_percent_rate,
    )
    state.ctrl += delta
    np.clip(state.ctrl, 0.0, 1.0, out=state.ctrl)


def compute_target_lengths(model: VGTRModel, state: RuntimeState) -> np.ndarray:
    """Compute env-major rod target lengths."""
    target_lengths = np.broadcast_to(
        model.rod_rest_length,
        (state.num_envs, model.rod_count),
    ).copy()
    if model.rod_count == 0:
        return target_lengths

    active_mask = model.rod_type == ROD_TYPE_ACTIVE
    elastic_mask = model.rod_type == ROD_TYPE_ELASTIC
    passive_mask = model.rod_type == ROD_TYPE_PASSIVE

    if np.any(active_mask) and state.ctrl.size:
        active_groups = model.rod_control_group[active_mask]
        active_limits = model.rod_length_limits[active_mask]
        target_lengths[:, active_mask] = (
            active_limits[:, 0][None, :]
            + (active_limits[:, 1] - active_limits[:, 0])[None, :] * state.ctrl[:, active_groups]
        )

    override_mask = np.isfinite(state.rod_target_override) & active_mask[None, :]
    if np.any(override_mask):
        clipped_override = np.clip(
            state.rod_target_override,
            model.rod_length_limits[:, 0][None, :],
            model.rod_length_limits[:, 1][None, :],
        )
        target_lengths = np.where(override_mask, clipped_override, target_lengths)

    if np.any(elastic_mask):
        target_lengths[:, elastic_mask] = np.clip(
            model.rod_rest_length[elastic_mask],
            model.rod_length_limits[elastic_mask, 0],
            model.rod_length_limits[elastic_mask, 1],
        )[None, :]
    if np.any(passive_mask):
        target_lengths[:, passive_mask] = np.clip(
            model.rod_rest_length[passive_mask],
            model.rod_length_limits[passive_mask, 0],
            model.rod_length_limits[passive_mask, 1],
        )[None, :]
    return target_lengths


def compute_forces(model: VGTRModel, state: RuntimeState) -> None:
    """Compute rod, gravity, and ground-contact forces."""
    state.forces.fill(0.0)
    state.rod_stalled.fill(False)

    if model.rod_count:
        start = state.qpos[:, model.rod_anchors[:, 0], :]
        end = state.qpos[:, model.rod_anchors[:, 1], :]
        rod_vec = end - start
        rod_lengths = np.linalg.norm(rod_vec, axis=2)
        safe_lengths = np.where(rod_lengths > 1e-12, rod_lengths, 1.0)
        unit = rod_vec / safe_lengths[:, :, None]
        target_lengths = compute_target_lengths(model, state)

        raw_force = model.config.k * (rod_lengths - target_lengths)
        clipped_force = np.clip(
            raw_force,
            model.rod_force_limits[:, 0][None, :],
            model.rod_force_limits[:, 1][None, :],
        )
        active_mask = model.rod_type == ROD_TYPE_ACTIVE
        stalled_by_force = np.abs(raw_force - clipped_force) > 1e-9
        stalled_by_length = (rod_lengths < model.rod_length_limits[:, 0][None, :] - 1e-9) | (
            rod_lengths > model.rod_length_limits[:, 1][None, :] + 1e-9
        )
        state.rod_stalled[...] = active_mask[None, :] & (stalled_by_force | stalled_by_length)

        rod_forces = unit * clipped_force[:, :, None]
        env_indices = np.arange(state.num_envs)[:, None]
        np.add.at(state.forces, (env_indices, model.rod_anchors[:, 0]), rod_forces)
        np.add.at(state.forces, (env_indices, model.rod_anchors[:, 1]), -rod_forces)

        state.rod_length[...] = rod_lengths
        state.rod_target_length[...] = target_lengths
        state.rod_axial_force[...] = clipped_force
        safe_target = np.where(np.abs(target_lengths) > 1e-9, target_lengths, 1.0)
        state.rod_strain[...] = (rod_lengths - target_lengths) / safe_target

    if model.config.gravity:
        state.forces[:, :, 2] -= model.anchor_mass[None, :] * model.config.gravity_factor

    z_pos = state.qpos[:, :, 2]
    vz = state.qvel[:, :, 2]
    state.contact_mask[...] = z_pos < 0.0
    spring_force = -model.config.ground_k * z_pos
    damping_force = -model.config.ground_d * vz
    normal_force = np.where(state.contact_mask, np.maximum(0.0, spring_force + damping_force), 0.0)
    state.forces[:, :, 2] += normal_force

    v_xy = state.qvel[:, :, :2]
    speed_xy = np.linalg.norm(v_xy, axis=2)
    fric_max = model.config.friction_factor * normal_force
    fric_mag = np.minimum(fric_max, 100.0 * speed_xy)
    safe_speed = np.where(speed_xy > 1e-6, speed_xy, 1.0)
    state.forces[:, :, :2] += -v_xy * (fric_mag / safe_speed)[:, :, None]


def integrate(model: VGTRModel, state: RuntimeState) -> None:
    """Integrate env-major anchor state with explicit Euler."""
    movable_indices = np.flatnonzero(~model.anchor_fixed)
    if movable_indices.size == 0:
        return
    safe_mass = np.where(model.anchor_mass > 1e-9, model.anchor_mass, 1.0)
    state.qvel[:, movable_indices, :] = (
        state.qvel[:, movable_indices, :]
        + (state.forces[:, movable_indices, :] / safe_mass[movable_indices][None, :, None])
        * model.config.h
    )
    state.qvel[:, movable_indices, :] *= model.config.damping_ratio

    speed = np.linalg.norm(state.qvel[:, movable_indices, :], axis=2)
    scale = np.where(speed > 10.0, 10.0 / np.where(speed > 1e-12, speed, 1.0), 1.0)
    state.qvel[:, movable_indices, :] *= scale[:, :, None]

    state.qpos[:, movable_indices, :] += state.qvel[:, movable_indices, :] * model.config.h
    below_ground = state.qpos[:, :, 2] < 0.0
    state.qvel[:, :, 2] = np.where(below_ground, 0.0, state.qvel[:, :, 2])
    state.qpos[:, :, 2] = np.where(below_ground, 0.0, state.qpos[:, :, 2])


def step_state(
    model: VGTRModel,
    state: RuntimeState,
    ctrl_target: np.ndarray | None = None,
) -> None:
    """Advance a runtime state by one step."""
    set_ctrl_target(model, state, ctrl_target)
    update_ctrl(model, state)
    compute_forces(model, state)
    integrate(model, state)
    state.time += model.config.h
    state.step_count += 1
    if state.record_frames:
        if state.frames is None:
            state.frames = []
        state.frames.append(state.qpos.tolist())
