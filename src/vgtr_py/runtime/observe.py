"""Observation builders for runtime sessions."""

from __future__ import annotations

import numpy as np

from ..model import VGTRModel
from .state import RuntimeState


def build_observation(model: VGTRModel, state: RuntimeState) -> np.ndarray:
    del model
    parts = [
        state.qpos.reshape(state.num_envs, -1),
        state.qvel.reshape(state.num_envs, -1),
        state.rod_length,
        state.rod_target_length,
        state.rod_axial_force,
        state.rod_strain,
        state.rod_stalled.astype(np.float64),
        state.ctrl,
        state.ctrl_target,
        state.contact_mask.astype(np.float64),
    ]
    return (
        np.concatenate(parts, axis=1, dtype=np.float64)
        if parts
        else np.zeros((state.num_envs, 0), dtype=np.float64)
    )
