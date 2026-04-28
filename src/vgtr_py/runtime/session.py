"""Public runtime session API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..model import VGTRModel, compile_workspace
from ..workspace import Workspace
from .alloc import make_state, reset_state
from .kernels import advance_script_targets, step_state
from .observe import build_observation
from .project import project_anchor_targets
from .reward import build_info, compute_reward
from .state import RuntimeState

ControlMode = Literal["direct", "projection"]


@dataclass(slots=True)
class RuntimeSession:
    """Unified runtime facade over env-major state."""

    model: VGTRModel
    state: RuntimeState
    control_mode: ControlMode = "direct"
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.control_mode not in {"direct", "projection"}:
            raise ValueError(f"unsupported control_mode {self.control_mode!r}")

    @classmethod
    def from_workspace(
        cls,
        workspace: Workspace,
        *,
        num_envs: int = 1,
        control_mode: ControlMode = "direct",
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> "RuntimeSession":
        model = compile_workspace(workspace)
        state = make_state(model, num_envs=num_envs, seed=seed)
        return cls(model=model, state=state, control_mode=control_mode, max_steps=max_steps)

    @property
    def num_envs(self) -> int:
        return self.state.num_envs

    @property
    def action_dim(self) -> int:
        if self.control_mode == "projection":
            return max(int(self.model.projection_anchor_indices.shape[0] * 3), 1)
        return max(int(self.model.control_group_count), 1)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        reset_state(self.model, self.state, seed=seed)
        return self.observe_batch(), self.info_batch()

    def advance_script_targets(self) -> None:
        advance_script_targets(self.model, self.state)

    def observe_batch(self) -> np.ndarray:
        return build_observation(self.model, self.state)

    def info_batch(self) -> dict[str, np.ndarray]:
        return build_info(self.model, self.state)

    def _normalize_direct_action(self, action: np.ndarray | None) -> np.ndarray | None:
        if self.model.control_group_count == 0:
            if action is None:
                return None
            action_array = np.asarray(action, dtype=np.float64)
            expected = (self.num_envs, 1)
            if action_array.ndim == 1 and action_array.shape == (1,):
                return None
            if action_array.shape == expected:
                return None
            raise ValueError(
                "direct control mode requires no control groups; pass None or a dummy shape (1,) / (E, 1)"
            )
        action_array = np.asarray(action, dtype=np.float64) if action is not None else self.state.ctrl_target
        expected = (self.num_envs, self.model.control_group_count)
        if action_array.ndim == 1:
            if action_array.shape[0] != self.model.control_group_count:
                raise ValueError(
                    f"expected direct action with shape ({self.model.control_group_count},), got {action_array.shape}"
                )
            action_array = np.broadcast_to(action_array, expected)
        if action_array.shape != expected:
            raise ValueError(f"expected direct action with shape {expected}, got {action_array.shape}")
        return action_array

    def _normalize_projection_action(self, action: np.ndarray | None) -> np.ndarray:
        if self.model.projection_anchor_indices.size == 0:
            raise ValueError("projection control mode requires at least one projection anchor")
        if action is None:
            raise ValueError("projection control mode requires an explicit action")
        target_dim = int(self.model.projection_anchor_indices.shape[0] * 3)
        action_array = np.asarray(action, dtype=np.float64)
        expected = (self.num_envs, target_dim)
        if action_array.ndim == 1:
            if action_array.shape[0] != target_dim:
                raise ValueError(
                    f"expected projection action with shape ({target_dim},), got {action_array.shape}"
                )
            action_array = np.broadcast_to(action_array, expected)
        if action_array.shape != expected:
            raise ValueError(f"expected projection action with shape {expected}, got {action_array.shape}")
        return action_array

    def _resolve_ctrl_target(
        self,
        action: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.control_mode == "projection":
            raw_action = self._normalize_projection_action(action)
            return project_anchor_targets(self.model, self.state, raw_action), raw_action
        return self._normalize_direct_action(action), None

    def _truncated(self) -> np.ndarray:
        if self.max_steps is None:
            return np.zeros(self.num_envs, dtype=np.bool_)
        return self.state.step_count >= self.max_steps

    def step_batch(
        self,
        action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        ctrl_target, reward_action = self._resolve_ctrl_target(action)
        step_state(self.model, self.state, ctrl_target)
        observation = self.observe_batch()
        reward = compute_reward(self.model, self.state, reward_action)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = self._truncated()
        info = self.info_batch()
        return observation, reward, terminated, truncated, info

    def step(
        self,
        action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        return self.step_batch(action)

    def _require_single_env(self) -> None:
        if self.num_envs != 1:
            raise ValueError("single-env convenience API requires num_envs == 1")

    def observe_one(self) -> np.ndarray:
        self._require_single_env()
        return self.observe_batch()[0]

    def info_one(self) -> dict[str, object]:
        self._require_single_env()
        return _single_info(self.info_batch())

    def step_one(
        self,
        action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._require_single_env()
        observation, reward, terminated, truncated, info = self.step_batch(action)
        return observation[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), _single_info(info)


def _single_info(info: dict[str, np.ndarray]) -> dict[str, object]:
    single: dict[str, object] = {}
    for key, value in info.items():
        if value.ndim == 0:
            single[key] = value.item()
        elif value.shape[0] == 1:
            item = value[0]
            if np.isscalar(item):
                single[key] = item.item() if hasattr(item, "item") else item
            else:
                single[key] = item.copy()
        else:
            single[key] = value.copy()
    return single
