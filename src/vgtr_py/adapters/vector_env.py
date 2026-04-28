"""Batched Gym-style adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..gym_compat import gym, spaces
from ..runtime.session import ControlMode, RuntimeSession
from ..workspace import Workspace


@dataclass(slots=True)
class VGTRVectorEnv(gym.Env):
    """Vectorized Gym-style environment backed by RuntimeSession."""

    session: RuntimeSession

    metadata = {"render_modes": []}

    def __post_init__(self) -> None:
        action_dim = self.session.action_dim
        action_shape = (self.session.num_envs, action_dim)
        obs_template = self.session.observe_batch().astype(np.float32)
        self.single_action_space = spaces.Box(
            low=np.full(action_dim, -10.0, dtype=np.float32),
            high=np.full(action_dim, 10.0, dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.full(action_shape, -10.0, dtype=np.float32),
            high=np.full(action_shape, 10.0, dtype=np.float32),
            shape=action_shape,
            dtype=np.float32,
        )
        self.single_observation_space = spaces.Box(
            low=np.full(obs_template.shape[1:], -np.inf, dtype=np.float32),
            high=np.full(obs_template.shape[1:], np.inf, dtype=np.float32),
            shape=obs_template.shape[1:],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(obs_template.shape, -np.inf, dtype=np.float32),
            high=np.full(obs_template.shape, np.inf, dtype=np.float32),
            shape=obs_template.shape,
            dtype=np.float32,
        )

    @property
    def model(self):
        return self.session.model

    @property
    def state(self):
        return self.session.state

    @classmethod
    def from_workspace(
        cls,
        workspace: Workspace,
        *,
        num_envs: int,
        max_steps: int = 1000,
        control_mode: ControlMode = "direct",
    ) -> VGTRVectorEnv:
        session = RuntimeSession.from_workspace(
            workspace,
            num_envs=num_envs,
            max_steps=max_steps,
            control_mode=control_mode,
        )
        return cls(session=session)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        del options
        return self.session.reset(seed=seed)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        return self.session.step_batch(action)
