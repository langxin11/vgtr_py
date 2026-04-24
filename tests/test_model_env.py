"""模型编译、数据生成与仿真步进测试。

覆盖 VGTRModel 编译、VGTRData 分配、Simulator 步进、
投影层及 VGTREnv 环境接口的核心行为。
"""

from pathlib import Path

import numpy as np

from vgtr_py.config import default_config
from vgtr_py.data import make_data
from vgtr_py.env import VGTREnv
from vgtr_py.model import compile_workspace
from vgtr_py.projection import project_anchor_targets
from vgtr_py.schema import (
    ControlGroupFile,
    RodGroupFile,
    SiteFile,
    WorkspaceFile,
    load_workspace_file,
)
from vgtr_py.sim import Simulator
from vgtr_py.vector_env import VectorVGTREnv
from vgtr_py.workspace import ROD_TYPE_ACTIVE, Workspace


def make_workspace() -> Workspace:
    return Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0], fixed=True, projection_target=False),
                "s2": SiteFile(pos=[1.2, 0.0, 0.0], projection_target=True),
            },
            rod_groups=[
                RodGroupFile(
                    name="g12",
                    site1="s1",
                    site2="s2",
                    rod_type="active",
                    control_group="drive_A",
                    length_limits=[0.8, 1.2],
                    force_limits=[-500.0, 500.0],
                )
            ],
            control_groups=[
                ControlGroupFile(name="drive_A", default_target=0.0),
            ],
            script=[[0.0, 1.0]],
            num_actions=2,
        ),
        default_config(),
    )


def test_compile_workspace_exposes_projection_targets_and_limits() -> None:
    workspace = make_workspace()
    model = compile_workspace(workspace)

    assert model.anchor_count == 2
    assert model.projection_anchor_indices.tolist() == [1]
    assert model.rod_type.tolist() == [ROD_TYPE_ACTIVE]
    np.testing.assert_allclose(model.rod_length_limits, np.asarray([[0.8, 1.2]]))
    np.testing.assert_allclose(model.rod_force_limits, np.asarray([[-500.0, 500.0]]))


def test_simulator_and_projection_update_runtime_stats() -> None:
    workspace = make_workspace()
    model = compile_workspace(workspace)
    data = make_data(model)

    ctrl_target = project_anchor_targets(model, data, np.asarray([[0.9, 0.0, 0.0]], dtype=np.float64))
    assert ctrl_target.shape == (1,)
    assert ctrl_target[0] > 0.0

    simulator = Simulator()
    simulator.step(model, data, ctrl_target)

    assert data.step_count == 1
    assert data.rod_length.shape == (1,)
    assert data.rod_target_length.shape == (1,)
    assert data.rod_axial_force.shape == (1,)
    assert data.rod_strain.shape == (1,)


def test_env_reset_step_returns_flat_observation_and_info() -> None:
    env = VGTREnv.from_workspace(make_workspace(), max_steps=1)

    obs, info = env.reset(seed=42)
    assert obs.ndim == 1
    assert "rod_length" in info
    assert "tracking_error" in info

    action = np.asarray([0.9, 0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert next_obs.ndim == 1
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is True
    assert next_info["rod_axial_force"].shape == (1,)


def test_vector_env_reset_step_returns_batched_observation_and_info() -> None:
    env = VectorVGTREnv.from_workspace(make_workspace(), num_envs=3, max_steps=2)

    obs, info = env.reset(seed=42)
    assert obs.ndim == 2
    assert obs.shape[0] == 3
    assert info["rod_length"].shape == (3, 1)
    assert info["tracking_error"].shape == (3,)

    action = np.asarray(
        [
            [0.9, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert next_obs.shape == obs.shape
    assert reward.shape == (3,)
    assert terminated.tolist() == [False, False, False]
    assert truncated.tolist() == [False, False, False]
    assert next_info["rod_axial_force"].shape == (3, 1)


def test_vector_env_matches_single_env_for_same_action() -> None:
    workspace = make_workspace()
    single_env = VGTREnv.from_workspace(workspace, max_steps=5)
    vector_env = VectorVGTREnv.from_workspace(workspace, num_envs=2, max_steps=5)

    single_env.reset(seed=7)
    vector_env.reset(seed=7)
    action = np.asarray([0.9, 0.0, 0.0], dtype=np.float32)

    single_obs, single_reward, _, _, single_info = single_env.step(action)
    vector_obs, vector_reward, _, _, vector_info = vector_env.step(
        np.broadcast_to(action, vector_env.action_space.shape)
    )

    np.testing.assert_allclose(vector_obs[0], single_obs)
    np.testing.assert_allclose(vector_reward[0], single_reward)
    np.testing.assert_allclose(vector_info["rod_length"][0], single_info["rod_length"])
    np.testing.assert_allclose(vector_info["rod_axial_force"][0], single_info["rod_axial_force"])


def test_example_config_uses_new_schema_fields() -> None:
    workspace_file = load_workspace_file(Path("configs/example.json"))

    assert workspace_file.sites["s2"].projection_target is True
    assert workspace_file.rod_groups[0].rod_type == "active"
    assert workspace_file.rod_groups[0].length_limits == [0.9, 1.2]
