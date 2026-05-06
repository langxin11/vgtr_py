"""模型编译、统一 runtime 与适配器测试。

覆盖 VGTRModel 编译、RuntimeSession 步进、投影层及 Gym 适配器核心行为。
"""

from pathlib import Path

import numpy as np
import pytest

from vgtr_py.adapters import VGTRGymEnv, VGTRVectorEnv
from vgtr_py.config import default_config
from vgtr_py.model import compile_workspace
from vgtr_py.runtime import RuntimeSession, make_state, project_anchor_targets
from vgtr_py.runtime.kernels import advance_script_targets
from vgtr_py.schema import (
    ControlGroupFile,
    RodGroupFile,
    SiteFile,
    WorkspaceFile,
    load_workspace_file,
)
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


def make_two_active_rods_one_group_workspace() -> Workspace:
    config = default_config()
    config.contraction_percent_rate = 1.0
    return Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0], fixed=True, projection_target=False),
                "s2": SiteFile(pos=[1.0, 0.0, 0.0], projection_target=True),
                "s3": SiteFile(pos=[2.0, 0.0, 0.0], projection_target=True),
            },
            rod_groups=[
                RodGroupFile(
                    name="g12",
                    site1="s1",
                    site2="s2",
                    rod_type="active",
                    control_group="shared",
                    length_limits=[0.8, 1.2],
                ),
                RodGroupFile(
                    name="g23",
                    site1="s2",
                    site2="s3",
                    rod_type="active",
                    control_group="shared",
                    length_limits=[0.8, 1.2],
                ),
            ],
            control_groups=[ControlGroupFile(name="shared", default_target=0.25)],
        ),
        config,
    )


def test_compile_workspace_exposes_projection_targets_and_limits() -> None:
    workspace = make_workspace()
    model = compile_workspace(workspace)

    assert model.anchor_count == 2
    assert model.projection_anchor_indices.tolist() == [1]
    assert model.active_rod_indices.tolist() == [0]
    assert model.rod_type.tolist() == [ROD_TYPE_ACTIVE]
    np.testing.assert_allclose(model.rod_length_limits, np.asarray([[0.8, 1.2]]))
    np.testing.assert_allclose(model.rod_force_limits, np.asarray([[-500.0, 500.0]]))


def test_runtime_session_and_projection_update_runtime_stats() -> None:
    workspace = make_workspace()
    model = compile_workspace(workspace)
    state = make_state(model, num_envs=1)
    session = RuntimeSession(model=model, state=state, control_mode="projection")

    ctrl_target = project_anchor_targets(
        model,
        state,
        np.asarray([[0.9, 0.0, 0.0]], dtype=np.float64),
    )
    assert ctrl_target.shape == (1, 1)
    assert ctrl_target[0, 0] > 0.0

    session.step_batch(np.asarray([0.9, 0.0, 0.0], dtype=np.float64))

    assert state.step_count.tolist() == [1]
    assert state.rod_length.shape == (1, 1)
    assert state.rod_target_length.shape == (1, 1)
    assert state.rod_axial_force.shape == (1, 1)
    assert state.rod_strain.shape == (1, 1)
    assert state.rod_ctrl_target.shape == (1, 1)


def test_runtime_session_single_env_helpers_and_env_view() -> None:
    model = compile_workspace(make_workspace())
    state = make_state(model, num_envs=2)
    session = RuntimeSession(model=model, state=state, control_mode="direct", max_steps=2)

    obs = session.observe_batch()
    assert obs.shape[0] == 2

    view = state.env_view(1)
    assert view.env_index == 1
    assert view.qpos.shape == (2, 3)
    assert view.qpos.flags.writeable is False
    assert view.ctrl.flags.writeable is False
    assert view.rod_ctrl.flags.writeable is False
    assert view.rod_ctrl_target.flags.writeable is False

    single_session = RuntimeSession(
        model=model, state=make_state(model, num_envs=1), control_mode="direct"
    )
    one_obs = single_session.observe_one()
    assert one_obs.ndim == 1
    with pytest.raises(ValueError, match="num_envs == 1"):
        session.observe_one()


def test_runtime_session_explicit_mode_rejects_wrong_action_shape() -> None:
    workspace = make_workspace()
    direct_session = RuntimeSession.from_workspace(workspace, control_mode="direct")
    projection_session = RuntimeSession.from_workspace(workspace, control_mode="projection")

    with pytest.raises(ValueError, match="direct action"):
        direct_session.step_batch(np.asarray([0.9, 0.0, 0.0], dtype=np.float64))
    with pytest.raises(ValueError, match="projection action"):
        projection_session.step_batch(np.asarray([0.2], dtype=np.float64))


def test_direct_mode_uses_per_active_rod_action_dimension() -> None:
    workspace = make_two_active_rods_one_group_workspace()
    model = compile_workspace(workspace)
    state = make_state(model, num_envs=2)
    session = RuntimeSession(model=model, state=state, control_mode="direct")

    assert session.action_dim == 2
    np.testing.assert_allclose(state.rod_ctrl_target, np.full((2, 2), 0.25))

    with pytest.raises(ValueError, match="direct action"):
        session.step_batch(np.asarray([0.5], dtype=np.float64))

    session.step_batch(np.asarray([0.0, 1.0], dtype=np.float64))

    np.testing.assert_allclose(state.rod_ctrl_target, np.asarray([[0.0, 1.0], [0.0, 1.0]]))
    np.testing.assert_allclose(state.rod_ctrl, np.asarray([[0.0, 1.0], [0.0, 1.0]]))
    np.testing.assert_allclose(state.rod_target_length[0], np.asarray([0.8, 1.2]))


def test_control_group_mode_expands_to_per_rod_targets() -> None:
    workspace = make_two_active_rods_one_group_workspace()
    model = compile_workspace(workspace)
    state = make_state(model, num_envs=1)
    session = RuntimeSession(model=model, state=state, control_mode="control_group")

    assert session.action_dim == 1
    session.step_batch(np.asarray([0.75], dtype=np.float64))

    np.testing.assert_allclose(state.ctrl_target, np.asarray([[0.75]]))
    np.testing.assert_allclose(state.rod_ctrl_target, np.asarray([[0.75, 0.75]]))
    np.testing.assert_allclose(state.rod_ctrl, np.asarray([[0.75, 0.75]]))


def test_projection_mode_outputs_per_rod_targets_without_group_average() -> None:
    workspace = make_two_active_rods_one_group_workspace()
    model = compile_workspace(workspace)
    state = make_state(model, num_envs=1)
    session = RuntimeSession(model=model, state=state, control_mode="projection")

    projected = project_anchor_targets(
        model,
        state,
        np.asarray([0.9, 0.0, 0.0, 2.1, 0.0, 0.0], dtype=np.float64),
    )

    assert session.action_dim == 6
    assert projected.shape == (1, 2)
    np.testing.assert_allclose(projected, np.asarray([[0.25, 1.0]]))


def test_advance_script_targets_uses_integer_script_indices() -> None:
    model = compile_workspace(make_workspace())
    state = make_state(model, num_envs=1)
    state.step_count[:] = int(model.config.num_steps_action) + 1

    advance_script_targets(model, state)

    assert state.i_action.tolist() == [1]
    np.testing.assert_allclose(state.ctrl_target, np.asarray([[1.0]], dtype=np.float64))
    np.testing.assert_allclose(state.rod_ctrl_target, np.asarray([[1.0]], dtype=np.float64))


def test_gym_env_reset_step_returns_flat_observation_and_info() -> None:
    env = VGTRGymEnv.from_workspace(make_workspace(), max_steps=1, control_mode="projection")

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
    env = VGTRVectorEnv.from_workspace(
        make_workspace(),
        num_envs=3,
        max_steps=2,
        control_mode="projection",
    )

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
    single_env = VGTRGymEnv.from_workspace(workspace, max_steps=5, control_mode="projection")
    vector_env = VGTRVectorEnv.from_workspace(
        workspace,
        num_envs=2,
        max_steps=5,
        control_mode="projection",
    )

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
