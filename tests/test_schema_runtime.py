"""Schema 与工作区运行时行为测试。

覆盖 WorkspaceFile 加载、边界限制与投影标志的往返一致性。
"""

import pytest

from vgtr_py.config import default_config
from vgtr_py.schema import ControlGroupFile, RodGroupFile, SiteFile, WorkspaceFile
from vgtr_py.workspace import Workspace


def test_workspace_roundtrip_preserves_projection_and_limits() -> None:
    workspace = Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0], fixed=True, projection_target=False),
                "s2": SiteFile(pos=[1.0, 0.0, 0.0], projection_target=True),
            },
            rod_groups=[
                RodGroupFile(
                    name="g12",
                    site1="s1",
                    site2="s2",
                    rod_type="active",
                    control_group="cg1",
                    length_limits=[0.6, 1.0],
                    force_limits=[-100.0, 100.0],
                )
            ],
            control_groups=[ControlGroupFile(name="cg1", default_target=0.3)],
        ),
        default_config(),
    )

    dumped = workspace.to_workspace_file()
    assert dumped.sites["s2"].projection_target is True
    assert dumped.rod_groups[0].rod_type == "active"
    assert dumped.rod_groups[0].length_limits == [0.6, 1.0]
    assert dumped.rod_groups[0].force_limits == [-100.0, 100.0]
    assert dumped.control_groups[0].default_target == 0.3


def test_length_limits_do_not_override_initial_rest_length() -> None:
    workspace = Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0]),
                "s2": SiteFile(pos=[1.0, 0.0, 0.0]),
            },
            rod_groups=[
                RodGroupFile(
                    name="g12",
                    site1="s1",
                    site2="s2",
                    length_limits=[0.6, 1.4],
                )
            ],
        ),
        default_config(),
    )

    assert workspace.topology.rod_rest_length.tolist() == [1.0]
    assert workspace.topology.rod_min_length.tolist() == [0.6]
    assert workspace.topology.rod_length_limits.tolist() == [[0.6, 1.4]]


def test_workspace_from_file_rejects_invalid_rod_type() -> None:
    with pytest.raises(ValueError, match="unsupported rod_type"):
        Workspace.from_workspace_file(
            WorkspaceFile(
                sites={
                    "s1": SiteFile(pos=[0.0, 0.0, 0.0]),
                    "s2": SiteFile(pos=[1.0, 0.0, 0.0]),
                },
                rod_groups=[
                    RodGroupFile(
                        name="g12",
                        site1="s1",
                        site2="s2",
                        rod_type="invalid-type",
                    )
                ],
            ),
            default_config(),
        )
