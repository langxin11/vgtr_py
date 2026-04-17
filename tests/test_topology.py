import numpy as np

from vgtr_py.config import default_config
from vgtr_py.schema import WorkspaceFile, SiteFile, RodGroupFile
from vgtr_py.topology import remove_selected_edges, remove_selected_vertices, select_anchor, select_rod_group
from vgtr_py.workspace import Workspace


def make_workspace() -> Workspace:
    """
    辅助函数：构建一个拓扑结构的 Workspace 用于测试。
    该拓扑包含4个锚点和3根杆组：
    锚点: s1, s2, s3, s4
    杆组: [s1-s2], [s2-s3], [s3-s4]
    """
    workspace = Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0]),
                "s2": SiteFile(pos=[1.0, 0.0, 0.0]),
                "s3": SiteFile(pos=[2.0, 0.0, 0.0]),
                "s4": SiteFile(pos=[3.0, 0.0, 0.0]),
            },
            rod_groups=[
                RodGroupFile(name="g1", site1="s1", site2="s2", control_group="cg1", enabled=True),
                RodGroupFile(name="g2", site1="s2", site2="s3", control_group="cg2", enabled=False),
                RodGroupFile(name="g3", site1="s3", site2="s4", control_group="cg3", enabled=True),
            ],
            control_groups=[],
        ),
        default_config(),
    )
    # 模拟 UI 选择
    select_anchor(workspace, 1, additive=False)  # 选中 s2
    select_rod_group(workspace, 1, additive=True) # 选中 g2 (使用增量模式以免清空点选择)
    return workspace


def test_remove_selected_vertices_keeps_rod_arrays_in_sync() -> None:
    """
    测试：删除当前所有被选中的锚点时，与之相关联的所有杆组，以及所有的杆组属性必须保持同步删除和更新。
    """
    # Arrange
    workspace = make_workspace()

    # Act: 删除选中锚点 (s2)
    # s2被删除后，关联的 g1 和 g2 也必须被级联删除，只剩下 g3
    removed = remove_selected_vertices(workspace)

    # Assert
    assert removed == 1
    # 验证只剩下原来的杆组 g3 (现在连接新的索引 1 和 2，即原来的 s3 和 s4)
    assert workspace.topology.rod_anchors.shape[0] == 1
    
    # 验证属性同步
    assert bool(workspace.topology.rod_enabled[0]) is True
    assert workspace.topology.rod_group_ids[0] == "g3"

    # 验证 UI 状态
    assert workspace.ui.anchor_status.shape == (3,)
    assert workspace.ui.rod_group_status.shape == (1,)
    assert np.count_nonzero(workspace.ui.anchor_status) == 0
    assert np.count_nonzero(workspace.ui.rod_group_status) == 0


def test_remove_selected_edges_keeps_rod_arrays_in_sync() -> None:
    """
    测试：删除当前所有被选中的杆组时，所有相关属性都必须同步删除。
    """
    # Arrange
    workspace = make_workspace()

    # Act: 删除选中杆组 (g2)
    removed = remove_selected_edges(workspace)

    # Assert
    assert removed == 1
    assert workspace.topology.rod_anchors.shape[0] == 2
    
    # 验证剩余的是 g1 和 g3
    assert workspace.topology.rod_group_ids == ["g1", "g3"]
    assert bool(workspace.topology.rod_enabled[0]) is True
    assert bool(workspace.topology.rod_enabled[1]) is True

    # 验证 UI 状态
    assert np.count_nonzero(workspace.ui.anchor_status) == 0
    assert np.count_nonzero(workspace.ui.rod_group_status) == 0
