import numpy as np

from vgtr_py.config import default_config
from vgtr_py.schema import LegacyWorkspaceFile
from vgtr_py.topology import remove_selected_edges, remove_selected_vertices
from vgtr_py.workspace import Workspace


def make_workspace() -> Workspace:
    """
    辅助函数：构建一个拓扑结构的 Workspace 用于测试。
    该拓扑包含4个顶点和3条边：
    顶点: 0, 1, 2, 3
    边: [0-1], [1-2], [2-3]
    其中，顶点1处于被选中状态 (vStatus=2)，边[1-2]处于被选中状态 (eStatus=2)。
    """
    return Workspace.from_legacy_file(
        LegacyWorkspaceFile(
            v=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            e=[
                [0, 1],
                [1, 2],
                [2, 3],
            ],
            fixedVs=[0, 0, 0, 0],
            lMax=[10.0, 20.0, 30.0],
            edgeChannel=[1, 2, 3],
            edgeActive=[1, 0, 1],
            maxContraction=[0.1, 0.2, 0.3],
            vStatus=[0, 2, 0, 0],  # 顶点1 被选中 (状态码为 2)
            eStatus=[0, 2, 1],  # 边[1-2] 被选中 (状态码为 2)
        ),
        default_config(),
    )


def test_remove_selected_vertices_keeps_edge_arrays_in_sync() -> None:
    """
    测试：删除当前所有被选中的顶点时，与之相关联的所有边，以及所有的边属性（通道、收缩率等）必须保持同步删除和更新。
    """
    # Arrange: 创建带有一个被选中顶点 (顶点1) 的拓扑结构
    workspace = make_workspace()

    # Act: 执行删除选中顶点的操作
    # 顶点1被删除后，与顶点1相连的边 [0-1] 和 [1-2] 也必须被级联删除，只剩下边 [2-3]
    removed = remove_selected_vertices(workspace)

    # Assert: 验证拓扑结构和属性的一致性
    assert removed == 1  # 验证删除了 1 个顶点

    # 验证只剩下原来的边 [2-3]（由于顶点0和1被删除或重排，新的顶点索引变为了 [1-2]）
    np.testing.assert_array_equal(
        workspace.topology.edges,
        np.asarray([[1, 2]], dtype=np.int32),
    )
    # 验证原边 [2-3] 的各项属性（l_max=30.0, maxContraction=0.3, channel=3, active=True）是否正确保留
    np.testing.assert_array_equal(workspace.topology.l_max, np.asarray([30.0], dtype=np.float64))
    np.testing.assert_array_equal(
        workspace.topology.max_contraction,
        np.asarray([0.3], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        workspace.topology.edge_channel,
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        workspace.topology.edge_active,
        np.asarray([True], dtype=np.bool_),
    )

    # 验证 UI 状态数组大小是否已正确裁剪，并且所有剩余元素的选中状态被清空
    assert workspace.ui.vertex_status.shape == (3,)
    assert workspace.ui.edge_status.shape == (1,)
    assert np.count_nonzero(workspace.ui.vertex_status) == 0
    assert np.count_nonzero(workspace.ui.edge_status) == 0


def test_remove_selected_edges_keeps_edge_arrays_in_sync() -> None:
    """
    测试：删除当前所有被选中的边时，所有与这些边相关的平行属性（如最大长度、收缩率等）都必须同步删除，并保持对应关系不变。
    """
    # Arrange: 创建带有一条被选中边 [1-2] 的拓扑结构
    workspace = make_workspace()

    # Act: 执行删除选中边的操作
    removed = remove_selected_edges(workspace)

    # Assert: 验证有且仅有一条边被删除
    assert removed == 1

    # 验证剩余的边为原来的 [0-1] 和 [2-3]
    np.testing.assert_array_equal(
        workspace.topology.edges,
        np.asarray([[0, 1], [2, 3]], dtype=np.int32),
    )
    # 验证剩余边对应的属性数组是否同步裁切正确（属性10.0与30.0对应第一和第三条边）
    np.testing.assert_array_equal(
        workspace.topology.l_max,
        np.asarray([10.0, 30.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        workspace.topology.max_contraction,
        np.asarray([0.1, 0.3], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        workspace.topology.edge_channel,
        np.asarray([1, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        workspace.topology.edge_active,
        np.asarray([True, True], dtype=np.bool_),
    )

    # 验证 UI 中的选中状态标志是否已被清空
    assert np.count_nonzero(workspace.ui.vertex_status) == 0
    assert np.count_nonzero(workspace.ui.edge_status) == 0
