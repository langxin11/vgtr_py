from vgtr_py.config import default_config
from vgtr_py.history import WorkspaceHistory, snapshots_equal
from vgtr_py.schema import LegacyWorkspaceFile
from vgtr_py.workspace import Workspace


def make_workspace() -> Workspace:
    """
    辅助函数：创建一个包含基础拓扑（3个顶点，2条边）的测试用 Workspace 实例。
    """
    return Workspace.from_legacy_file(
        LegacyWorkspaceFile(
            v=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            e=[
                [0, 1],
                [1, 2],
            ],
        ),
        default_config(),
    )


def test_workspace_history_undo_redo_roundtrip() -> None:
    """
    测试：Workspace 历史记录管理器的撤销 (Undo) 和重做 (Redo) 流程必须能够无损地来回切换状态。
    """
    # Arrange: 准备基础的 workspace 和对应的历史记录实例
    workspace = make_workspace()
    history = WorkspaceHistory()
    before = workspace.snapshot()

    # Act: 修改第一个顶点的 X 坐标，并将修改前的状态推入历史记录栈
    workspace.topology.vertices[0, 0] = 42.0
    history.push(before)

    # Assert Undo: 执行撤销操作，验证状态是否完全恢复到了修改前
    assert history.undo(workspace) is True
    assert snapshots_equal(workspace.snapshot(), before) is True

    # Assert Redo: 执行重做操作，验证新修改（42.0）是否被正确重新应用
    assert history.redo(workspace) is True
    assert workspace.topology.vertices[0, 0] == 42.0
