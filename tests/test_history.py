from vgtr_py.config import default_config
from vgtr_py.history import WorkspaceHistory, snapshots_equal
from vgtr_py.schema import WorkspaceFile, SiteFile, RodGroupFile
from vgtr_py.workspace import Workspace


def make_workspace() -> Workspace:
    """
    辅助函数：创建一个包含基础拓扑（3个锚点，2根杆组）的测试用 Workspace 实例。
    """
    return Workspace.from_workspace_file(
        WorkspaceFile(
            sites={
                "s1": SiteFile(pos=[0.0, 0.0, 0.0]),
                "s2": SiteFile(pos=[1.0, 0.0, 0.0]),
                "s3": SiteFile(pos=[2.0, 0.0, 0.0]),
            },
            rod_groups=[
                RodGroupFile(name="g1", site1="s1", site2="s2"),
                RodGroupFile(name="g2", site2="s2", site1="s3"),
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

    # Act: 修改第一个锚点的 X 坐标，并将修改前的状态推入历史记录栈
    workspace.topology.anchor_pos[0, 0] = 42.0
    history.push(before)

    # Assert Undo: 执行撤销操作，验证状态是否完全恢复到了修改前
    assert history.undo(workspace) is True
    assert snapshots_equal(workspace.snapshot(), before) is True

    # Assert Redo: 执行重做操作，验证新修改（42.0）是否被正确重新应用
    assert history.redo(workspace) is True
    assert workspace.topology.anchor_pos[0, 0] == 42.0
