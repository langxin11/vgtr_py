from pathlib import Path

import numpy as np

from vgtr_py.commands import (
    apply_edit,
    connect_selected_anchors,
    load_workspace_from_paths,
    save_workspace_to_path,
)
from vgtr_py.config import default_config
from vgtr_py.history import WorkspaceHistory
from vgtr_py.schema import WorkspaceFile
from vgtr_py.topology import add_joint, select_anchor
from vgtr_py.workspace import Workspace


def test_load_workspace_from_paths_builds_workspace(tmp_path: Path) -> None:
    """
    测试：加载工作区命令 (load_workspace_from_paths) 应该能够正确地从磁盘上的真实文件中
    组装为一个功能完整的 Workspace 实例。
    """
    # Arrange: 使用最新的 VGTR 模式写入最小可用的数据文件
    config_path = tmp_path / "config.json"
    example_path = tmp_path / "example.json"
    config_path.write_text('{"k": 20000}', encoding="utf-8")
    example_path.write_text(
        '{"sites": {"s1": {"pos": [0,0,0]}, "s2": {"pos": [1,0,0]}}, "rod_groups": [{"name": "g1", "site1": "s1", "site2": "s2"}]}',
        encoding="utf-8",
    )

    # Act
    workspace = load_workspace_from_paths(
        config_path=config_path,
        example_path=example_path,
    )

    # Assert
    assert isinstance(workspace, Workspace)
    assert workspace.topology.anchor_pos.shape == (2, 3)
    assert workspace.topology.rod_anchors.shape == (1, 2)


def test_apply_edit_pushes_to_history(tmp_path: Path) -> None:
    """
    测试：apply_edit 应该在执行动作且状态发生改变时，自动将修改前的状态压入历史记录。
    """
    # Arrange
    example_path = tmp_path / "example.json"
    example_path.write_text(
        '{"sites": {"s1": {"pos": [0,0,0]}}, "rod_groups": []}',
        encoding="utf-8",
    )
    workspace = load_workspace_from_paths(config_path=None, example_path=example_path)
    history = WorkspaceHistory()

    # Act
    # 执行一个会改变状态的操作 (增加一个锚点)
    apply_edit(workspace, history, lambda: add_joint(workspace))

    # Assert
    assert history.undo_stack_size == 1
    assert workspace.topology.anchor_pos.shape[0] == 2


def test_connect_selected_anchors_creates_rod(tmp_path: Path) -> None:
    """
    测试：connect_selected_anchors 应该在选中两个锚点时创建一条新连杆。
    """
    # Arrange
    example_path = tmp_path / "example.json"
    example_path.write_text(
        '{"sites": {"s1": {"pos": [0,0,0]}, "s2": {"pos": [1,0,0]}}, "rod_groups": []}',
        encoding="utf-8",
    )
    workspace = load_workspace_from_paths(config_path=None, example_path=example_path)
    history = WorkspaceHistory()

    # 选中两个点
    select_anchor(workspace, 0, additive=True)
    select_anchor(workspace, 1, additive=True)

    # Act
    connect_selected_anchors(workspace, history)

    # Assert
    assert workspace.topology.rod_anchors.shape[0] == 1
    assert history.undo_stack_size == 1
