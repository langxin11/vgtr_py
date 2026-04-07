from pathlib import Path

import numpy as np

from vgtr_py.commands import (
    apply_edit,
    connect_selected_vertices,
    load_workspace_from_paths,
    save_workspace_to_path,
)
from vgtr_py.config import default_config
from vgtr_py.history import WorkspaceHistory
from vgtr_py.schema import LegacyWorkspaceFile, load_legacy_workspace
from vgtr_py.topology import add_joint, select_vertex
from vgtr_py.workspace import Workspace


def test_load_workspace_from_paths_builds_workspace(tmp_path: Path) -> None:
    """
    测试：加载工作区命令 (load_workspace_from_paths) 应该能够正确地从磁盘上的真实文件中
    读取数据文件 (example) 和配置文件 (config)，并将其组装为一个功能完整的 Workspace 实例。
    """
    # Arrange: 在临时目录写入最小可用的 JSON 配置文件和拓扑结构文件
    config_path = tmp_path / "config.json"
    example_path = tmp_path / "example.json"
    config_path.write_text('{"directionalFriction": 1}', encoding="utf-8")
    example_path.write_text(
        '{"v": [[0, 0, 0], [1, 0, 0]], "e": [[0, 1]], "edgeChannel": [2]}',
        encoding="utf-8",
    )

    # Act: 触发指令加载文件
    workspace = load_workspace_from_paths(config_path=config_path, example_path=example_path)

    # Assert: 验证生成的工作区内容与存储的文件内容严格映射
    assert isinstance(workspace, Workspace)
    assert workspace.topology.vertices.shape == (2, 3)  # 解析出了两个三维顶点
    np.testing.assert_array_equal(workspace.topology.edge_channel, np.asarray([2], dtype=np.int32))
    assert workspace.ui.show_channel is True  # 存在通道数据，UI配置应默认展示通道色


def make_workspace() -> Workspace:
    """
    辅助函数：快速构造一个供内存操作测试用的 3顶点 2条边 拓扑的工作区。
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


def test_save_workspace_to_path_writes_legacy_json(tmp_path: Path) -> None:
    """
    测试：保存工作区命令 (save_workspace_to_path) 必须能将当前的运行时内存状态
    （包括所有顶点、边以及 UI 上的选中状态），完整保真地序列化并写入到磁盘目标路径的 JSON 文件中。
    （同时验证路径不存在时系统能自动创建各级父目录）
    """
    # Arrange: 使用基础拓扑，并手动篡改第一个顶点的 UI 选中状态；并指定一个深层目录
    workspace = make_workspace()
    workspace.ui.vertex_status[:] = np.asarray([2, 0, 0], dtype=np.int8)
    output_path = tmp_path / "nested" / "saved.json"

    # Act: 执行保存
    save_workspace_to_path(workspace, output_path)

    # Assert: 从保存的文件再反向读取回来进行校验
    saved = load_legacy_workspace(output_path)
    assert output_path.is_file()  # 确保自动建文件夹机制生效且文件产生
    assert saved.v == workspace.topology.vertices.tolist()  # 顶点坐标一致
    assert saved.e == workspace.topology.edges.tolist()  # 边关系一致
    assert saved.vStatus == [2, 0, 0]  # 选中状态没丢


def test_apply_edit_records_history_and_supports_undo_redo() -> None:
    """
    测试：通用的编辑包装器 (apply_edit) 可以捕获任何修改拓扑的业务函数（例如 add_joint），
    自动记录快照，并对接 WorkspaceHistory 支持来回的撤销 (Undo) 和重做 (Redo) 动作。
    """
    # Arrange: 准备拓扑与空白的历史栈
    workspace = make_workspace()
    history = WorkspaceHistory()

    # Act: 使用包装器执行一次真实拓扑操作——"在索引为1的位置添加联合节点"
    changed = apply_edit(workspace, history, lambda: add_joint(workspace, source_index=1))

    # Assert 1: 验证前置动作修改成功
    assert changed is True
    assert history.can_undo is True
    assert workspace.topology.vertices.shape == (4, 3)  # 原3个点变4个
    assert workspace.topology.edges.shape == (3, 2)  # 原2条边变3条

    # Assert 2: 验证撤销是否起效
    assert history.undo(workspace) is True
    assert workspace.topology.vertices.shape == (3, 3)
    assert workspace.topology.edges.shape == (2, 2)

    # Assert 3: 验证重做是否起效
    assert history.redo(workspace) is True
    assert workspace.topology.vertices.shape == (4, 3)
    assert workspace.topology.edges.shape == (3, 2)


def test_connect_selected_vertices_command_records_history() -> None:
    """
    测试：UI 面板提供的高级指令 connect_selected_vertices ("连接所选顶点")
    能够根据选中状态自动生成对应的边，并将此次操作存入历史堆栈。
    """
    # Arrange: 基础结构中存在 0, 1, 2 三个顶点，手动选中 0 和 2
    workspace = make_workspace()
    history = WorkspaceHistory()
    select_vertex(workspace, 0)
    select_vertex(workspace, 2, additive=True)

    # Act: 发生连接动作
    changed = connect_selected_vertices(workspace, history)

    # Assert: 原本 [0-1] [1-2] 两条边，现在应该增加了一条新边 [0-2]，共3条边
    assert changed is True
    assert workspace.topology.edges.shape == (3, 2)
    assert history.can_undo is True
