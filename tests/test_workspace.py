import numpy as np

from vgtr_py.config import default_config
from vgtr_py.kinematics import derive_rod_kinematics
from vgtr_py.schema import ControlGroupFile, RodGroupFile, SiteFile, WorkspaceFile
from vgtr_py.workspace import Workspace


def make_vgtr_workspace() -> Workspace:
    """
    辅助函数：构造一个基于 VGTR (变几何桁架) 格式的基准工作区实例。
    其中包含 2个锚点(site) 和 1组连接两点的驱动杆(rod_group)，并指定了它们的物理及控制属性。
    """
    workspace_file = WorkspaceFile(
        sites={
            "s1": SiteFile(pos=[0.0, 0.0, 0.0], fixed=True, radius=0.01, mass=2.0),
            "s2": SiteFile(pos=[2.0, 0.0, 0.0], radius=0.02, mass=3.0),
        },
        rod_groups=[
            RodGroupFile(
                name="g12",
                site1="s1",
                site2="s2",
                actuated=True,
                control_group="drive_A",
                rest_length=2.0,
                min_length=1.2,
                rod_radius=0.03,
                sleeve_half=[0.05, 0.05, 0.2],
                group_mass=4.0,
            )
        ],
        control_groups=[
            ControlGroupFile(name="drive_A", color=[1.0, 0.0, 0.0], default_target=0.25)
        ],
        script=[[0.0, 1.0]],
        numActions=2,
    )
    return Workspace.from_file_data(workspace_file, default_config())


def test_workspace_from_vgtr_file_populates_new_runtime_fields() -> None:
    """
    测试：验证从 VGTR 文件格式解析 Workspace 时，能在运行时数据结构中正确填充新的字段
    （如解析出的拓扑数组，属性列表等）。
    """
    # Arrange & Act: 从配置文件生成工作区
    workspace = make_vgtr_workspace()

    # Assert: 验证各属性是否按照文件的描述被正确解包到 numpy 维度数组或列表中
    assert workspace.storage_format == "vgtr"  # 识别到了 vgtr 文件格式
    assert workspace.topology.anchor_ids == ["s1", "s2"]
    assert workspace.topology.rod_group_ids == ["g12"]

    # 验证驱动杆两端的连接关系映射是否正确 (s1 -> index 0, s2 -> index 1)
    np.testing.assert_array_equal(
        workspace.topology.rod_anchors, np.asarray([[0, 1]], dtype=np.int32)
    )

    # 验证物理参数的正确映射
    np.testing.assert_allclose(
        workspace.topology.rod_rest_length, np.asarray([2.0], dtype=np.float64)
    )
    np.testing.assert_allclose(
        workspace.topology.rod_min_length, np.asarray([1.2], dtype=np.float64)
    )

    # 验证 UI 颜色配置（浮点 RGB 转 Uint8 (255)）
    np.testing.assert_array_equal(
        workspace.script.control_group_colors,
        np.asarray([[255, 0, 0]], dtype=np.uint8),
    )
    # 验证控制组目标值的填充
    np.testing.assert_allclose(
        workspace.physics.control_group_target,
        np.asarray([0.25], dtype=np.float64),
    )


def test_vgtr_workspace_roundtrips_to_workspace_file() -> None:
    """
    测试：序列化到文件对象和反序列化的过程互不干涉（Roundtrip）。
    对象转换为文件描述信息后内容必须一致。
    """
    # Arrange: 准备基础的工作区
    workspace = make_vgtr_workspace()

    # Act: 触发一次保存流程 (序列化到对象模型)
    dumped = workspace.to_workspace_file()

    # Assert: 校验保存出来的模型数据是否保真
    assert list(dumped.sites) == ["s1", "s2"]
    assert dumped.rod_groups[0].name == "g12"
    assert dumped.rod_groups[0].site1 == "s1"
    assert dumped.rod_groups[0].site2 == "s2"
    assert dumped.rod_groups[0].min_length == 1.2
    assert dumped.control_groups[0].name == "drive_A"
    assert dumped.script == [[0.0, 1.0]]


def test_derive_rod_kinematics_tracks_anchor_geometry() -> None:
    """
    测试：运动学推导引擎（derive_rod_kinematics）能否正确追踪并计算驱动杆基于两端锚点在真实空间的几何形状及姿态。
    """
    # Arrange: 构建基础工作区实例
    workspace = make_vgtr_workspace()

    # Act: 执行计算核心，导出这一个帧步对应的运动学信息
    kinematics = derive_rod_kinematics(workspace)

    # Assert: 验证基于锚点 s1(0,0,0) 和 s2(2,0,0) 得出的各项计算指标
    np.testing.assert_allclose(
        kinematics.anchor_distance, np.asarray([2.0], dtype=np.float64)
    )  # 杆长
    np.testing.assert_allclose(
        kinematics.direction, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    )  # 朝向X轴
    np.testing.assert_allclose(
        kinematics.midpoint, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    )  # 中点位置
    np.testing.assert_allclose(
        kinematics.left_tip_pos, np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    )  # 左端点
    np.testing.assert_allclose(
        kinematics.right_tip_pos, np.asarray([[2.0, 0.0, 0.0]], dtype=np.float64)
    )  # 右端点
    # 验证共享滑动状态（shared_slide_q）：基于长度计算的比率值
    np.testing.assert_allclose(kinematics.shared_slide_q, np.asarray([0.8], dtype=np.float64))
