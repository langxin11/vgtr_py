"""动态仿真数据。

定义单仿真实例的可变运行时状态容器，包含质点位置、速度、受力、
杆级统计量（长度、目标长度、轴力、应变、堵转标志）及控制量。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .model import VGTRModel

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class VGTRData:
    """单仿真实例的可变运行时状态。

    Attributes:
        qpos: 锚点位置数组，shape (anchor_count, 3)。
        qvel: 锚点速度数组，shape (anchor_count, 3)。
        forces: 锚点合力数组，shape (anchor_count, 3)。
        ctrl: 当前控制组激活值 [0, 1]，shape (control_group_count,)。
        ctrl_target: 控制组目标值 [0, 1]，shape (control_group_count,)。
        rod_length: 杆组当前长度，shape (rod_count,)。
        rod_target_length: 杆组目标长度，shape (rod_count,)。
        rod_target_override: 杆组目标长度覆盖值（per-rod 模式使用），shape (rod_count,)。
        rod_axial_force: 杆组轴向力，shape (rod_count,)。
        rod_strain: 杆组应变（长度误差/目标长度），shape (rod_count,)。
        rod_stalled: 杆组堵转标志，shape (rod_count,)。
        contact_mask: 锚点地面接触标志，shape (anchor_count,)。
        time: 累计仿真时间。
        step_count: 已执行仿真步数。
        i_action: 当前动作索引（脚本驱动时使用）。
        i_action_prev: 上一动作索引。
        record_frames: 是否记录轨迹帧。
        frames: 记录的轨迹帧列表，每帧为 qpos 的 list 副本。
    """

    qpos: FloatArray
    qvel: FloatArray
    forces: FloatArray
    ctrl: FloatArray
    ctrl_target: FloatArray
    rod_length: FloatArray
    rod_target_length: FloatArray
    rod_target_override: FloatArray
    rod_axial_force: FloatArray
    rod_strain: FloatArray
    rod_stalled: BoolArray
    contact_mask: BoolArray
    time: float = 0.0
    step_count: int = 0
    i_action: int = 0
    i_action_prev: int = 0
    record_frames: bool = False
    frames: list[list[list[float]]] | None = None


def make_data(model: VGTRModel, *, seed: int | None = None) -> VGTRData:
    """按模型默认值分配并初始化运行时数据。

    Args:
        model: 编译后的只读模型。
        seed: 随机种子（目前保留，未来用于随机化初始状态）。

    Returns:
        已初始化的 VGTRData 实例。
    """
    data = VGTRData(
        qpos=model.anchor_rest_pos.copy(),
        qvel=np.zeros_like(model.anchor_rest_pos),
        forces=np.zeros_like(model.anchor_rest_pos),
        ctrl=model.control_group_default_target.copy(),
        ctrl_target=model.control_group_default_target.copy(),
        rod_length=np.zeros(model.rod_count, dtype=np.float64),
        rod_target_length=np.zeros(model.rod_count, dtype=np.float64),
        rod_target_override=np.full(model.rod_count, np.nan, dtype=np.float64),
        rod_axial_force=np.zeros(model.rod_count, dtype=np.float64),
        rod_strain=np.zeros(model.rod_count, dtype=np.float64),
        rod_stalled=np.zeros(model.rod_count, dtype=np.bool_),
        contact_mask=np.zeros(model.anchor_count, dtype=np.bool_),
    )
    reset_data(model, data, seed=seed)
    return data


def reset_data(model: VGTRModel, data: VGTRData, *, seed: int | None = None) -> None:
    """将运行时数据重置为模型的静止状态。

    Args:
        model: 编译后的只读模型。
        data: 待重置的可变运行时状态。
        seed: 随机种子（目前保留）。
    """
    del seed  # reserved for future randomized resets
    data.qpos = model.anchor_rest_pos.copy()
    data.qvel = np.zeros_like(model.anchor_rest_pos)
    data.forces = np.zeros_like(model.anchor_rest_pos)
    data.ctrl = model.control_group_default_target.copy()
    data.ctrl_target = model.control_group_default_target.copy()
    data.rod_length = _rod_lengths(data.qpos, model.rod_anchors)
    data.rod_target_length = model.rod_rest_length.copy()
    data.rod_target_override = np.full(model.rod_count, np.nan, dtype=np.float64)
    data.rod_axial_force = np.zeros(model.rod_count, dtype=np.float64)
    data.rod_strain = np.zeros(model.rod_count, dtype=np.float64)
    data.rod_stalled = np.zeros(model.rod_count, dtype=np.bool_)
    data.contact_mask = np.zeros(model.anchor_count, dtype=np.bool_)
    data.time = 0.0
    data.step_count = 0
    data.i_action = 0
    data.i_action_prev = 0
    if data.record_frames:
        data.frames = []
    elif data.frames is None:
        data.frames = None


def _rod_lengths(qpos: FloatArray, rod_anchors: NDArray[np.int32]) -> FloatArray:
    """根据锚点位置计算杆组长度。

    Args:
        qpos: 锚点位置数组，shape (N, 3)。
        rod_anchors: 杆组两端锚点索引，shape (R, 2)。

    Returns:
        各杆组当前长度，shape (R,)。
    """
    if rod_anchors.size == 0:
        return np.zeros(0, dtype=np.float64)
    diff = qpos[rod_anchors[:, 1]] - qpos[rod_anchors[:, 0]]
    return np.linalg.norm(diff, axis=1)
