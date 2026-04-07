"""运动学相关计算模块。

管理并提供处理骨架或网格的正逆运动学方程等核心数学模型的基础例程。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .workspace import Workspace

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class RodKinematics:
    """杆组派生几何。"""

    direction: FloatArray
    anchor_distance: FloatArray
    midpoint: FloatArray
    shared_slide_q: FloatArray
    sleeve_pose: FloatArray
    left_tip_pos: FloatArray
    right_tip_pos: FloatArray


def derive_rod_kinematics(workspace: Workspace) -> RodKinematics:
    """由锚点位置和杆组参数推导套筒与杆端几何。"""
    topology = workspace.topology
    rod_count = topology.rod_anchors.shape[0]
    if rod_count == 0:
        return RodKinematics(
            direction=np.zeros((0, 3), dtype=np.float64),
            anchor_distance=np.zeros(0, dtype=np.float64),
            midpoint=np.zeros((0, 3), dtype=np.float64),
            shared_slide_q=np.zeros(0, dtype=np.float64),
            sleeve_pose=np.zeros((0, 7), dtype=np.float64),
            left_tip_pos=np.zeros((0, 3), dtype=np.float64),
            right_tip_pos=np.zeros((0, 3), dtype=np.float64),
        )

    left_tip = topology.anchor_pos[topology.rod_anchors[:, 0]]
    right_tip = topology.anchor_pos[topology.rod_anchors[:, 1]]
    direction = right_tip - left_tip
    anchor_distance = np.linalg.norm(direction, axis=1)
    safe = np.where(anchor_distance > 1e-12, anchor_distance, 1.0)
    direction = direction / safe[:, None]
    midpoint = (left_tip + right_tip) * 0.5

    sleeve_half = np.zeros(rod_count, dtype=np.float64)
    if topology.rod_sleeve_half.shape[0] == rod_count:
        sleeve_half = topology.rod_sleeve_half[:, 2]
    shared_slide_q = np.maximum(anchor_distance * 0.5 - sleeve_half, 0.0)

    sleeve_pose = np.zeros((rod_count, 7), dtype=np.float64)
    sleeve_pose[:, :3] = midpoint
    for i in range(rod_count):
        sleeve_pose[i, 3:] = quaternion_from_z_axis(direction[i])

    return RodKinematics(
        direction=direction,
        anchor_distance=anchor_distance,
        midpoint=midpoint,
        shared_slide_q=shared_slide_q,
        sleeve_pose=sleeve_pose,
        left_tip_pos=left_tip.copy(),
        right_tip_pos=right_tip.copy(),
    )


def quaternion_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """返回将局部 +Z 轴旋转到目标方向的四元数，格式为 wxyz。"""
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    target = direction / norm
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot >= 1.0 - 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot <= -1.0 + 1e-8:
        return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    axis = np.cross(source, target)
    axis /= np.linalg.norm(axis)
    half_angle = np.arccos(dot) / 2.0
    sin_half_angle = np.sin(half_angle)
    return np.asarray(
        [
            np.cos(half_angle),
            axis[0] * sin_half_angle,
            axis[1] * sin_half_angle,
            axis[2] * sin_half_angle,
        ],
        dtype=np.float64,
    )
