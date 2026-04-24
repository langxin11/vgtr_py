"""从锚点目标位置到控制组目标值的投影。

将高层的关键锚点目标位置（如末端执行器位置）通过运动学关系
反解为底层控制组的目标激活值 [0, 1]，供 Simulator 执行。
"""

from __future__ import annotations

import numpy as np

from .data import VGTRData
from .model import VGTRModel
from .workspace import ROD_TYPE_ACTIVE


def project_anchor_targets(
    model: VGTRModel,
    data: VGTRData,
    target_positions: np.ndarray,
) -> np.ndarray:
    """将关键锚点目标位置投影为控制组目标值。

    对每根主动杆，计算其在目标位形下的期望长度，再归一化为 [0, 1] 控制值，
    最后按控制组取平均。

    Args:
        model: 只读静态模型。
        data: 当前运行时状态（用于读取当前位形）。
        target_positions: 关键锚点的目标位置数组，shape (P, 3)。

    Returns:
        控制组目标值数组，shape (control_group_count,)。
    """
    if model.projection_anchor_indices.size == 0 or model.control_group_count == 0:
        return np.zeros(model.control_group_count, dtype=np.float64)

    target = np.asarray(target_positions, dtype=np.float64)
    reshaped = target.reshape(model.projection_anchor_indices.shape[0], 3)
    projected_qpos = data.qpos.copy()
    projected_qpos[model.projection_anchor_indices] = reshaped

    control_target = data.ctrl_target.copy()
    contributions = np.zeros(model.control_group_count, dtype=np.float64)
    counts = np.zeros(model.control_group_count, dtype=np.int32)

    for rod_index in np.flatnonzero(model.rod_type == ROD_TYPE_ACTIVE):
        anchor_pair = model.rod_anchors[rod_index]
        desired_length = float(
            np.linalg.norm(projected_qpos[anchor_pair[1]] - projected_qpos[anchor_pair[0]])
        )
        min_length, max_length = model.rod_length_limits[rod_index]
        if max_length <= min_length + 1e-9:
            normalized = 0.0
        else:
            normalized = (max_length - desired_length) / (max_length - min_length)
        group_index = int(model.rod_control_group[rod_index])
        contributions[group_index] += float(np.clip(normalized, 0.0, 1.0))
        counts[group_index] += 1

    mask = counts > 0
    control_target[mask] = contributions[mask] / counts[mask]
    np.clip(control_target, 0.0, 1.0, out=control_target)
    return control_target
