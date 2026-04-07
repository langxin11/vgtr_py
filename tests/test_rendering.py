import numpy as np

from vgtr_py.rendering import _quaternion_from_z_axis


def rotate_with_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    根据给定的四元数旋转一个三维向量。

    参数:
        vector: 需要旋转的原始三维向量（长度为3的 numpy 数组）
        quaternion: 表示旋转的四元数 [w, x, y, z]（长度为4的 numpy 数组）

    返回:
        旋转后的三维向量
    """
    w, x, y, z = quaternion
    q_vec = np.asarray([x, y, z], dtype=np.float64)
    uv = np.cross(q_vec, vector)
    uuv = np.cross(q_vec, uv)
    return vector + 2.0 * (w * uv + uuv)


def test_quaternion_from_z_axis_keeps_z_direction() -> None:
    """
    测试：如果目标方向已经是 Z 轴正向，则生成的四元数不应对 Z 轴向量产生任何旋转效果。
    """
    # Arrange: 构造目标向量 (0, 0, 2)
    quaternion = _quaternion_from_z_axis(np.asarray([0.0, 0.0, 2.0], dtype=np.float64))

    # Act: 尝试用该四元数旋转标准的 Z 轴单位向量 (0, 0, 1)
    rotated = rotate_with_quaternion(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), quaternion)

    # Assert: 旋转后的向量必须保持为 (0, 0, 1)
    np.testing.assert_allclose(rotated, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def test_quaternion_from_z_axis_rotates_to_x_direction() -> None:
    """
    测试：如果目标方向是 X 轴方向，则生成的四元数应该将 Z 轴基础向量正确旋转到 X 轴方向。
    """
    # Arrange: 构造指向 X 轴的目标向量 (3, 0, 0)
    quaternion = _quaternion_from_z_axis(np.asarray([3.0, 0.0, 0.0], dtype=np.float64))

    # Act: 尝试用该四元数旋转标准的 Z 轴单位向量 (0, 0, 1)
    rotated = rotate_with_quaternion(np.asarray([0.0, 0.0, 1.0], dtype=np.float64), quaternion)

    # Assert: 旋转后的向量必须是指向 X 轴的单位向量 (1, 0, 0)，考虑到计算误差设置了 atol
    np.testing.assert_allclose(rotated, np.asarray([1.0, 0.0, 0.0], dtype=np.float64), atol=1e-7)
