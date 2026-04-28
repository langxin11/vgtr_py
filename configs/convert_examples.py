"""
PneuMesh 示例数据转换脚本 (convert_examples.py)

【脚本功能】
该脚本用于将原始的 PneuMesh (旧版) JSON 示例数据批量转换为符合最新 VGTR 规范的目标配置格式。
主要处理以下核心数据的映射与规范化：
1. 节点 (Sites)：坐标转换与固定状态。
2. 连杆 (Rod Groups)：拓扑连接、控制通道绑定及激活状态。
3. 控制组 (Control Groups)：收缩目标与通道属性。
4. 动作脚本 (Script)：多通道动作时序的标准化。

【转换规则】
- 节点：s1, s2, ... (从 0 开始编号)
- 连杆：g1, g2, ... (从 0 开始编号)
- 控制通道：ch0, ch1, ...
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# --- 路径定义 ---
ROOT = Path(__file__).resolve().parents[1]
# 原始示例文件目录 (位于 pneumesh 静态资源中)
SOURCE_DIR = ROOT / "pneumesh" / "public" / "examples"
# 转换后文件的输出目录 (项目配置库)
TARGET_DIR = ROOT / "configs"
DEFAULT_LENGTH_DELTA = 0.1
DEFAULT_FORCE_LIMIT = 20000.0


def _site_name(index: int) -> str:
    """标准化节点命名 (Index -> ID)。"""
    return f"s{index + 1}"


def _rod_name(index: int) -> str:
    """标准化连杆组命名 (Index -> ID)。"""
    return f"g{index + 1}"


def _control_group_name(index: int) -> str:
    """标准化控制组命名 (Index -> ID)。"""
    return f"ch{index}"


def _bool(value: Any, *, default: bool = False) -> bool:
    """工具函数：安全转换布尔值，处理 None 和各种真值类型。"""
    if value is None:
        return default
    return bool(value)


def _distance(left: list[float], right: list[float]) -> float:
    """计算两个三维点之间的欧氏距离。"""
    return math.sqrt(sum((float(r) - float(l)) ** 2 for l, r in zip(left, right, strict=False)))


def _normalize_script(
    script: list[list[Any]], *, num_channels: int, num_actions: int
) -> list[list[bool]]:
    """
    动作脚本标准化逻辑：
    将原始不规则的嵌套列表转换为 (Channels x Actions) 的标准布尔矩阵。
    """
    normalized = [[False for _ in range(num_actions)] for _ in range(num_channels)]
    for i in range(min(num_channels, len(script))):
        row = script[i]
        for j in range(min(num_actions, len(row))):
            normalized[i][j] = _bool(row[j])
    return normalized


def convert_example(data: dict[str, Any]) -> dict[str, Any]:
    """
    数据转换核心算子：
    负责解析旧版字典结构并重新组装为新版规范字典。

    Args:
        data: 原始 JSON 解析后的字典数据。

    Returns:
        转换后的新版字典结构。
    """
    # 1. 提取基础拓扑数据
    # 兼容历史遗留字段 v0 和 v
    vertices = data.get("v") or data.get("v0") or []
    edges = data.get("e") or []
    # 哪些节点是固定在空中的
    fixed_vertices = data.get("fixedVs") or []

    # 2. 提取控制参数
    edge_channels = data.get("edgeChannel") or []
    edge_active = data.get("edgeActive") or []
    script = data.get("script") or []
    inflate_channel = data.get("inflateChannel") or []
    contraction_percent = data.get("contractionPercent") or []

    # 确定控制通道维度和动作步骤总数
    num_channels = int(data.get("numChannels") or len(script) or 0)
    num_actions = int(data.get("numActions") or max((len(row) for row in script), default=1))

    # 3. 构造节点集合 (Sites)
    sites: dict[str, dict[str, Any]] = {}
    for i, pos in enumerate(vertices):
        site: dict[str, Any] = {"pos": [float(coord) for coord in pos]}
        # 如果原始数据标记了固定属性，则写入新格式
        if i < len(fixed_vertices):
            site["fixed"] = _bool(fixed_vertices[i])
        site["projection_target"] = not _bool(site.get("fixed"), default=False)
        sites[_site_name(i)] = site

    # 4. 构造连杆组 (Rod Groups)
    rod_groups: list[dict[str, Any]] = []
    for i, edge in enumerate(edges):
        site1_name = _site_name(int(edge[0]))
        site2_name = _site_name(int(edge[1]))
        anchor_distance = _distance(sites[site1_name]["pos"], sites[site2_name]["pos"])
        min_length = anchor_distance
        max_length = min_length + DEFAULT_LENGTH_DELTA
        enabled = _bool(edge_active[i], default=True) if i < len(edge_active) else True
        control_group = (
            _control_group_name(int(edge_channels[i]))
            if i < len(edge_channels)
            else _control_group_name(i)
        )
        rod_group: dict[str, Any] = {
            "name": _rod_name(i),
            "site1": site1_name,
            "site2": site2_name,
            "actuated": True,
            "enabled": enabled,
            "control_group": control_group,
            "rod_type": "active",
            "length_limits": [min_length, max_length],
            "force_limits": [-DEFAULT_FORCE_LIMIT, DEFAULT_FORCE_LIMIT],
        }

        rod_groups.append(rod_group)

    # 5. 构造控制组属性 (Control Groups)
    control_groups: list[dict[str, Any]] = []
    for i in range(num_channels):
        group: dict[str, Any] = {"name": _control_group_name(i)}

        # 处理气动收缩的默认目标参数 (针对主动收缩通道)
        if i < len(inflate_channel) and i < len(contraction_percent):
            if _bool(inflate_channel[i]):
                group["default_target"] = float(contraction_percent[i])

        control_groups.append(group)

    # 6. 标准化动作时序脚本
    normalized_script = _normalize_script(
        script,
        num_channels=num_channels,
        num_actions=num_actions,
    )

    # 7. 组装最终结果
    return {
        "sites": sites,
        "rod_groups": rod_groups,
        "control_groups": control_groups,
        "script": normalized_script,
        "numActions": num_actions,
    }


def main() -> None:
    """
    主程序：
    遍历源文件夹中的所有 JSON 示例，逐个转换并保存。
    """
    source_files = sorted(SOURCE_DIR.glob("*.json"))
    if not source_files:
        print(f"Warning: 未在 {SOURCE_DIR} 找到任何示例文件。")
        return

    # 确保目标 configs 目录已创建
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for source_path in source_files:
        try:
            # 载入旧数据
            data = json.loads(source_path.read_text(encoding="utf-8"))
            # 执行转换
            converted = convert_example(data)

            # 序列化为新版 JSON (保留非 ASCII 字符，缩进 2 格)
            target_path = TARGET_DIR / source_path.name
            target_path.write_text(
                json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            count += 1
        except Exception as e:
            print(f"Error converting {source_path.name}: {e}")

    print(f"转换完成！共处理 {count} 个示例，已输出至: {TARGET_DIR}")


if __name__ == "__main__":
    main()
