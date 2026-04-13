"""
PneuMesh 示例数据转换脚本 (convert_examples.py)

该脚本用于将原始的 PneuMesh JSON 示例数据批量转换为符合最新规范的目标配置格式。
主要处理节点 (sites)、连杆 (rod_groups)、控制组 (control_groups) 以及动作脚本 (script) 等核心数据的映射和规范化。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 路径配置
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "pneumesh" / "public" / "examples"  # 原始示例文件目录
TARGET_DIR = ROOT / "configs"                           # 转换后文件的输出目录


def _site_name(index: int) -> str:
    """生成标准化节点名称 (例如: 's1', 's2')。"""
    return f"s{index + 1}"


def _rod_name(index: int) -> str:
    """生成标准化连杆名称 (例如: 'g1', 'g2')。"""
    return f"g{index + 1}"


def _control_group_name(index: int) -> str:
    """生成标准化控制组名称 (例如: 'ch0', 'ch1')。"""
    return f"ch{index}"


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def _normalize_script(script: list[list[Any]], *, num_channels: int, num_actions: int) -> list[list[bool]]:
    normalized = [[False for _ in range(num_actions)] for _ in range(num_channels)]
    for i in range(min(num_channels, len(script))):
        row = script[i]
        for j in range(min(num_actions, len(row))):
            normalized[i][j] = _bool(row[j])
    return normalized


def convert_example(data: dict[str, Any]) -> dict[str, Any]:
    """
    将原始示例数据结构解析并转换为目标配置格式。

    Args:
        data: 从原始 JSON 文件中读取的字典数据。

    Returns:
        符合最新规范的字典结构，包含 sites, rod_groups, control_groups, script 等。
    """
    # 提取基础拓扑和状态数据 (兼容不同的历史字段命名，如 v 和 v0)
    vertices = data.get("v") or data.get("v0") or []
    edges = data.get("e") or []
    fixed_vertices = data.get("fixedVs") or []
    
    # 提取控制与动作数据
    edge_channels = data.get("edgeChannel") or []
    edge_active = data.get("edgeActive") or []
    script = data.get("script") or []
    inflate_channel = data.get("inflateChannel") or []
    contraction_percent = data.get("contractionPercent") or []
    
    # 计算通道数和动作总数 (带有默认容错处理)
    num_channels = int(data.get("numChannels") or len(script) or 0)
    num_actions = int(data.get("numActions") or max((len(row) for row in script), default=1))

    # 构建 Sites (节点) 集合
    sites: dict[str, dict[str, Any]] = {}
    for i, pos in enumerate(vertices):
        site: dict[str, Any] = {"pos": [float(coord) for coord in pos]}
        # 仅当有固定点配置时才写入 fixed 属性
        if i < len(fixed_vertices):
            site["fixed"] = _bool(fixed_vertices[i])
        sites[_site_name(i)] = site

    # 构建 Rod Groups (连杆组) 列表
    rod_groups: list[dict[str, Any]] = []
    for i, edge in enumerate(edges):
        rod_group: dict[str, Any] = {
            "name": _rod_name(i),
            "site1": _site_name(int(edge[0])),
            "site2": _site_name(int(edge[1])),
            "actuated": True,
        }
        
        # 绑定控制通道和激活状态
        if i < len(edge_channels):
            rod_group["control_group"] = _control_group_name(int(edge_channels[i]))
        if i < len(edge_active):
            rod_group["enabled"] = _bool(edge_active[i], default=True)
            
        rod_groups.append(rod_group)

    # 构建 Control Groups (控制组) 列表
    control_groups: list[dict[str, Any]] = []
    for i in range(num_channels):
        group: dict[str, Any] = {"name": _control_group_name(i)}
        
        # 注入充气通道的默认收缩目标参数
        if i < len(inflate_channel) and i < len(contraction_percent):
            if _bool(inflate_channel[i]):
                group["default_target"] = float(contraction_percent[i])
                
        control_groups.append(group)

    normalized_script = _normalize_script(
        script,
        num_channels=num_channels,
        num_actions=num_actions,
    )

    return {
        "sites": sites,
        "rod_groups": rod_groups,
        "control_groups": control_groups,
        "script": normalized_script,
        "numActions": num_actions,
    }


def main() -> None:
    """
    主执行入口：
    扫描源目录中的所有 JSON 文件，依次进行数据格式转换，并输出至目标目录。
    """
    source_files = sorted(SOURCE_DIR.glob("*.json"))
    if not source_files:
        raise SystemExit(f"Error: No source examples found in {SOURCE_DIR}")

    # 确保目标目录存在
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for source_path in source_files:
        # 读取与转换
        data = json.loads(source_path.read_text(encoding="utf-8"))
        converted = convert_example(data)
        
        # 序列化写入
        target_path = TARGET_DIR / source_path.name
        target_path.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(target_path.name)

    # 打印转换报告
    print(f"Successfully generated {len(generated)} files into {TARGET_DIR}:")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
