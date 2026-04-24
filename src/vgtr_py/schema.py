"""数据模式与序列化层。

提供基于 msgspec 的工作区结构数据格式的校验与转换逻辑。专门用于处理磁盘文件与内存结构的双向映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec


class SiteFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """变几何桁架格式中的锚点定义。"""

    pos: list[float]
    description: str | None = None
    radius: float | None = None
    fixed: bool | int | None = None
    mass: float | None = None
    projection_target: bool | int = False
    metadata: dict[str, Any] | None = None


class RodGroupFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """变几何桁架格式中的杆组定义。"""

    name: str
    description: str | None = None
    site1: str
    site2: str
    rod_type: str | None = None
    actuated: bool | int = False
    enabled: bool | int = True
    control_group: str | None = None
    group_mass: float | None = None
    rod_radius: float | None = None
    sleeve_radius: float | None = None
    sleeve_display_half_length_ratio: float | None = None
    rest_length: float | None = None
    min_length: float | None = None
    max_contraction: float | None = None
    length_limits: list[float] = msgspec.field(default_factory=list)
    force_limits: list[float] = msgspec.field(default_factory=list)
    metadata: dict[str, Any] | None = None


class ControlGroupFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """控制组定义。"""

    name: str
    description: str | None = None
    color: list[float] = msgspec.field(default_factory=list)
    default_target: float | None = None
    enabled: bool | int = True
    metadata: dict[str, Any] | None = None


class WorkspaceFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """变几何桁架机器人工作区主格式。"""

    description: str | None = None
    sites: dict[str, SiteFile]
    rod_groups: list[RodGroupFile]
    control_groups: list[ControlGroupFile] = msgspec.field(default_factory=list)
    script: list[list[bool | int | float]] = msgspec.field(default_factory=list)
    num_actions: int | None = msgspec.field(default=None, name="numActions")
    anchor_frames: list[list[list[float]]] = msgspec.field(default_factory=list)
    metadata: dict[str, Any] | None = None


WorkspaceFileData = WorkspaceFile


def _decode_json_object(raw: bytes | str) -> dict[str, Any]:
    """将原始 JSON 解码为字典对象。

    Args:
        raw: JSON 字节串或字符串。

    Returns:
        解码后的字典。

    Raises:
        TypeError: 若解码结果不是字典。
    """
    decoded = msgspec.json.decode(raw)
    if not isinstance(decoded, dict):
        raise TypeError("workspace file must decode to a JSON object")
    return decoded


def decode_workspace_file(raw: bytes | str) -> WorkspaceFileData:
    """解码 VGTR 工作区文件。"""
    decoded = _decode_json_object(raw)
    if "sites" not in decoded or "rod_groups" not in decoded:
        raise ValueError(
            "workspace JSON must use VGTR schema with 'sites' and 'rod_groups'; "
            "legacy PneuMesh format is no longer supported"
        )

    rod_groups = decoded.get("rod_groups")
    if isinstance(rod_groups, list):
        for i, rod_group in enumerate(rod_groups):
            if isinstance(rod_group, dict) and "sleeve_half" in rod_group:
                raise ValueError(
                    f"rod_groups[{i}] uses deprecated field 'sleeve_half'; "
                    "use 'sleeve_radius' and 'sleeve_display_half_length_ratio'"
                )

    return msgspec.convert(decoded, type=WorkspaceFile)


def encode_workspace_file(workspace_file: WorkspaceFileData, *, indent: int = 2) -> bytes:
    """编码任一受支持的工作区文件格式。"""
    encoded = msgspec.json.encode(workspace_file)
    return msgspec.json.format(encoded, indent=indent)


def load_workspace_file(path: str | Path) -> WorkspaceFileData:
    """加载 VGTR 工作区文件。"""
    return decode_workspace_file(Path(path).read_bytes())


def dump_workspace_file(
    path: str | Path,
    workspace_file: WorkspaceFileData,
    *,
    indent: int = 2,
) -> None:
    """将工作区对象持久化到指定路径。"""
    Path(path).write_bytes(encode_workspace_file(workspace_file, indent=indent))


def as_plain_dict(workspace_file: WorkspaceFileData) -> dict[str, Any]:
    """将结构化工作区对象转换为纯 Python 字典。"""
    return msgspec.to_builtins(workspace_file)  # type: ignore[return-value]
