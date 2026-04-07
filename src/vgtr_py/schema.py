"""数据模式与序列化层。

提供基于 msgspec 的工作区结构数据格式的校验与转换逻辑。专门用于处理磁盘文件与内存结构的双向映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec


class LegacyWorkspaceFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """旧版 PneuMesh 工作区文件架构。"""

    v: list[list[float]]
    e: list[list[int]]
    v0: list[list[float]] = []
    fixedVs: list[bool | int] = []
    lMax: list[float] = []
    edgeChannel: list[int] = []
    edgeActive: list[bool | int] = []
    script: list[list[bool | int | float]] = []
    maxContraction: list[float] = []
    vStatus: list[int] = []
    eStatus: list[int] = []
    fStatus: list[int] = []
    numChannels: int | None = None
    numActions: int | None = None
    inflateChannel: list[bool | int] = []
    contractionPercent: list[float] = []
    v_frames: list[list[list[float]]] = []


class SiteFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """变几何桁架格式中的锚点定义。"""

    pos: list[float]
    radius: float | None = None
    fixed: bool | int | None = None
    mass: float | None = None


class RodGroupFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """变几何桁架格式中的杆组定义。"""

    name: str
    site1: str
    site2: str
    actuated: bool | int = False
    enabled: bool | int = True
    control_group: str | None = None
    group_mass: float | None = None
    rod_radius: float | None = None
    sleeve_half: list[float] = []
    rest_length: float | None = None
    min_length: float | None = None
    max_contraction: float | None = None


class ControlGroupFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """控制组定义。"""

    name: str
    color: list[float] = []
    default_target: float | None = None
    enabled: bool | int = True


class WorkspaceFile(msgspec.Struct, kw_only=True, omit_defaults=True):
    """变几何桁架机器人工作区主格式。"""

    sites: dict[str, SiteFile]
    rod_groups: list[RodGroupFile]
    control_groups: list[ControlGroupFile] = []
    script: list[list[bool | int | float]] = []
    numActions: int | None = None
    anchor_frames: list[list[list[float]]] = []


WorkspaceFileData = LegacyWorkspaceFile | WorkspaceFile


def _decode_json_object(raw: bytes | str) -> dict[str, Any]:
    decoded = msgspec.json.decode(raw)
    if not isinstance(decoded, dict):
        raise TypeError("workspace file must decode to a JSON object")
    return decoded


def decode_workspace_file(raw: bytes | str) -> WorkspaceFileData:
    """自动识别并解码旧版或 VGTR 工作区文件。"""
    decoded = _decode_json_object(raw)
    if "sites" in decoded and "rod_groups" in decoded:
        return msgspec.convert(decoded, type=WorkspaceFile)
    return msgspec.convert(decoded, type=LegacyWorkspaceFile)


def encode_workspace_file(workspace_file: WorkspaceFileData, *, indent: int = 2) -> bytes:
    """编码任一受支持的工作区文件格式。"""
    encoded = msgspec.json.encode(workspace_file)
    return msgspec.json.format(encoded, indent=indent)


def load_workspace_file(path: str | Path) -> WorkspaceFileData:
    """自动识别并加载旧版或 VGTR 工作区文件。"""
    return decode_workspace_file(Path(path).read_bytes())


def dump_workspace_file(
    path: str | Path,
    workspace_file: WorkspaceFileData,
    *,
    indent: int = 2,
) -> None:
    """将工作区对象持久化到指定路径。"""
    Path(path).write_bytes(encode_workspace_file(workspace_file, indent=indent))


def decode_legacy_workspace(raw: bytes | str) -> LegacyWorkspaceFile:
    """将给定 JSON 数据解码为旧版工作区对象。"""
    return msgspec.json.decode(raw, type=LegacyWorkspaceFile)


def encode_legacy_workspace(workspace_file: LegacyWorkspaceFile, *, indent: int = 2) -> bytes:
    """将 LegacyWorkspaceFile 对象编码为 JSON 字节流。"""
    encoded = msgspec.json.encode(workspace_file)
    return msgspec.json.format(encoded, indent=indent)


def load_legacy_workspace(path: str | Path) -> LegacyWorkspaceFile:
    """读取给定路径并解析旧版工作区。"""
    return decode_legacy_workspace(Path(path).read_bytes())


def dump_legacy_workspace(
    path: str | Path,
    workspace_file: LegacyWorkspaceFile,
    *,
    indent: int = 2,
) -> None:
    """将旧版工作区对象持久化到指定路径。"""
    Path(path).write_bytes(encode_legacy_workspace(workspace_file, indent=indent))


def as_plain_dict(workspace_file: WorkspaceFileData) -> dict[str, Any]:
    """将结构化工作区对象转换为纯 Python 字典。"""
    return msgspec.to_builtins(workspace_file)  # type: ignore[return-value]
