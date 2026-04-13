"""项目的核心入口。

该模块负责组装和构建应用程序，包括配置加载、依赖注入和运行时环境解析，为后续启动界面和引擎打下基础。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ui import VgtrUiApp


SOURCE_ROOT_ENV_VAR = "VGTR_SOURCE_ROOT"
DEFAULT_CONFIG_RELATIVE_PATH = Path("configs") / "config.json"
DEFAULT_EXAMPLE_RELATIVE_PATH = Path("configs") / "crawling-bot.json"


def project_root() -> Path:
    """返回项目根目录。

    Returns:
        项目根目录路径。
    """
    return Path(__file__).resolve().parents[2]


def candidate_source_roots() -> list[Path]:
    """返回候选源码根目录列表。

    搜索顺序为：环境变量路径、当前仓库根目录。
    返回值会做路径解析与去重。

    Returns:
        候选源码根目录列表。
    """
    raw_candidates = [
        Path(os.environ[SOURCE_ROOT_ENV_VAR]).expanduser()
        for _ in [None]
        if SOURCE_ROOT_ENV_VAR in os.environ
    ]
    raw_candidates.extend(
        [
            project_root(),
        ]
    )
    candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in raw_candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
    return candidates


def _has_default_assets(source_root: Path) -> bool:
    """检查源码根目录是否包含默认示例文件。

    Args:
        source_root: 待检查的源码根目录。

    Returns:
        若包含默认示例文件则返回 ``True``，否则返回 ``False``。
    """
    return (source_root / DEFAULT_EXAMPLE_RELATIVE_PATH).is_file()


def resolve_source_root(source_root: str | Path | None = None) -> Path:
    """解析可用的源码根目录。

    如果传入 ``source_root``，会优先验证该路径；否则按候选路径顺序自动搜索。
    返回值保证包含默认示例文件。

    Args:
        source_root: 手动指定的源码根目录。

    Returns:
        可用的源码根目录。

    Raises:
        FileNotFoundError: 找不到包含默认示例文件的源码根目录。
    """
    if source_root is not None:
        candidate = Path(source_root).expanduser().resolve(strict=False)
        if _has_default_assets(candidate):
            return candidate
        raise FileNotFoundError(
            "explicit source root does not contain default example: "
            f"{candidate / DEFAULT_EXAMPLE_RELATIVE_PATH}"
        )

    for candidate in candidate_source_roots():
        if _has_default_assets(candidate):
            return candidate

    raise FileNotFoundError(
        "could not locate default VGTR example; pass --example, "
        f"or set {SOURCE_ROOT_ENV_VAR} to a source tree that contains "
        f"{DEFAULT_EXAMPLE_RELATIVE_PATH}"
    )


def default_config_path(source_root: str | Path | None = None) -> Path:
    """返回默认配置文件路径。

    Args:
        source_root: 手动指定的源码根目录。

    Returns:
        默认配置文件路径。
    """
    return resolve_source_root(source_root) / DEFAULT_CONFIG_RELATIVE_PATH


def optional_default_config_path(source_root: str | Path | None = None) -> Path | None:
    """返回默认配置文件路径；若不存在则返回 ``None``。"""
    candidate = default_config_path(source_root)
    if candidate.is_file():
        return candidate
    return None


def default_example_path(source_root: str | Path | None = None) -> Path:
    """返回默认示例文件路径。

    Args:
        source_root: 手动指定的源码根目录。

    Returns:
        默认示例文件路径。
    """
    return resolve_source_root(source_root) / DEFAULT_EXAMPLE_RELATIVE_PATH


def resolve_runtime_paths(
    *,
    config_path: Path | None,
    example_path: Path | None,
) -> tuple[Path | None, Path]:
    """解析运行时使用的配置与示例路径。

    当参数缺失时，会从自动检测的源码根目录补齐默认路径。

    Args:
        config_path: 手动指定的配置文件路径。
        example_path: 手动指定的示例文件路径。

    Returns:
        配置文件路径（可空）与示例文件路径组成的元组。
    """
    if config_path is not None and example_path is not None:
        return config_path, example_path

    source_root = resolve_source_root()
    return (
        config_path or optional_default_config_path(source_root),
        example_path or default_example_path(source_root),
    )


def build_app(
    *,
    config_path: Path | None,
    example_path: Path,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> VgtrUiApp:
    """创建并返回 ``VgtrUiApp`` 实例。

    Args:
        config_path: 配置文件路径，可选。
        example_path: 示例文件路径。
        host: 服务监听地址。
        port: 服务端口。

    Returns:
        已初始化的 ``VgtrUiApp`` 实例。
    """
    from .ui import create_app

    return create_app(
        config_path=config_path,
        example_path=example_path,
        host=host,
        port=port,
    )
