import importlib
from pathlib import Path

import pytest

from vgtr_py.app import (
    DEFAULT_CONFIG_RELATIVE_PATH,
    DEFAULT_EXAMPLE_RELATIVE_PATH,
    SOURCE_ROOT_ENV_VAR,
    default_config_path,
    default_example_path,
    resolve_runtime_paths,
    resolve_source_root,
)


def make_source_root(tmp_path: Path) -> Path:
    """
    辅助函数：在临时的文件系统中构建一个模拟的项目根目录环境。
    包含必需的默认配置文件和示例文件结构。
    """
    source_root = tmp_path / "pneumesh"
    config_path = source_root / DEFAULT_CONFIG_RELATIVE_PATH
    example_path = source_root / DEFAULT_EXAMPLE_RELATIVE_PATH

    # 创建逐级父目录
    config_path.parent.mkdir(parents=True, exist_ok=True)
    example_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入基础测试内容
    config_path.write_text("{}", encoding="utf-8")
    # 使用正确的 VGTR 模式写入示例
    example_path.write_text('{"sites": {}, "rod_groups": []}', encoding="utf-8")
    return source_root


def test_default_paths_use_env_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    测试：如果环境变量中指定了 SOURCE_ROOT_ENV_VAR，应用应该优先且正确地
    将其作为解析默认配置和数据文件路径的基准根目录。
    """
    # Arrange: 准备带有标准文件的假根目录，并劫持环境变量指向它
    source_root = make_source_root(tmp_path)
    monkeypatch.setenv(SOURCE_ROOT_ENV_VAR, str(source_root))

    # Act & Assert: 断言解析出的根目录和组合出的子路径是符合预期的
    assert resolve_source_root() == source_root.resolve()
    assert default_config_path() == source_root / DEFAULT_CONFIG_RELATIVE_PATH
    assert default_example_path() == source_root / DEFAULT_EXAMPLE_RELATIVE_PATH


def test_resolve_runtime_paths_supports_partial_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试：如果只提供部分特定的路径（例如仅覆盖 config_path，而不覆盖 example_path），
    系统能够智能地处理：提供的用指定的新路径，没提供的去默认根目录下找。
    """
    # Arrange: 构造假根目录，劫持环境变量，同时在系统另一处伪造一个用户自定义的配置文件
    source_root = make_source_root(tmp_path)
    explicit_config = tmp_path / "custom-config.json"
    explicit_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(SOURCE_ROOT_ENV_VAR, str(source_root))

    # Act: 模拟用户只传入了 config 的绝对路径，example 传入 None (使用默认)
    resolved_config, resolved_example = resolve_runtime_paths(
        config_path=explicit_config,
        example_path=None,
    )

    # Assert: 验证混合解析的准确性
    assert resolved_config == explicit_config  # 使用了自定义覆盖的配置文件
    assert (
        resolved_example == source_root / DEFAULT_EXAMPLE_RELATIVE_PATH
    )  # 自动回退找了默认的例子文件


def test_resolve_source_root_raises_clear_error_when_defaults_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    测试：如果根目录找不到（环境变量没配置，且缺省候选列表也都没有有效路径），
    必须抛出一个清晰明了的 FileNotFoundError，提示用户手动通过参数指定配置路径。
    """
    # Arrange: 强制抹除环境变量，并且将内置找根目录的 fallback 候选列表硬插空数组
    app_module = importlib.import_module("vgtr_py.app")
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(app_module, "candidate_source_roots", lambda: [])

    # Act & Assert: 验证是否干净利落地抛出包含具体指导信息的异常
    with pytest.raises(FileNotFoundError, match="pass --example"):
        resolve_source_root()
