from pathlib import Path

from typer.testing import CliRunner

import vgtr_py

runner = CliRunner()


class DummyServer:
    """
    辅助类：用于 Mock (伪造) 底层的 HTTP Server 实例，
    主要是为了验证 Server 是否按预期进入了长期阻塞的监听状态，而不去真的开端口。
    """

    def __init__(self) -> None:
        self.slept_forever = False

    def sleep_forever(self) -> None:
        self.slept_forever = True


class DummyViewerApp:
    """
    辅助类：用于 Mock (伪造) 真实的 ViewerApp 应用，
    阻断对实际 Viewer 服务器的启动调用，避免测试代码被真实的网络服务阻塞卡死。
    """

    def __init__(self) -> None:
        self.started = False
        self.server = DummyServer()

    def start(self) -> None:
        self.started = True


def test_serve_uses_resolved_paths(monkeypatch) -> None:
    """
    测试：CLI 的 `serve` 命令是否正确地将命令行参数（Host、Port）
    以及路径解析器返回的路径组装起来，并成功把它们喂给应用工厂模块（build_app）。
    """
    # Arrange: 准备期望的硬编码返回值和测试用的收集器 (captured)
    expected_config = Path("/tmp/config.json")
    expected_example = Path("/tmp/example.json")
    captured: dict[str, object] = {}
    viewer_app = DummyViewerApp()

    # 1. 劫持路径解析器 (resolve_runtime_paths)
    def fake_resolve_runtime_paths(
        *, config_path: Path | None, example_path: Path | None
    ) -> tuple[Path, Path]:
        captured["resolve_args"] = (config_path, example_path)
        return expected_config, expected_example

    # 2. 劫持应用装配工厂 (build_app)
    def fake_build_app(
        *, config_path: Path, example_path: Path, host: str, port: int
    ) -> DummyViewerApp:
        captured["build_args"] = {
            "config_path": config_path,
            "example_path": example_path,
            "host": host,
            "port": port,
        }
        return viewer_app

    monkeypatch.setattr(vgtr_py, "resolve_runtime_paths", fake_resolve_runtime_paths)
    monkeypatch.setattr(vgtr_py, "build_app", fake_build_app)

    # Act: 模拟在命令行输入 `vgtr_py serve --host 127.0.0.1 --port 9000`
    result = runner.invoke(vgtr_py.cli, ["serve", "--host", "127.0.0.1", "--port", "9000"])

    # Assert: 验证执行流
    assert result.exit_code == 0
    # 由于命令行没指定 --config/--example，所以传递给解析器的参数应该都是 None
    assert captured["resolve_args"] == (None, None)
    # 给 build_app 传入的参数应当是上面解析器返回的模拟路径和 CLI 的 host/port
    assert captured["build_args"] == {
        "config_path": expected_config,
        "example_path": expected_example,
        "host": "127.0.0.1",
        "port": 9000,
    }
    # 确保成功发出了起服务的命令
    assert viewer_app.started is True
    assert viewer_app.server.slept_forever is True


def test_plan_prints_implementation_plan_path() -> None:
    """
    测试：CLI 的 `plan` 子命令能够执行并在标准输出 (stdout) 中
    打印出实现计划 (Implementation Plan) 文档的相对路径位置。
    """
    # Act: 模拟执行 plan 命令
    result = runner.invoke(vgtr_py.cli, ["plan"])

    # Assert: 无报错退出，且能在输出文本的末尾找到对应文档结构名称
    assert result.exit_code == 0
    assert result.stdout.strip().endswith("docs/IMPLEMENTATION_PLAN.md")
