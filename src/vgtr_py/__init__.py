"""包初始化与Typer CLI。

此模块定义了项目的入口点组或命令行接口挂载点，便于通过命令行驱动 PneuMesh 系统。
"""

import logging
from pathlib import Path

import typer

from .app import build_app, resolve_runtime_paths

cli = typer.Typer(no_args_is_help=True)
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    exists=True,
    dir_okay=False,
    help="Simulation config JSON path. If omitted, auto-discover from source root.",
)
EXAMPLE_OPTION = typer.Option(
    None,
    "--example",
    exists=True,
    dir_okay=False,
    help="Example workspace JSON path. If omitted, auto-discover from source root.",
)


@cli.command()
def plan() -> None:
    """输出实现计划文档路径。"""
    project_root = Path(__file__).resolve().parents[2]
    typer.echo(project_root / "docs" / "implementation-plan.md")


@cli.command()
def serve(
    config: Path | None = CONFIG_OPTION,
    example: Path | None = EXAMPLE_OPTION,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """启动基于 Viser 的 VGTR 交互式查看器。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    resolved_config, resolved_example = resolve_runtime_paths(
        config_path=config, example_path=example
    )
    viewer_app = build_app(
        config_path=resolved_config,
        example_path=resolved_example,
        host=host,
        port=port,
    )
    viewer_app.start()
    typer.echo(f"Serving VGTR at http://{host}:{port}")
    typer.echo(f"Config: {resolved_config if resolved_config is not None else '<python-default>'}")
    typer.echo(f"Example: {resolved_example}")
    viewer_app.server.sleep_forever()


def main() -> None:
    cli()
