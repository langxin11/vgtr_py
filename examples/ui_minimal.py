"""最小化完整 UI 启动器。

演示流程：
    Workspace JSON -> VgtrUiApp -> Viser UI with editor and simulator controls

Usage:
    uv run python examples/ui_minimal.py
    open http://127.0.0.1:8080
"""

from __future__ import annotations

from pathlib import Path

from vgtr_py.app import build_app


def main() -> None:
    app = build_app(
        config_path=None,
        example_path=Path("configs/example.json"),
        host="127.0.0.1",
        port=8080,
    )
    app.start()
    print("UI is serving at http://127.0.0.1:8080")
    print("Press Ctrl+C to stop.")
    app.server.sleep_forever()


if __name__ == "__main__":
    main()
