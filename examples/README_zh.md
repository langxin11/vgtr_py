# Python 示例

这些示例演示了从轻量级无界面脚本到基于 Viser 的前端应用程序的运行时 API。
建议从无界面运行时和批量环境示例入手。在仓库根目录下执行命令。

## 无界面运行时

```bash
uv run python examples/sim_minimal.py
```

展示核心路径：

```text
Workspace JSON -> VGTRModel -> RuntimeSession.step_batch()
```

## 投影控制

```bash
uv run python examples/projection_minimal.py
```

将目标锚点位置投影到控制目标值，然后推进仿真。

## Gym 风格环境

```bash
uv run python examples/env_minimal.py
```

创建 `VGTRGymEnv`，调用 `reset()` 并执行若干步 `step()`。

## 批量环境

```bash
uv run python examples/vector_env_minimal.py
```

创建具有多个 CPU 向量化环境的 `VGTRVectorEnv`，并步进批量动作数组。

## 运行时 Viser 循环（单环境）

```bash
uv run python examples/runtime_viser_loop.py
uv run python examples/runtime_viser_loop.py --example configs/example.json --port 8081
```

单环境运行时/渲染路径：

```text
Workspace JSON -> RuntimeSession -> SceneRenderer/Viser
```

在浏览器中打开打印的 Viser URL。

## 批量运行时 Viser 循环

```bash
uv run python examples/runtime_viser_batch_loop.py
uv run python examples/runtime_viser_batch_loop.py --num-envs 9 --hide-others --selected-env 0
uv run python examples/runtime_viser_batch_loop.py --mode fixed --control-mode projection
```

在单个 Viser 场景中使用批处理网格句柄渲染多个 `RuntimeSession` 环境。
支持 `direct`、`control_group` 和 `projection` 控制模式，以及 `sine`/`fixed` 作动模式。默认 `GeoTrussRover.json` 使用 `direct` 模式逐主动杆驱动；需要联动控制组通道时使用 `control_group`。

## 手动控制 Viser 循环

```bash
uv run python examples/runtime_viser_loop_manual_control.py
```

单环境循环，支持三种作动模式：GUI 滑块（交互式）、正弦波（开环步态）和脚本回放（工作区脚本矩阵）。

## 完整 UI

```bash
uv run python examples/ui_minimal.py
```

在 `http://127.0.0.1:8080` 启动完整的 Viser 编辑器和仿真 UI。

在浏览器中打开后可使用完整的 Editor + Physics 标签页，包括批量模式控制
（Physics 标签页中的 Environment 文件夹）和单环境编辑工具。
