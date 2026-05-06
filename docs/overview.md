# Project Overview

`vgtr-py` 是一个面向 VGTR（变几何桁架机器人）的轻量级编辑器、预览仿真器与 RL 实验平台。

## 设计理念

- **轻量**：纯 Python + NumPy，零深度学习框架依赖即可运行完整仿真
- **分层**：严格区分编辑态（Workspace）与运行态（Model/RuntimeState/RuntimeSession），建模层与求解层解耦
- **真实**：杆件类型化（被动/主动/弹性），引入行程/出力限制与堵转检测，拒绝"橡皮网"假物理
- **可扩展**：同一份模型可驱动单环境、批量环境，为 RL 训练提供干净接口

## 架构总览

```
                          +-----------+
                          | Workspace |  编辑态 / 导入导出 / 历史记录
                          +-----+-----+
                                | compile_workspace()
                          +-----v-----+
                          | VGTRModel |  只读静态模型（拓扑、物理参数、脚本）
                          +-----+-----+
                                |
                          make_state(num_envs=N)
                                |
                          +-----v------+
                          | RuntimeState|  env-major 动态状态 (N, ...)
                          +-----+------+
                                |
                         RuntimeSession
                      step / observe / info
                                |
              +-----------------+------------------+
              |                                    |
        +-----v------+                      +------v-------+
        | VGTRGymEnv |                      | VGTRVectorEnv|  Gymnasium 接口
        +------------+                      +--------------+

        投影层（可选，降维控制）：
        runtime.project.project_anchor_targets()
        目标锚点位置 → 逐主动杆目标值
```

## 核心模块

| 模块 | 职责 |
|------|------|
| `workspace` | 工作区容器：拓扑、物理参数、脚本、UI 状态 |
| `model` | 编译 workspace → 不可变 `VGTRModel`，含预计算的主动杆索引、行程/力限、控制组映射 |
| `runtime` | 统一运行时内核：`RuntimeState`、`RuntimeSession`、projection/observation/reward helpers |
| `adapters` | Gymnasium 适配层：`VGTRGymEnv` / `VGTRVectorEnv` |
| `kinematics` | 杆组派生几何：方向、中点、套筒姿态（position + quat） |
| `topology` | 图论操作：增删锚点/杆组、拖拽、固定、控制组分配 |
| `commands` | UI 命令层：将业务指令翻译为拓扑操作，自动接管 undo/redo |
| `history` | 快照栈：最多 100 条 undo/redo 记录 |
| `rendering` | Viser 3D 渲染：三段式杆组（sleeve + 左右活塞），单/批量场景 |
| `config` | 配置 dataclass：`SimulationConfig`, `RobotConfig` |
| `schema` | JSON 文件格式 (`sites + rod_groups`) 的解析与兼容 |

## 快速链接

- [架构设计](architecture.md) — 分层设计、数据流、控制架构
- [物理模型](physics.md) — 当前显式欧拉模型与 XPBD 演进方向
- [UI 交互](ui-interaction.rst) — 编辑器与仿真面板操作逻辑
- [开发路线](roadmap.md) — 已完成项、近期任务、阶段规划
- [API](api.rst) — 自动生成模块索引
