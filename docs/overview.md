# Project Overview

`vgtr-py` 是一个面向 VGTR（变几何桁架机器人）的轻量级编辑器、预览仿真器与 RL 实验平台。

## 设计理念

- **轻量**：纯 Python + NumPy，零深度学习框架依赖即可运行完整仿真
- **分层**：严格区分编辑态（Workspace）与运行态（Model/Data/Simulator），建模层与求解层解耦
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
              +-----------------+------------------+
              |                                    |
         make_data()                       make_batch_data(N)
              |                                    |
        +-----v-----+                      +-------v--------+
        |  VGTRData |                      | BatchVGTRData  |  (num_envs, ...)
        +-----+-----+                      +-------+--------+
              |                                    |
       Simulator.step()                   BatchSimulator.step()
         (sim.py)                            (batch.py)
              |                                    |
        +-----v-----+                      +-------v--------+
        |   VGTREnv |                      | VectorVGTREnv  |  Gymnasium 接口
        +-----------+                      +----------------+

        投影层（可选，降维控制）：
        project_anchor_targets() / _project_anchor_targets_batch()
        目标锚点位置 → 控制组目标值
```

## 核心模块

| 模块 | 职责 |
|------|------|
| `workspace` | 工作区容器：拓扑、物理参数、脚本、UI 状态 |
| `model` | 编译 workspace → 不可变 `VGTRModel`，含预计算的杆组索引、行程/力限、控制组映射 |
| `data` | 单环境运行时状态 `VGTRData`：qpos, qvel, forces, ctrl, rod 统计量 |
| `sim` | 无状态仿真步进 `Simulator.step(model, data, action)` |
| `batch` | 批量运行时 `BatchVGTRData` + `BatchSimulator`，CPU NumPy 向量化 |
| `env` | 单环境 Gymnasium 包装 `VGTREnv` |
| `vector_env` | 批量 Gymnasium 包装 `VectorVGTREnv`，支持 `(N, action_dim)` 动作输入 |
| `projection` | 锚点目标位置 → 控制组目标值的可行性投影 |
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
