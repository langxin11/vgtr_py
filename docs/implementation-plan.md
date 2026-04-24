# vgtr-py 实现计划（精简版）

本文档记录了 `vgtr-py` 的实现约定、架构策略与下一步工作项。

## 1. 项目定位

`vgtr-py` 是一个面向 VGTR（变几何桁架机器人）的：

- 结构编辑器
- 交互预览器
- 轻量仿真器

当前路线为 VGTR-only，不再显式兼容 PneuMesh 旧格式。

## 2. 核心架构策略：边界隔离 (Boundary Isolation)

为了兼顾底层算法的通用性与上级业务的专一性，项目采用以下术语隔离策略：

-   **对内（图论语义）**：底层拓扑逻辑 (`topology.py`) 使用 `vertex` (顶点) 和 `edge` (边) 作为纯粹的数学概念。
-   **对外（业务语义）**：上层 UI (`ui.py`) 和配置文件严格使用 `anchor` (锚点)、`rod_group` (杆组) 和 `control_group` (控制组)。
-   **桥接（指令层）**：`commands.py` 负责将 UI 的业务指令（如 `add_joint_from_anchor`）翻译为底层的图论操作。

## 3. 当前已完成

- **术语对齐**：全面收敛至 `anchor / rod_group / control_group` 业务语义。
- **工作区主格式**：`sites + rod_groups`（支持可选 `control_groups/script`）。
- **加载策略收敛**：JSON 仅作为点位与连接输入，结构/显示参数统一由 `robot_config` 提供。
- **命令层抽离**：`commands.py` 完美桥接 UI 与底层，自动接管 Undo/Redo 及快照管理。
- **编辑能力**：支持锚点选择、延伸添加、杆组连接、删除、拖拽、固定、批量分配控制组。
- **仿真内核分层**：`VGTRModel`（只读静态模型）+ `VGTRData`（动态状态）+ `Simulator`（无状态步进）三层拆分，支持无 UI 的纯脚本/批量调用。
- **真实执行器模型**：杆件类型化（被动刚性 / 主动伸缩 / 柔性），引入行程限制 `rod_length_limits`、出力限制 `rod_force_limits`、堵转检测 `rod_stalled`，主动杆不再是无限受力的“橡皮筋”。
- 预览仿真：基于显式欧拉积分（Semi-implicit Euler）的轻量仿真，支持变长度杆组约束、脚本驱动、重力、阻尼、地面摩擦。
- **渲染系统**：基于 Viser 的三段式杆组渲染（sleeve + left + right），支持套筒中心对齐。
- **RL 环境接口**：`VGTREnv` 提供 Gymnasium 风格 `reset/step` API，观测空间包含杆级力学统计量（轴力、应变、堵转标志）。
- **可行性投影层**：`projection.py` + `kinematics.py` 提供节点目标到杆长目标的约束投影能力，支撑高层低维控制。

## 4. 运行时数据约定

### 4.1 拓扑层 (TopologyState)

- `anchor_ids` / `anchor_pos` / `anchor_fixed`
- `rod_group_ids` / `rod_anchors`
- `rod_rest_length` / `rod_min_length` / `rod_control_group`

### 4.2 控制层 (ScriptState)

- `control_group_ids` / `control_group_colors`
- `control_group_enabled`
- `script` (shape: `num_control_groups x num_actions`)

### 4.3 UI 层 (UiState)

- `anchor_status` / `rod_group_status`
- `editing / simulate / record`

## 5. 配置参数约定

当前仅保留并使用以下参数：

- `k`, `h`, `dampingRatio`
- `contractionInterval`, `contractionLevels`, `contractionPercentRate`
- `gravityFactor`, `gravity`, `frictionFactor`
- `defaultMinLength`, `numStepsActionMultiplier`

## 6. 近期任务 (Upcoming Tasks)

1. **Schema 校验强化**：为输入 JSON 提供更友好的错误提示（缺失字段、类型错误）。
2. **UI 逻辑拆分**：进一步解耦 `ui.py` 中的事件处理逻辑，提取到独立的组件类中。
3. **扩展回归测试**：覆盖极端收缩参数、多控制组并发、拖拽历史补录等边缘场景。
4. **RL 训练闭环验证**：基于 `VGTREnv` 跑通首个 PPO/SAC 随机 rollout，验证观测空间与奖励设计。
5. **交互式动作脚本序列编辑器**：在 UI 中提供通道-动作网格编辑器，替代纯数值滑块方式。

## 7. 验收标准

1. `vgtr-py serve` 可直接启动默认示例。
2. 全量测试通过。
3. 文档与代码实现保持 100% 同步。
