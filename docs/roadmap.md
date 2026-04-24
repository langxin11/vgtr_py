# 开发路线

## 项目定位

`vgtr-py` 面向 VGTR（变几何桁架机器人）的：

- 结构编辑器
- 交互预览器
- 轻量仿真器
- RL 实验平台

当前 VGTR-only，不再显式兼容 PneuMesh 旧格式。

## 已完成

- **术语对齐**：全面收敛至 `anchor / rod_group / control_group` 业务语义
- **工作区主格式**：`sites + rod_groups`（支持可选 `control_groups / script`）
- **加载策略收敛**：JSON 仅作为点位与连接输入，结构/显示参数统一由 `robot_config` 提供
- **命令层抽离**：`commands.py` 桥接 UI 与底层，自动接管 undo/redo 及快照管理
- **编辑能力**：支持锚点选择、延伸添加、杆组连接、删除、拖拽、固定、批量分配控制组
- **仿真内核分层**：`VGTRModel`（只读静态模型）+ `VGTRData`（动态状态）+ `Simulator`（无状态步进）三层拆分，支持无 UI 的纯脚本/批量调用
- **真实执行器模型**：杆件类型化（被动刚性 / 主动伸缩 / 柔性），引入行程限制 `rod_length_limits`、出力限制 `rod_force_limits`、堵转检测 `rod_stalled`
- **轻量仿真**：基于显式欧拉积分，支持变长度杆组约束、脚本驱动、重力、阻尼、地面摩擦
- **渲染系统**：基于 Viser 的三段式杆组渲染（sleeve + left + right），支持套筒中心对齐
- **RL 环境接口**：`VGTREnv` 提供 Gymnasium 风格 `reset/step` API，观测空间包含杆级力学统计量
- **可行性投影层**：`projection.py` + `kinematics.py` 提供节点目标到杆长目标的约束投影能力
- **CPU 批量 rollout 基线**：`BatchVGTRData` + `BatchSimulator` + `VectorVGTREnv` 支持同构多环境的 NumPy 向量化步进

## 近期任务

1. **Schema 校验强化**：为输入 JSON 提供更友好的错误提示（缺失字段、类型错误）
2. **UI 逻辑拆分**：进一步解耦 `ui.py` 中的事件处理逻辑
3. **扩展回归测试**：覆盖极端收缩参数、多控制组并发、拖拽历史补录等边缘场景
4. **RL 训练闭环验证**：基于 `VectorVGTREnv` 跑通首个 PPO/SAC 随机 rollout，验证观测空间与奖励设计
5. **批量仿真性能评估**：对比单环境循环、多进程 rollout 与 CPU batch 的吞吐差异
6. **交互式动作脚本序列编辑器**：在 UI 中提供通道-动作网格编辑器

## 后续方向（视需求启动）

- **GPU 后端迁移**：若训练吞吐成为瓶颈，将 `batch.py` 计算内核迁移到 JAX/PyTorch/Warp
- **XPBD 求解器**：若大刚度场景稳定性不足，将显式欧拉升级为 XPBD（详见 [物理模型](physics.md)）
- **大规模并行训练**：GPU 原生后端 + 数千并行环境的 RL 训练

## 验收标准

1. `vgtr-py serve` 可直接启动默认示例
2. 全量测试通过
3. 文档与代码实现保持同步
