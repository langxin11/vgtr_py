# 仿真器架构设计

本文档描述 `vgtr-py` 从"编辑器驱动的轻量仿真器"演进到"可程序化调用、可用于 RL/批量 rollout 的仿真内核"的架构方案。

核心理念借鉴 MuJoCo 等引擎在软件接口分层上的优点（静态模型与动态状态分离、`compile/reset/step` 明确分层），同时在物理建模上向**真实机器人执行器**靠拢，并建立适合 RL 训练的**多层控制抽象**。

## 1. 四层分离

系统严格拆分为四层：

```
Workspace ──compile──→ VGTRModel ──make_data──→ VGTRData
                                                    │
                                          Simulator.step(model, data, action)
```

### 1.1 Workspace — 编辑态

- 可变状态中心：拓扑、物理参数、脚本、UI 状态
- 负责导入导出（JSON `sites + rod_groups` 格式）、快照、历史记录
- 仅用于编辑/预览场景，**不参与纯计算/RL 调用路径**

### 1.2 VGTRModel — 只读静态模型

`compile_workspace()` 将 Workspace 编译为不可变模型：

```python
@dataclass(slots=True)
class VGTRModel:
    anchor_count: int
    rod_count: int
    control_group_count: int

    # 拓扑
    anchor_rest_pos: np.ndarray        # (N, 3)
    anchor_fixed: np.ndarray           # (N,) bool
    anchor_mass: np.ndarray            # (N,)
    rod_anchors: np.ndarray            # (R, 2) int32

    # 执行器类型与极限
    rod_type: np.ndarray               # (R,) 0:被动 1:主动 2:弹性
    rod_rest_length: np.ndarray        # (R,)
    rod_length_limits: np.ndarray      # (R, 2) [L_min, L_max]
    rod_force_limits: np.ndarray       # (R, 2) [F_min, F_max]
    rod_control_group: np.ndarray      # (R,) int32

    # 控制
    control_group_default_target: np.ndarray  # (C,)
    script: np.ndarray                 # (C, num_actions)

    # 投影
    projection_anchor_indices: np.ndarray

    # 全局参数
    config: SimulationConfig
```

### 1.3 VGTRData / BatchVGTRData — 动态状态

单环境 (`VGTRData`) 与批量 (`BatchVGTRData`) 均保存完全确定性的运行时状态：

```python
# 单环境 — 全部为 (N,3) 或 (R,) 形状
qpos, qvel, forces
ctrl, ctrl_target
rod_length, rod_target_length, rod_axial_force, rod_strain, rod_stalled
contact_mask
time, step_count

# 批量 — 第一维为环境索引
qpos:   (num_envs, N, 3)
qvel:   (num_envs, N, 3)
ctrl:   (num_envs, C)
...
```

`reset_data()` / `reset_batch_data()` 将状态重置为静止初始态。

### 1.4 Simulator / BatchSimulator — 无状态步进

两个步进器均为纯函数：`step(model, data, action=None)`，无内部状态，不依赖 UI。

单步流程：
1. 若传入 `action`，写入 `data.ctrl_target`（clamp [0,1]）
2. 平滑逼近控制值（受 `contraction_percent_rate` 限制）
3. 计算全场受力（杆力 + 重力 + 地面 + 摩擦）
4. 显式欧拉积分更新 qpos/qvel
5. 更新 `time` 和 `step_count`

`BatchSimulator.step()` 采用完全向量化的 NumPy 操作，一次性处理 `(num_envs, ...)` 维度的所有环境。

## 2. 单环境 vs 批量路径

两套路径共享同一份 `VGTRModel`，仅在运行时数据形状和步进器上不同：

| 维度 | 单环境 | 批量 |
|------|--------|------|
| 运行时数据 | `VGTRData` | `BatchVGTRData` |
| 形状 | `(N, 3)` | `(num_envs, N, 3)` |
| 步进器 | `Simulator` | `BatchSimulator` |
| Gym 包装 | `VGTREnv` | `VectorVGTREnv` |
| 动作输入 | `(action_dim,)` | `(num_envs, action_dim)` 或广播 |
| 后端 | NumPy | NumPy 向量化 |
| 用途 | 交互预览、调试 | RL 并行采样、性能实验 |

批量层属于 CPU NumPy 向量化基线。未来若训练吞吐成为瓶颈，可迁移到 JAX/PyTorch/Warp 等 GPU 友好后端。

## 3. 控制架构：两级抽象

不要直接让 RL 暴力控制所有杆长。采用**两级控制**：

```
RL 策略 → 关键节点目标位置 → Projection → ctrl_target → Simulator
  (低维)      (任务空间)     (可行性投影)   (执行器空间)
```

### 3.1 直接模式

动作直接写入 `ctrl_target`（值域 [0,1]），按控制组驱动主动杆：

```
action[group] → ctrl_target[group]
ctrl → smooth_interp → target_length = lerp(L_min, L_max, ctrl[group])
```

### 3.2 投影模式

动作 = 关键锚点目标位置。Projection 层将目标位置转换为控制组值：

1. 将目标位置覆盖到 `projection_anchor_indices`
2. 对每个主动杆，计算锚点在新位置下的期望长度
3. 将期望长度归一化到 [0,1]（0=L_min, 1=L_max）
4. 按控制组取平均 → `ctrl_target`

这使外部策略只需输出少量关键节点的 3D 位置，而非直接操作所有控制组。

## 4. RL 接口设计

### 4.1 观测空间

显式暴露杆级力学统计量，**绝不让智能体把结构当成"橡皮网"来学**：

```
obs = [qpos, qvel, rod_length, rod_target_length,
       rod_axial_force, rod_strain, rod_stalled,
       ctrl, ctrl_target, contact_mask]
```

### 4.2 奖励函数

```
reward = -(tracking_error
           + 0.1 * stalled_count      # 堵转惩罚
           + 0.01 * passive_force      # 被动杆受力惩罚
           + |strain|)                 # 结构内应力惩罚
```

设计原则：
- 被动杆受力/形变惩罚 → 要求维持刚性
- 堵转惩罚 → 防止超行程/超力操作
- 内应力惩罚 → 降低自交和强行对抗

## 5. 渲染架构

基于 Viser 的三段式杆组渲染：

```
sleeve (刚性套筒) + left_piston + right_piston (伸缩活塞)
```

- 单环境：`SceneRenderer.render()` → `/world/...`
- 批量预览：`SceneRenderer.render_batch()` → `/world/batch/...` batched mesh handles
- 支持 `selected_env` 与 `show_only_selected` 控制批量视图

## 6. 关键设计决策

### 6.1 术语边界隔离

| 层次 | 术语 | 用途 |
|------|------|------|
| 图论层 (`topology.py`) | vertex / edge | 纯数学操作 |
| 业务层 (UI / config) | anchor / rod_group / control_group | 用户可见 |
| 桥接层 (`commands.py`) | 翻译 | UI 指令 → 图论操作 |

### 6.2 执行器类型化

| 类型 | 目标长度 | use case |
|------|----------|----------|
| Active (1) | `lerp(L_min, L_max, ctrl)` | 可控伸缩 |
| Passive (0) | `clamp(rest, L_min, L_max)` | 刚性连杆 |
| Elastic (2) | `clamp(rest, L_min, L_max)` | 柔性结构 |

### 6.3 物理建模真实性

主动杆不是无限受力的"橡皮筋"——引入：
- 行程限制 `[L_min, L_max]`
- 出力限制 `[F_min, F_max]`
- 堵转检测：力截断 或 行程超限 → `rod_stalled=True`
