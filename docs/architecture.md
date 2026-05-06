# 仿真器架构设计

本文档描述 `vgtr-py` 从"编辑器驱动的轻量仿真器"演进到"可程序化调用、可用于 RL/批量 rollout 的仿真内核"的架构方案。

核心理念借鉴 MuJoCo 等引擎在软件接口分层上的优点（静态模型与动态状态分离、`compile/reset/step` 明确分层），同时在物理建模上向**真实机器人执行器**靠拢，并建立适合 RL 训练的**多层控制抽象**。

控制接口的底层真值是**主动杆执行器**，而不是控制组。控制组只用于 UI 联动、脚本复用、批量调参和兼容旧 JSON；运行时内核应当能直接表达“每根主动杆一个目标”的控制语义。

## 1. 四层分离

系统严格拆分为四层：

```
Workspace ──compile_workspace()──→ VGTRModel ──make_state()──→ RuntimeState
                                                                    │
                                                        RuntimeSession.step(action)
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
    config: SimulationConfig
    robot_config: RobotConfig

    # 锚点
    anchor_ids: list[str]
    anchor_count: int
    anchor_rest_pos: np.ndarray        # (N, 3)
    anchor_fixed: np.ndarray           # (N,) bool
    anchor_mass: np.ndarray            # (N,)
    anchor_radius: np.ndarray          # (N,)
    anchor_projection_target: np.ndarray  # (N,) bool
    projection_anchor_indices: np.ndarray  # (K,) int32

    # 杆组
    rod_group_ids: list[str]
    rod_count: int
    rod_anchors: np.ndarray            # (R, 2) int32
    rod_type: np.ndarray               # (R,) 0:被动 1:主动 2:弹性
    rod_enabled: np.ndarray            # (R,) bool
    rod_control_group: np.ndarray      # (R,) int32
    rod_rest_length: np.ndarray        # (R,)
    rod_length_limits: np.ndarray      # (R, 2) [L_min, L_max]
    rod_force_limits: np.ndarray       # (R, 2) [F_min, F_max]
    rod_group_mass: np.ndarray         # (R,)
    rod_radius: np.ndarray             # (R,)
    rod_sleeve_half: np.ndarray        # (R, 3)
    active_rod_indices: np.ndarray     # (A,) int32

    # 控制组：上层联动抽象，不是底层执行器接口
    control_group_ids: list[str]
    control_group_count: int
    control_group_enabled: np.ndarray  # (C,) bool
    control_group_default_target: np.ndarray  # (C,)
    control_group_colors: np.ndarray   # (C, 3) uint8

    # 脚本
    script: np.ndarray                 # (C, num_actions)
    num_actions: int
```

### 1.3 RuntimeState — 动态状态

统一的 env-major 运行时状态，单环境即 `num_envs=1`：

```python
@dataclass(slots=True)
class RuntimeState:
    qpos: np.ndarray            # (E, N, 3)
    qvel: np.ndarray            # (E, N, 3)
    forces: np.ndarray          # (E, N, 3)
    ctrl: np.ndarray            # (E, C)
    ctrl_target: np.ndarray     # (E, C)
    rod_ctrl: np.ndarray        # (E, A)
    rod_ctrl_target: np.ndarray # (E, A)
    rod_length: np.ndarray      # (E, R)
    rod_target_length: np.ndarray  # (E, R)
    rod_target_override: np.ndarray  # (E, R)
    rod_axial_force: np.ndarray     # (E, R)
    rod_strain: np.ndarray          # (E, R)
    rod_stalled: np.ndarray         # (E, R) bool
    contact_mask: np.ndarray        # (E, N) bool
    time: np.ndarray            # (E,)
    step_count: np.ndarray      # (E,) int64
    i_action: np.ndarray        # (E,) int64
    i_action_prev: np.ndarray   # (E,) int64
```

`make_state()` / `reset_state()` 创建并初始化状态（位于 `runtime/alloc.py`）。

### 1.4 RuntimeSession — 统一步进入口

`RuntimeSession` 是面向用户的公开门面，内部委托 `runtime/kernels.py` 中的纯函数执行物理计算。单环境和批量环境使用同一个类，通过 `num_envs` 区分。

```python
session = RuntimeSession.from_workspace(workspace, num_envs=4)
session.reset(seed=42)
session.step_batch(action)  # action shape: (num_envs, action_dim)
```

单步流程（`runtime/kernels.py`）：
1. 若传入 `action`，解析为逐主动杆目标 `state.rod_ctrl_target`（clamp [0,1]）
2. 若动作来自控制组，则先由控制组映射生成逐杆目标
3. `update_ctrl()` — 平滑逼近逐杆控制值（受 `contraction_percent_rate` 限制）
4. `compute_forces()` — 计算全场受力（杆力 + 重力 + 地面 + 摩擦）
5. `integrate()` — 显式欧拉积分更新 qpos/qvel
6. `update_time()` — 更新 `time` 和 `step_count`

`step_batch()` 采用完全向量化的 NumPy 操作，一次性处理 `(num_envs, ...)` 维度的所有环境。

## 2. 单环境 vs 批量路径

两套路径共享同一份 `VGTRModel` 和 `RuntimeSession`，仅在 `num_envs` 和 Gym 适配器上不同：

| 维度 | 单环境 | 批量 |
|------|--------|------|
| 运行时状态 | `RuntimeState` (E=1) | `RuntimeState` (E=N) |
| 形状 | `(1, N, 3)` | `(num_envs, N, 3)` |
| 会话 | `RuntimeSession` (num_envs=1) | `RuntimeSession` (num_envs=N) |
| Gym 包装 | `VGTRGymEnv` | `VGTRVectorEnv` |
| 动作输入 | `(action_dim,)` | `(num_envs, action_dim)` |
| 底层控制维度 | 主动杆数量 | 主动杆数量 |
| 后端 | NumPy | NumPy 向量化 |
| 用途 | 交互预览、调试 | RL 并行采样、性能实验 |

批量层属于 CPU NumPy 向量化基线。未来若训练吞吐成为瓶颈，可迁移到 JAX/PyTorch/Warp 等 GPU 友好后端。

## 3. 控制架构：逐杆执行器 + 上层抽象

底层执行器空间按主动杆定义：

```
rod_ctrl_target[active_rod] ∈ [0, 1]
0 → L_min
1 → L_max
```

这使最底层控制接口始终是“每根主动杆一个目标”。控制组、脚本、UI 滑块和投影控制都只是生成逐杆目标的上层路径。

### 3.1 逐杆直接模式

动作直接写入 `rod_ctrl_target`（值域 [0,1]），逐主动杆驱动：

```
action[active_rod] → rod_ctrl_target[active_rod]
rod_ctrl → smooth_interp → target_length = lerp(L_min, L_max, rod_ctrl[active_rod])
```

这是最底层、最明确的控制接口，适合 RL 训练、批量 rollout 和需要完全逐杆独立控制的脚本。

### 3.2 控制组模式

控制组是上层联动抽象。动作写入 `ctrl_target`，再展开成逐杆目标：

```
action[group] → ctrl_target[group]
rod_ctrl_target[rod] = ctrl_target[rod_control_group[rod]]
```

因此，有控制组时仍然可以逐杆控制；逐杆接口不应被控制组遮蔽。控制组只负责“多个杆共享同一个上层通道”的便利表达。

### 3.3 投影模式

动作 = 关键锚点目标位置。`project_anchor_targets()` 将目标位置转换为逐主动杆目标：

1. 将目标位置覆盖到 `projection_anchor_indices`
2. 对每个主动杆，计算锚点在新位置下的期望长度
3. 将期望长度归一化到 [0,1]（0=L_min, 1=L_max）
4. 直接输出逐杆 `rod_ctrl_target`，输出维度 = 主动杆数量

投影层不应按控制组取平均。控制组平均会丢失同组内不同杆的几何差异，并让“有控制组但仍想逐杆控制”的场景变得困难。若调用方需要控制组语义，应在更高层显式把逐杆目标聚合或绑定，而不是让投影内核隐式合并。

### 3.4 运行时模式

当前运行时代码采用显式控制模式：

- `direct`：action 维度 = 主动杆数量，直接写入 `rod_ctrl_target`。
- `control_group`：action 维度 = 控制组数量，写入 `ctrl_target` 后展开到 `rod_ctrl_target`。
- `projection`：action 维度 = projection target 锚点数量 × 3，投影后直接输出逐主动杆目标。
- `rod_target_override`：保留给手动调试，直接覆盖最终物理目标长度。

## 4. RL 接口设计

### 4.1 观测空间

显式暴露杆级力学统计量，**绝不让智能体把结构当成"橡皮网"来学**：

```
obs = [qpos, qvel, rod_length, rod_target_length,
       rod_axial_force, rod_strain, rod_stalled,
       rod_ctrl, rod_ctrl_target,
       ctrl, ctrl_target, contact_mask]
```

`rod_ctrl / rod_ctrl_target` 是执行器级观测；`ctrl / ctrl_target` 仅用于保留控制组层的可解释性和兼容性。

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

### 6.1 模块职责

| 模块 | 职责 |
|------|------|
| `topology.py` | 纯图论操作（vertex / edge） |
| `workspace.py` | 可变编辑态状态中心 |
| `model.py` | 只读编译模型（`compile_workspace()`） |
| `runtime/state.py` | 统一 env-major 运行时状态 |
| `runtime/kernels.py` | 纯函数物理计算（`step_state`, `compute_forces`, `integrate`） |
| `runtime/session.py` | 公开门面（`RuntimeSession`） |
| `runtime/alloc.py` | 状态分配与重置（`make_state`, `reset_state`） |
| `runtime/project.py` | 投影控制（`project_anchor_targets`，输出逐主动杆目标） |
| `runtime/observe.py` | 观测构建（`build_observation`） |
| `runtime/reward.py` | 奖励与信息（`compute_reward`, `build_info`） |
| `adapters/gym_env.py` | 单环境 Gym 适配器（`VGTRGymEnv`） |
| `adapters/vector_env.py` | 批量 Gym 适配器（`VGTRVectorEnv`） |

### 6.2 执行器类型化

| 类型 | 目标长度 | 用途 |
|------|----------|------|
| Active (1) | `lerp(L_min, L_max, rod_ctrl)` | 可控伸缩 |
| Passive (0) | `clamp(rest, L_min, L_max)` | 刚性连杆 |
| Elastic (2) | `clamp(rest, L_min, L_max)`（当前与 Passive 相同） | 柔性结构（预留） |

### 6.3 物理建模真实性

主动杆不是无限受力的"橡皮筋"——引入：
- 行程限制 `[L_min, L_max]`
- 出力限制 `[F_min, F_max]`
- 堵转检测：力截断 或 行程超限 → `rod_stalled=True`
