# VGTR Simulator API 与架构演进设计

本文档给出了 `vgtr-py` 从“编辑器驱动的轻量仿真器”演进到“可被程序化调用、可用于 RL/批量 rollout 的仿真内核”的完整方案。

目标不仅是吸收 MuJoCo 等引擎在软件接口分层上的优点（静态模型与动态状态分离、`compile/reset/step` 明确分层），还要在物理建模上向**真实机器人执行器**靠拢，并建立适合 RL 训练的**多层控制抽象**。

## 1. 总体路线与核心理念

采用 **“底层真实执行器 + 高层节点/任务空间控制”** 的双层方案。
- **软件架构**：将静态拓扑与动态状态完全解耦，使得单步求解路径不再依赖 UI，同一份模型可驱动单环境和批量并行环境。
- **物理建模**：底层保证物理真实性，抛弃纯粹的“橡皮网”弹簧，引入具备力学极限和堵转特性的真实执行器模型。
- **控制抽象**：高层负责降低控制维度与训练难度，通过可行性投影将任务空间意图转化为底层安全的驱动指令。

## 2. 软件架构：四层拆分目标

当前引擎的单步路径依赖 UI 状态（如 `ui.simulate`, `ui.editing`），导致难以被外部纯计算环境调用。建议将系统严格拆分为四层：

1. `Workspace`：负责编辑态、导入导出、UI 绑定、历史记录。
2. `VGTRModel`：只读静态模型（拓扑、质量、约束、物理参数预计算）。
3. `VGTRData`：单环境动态状态（位置、速度、控制量、接触缓存、每步的杆级统计量）。
4. `Simulator`：对 `VGTRModel + VGTRData` 执行无状态的求解步进。

数据流转关系：
```text
Workspace --compile--> VGTRModel
VGTRModel + seed -----> VGTRData
Simulator.step(model, data, action)
```

## 3. 物理内核改进：真实执行器模型

把当前引擎从统一的“弹簧杆”升级为**“杆件类型化执行器模型”**，区分：
- **被动刚性杆** (Passive Rigid)
- **主动伸缩杆** (Active Actuator)
- **柔性实验杆** (Soft/Elastic)

对于主动杆，在单步仿真中必须引入以下真实物理约束：
- **行程限制**：`L_min` / `L_max`
- **出力限制**：最大推拉力 `F_max`
- **自锁与堵转 (Stall)**：超载时停止运动，并在未激活时保持高刚度锁止状态。

**每步输出杆级统计量（写入 `VGTRData`）：**
- `current_length` (当前长度)
- `target_length` (目标长度)
- `axial_force` (轴向受力)
- `strain` (应变)
- `stalled_flag` (是否处于堵转状态)

*注：前期保留现有接触与积分框架，增量式在 `engine.py` 演进，不急于整体替换 XPBD 求解器。*

## 4. RL 与控制层改进：两级控制架构

不要直接让 RL 暴力控制所有杆长，改为**两级控制架构**，将“多杆强耦合问题”转化为“少量关键节点目标 + 约束投影问题”：

1. **高层动作 (RL / 规划层)**：输出关键节点的目标位置、期望姿态或机器人的宏观形态模态。
2. **中间层 (Projection)**：可行性投影（约束优化 / 阻尼最小二乘 / 逆运动学），将高层意图转化为符合物理极限的动作。
3. **底层执行 (Simulator)**：将投影后的结果转成主动杆目标长度，交由仿真器执行。

### 奖励与观测设计原则
把底层的力学统计量显式暴露给 RL 观测空间，并在 Reward 中加入惩罚，**绝不让智能体把结构当成“橡皮网”来学**：
- **Observation 加入**：杆长误差、杆轴向力、堵转标志、地面接触状态。
- **Reward 加入**：
  - 被动杆受力/形变惩罚（要求维持刚性）
  - 超行程惩罚
  - 堵转 (Stall) 惩罚
  - 结构过度内应力惩罚（降低自交和强行对抗）

## 5. 模块代码结构草图

### 5.1 VGTRModel
预计算的只读静态缓存：
```python
@dataclass(slots=True)
class VGTRModel:
    anchor_count: int
    rod_count: int
    
    # 执行器类型与极限
    rod_type: np.ndarray               # 0:被动, 1:主动, 2:柔性
    rod_rest_length: np.ndarray        # (R,)
    rod_length_limits: np.ndarray      # (R, 2) [L_min, L_max]
    rod_force_limits: np.ndarray       # (R, 2) [-F_pull, F_push]
    
    # 物理全局参数与拓扑 (略)
    ...
```

### 5.2 VGTRData
完全确定性的动态状态与力学统计快照：
```python
@dataclass(slots=True)
class VGTRData:
    qpos: np.ndarray                   # (N, 3)
    qvel: np.ndarray                   # (N, 3)
    
    # 杆级统计量 (暴露给 RL)
    rod_length: np.ndarray             # (R,)
    rod_axial_force: np.ndarray        # (R,)
    rod_stalled: np.ndarray            # (R,) bool
    
    ctrl: np.ndarray                   # (C,) 当前激活值
    ctrl_target: np.ndarray            # (C,)
    
    # 接触缓存与时间
    contact_cache: np.ndarray 
    time: float = 0.0
```

## 6. 面向外部的 API 风格

推荐保留两层 API 以适应不同场景。

### 6.1 低层仿真 API (MuJoCo 风格)
面向研究、性能控制、批量化：
```python
from vgtr_py.model import compile_workspace
from vgtr_py.data import make_data, reset_data
from vgtr_py.sim import Simulator

model = compile_workspace(workspace)
data = make_data(model)
sim = Simulator()

reset_data(model, data, seed=42)
sim.step(model, data, action) # action 更新 ctrl_target
```

### 6.2 高层环境 API (Gymnasium 风格)
面向 RL 框架对接：
```python
env = VGTREnv.from_workspace(workspace)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

## 7. UI / Viewer 的重新定位

Viewer 应只做展示与轻交互，**不再承载仿真规则**。右侧面板改为 Tab 化，与 `mjviser` 最佳实践对齐：

- **Tab 1: Editor (✏️)** - 拓扑编辑与搭建
- **Tab 2: Physics (⚙️)** - 物理参数、步进调试、场景控制
- **Tab 3: Actuation (🎮)** - 控制组、脚本测试
- *(后续扩展)* **Metrics / Goals** - 力学状态监控、任务目标设置

## 8. 开发落地阶段清单 (Phase 1 ~ 5)

**遵循原则：先把底层变“更硬、更真”，再把高层变“更低维、更聪明”。**

### Phase 1: 软硬解耦，建立纯净内核
- 解除 `engine.step()` 与 UI 的依赖。
- 引入 `VGTRModel`, `VGTRData`, `Simulator` 三个核心类。
- 实现 `compile_workspace()`, `make_data()`, `step()`。
- **里程碑**：能够在无 UI 的纯脚本环境下，加载已有模型并稳定运行仿真。

### Phase 2: 执行器物理特性升级
- 加入“杆件类型化”支持（区分被动、主动）。
- 在积分求解中加入物理极限：行程限制 (Limits)、力限 (Max Force)、自锁与堵转 (Stall)。
- 完善 `VGTRData` 中的统计量字段更新。
- **里程碑**：UI 中测试主动推拉，遇到障碍物或拉伸极限时表现出真实的堵转和反作用力，而非无限穿透。

### Phase 3: RL 观测与奖励接入
- 构建 `Gymnasium Env` 包装器。
- 将 Phase 2 中计算出的杆级统计量（轴力、误差、堵转）暴露到 `observation`。
- 实现针对结构内应力和违规操作的 `reward` 惩罚函数。
- **里程碑**：跑通第一个 PPO / SAC 随机 rollout，确保能获取到完整的力学观测。

### Phase 4: 高层节点空间控制与可行性投影
- 在 Env 层实现动作降维：外部只输出关键节点目标。
- 实现中间层的可行性投影（Projection）逻辑，将目标转换为底层的 `ctrl_target`。
- **里程碑**：使用较小规模的网络成功训练出一个基于节点目标的行走步态。

### Phase 5: 求解器性能与并行化进阶 (可选)
- 支持多实例并行 rollout 或 Batched 仿真。
- 评估现有的罚函数方法是否成为瓶颈。
- 若刚度问题严重，再考虑将底层积分器升级为 XPBD 或引入硬约束求解器。
- **里程碑**：支持超大规模并行训练，或实现极度刚性结构的稳定仿真。
