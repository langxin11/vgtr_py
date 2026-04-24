# 物理模型

本文档描述当前仿真引擎的物理模型，以及未来可能引入的 XPBD 求解器方案。

## 1. 当前模型：显式欧拉积分

位置在 `src/vgtr_py/sim.py` 和 `src/vgtr_py/batch.py`。

### 1.1 积分方案

显式欧拉（Semi-implicit Euler）：

```
v += (F / m) * h
v *= damping
x += v * h
```

- 时间步长 `h = 0.001`（默认）
- 阻尼系数 `damping_ratio = 0.95`（每步速度缩放）
- 速度限幅：> 10 m/s 时截断
- 固定锚点 `(anchor_fixed == True)` 不参与积分
- 地面穿透硬修正：`z < 0` 时强制 `z=0, vz=0`

### 1.2 杆组轴向弹簧力

每个杆组等价于一个线性弹簧，力沿杆方向施加于两端锚点：

```
F = k * (current_length - target_length)  # 胡克定律
F = clamp(F, F_min, F_max)                # 出力限幅
```

目标长度因杆件类型而异：

| 类型 | 目标长度 |
|------|----------|
| Active | `L_min + (L_max - L_min) * ctrl[group]`，ctrl ∈ [0,1] |
| Passive | `clamp(rest_length, L_min, L_max)` |
| Elastic | `clamp(rest_length, L_min, L_max)` |

`rod_target_override` 可逐杆覆盖目标长度（用于手动调试）。

### 1.3 控制平滑

控制值不直接跳变，而是按速率限制平滑逼近：

```
delta = clip(ctrl_target - ctrl, -rate, rate)
ctrl += delta
```

`contraction_percent_rate` 默认 1e-3，模拟真实执行器的响应延迟。

### 1.4 堵转检测

杆件在以下任一条件成立时标记为堵转（`rod_stalled = True`）：

- **力饱和**：`|raw_force - clipped_force| > 1e-9`（输出力被限幅截断）
- **行程超限**：`current_length < L_min - 1e-9` 或 `current_length > L_max + 1e-9`

堵转信息写入 `data.rod_stalled`，暴露给 RL 观测空间和奖励函数。

### 1.5 外力

**重力**：
```
F_z -= mass * gravity_factor
```
`gravity_factor` 默认 180，可通过 `gravity=False` 关闭。

**地面接触**（惩罚法，非约束求解）：
```
Fn = max(0, -ground_k * z - ground_d * vz)    # 弹簧-阻尼
F_z += Fn
```
`ground_k=100000`, `ground_d=500`。

**库仑摩擦**：
```
F_fric = -normalize(v_xy) * min(friction_factor * Fn, 100 * |v_xy|)
```
`friction_factor` 默认 0.5。

### 1.6 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k` | 20000 | 杆组刚度 |
| `h` | 0.001 | 时间步长 |
| `damping_ratio` | 0.95 | 每步速度衰减 |
| `gravity_factor` | 180 | 重力加速度 |
| `friction_factor` | 0.5 | 库仑摩擦系数 |
| `ground_k` | 100000 | 地面弹簧刚度 |
| `ground_d` | 500 | 地面阻尼 |
| `contraction_percent_rate` | 1e-3 | 控制速率限制 |

## 2. 局限性

显式欧拉方案在以下场景存在已知局限：

- **大刚度不稳定**：`k` 较大时，需极小 `h` 才能维持稳定
- **穿透**：地面接触使用惩罚法，高速碰撞下可能穿透
- **无硬约束**：杆长约束是软弹簧，非刚性约束
- **速率限制**：CPU NumPy，单环境步进 ~μs 级；批量环境下受 CPU 核心数约束

## 3. XPBD 提案（未来方向）

> **状态：未实现。** 当前代码使用显式欧拉。XPBD 是解决大刚度稳定性问题的候选方案，尚未进入开发排期。

### 3.1 动机

XPBD（eXtended Position-Based Dynamics）的优势：

- **无条件稳定**：即使在大时间步长下也能处理极高刚度（甚至完全刚性）的约束
- **柔顺度控制**：通过物理 Compliance 精确模拟从刚性连杆到弹性气动肌肉的各种特性
- **计算效率**：无需复杂质量矩阵求逆，适合实时交互

### 3.2 算法概要

将每个杆组建模为距离约束 `C(x_i, x_j) = |x_i - x_j| - L_t = 0`。

每步执行：

**预测**（仅外力）：
```
v += dt * w * f_ext
p = x
x += dt * v
```

**约束投影**（N 次迭代，通常 3-10 次）：
```
n = (x_i - x_j) / |x_i - x_j|
Δλ = (-C - α̃ λ) / (w_i + w_j + α̃)     # α̃ = α / dt²
x_i += w_i * Δλ * n
x_j -= w_j * Δλ * n
λ += Δλ
```

**速度更新与阻尼**：
```
v = (x - p) / dt
v *= (1 - damping)
```

### 3.3 实施要点

- **Jacobi 并行化**：避免 Gauss-Seidel 顺序依赖，独立计算所有约束修正量后按锚点平均应用
- **稀疏累加**：利用 `np.add.at` 将杆组修正量累加到锚点
- **驱动集成**：在每步迭代前，通过 `lerp(L_min, L_max, ctrl)` 动态计算 L_t
