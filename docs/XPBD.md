# XPBD 物理解算器设计规范 (提案)

本文档描述了变几何桁架机器人（VGTR）从当前的显式动力学升级为 **XPBD (eXtended Position-Based Dynamics)** 算法的技术方案。

## 1. 为什么要引入 XPBD？

当前 `engine.py` 使用基于胡克定律的显式欧拉积分（Explicit Euler）。该方案在杆组刚度（$k$）较大时，为了维持稳定，必须使用极小的时间步长（$dt$），否则系统容易发散（Blowing up）。

**XPBD 的优势：**
- **无条件稳定**：即使在大时间步长下也能处理极高刚度（甚至完全刚性）的约束。
- **柔顺度控制**：通过引入物理柔顺度（Compliance），可精确模拟从刚性连杆到弹性气动肌肉的各种特性。
- **计算效率**：省去了计算复杂的质量矩阵求逆，适合实时交互。

## 2. 核心变量定义

在 `vgtr-py` 的数据模型中：

### 2.1 锚点 (Anchors)
每个锚点视为一个物理粒子：
- 位置: $\mathbf{x}$ (当前位置), $\mathbf{p}$ (预测位置)
- 速度: $\mathbf{v}$
- 质量倒数: $w = 1/m$（若 `anchor_fixed` 为 True，则 $w = 0$）

### 2.2 杆组约束 (Rod Group Constraints)
每个杆组对应一个距离约束：
- 目标长度: $L_t \in [L_{min}, L_{max}]$ (由 `control_group` 驱动)
- 物理刚度: $k$
- 柔顺度: $\alpha = 1/k$
- 拉格朗日乘子: $\lambda$ (每步迭代开始时初始化为 0)

## 3. 算法流程 (Integration Loop)

在 `step()` 函数中，每步循环执行以下步骤：

### 3.1 预测 (Prediction)
仅考虑外力（如重力 $\mathbf{g}$），更新预测位置。
$$ \mathbf{v} \leftarrow \mathbf{v} + \Delta t \cdot w \cdot \mathbf{f}_{ext} $$
$$ \mathbf{p} \leftarrow \mathbf{x} $$
$$ \mathbf{x} \leftarrow \mathbf{x} + \Delta t \cdot \mathbf{v} $$

### 3.2 约束投影 (Constraint Projection)
通过 $N$ 次迭代（通常 3-10 次）修正位置。对于每个杆组约束 $C(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\| - L_t = 0$：

1.  **计算梯度**：$\mathbf{n} = \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|}$
2.  **计算乘子增量**：引入时间步缩放的柔顺度 $\tilde{\alpha} = \frac{\alpha}{\Delta t^2}$
    $$ \Delta \lambda = \frac{-C(\mathbf{x}_i, \mathbf{x}_j) - \tilde{\alpha} \lambda}{w_i + w_j + \tilde{\alpha}} $$
3.  **更新位置**：
    $$ \mathbf{x}_i \leftarrow \mathbf{x}_i + w_i \Delta \lambda \mathbf{n} $$
    $$ \mathbf{x}_j \leftarrow \mathbf{x}_j - w_j \Delta \lambda \mathbf{n} $$
    $$ \lambda \leftarrow \lambda + \Delta \lambda $$

### 3.3 速度更新与阻尼 (Velocity Update)
$$ \mathbf{v} \leftarrow \frac{\mathbf{x} - \mathbf{p}}{\Delta t} $$
$$ \mathbf{v} \leftarrow \mathbf{v} \cdot (1.0 - \text{damping}) $$

## 4. NumPy 向量化实施建议

在 Python 环境下，为了保持性能，必须避免 Python 原生循环：

1.  **Jacobi 并行化**：约束投影默认是 Gauss-Seidel 风格（顺序更新），难以向量化。建议采用 Jacobi 风格：独立计算所有约束的修正量，然后对每个锚点的总修正量取平均应用。
2.  **稀疏矩阵掩码**：利用 NumPy 的 `np.add.at` 将杆组的修正量累加到锚点上。
3.  **驱动集成**：在每步迭代前，通过 `lerp(rod_min_length, rod_rest_length, control_group_value)` 动态计算 $L_t$。

## 5. 结论

通过引入 XPBD，`vgtr-py` 可以实现极其顺滑的交互体验。即使用户在界面上剧烈拖拽锚点，或设置极端的收缩参数，仿真系统也能保持物理上的稳健性，不会出现模型“爆炸”的现象。
