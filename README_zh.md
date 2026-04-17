# ⚙️ vgtr-py

<p align="center">
  <strong>变几何桁架机器人 (VGTR) 交互式设计与预览仿真环境</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange" alt="Status: Active Development">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Architecture-Boundary%20Isolation-green" alt="Architecture: Boundary Isolation">
  <img src="https://img.shields.io/badge/Tests-Passing-success" alt="Tests">
</p>

---

`vgtr-py` 是一个面向 Python 的**变几何桁架机器人 (Variable-Geometry Truss Robot, VGTR)** 编辑器与预览仿真器。

它在 [PneuMesh](https://github.com/riceroll/pneumesh) 的原型基础上，针对 VGTR 语义进行了全新的架构迁移。通过**锚点 (Anchors)**、**杆组 (Rod Groups)** 与 **控制组 (Control Groups)** 描述机器人的结构，并提供了基于物理受力的实时行为预览。

## 🎯 核心特性

- 🏗️ **轻量级拓扑编辑**：支持锚点延伸、批量连接、固定/解固定等专为桁架结构优化的交互。
- 🚀 **力驱动的动力学仿真**：
    - 基于显式欧拉 (Explicit Euler) 积分的实时受力解算。
    - **地面模型**：实现基于穿透深度的弹簧-阻尼惩罚力反馈。
    - **库仑摩擦**：支持受正压力约束的平滑库仑摩擦模型，模拟真实的蹬地爬行。
- 🎨 **移动副机械渲染**：采用三段式杆组渲染（套筒 + 双活塞杆），直观呈现杆件伸缩过程中的机械滑动感。
- 🎛️ **实时参数化控制**：
    - **在线调优**：支持在仿真运行中实时调节刚度、阻尼、摩擦系数等物理参数。
    - **动态驱动**：自动根据控制组生成 UI 滑块，支持手动驱动与脚本控制。

## 🏛️ 架构设计：边界隔离策略

项目采用“对内图论，对外业务”的隔离策略以保证代码的长期可维护性：

- **对内**：底层算法 (`topology.py`) 使用 `vertex` 和 `edge` 概念，保持数学逻辑的简洁。
- **对外**：上层业务与 UI (`ui.py`) 严格使用 `anchor`、`rod_group` 与 `control_group` 语义。
- **翻译**：指令层 (`commands.py`) 负责术语映射，并接管基于快照的 **Undo/Redo** 逻辑。

## 🛠️ 技术栈

- **数值计算**: `numpy`
- **3D 交互/GUI**: `viser`
- **数据序列化**: `msgspec` (使用 `sites` + `rod_groups` 磁盘格式)
- **CLI 工具**: `typer`
- **质量保证**: `pytest` & `ruff`

## 🗺️ 路线图状态

- [x] 基于 `anchor/rod_group` 的业务语义重构
- [x] 实现地面惩罚力与库仑摩擦模型
- [x] 开发三段式活塞杆机械渲染系统
- [x] 建立基于快照的全局历史记录系统
- [x] 核心模块已有单元测试并当前全部通过
- [ ] 交互式动作脚本序列编辑器 (Action Sequence Editor)

## 🚀 快速开始

环境要求：**Python 3.12+**。推荐使用 `uv` 进行开发。

```bash
# 1. 检出并安装依赖
git clone https://github.com/langxin11/vgtr_py.git
cd vgtr_py
uv sync --dev

# 2. 启动交互式预览器
uv run vgtr-py serve
```

## 🧪 自动化测试

运行测试前推荐禁用环境插件干扰：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

---

更多架构细节请参阅 [实现计划 (implementation-plan.md)](docs/implementation-plan.md)。
