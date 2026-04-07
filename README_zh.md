# ⚙️ vgtr-py

<p align="center">
  <strong>变几何桁架机器人（VGTR）编辑器与仿真器</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange" alt="Status: Active Development">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Code%20Style-Google-blueviolet" alt="Code Style: Google">
  <img src="https://img.shields.io/badge/Tests-Passing-success" alt="Tests">
</p>

---

`vgtr-py` 是一个面向 Python 的**变几何桁架机器人（Variable-Geometry Truss Robot, VGTR）**编辑器与预览仿真器。

它借鉴了 [PneuMesh](https://github.com/riceroll/pneumesh) 的简化动力学、轻量编辑交互和颜色组选控思路，将其迁移到更贴合 VGTR 的现代 Python 架构中，并重新定义了锚点、杆组与控制组语义。

## 🎯 项目目标

这个项目保留了原始 PneuMesh 中核心且高效的优势：

- 🏗️ **轻量级的结构编辑工作流**
- 🚀 **简化的弹簧网络计算动力学**
- 🎛️ **基于控制组与执行序列的脚本控制模型**
- 🔄 **基于标准 JSON 的工作区交换格式**

> [!WARNING]  
> **定位声明**：在 v1 阶段，本项目旨在提供一个快速迭代和验证思路的轻量级交互设计工具，**不打算**替代如 MuJoCo 或 Isaac Gym 等高保真机器人仿真引擎。

## 🏛️ 项目架构与状态

目前项目代码库核心组件已搭建完成：

- 🗄️ **Workspace (`workspace.py`, `schema.py`, `topology.py`)**：核心状态中心，负责拓扑结构管理与基于 Msgspec 的数据持久化。
- ⚙️ **Engine (`engine.py`, `kinematics.py`)**：数值仿真内核，基于欧拉积分驱动物理计算与运动学更新。
- 🎨 **Renderer (`rendering.py`)**：将复杂内部结构高保真地映射为三维可视化场景。
- 💻 **UI (`ui.py`, `commands.py`)**：基于 Viser 的前端交互层以及基于 Typer 的 CLI 工具集。
- 🕒 **History (`history.py`)**：基于快照 (Snapshot) 设计的完整撤销/重做操作栈机制。

*注：当前 `src/vgtr_py/` 下的所有源码均严格遵照已本地化的 **Google Python (中文)** 代码文档规范。*

## 🛠️ 技术栈

- 🧮 **`numpy`**: 核心矩阵与模拟数组运算
- 🌍 **`viser`**: 基于浏览器的响应式 3D 可视化与实时 GUI 绑定
- 📦 **`msgspec`**: 严格的数据 Schema 校验与高性能的 JSON I/O
- ⌨️ **`typer`**: 统一且友好的 CLI 命令行接口
- 🧪 **`pytest` & `ruff`**: 单元测试支持与严苛的代码质量检测

## 🗺️ 近距离里程碑 (路线图)

- [x] 与原始 PneuMesh 结构兼容的 JSON Schema 解析层
- [x] 基于 Python + NumPy 的核心力学内核逻辑
- [x] 基于 Viser 的三维交互渲染框架 (支持拖拽、多选、删除与连边)
- [x] 基于快照机制的全局撤销/重做 (Undo/Redo) 历史系统
- [x] 系统核心模块的规范化及单元测试用例全覆盖
- [ ] 完整支持基于时间轴序列的脚本执行系统 (Script Grid Editor)

## 🚀 快速开始

目前环境强制依赖 **Python 3.12+**。推荐使用虚拟环境或 `uv`。

```bash
# 1. 激活虚拟环境 (必要)
source .venv/bin/activate

# 2. 安装项目运行于测试依赖
pip install -e ".[test]"

# 3. 启动应用
vgtr-py serve
```

## 🧪 测试与质量

本项目接入并强制保持了完整的单元测试网。为避免与系统中的其他插件发生污染冲突（如较旧的 ROS 插件），推荐干净启动：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

---

如果你要继续参与开发并了解更深的技术细节与架构迁移决策，请参阅 [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)。
