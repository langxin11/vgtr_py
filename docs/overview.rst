Project Overview
================

``vgtr-py`` 是一个面向 Python 的变几何桁架机器人编辑器与预览仿真器。

Goals
-----

- 保留轻量级结构编辑工作流
- 保留显式积分与变长度杆组约束的轻量动力学
- 保留通道与动作驱动的脚本模型
- 保留基于 JSON 的工作区交换格式

Current Focus
-------------

当前阶段优先关注以下方向：

- 数据模型与工作区结构稳定化
- NumPy 求解核心演进
- 基于 Viser 的浏览器交互式预览
- 编辑、历史记录与命令层整理

Architecture
------------

当前代码组织围绕以下关系展开：

- ``Workspace`` 作为状态中心，负责编辑态、导入导出、历史记录。
- ``VGTRModel`` / ``VGTRData`` 提供只读静态模型与动态运行时分隔。
- ``Simulator`` (``sim``) 负责无状态求解步进。
- ``Engine`` (``engine``) 负责拓扑预计算与状态数组同步。
- ``UI`` 负责读取和编辑工作区，提供 Tab 化交互面板。
- ``Renderer`` 负责场景映射与显示。

Key Modules
-----------

- ``vgtr_py.workspace``: 工作区及状态容器
- ``vgtr_py.model``: 静态模型编译 (``VGTRModel``)
- ``vgtr_py.data``: 动态状态生成与重置 (``VGTRData``)
- ``vgtr_py.sim``: 仿真步进核心 (``Simulator``)
- ``vgtr_py.engine``: 预计算与状态数组同步
- ``vgtr_py.env``: Gymnasium 风格环境包装器
- ``vgtr_py.projection``: 可行性投影与约束求解
- ``vgtr_py.schema``: JSON schema 和兼容层
- ``vgtr_py.kinematics``: 杆组派生几何与姿态推导
- ``vgtr_py.rendering``: Viser 场景渲染
- ``vgtr_py.ui``: GUI 和交互事件绑定
- ``vgtr_py.commands``: 面向 UI 的命令层
- ``vgtr_py.gym_compat``: Gymnasium 接口兼容层
