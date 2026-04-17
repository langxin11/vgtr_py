UI 交互操作逻辑
================

本文档描述 ``vgtr_py.ui.VgtrUiApp`` 当前实现中的交互语义。内容以代码行为为准，便于使用者和开发者统一认知。

总体结构
--------

UI 由三个主要面板构成：

- ``Workspace``: 文件加载/保存、运行与全局开关。
- ``Editing Tools``: 仅在编辑模式可见，包含选择与拓扑操作。
- ``Playback``: 手动步进与运行时重置。

关键状态字段位于 ``workspace.ui``：

- ``simulate``: 是否自动推进仿真。
- ``editing``: 是否处于编辑模式。
- ``moving_anchor`` / ``moving_body``: 拖拽时是否允许在编辑态推进仿真。
- ``show_control_group``: 是否显示控制组信息。


Workspace 面板行为
------------------

Load / Save
^^^^^^^^^^^

- ``Load Workspace`` 使用 ``Config Path`` 与 ``Example Path`` 重新构造 ``Workspace``。
- 加载成功后会清空 Undo/Redo 历史并同步 GUI 状态。
- ``Save Workspace`` 将当前工作区导出到 ``Save Path`` 指定位置。

Undo / Redo
^^^^^^^^^^^

- ``Undo`` 与 ``Redo`` 通过 ``WorkspaceHistory`` 回放快照。
- 成功撤销/重做后会：

  - 清除拖拽中的临时快照状态；
  - 同步 GUI 勾选值；
  - 刷新渲染和状态面板。

运行控制
^^^^^^^^

- ``Simulate`` 控制后台循环是否自动步进。
- ``Editing`` 切换编辑态，并在进入编辑时恢复初始姿态。
- ``Simulation Steps / Second`` 控制每秒累计步数（非渲染帧率）。
- ``Render Hz`` 控制后台循环睡眠周期与渲染刷新频率。


Editing Tools 面板行为
----------------------

该面板仅在 ``editing=True`` 时可见。关闭编辑态时会自动清理移动态与拖拽状态。

Move Anchor
^^^^^^^^^^^

- ``Move Anchor`` 开启后允许通过变换控制器拖拽锚点。
- 若当前不在编辑态，勾选 ``Move Anchor`` 会自动切入编辑态。

Selection 分组
^^^^^^^^^^^^^^

- ``Clear Selection``: 清空锚点与杆组的选择状态。
- ``Fix Selected``: 将选中锚点设为固定。
- ``Unfix Selected``: 取消选中锚点固定状态。

Topology 分组
^^^^^^^^^^^^^

- ``Add Anchor``:

  - 恰好选择 1 个锚点时，从该锚点派生新增锚点，并自动连一条新杆组。
  - 选择 0 个或多个（≥2）锚点时，从选择中心（或原点）新增锚点，不自动连接。

- ``Add Rod Group``:

  - 连接当前选中锚点集合；
  - 少于 2 个选中锚点时会提示无法连接。

- ``Remove Selected Rod Groups``: 删除选中杆组。
- ``Remove Selected Anchors``: 删除选中锚点（并处理关联拓扑）。
- ``Center Model``: 对当前模型执行居中命令。

Click Mode
^^^^^^^^^^

``Click Mode`` 影响点击行为，支持三种模式：

- ``replace``: 单击后替换选择集（杆组点击同样表现为替换）。
- ``toggle``: 单击切换当前元素的选中状态，支持多选。
- ``add-child``: 仅对锚点点击生效，点击锚点后直接从该锚点新增子锚点；杆组点击仍走杆组选择逻辑。


场景点击与拖拽语义
------------------

锚点点击
^^^^^^^^

- 锚点点击回调会读取 ``Click Mode``：

  - ``add-child``: 立即执行新增锚点命令；
  - 其他模式：执行锚点选择（replace/toggle）。

- 每次点击后，文件状态栏会显示当前选中锚点索引。

杆组点击
^^^^^^^^

- 杆组点击统一调用按模式选择：

  - ``toggle``: 切换杆组选中状态；
  - 其他模式：替换式选择。

- 点击后文件状态栏显示当前选中杆组索引。

拖拽编辑与历史记录
^^^^^^^^^^^^^^^^^^

- 拖拽开始时缓存 ``workspace.snapshot()``。
- 拖拽过程中持续更新选中锚点位置。
- 拖拽结束时比较起始快照与当前状态：

  - 仅在确实发生变化时写入历史栈；
  - 无变化则不产生历史记录。


Playback 与后台循环
-------------------

Playback 面板
^^^^^^^^^^^^^

- ``Step Once``: 按 ``Manual Step Count`` 手动推进 ``n`` 步。
- ``Reset Runtime``: 重置运行时状态（如步数计数等）。

后台循环触发条件
^^^^^^^^^^^^^^^^

后台线程持续运行，但是否步进由 ``_should_step_simulation`` 决定：

- ``simulate=False``: 不步进。
- ``simulate=True`` 且 ``editing=False``: 持续自动步进。
- ``simulate=True`` 且 ``editing=True``: 仅在拖拽中且允许移动（``moving_anchor`` 或 ``moving_body``）时步进。

该设计使编辑态默认更稳态，仅在显式拖拽交互时联动仿真。


状态面板说明
------------

状态面板实时展示：

- Anchor / Rod Group / Control Group 数量；
- 总步数、仿真步频和渲染频率；
- 当前 ``Click Mode``；
- 选中锚点与杆组的数量及索引明细。

