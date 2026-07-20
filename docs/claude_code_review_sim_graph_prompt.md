# 实施提示词：规则仿真 / 方案评审 / 数据导入 三界面可视化调整

> 你（Claude Code）是一名资深前端 / 数据可视化工程师。请基于以下规格修改 LLM4DRD 排产系统前端代码，完成【规则仿真】【方案评审】【数据导入】三个界面的指定调整。行号可能随修改偏移，请以**函数名**为准用 grep 定位。完成后必须本地起服务实测验证，不得只改代码不验证。

> **⚠️ 硬性约束：实现过程中禁止使用 `superpowers` skill（及任何以 superpowers 为名的命令 / 子能力）。本次所有改动仅使用原生代码编辑能力完成，不得触发、调用或依赖 superpowers。**

---

## 0. 项目与技术上下文

- 项目根目录：`/Users/zhouwentao/Desktop/llm4drd`
- 前端主逻辑：`frontend/app_v2.js`
- 前端样式：`frontend/app_v2.css`
- 设计 token：`frontend/design-system.css`（浅色主题）
- 本地验证：项目根目录执行 `python run_server.py`，浏览器打开 `index_v2.html`，进入对应界面。

### 设计 token（必须复用，保持一致性）
- 主色 `--primary: #2f6feb`、强调色 `--accent: #c2620a`、成功/已完成 `--success: #12a150`、信息/进行中 `--info: #0e8e7f`
- 圆角 `--radius-md`、阴影 `--shadow-sm`、弱化文字 `--text-soft`、容器留白 `--inset`
- 选中态 `.is-selected`、最佳值 `.is-best` 已存在，直接复用，不要新建同名类。
- 方案配色板 `SCHEME_COLOR_TOKENS` / `schemeColorToken(idx)` 已存在，沿用。

### 命名规范（沿用既有约定，不要破坏）
- 候选方案显示名一律用 `方案一 / 方案二 / 方案三…`，**禁止**出现 `S-` + 乱码（由 `getReviewCandidates()` 约 1068 控制，本次不改命名逻辑，但若改动涉及显示处请保持此规范）。
- `id` / `solutionId` 保留原始 `S-xxx`，勾选、导出、AI 评审映射不变。
- **订单一律按「订单号」展示**（即 `order_id` / 订单编号），**不得**以 `id.name`、`order_id · order_name` 等 "id+name" 拼接形式展示。

### 关键函数 / 状态（grep 定位）
- 订单标签函数：默认 `orderComboboxLabel`（约 455，`return [order.order_id, order.order_name].filter(Boolean).join(" · ")`——本次改为只返回订单号）；`renderOrderCombobox()`（约 463）、`mountOrderComboboxes()`（约 480）、`ReviewRuntime.createOrderComboboxController`（控制单选/搜索行为，需扩展为多选）。
- 规则仿真：`renderWorkflowStep3()`（约 2744），甘特在约 2779 调 `renderTimeline(app.simResult.gantt, {title, canvasId:"gantt-sim", solutionIds})`；当前**无订单筛选框**。
- 甘特统一渲染：`renderTimeline()`（约 1988，vis-timeline 封装）；甘特左侧机器名标签样式在 `frontend/app_v2.css` 的 `.gantt-canvas .vis-labelset .vis-label .vis-inner`（约 2183）、`.vis-nested-group`（约 2497）。
- 方案评审总渲染：`renderReviewLibraryTab()`（约 3255）
- 方案对比表：`renderReviewCandidateComparison()`（约 2975）——小字在约 3001
- 机器分类利用率：`renderReviewTypeUtilization()`（约 3029）
- 甘特联动：`renderReviewGantt()`（约 3235）、`reviewGanttControlsHtml()`（约 3182，当前只渲染订单筛选）、`buildReviewGanttData()`（约 3093，当前按机器→方案 nestedGroups 做多方案对比）
- 候选数据：`getReviewCandidates()`（1068）、`getSelectedReviewCandidates()`（1117）；排程明细在 `app.reviewRead.schemes[solutionId]`
- 操作列交互：`focus-candidate`（查看详情，约 3005）、`send-candidate-to-ai`、`export-selected-solution`
- 图谱：`layoutGraph()`（约 2576，**当前按节点类型分泳道、加家族带 = 层次感来源**）、`GRAPH_LANE_CONFIG` / `GRAPH_LANE_ORDER`、`renderInteractiveGraph()`（约 4057，节点详情与关系解释在此）、`renderGraphSection()`（约 2718）

---

## Scope 1：【规则仿真】界面

### 1.1 订单筛选：按订单号展示 + 多选 + 全选（Excel 风格筛选框）
- **订单显示改为订单号**：定位订单标签函数（约 455，默认 `return [order.order_id, order.order_name].join(" · ")`），**改为只返回订单号**（`order_id` / 订单编号），去掉 `· order_name` 拼接；`__all__` / 「全部订单」特殊项保留。该改动对所有用到 `renderOrderCombobox` 的甘特（规则仿真 + 方案评审）一并生效。
- **升级为 Excel 风格多选筛选框**：当前 `renderOrderCombobox`（约 463）是「搜索输入框 + 单选按钮列表」。改为类似 Excel 筛选下拉：
  - 下拉内**每个选项前带 checkbox**，支持**多选**；列表**顶部增加「全选」checkbox**（勾选即选中该方案/结果下的全部订单）。
  - 选中态以勾选集合（数组）维护，而非单个 `current`；相应改造 `mountOrderComboboxes`（约 480）与 `ReviewRuntime.createOrderComboboxController` 的 `select` 回调，使其接收订单数组。
  - 保留模糊搜索（输入框按订单号过滤）。
- **在规则仿真甘特上方新增该订单多选筛选框**（当前缺失）：筛选 `app.simResult.gantt`，选中 0 个 = 全部订单；选中 1+ 个 = 仅显示这些订单关联的工序。交互、视觉与方案评审甘特订单框一致。
- 验收：规则仿真运行后，甘特上方出现订单多选筛选框；订单以**订单号**呈现（无 id.name）；可勾选多个订单、可点「全选」；勾选后甘特仅显示所选订单工序。

### 1.2 甘特图：首行首列文字被盖住，修复
- 现象：甘特图**第一行（最上方机器分组）的左侧标签文字**被遮挡 / 裁切（疑似 `.vis-labelset .vis-label .vis-inner` 宽度不足、`overflow` 裁切，或时间轴内容 / 嵌套分组缩进 `padding-left` 压住标签）。
- 修复（仅改 `.gantt-canvas` 相关 CSS 与必要的 `renderTimeline` 容器，定位约 2183 / 2497）：
  - 确保左侧 labelset 宽度足以容纳机器名，`.vis-inner` 不被 `overflow:hidden` 裁掉；必要时增大 labelset 宽度 / 设为 `overflow: visible` / 提升 `z-index` 高于时间轴前景。
  - 移除或修正导致首行标签被压住的嵌套分组 `padding-left`（本次甘特不再需要多方案嵌套分组）。
  - 首行（最上方机器行）标签完整可见，不被第一条甘特条或时间轴覆盖。
- 验收：规则仿真甘特最上方机器分组的左侧标签文字完整显示，无遮挡、无裁切。

### 1.3 甘特图基础能力（规则仿真 = 统一甘特的基础实例）
- 规则仿真甘特是「单结果（=当前规则）、按机器分组、订单多选」的实例，作为后续方案评审甘特的统一基础。统一要求：
  - **按机器分组**（vis groups：机器行）。
  - **全量渲染，默认窗口 4 天**：数据层传入完整排程（覆盖 D1…Dn 全量天数）；打开界面时用 vis timeline `setWindow` 把可视范围默认聚焦前 4 天（D1–D4），其余天数可横向滚动 / 缩放查看。**禁止以截断数据的方式只保留前 4 天**。
  - 订单筛选含「全选」与「所有订单」语义（见 1.1）。
  - 排版 / 卡片头 / 空态沿用既有 design-system（圆角、阴影、机器分组、时间轴网格、「现在」竖线等），标题为「规则仿真甘特图 · {规则名}」。

---

## Scope 2：【方案评审】界面

### 2.1 甘特图：以规则仿真甘特为基础迁移 + 增加方案单选
- **迁移规则仿真甘特到此界面**：复用 Scope 1 的统一甘特渲染（按机器分组 + 订单多选 + 全选 + 全量天数 / 默认 4 天窗口）。
- **在原有功能基础上增加「方案」选择，单选即可**：
  - 新增**方案单选**筛选控件（从候选方案中选 1 个；与 `app.reviewSelection` 解耦，单独状态如 `app.reviewRead.schemeId`）。选方案后**默认取该方案第一个订单**填入订单筛选。
  - 订单：**多选**（复用升级后的 `renderOrderCombobox` 多选模式，含「全选」/「所有订单」）。选中 0 个具体订单 = 该方案全部订单；选中 1+ 个具体订单 = 仅显示这些订单关联的工序。
  - **甘特只展示单个方案的排程**，**不再并排对比多个方案**（移除 `buildReviewGanttData` 约 3093 的 `nestedGroups` 中「机器 → 方案」多方案子树与方案色图例对比）。
  - 订单 / 方案筛选切换时，重新拉取或过滤该方案排程（复用 `loadReviewData` / 现有过滤逻辑），甘特重渲染。
- **保留两个筛选框：方案 + 订单**（方案单选、订单多选）。
- 文案 / 卡片头同步调整：标题如「方案排程甘特（按机器）」，说明改为「选择单个方案与一个或多个订单，查看该方案在所选订单上的排产；默认显示前 4 天，可滚动查看全部」。
- 验收：选方案 A → 甘特显示 A 全部订单排产、订单框默认「所有订单」、打开时可视窗口默认 4 天但数据含全部天数（可滚动/缩放看到 D5+）；选「全选」→ 显示该方案下所有订单；取消「所有订单」并多选订单 X、Y → 甘特只剩 X、Y 的工序；切换方案 → 甘特整体换为另一方案排程，无多方案并排。

### 2.2 方案对比表：方案列去掉小字
- 位置：`renderReviewCandidateComparison()` 约 3001 的 `<small>${candidateSourceLabel(item)} · ${candidateModeLabel(item)}</small>`。
- 改动：**删除该行 `<small>` 小字**，方案列只保留方案名（含方案色点 `.scheme-dot`）。来源/模式信息若需保留，可移入「查看详情」面板，但默认不铺在对比表内。
- 验收：对比表每行只有方案名 + KPI + 操作列，无副标题小字。

### 2.3 新增「每日机器分类利用率」+ 查看详情趋势
- 背景：当前 `renderReviewTypeUtilization()` 只给「每机器类型 × 每方案」的**聚合**利用率（无按天）。需新增**按天**维度。
- 计算口径（前端从排程明细计算，避免后端改动）：
  - 数据源：每个方案的 `schedule`（= `app.reviewRead.schemes[solutionId]`），每条 op 含 `machine_id/machine_name, start, end, status`。
  - 机器类型归属：用实例机器主数据把 `machine_id` 映射到机器类型（schedule 已带 `machine_type` 直接用；否则 join `/api/instance` 或 `app.instanceDetails` 机器列表）。
  - 按天分桶：以 `plan_start_at`（无则用 fallback 基准）为第 0 天起点，按 24h（或班次时长）滚动分桶，得到 D1、D2…Dn。
  - 单机类型日利用率 = 该类型当日 busy 时长 /（该类型机器数 × 当日窗口小时数）。
- 表格呈现：**行 = 机器类型，列 = 天（D1…Dn）**，单元格 = 该类型当天利用率 %；每行最佳值加粗（`.is-best`）。无数据的天显示 `-`。
- 查看详情趋势：操作列「查看详情」按钮（`focus-candidate`，约 3005）**改作打开该方案每日利用率趋势**：
  - 形式：抽屉 / 弹层 / 卡片内展开区，标题「{方案名} 每日机器分类利用率趋势」。
  - 内容：每个机器类型一条**日利用率折线（或面积）趋势**（横轴 D1…Dn，纵轴利用率%），与上方表格同色同口径。可用轻量 SVG / Canvas 自绘，不引入新图表库。
  - 原有「置顶 + 高亮」行为如仍需要，可改由点击方案名或保留独立动作，但**默认查看详情 = 打开趋势**。
- 数据获取：优先用已加载 `app.reviewRead.schemes`；未加载则触发 `loadReviewData` 后再计算。
- 验收：勾选某方案 → 利用率区出现「机器类型 × 天」表格；点查看详情 → 弹出该方案每日利用率趋势折线，与表格口径一致。

### 2.4 机器分类利用率对比默认折叠
- 当前 `renderReviewTypeUtilization()` 渲染后默认**展开**。改为**默认折叠**（collapsed）。
- 实现：用 `<details>` + `<summary>`，或自定义折叠头（点击展开/收起，箭头旋转），默认不设置 `open`。折叠头显示「机器分类利用率对比」+ 当前勾选方案数等摘要。
- 验收：进入方案评审页，利用率卡默认收起；点击展开后内容正常。

---

## Scope 3：【数据导入】界面 —— 图谱改回扁平（7月15号图）

- 背景：当前 `layoutGraph()`（约 2576）按节点类型分**泳道（swimlane）**、加**家族带（families）**、层级排序——即用户说的「层次感」。用户认为更凌乱，**要改回 7月15号那版的扁平布局（无层次感）**。
- 改动（仅改 `layoutGraph()` 及其依赖的泳道/家族逻辑）：
  - **移除按类型的竖向泳道分隔带**（`GRAPH_LANE_CONFIG` 的 lane 背景/宽度计算、`lanes` 家族带 `families`）。
  - **移除层级排序与家族分组**带来的强结构；节点改为**扁平排布**（按类型聚簇但不画分隔带；或力导向/网格平铺；节点间用边自然连接）。
  - 保留节点**类型配色**（`graphTypeColor`）、**形状/尺寸编码**、节点间边样式。
  - 画布平移/缩放、视口适配（`fitGraphViewport` 等）保留。
- **明确保留（不要动）**：
  - 节点详情面板（彩色类型徽章 + 名称 + entity_id + 关键属性 + 上下游 + 关系分解）——在 `renderInteractiveGraph()`（约 4057）内。
  - 关系解释（边的类型/含义说明、图例）。
  - 你此前对文字类、按钮类的修改（构建图谱按钮、统计网格、面包屑等）。
- 验收：进入数据导入 → 可交互有向异构图节点**不再按类型分泳道/层级**，整体扁平（即 7月15号图风格）；点击节点仍能看到详情与关系解释；文字/按钮无回归。

> 备注：「7月15号图」按"去掉层次感、扁平化、回到 7月15日那版布局"实现。若你指的是某个具体参考图（如图15）的精确样式，请提供该图或描述其布局，再据此微调 `layoutGraph`。

---

## 通用约束
- **禁止使用 `superpowers` skill**：本次所有实现**不得**调用、触发或依赖 `superpowers` 及其任何子能力 / 命令，仅用原生代码编辑能力完成上述全部修改。
- 仅修改 `frontend/app_v2.js` 与 `frontend/app_v2.css`（后端排程数据已在前端可用，无需改 `api/server.py`；如确有必要新增端点，先说明再动）。
- 保持 `design-system.css` token 一致，不引入新硬编码色值（用 token 或既有 `SCHEME_COLOR_TOKENS`）。
- 命名规范不变：方案显示名用「方案一/方案二…」，禁止 `S-` 乱码；**订单一律按订单号展示，禁止 id.name 拼接**。
- 改动后必须本地起服务验证，覆盖三界面：① 规则仿真订单多选/全选 + 订单号展示 + 首行首列标签无遮挡；② 方案评审甘特（方案单选/订单多选含全选/单方案/全量天数默认4天）+ 对比表无小字 + 每日利用率表 + 查看详情趋势 + 利用率默认折叠；③ 数据导入图谱扁平无层级且详情完好。
- 性能：大实例（千级机器/万级工序）下甘特与利用率计算需分页/抽样/增量，不得一次性全量渲染卡死（参照现有 `GANTT_MAX_GROUPS` / 订单抽样逻辑）。
