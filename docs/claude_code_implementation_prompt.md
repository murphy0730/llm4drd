# 实施提示词：LLM4DRD 排产系统前端可视化优化

> 你（Claude Code）是一名资深前端 / 数据可视化工程师。请基于以下规格修改 LLM4DRD 排产系统前端代码，实现「方案评审库视图」与「图谱 / 甘特图」的优化。行号可能随修改偏移，请以**函数名**为准用 grep 定位。

---

## 0. 项目与技术上下文

- 项目根目录：`/Users/zhouwentao/Desktop/llm4drd`
- 前端主逻辑：`frontend/app_v2.js`
- 前端样式：`frontend/app_v2.css`
- 设计 token：`frontend/design-system.css`（浅色主题）
- 本地验证：项目根目录执行 `python run_server.py`，浏览器打开 `index_v2.html`，进入「方案评审」页 / 图谱 / 甘特图。

### 设计 token（必须复用，保持一致性）
- 主色 `--primary: #2f6feb`
- 强调色 `--accent: #c2620a`
- 成功 / 已完成 `--success: #12a150`
- 信息 / 进行中 `--info: #0e8e7f`
- 圆角 `--radius-md`、阴影 `--shadow-sm`、弱化文字 `--text-soft`、容器留白 `--inset`
- 选中态 `.is-selected`、最佳值 `.is-best` 已存在，直接复用，不要新建同名类。

### 关键函数 / 状态（grep 定位）
- `renderReviewLibraryTab()`（约 2830）——方案评审库视图总渲染
- `renderCandidateCards()`（约 2679，将被停用）
- `renderReviewCandidateComparison()`（约 2737）——全量 KPI 对比表
- `renderReviewTypeUtilization()`（约 2777）——机器分类利用率对比
- `getReviewCandidates()`（约 904-921）——候选方案组装与命名
- `getSelectedReviewCandidates()`——返回当前勾选集合
- `normalizeCandidate()`（约 858）——候选标准化
- `renderTimeline()`（vis-timeline 甘特渲染）
- 勾选状态：`app.reviewSelection`（共享上限 4）；聚焦：`app.reviewDetailId`
- 勾选交互：`data-action="toggle-candidate"` + `data-id`
- 操作交互：`export-selected-solution` / `focus-candidate` / `send-candidate-to-ai`
- 启发式：`CONFIG.HEURISTIC_RULES`、`HEURISTIC_RULE_BLURB`、`load-heuristic-references`、`data-heuristic-select`
- 图谱：`renderInteractiveGraph()`（约 3984）、`layoutGraph()`、`buildGraphViewModel()`、`buildOrderScopedNodeSet()`、`graphTypeColor`（约 1698）
- 甘特：`mountGantts()`（约 4510）

---

## Scope A：方案评审库视图优化（已与用户确认，直接实现）

### A.0 已确认决策
1. **添加启发式参照：保留并优化**——用 chip 选择器替换 `<select multiple>` + 按钮；点击即载入为候选行并进入统一对比与联动。
2. **甘特联动方式：按机器分组**——每台机器行内呈现每个勾选方案的排产，每方案独立配色（vis `nestedGroups`：机器 → 方案）。
3. **勾选范围：沿用共享上限 4**——`app.reviewSelection` 与 AI 评审共用，勾选即同时驱动「利用率对比 + 甘特 + AI 评审」。
4. **命名规范：不使用 `S-` + 乱码**——候选方案显示名一律用 `方案一 / 方案二 / 方案三…`；已有可读名（基线方案、启发式参考·X、精确冠军参考）不变；任何回退到 `solution_id`（即 `S-xxx`）的位置改为干净标签。

### A.1 合并方案卡片 → 统一对比卡
- 在 `renderReviewLibraryTab` 中**删除对 `renderCandidateCards` 的调用**，并停用 / 移除 `renderCandidateCards` 函数体，避免死代码。
- 改造 `renderReviewCandidateComparison` 为「方案对比」表，列结构：
  `[选] | 方案(名称 + 副标题：来源·模式) | 主目标 KPI… | 其他 KPI… | [操作]`
- 每行最左为勾选框，复用 `toggle-candidate` 与 `app.reviewSelection`；选中行加 `.is-selected`（左侧主色竖条 + 浅色底纹）。
- 保留「每列最佳值加粗」`.is-best`；保留「聚焦方案置顶 / 高亮」（行内「查看详情」置顶并设 `app.reviewDetailId`）。
- 基线方案、`S-d646416335`（现应显示为 `方案一` 等）直接作为首行展示。

### A.2 勾选框联动（核心）
- `renderReviewTypeUtilization`：将当前基于 `reviewComparePreview`（只看 `reviewDetailId`）改为基于 `getSelectedReviewCandidates()`（勾选集合）；列头即各勾选方案名 + 方案色图例；未勾选时显示空态提示（如「请勾选方案以对比利用率」）。
- 甘特：将末尾只渲染 `app.reviewDetailId` 单方案的 `renderTimeline`（约 2879-2884）改为渲染勾选集——新增 `renderReviewGantt(selected)`：**按机器分组**（vis `nestedGroups`：机器 → 方案），每个勾选方案一种颜色（稳定色板 `--primary / --accent / --success / --info` 循环）。
- 给勾选集合分配稳定色板（按 token 循环），该色贯穿：勾选态、对比表高亮、利用率列头、甘特条块 className（`.scheme-c-0..3`），形成视觉闭环。

### A.3 操作列（导出能力）
- 对比表末尾新增【操作】列，每行含三个紧凑 icon 按钮：导出（复用 `export-selected-solution` + `data-id`）、查看详情（`focus-candidate`）、送入 AI（`send-candidate-to-ai`）。
- 保留系统页「导出与交付」（模板 / CSV / 选中方案 Excel）作为批量入口，与逐行导出互补。

### A.4 方案池工具 chip 化
- 把「添加启发式参照」的 `<select multiple>` + 按钮（约 2862-2868）替换为内联 chip 列表：规则名来自 `CONFIG.HEURISTIC_RULES`（ATC / EDD / SPT / CR / FIFO / LPT），一句话说明来自 `HEURISTIC_RULE_BLURB`。
- 点击 chip 即切换 `app.heuristicSelection` 并即时加载为候选行（复用 / 改写 `load-heuristic-references` 逻辑），自动进入统一对比表与联动。
- 事件由 `data-heuristic-select` 改为 chip 点击 `data-action="toggle-heuristic"`。
- 卡片加一句价值说明（「教科书基准锚点：衡量优化器相对经典派工规则的提升」）；视觉统一到 `.tag` / `.chip` 体系。

### A.5 命名规范（仅改显示名，不动 id）
修改 `getReviewCandidates()`（约 904-921）：
- Pareto / 优化解分支：`name: item.solution_id || \`Pareto-${index + 1}\`` → 改为 `name: \`方案${index + 1}\``。
- 其余任何 `name` 回退到 `solution_id` 处（如 `reference_solutions` 缺 `rule_name` 时），一律改为干净标签（如 `参照方案${n}`），确保界面不再出现 `S-` + 乱码。
- `id` / `solutionId` 仍保留原始 `S-xxx`，勾选、AI 评审、导出映射完全不变。
- 影响范围：对比表、利用率列头、甘特标题、AI 选项、导出文件名全部自动跟随 `item.name`，无需逐处改。

### A.6 美观与一致性
- 复用 design-system.css token（圆角、阴影、主色、弱化文字）。
- 选中态 / 最佳值 / 聚焦三态用同一套语义色。
- 表格置于 `--inset` 容器内横向滚动；行高加大、表头 sticky；用分区标题 + 留白替代多余分隔线，减少卡片堆叠。

### A.7 文件改动清单
- `frontend/app_v2.js`：合并卡片进对比表；利用率改用勾选集；库页顺序整理；甘特支持勾选集（按机器分组 + 方案色）；方案池 chip 化；`getReviewCandidates` 命名改造。
- `frontend/app_v2.css`：对比表 / 勾选列 / 操作列 / 联动色（`.scheme-c-0..3`）/ 机器分组甘特 / chip 样式。

### A.8 验证（本地，必须执行）
1. `python run_server.py` → 方案评审库。
2. 勾选「基线方案·ATC」与「方案一」「方案二」→ 核对：
   - 二者进统一对比表首行并高亮（`.is-selected`），显示名为「方案一 / 方案二」而非 `S-xxx`；
   - 利用率对比只显勾选列（带方案色列头）；
   - 甘特按机器分组、每台机器呈现多方案排产且各自配色；
   - 操作列可逐行导出；取消勾选即退出联动。
3. 方案池用 chip 选 ATC / EDD 等即载入为候选行并进入联动。
4. 做对比度 / 色盲友好检查（配色区分度）。

---

## Scope B：图谱与甘特图优化（方案已提出，实现前必须先向用户确认两处决策）

> 以下为已设计的优化方案，用户尚未确认关键决策，**请勿直接落地**，先就 B.3 决策点与用户对齐后再实现。

### B.1 图谱（renderInteractiveGraph）
- **配色**：规划链冷色（订单=靛蓝 `#185FA5`、任务=蓝绿 `#0F6E56`、工序=绿 `#3B6D11`），资源层暖 / 彩色（机器=琥珀 `#854F0B`、工装=玫红 `#993556`、人员=紫 `#534AB7`）。替换现有 `graphTypeColor`（现 `order #0f4c81` 与 `task #2c7fb0` 同蓝易混）。
- **形状 + 尺寸双编码**（色盲友好）：订单=大圆角矩形 / 任务=中矩形 / 工序=小圆角矩形 / 机器=圆形 / 工装=菱形 / 人员=三角。
- **布局**：层级泳道——每列加淡色竖向泳道背景 + 顶部层级标签（订单 / 任务 / 工序 / 资源）；结构边实线、资源边虚线。
- **按订单过滤**：顶部加「订单」下拉（复用甘特图同款交互），选中即把图 scope 到该订单簇（订单 + 任务 + 工序 + 挂载机器 / 工装 / 人员），显示面包屑统计（`当前订单：A · 2 任务 / 3 工序 / 1 机器`）。在 `app.graphView` 加 `orderFilter`，接入 `buildOrderScopedNodeSet`。
- **信息呈现**：右栏节点详情（彩色类型徽章 + 名称 + entity_id + 关键属性 key-value + 上下游 + 关系分解）；画布常驻图级统计条（订单 / 任务 / 工序 / 机器总数、边数）+ 底部图例（颜色 ↔ 类型 ↔ 形状、线型 ↔ 关系）。

### B.2 甘特图（renderTimeline / mountGantts）
- 新增「分组方式」切换：按机器（现状）/ 按订单层级。按订单层级时用 vis `nestedGroups` 建树：订单 ▸ 任务令 ▸ 工序，工序行标注所用机器（`工序 O1 · M1`）。
- 订单过滤保持首要；选中订单后层级收拢到该订单子树。
- 配色与图谱联动（订单同色）：默认按状态着色（已完成绿 / 进行中蓝 / 未来琥珀），增加「按订单着色」模式；选中订单时该订单条块高亮、其余淡化。
- 美观：条块圆角 + 轻投影、行高加大、时间轴网格清晰、加「现在」竖线（有真实 `plan_start_at` 时）；图例 / 汇总条保留并打磨。

### B.3 待确认决策（实现前必问用户）
1. 图谱「按订单过滤」要**单选**一个订单，还是支持**多选 / 高亮多个**订单？
2. 甘特图默认视图保持「按机器」，还是默认切到「按订单层级」？

---

## 通用约束
- 仅修改 `frontend/app_v2.js` 与 `frontend/app_v2.css`（Scope B 需先确认）。
- 保持 design-system.css token 一致，不引入新硬编码色值（用 token 或 Scope 内明确给出的色值）。
- 改动后必须本地起服务验证，不得只改代码不验证。
- 优先实现 Scope A（已确认），Scope B 待决策后再做。
