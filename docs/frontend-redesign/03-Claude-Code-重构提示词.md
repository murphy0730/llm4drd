# Claude Code 任务：LLM4DRD 前端重构实施

> 用法：把本文件整段作为提示词发给 Claude Code（工作目录 = 本仓库根目录）。
> 设计稿是唯一视觉与交互来源；后端缺什么能力就补什么能力。

---

## 0. 你的任务

根据设计稿 `docs/frontend-redesign/02-设计稿.html`（下称 **设计稿**），对 `frontend/` 下的现有前端做**整站重构改造**，并把设计稿里新增、但后端尚不具备的能力补齐到 `api/server.py`。

约束一句话：**视觉/交互以设计稿为准，数据/接口以后端为准；设计稿里的演示数据必须全部替换为真实 API 数据。实现任务不要使用superpowers skill。**

---

## 1. 关键文件

| 角色 | 路径 |
|---|---|
| 设计稿（视觉+交互唯一来源） | `docs/frontend-redesign/02-设计稿.html` |
| 信息架构分析（功能清单与保留约定） | `docs/frontend-redesign/01-功能与信息架构分析.md` |
| 现有前端入口 | `frontend/index_v2.html` |
| 现有前端逻辑 | `frontend/app_v2.js`（约 6000 行） |
| 现有样式 | `frontend/app_v2.css`、`frontend/design-system.css` |
| 现有第三方依赖 | `frontend/vendor/`（vis-timeline、cytoscape、dagre，**沿用**） |
| 后端服务 | `api/server.py`（Flask，`@app.post/@app.get` 路由） |
| 优化/评审相关 | `optimization/`、`api/review_read.py` |

先通读设计稿和分析文档，再读 `app_v2.js` 的关键渲染函数（下面第 4 节给了行号锚点）。

---

## 2. 绝对不可破坏的既有约定（违反即返工）

1. **方案显示命名**：界面一律显示「方案一 / 方案二 / 方案三…」，**禁止**出现后端 `S-xxxxxxxx` 式 solution_id。参考 `app_v2.js` 的 `getReviewCandidates()`（约 904–921 行）。只改显示 name，不动 id/solutionId。
2. **评审勾选上限 4 个**，与 AI 评审助手**共用同一个勾选集**（`app.reviewSelection`），超限要 toast 提示。
3. **状态样式语义**：复用 `.is-selected`（选中）/ `.is-best`（最优/冠军）。
4. **设计 token**：沿用 `design-system.css` 的变量——`--primary: #2f6feb`、`--accent: #c2620a`、`--success: #12a150`、`--info: #0e8e7f`。新样式用这些变量，不要另造色值。
5. **甘特图组件**：继续使用 `vis-timeline`（`frontend/vendor/`），**甘特本体样式、停机遮罩、在制锁定、条块按订单标识色着色全部保持不变**。只能改甘特上方的筛选菜单，不能改甘特渲染。
6. **图谱组件**：继续使用 `cytoscape` + `cytoscape-dagre`（`frontend/vendor/`），节点按类型颜色+形状+尺寸三重编码、滚轮缩放、拖拽平移保持不变。

---

## 3. 重构后的信息架构（以此为准）

```
┌ 顶栏：品牌 + 场景 pill + 实例 KPI 条 + LLM 状态
├ 流程进度条（全局常驻，5 步）：数据导入 → 图谱构建 → 规则仿真 → 优化求解 → 方案评审
│   每步带状态（✓完成 / ●当前 / 待开始），可点击跳转，当前步高亮
├ 侧栏导航（无步骤编号，按组）：
│   调度流程：数据导入 · 图谱构建 · 优化求解 · 方案评审（各带状态灯/徽标）
│   分析：工作概览
│   系统：导出与交付 · 大模型连接 · 系统状态
└ 主区：页面只保留「前置检查条 + 内容」，删除每页大标题 Header 区
```

**工作流只由顶部进度条表达**，侧栏和页头不再重复步骤名（这是 v2 的核心修订点，不要再加回去）。

---

## 4. 逐页实施要求

### 4.1 数据导入（import）
- 上传 Hero 区 + 9 类 Sheet 标签；数据强校验改为「通过项/警告/错误」三档大色块摘要 + 明细表。
- 复用现有逻辑：`renderValidationPanel()`（app_v2.js ~2944）、导入流程 `handleImportFile`（~4871）。
- API：`POST /api/instance/import-excel`、`GET /api/instance/validate`、`GET /api/instance/validate/export`、`GET /api/instance/template`。

### 4.2 图谱构建（graph）—— 本页布局变化最大
- 基于现有「可交互有向异构图」重构，参考 `renderInteractiveGraph()`（app_v2.js ~4071–4375）。
- **画布最大化**：整页宽 + 最小高 560px；右侧详情栏收窄到 320px。
- 工具栏**只保留**：视图模式（焦点邻域/全关系视图）+ **按订单过滤**一个下拉框。
  - ⚠️ 删除设计稿迭代中已去掉的：订单号跳转输入框、节点搜索框（**不要再实现**）。
- 保留：面包屑统计（订单▸任务▸工序▸机器/工装/人员▸关系）、节点层级/关系层级筛选 chip、画布统计、缩放/适配/重置/全屏、右侧节点详情与关系分解。
- API：`POST /api/graph/build`、`GET /api/graph/meta|nodes|edges|order/{id}|status/{task_id}`。

### 4.3 规则仿真（simulate）
- 规则选择改卡片式（ATC/EDD/SPT/CR/FIFO/LPT + 一句话业务解释）。
- 指标 KPI 卡（总延误/总周期/净可用利用率/完成工序）+ 仿真不完整告警通栏。
- **甘特筛选只保留 4 个，排成一排**：订单（多选）、机器类型（多选）、工序状态、分组（按订单层级/按机器资源）。一排放不下时筛选行内部横向滚动。
  - ⚠️ 删除：仅含停机、机器号搜索、工序搜索、时间窗输入（**不要再实现**）。
  - 参考现有筛选渲染（app_v2.js ~2230–2308），做减法。
- API：`POST /api/simulate`。

### 4.4 优化求解（optimize）
- 左侧参数配置（sticky）+ 右侧运行状态。
- **多目标选择改紧凑 chip 流式网格**（目标很多时自然换行，方向 min/max 小字标注），参考 `renderObjectiveSelectors`（~2866）做样式重构。
- 运行状态：进度环 + 阶段指示（近似广搜 → 精确精修 → Pareto 前沿重算）+ 4 个 KPI 卡 + 运行日志流。
- API：`GET /api/optimize/objectives`、`POST /api/optimize/hybrid`、`GET /api/optimize/hybrid/status/{task_id}`、`GET /api/optimize/hybrid/result/{task_id}`。

### 4.5 方案评审（review）—— 功能最密的一页
**结构**：标题带（含 Pareto/启发式/精确冠军/已选 计数）→ Tab 条（方案库 / 精确冠军参考 / AI 评审助手）→ Tab 内容。

**通用滚动规则（总要求）**：任何 Tab 内容过宽或过高，**只在该 Tab 容器内部出滚动条**，页面框架（标题带、Tab 条、前置检查条）固定不动。设计稿里对应 `.rtab-scroll`。

**Tab 1 方案库**：
- **方案对比表**：方案多了表体纵向滚动 + 表头吸顶。基线方案（ATC）每个 KPI 单元格带 Δ 标注（箭头随升降、绿=改善/红=变差、括号百分比），最优值加粗标「最优」。参考 `renderReviewCandidateComparison`（app_v2.js ~3077–3147）。
- **列配置**：下拉面板勾选要展示的 KPI 列；**默认显示 3 个主目标 + 2 个常用 KPI 共 5 列**，主目标列带标记；隐藏集存 localStorage（沿用 `review-compare-hidden-cols-v1`）。KPI 全集来自目标目录接口。
- **操作列**（每行 3 个小按钮）：
  - `◎ 详情`：联动下方「机器分类利用率对比」和「排程甘特」切换到该方案（高亮当前）。对应现有 `focus-candidate`。
  - `✦ AI 评审`：把该方案送入 AI 评审勾选集并切到 AI Tab。对应 `send-candidate-to-ai`。
  - `⬇ 导出`：导出该方案 Excel。对应 `export-selected-solution`。
- **机器分类利用率对比 · 当前方案**：机器类型 × D1..Dn 逐日利用率矩阵；≥90% 过载红、<60% 偏低黄、行内最佳绿加粗。参考 `renderReviewTypeUtilization`（~3196–3258）+ 趋势 `renderDailyUtilTrendSvg`（~3170）。
  - API：`GET /api/optimize/hybrid/result/{task_id}/machine-type-daily-utilization`。
- **排程甘特 · 当前方案**：与仿真甘特**同一组件同一样式**，筛选同样只保留 4 个一排；随「◎ 详情」联动。参考 `renderReviewGantt`（~3290–3317）。
  - API：`GET /api/optimize/hybrid/result/{task_id}/schedule`、`.../review-orders`。

**Tab 2 精确冠军参考**：单目标冠军 + 加权单目标冠军两个表单 + 最新冠军摘要（自动纳入方案库）。参考 `renderReviewExactTab`（~3388–3430）。
- API：`GET /api/exact/objectives`、`POST /api/optimize/exact-reference`。

**Tab 3 AI 评审助手**：已选方案摘要表 + 全量指标矩阵 + 对话流（比较/按诉求推荐/追问）。参考 `renderReviewAiTab`（~3455–3504）。
- API：`POST /api/ai/pareto/compare|recommend|ask`。

### 4.6 工作概览（dashboard）
Hero（下一步引导 + 主目标/方案数/关注方案）+ KPI 卡 + 三列（问题规模/优化进展/当前推荐关注）。参考 `renderDashboard`（~1658–1732）。

### 4.7 导出与交付（export，从系统 Tab 提升为独立页）
下载模板 / 导出实例 CSV / 导出当前方案 Excel（以评审聚焦方案为准）。
- API：`GET /api/instance/template`、`GET /api/instance/csv`、`POST /api/optimize/hybrid/export-solution`。

### 4.8 大模型连接（llm）/ 系统状态（system）
- LLM：Base URL / 模型 / API Key + 保存/测试。API：`GET/PUT /api/config/llm`、`POST /api/config/llm/test`。
- 系统状态：健康、数据库、当前任务、前端版本、工作上下文。API：`GET /api/health`。

---

## 5. 需要补齐的后端能力（设计稿有、后端可能缺的）

设计稿大部分功能后端已有（见第 4 节 API 清单）。请**先核对** `api/server.py`，只补真正缺的，可能包括：

1. **顶部流程进度条的聚合状态**：5 步各自的完成度（实例已导入？校验过？图谱建了吗？仿真跑了吗？优化状态？已选方案数？）。若现有 `/api/workflow/progress` 不足以一次性喂给进度条，**扩展该接口或新增 `/api/workflow/overview`**，返回：
   ```json
   { "steps": [
     {"key":"import","state":"done|current|todo|blocked","label":"数据导入","detail":"校验通过 · 3 警告"},
     {"key":"graph","state":"...","label":"图谱构建","detail":"428 节点 · 1204 边"},
     {"key":"simulate","state":"...","label":"规则仿真","detail":"ATC · 可行"},
     {"key":"optimize","state":"...","label":"优化求解","detail":"运行中 · 63%"},
     {"key":"review","state":"...","label":"方案评审","detail":"已选 2/4"} ] }
   ```
2. **优化运行日志流**：设计稿右侧有求解器关键事件日志。若 `hybrid/status` 不含事件流，在优化任务里记录关键事件（初始化/每代发现非支配解/阶段切换/精修轮次），经 `.../status/{task_id}` 或新接口 `.../logs/{task_id}` 暴露。
3. **优化阶段细分**：进度条的三阶段（近似广搜/精确精修/Pareto 重算）需要后端在 status 里给出 `phase` 字段与百分比。
4. **实例 KPI 条**（顶栏订单/任务/工序/机器/工装/人员计数）：若 `/api/instance/details` 已含则直接用，不重复造。

原则：**能复用现有接口就不新增**；新增接口遵循现有 Flask 风格（`@app.get/@app.post`，JSON 返回，线程池处理长任务，参考已有端点写法）。

---

## 6. 工程要求

- **不改技术栈**：原生 HTML + JS + CSS，沿用 `frontend/vendor/` 的 vis-timeline / cytoscape。不引入 React/Vue/构建工具。
- 样式写进 `app_v2.css`（重构）并复用 `design-system.css` 的 token；深色模式变量保留。
- 路由/导航用现有的 hash + `data-nav` 机制扩展（新增 `graph`、`export` 两个一级页面）。
- 所有异步长任务（导入、校验、图谱构建、优化）沿用轮询 + 进度反馈，不阻塞 UI。
- 每完成一个页面自查：数据来自真实 API、显示命名符合约定、空态/加载态/错误态齐全。

---

## 7. 验收清单（逐项过）

- [ ] 顶栏 KPI 条 + LLM 状态实时；场景 pill 可点击回导入页
- [ ] 流程进度条 5 步状态正确、可点击跳转、当前步高亮
- [ ] 侧栏无步骤编号，状态灯/徽标准确
- [ ] 页面无大标题 Header 区，前置检查条正常
- [ ] 图谱页：仅「视图模式 + 按订单过滤」，画布够大，右侧详情完整
- [ ] 仿真/评审甘特：4 个筛选一排，甘特样式与原版一致
- [ ] 优化目标 chip 网格；运行进度环 + 三阶段 + 日志流
- [ ] 评审对比表：滚动吸顶、Δ 标注、列配置（默认 5 列、可勾选、存 localStorage）
- [ ] 操作列三按钮：详情联动利用率+甘特、AI 评审、导出
- [ ] 利用率按天矩阵：过载红/偏低黄/最佳绿
- [ ] 方案显示「方案一/二…」，无 solution_id 乱码
- [ ] 勾选上限 4 且与 AI 评审共用
- [ ] 各 Tab 内容超限只在 Tab 内滚动，页面框架固定
- [ ] 后端新增接口有对应测试或最小验证

---

## 8. 建议实施顺序

1. 骨架：顶栏 + 流程进度条 + 侧栏 + 页面容器（先做导航与状态流转）
2. 数据导入 / 图谱构建（图谱布局改动最大，早做）
3. 规则仿真（甘特筛选做减法）
4. 优化求解（目标 chip + 进度区 + 后端日志/阶段接口）
5. 方案评审（最复杂，放后面：对比表 → 列配置 → 操作列联动 → 利用率矩阵 → 甘特）
6. 工作概览 / 导出 / LLM / 系统状态
7. 联调 + 验收清单逐项过

开始吧。遇到设计稿与现有实现对不上的地方，**以设计稿为准**并在代码注释里说明取舍。
