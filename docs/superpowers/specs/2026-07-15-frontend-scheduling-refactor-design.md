# 前台调度界面精简重构 — 设计

- 日期：2026-07-15
- 目标文件：`frontend/index_v2.html`、`frontend/app_v2.js`、`frontend/app_v2.css`
- v1 文件（`frontend/index.html`、`app.js`、`app.css`）确认无后端引用后一并删除

## 背景与目标

现有前台调度界面存在大量占位标题、失效按钮和无入口的死页，干扰实际排产流程。本次重构把调度流程精简为 4 步，删除所有无意义内容，使界面聚焦于真实工作流：

```
01 数据导入 → 02 仿真与洞察 → 03 优化求解 → 04 方案评审
```

## 1. 顶部栏（topbar）

| 现状 | 改为 |
|---|---|
| 品牌区 + 当前实例 pill | 保留 |
| `订单 / 工序`（一个框，`summary.orders / operations`） | 拆成三项计数：**订单** `summary.orders`、**任务令** `summary.tasks`、**工序** `summary.operations` |
| `资源`（一个框） | 拆成三项：**机器 / 工装 / 人员**（`summary.machines / toolings / personnel`） |
| `优化状态` 框 | 删除 |
| `刷新`、`帮助` 按钮（`topbar-actions`） | 整块删除，含 `refresh-all`、`toggle-help` 的 action handler |

对应更新 `updateTopbar()` 中 `topbar-orders-ops`、`topbar-resources`、`topbar-opt-status` 的写值逻辑（改为分别写入新增的计数元素）。

## 2. 右侧上下文面板

- `<aside class="context-panel" id="context-panel">` 整块删除（含内部四张 context-card）。
- 删除 `收起/展开` toggle 及 `toggle-panel` action handler。
- 删除所有只服务于该面板的 `panel-*` 元素写值逻辑与相关 CSS。
- 布局从「侧边栏 + 主舞台 + 右面板」三栏改为「侧边栏 + 主舞台」两栏。

## 3. 左侧导航（调度流程）

```
01 数据导入      data-nav="new-scene"        (原「新建与导入」重命名)
02 仿真与洞察    data-nav="simulate"         (原 03)
03 优化求解      data-nav="optimize-launch"  (原 04)
04 方案评审      data-nav="solution-review"  (原 05)
```

- 删除「实例与约束」导航项（`data-nav="config"`）及其 `nav-index`。
- 重新编号 01–04。
- 辅助分组：保留「工作概览」（`data-nav="dashboard"`）。
- 系统分组：保留「大模型连接」「系统状态」。

## 4. 页面级改动

### ① 数据导入（`page-new-scene`）
- 删除页面顶部标题块 `page-header`（eyebrow「Create」+ h2 + 描述 + `同步当前实例` 按钮）。
- 工作区顶部直接是**导入与模板**卡片。
- 卡片下方接**数据强校验**面板（从 `page-config` 迁移过来的 `renderValidation` 输出）。
- 删除**快速生成**表单 `form-generate-v2`，及 `generate-instance`、`fill-now-start` 相关 action 与逻辑。
- 导入完成后**当页立即触发并展示数据强校验结果**，不再跳转到「实例与约束」。（校验本身已在导入流程中运行；改动点是把结果渲染在本页而非跳转。）

### ② 实例与约束（`page-config`）
- 数据强校验迁到①后，**整页删除**：
  - `index_v2.html` 中 `<section id="page-config">` DOM。
  - `app_v2.js` 中 config 渲染函数、`configTab` 状态、5 个 config tab（instance / orders / operations / resources / downtime）渲染逻辑。
  - `NAV_MAP` 中 `config`、`instance-setup`、`order-maintenance`、`resource-maintenance`、`downtime-management` 路由项。

### ③ 仿真与洞察（`workflow` 页 step 3）
- 删除中间「工作台」标题块与 `workflow-rail`（每步示例）。
- **只保留图谱视图 / 完整仿真页两个 Tab**。
- 进入该页默认 `workflowFocus: "graph"`（图谱视图）。
- 两个视图（graph / simulate focus）功能均保留不变。

### ④ 优化求解（`workflow` 页 step 4）
- 删除「多目标优化配置」卡片**以上**的所有内容（实例摘要、图谱构建、规则仿真等占位块）。
- 从「多目标优化配置」开始向下的内容全部保留。

### ⑤ 方案评审（`page-review`）
- 删除页面顶部标题块 `page-header`。
- 保留下方 `方案库 / 精确冠军参考 / AI 评审` Tab 工作流。

## 5. 关联清理

- **`page-insights`（结构/资源/瓶颈）死页：删除。** 当前侧边栏无入口，已是死页。删除其 DOM、`NAV_MAP` 中 `insights` / `structure-analysis` / `resource-analysis` / `bottleneck-analysis` 路由，及其渲染函数。
- 删除因上述改动而成为孤儿的 JS：`refreshAll`、`toggle-help`、`toggle-panel`、config 渲染族、快速生成族、workflow-rail 渲染、insights 渲染族。
- 删除只服务于已删元素的 CSS 规则（context-panel、topbar-actions、快速生成表单、workflow-rail 等）。

## 实现方式

③④ 同属动态 `workflow` 页（rail + content 结构）。采用**改渲染逻辑、不新建页面结构**的方式：
- step3 渲染改为「只渲染 graph / simulate 两个 Tab、默认 graph」。
- step4 渲染改为「只渲染优化配置及以下」。
- 去掉共享的 rail / header 渲染。

这样 diff 最小且贴合现有架构。

## 验证标准

1. 顶部栏显示 订单/任务令/工序 三项 + 机器/工装/人员 三项，无优化状态、无刷新/帮助按钮，无右侧面板。
2. 左侧导航为 4 步（数据导入/仿真与洞察/优化求解/方案评审）+ 工作概览 + 系统项。
3. 数据导入页：无标题块，导入与模板在顶部，数据强校验在下方；导入 Excel 后当页显示校验结果。
4. 仿真与洞察页：默认图谱视图，两个 Tab 可切换且功能正常。
5. 优化求解页：只从多目标优化配置开始。
6. 方案评审页：无标题块，三 Tab 工作流正常。
7. 无残留死页、死按钮、控制台报错；v1 文件已删除且服务器正常启动。
