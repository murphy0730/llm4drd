# 前台调度界面精简重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把前台调度界面精简为「数据导入 → 仿真与洞察 → 优化求解 → 方案评审」4 步，删除占位标题、失效按钮、死页与 v1 文件。

**Architecture:** 纯原生 JS 单页应用。页面骨架在 `frontend/index_v2.html`，逻辑与动态渲染在 `frontend/app_v2.js`，样式在 `frontend/app_v2.css`。改动以「删 DOM + 删/改渲染函数 + 修 NAV_MAP/状态」为主，不新增页面结构。

**Tech Stack:** 原生 HTML/CSS/JS（无框架、无打包器）；后端 FastAPI（`run_server.py` 提供 `/static/*` 与页面）。

## Global Constraints

- 目标文件仅限：`frontend/index_v2.html`、`frontend/app_v2.js`、`frontend/app_v2.css`；并删除 v1：`frontend/index.html`、`frontend/app.js`、`frontend/app.css`。
- 无 JS 单元测试框架。每个任务的验证 = ①静态 grep 断言（确认删除/新增字符串）②浏览器冒烟：启动服务后加载页面、切到相关页、**控制台无报错**。
- 每次改 `index_v2.html` 引用的静态资源后，把 HTML 里的 `?v=` 版本号递增，避免浏览器缓存旧文件。
- 设计依据：`docs/superpowers/specs/2026-07-15-frontend-scheduling-refactor-design.md`。
- 频繁提交：每个 Task 结束提交一次。

### 通用验证命令（多个任务复用）

- 启动服务：`python run_server.py`（默认监听，见文件内端口；假设 `http://127.0.0.1:8000/`）。
- 浏览器冒烟：用 claude-in-chrome 打开页面 → 依次点左侧导航 → 用 `read_console_messages` 确认无 `error`。
- 静态断言示例：`grep -c "关键字" frontend/index_v2.html`（期望 0 或 N）。

---

### Task 1: 删除 v1 遗留文件

**Files:**
- Delete: `frontend/index.html`
- Delete: `frontend/app.js`
- Delete: `frontend/app.css`

**Interfaces:**
- Consumes: 无
- Produces: 无

- [ ] **Step 1: 确认后端/HTML 不引用 v1 文件**

Run:
```bash
cd /Users/zhouwentao/Desktop/llm4drd && \
grep -rn "app\.js\|app\.css\|index\.html" run_server.py api/ frontend/index_v2.html | grep -v "app_v2\|app_v2.css"
```
Expected: 无输出（或仅注释）。若有真实引用，停止并上报。

- [ ] **Step 2: 删除文件**

```bash
cd /Users/zhouwentao/Desktop/llm4drd && git rm frontend/index.html frontend/app.js frontend/app.css
```

- [ ] **Step 3: 冒烟验证 v2 仍可用**

启动服务并加载 `/`（v2 页面）；确认页面正常渲染、控制台无 404/报错。

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: remove legacy v1 frontend files"
```

---

### Task 2: 顶部栏计数拆分 + 删除优化状态与操作按钮

**Files:**
- Modify: `frontend/index_v2.html:34-53`（`topbar-status` + `topbar-actions`）
- Modify: `frontend/app_v2.js:1383-1393`（`updateShell` 顶部写值）

**Interfaces:**
- Consumes: `getSceneSummary()` 返回 `{ orders, tasks, operations, machines, toolings, personnel }`
- Produces: 顶部元素 id：`topbar-orders`、`topbar-tasks`、`topbar-operations`、`topbar-machines`、`topbar-toolings`、`topbar-personnel`

- [ ] **Step 1: 替换 topbar-status 与删除 topbar-actions（index_v2.html）**

把 `index_v2.html` 第 34–53 行（`<div class="topbar-status">…</div>` 到 `<div class="topbar-actions">…</div>` 整段）替换为：

```html
      <div class="topbar-status">
        <div class="status-box"><span>订单</span><strong id="topbar-orders">-</strong></div>
        <div class="status-box"><span>任务令</span><strong id="topbar-tasks">-</strong></div>
        <div class="status-box"><span>工序</span><strong id="topbar-operations">-</strong></div>
        <div class="status-box"><span>机器</span><strong id="topbar-machines">-</strong></div>
        <div class="status-box"><span>工装</span><strong id="topbar-toolings">-</strong></div>
        <div class="status-box"><span>人员</span><strong id="topbar-personnel">-</strong></div>
      </div>
```

（即：删除原 `订单/工序`、`资源`、`优化状态` 三个框，改为 6 个独立计数框；整块删除 `topbar-actions`（刷新 + 帮助）。）

- [ ] **Step 2: 更新 updateShell 顶部写值（app_v2.js）**

把 `app_v2.js` 第 1383–1393 行（从 `el("topbar-orders-ops")...` 到 `: "未启动";` 结束的优化状态块）替换为：

```javascript
  el("topbar-orders").textContent = hasScene ? formatInt(summary.orders) : "-";
  el("topbar-tasks").textContent = hasScene ? formatInt(summary.tasks) : "-";
  el("topbar-operations").textContent = hasScene ? formatInt(summary.operations) : "-";
  el("topbar-machines").textContent = hasScene ? formatInt(summary.machines) : "-";
  el("topbar-toolings").textContent = hasScene ? formatInt(summary.toolings) : "-";
  el("topbar-personnel").textContent = hasScene ? formatInt(summary.personnel) : "-";
```

- [ ] **Step 3: 递增 HTML 资源版本号**

`index_v2.html` 顶部把 `app_v2.js?v=...` 与 `app_v2.css?v=...` 的版本号各加一（例如 `-29` → `-30`）。

- [ ] **Step 4: 静态断言**

```bash
grep -c "topbar-opt-status\|topbar-orders-ops\|topbar-actions\|refresh-all\|topbar-resources" frontend/index_v2.html
```
Expected: `0`

- [ ] **Step 5: 冒烟验证**

加载并导入/生成一个实例，确认顶部显示 订单/任务令/工序/机器/工装/人员 六项数字，无优化状态、无刷新/帮助按钮，控制台无报错。

- [ ] **Step 6: Commit**

```bash
git add frontend/index_v2.html frontend/app_v2.js && git commit -m "feat: split topbar into order/task/op + machine/tooling/personnel counts, drop opt-status and actions"
```

---

### Task 3: 删除右侧上下文面板

**Files:**
- Modify: `frontend/index_v2.html:259-305`（`<aside class="context-panel">…</aside>` 整段删除）
- Modify: `frontend/app_v2.js:1395-1422`（`updateShell` 中 `panel-*` 写值整段删除）
- Modify: `frontend/app_v2.js:6412-6414`（`toggle-panel` action handler 删除）
- Modify: `frontend/app_v2.css`（`.context-panel`、`.panel-toggle`、`.context-card` 等只服务该面板的样式删除）

**Interfaces:**
- Consumes: 无
- Produces: workspace 变为「侧边栏 + 主舞台」两栏

- [ ] **Step 1: 删除 aside.context-panel（index_v2.html）**

删除 `index_v2.html` 第 259–305 行整个 `<aside class="context-panel" id="context-panel">…</aside>`。

- [ ] **Step 2: 删除 updateShell 中 panel 写值（app_v2.js）**

删除第 1395–1422 行（从 `el("panel-scene-name")...` 到 `el("panel-opt-approx")...` 结束）——这些元素已不存在，`el()` 会报空。保留其后的 `document.querySelectorAll(".requires-scene")...`（1424 起）。

> 注意：若 `getSelectedReviewCandidate()`、`objectiveShortList()` 等在此块外无其他调用可保留；本步骤只删 `el("panel-...")` 赋值行，不删这些辅助函数。

- [ ] **Step 3: 删除 toggle-panel handler（app_v2.js）**

删除第 6412–6414 行 `if (action === "toggle-panel") { ... }` 整个分支。

- [ ] **Step 4: 删除相关 CSS**

在 `app_v2.css` 中删除仅用于该面板的规则：`.context-panel`、`.panel-toggle`、`.context-card`、`.context-card-head`、以及 `.workspace` 里为三栏定义的 `grid-template-columns`（改为两栏，例如 `240px 1fr`）。用 grep 定位：`grep -n "context-panel\|panel-toggle\|context-card\|context-grid\|help-links" frontend/app_v2.css`。（`context-grid` 若被 `renderKeyValueGrid` 复用则保留——先 grep 确认 `context-grid` 在 JS 中仍被使用，是则保留该类。）

- [ ] **Step 5: 静态断言**

```bash
grep -c "context-panel\|toggle-panel\|panel-scene-name\|panel-opt-status" frontend/index_v2.html frontend/app_v2.js
```
Expected: 均为 `0`

- [ ] **Step 6: 冒烟验证**

加载页面，确认右侧无面板、主区域占满、控制台无报错；切换各页无异常。

- [ ] **Step 7: 递增版本号并 Commit**

递增 `?v=`；`git add -A && git commit -m "feat: remove right-side context panel and its toggle/writes/styles"`

---

### Task 4: 左侧导航重排（重命名 + 删「实例与约束」+ 重编号）

**Files:**
- Modify: `frontend/index_v2.html:57-64`（`sidebar-group` 调度流程）

**Interfaces:**
- Consumes: `data-nav` 路由键仍用现有：`new-scene` / `simulate` / `optimize-launch` / `solution-review`
- Produces: 4 步导航

- [ ] **Step 1: 替换调度流程导航（index_v2.html）**

把第 58–63 行的调度流程按钮替换为：

```html
          <div class="sidebar-title">调度流程</div>
          <button class="nav-item" type="button" data-nav="new-scene"><span class="nav-index">01</span>数据导入</button>
          <button class="nav-item requires-scene" type="button" data-nav="simulate"><span class="nav-index">02</span>仿真与洞察</button>
          <button class="nav-item requires-scene" type="button" data-nav="optimize-launch"><span class="nav-index">03</span>优化求解</button>
          <button class="nav-item requires-scene" type="button" data-nav="solution-review"><span class="nav-index">04</span>方案评审</button>
```

（删除原 `data-nav="config"` 的「实例与约束」项；`新建与导入`→`数据导入`；重编号 01–04。辅助/系统分组不动。）

- [ ] **Step 2: 静态断言**

```bash
grep -c "data-nav=\"config\"\|新建与导入" frontend/index_v2.html
```
Expected: `0`

- [ ] **Step 3: 冒烟验证**

左侧显示 4 步 + 工作概览 + 大模型连接/系统状态；逐一点击可正常切页，控制台无报错。

- [ ] **Step 4: 递增版本号并 Commit**

`git add frontend/index_v2.html && git commit -m "feat: rename to 数据导入 and collapse sidebar flow to 4 steps"`

---

### Task 5: 数据导入页（删标题块 + 删快速生成 + 迁入数据强校验）

**Files:**
- Modify: `frontend/index_v2.html:76-150`（`<section id="page-new-scene">`）
- Modify: `frontend/app_v2.js:5450-5464`（`renderCurrentPage` 增加 new-scene 分支）
- Modify: `frontend/app_v2.js:5691-5730`（`handleImportFile` 导入后不再跳转 config）

**Interfaces:**
- Consumes: `renderValidationPanel()`（app_v2.js:3754，返回校验面板 HTML）、`handleRunValidation(silent)`（5670）
- Produces: 页面新增容器 `id="new-scene-validation"`；数据导入页承载校验结果

- [ ] **Step 1: 重写 page-new-scene（index_v2.html）**

把第 76–150 行整个 `<section class="page active" id="page-new-scene">…</section>` 替换为（删标题块、删快速生成、保留导入卡片、追加校验容器）：

```html
        <section class="page active" id="page-new-scene">
          <div class="page-body">
            <article class="surface-card">
              <div class="card-head">
                <h3>导入与模板</h3>
                <p>用于接入真实业务数据、工艺结构和初始在制状态。</p>
              </div>
              <div class="upload-panel">
                <div class="upload-dropzone">
                  <strong>拖入 Excel 或点击选择文件</strong>
                  <span>支持 planning_context / orders / tasks / operations / machines / toolings / personnel / downtimes / initial_state</span>
                  <input id="import-file" type="file" accept=".xlsx,.xls,.xlsm" hidden>
                  <div class="upload-actions">
                    <button class="btn btn-primary" type="button" data-action="trigger-import">选择 Excel</button>
                    <button class="btn btn-ghost" type="button" data-action="download-template">下载模板</button>
                  </div>
                </div>
                <div class="import-progress" id="import-progress" hidden aria-live="polite">
                  <div class="import-progress-head">
                    <span class="import-spinner" aria-hidden="true"></span>
                    <strong id="import-progress-label">正在上传 Excel…</strong>
                  </div>
                  <div class="import-progress-track"><i id="import-progress-bar" style="width:0%"></i></div>
                  <p class="import-progress-note" id="import-progress-note">请勿关闭页面，导入完成后会在本页显示数据强校验结果。</p>
                </div>
              </div>
              <div class="hint-list">
                <div>模板已包含工装、人员、停机和初始在制状态字段。</div>
                <div>任务层交期可留空，默认继承订单交期。</div>
                <div>排产跨度天数由后端自动估算，并在不足时自动扩展。</div>
              </div>
            </article>
            <div id="new-scene-validation"></div>
          </div>
        </section>
```

- [ ] **Step 2: renderCurrentPage 渲染校验到数据导入页（app_v2.js）**

在 `renderCurrentPage()`（5450）中，`updateShell();` 之后加入 new-scene 分支：

```javascript
  if (app.currentPage === "new-scene") {
    const box = el("new-scene-validation");
    if (box) box.innerHTML = app.currentScene ? renderValidationPanel() : "";
    if (!app.validation && !app.validationBusy && app.currentScene) handleRunValidation(true);
  }
```

- [ ] **Step 3: 导入后停留在本页并渲染校验（app_v2.js）**

查看 `handleImportFile`（5691–5730）：把导入成功后 `navigate("config")` / 跳转「实例与约束」的逻辑改为停留在 `new-scene` 并重渲染。定位：`grep -n "config\|navigate(" app_v2.js | sed -n '/569[0-9]\|57[0-3][0-9]/p'`。将跳转目标改为：

```javascript
    if (app.currentPage !== "new-scene") await navigate("new-scene");
    else await renderCurrentPage();
```

并把 toast 文案里「请在“实例与约束”页查看明细」改为「校验结果已在本页显示」。

- [ ] **Step 4: 静态断言**

```bash
grep -c "form-generate-v2\|快速生成" frontend/index_v2.html
```
Expected: `0`
```bash
grep -c "new-scene-validation" frontend/index_v2.html frontend/app_v2.js
```
Expected: 均 `>=1`

- [ ] **Step 5: 冒烟验证**

进入数据导入页：无标题块、无快速生成表单、顶部是导入卡片；导入一个 Excel 后，**本页下方**出现数据强校验结果（不跳转）；控制台无报错。

- [ ] **Step 6: 递增版本号并 Commit**

`git add -A && git commit -m "feat: data-import page hosts validation, drop title block and quick-generate"`

---

### Task 6: 删除「实例与约束」整页

**Files:**
- Modify: `frontend/index_v2.html:220-238`（`<section id="page-config">` 删除）
- Modify: `frontend/app_v2.js`：删除 `renderConfig`(4059)、`renderConfigInstanceTab`(3808)、`renderConfigOrdersTab`(3865)、`renderConfigOperationsTab`(3916)、`renderConfigResourcesTab`(3968)、`renderConfigDowntimeTab`(4009)
- Modify: `frontend/app_v2.js:78-82`（`NAV_MAP` 删 config 系路由）
- Modify: `frontend/app_v2.js:5457-5462`（`renderCurrentPage` 删 config 分支）
- Modify: `frontend/app_v2.js:132`（删 `configTab` 状态）、`:188`（`SIDEBAR_GROUPS.config`）
- Modify: `frontend/app_v2.js:1511-1512`、`:3542`（把指向 config 的 nav-jump 改到 new-scene）

**Interfaces:**
- Consumes: 无（`renderValidationPanel` 已迁到数据导入页，保留该函数）
- Produces: 无 config 页

- [ ] **Step 1: 删除 page-config DOM（index_v2.html）**

删除第 220–238 行整个 `<section class="page" id="page-config">…</section>`。

- [ ] **Step 2: 删除 config 渲染函数（app_v2.js）**

删除 `renderConfigInstanceTab`、`renderConfigOrdersTab`、`renderConfigOperationsTab`、`renderConfigResourcesTab`、`renderConfigDowntimeTab`、`renderConfig` 六个函数整体。**注意保留 `renderValidationPanel`（3754）**——它已被数据导入页复用。

> `renderConfigInstanceTab` 内含 `${renderValidationPanel()}`；删除该函数不影响独立的 `renderValidationPanel`。

- [ ] **Step 3: 删 NAV_MAP config 路由（app_v2.js:78-82）**

删除 `config`、`instance-setup`、`order-maintenance`、`resource-maintenance`、`downtime-management` 五个键。

- [ ] **Step 4: 删 renderCurrentPage config 分支 + 状态**

删除 5457–5462 的 `if (app.currentPage === "config") {…}` 整块；删除 app 状态里 `configTab: "instance",`（132）；删除 `SIDEBAR_GROUPS.config`（188）；`groupForNav`/`expandSidebarGroup` 若因此有对 config 的引用，用 grep 清理（`grep -n "configTab\|\"config\"\|'config'" app_v2.js`）。

- [ ] **Step 5: 修复指向 config 的跳转**

- `renderDashboard` flowSteps（1511–1512）：把 `nav: "config"` 两处改为 `nav: "new-scene"`，label 相应保留（如「实例准备」「约束校验」）。
- 仿真 infeasible banner（3542）：`data-nav-jump="config"` 改为 `data-nav-jump="new-scene"`，文案「去查看校验结果」保留。
- grep 兜底：`grep -n "nav-jump=\"config\"\|navigate(\"config\")\|nav: \"config\"" app_v2.js` → 期望 0。

- [ ] **Step 6: 静态断言**

```bash
grep -c "page-config\|configTab\|renderConfig\b\|data-config-tab" frontend/index_v2.html frontend/app_v2.js
```
Expected: 均 `0`

- [ ] **Step 7: 冒烟验证**

左侧无「实例与约束」；从工作概览点「约束校验」跳到数据导入页；仿真页 infeasible 提示跳转到数据导入页；控制台无报错。

- [ ] **Step 8: 递增版本号并 Commit**

`git add -A && git commit -m "feat: delete 实例与约束 page and config renderers, repoint jumps to data-import"`

---

### Task 7: 仿真与洞察页（workflow step3）删标题块与 rail，默认图谱视图

**Files:**
- Modify: `frontend/index_v2.html:188-200`（`<section id="page-workflow">` 的 page-header 与 workflow-rail）
- Modify: `frontend/app_v2.js:3740`（`renderWorkflow` 不再调用 `renderWorkflowRail`）
- Modify: `frontend/app_v2.js:83-84`（`NAV_MAP` 的 `graph`/`simulate` 默认 focus）

**Interfaces:**
- Consumes: `renderWorkflowStep3()`（3498，已含图谱视图/完整仿真页两个 focus tab，3597–3601），`renderWorkflowStep4()`（3631）
- Produces: workflow 页无标题块/无 rail

- [ ] **Step 1: 精简 page-workflow DOM（index_v2.html）**

把第 188–200 行 `<section class="page" id="page-workflow">…</section>` 替换为（删 page-header、删 workflow-rail 容器）：

```html
        <section class="page" id="page-workflow">
          <div class="page-body">
            <div class="workflow-content" id="workflow-content"></div>
          </div>
        </section>
```

- [ ] **Step 2: renderWorkflow 停止渲染 rail 与标题（app_v2.js）**

在 `renderWorkflow()`（3720）中：删除第 3721–3739 行对 `page-header h2`/`p` 的 querySelector 与 textContent 赋值（DOM 已删，`el("page-workflow")?.querySelector` 会返回 null，虽已有 `if` 保护，但删除更干净）；删除第 3740 行 `renderWorkflowRail();` 调用。保留 `workflow-content` 渲染逻辑（3741 起）。

> `renderWorkflowRail`（3418）若无其他调用则一并删除；先 `grep -n "renderWorkflowRail" app_v2.js` 确认仅此一处调用。

- [ ] **Step 3: 确认默认图谱视图（app_v2.js:83-84）**

`NAV_MAP.simulate` 当前 `workflowFocus: "simulate"`。改为 `workflowFocus: "graph"`，使点击「仿真与洞察」默认进图谱视图：

```javascript
  graph: { page: "workflow", workflowStep: 3, workflowFocus: "graph", requiresScene: true },
  simulate: { page: "workflow", workflowStep: 3, workflowFocus: "graph", requiresScene: true },
```

（`renderWorkflowStep3` 内 `const focus = app.workflowFocus || "graph"` 已保证 tab 切换；两个 tab 功能不变。）

- [ ] **Step 4: 静态断言**

```bash
grep -c "workflow-rail\|Workflow Studio" frontend/index_v2.html
```
Expected: `0`

- [ ] **Step 5: 冒烟验证**

点「仿真与洞察」：无「工作台」标题、无 rail；默认显示**图谱视图** tab；切到**完整仿真页** tab 可运行仿真、显示甘特与明细；控制台无报错。

- [ ] **Step 6: 递增版本号并 Commit**

`git add -A && git commit -m "feat: strip workflow header/rail, default 仿真与洞察 to graph view"`

---

### Task 8: 优化求解页（workflow step4）删「多目标优化配置」以上内容

**Files:**
- Modify: `frontend/app_v2.js:3635-3647`（`renderWorkflowStep4` 顶部 decision-band）

**Interfaces:**
- Consumes: 无
- Produces: step4 从「多目标优化配置」卡片开始

- [ ] **Step 1: 删除 decision-band（app_v2.js）**

在 `renderWorkflowStep4()`（3631）的返回模板中，删除第 3636–3647 行整个 `<article class="surface-card decision-band">…</article>`（含 Optimization Control 抬头、主目标数/推荐预算/目标方案统计条）。保留其后从 `<article class="surface-card">` + `<h3>多目标优化配置</h3>`（3648）开始的所有内容。

- [ ] **Step 2: 静态断言**

```bash
grep -c "decision-band\|Optimization Control" frontend/app_v2.js
```
Expected: `0`（若 `.decision-band` CSS 无其他页面使用，可在 app_v2.css 一并删除该规则）

- [ ] **Step 3: 冒烟验证**

点「优化求解」：页面第一张卡片即「多目标优化配置」；目标勾选、预算、启动优化均正常；控制台无报错。

- [ ] **Step 4: 递增版本号并 Commit**

`git add -A && git commit -m "feat: drop decision-band above 多目标优化配置 on optimize page"`

---

### Task 9: 方案评审页删标题块

**Files:**
- Modify: `frontend/index_v2.html:202-218`（`<section id="page-review">` 的 page-header）

**Interfaces:**
- Consumes: `renderReview()`（4291，渲染到 `#review-content`）
- Produces: 无标题块的评审页

- [ ] **Step 1: 删除 page-review 的 page-header（index_v2.html）**

删除第 203–209 行 `<div class="page-header">…</div>`，保留其后的 `<div class="page-body">`（tab-strip + `#review-content`）。结果：

```html
        <section class="page" id="page-review">
          <div class="page-body">
            <div class="tab-strip">
              <button class="tab-btn" type="button" data-review-tab="library">方案库</button>
              <button class="tab-btn" type="button" data-review-tab="exact">精确冠军参考</button>
              <button class="tab-btn" type="button" data-review-tab="ai">AI 评审助手</button>
            </div>
            <div id="review-content"></div>
          </div>
        </section>
```

- [ ] **Step 2: 静态断言**

```bash
grep -c "Solution Review" frontend/index_v2.html
```
Expected: `0`

- [ ] **Step 3: 冒烟验证**

点「方案评审」：无标题块，三 tab（方案库/精确冠军参考/AI 评审）正常切换与工作；控制台无报错。

- [ ] **Step 4: 递增版本号并 Commit**

`git add frontend/index_v2.html && git commit -m "feat: remove title block on 方案评审 page"`

---

### Task 10: 删除死页 page-insights（结构/资源/瓶颈）

**Files:**
- Modify: `frontend/index_v2.html:167-186`（`<section id="page-insights">` 删除）
- Modify: `frontend/app_v2.js:72-75`（`NAV_MAP` 删 insights 系路由）
- Modify: `frontend/app_v2.js:3199`（`renderInsights` 删除）、`:5454`（renderCurrentPage insights 分支）、`:131`（`insightTab` 状态）、`:187`（`SIDEBAR_GROUPS.insights`）

**Interfaces:**
- Consumes: 无
- Produces: 无 insights 页

- [ ] **Step 1: 删除 page-insights DOM（index_v2.html）**

删除第 167–186 行整个 `<section class="page" id="page-insights">…</section>`。

- [ ] **Step 2: 删除 renderInsights 与调用（app_v2.js）**

删除 `renderInsights()`（3199，到其闭合）；删除 5454 行 `if (app.currentPage === "insights") renderInsights();`；删除 5800 行内对 `"insights"` 的判断（`grep -n '"insights"' app_v2.js` 逐个处理）。

- [ ] **Step 3: 删 NAV_MAP insights 路由 + 状态 + 分组**

删除 72–75 的 `insights`/`structure-analysis`/`resource-analysis`/`bottleneck-analysis`；删除 app 状态 `insightTab: "structure",`（131）与 `resolved.insightTab` 赋值（1446）；删除 `SIDEBAR_GROUPS.insights`（187）。

- [ ] **Step 4: 处理指向 insights 的残留跳转**

`renderWorkflowStep3` 第 3514 行 `data-nav-jump="structure-analysis"`（查看洞察）：该按钮目标页已删。删除该按钮（`<button ... 查看洞察</button>`）。grep 兜底：`grep -n "structure-analysis\|bottleneck-analysis\|resource-analysis\|nav-jump=\"insights\"" app_v2.js` → 期望 0。

- [ ] **Step 5: 静态断言**

```bash
grep -c "page-insights\|renderInsights\|insightTab\|data-insight-tab" frontend/index_v2.html frontend/app_v2.js
```
Expected: 均 `0`

- [ ] **Step 6: 冒烟验证**

全流程点一遍（数据导入→仿真与洞察→优化求解→方案评审→工作概览→系统）；无死链、无「查看洞察」跳空；控制台无报错。

- [ ] **Step 7: 递增版本号并 Commit**

`git add -A && git commit -m "chore: delete dead page-insights and its routes/renderers"`

---

### Task 11: 孤儿清理与全流程终检

**Files:**
- Modify: `frontend/app_v2.js`：删除 `refreshAll`(6259)、`toggle-help` handler(6270)、`handleGenerateInstance`(5615)、`fill-now-start`/`generate-instance` action、`renderWorkflowRail`(若 Task7 未删)
- Modify: `frontend/app_v2.css`：删除仅服务已删元素的样式

**Interfaces:**
- Consumes: 无
- Produces: 无孤儿函数/handler

- [ ] **Step 1: 定位并删除孤儿**

逐个 grep 确认无调用后删除：
```bash
grep -n "refreshAll\|toggle-help\|handleGenerateInstance\|generate-instance\|fill-now-start\|form-generate-v2\|gen-orders" frontend/app_v2.js
```
- `refreshAll`（6259）+ 其唯一调用点（6269 `refresh-all`，已随 topbar-actions 删）→ 删函数与残留 action 分支。
- `toggle-help`（6270）→ 删该 action 分支。
- `handleGenerateInstance`（5615）与 `generate-instance`/`fill-now-start` action 分支（快速生成已删）→ 删。
- 相关 `gen-*` 读取逻辑随之删除。

- [ ] **Step 2: 删除孤儿 CSS**

```bash
grep -n "decision-band\|workflow-rail\|form-generate\|topbar-actions" frontend/app_v2.css
```
删除仅用于已删元素的规则。保留仍被引用的通用类（`.status-box`、`.surface-card`、`.workflow-focus-tabs`、`.focus-tab`、`.tab-strip` 等）。

- [ ] **Step 3: 全量静态断言**

```bash
cd /Users/zhouwentao/Desktop/llm4drd && for k in "topbar-opt-status" "context-panel" "page-config" "page-insights" "form-generate-v2" "decision-band" "workflow-rail" "refresh-all" "toggle-help" "toggle-panel"; do echo "$k => $(grep -rc "$k" frontend/index_v2.html frontend/app_v2.js | paste -sd+ | bc)"; done
```
Expected: 每项 `=> 0`

- [ ] **Step 4: 全流程浏览器终检**

启动服务，用 claude-in-chrome：
1. 顶部：订单/任务令/工序/机器/工装/人员六项、无优化状态、无刷新/帮助、无右侧面板。
2. 侧栏：4 步 + 工作概览 + 大模型连接/系统状态。
3. 数据导入：导入 Excel → 本页显示强校验结果。
4. 仿真与洞察：默认图谱视图，两 tab 均正常。
5. 优化求解：首卡为多目标优化配置，可启动优化。
6. 方案评审：三 tab 正常。
7. 全程 `read_console_messages` 无 `error`。

- [ ] **Step 5: 递增版本号并 Commit**

`git add -A && git commit -m "chore: remove orphaned handlers/functions/styles after UI refactor"`

---

## Self-Review 记录

- **Spec coverage：** 顶部栏(§1→T2)、右面板(§2→T3)、导航(§3→T4)、数据导入(§4①→T5)、实例与约束删除(§4②→T6)、仿真与洞察(§4③→T7)、优化求解(§4④→T8)、方案评审(§4⑤→T9)、page-insights(§5→T10)、孤儿清理(§5→T11)、v1 删除(→T1)。全部覆盖。
- **Placeholder scan：** 无 TBD/TODO；每个改动含具体行号锚点与替换代码或精确 grep 定位。
- **Type/命名一致：** 顶部新增 id（`topbar-orders/tasks/operations/machines/toolings/personnel`）在 T2 定义并使用；`renderValidationPanel` 在 T5 复用、T6 明确保留；`workflowFocus:"graph"` 在 T7 统一。
- **风险提示：** 行号基于当前快照，逐任务提交后后续任务行号会漂移——实施时以「函数名/字符串锚点 + grep」为准，行号仅作参考。
