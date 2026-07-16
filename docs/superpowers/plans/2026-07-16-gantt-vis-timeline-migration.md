# 甘特图迁移 (vis-timeline) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 vis-timeline 替换手写的只读甘特图 `renderTimeline`，实现美观、可左右滑动/缩放、方案切换正确渲染。

**Architecture:** 复刻现有 cytoscape 的「`innerHTML` 占位 + `requestAnimationFrame` 命令式挂载」模式。`renderTimeline` 改为返回占位容器并把数据写入 `app.pendingGantts`；`mountGantts()` 销毁旧实例后按容器 id 取数据实例化 `vis.Timeline`。数据映射、遮罩、兜底基准集中在纯函数 `buildGanttData`。

**Tech Stack:** vanilla JS（无构建、无 npm），vis-timeline standalone UMD 包直引 `frontend/vendor/`，FastAPI 静态托管，`uvicorn ... --port 8888` 启动。

## Global Constraints

- 不引入构建工具 / npm / 测试框架；vis-timeline 以 UMD 脚本直引 `frontend/vendor/`（复刻 cytoscape/dagre 的引入方式）。
- 只读甘特：`editable: false`、`selectable: false`，不做拖拽编辑。
- 视觉融入现有 design-system：复用 `.status-completed/processing/future` 与 offshift/planned/unplanned 类及现有配色；轴/网格/标签走 design-system CSS 变量，跟随 light/dark。
- 静态资源引用一律带 `?v=` 版本号（与现有一致）。
- 验证以浏览器驱动真实 app（`http://localhost:8888/`，按 README 打开首页）+ 控制台检查为主；纯函数用一次性 node assert 脚本（放 scratchpad，跑完即弃）。
- 现有 3 处调用点：`app_v2.js:2811`（仿真）、`app_v2.js:3072`（方案详情）、`app_v2.js:3117`（精确参考）。

---

## 文件结构

- `frontend/vendor/vis-timeline-graph2d.min.js`（新增）— vis-timeline standalone UMD，暴露全局 `vis`
- `frontend/vendor/vis-timeline-graph2d.min.css`（新增）— vis-timeline 基础样式
- `frontend/index_v2.html`（修改）— 引入上述 vendor；bump `app_v2.js?v=`
- `frontend/app_v2.js`（修改）— 新增 `buildGanttData`、`mountGantts`；改造 `renderTimeline`；`app` 增加 `pendingGantts`/`ganttInstances`；3 处调用点触发挂载
- `frontend/app_v2.css`（修改）— vis-timeline 覆盖样式，融入 design-system

---

### Task 1: 引入 vis-timeline vendor 资源并在 HTML 中引用

**Files:**
- Create: `frontend/vendor/vis-timeline-graph2d.min.js`
- Create: `frontend/vendor/vis-timeline-graph2d.min.css`
- Modify: `frontend/index_v2.html:155-158`

**Interfaces:**
- Consumes: 无
- Produces: 全局 `window.vis`，其中 `vis.Timeline`、`vis.DataSet` 可用

- [ ] **Step 1: 下载 vis-timeline standalone UMD 包到 vendor/**

Run（若该版本 404，改用 `https://unpkg.com/vis-timeline/` 页面列出的最新 7.x 版本号）：
```bash
cd /Users/zhouwentao/Desktop/llm4drd/frontend/vendor
curl -fsSL -o vis-timeline-graph2d.min.js "https://unpkg.com/vis-timeline@7.7.3/standalone/umd/vis-timeline-graph2d.min.js"
curl -fsSL -o vis-timeline-graph2d.min.css "https://unpkg.com/vis-timeline@7.7.3/styles/vis-timeline-graph2d.min.css"
```

- [ ] **Step 2: 验证文件非空且是 UMD 包**

Run:
```bash
cd /Users/zhouwentao/Desktop/llm4drd/frontend/vendor
wc -c vis-timeline-graph2d.min.js vis-timeline-graph2d.min.css
grep -c "Timeline" vis-timeline-graph2d.min.js
```
Expected: js 文件 > 100KB，css > 5KB，`grep -c` 返回 > 0。

- [ ] **Step 3: 在 index_v2.html 引入 vendor**

在 `frontend/index_v2.html` 的 `<head>` 内、现有 `design-system.css` 引用之后加 CSS（`index_v2.html:11` 附近）：
```html
  <link rel="stylesheet" href="/static/vendor/vis-timeline-graph2d.min.css?v=7.7.3">
```
在现有 cytoscape 脚本组之后、`app_v2.js` 之前加 JS（`index_v2.html:157` 之后）：
```html
  <script src="/static/vendor/vis-timeline-graph2d.min.js?v=7.7.3"></script>
```

- [ ] **Step 4: 启动 app 并验证全局 vis 可用**

Run（后台启动，若已运行可跳过）：
```bash
cd /Users/zhouwentao/Desktop/llm4drd && uvicorn llm4drd_platform.api.server:app --reload --port 8888
```
用浏览器打开 `http://localhost:8888/`，在控制台执行：
```js
typeof vis, typeof vis.Timeline, typeof vis.DataSet
```
Expected: `"object" "function" "function"`。

- [ ] **Step 5: Commit**

```bash
git add frontend/vendor/vis-timeline-graph2d.min.js frontend/vendor/vis-timeline-graph2d.min.css frontend/index_v2.html
git commit -m "feat: vendor vis-timeline and load it in index_v2"
```

---

### Task 2: 新增 buildGanttData 纯函数（映射 + 遮罩 + 兜底基准）

**Files:**
- Modify: `frontend/app_v2.js`（在 `renderTimeline` 之前新增函数，约 `app_v2.js:1314` 之前）
- Test: `/private/tmp/claude-501/-Users-zhouwentao-Desktop-llm4drd/b7767967-8b03-4608-b911-286ea5e8ce01/scratchpad/test_build_gantt.mjs`（一次性，跑完即弃）

**Interfaces:**
- Consumes: 现有 `asArray`、`normalizeScheduleStatus`、`getMachineMap`、`buildMachineOverlays`、`offsetToDateTime`、`formatDateTime`、`tryParseDate`、`escapeHtml`（均已在 app_v2.js 定义）
- Produces: `buildGanttData(entries, options) -> { groups: Array<{id,content}>, items: Array<{id,group,start,end,content,className,title,type?}>, window: {start:string,end:string}, hasRealBase:boolean } | null`
  - `start`/`end` 为 ISO 字符串；遮罩项 `type: "background"`；`window` 为初始可视窗口（两端各留 padding）；无有效数据返回 `null`

- [ ] **Step 1: 写一次性 node 断言脚本（先跑通算法）**

Create `.../scratchpad/test_build_gantt.mjs`（独立复制纯逻辑断言，不依赖 DOM/app）：
```js
const FALLBACK_BASE = "2000-01-01T00:00:00.000Z";
const PLAN_BASE = "2024-06-01T00:00:00.000Z";
function offsetToISO(offset, base) {
  return new Date(new Date(base).getTime() + Number(offset) * 3600 * 1000).toISOString();
}
function buildGanttData(entries, { planStartAt } = {}) {
  const hasRealBase = Boolean(planStartAt);
  const base = hasRealBase ? planStartAt : FALLBACK_BASE;
  const normalized = entries
    .map((e) => ({ machineId: e.machine_id || "unknown", machineName: e.machine_name || e.machine_id || "未知资源", opId: e.op_id || "-", start: Number(e.start), end: Number(e.end), status: e.status || "future" }))
    .filter((e) => !Number.isNaN(e.start) && !Number.isNaN(e.end) && e.end > e.start);
  if (!normalized.length) return null;
  const groupsMap = new Map();
  normalized.forEach((e) => { if (!groupsMap.has(e.machineId)) groupsMap.set(e.machineId, e.machineName); });
  const groups = Array.from(groupsMap, ([id, content]) => ({ id, content }));
  const horizonStart = Math.min(...normalized.map((e) => e.start));
  const horizonEnd = Math.max(...normalized.map((e) => e.end));
  const items = normalized.map((e, i) => ({ id: `op-${i}`, group: e.machineId, start: offsetToISO(e.start, base), end: offsetToISO(e.end, base), content: e.opId, className: `status-${e.status}`, title: `${e.opId}` }));
  const padH = Math.max((horizonEnd - horizonStart) * 0.02, 1);
  return { groups, items, hasRealBase, window: { start: offsetToISO(horizonStart - padH, base), end: offsetToISO(horizonEnd + padH, base) } };
}
import assert from "node:assert";
const out = buildGanttData([
  { machine_id: "M1", machine_name: "机器A", op_id: "OP1", start: 0, end: 2, status: "completed" },
  { machine_id: "M1", op_id: "OP2", start: 2, end: 5, status: "processing" },
  { machine_id: "M2", op_id: "OP3", start: 1, end: 3, status: "future" },
  { machine_id: "M2", op_id: "BAD", start: 3, end: 3, status: "future" },
], { planStartAt: PLAN_BASE });
assert.equal(out.groups.length, 2, "两个机器组");
assert.equal(out.items.length, 3, "过滤 end<=start");
assert.equal(out.items[0].className, "status-completed");
assert.equal(out.items[0].start, "2024-06-01T00:00:00.000Z");
assert.equal(out.items[1].start, "2024-06-01T02:00:00.000Z");
const fb = buildGanttData([{ machine_id: "M1", op_id: "X", start: 0, end: 1, status: "future" }], {});
assert.equal(fb.hasRealBase, false);
assert.equal(fb.items[0].start, "2000-01-01T00:00:00.000Z");
assert.equal(buildGanttData([{ machine_id: "M1", op_id: "Z", start: 3, end: 3 }], {}), null, "无有效数据返回 null");
console.log("ALL PASS");
```

- [ ] **Step 2: 运行脚本确认失败→通过**

Run: `node "/private/tmp/claude-501/-Users-zhouwentao-Desktop-llm4drd/b7767967-8b03-4608-b911-286ea5e8ce01/scratchpad/test_build_gantt.mjs"`
Expected: 打印 `ALL PASS`（先跑通算法，再落地到 app_v2.js，保证同算法）。

- [ ] **Step 3: 在 app_v2.js 落地 buildGanttData**

先阅读 `buildMachineOverlays`（约 `app_v2.js:1260-1313`）确认其内部原始小时偏移字段。在 `renderTimeline`（`app_v2.js:1315`）之前插入：
```js
const GANTT_FALLBACK_BASE = "2000-01-01T00:00:00.000Z";

function ganttOffsetToISO(offset, base) {
  return new Date(new Date(base).getTime() + Number(offset) * 3600 * 1000).toISOString();
}

function buildGanttData(entries, options = {}) {
  const planStartAt = tryParseDate(app.instanceDetails?.plan_start_at);
  const hasRealBase = Boolean(planStartAt);
  const base = hasRealBase ? planStartAt.toISOString() : GANTT_FALLBACK_BASE;

  const normalized = asArray(entries)
    .map((item) => ({
      machineId: item.machine_id || item.machine_name || item.resource_id || "unknown",
      machineName: item.machine_name || item.machine_id || item.resource_name || "未知资源",
      opId: item.op_id || item.operation_id || item.id || "-",
      orderId: item.order_id || "-",
      taskId: item.task_id || "-",
      start: Number(item.start ?? item.start_time ?? 0),
      end: Number(item.end ?? item.end_time ?? 0),
      status: normalizeScheduleStatus(item.status),
      statusLabel: item.status_label || (normalizeScheduleStatus(item.status) === "completed" ? "已完成" : normalizeScheduleStatus(item.status) === "processing" ? "进行中" : "未来排产"),
    }))
    .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start);

  if (!normalized.length) return null;

  const groupsMap = new Map();
  normalized.forEach((item) => { if (!groupsMap.has(item.machineId)) groupsMap.set(item.machineId, item.machineName); });
  const groups = Array.from(groupsMap, ([id, content]) => ({ id, content }));

  const machineMap = getMachineMap();
  const horizonStart = Math.min(...normalized.map((i) => i.start));
  const horizonEnd = Math.max(...normalized.map((i) => i.end));

  const items = [];
  normalized.forEach((item, index) => {
    items.push({
      id: `op-${index}`,
      group: item.machineId,
      start: ganttOffsetToISO(item.start, base),
      end: ganttOffsetToISO(item.end, base),
      content: escapeHtml(item.opId),
      className: `status-${item.status}`,
      title: `${item.statusLabel} · ${item.opId}\n订单:${item.orderId} 任务:${item.taskId}\n${hasRealBase ? `${formatDateTime(ganttOffsetToISO(item.start, base))} ~ ${formatDateTime(ganttOffsetToISO(item.end, base))}` : `相对 ${item.start}h ~ ${item.end}h`}`,
    });
  });

  // 遮罩：班次外 / 停机 -> background 项
  groups.forEach((g) => {
    const overlays = buildMachineOverlays(machineMap.get(g.id), horizonStart, horizonEnd);
    overlays.forEach((ov, i) => {
      const cls = ov.className.includes("unplanned") ? "unplanned" : ov.className.includes("planned") ? "planned" : "offshift";
      items.push({
        id: `bg-${g.id}-${i}`,
        group: g.id,
        start: ganttOffsetToISO(ov.startOffset, base),
        end: ganttOffsetToISO(ov.endOffset, base),
        type: "background",
        className: cls,
      });
    });
  });

  const padH = Math.max((horizonEnd - horizonStart) * 0.02, 1);
  return {
    groups,
    items,
    hasRealBase,
    window: { start: ganttOffsetToISO(horizonStart - padH, base), end: ganttOffsetToISO(horizonEnd + padH, base) },
  };
}
```
> 注：`buildMachineOverlays` 现有返回项字段为渲染用的 `left/width` 百分比字符串。本函数需要原始小时偏移，故让 `buildMachineOverlays` 同时返回 `startOffset`/`endOffset`（其内部计算 left/width 时已有这些偏移量，直接透出即可）。若字段名不同，以实际为准并在本函数对应调整。`renderTimeline` 旧路径此时已被 Task 3 替换，不再消费 `left/width`，改动安全。

- [ ] **Step 4: 浏览器控制台冒烟验证**

打开 `http://localhost:8888/`，进入有甘特数据的页面后控制台执行：
```js
const d = buildGanttData(app.simResult.gantt, {});
console.log(d.groups.length, d.items.length, d.hasRealBase, d.window);
```
Expected: `groups`/`items` 数量合理（items ≥ 工序数），`window.start` < `window.end`，无报错。

- [ ] **Step 5: Commit**

```bash
git add frontend/app_v2.js
git commit -m "feat: add buildGanttData mapping for vis-timeline"
```

---

### Task 3: 改造 renderTimeline 为占位容器 + 写入 pendingGantts

**Files:**
- Modify: `frontend/app_v2.js:1315-1429`（`renderTimeline` 函数体）
- Modify: `frontend/app_v2.js`（`app` 状态对象定义处，新增 `pendingGantts`/`ganttInstances`）

**Interfaces:**
- Consumes: `buildGanttData`（Task 2）、现有 `renderEmptyState`、`formatInt`、`escapeHtml`
- Produces: `renderTimeline(entries, options)` 返回占位 HTML（含 `surface-card` 头部/summary/legend + `<div class="gantt-canvas" id="...">`），副作用写 `app.pendingGantts.set(id, {entries, options})`；`app.pendingGantts: Map`、`app.ganttInstances: Array`

- [ ] **Step 1: 在 app 状态对象新增字段**

用 `grep -n "^const app = {" frontend/app_v2.js` 定位全局状态定义，新增：
```js
  pendingGantts: new Map(),
  ganttInstances: [],
```

- [ ] **Step 2: 重写 renderTimeline 为占位容器**

将 `renderTimeline`（`app_v2.js:1315-1429`）整体替换为：
```js
function renderTimeline(entries, options = {}) {
  const data = buildGanttData(entries, options);
  if (!data) {
    return renderEmptyState("暂无甘特数据", "当前方案还没有可显示的资源排程。");
  }
  const id = options.canvasId || `gantt-${(options.title || "t").replace(/[^a-zA-Z0-9]/g, "").slice(0, 24)}`;
  app.pendingGantts.set(id, { entries, options });

  const statusCounts = data.items.reduce((acc, it) => {
    if (it.type === "background") return acc;
    const key = (it.className || "").replace("status-", "");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, { completed: 0, processing: 0, future: 0 });

  return `
    <div class="surface-card">
      <div class="card-head">
        <h3>${escapeHtml(options.title || "资源甘特图")}</h3>
        <p>可滚轮缩放、左右拖拽平移查看全程；条块颜色区分已完成 / 进行中 / 未来排产，斜纹遮罩显示班次外与停机占用。</p>
      </div>
      <div class="timeline-summary-strip">
        <div class="timeline-summary-card"><span>资源行数</span><strong>${formatInt(data.groups.length)}</strong></div>
        <div class="timeline-summary-card"><span>已完成 / 进行中 / 未来</span><strong>${formatInt(statusCounts.completed)} / ${formatInt(statusCounts.processing)} / ${formatInt(statusCounts.future)}</strong></div>
        <div class="timeline-summary-card"><span>时间基准</span><strong>${data.hasRealBase ? "计划起始时间" : "相对小时（无 plan_start_at）"}</strong></div>
      </div>
      <div class="legend">
        <span class="legend-item"><span class="legend-swatch status-completed"></span>已完成</span>
        <span class="legend-item"><span class="legend-swatch status-processing"></span>进行中</span>
        <span class="legend-item"><span class="legend-swatch status-future"></span>未来排产</span>
        <span class="legend-item"><span class="legend-swatch offshift"></span>班次外</span>
        <span class="legend-item"><span class="legend-swatch planned"></span>计划停机</span>
        <span class="legend-item"><span class="legend-swatch unplanned"></span>非计划停机</span>
      </div>
      <div class="gantt-canvas" id="${escapeHtml(id)}"></div>
    </div>
  `;
}
```
> 3 处调用点分别传入稳定 `canvasId`（Task 4 Step 2 处理），以区分方案。

- [ ] **Step 3: 浏览器验证占位容器与数据写入**

打开 `http://localhost:8888/` 进入仿真页，控制台执行：
```js
document.querySelectorAll(".gantt-canvas").length, app.pendingGantts.size
```
Expected: 至少 1 个 `.gantt-canvas`，`pendingGantts.size ≥ 1`（此时容器仍为空，Task 4 才挂载）。

- [ ] **Step 4: Commit**

```bash
git add frontend/app_v2.js
git commit -m "feat: renderTimeline emits gantt placeholder and queues data"
```

---

### Task 4: 新增 mountGantts 并在 3 处调用点触发挂载

**Files:**
- Modify: `frontend/app_v2.js`（新增 `mountGantts`，建议紧邻 `mountLegacyCytoscapeGraph` 即 `app_v2.js:4210` 之前）
- Modify: `frontend/app_v2.js:2811`、`:3072`、`:3117`（传入稳定 canvasId）
- Modify: `frontend/app_v2.js`（3 处外层渲染函数的 `container.innerHTML = ...` 之后触发挂载）

**Interfaces:**
- Consumes: `app.pendingGantts`、`app.ganttInstances`（Task 3）、`buildGanttData`（Task 2）、全局 `vis.Timeline`/`vis.DataSet`（Task 1）
- Produces: `mountGantts()`（销毁旧实例 + 挂载当前活动页未绑定容器）

- [ ] **Step 1: 新增 mountGantts 函数**

```js
function mountGantts() {
  if (typeof window.vis === "undefined" || typeof window.vis.Timeline !== "function") return;
  app.ganttInstances.forEach((t) => { try { t.destroy(); } catch (_) {} });
  app.ganttInstances = [];
  document.querySelectorAll(".page.active .gantt-canvas:not([data-bound='1'])").forEach((el) => {
    const payload = app.pendingGantts.get(el.id);
    if (!payload) return;
    const data = buildGanttData(payload.entries, payload.options);
    if (!data) return;
    el.dataset.bound = "1";
    const timeline = new vis.Timeline(
      el,
      new vis.DataSet(data.items),
      new vis.DataSet(data.groups),
      {
        editable: false,
        selectable: false,
        zoomable: true,
        moveable: true,
        horizontalScroll: true,
        stack: false,
        margin: { item: 6, axis: 8 },
        orientation: { axis: "top" },
        zoomMin: 1000 * 60 * 30,
        zoomMax: 1000 * 60 * 60 * 24 * 90,
        start: data.window.start,
        end: data.window.end,
        showTooltips: true,
      }
    );
    app.ganttInstances.push(timeline);
  });
}
```

- [ ] **Step 2: 给 3 处调用点传稳定 canvasId**

- `app_v2.js:2811`：`renderTimeline(app.simResult.gantt, { title: \`规则仿真甘特图 · ${app.simRule}\`, canvasId: "gantt-sim" })`
- `app_v2.js:3072`：`renderTimeline(focused.schedule, { title: \`方案详情甘特图 · ${focused.name}\`, canvasId: \`gantt-plan-${focused.id}\` })`
- `app_v2.js:3117`：`renderTimeline(app.exactReference.schedule, { title: "精确冠军参考甘特图", canvasId: "gantt-exact" })`

- [ ] **Step 3: 在 3 处外层渲染函数末尾触发挂载**

定位 `app_v2.js:2811/3072/3117` 各自所在的外层渲染函数（即执行 `container.innerHTML = ...` 的函数）。在其 `innerHTML` 赋值之后追加：
```js
requestAnimationFrame(() => mountGantts());
```
> 若某处已有 `requestAnimationFrame(() => mountLegacyCytoscapeGraph())`，在同一回调内追加 `mountGantts();` 即可，避免重复排帧。`data-bound` 守卫保证多次触发幂等。

- [ ] **Step 4: 浏览器端到端验证渲染**

打开 `http://localhost:8888/`，依次访问仿真页、方案详情页、精确参考页：
- 每处 `.gantt-canvas` 内出现 vis-timeline 画布（机器分组行 + 彩色条块）。
- 控制台 `app.ganttInstances.length` ≥ 1；无红色报错。
- 滚轮可缩放、拖拽可左右平移。

- [ ] **Step 5: Commit**

```bash
git add frontend/app_v2.js
git commit -m "feat: mount vis-timeline instances via mountGantts at 3 call sites"
```

---

### Task 5: vis-timeline 覆盖样式，融入 design-system（含 light/dark）

**Files:**
- Modify: `frontend/app_v2.css`（文件末尾新增 vis-timeline 覆盖段）

**Interfaces:**
- Consumes: 现有 `.status-completed/processing/future`、`.offshift/.planned/.unplanned` 配色（`app_v2.css:964-1005`）、design-system 变量
- Produces: 融入设计系统的甘特图外观

- [ ] **Step 1: 追加覆盖样式**

在 `frontend/app_v2.css` 末尾追加：
```css
/* —— vis-timeline 融入 design-system —— */
.gantt-canvas { width: 100%; min-height: 220px; }
.gantt-canvas .vis-timeline {
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  font-family: var(--font);
  background: var(--surface);
  overflow: hidden;
}
.gantt-canvas .vis-panel,
.gantt-canvas .vis-labelset .vis-label,
.gantt-canvas .vis-time-axis .vis-grid.vis-minor { border-color: var(--line); }
.gantt-canvas .vis-time-axis .vis-text,
.gantt-canvas .vis-labelset .vis-label { color: var(--text-soft); }
.gantt-canvas .vis-labelset .vis-label .vis-inner { font-weight: 600; color: var(--text); padding: 4px 8px; }

/* 条块：复用现有状态渐变，去掉 vis 默认边框 */
.gantt-canvas .vis-item {
  border: none;
  border-radius: var(--radius-sm);
  color: #fff;
  font-size: 11px;
  box-shadow: var(--shadow-sm);
}
.gantt-canvas .vis-item.status-completed { background: linear-gradient(135deg, #7e8da0, #566779); }
.gantt-canvas .vis-item.status-processing { background: linear-gradient(135deg, var(--accent-soft), var(--accent)); }
.gantt-canvas .vis-item.status-future { background: linear-gradient(135deg, var(--primary), #3479b4); }
.gantt-canvas .vis-item .vis-item-content { padding: 2px 6px; font-weight: 700; }

/* 背景遮罩：复用现有斜纹 */
.gantt-canvas .vis-item.vis-background.offshift { background: repeating-linear-gradient(135deg, rgba(123,142,163,0.55), rgba(123,142,163,0.55) 8px, rgba(123,142,163,0.28) 8px, rgba(123,142,163,0.28) 16px); }
.gantt-canvas .vis-item.vis-background.planned { background: repeating-linear-gradient(135deg, rgba(204,122,0,0.5), rgba(204,122,0,0.5) 8px, rgba(204,122,0,0.24) 8px, rgba(204,122,0,0.24) 16px); }
.gantt-canvas .vis-item.vis-background.unplanned { background: repeating-linear-gradient(135deg, rgba(179,58,47,0.5), rgba(179,58,47,0.5) 8px, rgba(179,58,47,0.24) 8px, rgba(179,58,47,0.24) 16px); }

/* tooltip 融入 */
.gantt-canvas .vis-tooltip {
  background: var(--surface-strong);
  color: var(--text);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-md);
  font-family: var(--font);
  white-space: pre-line;
}
.gantt-canvas .vis-current-time { background: var(--danger); }
```

- [ ] **Step 2: 浏览器目视验证 light / dark**

打开 `http://localhost:8888/` 甘特页：
- light 主题下条块/轴/网格/标签清晰，条块配色与旧甘特一致。
- 切到 dark 主题（现有主题切换），背景、文字、网格线跟随变暗，无白底刺眼块。
- 遮罩斜纹两主题下均可见。
- tooltip 悬浮显示多行内容且样式融入。

- [ ] **Step 3: Commit**

```bash
git add frontend/app_v2.css
git commit -m "style: theme vis-timeline gantt into design-system"
```

---

### Task 6: 端到端验收（方案切换 / 缩放滑动 / 兜底 / 空数据）+ bump 版本

**Files:**
- Modify: `frontend/index_v2.html`（bump `app_v2.js?v=`）

**Interfaces:**
- Consumes: 前 5 个 Task 的成果
- Produces: 通过 spec 全部验证标准的可用甘特图

- [ ] **Step 1: 方案切换无残留验证**

打开评审 / 方案库页，依次点击不同方案：
- 每次切换后甘特图更新为对应方案数据。
- 控制台 `app.ganttInstances.length` 不随切换累积增大（旧实例已 destroy）。
- 无重复 tooltip、无 `Cannot read properties` 等报错。
连续切换 5 次以上确认稳定。

- [ ] **Step 2: 缩放 / 左右滑动验证**

在任一甘特图上：滚轮放大缩小时间尺度；按住拖拽左右平移，能看到数据两端；机器名列在横向滚动时保持固定。

- [ ] **Step 3: 兜底基准与空数据验证**

- 找一个 `app.instanceDetails.plan_start_at` 为空的实例（或控制台临时 `app.instanceDetails.plan_start_at = null` 后重新渲染），确认甘特图仍按相对小时渲染、summary 显示「相对小时（无 plan_start_at）」。
- 对空排程方案，确认显示「暂无甘特数据」空状态，无报错。
> 若临时改动了 `app.instanceDetails`，刷新页面还原。

- [ ] **Step 4: bump app_v2.js 版本号**

在 `frontend/index_v2.html` 将 `app_v2.js?v=...` 版本号递增（如 `20260716-8`），确保缓存刷新。

- [ ] **Step 5: 最终提交**

```bash
git add frontend/index_v2.html
git commit -m "chore: bump app_v2.js cache-busting version for vis-timeline gantt"
```

---

## Self-Review 记录

- **Spec 覆盖：** 集成方式(T1) / 渲染架构(T3,T4) / 实例生命周期与方案切换(T4,T6.1) / 数据映射+遮罩+兜底(T2) / 样式融入 light-dark(T5) / 只读+缩放滑动交互(T4,T6.2) / 空数据(T3,T6.3) — 全部有对应任务。
- **占位符扫描：** 无 TBD/TODO；代码步骤均给出完整代码。`buildMachineOverlays` 字段名一处标注「以实际为准」，因该函数需先阅读确认，已明确处理方式而非留空。
- **类型一致：** `buildGanttData` 返回 `{groups,items,hasRealBase,window}` 或 `null`，T2 定义、T3/T4 消费一致；`pendingGantts`(Map)/`ganttInstances`(Array) T3 定义、T4 消费一致；`canvasId` T3 读取、T4 传入一致。
