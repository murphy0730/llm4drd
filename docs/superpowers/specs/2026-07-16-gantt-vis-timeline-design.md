# 甘特图迁移设计：手写 renderTimeline → vis-timeline

**日期：** 2026-07-16
**分支：** refactor/frontend-scheduling-ui
**目标：** 用 vis-timeline 替换当前手写的只读甘特图，实现更美观的视觉、可左右滑动/缩放查看，并保持点击不同方案时正确切换。

---

## 背景与现状

当前甘特图是手写实现 `renderTimeline`（`frontend/app_v2.js:1315`）：vanilla JS 字符串模板 + `div` 绝对定位 + 百分比宽度，注入 `innerHTML`。共 3 处只读调用：

- `app_v2.js:2811` — 规则仿真甘特图
- `app_v2.js:3072` — 方案详情甘特图
- `app_v2.js:3117` — 精确冠军参考甘特图

**主要痛点：**
1. 整个时间窗被压进 100% 宽度，工序多/跨度大时条块挤成一团，无法看清。
2. 不能缩放 / 左右滑动 / 平移，只有原生 `title` 提示。
3. 每次全量 `innerHTML` 重拼，大数据量渲染卡顿。

**约束与决策（已与用户确认）：**
- 只读展示，**不需要拖拽编辑**排产。
- 需求：美观、可左右滑动查看、点击不同方案切换渲染不同甘特图。
- 视觉基调：**融入现有 design-system**，不用 vis-timeline 默认样式。
- 集成方式：**UMD 直引 `vendor/`，不引入构建工具**——复刻现有 cytoscape 的加载与挂载模式，保持「script 直引」架构一致，风险最低。

## 选型结论

**vis-timeline**（MIT）。理由：其 `groups`(机器) × `items`(时间条) 数据模型与现有 `renderTimeline` 几乎一一对应；缩放/平移/tooltip 只读交互开箱即用；可作为独立 UMD 脚本引入，不改变现有架构。ECharts 仅在工序规模达上万级或需统一图表栈时才更优——当前场景 vis-timeline 更省事。

---

## 架构设计

### 1. 集成方式

- 将 vis-timeline standalone UMD 包放入 `frontend/vendor/`：
  - `vis-timeline-graph2d.min.js`
  - `vis-timeline-graph2d.min.css`
- 在 `frontend/index_v2.html` 加入对应 `<script>` / `<link>`（紧邻现有 cytoscape/dagre 引入处，带 `?v=` 版本号）。

### 2. 渲染架构（复刻 `mountLegacyCytoscapeGraph`）

现有模式：`container.innerHTML = html` 之后 `requestAnimationFrame(() => mountLegacyCytoscapeGraph())` 挂载命令式组件（`app_v2.js:2914` / `4210`）。甘特图复刻同一模式：

1. `renderTimeline(entries, options)` 不再返回完整甘特 HTML，而是返回**占位容器**：外层 `surface-card`（标题、summary strip、图例保持不变）+ 一个空的 `<div class="gantt-canvas" id="...">`。同时把本次 `{entries, options}` 存入 `app.pendingGantts`（`Map<容器id, 数据>`）。
2. 在 3 处调用点所在的外层渲染函数末尾（`container.innerHTML = ...` 之后）触发 `requestAnimationFrame(() => mountGantts())`。

容器 id 按方案标识生成，天然区分不同方案：`gantt-sim`、`gantt-plan-${focused.id}`、`gantt-exact`。

### 3. 实例生命周期与方案切换

```
function mountGantts() {
  // 1. 销毁上一轮所有旧实例（复刻 cyGraphInstance.destroy）
  app.ganttInstances.forEach(t => t.destroy());
  app.ganttInstances = [];
  // 2. 对当前活动页里每个未绑定的甘特容器实例化
  document.querySelectorAll(".page.active .gantt-canvas:not([data-bound='1'])")
    .forEach(el => {
      el.dataset.bound = "1";
      const { entries, options } = app.pendingGantts.get(el.id);
      const { items, groups, timelineOptions } = buildGanttData(entries, options);
      const timeline = new vis.Timeline(el, items, groups, timelineOptions);
      app.ganttInstances.push(timeline);
    });
}
```

- `data-bound` 幂等守卫，防重复挂载（复刻 cytoscape）。
- 挂新实例前销毁所有旧实例，避免内存泄漏 / 事件残留。
- 切换方案：新 `innerHTML` → 新容器（未绑定）→ `mountGantts` 销毁旧实例并按新方案数据全新挂载，时间窗口自动重置到新方案数据范围。

### 4. 数据映射

| 现有 `renderTimeline` | vis-timeline |
|---|---|
| 按 `machineId` 分组的机器行 | groups：`{ id: machineId, content: machineName }` |
| 工序 `{start, end, opId, status}` | items：`{ group: machineId, start, end, content: opId, className: 'status-'+status, title: tooltip }` |
| 班次外 / 停机遮罩（`buildMachineOverlays`） | items：`{ type: 'background', group, start, end, className: 'offshift'|'planned'|'unplanned' }` |

- **时间映射**：复用 `offsetToDateTime(offset)`（offset 小时 × `plan_start_at` → ISO），vis-timeline 直接解析 ISO。
- **无 `plan_start_at` 兜底**：`offsetToDateTime` 此时返回 `""`。改为使用固定兜底基准（如 `2000-01-01T00:00:00`）+ offset 小时，保证仍可按相对小时显示；轴标签在有真实基准时显示日期时间，无基准时显示相对小时。
- **空数据**：沿用 `renderEmptyState("暂无甘特数据", ...)`，在 `renderTimeline` 早返回，不进入挂载队列。

### 5. 样式（融入设计系统）

- item 通过 `className` 复用现有 `.status-completed/processing/future` 与三种遮罩斜纹类，直接沿用现有配色（completed 灰蓝 / processing 橙 / future 蓝渐变）。
- 覆盖 vis-timeline 默认 CSS：轴、网格线、分组标签、条块圆角、字体、阴影全部改用 design-system 变量（`--surface`/`--line`/`--text`/`--radius-sm`/`--font`/`--shadow-sm`），**自动跟随 light/dark**。
- 保留 `surface-card` 外壳：标题、summary strip（时间窗口 / 资源行数 / 状态计数）、图例仍由 `renderTimeline` 返回的 HTML 承载；vis-timeline 只接管中间画布区。

### 6. 交互（只读 + 可滑动）

- `editable: false`、`selectable: false`。
- `zoomable: true` + `moveable: true` + `horizontalScroll: true`：滚轮缩放时间尺度，拖拽/滚动左右平移看全程。
- `zoomMin` / `zoomMax` 限定缩放范围。
- 初始窗口 `start`/`end` = 数据范围两端各留 padding，进入即见全貌。
- tooltip 用 item `title`（vis-timeline 原生浮层），内容沿用现有「状态·工序·订单·任务·起止时间」。
- 资源行左侧固定（groups 天然左固定），横向滚动机器名不动。
- 响应式：宽度自适应 `surface-card`；高度按机器行数自动，超高时容器内纵向滚动。

---

## 涉及文件

- `frontend/vendor/vis-timeline-graph2d.min.js`（新增）
- `frontend/vendor/vis-timeline-graph2d.min.css`（新增）
- `frontend/index_v2.html`：新增 vendor 引入；bump `app_v2.js` 版本号
- `frontend/app_v2.js`：
  - 改造 `renderTimeline` → 返回占位容器 + 写入 `app.pendingGantts`
  - 新增 `buildGanttData(entries, options)`（数据映射，含遮罩、兜底基准）
  - 新增 `mountGantts()`
  - `app` 状态新增 `pendingGantts`、`ganttInstances`
  - 3 处调用点所在外层渲染函数末尾触发 `mountGantts`（复刻 cytoscape 的 `requestAnimationFrame` 触发）
- `frontend/app_v2.css`：新增 vis-timeline 覆盖样式（沿用现有状态色/遮罩类）

## 风险与权衡

- **DOM 渲染量**：vis-timeline 为 DOM 渲染，工序达上万级时性能不如 Canvas（ECharts）。当前场景规模足够；若后续超万级，再评估切 ECharts。
- **默认样式覆盖成本**：需要一组 CSS 覆盖才能融入设计系统；已在样式节明确范围，可控。
- **多处触发 mount**：3 处调用点分别触发 `mountGantts`，需确保每处外层渲染函数末尾都接上；`data-bound` 守卫保证幂等。

## 验证标准

1. 3 处甘特图（仿真 / 方案详情 / 精确参考）均正常渲染，条块着色、遮罩、summary strip、图例与现有一致。
2. 可滚轮缩放、左右拖拽平移查看完整时间范围。
3. 在方案库/评审页点击不同方案，甘特图正确切换、无旧实例残留（无重复 tooltip / 无控制台报错）。
4. light / dark 主题下样式均正确。
5. `plan_start_at` 缺失时仍能按相对小时显示，不空白。
6. 空数据时显示「暂无甘特数据」空状态。
