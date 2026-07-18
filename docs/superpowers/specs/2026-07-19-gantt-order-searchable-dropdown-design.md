# 甘特图可搜索订单下拉框 设计

日期：2026-07-19

## 背景与问题

甘特图区域用原生 `<select>` 选择订单。当订单数量很多时，原生下拉难以定位目标订单：只能靠浏览器逐字母跳转、无法模糊筛选、也无法控制展示行数。

涉及两处订单选择框（本次改造范围）：

- 主方案/排产甘特图：`data-gantt-order-select`（`frontend/app_v2.js:1869`）
- 评审联动甘特图：`data-review-gantt-order`（`frontend/app_v2.js:2978`）

图谱页的 `data-graph-order-select` 不在本次范围内。

## 需求

1. 下拉选择订单。
2. 支持模糊筛选，**仅匹配订单号（id）**，不匹配订单名。
3. 订单可能很多：默认展示约 10 条，超出向下滚动。
4. **下拉项仅显示订单号（id），不显示订单名。**

## 关键现状（决定设计取向）

- 两处订单选项在客户端均已完整可得：
  - 主图 client 模式来自 `allEntries`，server 模式来自 `serverOrders.orders`（含 id + name）。
  - 评审图来自 `state.orders`（含 id + name）。
  - 因此模糊筛选纯客户端完成，无需服务端搜索。
- 两处选中订单都会触发整页/整块重渲染（innerHTML 字符串重建）：
  - 主图：`ganttOrderFilter[canvas] = value; renderCurrentPage()`；server 模式走 `loadPlanGantt(taskId, solutionId, value)`。
  - 评审图：`loadReviewGantt(taskId, ids, value)`。
- 该重渲染模型会销毁任何有状态的第三方组件实例。因此选择“零依赖自建组件”，而非引入 Tom Select / Choices.js 等库。
- 本设计中下拉项仅显示订单号（id）；原生 select 现有的 `id · name` 标签不再沿用。

## 方案：可复用自建 combobox

新增一个可复用渲染函数（示意名 `renderOrderCombobox`），生成与现有代码一致的 HTML 字符串，通过事件委托接线。两处订单选择框复用同一组件。

### DOM 结构

收起态是一个输入框；展开态在其下叠加下拉列表。

```
<div class="order-combobox" data-order-combobox data-canvas="{id}" data-role="gantt|review">
  <input type="search" class="order-combobox-input"
         placeholder="搜索订单号…" value="{当前选中项：订单号，或“全部订单”}" autocomplete="off">
  <ul class="order-combobox-list" hidden>
    <!-- 仅主图 allowAll 时提供 -->
    <li data-value="__all__">全部订单（N 道工序）</li>
    <li data-value="{id}">{id}</li>
    ...
  </ul>
</div>
```

- 列表容器 `max-height` 约等于 10 行高度，`overflow-y:auto`，超出滚动。
- 列表项仅显示订单号（id）；过滤依据同为 id。

### 行为

- **展开**：点击 / 聚焦输入框 → 显示列表，默认列出全部选项。
- **过滤**：输入文字 → 对订单 id 做大小写不敏感的**子串匹配**（不匹配订单名），实时过滤；`__all__` 项在有查询词时隐藏；无匹配显示“无匹配订单”占位行。
- **键盘**：↑ / ↓ 移动高亮，Enter 选中高亮项，Esc 收起。
- **鼠标**：点击列表项选中；点击组件外部收起（全局 `document` click 监听）。
- **选中**：复用现有各自的 change 逻辑——
  - 主图 client 模式：设置 `app.ganttOrderFilter[canvas]` 后 `renderCurrentPage()`；
  - 主图 server 模式：`loadPlanGantt(taskId, solutionId, value)`；
  - 评审图：`loadReviewGantt(taskId, ids, value)`。
- 选中后整页重渲染，组件回到收起态，输入框显示新选中项标签——**天然无需跨渲染保存展开态**。

### 状态管理

展开/收起、高亮项、查询词均为**单次渲染生命周期内的临时 DOM 状态**，不写入 `app`。选中即触发整页重渲染并自然重置。这正是选自建组件（而非第三方库）的关键：无实例创建/销毁的生命周期负担。

### 接线

- 在现有事件委托处新增分支：
  - 输入框 `focus` / 点击 → 展开；
  - `input` 事件 → 按 id 子串过滤重绘列表项；
  - 点击 `[data-value]` 列表项 → 收起并调用对应选中逻辑；
  - `keydown` → 键盘导航。
- 全局 `document` click 监听：点击组件外部收起所有展开的 combobox。
- 移除/替换原两处 `<select>` 的 change 处理分支（`data-gantt-order-select`、`data-review-gantt-order`），改为由 combobox 选中逻辑触发同样的调用。

### 样式

在 `frontend/app_v2.css` 新增 `.order-combobox` 相关类：输入框、下拉列表、列表项、高亮态、滚动容器、无匹配占位。沿用现有 `field-inline` 排版与下拉视觉风格。

## 验证

项目无既有前端测试框架，采用手动验证，逐场景走查：

1. 小实例（多订单）：默认展示约 10 条并可滚动；输入订单号片段可模糊筛选；输入订单名片段**不**产生匹配（确认只匹配 id）；“全部订单”可选并生效。
2. 大实例 server 模式：选中订单触发 `loadPlanGantt`，甘特图按订单重取数。
3. 评审联动甘特图：选中订单触发 `loadReviewGantt`，多方案按订单重排。
4. 键盘导航（↑/↓/Enter/Esc）与点击外部收起均正常。

## 非目标

- 不改图谱页订单过滤下拉。
- 不引入第三方前端依赖。
- 不做服务端订单搜索（客户端已有完整订单列表）。
