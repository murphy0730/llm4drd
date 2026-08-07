# LLM4DRD 前端 Design System

> **2026-08-06 更新**：本文档第 1 稿基于旧组件提取，已被重设计版取代。现行有效的设计系统为：
> - 组件库样式：`frontend/design-system.v2.css`（tokens + 全部组件 + 暗色模式）
> - 可视化预览：`frontend/design-system.html`（浏览器直接打开，或经 `/static/design-system.html`）
> - 应用侧落地：`frontend/app_v3.css` 将 JS 渲染的既有类名全部映射到 v2 视觉语言；新界面入口 `frontend/index_v3.html`
>
> 本文档保留的设计原则（密度、字重层级、数字等宽、tone 体系、渐进呈现）仍然有效；具体类名规格以 design-system.v2.css 与预览页为准。

> 文档类型：设计系统规范（tokens + 组件 + 使用规则）
> 用途：后续添加功能时复用组件的统一依据；也是样式收敛（重构方案 P3）的目标态
> 依据：`docs/frontend_ui_ux_refactor_plan.md` 设计原则；视觉语言以现有 `design-system.css` + `app_v2.css` v3 代为基准收敛
> 气质目标：**美观、简洁、紧凑**——专业工具感，信息密度高但不拥挤
> 编制日期：2026-08-06

---

## 1. 设计原则

1. **密度优先，呼吸有度**：这是专业调度工具，一屏要装下大量数据。默认采用紧凑规格（正文 12.5px、控件高 30-36px、卡片内边距 18-20px），但通过统一留白节奏和层级对比避免拥挤感。
2. **层级靠字重与灰度，不靠装饰**：信息层级用字号阶梯（4 级）、字重（550/600/650）和三级文字灰度表达；边框、阴影、色块保持克制（卡片无边框阴影或仅 1px 浅线）。
3. **数字是主角**：所有指标、ID、时间使用等宽数字字体（`--font-mono` + `tabular-nums`），保证表格与 KPI 纵向对齐、跳动时不抖动。
4. **语义色只在需要时出现**：成功/警告/错误/进行中有固定语义色与固定浅底配对，不用于装饰。中性色承担 90% 的界面。
5. **状态即信息**：每个异步对象（任务、校验、方案）的状态用同一套 tone 体系表达（见第 4 章），用户在任何页面看到同样的颜色语言。
6. **一个动作一个主按钮**：每个视图区块最多一个 `btn-primary`，其余降级为 secondary/ghost/xs，避免多主色争夺注意力。
7. **渐进呈现**：默认展示完成主任务所需的最少信息；进阶参数、技术详情、诊断明细收进折叠区（`<details>`、抽屉、二级 Tab）。

---

## 2. 设计 Tokens

> 单一来源原则：tokens 只定义在 `design-system.css` 的 `:root` 中；组件样式只允许引用变量，禁止硬编码颜色/字号/圆角。现有 `app_v2.css` 中的硬编码值（如 `.btn-primary` 的 `#2f6feb`）属待收敛项。

### 2.1 色彩

#### 中性色（界面骨架）

| Token | 浅色 | 深色 | 用途 |
| --- | --- | --- | --- |
| `--canvas` | `#fbfbfc` | `#08090a` | 页面底层背景 |
| `--surface` | `#ffffff` | `#141517` | 卡片、表格、导航等前景面 |
| `--surface-strong` | `#f7f8f8` | `#1a1b1e` | 表头、分段控件底、hover 态 |
| `--inset` | `#f4f5f6` | `#0b0c0e` | 进度条轨道、代码块底 |
| `--line` | rgba(9,10,14,.08) | rgba(255,255,255,.08) | 常规分隔线/边框 |
| `--line-strong` | rgba(9,10,14,.16) | rgba(255,255,255,.15) | 输入框边框、强调分隔 |

#### 文字（三级灰度 + 反色）

| Token | 浅色 | 深色 | 用途 |
| --- | --- | --- | --- |
| `--text` | `#0e0f11` | `#f4f5f6` | 正文、标题 |
| `--text-soft` | `#5b5f66` | `#969aa1` | 次要说明、未激活导航 |
| `--text-faint` | `#8a8f96` | `#62666d` | 标签、辅助提示、表头 |

规则：任意文字组合对比度 ≥ 4.5:1；`--text-faint` 仅用于 12px 以下辅助文字，不用于关键操作。

#### 语义色（主色 + 四态）

| 语义 | 实色 token（浅/深） | 浅底 token | 用途 |
| --- | --- | --- | --- |
| 主色 Primary | `#2f6feb` / `#4c8dff` | `--primary-soft` | 主按钮、激活态、链接、进行中 |
| 成功 Success | `#12a150` / `#3dd68c` | `--success-soft` | 完成、有效、通过 |
| 警告 Warning | `#b45309` / `#ffc53d` | `--warning-soft` | 建议处理、已过期、低利用率 |
| 错误 Danger | `#dc2626` / `#ff6369` | `--danger-soft` | 失败、阻断、超限 |
| 信息 Info | `#0e8e7f` / `#2dd4bf` | `--info-soft` | AI、中性提示 |
| 强调 Accent | `#c2620a` | `--accent-tint` | 插单等专项情景的标识色（限情景域使用） |

配对规则：语义浅底 = 实色 8-12% 透明度（浅色）/ 12-22%（深色），文字用对应实色。禁止跨语义混用（如成功文字配警告底）。

#### 品牌渐变（仅 3 处）

`linear-gradient(148deg, #2f6feb, #6d5de8)`：仅用于品牌标、进度条填充、工作台 Hero。组件内禁止新增渐变。

### 2.2 字体与字号阶梯

```css
--font: "Inter", "PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei", sans-serif;
--font-display: "Geist", "PingFang SC", sans-serif;      /* 标题 */
--font-mono: "Geist Mono", ui-monospace, Menlo, monospace; /* 数字/ID/代码 */
```

| 阶梯 | 字号 | 字重 | 用途 | 示例类 |
| --- | --- | --- | --- | --- |
| T1 页面级标题 | 20px | 620-650 | Hero、工作区头 | `.hero h3`、`.exact-workspace-head h2` |
| T2 卡片标题 | 15px | 600 | card-head | `.card-head h3` |
| T3 强调正文 | 13-14.5px | 550-600 | 小节标题、列表主行 | `.exact-step-title h3` |
| T4 正文 | 12.5px | 400-550 | 正文、按钮、表格、Tab | `.btn`、`.tabs .tab`、`td` |
| T5 辅助 | 11-11.5px | 400-600 | 说明、表单标签、行内操作 | `.subtle`、`label`、`.op-btn` |
| T6 标签 | 9.5-10.5px | 400-600 | 表头、chip、统计小字 | `th`、`.chip`、`.kpi-card span` |

KPI 大数字：20px / 600 / `--font-mono` / `tabular-nums` / `letter-spacing: -0.02em`。
表头规范：10px、大写、字距 0.04-0.05em、`--text-faint`。

### 2.3 间距（4px 基网）

| Token 建议 | 值 | 用途 |
| --- | --- | --- |
| `--space-1` | 4px | 图标与文字间距、紧凑行内 |
| `--space-2` | 8px | 控件组内间距、chip 间距 |
| `--space-3` | 12px | 卡片内小节间距、KPI 网格 gap |
| `--space-4` | 16px | 卡片间距（`.stack` gap）、表单组 |
| `--space-5` | 20px | 卡片内边距（横向） |
| `--space-6` | 24px | 页面区块间距 |
| `--space-8` | 32px | 页面主体横向留白 |

卡片内边距固定 `18px 20px`；页面主体 `padding: 6px 28px 56px`（顶 6px 因 precheck 自带 margin）。

### 2.4 圆角 / 阴影 / 动效 / 层级

| 类别 | Token | 值 | 用途 |
| --- | --- | --- | --- |
| 圆角 | `--radius-sm` | 4-6px | 按钮 xs、输入框、chip 内点 |
| | `--radius-md` | 8px | 按钮、卡片小件、Tab 容器 |
| | `--radius-lg` | 12px | 页面级卡片（surface-card、export-card） |
| | `--radius-full` | 999px | chip、lamp、进度条 |
| 阴影 | `--shadow-sm` | 0 1px 2px /5% | Tab 激活浮起、缩放控件 |
| | `--shadow-md` | 0 8px 24px /8% | toast、上传图标 |
| | `--shadow-lg` | 0 16px 44px /12% | 弹窗、下拉面板 |
| 动效 | `--ease` | cubic-bezier(0.16,1,0.3,1) | 唯一缓动曲线 |
| | 时长 | 0.13-0.16s 交互动效；0.22-0.28s 区块入场；0.35-0.4s 进度条 | |
| 层级 | z-index | 2 表头吸附；30 顶栏/下拉面板；弹窗 overlay 最高 | 不设 token，按层注释管理 |

动效规则：只动 `transform` 与 `opacity`；`prefers-reduced-motion` 全局禁用； hover 位移统一 `translateY(-1px)`。

---

## 3. 布局系统

### 3.1 应用骨架

```
┌──────────────── topbar (56px) ────────────────┐
│ brand │ scene-pill │ kpi-strip │ llm-status   │
├──────────┬─────────────────────────────────────┤
│ sidebar  │  main (独立滚动)                     │
│ 224px    │  page-body max-width 1560px 居中     │
└──────────┴─────────────────────────────────────┘
```

- `.app`：CSS Grid，`grid-template-rows: var(--topbar-h) 1fr` / `columns: var(--sidebar-w) 1fr`；折叠态 `.app.is-sidebar-collapsed` 侧栏归零。
- 页面容器三档宽度：默认 1560px（`.page-body`）、窄版 880px（`.page-body--narrow`，表单/设置类）、通栏（`.page-body--wide` / `.flush`，评审、图谱等画布类）。
- 主滚动在 `.main`，页面入场动效 `page-enter`（0.28s 上浮 5px 淡入）。

### 3.2 网格工具（已有，直接复用）

| 类 | 结构 | 用途 |
| --- | --- | --- |
| `.stack` | 单列 gap 16px | 页面区块纵向堆叠（默认） |
| `.grid-2 / .grid-3 / .grid-4` | 等分列 gap 16/12px | 详情卡、KPI 行 |
| `.kpi-grid` | 4 列 gap 12px | KPI 卡专用 |
| `.form-grid` | 2 列 | 表单（优化页内可加密到 5 列） |
| `.mt-16` | 上间距 | 区块补距 |

规则：grid 子元素必须 `min-width: 0`（宽表在列内自滚动，不撑爆轨道，已有全局兜底 app_v2.css:3318）；1200px 以下 `grid-4` 收为 2 列。

### 3.3 响应式断点（收敛后唯一口径）

| 断点 | 行为 |
| --- | --- |
| ≤1340px | 侧栏 224→210px |
| ≤1200px | grid-4→2 列；图谱详情栏转为画布下方 |
| ≤1080px | 侧栏转顶部横向滚动条；顶栏隐藏 KPI 条与 scene-pill；页边距收 18px |
| ≤720-900px | 业务组件单列（统计带、精确结果栅格等） |

新组件只需声明自己在这些断点下的栅格变化，禁止新建断点。

---

## 4. 状态语义体系（Tone System）

全应用唯一的状态语言，与重构方案的七态工作流模型对应：

| Tone | 类名 | 色 | 适用 |
| --- | --- | --- | --- |
| 成功/有效 | `.ok` / `.success` | success | 校验通过、结果有效、任务完成 |
| 警告/过期 | `.warn` / `.warning` | warning | 有警告、结果已过期、建议处理 |
| 错误/阻断 | `.err` / `.danger` | danger | 校验失败、任务失败、被阻塞 |
| 信息/引导 | `.info` | info/primary | 引导提示、AI 相关 |
| 进行中 | `.run` / `.on` / `.active` | primary | 任务运行中（配脉冲动画 `v3pulse`）、选中态 |
| 历史记录 | （灰）| text-faint | 仅历史，降透明度 + 标注 |

承载组件三件套（同一 tone 在三处形态一致）：

- **指示灯** `.lamp`（7px 圆点，侧栏/列表行内）：`.ok/.warn/.err/.run`；
- **横幅** `.precheck`（页顶通栏）：仅保留阻断与错误类提示 + 一个动作链接（`.link-btn`），成功态不渲染；
- **状态 chip** `.chip.ok/.warn/.err/.info`（表格行、卡片头内）。

规则：同一对象的状态在同一时刻只允许一处"通栏级"表达；行内/点状表达不限。新增状态场景必须从本表选 tone，不得自造颜色语义。

---

## 5. 组件库

> 以下组件均已存在并经过生产验证，类名与 `app_v2.css` v3 代一一对应。新功能**先查此库**，能组合就不新建。

### 5.1 按钮 Button

| 变体 | 类 | 规格 | 用途 |
| --- | --- | --- | --- |
| 主按钮 | `.btn .btn-primary` | 高 33px，padding 6×14，12.5px/550，radius 8 | 每区块唯一主动作 |
| 次按钮 | `.btn .btn-secondary` | 同上 + 1px 边框 | 常规动作 |
| 幽灵按钮 | `.btn .btn-ghost` | 弱边框，文字 soft | 低频/辅助动作 |
| 小按钮 | `.btn .btn-xs` | 高 27px，11.5px，radius 6 | 工具栏、画布控件 |
| 行内操作 | `.op-btn`（+`.detail/.ai/.exp`） | 11px，padding 4×9 | 表格行内多操作，hover 按语义着色 |
| 文字链接 | `.link-btn` | 下划线，继承色 | 横幅/行内跳转 |
| 危险 | `.btn-danger` | danger 文字 | 删除等（需二次确认弹窗配合） |

规则：禁用态统一 `opacity .45 + not-allowed`，不得改色；loading 态按钮禁用并替换文案（如"运行中…"），配合 spinner（见 5.8）。

### 5.2 卡片 Card

```html
<div class="surface-card">
  <div class="card-head"><div><h3>标题</h3><p>一句说明</p></div><!-- 右侧动作区 --></div>
  …内容…
</div>
```

- 规格：radius 12、padding 18×20、1px `--line`、shadow-sm（静态，不做 hover 浮起）；
- 卡片头：T2 标题 + T4 灰说明 + 右侧动作区（按钮用 `.compare-head-btn` 或 `.btn-xs` 等高对齐）；
- 变体：`is-collapsed`（整卡折叠，只留头）；左色条强调（如 `.insertion-control-strip` 的 `border-left: 3px solid`）仅用于情景域标识；
- 空态：`.renderEmptyState` 产物（图标 + 一句引导 + 可选动作），所有"无数据"场景统一使用，不自绘。

### 5.3 指标展示 KPI / KV

| 组件 | 类 | 规格 | 用途 |
| --- | --- | --- | --- |
| KPI 卡 | `.kpi-card`（入 `.kpi-grid`） | 标签 11px + 数值 20px mono + 注脚 10.5px | 页面级 3-4 个核心指标 |
| 指标条 | `.kpi-strip` 单元格 | 9.5px 标签 + 11.5px mono 值，竖线分隔 | 顶栏等横向紧凑区 |
| KV 网格 | `.kv-grid` | 3 列，10.5px 标签 + 13px mono 值 | 属性组、实例摘要 |
| 迷你 KV | `.mini-kv` | 2 列，9.5px + 12.5px mono | 侧栏/详情面板内 |
| 结果带 | `.insertion-result-band` | 4 格 1px 缝网格 + 可通栏注脚 | 任务结果关键事实 |

规则：数字一律 mono + tabular-nums；KPI 卡每视图 ≤4 张，超出改用 KV 网格。

### 5.4 表格 Table

```html
<div class="tbl-wrap"><table>
  <thead><tr><th>…</th></tr></thead>
  <tbody><tr><td>…</td></tr></tbody>
</table></div>
```

- 规格：容器 1px 边框 + radius 8 + 自滚动；表头 10px 大写、吸附（`sticky top:0`）；单元格 padding 10×13、12.5px；
- 行状态：`tr.is-selected`（primary-soft 底）、`tr.is-best-row`（success-soft 底）、`td.is-best`（单元格最优）；
- 行内差异：`.delta.good/.bad/.flat`（mono 10px，↑↓ 箭头）；
- 行内操作列：`.row-ops > .op-btn`（≤3 个，超出收进溢出菜单）；
- 宽表规则：横向滚动时首列保持语义锚点（对象名列尽量固定或加粗）；列配置用 `.col-config` 折叠面板（见 5.10）。

### 5.5 标签类 Chip / Pill / Lamp

| 组件 | 类 | 形态 | 用途 |
| --- | --- | --- | --- |
| 状态 chip | `.chip` + tone | 圆角胶囊 10.5px | 状态、来源标识（基准/参照/Pareto/精确） |
| 筛选 chip | `.fchip`（`.on/.off`） | 带色点 + 计数 | 图谱/列表的开关式筛选 |
| 实体 pill | `.pill` | 带色点，可点击 | 上下游实体跳转 |
| 分段统计 | `.review-stat-chips` | 竖线分隔组 | Tab 行旁的紧凑计数 |
| 徽标 | `.badge` | mono 胶囊 | 导航计数（如 3/4） |

规则：来源标识统一四类文案与固定 tone（基准=neutral、参照=info、Pareto=blue、精确=accent），全应用一致。

### 5.6 导航与切换

| 组件 | 类 | 用途 |
| --- | --- | --- |
| 侧栏导航 | `.nav-item`（`.active`，内含 `.lamp`/`.badge`） | 一级入口，组标题 `.nav-title` |
| 分段控件 | `.tabs > .tab` | 页内 Tab（唯一 Tab 形态，禁用旧 `.tab-strip/.tab-btn`） |
| 模式切换 | `.graph-mode > button.on` | 工具栏内二选一式切换 |
| 面包屑统计 | `.graph-breadcrumb` | 画布语境的对象 + 统计 |

### 5.7 表单 Form

- 输入件：`input/select/textarea` 高 36px、radius 6、`--line-strong` 边框；focus 态 primary 边框 + 3px 光晕；表格内编辑用紧凑款（高 33px、11.5px）；
- 布局：`.form-grid` 2 列，label 11.5px soft 上置；
- 开关：`.cold-start-control` 模式（轨道 + 滑块 + 标题/说明双行文案）——所有二元开关复用此结构；
- 选择器：单选 `.single-select-filter`、多选 `.multi-order-filter`（Excel 风格下拉，带搜索与全选）；远程搜索用 review_runtime 的 combobox 控制器（200ms 防抖、键盘导航）；
- 选择卡：`.policy-choice`（radio + 标题 + 说明，选中 inset 色条）——策略类二选一场景；
- 校验提示：权重合计条 `.exact-weight-summary` 模式（进度条 + 合计值，超限 `is-error` 变 danger）。

### 5.8 进度与运行态

| 组件 | 类 | 用途 |
| --- | --- | --- |
| 进度条 | `.bar-track > .bar-fill`（或 `.exact-progress`） | 通用百分比 |
| 进度环 | `.ring`（SVG + `.pct`） | 优化等主任务大进度 |
| 阶段轨道 | `.phase-line > .phase`（`.on/.done`） / `.exact-phase-track` | 多阶段任务步骤指示 |
| 状态卡 | `.optimize-run-status` / `.graph-build-status`（tone 变体） | 长任务运行区（标题 + 进度 + meta + 失败详情 `<details>`） |
| 日志 | `.log-box` | 深色等宽日志流（`#101216` 底） |
| 上传进度 | `.import-progress`（spinner + track + note） | 文件上传 |
| spinner | `.import-spinner` | 通用加载 |

规则：运行中组件必须给出"已耗时 + 当前阶段"；停滞检测用文字标注（"真实进度静止 x s"），不改颜色语义。

### 5.9 反馈 Toast / Modal / 横幅

- Toast：`#toast-stack` 右上角堆叠，3.2s 自消，四型 info/success/warning/error；用于轻量结果反馈，不承载必须阅读的信息；
- Modal：`.error-modal-overlay` 体系（遮罩 + radius 12 面板 + 标题 + 操作行 + 技术详情 `<details>`）；保存/列表/删除确认复用同一结构；删除必须二次确认；
- 横幅 `.precheck`：见第 4 章；
- 行内警示：`.budget-hint` 模式（accent-tint 底圆角条）用于参数建议类轻提示。

### 5.10 折叠与渐进呈现

| 模式 | 实现 | 适用 |
| --- | --- | --- |
| 详情折叠 | 原生 `<details>/<summary>` | 技术详情、诊断明细、粘贴 CSV 区 |
| 面板弹出 | `.col-config`（summary + 绝对定位面板，shadow-lg） | 列配置等局部设置 |
| 整卡折叠 | `.is-collapsed` | 利用率等可收起的大卡 |
| 抽屉（规划中） | AI 评审面板 | 右侧滑出，继承当前上下文 |

### 5.11 画布类（图谱 / 甘特）

- 工具栏 `.graph-toolbar`：左对齐控件组 + `.sep` 竖分隔 + 右侧主操作（`.graph-rebuild-btn` 模式）；
- 画布悬浮控件 `.graph-zoom-ctl`：毛玻璃圆角条，btn-xs；
- 详情侧栏 `.graph-detail-v3`：320px 固定宽、`dh` 头 + `dbody` 分节（节标题 `.dsec-t`）；
- 甘特筛选行 `.gantt-filter-row`：筛选项一排，过宽换行不裁切（下拉面板不被 overflow 裁掉）；
- 甘特分页器 `.gantt-pager`、图例行：与筛选行同高同 padding；
- 订单标识：`.scheme-dot`/`.odot` 9px 圆角色点，与甘特写色一致。

### 5.12 对话 Chat

`.chat-stream > .bubble.ai/.me`（AI 左对齐浅底、用户右对齐主色实底）；输入区 `.chat-input`（textarea + 发送）；快捷动作 `.quick-acts`（chip 按钮）。busy 态全禁用 + "AI 正在分析…"。

### 5.13 向导与步骤

- 纵向步骤卡：`.exact-step`（序号块 36px mono + 内容区，hover 浅底）——多步配置流的标准形态；
- 步骤序号 `.exact-step-no`；步骤间用 1px 分隔线，不用卡片套卡片；
- 底部动作条 `.exact-action-bar`（左说明 + 右主按钮 min-width 142px）。

### 5.14 业务专用区块（复用其结构模式）

| 模式 | 来源 | 可复用场景 |
| --- | --- | --- |
| Hero 横幅 | `.hero`（深色渐变 + 右侧 cells 统计） | 工作台顶部（每应用 ≤1 个） |
| 拖拽上传 | `.drop-hero`（虚线框 + 图标 + sheet-tags） | 任何文件导入入口 |
| 校验摘要 | `.valid-summary`（三档大色块计数） | 三分类统计摘要 |
| 导出卡 | `.export-card`（图标 + 文案 + 右侧按钮，`.highlight` 主推荐） | 交付列表项 |
| 权重行 | `.exact-weight-row`（checkbox + 名称 + 数值输入） | 带数值的多选列表 |

---

## 6. 新增功能时的使用规则

### 6.1 组件选择决策

```text
要展示什么？
├─ 状态/结论 → tone 三件套（lamp / chip / precheck），先选 tone（第 4 章）
├─ 指标数字 → KPI 卡(≤4) / KV 网格 / 结果带
├─ 对象列表 → 表格（5.4）；行内操作 ≤3 用 op-btn
├─ 大段内容块 → surface-card + stack
要用户做什么？
├─ 主任务动作 → btn-primary（区块内唯一）
├─ 配置参数 → form-grid + 输入件；二元开关 → cold-start 结构
├─ 多步流程 → exact-step 向导
├─ 异步任务 → 状态卡 + 进度组件（5.8），必须有耗时与阶段
没有现成组合？
└─ 先按 6.2 规则扩展，再考虑新建
```

### 6.2 扩展与新建规则

1. **先查库**：第 5 章已有组件必须优先复用；两个组件的间距组合不构成新组件。
2. **变体优先**：新需求优先做成现有组件的变体类（如 `.chip.xxx`、`.btn .btn-yyy`），变体只允许改 token 引用，不改结构尺寸。
3. **新建条件**（需同时满足）：跨 ≥2 个页面复用、无法用现有组件组合、有明确的独立状态集。新建组件需补入本文档第 5 章。
4. **命名**：结构类名小写连字符（`.insertion-result-band`）；状态类用 `.on/.active/.is-*` 与 tone 类 `.ok/.warn/.err/.info`；禁止语义不明的工具类泛滥（现有 `.stack/.grid-*` 已够用）。
5. **CSS 纪律**：只引用 token，不硬编码色值/圆角/字号；组件样式按"区块注释"归入 app 样式对应页面段；不动 `:root` 以外的既有变量定义；dark 模式由 token 自动适配，组件不写暗色特例。
6. **紧凑校验**：新区块自查——正文 ≥12px？控件高 27-36px？数字 mono？主按钮唯一？空态用统一组件？

### 6.3 禁用清单（历史遗留，禁止新代码使用）

| 禁用 | 替代 |
| --- | --- |
| `.tab-strip / .tab-btn` | `.tabs / .tab` |
| `.status-chip`（旧） | `.chip` + tone |
| `.workflow-overview / .readiness-panel / .executive-hero / .decision-band` | 步骤条（规划中）/ `.kpi-grid` / `.hero` |
| `.objective-pill` | `.obj-chip` |
| app_v2.css 旧 v2 代一切规则（`.topbar-status/.nav-parent/.page-header` 等） | 对应 v3 组件 |
| 硬编码 `#2f6feb` 等品牌色 | `var(--primary)` 等 token |

---

## 7. 落地说明

1. **Token 单一来源**：`design-system.css` 的 `:root` 为唯一 token 定义处；`app_v2.css` 的 `:root`（旧值：primary `#0f4c81`、topbar 78px 等）在样式收敛阶段删除——本系统所有规格以第 2 章为准，即现行 design-system.css 值。
2. **类名映射**：本文档类名与现行 v3 组件 100% 对应，新功能今天即可按此复用；待收敛的是旧代码，不是本规范。
3. **重构对齐**：步骤条、AI 抽屉、交付记录等规划中新组件落地时，按 6.2 新建条件补入第 5 章，并沿用本文档 token 与 tone 体系。
4. **验收方式**：新功能 UI 评审对照 6.2 紧凑校验清单；视觉回归以 1024/1280/1440 三宽度截图比对。
