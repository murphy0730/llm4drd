# 图谱渲染层换回 cytoscape（手术式，仅绘图层）

> 给 Claude Code 执行的提示词。目标：把当前自绘 SVG 图谱的"画图那一块"换回 cytoscape 力导向/层次布局，其余 UI 保持现在（7/18 之后）的样子。

## 硬性约束（务必遵守）
- **禁止使用 superpowers skill**（不要调用 /superpowers 或任何会生成 design-spec/plan 文档的流程），只用原生代码编辑能力直接实现。
- 节点图标/形状/配色要与 cytoscape 原版图谱保持一致（参考 `8545cc0` 的 cytoscape 样式定义），不要采用当前 SVG 版那套字符图标。

## 背景
- 仓库：`/Users/zhouwentao/Desktop/llm4drd`
- 当前前端图谱是 7/18 提交 `cbcbf9d`（"remove cytoscape/dagre graph deps and rework frontend v2"）后重做的自绘 SVG 图谱，观感不够好。
- 漂亮的 cytoscape 版在提交 `8545cc0`（2026-07-13，"cytoscape graph visualization"）：引入了 `frontend/vendor/cytoscape.min.js`、`dagre.min.js`、`cytoscape-dagre.js`，在 `index_v2.html` 用 `<script>` 引入，在 `app_v2.js` 用 `mountLegacyCytoscapeGraph()` 挂载。
- 目标：**只把"画图那一块"换回 cytoscape 力导向/层次布局**；菜单、节点类型筛选、搜索、节点详情面板、关系解释文字全部保持现在（HEAD）的样子。

## 范围
- 改：① `frontend/vendor/` 恢复 3 个库文件；② `frontend/index_v2.html` 在 `app_v2.js` 前加 3 行 `<script>`；③ `frontend/app_v2.js` 中 `renderInteractiveGraph()`（当前约第 4117 行）里那段 `<svg>…</svg>` 绘图代码换成 cytoscape 容器 + 挂载逻辑，并复用/新增 cytoscape 挂载函数。
- 不改：评审对比、图缓存、甘特搜索、`renderGraphSection()` 的菜单/筛选/详情/关系解释部分、`buildGraphViewModel()` 的筛选逻辑（保留筛选，但忽略它算出的 SVG lane 坐标，筛选结果照常喂给 cytoscape）。

## 具体步骤
1. 恢复 vendor 文件（已删除，从 8545cc0 取回）：
   ```
   git -C /Users/zhouwentao/Desktop/llm4drd checkout 8545cc0 -- \
     frontend/vendor/cytoscape.min.js \
     frontend/vendor/dagre.min.js \
     frontend/vendor/cytoscape-dagre.js
   ```
   确认这三个文件出现在 `frontend/vendor/` 且能被 `/static/vendor/*` 访问（核对 `run_server.py` 的静态目录映射）。

2. `index_v2.html`：在 `<script src="/static/app_v2.js?v=...">` 之前插入（版本号沿用库版本，并把 `app_v2.js` 的 `?v=` 版本号 +1 避免缓存）：
   ```
   <script src="/static/vendor/cytoscape.min.js?v=3.30.4"></script>
   <script src="/static/vendor/dagre.min.js?v=0.8.5"></script>
   <script src="/static/vendor/cytoscape-dagre.js?v=2.5.0"></script>
   ```

3. `app_v2.js`：
   - 参考 `git show 8545cc0:frontend/app_v2.js` 里的 `legacyGraphDataset()`、`renderLegacyCytoscapeGraph()`、`mountLegacyCytoscapeGraph()`、`legacyGraphDetailContent()`、`focusLegacyCytoscapeNode()`，把 cytoscape 渲染逻辑重新引入当前文件（这些函数已在 cbcbf9d 被删，需要移植/重写）。
   - 在 `renderInteractiveGraph()` 中：保留 `buildGraphViewModel()` 的调用与筛选（visibleNodes/visibleEdges、selectedNode、nodeType 过滤、搜索），但**不要再用它算出的 SVG lane 坐标**。把原 `<svg>…</svg>` 整段（当前约 4147–4233 行，从 `const svg = \`` 到对应闭合 `\`;` 及在 HTML 中的引用处）替换为：
     ```
     <div class="graph-cytoscape-canvas" data-graph-canvas></div>
     ```
     并在该段渲染后 `requestAnimationFrame(() => mountInteractiveCytoscapeGraph())` 挂载 cytoscape。
   - 新建 `mountInteractiveCytoscapeGraph()`：用 `app.graphNodes`/`app.graphEdges`（经现有 `normalizeGraphNode`/`normalizeGraphEdge` 归一化）构造 cytoscape elements；layout 用 `{ name: 'dagre', rankDir: 'TB' }` 或 `'cose'`；**节点图标/形状/配色严格对齐 `8545cc0` 的 cytoscape 样式**（参考其 `mountLegacyCytoscapeGraph` 里的 `style` 数组：各 node_type 对应的 shape、background-color、label、icon/字符），不要使用当前 SVG 版的 `GRAPH_TYPE_CHARS` 字符图标体系；监听节点 `tap` 事件 → 调用现有选中逻辑（等同于 `data-action="focus-graph-node"` 的处理，更新 `app.selectedGraphNodeId` 并重渲染详情面板/关系解释）。
   - 在 `app_v2.css` 增加：`.graph-cytoscape-canvas { width:100%; height:560px; }` 及必要的背景/圆角，使其与当前设计系统观感一致。

4. 验证：
   - 起服务（`python run_server.py` 或项目现有启动命令），浏览器打开图谱视图，点"重新构建图谱"。
   - 确认：cytoscape 力导向/层次图正常渲染、可拖拽缩放；**节点图标/形状与 cytoscape 原版一致**；点节点 → 详情面板与关系解释照常更新；节点类型筛选、搜索框仍生效；菜单与评审等功能不受影响。
   - 控制台无 cytoscape 报错；三个 vendor 文件均返回 200。

## 注意
- 不要 `git revert cbcbf9d`（会丢掉 7/18 之后大量功能），只做上述局部改动。
- 改完务必更新 `index_v2.html` 里相关 `?v=` 缓存号，否则浏览器用旧缓存。
- 若 cytoscape 在大数据量下卡顿，限制初始渲染节点数（沿用现有 `GRAPH_ALL_NODE_LIMIT` 等常量）。
