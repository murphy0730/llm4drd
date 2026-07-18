# 修复提示词：仿真结果 Excel 导出无效 + 方案评审操作列下载按钮

> 你是资深全栈工程师。请在 LLM4DRD 项目（`/Users/zhouwentao/Desktop/llm4drd`）中**定位并修复**下面两个功能缺陷。前端逻辑在 `frontend/app_v2.js` 与 `frontend/app_v2.css`，后端在 `api/server.py`。修复后必须本地起服务（`python run_server.py`）实测验证，不要只改代码不验证。

---

## 0. 技术上下文（已勘查，行号可能偏移，以 grep 实际结果为准）

- 前端 API 封装：`api.exportSimExcel()`（`frontend/app_v2.js:318`）→ `api.request("/simulate/export-excel")`；
  `api.exportOptimizeSolution(taskId, solutionId)`（`:339`）→ `api.request("/optimize/hybrid/export-solution", {...})`。
- `api.request`（`:218-246`）：当 `content-type` 为 `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` / `octet-stream` / `zip` 时返回 `response.blob()`，否则按 json/text。两个导出端点后端都返回该 Excel media_type，故前端能拿到 Blob。
- `downloadBlob(blob, filename)`（`:703-712`）：标准 `URL.createObjectURL` + `<a download>.click()`，机制正常。
- 点击委托：`document.addEventListener("click", ...)`（`:5466`），`event.target.closest("[data-action]")`（`:5506`）——**全文档级委托**，任何 `data-action` 按钮都会触发 `handleAction`（`:5265`）。

---

## 1. 缺陷一：规则仿真运行完后，点「导出仿真结果 Excel」没效果 / 导不出内容

### 现象
规则仿真跑完、结果已展示，但点「导出仿真结果 Excel」按钮（该按钮在仿真结果区，渲染条件 `app.simResult && !isRunning`，见 `:5000`）后，要么完全没反应，要么下载不到内容。

### 前端链路（已确认正常，可排除）
- 按钮 `data-action="export-sim-excel"`（`:5000`）→ 委托命中 `handleAction` 的 `if (action === "export-sim-excel")`（`:5349-5355`）：`await api.exportSimExcel()` → `downloadBlob(...)` → toast 成功；异常则 `toast("导出失败：…", "error")`。
- 此链路绑定与 Blob 下载机制均正常，**不要在这块反复排查"按钮没绑定"**。

### 后端链路（重点排查）
- 导出接口 `POST /api/simulate/export-excel`（`api/server.py:2159-2176`）：
  - 若 `last_sim_payload is None` → `HTTPException(400, "尚未运行仿真…")` → 前端会弹「导出失败：尚未运行仿真」的 toast（用户可能忽略这条提示，误以为"没效果"）。
  - 否则 `data = _build_sim_export_excel(last_sim_payload, current_shop)`（`:2043`，从 `payload["gantt"]` 取数据，`:2045/2063`）。
- `last_sim_payload` 是**模块级全局变量**（`api/server.py:54`），赋值点：
  - 实时 `POST /api/simulate` 处理器末尾 `last_sim_payload = payload`（`:2867`，且 `_save_workflow_progress("simulation", payload)` 持久化）；
  - 进程启动时 `_restore_workflow_progress()` 从快照恢复（`:181`，但**实例版本对不上会被过滤掉**，`:169-171`）。
- **最可能的根因（按优先级排查）**：
  1. **前后端状态错位**：前端 `app.simResult` 是持久化/历史态（可能来自上一会话、或切换实例后前端仍显示旧结果），但后端 `last_sim_payload` 已因**进程重启 / 切换实例导致快照被过滤 / 当前 `current_shop` 与存 payload 的实例不一致**而失效或为空 → 导出 400。请确认 `app.simResult` 与后端 `last_sim_payload` 是否同源同实例。
  2. **走了不经过 `/api/simulate` 的路径**：如「参照解/启发式」走 `POST /simulate/reference-solutions`、`POST /simulate/compare`，这些路径**不会**设置 `last_sim_payload`，若用户以为那是"规则仿真"去导出，就会 400。
  3. **gantt 为空**：`_build_sim_export_excel` 仅写表头、无数据行（导出文件存在但"没内容"）。检查 `last_sim_payload["gantt"]` 是否非空。

### 修复方向（请先复现、再定方案）
- **复现**：`python run_server.py` → 生成实例 → 跑一次规则仿真（ATC 等）→ 立即点导出 → 打开浏览器 DevTools 的 Network + Console，记录：是否发请求？HTTP 状态码？响应体？前端有无 toast/报错？下载的文件是否空？
- 若确认是"状态错位 / 重启后丢失"：让导出更健壮——要么导出时直接复用**前端已持有的 `app.simResult.gantt`**（前端拼 Excel，或把 gantt 回传后端生成），要么保证 `last_sim_payload` 在每次仿真后可靠持久化并在导出前按**当前实例版本**恢复（参考 `:169-171` 的版本过滤逻辑，别让错版本快照污染）。
- 若确认是"走错路径"：在导出前给出明确提示，或让这些参照/对比路径也写入 `last_sim_payload`（注意与"当前展示的仿真"语义一致）。
- 不要在按钮绑定、Blob 下载这些已确认正常的环节浪费时间。

---

## 2. 缺陷二：方案评审「方案对比」缺少操作列 / 下载按钮不能下该方案的排产

### 现象
方案评审（库视图）的「方案对比」表，末尾应有一个「操作」列，里面要有**下载按钮**，能把**该行方案的排产方案**下载下来；但当前要么这一列看不到，要么点了下载没反应。

### 代码现状（已勘查）
- 对比表 `renderReviewCandidateComparison`（`frontend/app_v2.js:2736-2788`）：
  - 表头 `headers = ["选", "方案", ...allKeys, "操作"]`（`:2772`）——**「操作」列表头在 HEAD 是存在的**；
  - 行末 `<td class="compare-ops">` 含三个按钮（`:2765-2769`）：`focus-candidate`（查看详情）、`send-candidate-to-ai`（送入 AI）、`export-selected-solution`（导出，⬇）。
- **所以"列不见了"很可能是运行的是旧构建/被浏览器缓存**：请先确认你实际部署/打开的前端是否包含上述 HEAD 改动（`index_v2.html` 有静态版本号机制，可能未刷新；或浏览器硬缓存）。若运行版确实没有该列，按 `:2736-2788` 的结构补回「操作」列（表头 + 行末 `<td class="compare-ops">` 三按钮）即可，并保持与 `.compare-table` 样式一致。
- **更关键的真问题在下载按钮失效**：点击 `export-selected-solution` → `handleAction` 分发到 `handleExportSolution(id)`（`:5407` → `:5151`）。该函数（`:5151-5168`）有硬门槛：
  ```js
  if (!app.optimizeTaskId) {
    toast("当前导出依赖优化任务上下文，请先运行优化。", "warning");
    return;   // ← 直接返回，什么也不下
  }
  const blob = await api.exportOptimizeSolution(app.optimizeTaskId, candidate.id);
  ```
  **评审库里的基线方案、启发式参照、仿真类候选大多没有 `app.optimizeTaskId`**（那是混合优化任务的上下文），于是点下载只会弹一句警告、永远下不来。这正是用户感知到的"操作列/下载没用"。

### 修复方向（核心：让下载钮对任意候选类型都能下到该方案的排产）
1. **先确认运行版状态**：若「操作」列在 HEAD 已存在，请清理缓存/刷新到最新构建；若确缺失则按 `:2736-2788` 补回。
2. **打通下载**：改写 `handleExportSolution`（`:5151`），按候选来源路由，去掉对 `app.optimizeTaskId` 的硬依赖：
   - 候选来自混合优化任务（有 `optimizeTaskId`）：维持现有 `api.exportOptimizeSolution(taskId, id)`；
   - 候选是**基线 / 启发式参照 / 仿真类**：这些方案的排产数据应能从 `getReviewCandidates()` 返回的对象或其来源取到（前端对比表与甘特已能渲染它们，说明数据可达）。请定位每种候选类型的排产数据存放位置（例如仿真类关联 `app.simResult` / `last_sim_payload`、启发式/基线关联其各自的 schedule 结构），实现对应的下载：
     - 优先：后端新增或复用统一端点，按 `solution_id`（必要时加来源/任务标识）返回该方案的排产 Excel；
     - 或：前端直接基于候选已有的排产明细（与甘特/对比表同源的数据）用 SheetJS/`downloadBlob` 在客户端生成 `.xlsx`。
   - 无论哪种，下载文件名用该方案的**显示名**（`item.name`，即「方案一/方案二…」，不要 `S-xxx` 乱码），如 `${item.name}_排产.xlsx`。
3. **验证下载内容**：导出的 Excel 必须是该方案的**排程明细**（工序/任务/机器/起止时间等），不是空文件也不是别方案的。

---

## 3. 通用要求
- 修改集中在 `frontend/app_v2.js`（前端导出/下载逻辑、必要时 `app_v2.css` 微调「操作」列/按钮样式）与 `api/server.py`（若需新增/调整导出端点）。
- 复用既有 `downloadBlob`、`toast`、`.op-btn`、`.compare-ops` 等模式，不要另起一套。
- 不要破坏 `api.request` 的 Blob 返回约定。
- **必须本地实测**：① 跑规则仿真 → 导出 Excel → 确认下载成功且内容非空；② 方案评审库视图 → 确认「操作」列存在 → 对任意类型候选（基线、启发式、仿真、优化）点下载钮都能拿到该方案的排产 Excel。
- 交付：给出每个缺陷的**根因结论 + 改动点（文件:行号）+ 验证结果**；改动前若对方案有歧义，先简单说明再动手。

---

## 4. 附：关键文件:行号速查
- 仿真导出按钮：`frontend/app_v2.js:5000`；handler：`5349-5355`；API：`318`
- 后端导出接口：`api/server.py:2159-2176`；构造：`2043-2157`；`last_sim_payload` 赋值：`2867` / 恢复 `181`
- 方案对比表：`frontend/app_v2.js:2736-2788`（表头 `2772`、操作单元格 `2765-2769`）
- 下载 handler：`handleExportSolution` `:5151-5168`；分发 `:5407`；API `exportOptimizeSolution` `:339`
- 后端方案导出：`api/server.py:3425-3439`
- 下载工具：`downloadBlob` `:703-712`；请求封装：`api.request` `:218-246`
