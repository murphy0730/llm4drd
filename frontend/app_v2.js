const CONFIG = {
  API_BASE: "/api",
  HISTORY_KEY: "llm4drd_v2_scene_history",
  OPT_POLL_MS: 1500,
  TABLE_LIMIT: 40,
  HEURISTIC_RULES: ["ATC", "EDD", "SPT", "CR", "FIFO", "LPT"],
  GRAPH_FOCUS_NODE_LIMIT: 80,
  GRAPH_ALL_NODE_LIMIT: 150,
  GRAPH_ALL_EDGE_LIMIT: 320,
  // vis-timeline 在数千分组/数万条目时会锁死主线程直至页面崩溃，超限时只画最忙的前 N 台机器
  GANTT_MAX_GROUPS: 40,
  // 甘特每页机器行数（筛选/分页后单页渲染上限，沿用原 40 台的 vis 安全值）
  GANTT_PAGE_SIZE: 40,
  // 甘特图"全部订单"选项仅在条目数不超过该值时提供，大实例强制按订单聚焦
  GANTT_ALL_ORDERS_MAX_OPS: 2000,
  // 按订单层级分组时每页工序行数（行=工序，树形表头行为订单/任务令）
  GANTT_ORDER_PAGE_SIZE: 60,
};

const ORDER_SEARCH_DEBOUNCE_MS = 200;
const ORDER_SEARCH_LIMIT = 50;

const GRAPH_NODE_ORDER = ["order", "task", "operation", "machine", "tooling", "personnel"];
const GRAPH_TYPE_LABELS = {
  order: "订单",
  task: "任务",
  operation: "工序",
  machine: "机器",
  tooling: "工装",
  personnel: "人员",
  other: "其他",
};
const GRAPH_EDGE_GROUP_LABELS = {
  structure: "结构链路",
  resource: "资源可行",
  other: "其他关系",
};
// 边类型中文名：关系分解/关联边表/悬停共用，缺省回退 humanizeCodeLabel
const GRAPH_EDGE_TYPE_LABELS = {
  order_has_task: "订单包含任务",
  task_predecessor: "任务前驱",
  task_has_operation: "任务包含工序",
  operation_sequence: "工序顺序",
  op_depends_task: "工序依赖任务",
  machine_eligible: "机器可行",
  tooling_eligible: "工装可行",
  personnel_eligible: "人员可行",
};

const PRIMARY_KPI_LABELS = {
  total_tardiness: "总延误",
  makespan: "总周期",
  avg_utilization: "全周期利用率",
  critical_utilization: "关键设备全周期利用率",
  avg_active_window_utilization: "活跃窗口利用率",
  critical_active_window_utilization: "关键设备活跃窗口利用率",
  avg_net_available_utilization: "净可用利用率",
  critical_net_available_utilization: "关键设备净可用利用率",
  total_wait_time: "总等待时间",
  avg_wait_time: "平均等待时间",
  avg_flowtime: "平均流程时间",
  total_completion_time: "总完工时间",
  tooling_utilization: "工装利用率",
  personnel_utilization: "人员利用率",
  assembly_sync_penalty: "装配同步惩罚",
  tardy_job_count: "延误任务数",
  main_order_tardy_total_time: "主订单延误总时长",
  main_order_tardy_ratio: "主订单延误比例",
  bottleneck_load_balance: "瓶颈负载均衡",
};

const REVIEW_KPI_KEYS = [
  "total_tardiness",
  "makespan",
  "avg_net_available_utilization",
  "avg_active_window_utilization",
  "avg_utilization",
  "total_wait_time",
  "avg_wait_time",
  "avg_flowtime",
  "assembly_sync_penalty",
  "tardy_job_count",
  "main_order_tardy_total_time",
  "main_order_tardy_ratio",
  "tooling_utilization",
  "personnel_utilization",
];

const NAV_MAP = {
  // “当前实例”页面已移除；旧书签 hash 统一落到“数据导入”。
  "scene-library": { page: "new-scene" },
  "new-scene": { page: "new-scene" },
  dashboard: { page: "dashboard", requiresScene: true },
  "solution-review": { page: "review", reviewTab: "library", requiresScene: true },
  // 旧的通用 "workflow" 书签 hash（原 rail 导航已移除）统一落到「规则仿真」。
  workflow: { page: "workflow", workflowStep: 3, requiresScene: true },
  // 图谱视图已并入「数据导入」页，旧的 graph 书签 hash 落到该页。
  graph: { page: "new-scene" },
  simulate: { page: "workflow", workflowStep: 3, requiresScene: true },
  "optimize-config": { page: "workflow", workflowStep: 4, requiresScene: true },
  "optimize-launch": { page: "workflow", workflowStep: 4, requiresScene: true },
  "pareto-library": { page: "review", reviewTab: "library", requiresScene: true },
  "exact-reference": { page: "review", reviewTab: "exact", requiresScene: true },
  "ai-review": { page: "review", reviewTab: "ai", requiresScene: true },
  "llm-config": { page: "system", systemTab: "llm" },
  "export-data": { page: "system", systemTab: "export" },
  settings: { page: "system", systemTab: "settings" },
};

const app = {
  currentPage: "new-scene",
  currentNav: "new-scene",
  currentSceneId: null,
  currentScene: null,
  sceneHistory: [],
  instanceDetails: null,
  validation: null,
  validationBusy: false,
  validationCollapsed: false,
  importBusy: false,
  simBusy: false,
  instanceDb: null,
  downtimes: [],
  graphMeta: null,
  graphNodes: [],
  graphEdges: [],
  graphBuildTaskId: null,
  graphBuildStatus: null,
  graphBuildPollTimer: null,
  graphBuildPollFailures: 0,
  selectedGraphNodeId: null,
  selectedGraphOrderId: null,
  graphOrderOptions: [],
  graphView: defaultGraphView(),
  simResult: null,
  simStatus: null,
  simElapsedTimer: null,
  simRule: "ATC",
  referenceSolutions: [],
  // 规则参照方案的加载状态：key = 排序规则 + 目标键，防并发重复；cachedRules/missingRules
  // 供 chip 标记就绪/未计算；computing 为正在后台仿真的规则（spinner）。
  referenceSolutionsState: { key: "", loading: false, error: null, cachedRules: [], missingRules: [], computing: [] },
  optimizeTaskId: null,
  optimizeStatus: null,
  optimizeResult: null,
  exactReference: null,
  optimizeObjectiveCatalog: [],
  exactObjectiveCatalog: [],
  llmConfig: null,
  health: null,
  systemTab: "llm",
  reviewTab: "library",
  workflowStep: 1,
  filters: { orders: "", operations: "", resources: "", downtime: "" },
  reviewSelection: [],
  reviewDetailId: null,
  aiConversation: [],
  aiBusy: false,
  aiLastRecommendedId: null,
  pollTimer: null,
  optimizePollFailures: 0,
  optimizeForm: {
    objectiveKeys: ["total_tardiness", "makespan", "avg_net_available_utilization"],
    targetSolutionCount: 10,
    timeLimitS: 45,
    recommendedTimeLimitS: 45,
    timeLimitTouched: false,
    populationSize: 18,
    generations: 8,
    coarseTimeRatio: 0.7,
    refineRounds: 1,
    alnsAggression: 1.0,
    baselineRuleName: "ATC",
  },
  exactForm: {
    mode: "single",
    objectiveKey: "makespan",
    timeLimitS: 45,
    weights: {},
  },
  sidebarExpanded: {
    optimize: false,
  },
  pendingGantts: new Map(),
  ganttInstances: [],
  // 用户手动调整的可视时间范围：canvasId -> { viewKey, window }。
  // 方案、订单或分组变化时 viewKey 变化并恢复默认窗口；机器筛选和分页沿用当前范围。
  ganttViewWindows: {},
  // 甘特图订单筛选：canvasId -> 选中的订单 id（"__all__" 表示全部，仅小实例提供）
  ganttOrderFilter: {},
  // 甘特图机器筛选与分页：canvasId -> { type, downtimeOnly, query, page }
  ganttMachineFilter: {},
  // 甘特图订单内二级筛选：canvasId -> { status, query, from, to }
  ganttEntryFilter: {},
  // 甘特图分组方式：canvasId -> "order"（订单▸任务令▸工序）| "machine"（按机器资源）
  ganttGroupMode: {},
  // 方案详情甘特图的服务端取数状态：key = taskId::solutionId，切方案即自动失效
  planGantt: { key: null, taskId: null, solutionId: null, orders: [], orderId: null, entries: [], totalOperations: 0, loading: false, error: null },
  // 评审批量读取状态：选中方案、订单排程和利用率由一个 review-data 请求共同维护。
  reviewRead: emptyReviewRead(),
  orderComboboxSources: new Map(),
  orderComboboxMounts: new Map(),
};

function emptyReviewRead() {
  return {
    selectionKey: null,
    scheduleKey: null,
    orderId: null,
    schemes: {},
    utilization: null,
    failedIds: [],
    failureMessages: {},
    loading: false,
    error: null,
  };
}

function defaultGraphView() {
  return {
    mode: "focus",
    search: "",
    maxOrders: 6,
    zoom: 1,
    panX: 0,
    panY: 0,
    nodeTypes: Object.fromEntries(GRAPH_NODE_ORDER.map((type) => [type, true])),
    edgeGroups: { structure: true, resource: true, other: true },
    positions: {},
  };
}

const SIDEBAR_GROUPS = {
  optimize: ["optimize-config", "optimize-launch", "pareto-library", "exact-reference", "ai-review"],
};

const api = {
  async request(endpoint, options = {}) {
    const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, options);
    if (!response.ok) {
      const text = await response.text();
      let message = text;
      try {
        const payload = JSON.parse(text);
        if (typeof payload?.detail === "string") message = payload.detail;
        else if (Array.isArray(payload?.detail)) {
          message = payload.detail.map((item) => item.msg || JSON.stringify(item)).join("；");
        } else if (payload?.message) message = payload.message;
      } catch (_) {
        // Keep the original response text when the server did not return JSON.
      }
      // 带上状态码：调用方要能区分"后端说没有"(4xx)和"根本没拿到"(5xx/超时)。
      const error = new Error(message || `请求失败（HTTP ${response.status}）`);
      error.status = response.status;
      throw error;
    }
    const contentType = response.headers.get("content-type") || "";
    if (
      contentType.includes("application/octet-stream") ||
      contentType.includes("application/vnd.openxmlformats") ||
      contentType.includes("text/csv")
    ) {
      return response.blob();
    }
    if (contentType.includes("application/json")) return response.json();
    return response.text();
  },

  async json(endpoint, method = "GET", body = null) {
    return this.request(endpoint, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
  },

  health() { return this.json("/health"); },
  getInstanceDetails(lite = false) { return this.json(`/instance/details${lite ? "?lite=1" : ""}`); },
  getInstanceDb() { return this.json("/instance/db"); },
  importExcel(file, onProgress) {
    // 用 XHR 而非 fetch，才能拿到上传进度用于进度条展示。
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${CONFIG.API_BASE}/instance/import-excel`);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && typeof onProgress === "function") {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };
      xhr.onerror = () => reject(new Error("网络错误，文件未能上传到服务器"));
      xhr.onload = () => {
        let payload = null;
        try { payload = JSON.parse(xhr.responseText); } catch (_) { /* 非 JSON 响应按原文处理 */ }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(payload ?? xhr.responseText);
          return;
        }
        let message = xhr.responseText;
        if (typeof payload?.detail === "string") message = payload.detail;
        else if (Array.isArray(payload?.detail)) message = payload.detail.map((item) => item.msg || JSON.stringify(item)).join("；");
        reject(new Error(message || `请求失败（HTTP ${xhr.status}）`));
      };
      xhr.send(formData);
    });
  },
  validateInstance(force = false) { return this.json(`/instance/validate${force ? "?force=true" : ""}`); },
  getWorkflowProgress() { return this.json("/workflow/progress"); },
  saveReviewProgress(payload) { return this.json("/workflow/review", "PUT", payload); },
  exportValidation() { return this.request("/instance/validate/export"); },
  downloadTemplate() { return this.request("/instance/template"); },
  exportCsv() { return this.request("/instance/csv"); },
  updateOrder(id, payload) { return this.json(`/instance/order/${id}`, "PUT", payload); },
  updateTask(id, payload) { return this.json(`/instance/task/${id}`, "PUT", payload); },
  updateOperation(id, payload) { return this.json(`/instance/operation/${id}`, "PUT", payload); },
  updateMachine(id, payload) { return this.json(`/instance/machine/${id}`, "PUT", payload); },
  getDowntimes() {
    return this.json("/downtime").then((payload) => {
      if (Array.isArray(payload)) return payload;
      if (Array.isArray(payload?.downtimes)) return payload.downtimes;
      return [];
    });
  },
  addDowntime(payload) { return this.json("/downtime", "POST", payload); },
  updateDowntime(id, payload) { return this.json(`/downtime/${id}`, "PUT", payload); },
  deleteDowntime(id) { return this.request(`/downtime/${id}`, { method: "DELETE" }); },
  buildGraph() { return this.json("/graph/build", "POST"); },
  getGraphBuildStatus(taskId) { return this.json(`/graph/status/${taskId}`); },
  getGraphMeta() { return this.json("/graph/meta"); },
  getGraphNodes(limit = 60, offset = 0, nodeType = null) {
    return this.json(`/graph/nodes?limit=${limit}&offset=${offset}${nodeType ? `&node_type=${encodeURIComponent(nodeType)}` : ""}`);
  },
  getGraphEdges(limit = 80, offset = 0) { return this.json(`/graph/edges?limit=${limit}&offset=${offset}`); },
  getGraphOrder(orderId) { return this.json(`/graph/order/${encodeURIComponent(orderId)}`); },
  searchGraphOrder(query) { return this.json(`/graph/orders/search?q=${encodeURIComponent(query)}`); },
  simulate(ruleName) { return this.json("/simulate", "POST", { rule_name: ruleName }); },
  exportSimExcel() { return this.request("/simulate/export-excel"); },
  simulateReferenceSolutions(ruleNames, objectiveKeys, onlyCached = false) {
    return this.json("/simulate/reference-solutions", "POST", {
      rule_names: ruleNames,
      objective_keys: objectiveKeys,
      only_cached: onlyCached,
    });
  },
  getOptimizeObjectives() { return this.json("/optimize/objectives"); },
  startHybridOptimize(payload) { return this.json("/optimize/hybrid", "POST", payload); },
  getOptimizeStatus(taskId) { return this.json(`/optimize/hybrid/status/${taskId}`); },
  getOptimizeResult(taskId) { return this.json(`/optimize/hybrid/result/${taskId}`); },
  getOptimizeSolutionSchedule(taskId, solutionId, orderId) {
    const params = new URLSearchParams({ solution_id: solutionId });
    if (orderId) params.set("order_id", orderId);
    return this.json(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/schedule?${params.toString()}`);
  },
  getMachineTypeUtilization(taskId, solutionIds) {
    const params = new URLSearchParams({ solution_ids: asArray(solutionIds).join(",") });
    return this.json(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/machine-type-utilization?${params.toString()}`);
  },
  getReviewData(taskId, solutionIds, orderId, includeUtilization, signal) {
    const params = new URLSearchParams({
      solution_ids: ReviewRuntime.normalizeIds(solutionIds).join(","),
      include_utilization: includeUtilization ? "true" : "false",
    });
    if (orderId) params.set("order_id", orderId);
    return this.request(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/review-data?${params}`, { signal });
  },
  searchReviewOrders(taskId, solutionIds, query, signal) {
    const params = new URLSearchParams({
      solution_ids: ReviewRuntime.normalizeIds(solutionIds).join(","),
      q: query || "",
      limit: "50",
    });
    return this.request(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/review-orders?${params}`, { signal });
  },
  exportOptimizeSolution(taskId, solutionId) {
    return this.request("/optimize/hybrid/export-solution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, solution_id: solutionId }),
    });
  },
  getExactObjectives() { return this.json("/exact/objectives"); },
  createExactReference(payload) { return this.json("/optimize/exact-reference", "POST", payload); },
  getLlmConfig() { return this.json("/config/llm"); },
  setLlmConfig(payload) { return this.json("/config/llm", "PUT", payload); },
  testLlmConfig() { return this.json("/config/llm/test", "POST"); },
  aiCompare(payload) { return this.json("/ai/pareto/compare", "POST", payload); },
  aiRecommend(payload) { return this.json("/ai/pareto/recommend", "POST", payload); },
  aiAsk(payload) { return this.json("/ai/pareto/ask", "POST", payload); },
};

const reviewDataClient = ReviewRuntime.createClient({
  fetchReviewData: (args, signal) => api.getReviewData(
    args.taskId,
    args.ids,
    args.orderId,
    args.includeUtilization,
    signal,
  ),
  fetchOrders: (args, signal) => api.searchReviewOrders(
    args.taskId,
    args.ids,
    args.query,
    signal,
  ),
});

let reviewReadRequestGeneration = 0;
let pendingReviewScheduleKey = null;

function invalidateReviewReadRequest() {
  reviewReadRequestGeneration += 1;
  pendingReviewScheduleKey = null;
}

function isCurrentReviewReadRequest(generation, selectionKey, scheduleKey) {
  return (
    generation === reviewReadRequestGeneration &&
    selectionKey === app.reviewRead.selectionKey &&
    scheduleKey === pendingReviewScheduleKey
  );
}

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function orderComboboxLabel(order) {
  if (!order) return "";
  if (order.order_id === "__all__") return order.order_name || "全部订单";
  return [order.order_id, order.order_name].filter(Boolean).join(" · ");
}

function renderOrderCombobox(config) {
  app.orderComboboxSources.set(config.id, config);
  const listId = `${config.id}-list`;
  return `
    <div class="order-combobox" data-order-combobox="${escapeHtml(config.id)}">
      <input type="search" role="combobox" aria-autocomplete="list"
        aria-expanded="false" aria-controls="${escapeHtml(listId)}"
        value="${escapeHtml(orderComboboxLabel(config.selected))}" placeholder="输入订单号模糊搜索">
      <div class="order-combobox-list" id="${escapeHtml(listId)}"
        role="listbox" hidden></div>
    </div>
  `;
}

function mountOrderComboboxes() {
  app.orderComboboxMounts.forEach((controller, mountedContainer) => {
    if (mountedContainer.isConnected) return;
    controller.dispose();
    app.orderComboboxMounts.delete(mountedContainer);
  });
  document.querySelectorAll(".page.active [data-order-combobox]:not([data-order-combobox-bound='1'])").forEach((container) => {
    const source = app.orderComboboxSources.get(container.dataset.orderCombobox);
    const input = container.querySelector('[role="combobox"]');
    const list = container.querySelector('[role="listbox"]');
    if (!source || !input || !list) return;
    container.dataset.orderComboboxBound = "1";

    let controller;
    const renderState = (state) => {
      list.innerHTML = state.results.map((order, index) => {
        const optionId = `${source.id}-option-${index}`;
        const active = index === state.activeIndex;
        return `<button type="button" id="${escapeHtml(optionId)}" class="order-combobox-option${active ? " is-active" : ""}" role="option" aria-selected="${active ? "true" : "false"}" data-order-result="${index}">${escapeHtml(orderComboboxLabel(order))}</button>`;
      }).join("");
      list.hidden = !state.open;
      input.setAttribute("aria-expanded", state.open ? "true" : "false");
      if (state.open && state.activeIndex >= 0) {
        input.setAttribute("aria-activedescendant", `${source.id}-option-${state.activeIndex}`);
        list.querySelector(".is-active")?.scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
      list.querySelectorAll("[data-order-result]").forEach((option) => {
        option.addEventListener("click", () => {
          controller.choose(state.results[Number(option.dataset.orderResult)]);
        });
      });
    };

    controller = ReviewRuntime.createOrderComboboxController({
      search: source.search,
      select: async (order) => {
        input.value = orderComboboxLabel(order);
        await source.select(order);
      },
      delay: ORDER_SEARCH_DEBOUNCE_MS,
      limit: ORDER_SEARCH_LIMIT,
      onState: renderState,
    });
    app.orderComboboxMounts.set(container, controller);

    input.addEventListener("keydown", async (event) => {
      if (event.key === "Escape") {
        controller.close();
        return;
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const delta = event.key === "ArrowDown" ? 1 : -1;
        controller.move(delta);
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        await controller.chooseActive();
      }
    });

    input.addEventListener("input", () => {
      controller.input(input.value);
    });
  });
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function formatInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("zh-CN");
}

function formatDurationMs(ms) {
  const value = Number(ms) || 0;
  if (value < 1000) return `${Math.round(value)}ms`;
  return `${(value / 1000).toFixed(2)}s`;
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatDurationHours(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(digits)}h`;
}

// 渲染仿真不可行的逐工序根因明细（可展开），数据来自 /api/simulate 的 diagnosis_detail
function renderInfeasibleDetail(detail) {
  if (!detail || !Array.isArray(detail.reasons) || detail.reasons.length === 0) return "";
  const rows = detail.reasons.map((reason) => {
    const marker = reason.is_root
      ? `<span class="sim-reason-tag sim-reason-tag--root">根因</span>`
      : `<span class="sim-reason-tag sim-reason-tag--derived">级联</span>`;
    const ids = Array.isArray(reason.hint_ids) && reason.hint_ids.length
      ? `<div class="sim-reason-ids">涉及：${escapeHtml(reason.hint_ids.join("、"))}</div>`
      : "";
    const examples = Array.isArray(reason.examples) && reason.examples.length
      ? `<div class="sim-reason-examples">示例工序：${escapeHtml(reason.examples.join("，"))}</div>`
      : "";
    return `
      <li class="sim-reason ${reason.is_root ? "sim-reason--root" : "sim-reason--derived"}">
        <div class="sim-reason-head">${marker}<span class="sim-reason-label">${escapeHtml(reason.label || reason.category || "未知原因")}</span><span class="sim-reason-count">${formatInt(reason.count)} 道工序</span></div>
        ${ids}
        ${examples}
      </li>`;
  }).join("");
  const bottlenecks = Array.isArray(detail.bottlenecks) ? detail.bottlenecks : [];
  const bottleneckBlock = bottlenecks.length ? `
    <div class="sim-bottleneck-block">
      <div class="sim-bottleneck-title">瓶颈工序 TOP（其未排导致最多下游受阻，优先增加机台/班次）</div>
      <ol class="sim-bottleneck-list">
        ${bottlenecks.map((item) => `
          <li>
            <strong>${escapeHtml(item.op_name || item.op_id)}</strong>（${escapeHtml(item.op_id)}）：
            工艺 ${escapeHtml(item.process_type || "-")}，可用机器 ${formatInt(item.eligible_machine_count)} 台，
            加工 ${item.processing_time}h，阻塞下游 <strong>${formatInt(item.blocked_downstream)}</strong> 道工序
          </li>`).join("")}
      </ol>
    </div>` : "";
  return `
    <details class="sim-infeasible-detail">
      <summary>展开逐工序根因分类（${detail.reasons.length} 类，未排 ${formatInt(detail.unscheduled)} 道）</summary>
      ${bottleneckBlock}
      <ul class="sim-reason-list">${rows}</ul>
      <p class="sim-reason-tip">先处理标记为“根因”的项；“级联”类工序会在上游根因修复后自动恢复。</p>
    </details>`;
}

function formatDurationSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  if (Number(value) < 60) return `${Number(value).toFixed(1)}s`;
  return `${(Number(value) / 60).toFixed(1)}m`;
}

function tryParseDate(value) {
  if (value === null || value === undefined || value === "") return null;
  let candidate = value;
  if (typeof candidate === "string" && /^[0-9]+(\.[0-9]+)?$/.test(candidate.trim())) {
    candidate = Number(candidate);
  }
  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    if (candidate < 1e11) {
      candidate *= 1000;
    }
  }
  const date = new Date(candidate);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatDateTime(value) {
  const date = tryParseDate(value);
  if (!date) return "-";
  return date.toLocaleString("zh-CN", {
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function toDateTimeLocalValue(value) {
  const date = tryParseDate(value);
  if (!date) return "";
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hour = String(date.getHours()).padStart(2, "0");
  const minute = String(date.getMinutes()).padStart(2, "0");
  return `${year}-${month}-${day}T${hour}:${minute}`;
}

function offsetToDateTime(offset) {
  const base = tryParseDate(app.instanceDetails?.plan_start_at);
  if (!base || offset === null || offset === undefined || Number.isNaN(Number(offset))) return "";
  return new Date(base.getTime() + Number(offset) * 3600 * 1000).toISOString();
}

function offsetToDateTimeLocal(offset) {
  return toDateTimeLocalValue(offsetToDateTime(offset));
}

function formatTimelineLabel(offset) {
  return formatDateTime(offsetToDateTime(offset));
}

function normalizeScheduleStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "completed") return "completed";
  if (normalized === "processing" || normalized === "in_progress") return "processing";
  return "future";
}

function toFlexibleDateTimeLocal(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "number") return offsetToDateTimeLocal(value);
  const numeric = Number(value);
  if (!Number.isNaN(numeric) && String(value).trim() !== "" && !String(value).includes("-") && !String(value).includes(":")) {
    return offsetToDateTimeLocal(numeric);
  }
  return toDateTimeLocalValue(value);
}

function isPercentMetric(key) {
  return key.includes("utilization") || key.endsWith("_ratio");
}

function isCountMetric(key) {
  return key.endsWith("_count") || key === "completed_operations" || key === "total_operations";
}

function statusChip(label, tone = "info") {
  return `<span class="status-chip ${tone}">${escapeHtml(label)}</span>`;
}

function renderCollapseButton(action, collapsed, subject) {
  const label = `${collapsed ? "展开" : "折叠"}${subject}`;
  return `
    <button class="collapse-btn" type="button" data-action="${escapeHtml(action)}" aria-expanded="${collapsed ? "false" : "true"}" aria-label="${escapeHtml(label)}" title="${escapeHtml(label)}">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
    </button>
  `;
}

function graphBuildIsRunning() {
  return ["queued", "running"].includes(String(app.graphBuildStatus?.status || "").toLowerCase());
}

function optimizeIsRunning() {
  return ["submitting", "started", "queued", "running"].includes(String(app.optimizeStatus?.status || "").toLowerCase());
}

function optimizePhaseLabel(phase) {
  return ({
    submitting: "提交任务",
    initializing: "初始化",
    graph_context_loading: "加载图上下文",
    graph_context_building: "构建图上下文",
    coarse: "候选广搜",
    exact_promotion: "精确评估",
    elite_refine: "精英精修",
    finalize: "整理结果",
    done: "已完成",
    error: "执行失败",
    connection: "连接异常",
    submit: "提交失败",
  })[phase] || humanizeCodeLabel(phase || "initializing");
}

function optimizeProgress(status) {
  return window.OptimizeProgress.optimizeProgress(status);
}

function renderOptimizeStatus() {
  const status = app.optimizeStatus;
  if (!status) return "";
  const state = String(status.status || "running").toLowerCase();
  const failed = state === "error" || state === "failed";
  const done = state === "done" || state === "completed" || state === "success";
  const activity = window.OptimizeProgress.optimizeActivity(status);
  const tone = failed ? "danger" : done ? "success" : activity.stalled ? "warning" : "info";
  const label = failed ? "优化失败" : done ? "优化完成" : state === "submitting" ? "正在提交优化任务" : "优化正在运行";
  const progress = optimizeProgress(status);
  const message = failed
    ? (status.error || status.message || "未收到具体错误说明")
    : done
      ? (status.message || "优化完成，方案已可用于评审")
      : (activity.message || status.message || "任务已提交，正在等待优化器返回进度");
  const phaseCompleted = Number(status.phase_completed);
  const phaseTotal = Number(status.phase_total);
  const phaseWork = Number.isFinite(phaseCompleted) && Number.isFinite(phaseTotal) && phaseTotal > 0
    ? `${formatInt(phaseCompleted)} / ${formatInt(phaseTotal)}`
    : "等待工作量明细";
  return `
    <article class="optimize-run-status ${tone}" id="optimize-run-status" role="status" aria-live="polite">
      <div class="optimize-run-head">
        <div>
          <span class="eyebrow">Live optimization</span>
          <h3>${escapeHtml(label)}</h3>
        </div>
        ${statusChip(failed ? "失败" : done ? "完成" : `${progress}%`, tone)}
      </div>
      <div class="optimize-run-progress"><i style="width:${progress}%"></i></div>
      <p class="optimize-run-message">${escapeHtml(message)}</p>
      <div class="optimize-run-meta">
        <span>阶段：${escapeHtml(optimizePhaseLabel(status.phase || state))}</span>
        <span>任务 ID：${escapeHtml(app.optimizeTaskId || "待分配")}</span>
        <span>已耗时：${formatDurationSeconds(status.elapsed_s || 0)}</span>
        <span>当前代数：${formatInt(status.current_generation || 0)} / ${formatInt(status.config?.generations || app.optimizeForm.generations)}</span>
        <span>阶段工作量：${escapeHtml(phaseWork)}</span>
        <span>最近真实进度：${activity.lastRealProgressAt ? escapeHtml(formatDateTime(activity.lastRealProgressAt)) : "等待首次真实进度"}</span>
        <span>真实进度静止：${formatDurationSeconds(activity.secondsSinceRealProgress)}</span>
        <span>连接心跳：${status.updated_at || status.received_at ? escapeHtml(formatDateTime(status.updated_at || status.received_at)) : "等待首次心跳"}</span>
      </div>
      ${activity.stalled && !failed && !done ? `<div class="graph-build-warning">${escapeHtml(activity.message)}</div>` : ""}
      ${failed ? `
        <div class="optimize-error-detail">
          <strong>${escapeHtml(status.error_type ? `错误类型：${status.error_type}` : "失败原因")}</strong>
          <span>请根据上方原因检查实例数据与优化参数后重试；若仍失败，可将下方技术详情提供给开发人员。</span>
          ${status.technical_detail ? `<details><summary>查看技术详情</summary><pre>${escapeHtml(status.technical_detail)}</pre></details>` : ""}
        </div>
      ` : ""}
      ${!failed ? `
        <div class="optimize-run-stats">
          <span>近似评估 <strong>${formatInt(status.approximate_evaluations || 0)}</strong></span>
          <span>精确评估 <strong>${formatInt(status.exact_evaluations || 0)}</strong></span>
          <span>候选池 <strong>${formatInt(status.coarse_pool_size || status.archive_size || 0)}</strong></span>
          <span>可行率 <strong>${formatPercent(status.feasible_ratio || 0)}</strong></span>
        </div>
      ` : ""}
    </article>
  `;
}

function renderGraphBuildStatus() {
  const status = app.graphBuildStatus;
  if (!status) return "";
  const state = String(status.status || "queued").toLowerCase();
  const elapsed = Number(status.elapsed_s || 0);
  const estimate = status.estimate || {};
  const tone = state === "error" ? "danger" : state === "done" ? "success" : elapsed >= 30 ? "warning" : "info";
  const label = state === "error" ? "构建失败" : state === "done" ? "构建完成" : state === "queued" ? "等待执行" : "正在构建";
  const longHint = state === "running" && elapsed >= 10
    ? `<div class="graph-build-long-hint">${elapsed >= 60 ? "构建已持续超过 1 分钟。系统仍在处理，请勿重复点击或关闭服务。" : "数据量较大，构建需要一些时间；页面会持续更新进度。"}</div>`
    : "";
  return `
    <article class="graph-build-status ${tone}" id="graph-build-status-panel" role="status" aria-live="polite">
      <div class="graph-build-head">
        <div>
          <span class="eyebrow">Graph build</span>
          <h3>${escapeHtml(label)}</h3>
        </div>
        ${statusChip(`${formatInt(status.progress || 0)}%`, tone)}
      </div>
      <div class="graph-build-progress"><i style="width:${Math.max(0, Math.min(100, Number(status.progress || 0)))}%"></i></div>
      <p>${escapeHtml(status.error || status.message || "正在准备图谱构建任务")}</p>
      <div class="graph-build-meta">
        <span>阶段：${escapeHtml(humanizeCodeLabel(status.stage || "queued"))}</span>
        <span>已耗时：${formatDurationSeconds(elapsed)}</span>
        <span>超时限制：${formatDurationSeconds(status.timeout_s || 180)}</span>
        ${estimate.estimated_nodes !== undefined ? `<span>预计节点：${formatInt(estimate.estimated_nodes)}</span>` : ""}
        ${estimate.estimated_edges !== undefined ? `<span>预计边：${formatInt(estimate.estimated_edges)}</span>` : ""}
        ${estimate.machine_edges !== undefined ? `<span>机器关系：${formatInt(estimate.machine_edges)}</span>` : ""}
        ${estimate.tooling_edges !== undefined ? `<span>工装关系：${formatInt(estimate.tooling_edges)}</span>` : ""}
        ${estimate.personnel_edges !== undefined ? `<span>人员关系：${formatInt(estimate.personnel_edges)}</span>` : ""}
      </div>
      ${status.warning ? `<div class="graph-build-warning">${escapeHtml(status.warning)}</div>` : ""}
      ${longHint}
    </article>
  `;
}

function syncGraphBuildControls() {
  const running = graphBuildIsRunning();
  document.querySelectorAll('[data-action="build-graph"]').forEach((button) => {
    button.disabled = running;
    button.setAttribute("aria-busy", running ? "true" : "false");
    button.textContent = running ? `正在构建 ${formatInt(app.graphBuildStatus?.progress || 0)}%` : "构建图谱";
  });
}

function toast(message, type = "info") {
  const stack = el("toast-stack");
  if (!stack) return;
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  stack.appendChild(node);
  window.setTimeout(() => node.remove(), 3200);
}

function showErrorModal(title, message, detail = "") {
  document.querySelector(".error-modal-overlay")?.remove();
  const overlay = document.createElement("div");
  overlay.className = "error-modal-overlay";
  overlay.innerHTML = `
    <div class="error-modal" role="alertdialog" aria-modal="true" aria-labelledby="error-modal-title">
      <h3 id="error-modal-title">${escapeHtml(title)}</h3>
      <p>${escapeHtml(message || "未收到具体错误说明")}</p>
      ${detail ? `<details><summary>查看技术详情</summary><pre>${escapeHtml(detail)}</pre></details>` : ""}
      <div class="form-actions"><button class="btn btn-primary" type="button" data-modal-close>我知道了</button></div>
    </div>
  `;
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay || event.target.closest("[data-modal-close]")) overlay.remove();
  });
  document.body.appendChild(overlay);
  overlay.querySelector("[data-modal-close]")?.focus();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function loadSceneHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CONFIG.HISTORY_KEY) || "[]");
    app.sceneHistory = Array.isArray(parsed) ? parsed : [];
  } catch {
    app.sceneHistory = [];
  }
}

function saveSceneHistory() {
  localStorage.setItem(CONFIG.HISTORY_KEY, JSON.stringify(app.sceneHistory.slice(0, 12)));
}

function renderEmptyState(title, description, primaryAction = "", secondaryAction = "") {
  return `
    <div class="empty-shell">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(description)}</p>
      <div class="empty-actions">
        ${primaryAction}
        ${secondaryAction}
      </div>
    </div>
  `;
}

function syncTabButtons(attrName, activeValue) {
  document.querySelectorAll(`[${attrName}]`).forEach((node) => {
    node.classList.toggle("active", node.getAttribute(attrName) === activeValue);
  });
}

function rememberCurrentScene(sourceLabel) {
  if (!app.currentScene) return;
  const entry = {
    id: `${app.currentScene.id || "scene"}-${Date.now()}`,
    name: app.currentScene.name || "当前活动实例",
    orders: app.currentScene.summary?.orders || 0,
    operations: app.currentScene.summary?.operations || 0,
    machines: app.currentScene.summary?.machines || 0,
    toolings: app.currentScene.summary?.toolings || 0,
    planStartAt: app.instanceDetails?.plan_start_at || "",
    recordedAt: new Date().toISOString(),
    sourceLabel,
  };
  app.sceneHistory = [entry, ...app.sceneHistory.filter((item) => item.planStartAt !== entry.planStartAt)].slice(0, 12);
  saveSceneHistory();
}

function getSceneSummary() {
  return app.instanceDetails?.summary || app.currentScene?.summary || {};
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function estimateOptimizeBudgetSeconds(form = app.optimizeForm, summary = getSceneSummary()) {
  const orders = Number(summary.orders || 0);
  const tasks = Number(summary.tasks || 0);
  const operations = Number(summary.operations || 0);
  const machines = Number(summary.machines || 0);
  const toolings = Number(summary.toolings || 0);
  const personnel = Number(summary.personnel || 0);
  const inProgress = Number(summary.ops_in_progress || 0);
  const objectiveCount = Math.max(1, asArray(form.objectiveKeys).length);
  const resourceFootprint = machines + toolings + personnel;
  const structuralScale =
    operations +
    tasks * 0.45 +
    orders * 1.8 +
    resourceFootprint * 0.7 +
    inProgress * 0.9;
  const scaleSeconds = 12 + Math.pow(Math.max(structuralScale, 1) / 110, 0.88) * 10;
  const searchFactor =
    0.9 +
    (Number(form.populationSize || 0) / 18) * 0.42 +
    (Number(form.generations || 0) / 8) * 0.48 +
    Math.max(0, Number(form.refineRounds || 0) - 1) * 0.24 +
    Math.max(0, Number(form.alnsAggression || 1) - 1) * 0.18 +
    (Number(form.targetSolutionCount || 0) / 10) * 0.18 +
    objectiveCount * 0.16 +
    (1 - clamp(Number(form.coarseTimeRatio || 0.7), 0.2, 0.95)) * 0.55;
  return Math.round(clamp(scaleSeconds * searchFactor, 15, 900));
}

function refreshOptimizeBudgetRecommendation(options = {}) {
  const preserveManual = options.preserveManual !== false;
  const recommended = estimateOptimizeBudgetSeconds();
  app.optimizeForm.recommendedTimeLimitS = recommended;
  if (!preserveManual || !app.optimizeForm.timeLimitTouched || !Number(app.optimizeForm.timeLimitS)) {
    app.optimizeForm.timeLimitS = recommended;
  }
  return recommended;
}

function updateOptimizeBudgetHint() {
  const hint = el("opt-budget-hint");
  const input = el("opt-time-limit");
  if (!hint || !input) return;
  const recommended = refreshOptimizeBudgetRecommendation({ preserveManual: true });
  const manual = Number(input.value || app.optimizeForm.timeLimitS || 0);
  hint.textContent = app.optimizeForm.timeLimitTouched
    ? `建议约 ${recommended} 秒。当前保留手动值 ${manual || recommended} 秒，可随时恢复建议值。`
    : `已按当前规模与参数自动推荐 ${recommended} 秒，可继续手动修改。`;
}

function getMachineMap() {
  const map = new Map();
  asArray(app.instanceDetails?.machines).forEach((item) => map.set(item.machine_id || item.id, item));
  return map;
}

function getObjectiveLabel(key) {
  // 目录优先：名称与优化求解页（同一份后端目标目录）逐字一致，避免硬编码漂移。
  const fromResult = asArray(app.optimizeResult?.objective_catalog).find((item) => item.key === key)?.label;
  if (fromResult) return fromResult;
  const fromCatalog = asArray(app.optimizeObjectiveCatalog).find((item) => item.key === key)?.label;
  if (fromCatalog) return fromCatalog;
  return PRIMARY_KPI_LABELS[key] || key; // 目录未覆盖时的兜底
}

function metricValue(candidate, key) {
  if (!candidate) return null;
  if (candidate.objectives && candidate.objectives[key] !== undefined) return candidate.objectives[key];
  if (candidate.metrics && candidate.metrics[key] !== undefined) return candidate.metrics[key];
  if (candidate.summary && candidate.summary[key] !== undefined) return candidate.summary[key];
  if (candidate.raw?.analytics_summary && candidate.raw.analytics_summary[key] !== undefined) return candidate.raw.analytics_summary[key];
  return null;
}

function metricDisplay(candidate, key) {
  const value = metricValue(candidate, key);
  if (value === null || value === undefined) return "-";
  if (isPercentMetric(key)) return formatPercent(value);
  if (isCountMetric(key)) return formatInt(value);
  return formatDurationHours(value);
}

function objectiveShortList(keys) {
  return asArray(keys).map((key) => getObjectiveLabel(key)).join(" / ") || "-";
}

function activePrimaryObjectiveKeys() {
  return asArray(app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys).filter(Boolean);
}

function humanizeCodeLabel(value) {
  const text = String(value || "").trim();
  if (!text) return "-";
  return text.replace(/[_\-]+/g, " ").replace(/\s+/g, " ").trim();
}

function normalizeCandidate(raw, overrides = {}) {
  if (!raw) return null;
  const solutionId = raw.solution_id || raw.rule_name || raw.id || `solution-${Math.random().toString(36).slice(2, 8)}`;
  const source = overrides.source || raw.source || raw.rule_name || "candidate";
  const name = overrides.name || raw.name || raw.rule_name || raw.solution_name || raw.exact_info?.label || solutionId;
  const mergedMetrics = {
    ...(raw.metrics || {}),
    ...(raw.objectives || {}),
  };
  const mergedSummary = {
    ...(raw.analytics_summary || {}),
    ...(raw.summary || {}),
  };
  return {
    id: solutionId,
    solutionId,
    name,
    source,
    heuristicRuleName: raw.rule_name || overrides.heuristicRuleName || null,
    feasible: raw.feasible !== false,
    evaluationMode: raw.evaluation_mode || "exact",
    objectives: raw.objectives || {},
    metrics: mergedMetrics,
    summary: mergedSummary,
    schedule: asArray(raw.schedule),
    deltaVsBaseline: raw.delta_vs_baseline || {},
    candidate: raw.candidate || {},
    raw,
  };
}

function getReviewCandidates() {
  const items = [];
  if (app.optimizeResult?.baseline) {
    items.push(normalizeCandidate(app.optimizeResult.baseline, {
      source: "baseline",
      name: `基线方案 · ${app.optimizeResult.baseline.rule_name || "ATC"}`,
    }));
  }
  asArray(app.optimizeResult?.solutions).forEach((item, index) => {
    items.push(normalizeCandidate(item, {
      source: item.source || "pareto",
      name: `方案${index + 1}`,
    }));
  });
  asArray(app.optimizeResult?.reference_solutions).forEach((item, index) => {
    items.push(normalizeCandidate(item, {
      source: item.source || "reference",
      name: item.rule_name ? `启发式参考 · ${item.rule_name}` : `参照方案${index + 1}`,
    }));
  });
  asArray(app.referenceSolutions).forEach((item, index) => {
    items.push(normalizeCandidate(item, {
      source: item.source || "heuristic",
      name: item.rule_name ? `启发式参考 · ${item.rule_name}` : `参照方案${index + 1}`,
    }));
  });
  if (app.exactReference) {
    items.push(normalizeCandidate(app.exactReference, {
      source: "exact_reference",
      name: app.exactReference.exact_info?.label || app.exactReference.solution_id || "精确冠军参考",
    }));
  }
  const uniq = new Map();
  items.filter(Boolean).forEach((item) => uniq.set(item.id, item));
  return Array.from(uniq.values());
}

function ensureReviewSelection() {
  const candidates = getReviewCandidates();
  const ids = new Set(candidates.map((item) => item.id));
  app.reviewSelection = app.reviewSelection.filter((id) => ids.has(id));
  if (!app.reviewSelection.length && candidates.length) {
    app.reviewSelection = candidates.slice(0, Math.min(3, candidates.length)).map((item) => item.id);
  }
  if (!app.reviewDetailId && candidates.length) {
    app.reviewDetailId = app.reviewSelection[0] || candidates[0].id;
  }
}

function getSelectedReviewCandidates() {
  const map = new Map(getReviewCandidates().map((item) => [item.id, item]));
  return app.reviewSelection.map((id) => map.get(id)).filter(Boolean);
}

function getSelectedReviewCandidate() {
  const candidates = getReviewCandidates();
  const map = new Map(candidates.map((item) => [item.id, item]));
  return map.get(app.reviewDetailId) || map.get(app.aiLastRecommendedId) || map.get(app.reviewSelection[0]) || candidates[0] || null;
}

// 勾选集稳定色板（共享上限 4）：按方案在勾选集合中的位置分配 --primary/--accent/--success/--info，
// 同一色贯穿勾选态、对比表高亮、利用率列头、甘特条块，形成视觉闭环。
const SCHEME_COLOR_TOKENS = ["var(--primary)", "var(--accent)", "var(--success)", "var(--info)"];
function schemeColorIndex(id) {
  const idx = app.reviewSelection.indexOf(id);
  return idx < 0 ? -1 : idx % SCHEME_COLOR_TOKENS.length;
}
function schemeColorToken(index) {
  return SCHEME_COLOR_TOKENS[((index % SCHEME_COLOR_TOKENS.length) + SCHEME_COLOR_TOKENS.length) % SCHEME_COLOR_TOKENS.length];
}

const CANDIDATE_SOURCE_LABEL = {
  baseline: "基线",
  pareto: "Pareto 优化",
  reference: "启发式参照",
  heuristic: "启发式参照",
  exact_reference: "精确冠军",
};
function candidateSourceLabel(item) {
  return CANDIDATE_SOURCE_LABEL[String(item?.source || "")] || humanizeCodeLabel(item?.source || "候选");
}
function candidateModeLabel(item) {
  const mode = String(item?.evaluationMode || "").toLowerCase();
  if (mode === "exact") return "精确评估";
  if (mode === "simulation" || mode === "simulated") return "仿真评估";
  return item?.evaluationMode ? humanizeCodeLabel(item.evaluationMode) : "评估";
}

function renderPrimaryObjectiveBadges(keys = activePrimaryObjectiveKeys()) {
  const items = asArray(keys).filter(Boolean);
  if (!items.length) return "";
  return `
    <div class="tag-row">
      ${items.map((key) => `<span class="tag">${escapeHtml(getObjectiveLabel(key))}</span>`).join("")}
    </div>
  `;
}

function renderCandidateMetricMatrix(candidates, title = "主目标 + 全量 KPI 对比") {
  const items = asArray(candidates).filter(Boolean);
  if (!items.length) return "";
  const primaryKeys = activePrimaryObjectiveKeys();
  const extraKeys = REVIEW_KPI_KEYS.filter((key) => !primaryKeys.includes(key));
  return `
    <article class="surface-card">
      <div class="card-head">
        <h3>${escapeHtml(title)}</h3>
        <p>先看本次业务选择的主目标，再补充其他核心 KPI，便于业务和 AI 统一口径做比较与推荐。</p>
      </div>
      ${renderPrimaryObjectiveBadges(primaryKeys)}
      ${renderSimpleTable(
        ["方案", "来源", ...primaryKeys.map((key) => getObjectiveLabel(key)), ...extraKeys.map((key) => getObjectiveLabel(key))],
        items.map((item) => [
          escapeHtml(item.name),
          escapeHtml(item.source),
          ...primaryKeys.map((key) => metricDisplay(item, key)),
          ...extraKeys.map((key) => metricDisplay(item, key)),
        ]),
      )}
    </article>
  `;
}

function buildAiSelection() {
  const selected = getSelectedReviewCandidates();
  return {
    solution_ids: selected.filter((item) => !item.heuristicRuleName).map((item) => item.id),
    heuristic_rule_names: selected.filter((item) => item.heuristicRuleName).map((item) => item.heuristicRuleName),
  };
}

function entityIdFromGraphId(id) {
  const value = String(id || "");
  const marker = value.indexOf(":");
  return marker >= 0 ? value.slice(marker + 1) : value;
}

function overlapSpan(startA, endA, startB, endB) {
  const start = Math.max(Number(startA), Number(startB));
  const end = Math.min(Number(endA), Number(endB));
  if (Number.isNaN(start) || Number.isNaN(end)) return 0;
  return Math.max(0, end - start);
}

// 机器的班次日历按内容去重后共享（详见 server._instance_details）：机器上只有
// 一个 shift_calendar_id，真正的班次在 details.shift_calendars 里。
function machineShifts(machine) {
  const calendarId = machine?.shift_calendar_id;
  if (calendarId) {
    const shared = app.instanceDetails?.shift_calendars?.[calendarId];
    if (shared) return asArray(shared);
  }
  // 兼容仍内联 shifts/shift_windows 的旧响应
  return asArray(machine?.shift_windows).length ? asArray(machine.shift_windows) : asArray(machine?.shifts);
}

function machineShiftWindows(machine) {
  const entries = machineShifts(machine);
  if (entries.length && entries[0]?.start !== undefined) {
    return entries
      .map((item) => ({
        start: Number(item.start),
        end: Number(item.end),
      }))
      .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start)
      .sort((a, b) => a.start - b.start);
  }
  return entries
    .map((item) => {
      const start = Number(item.day || 0) * 24 + Number(item.start_hour ?? item.start ?? 0);
      const end = Number(item.day || 0) * 24 + Number(item.end_hour ?? item.end ?? (Number(item.start_hour ?? item.start ?? 0) + Number(item.hours ?? 0)));
      return { start, end };
    })
    .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start)
    .sort((a, b) => a.start - b.start);
}

function machineDowntimeRows(machine) {
  return (asArray(machine?.downtimes).length ? asArray(machine?.downtimes) : asArray(app.downtimes).filter((item) => item.machine_id === (machine?.machine_id || machine?.id)))
    .map((item) => ({
      id: item.id,
      type: String(item.downtime_type || "planned").toLowerCase(),
      start: Number(item.start_time ?? item.start),
      end: Number(item.end_time ?? item.end),
    }))
    .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start)
    .sort((a, b) => a.start - b.start);
}

function machineAvailableSpanBetween(machine, start, end) {
  if (start === null || start === undefined || end === null || end === undefined) return 0;
  const startValue = Number(start);
  const endValue = Number(end);
  if (Number.isNaN(startValue) || Number.isNaN(endValue) || endValue <= startValue) return 0;
  const shifts = machineShiftWindows(machine);
  if (!shifts.length) return endValue - startValue;
  let available = 0;
  shifts.forEach((shift) => {
    available += overlapSpan(startValue, endValue, shift.start, shift.end);
  });
  machineDowntimeRows(machine).forEach((item) => {
    available -= overlapSpan(startValue, endValue, item.start, item.end);
  });
  return Math.max(0, available);
}

function flattenTaskRecords() {
  const rows = [];
  asArray(app.instanceDetails?.orders).forEach((order) => {
    asArray(order.tasks).forEach((task) => {
      rows.push({
        ...task,
        order_id: order.id,
        order_name: order.name,
        order_due_at: order.due_at,
        order_release_at: order.release_at,
        order_priority: order.priority,
      });
    });
  });
  return rows;
}

function flattenOperationRecords() {
  const rows = [];
  asArray(app.instanceDetails?.orders).forEach((order) => {
    asArray(order.tasks).forEach((task) => {
      asArray(task.ops).forEach((op) => {
        rows.push({
          ...op,
          order_id: order.id,
          order_name: order.name,
          order_due_at: order.due_at,
          task_name: task.name,
          task_due_at: task.due_at,
          task_derived_due_at: task.derived_due_at,
          task_critical_slack: task.critical_slack,
          task_is_main: task.is_main,
        });
      });
    });
  });
  return rows;
}

function formatSlackDisplay(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  return `${numeric.toFixed(1)}h`;
}

async function refreshHealth(silent = false) {
  try {
    app.health = await api.health();
  } catch (error) {
    if (!silent) toast(`健康检查失败：${error.message}`, "warning");
  }
}

async function loadInstanceBundle() {
  // details 走 lite（跳过 orders 明细，大实例 394MB → MB 级）；
  // instanceDb 在全仓库无消费方（死状态），不再每次导入后全量下载原始表。
  const [details, downtimes] = await Promise.all([
    api.getInstanceDetails(true),
    api.getDowntimes().catch(() => []),
  ]);
  app.instanceDetails = details;
  app.instanceDb = null;
  app.downtimes = Array.isArray(downtimes?.downtimes)
    ? downtimes.downtimes
    : (Array.isArray(downtimes) ? downtimes : []);
  refreshOptimizeBudgetRecommendation({ preserveManual: true });
}

// noInstance = 后端明确说"库里没有实例"(4xx)；其余异常(超时/5xx/网络)属于
// "拿不到"，不是"没有"。两者混为一谈时，一次失败的请求会伪装成未导入数据，
// 把用户退回导入页并跳过流程进度恢复——这正是大实例下 details 返回 470MB
// 超时后的表现。
function isNoInstanceError(error) {
  const status = error?.status ?? error?.response?.status;
  return status === 400 || status === 404;
}

async function syncCurrentScene(silent = false) {
  try {
    await loadInstanceBundle();
    app.currentSceneId = "current-instance";
    app.currentScene = {
      id: "current-instance",
      name: "当前实例",
      summary: app.instanceDetails.summary || {},
    };
    rememberCurrentScene("当前后端实例");
    await refreshHealth(true);
    updateShell();
    return true;
  } catch (error) {
    app.instanceDetails = null;
    app.instanceDb = null;
    app.currentScene = null;
    app.currentSceneId = null;
    app.downtimes = [];
    updateShell();
    if (isNoInstanceError(error)) {
      if (!silent) toast("当前没有可用实例，请先生成或导入。", "warning");
    } else {
      // 加载失败 ≠ 没有数据：必须说清楚，否则用户只会看到"请导入数据"，
      // 以为之前导入的实例和跑完的流程都丢了。
      console.error("加载当前实例失败", error);
      toast(`加载当前实例失败：${error.message}。数据仍在库中，请重试或查看后端日志。`, "warning");
    }
    return false;
  }
}

async function ensureSceneLoaded() {
  if (app.currentScene) return true;
  return syncCurrentScene(true);
}

function setActiveNav(navKey) {
  document.querySelectorAll("[data-nav]").forEach((node) => {
    node.classList.toggle("active", node.dataset.nav === navKey);
  });
  syncSidebarHierarchy();
}

function groupForNav(navKey) {
  return Object.entries(SIDEBAR_GROUPS).find(([, navs]) => navs.includes(navKey))?.[0] || null;
}

function expandSidebarGroup(groupKey, expanded = true) {
  Object.keys(app.sidebarExpanded).forEach((key) => {
    app.sidebarExpanded[key] = key === groupKey ? expanded : false;
  });
}

function syncSidebarHierarchy() {
  document.querySelectorAll(".nav-parent").forEach((parent) => {
    const submenu = parent.nextElementSibling?.classList.contains("nav-submenu") ? parent.nextElementSibling : null;
    if (!submenu) return;
    const groupKey = "optimize";
    const activeInGroup = SIDEBAR_GROUPS[groupKey]?.includes(app.currentNav);
    const expanded = !!app.sidebarExpanded[groupKey] || activeInGroup;
    submenu.classList.toggle("is-collapsed", !expanded);
    parent.classList.toggle("is-expanded", expanded);
    parent.classList.toggle("active-branch", activeInGroup);
  });
}

function showPage(pageName) {
  document.querySelectorAll(".page").forEach((page) => {
    page.classList.toggle("active", page.id === `page-${pageName}`);
  });
}

function updateShell() {
  const hasScene = !!app.currentScene;
  const summary = getSceneSummary();
  el("topbar-scene-name").textContent = hasScene ? app.currentScene.name : "未加载场景";
  el("topbar-orders").textContent = hasScene ? formatInt(summary.orders) : "-";
  el("topbar-tasks").textContent = hasScene ? formatInt(summary.tasks) : "-";
  el("topbar-operations").textContent = hasScene ? formatInt(summary.operations) : "-";
  el("topbar-machines").textContent = hasScene ? formatInt(summary.machines) : "-";
  el("topbar-toolings").textContent = hasScene ? formatInt(summary.toolings) : "-";
  el("topbar-personnel").textContent = hasScene ? formatInt(summary.personnel) : "-";

  document.querySelectorAll(".requires-scene").forEach((node) => {
    node.classList.toggle("is-disabled", !hasScene);
  });
}

async function navigate(navKey, pushHash = true) {
  const resolved = NAV_MAP[navKey] || { page: navKey };
  if (resolved.requiresScene) {
    const ready = await ensureSceneLoaded();
    if (!ready) {
      toast("请先在“数据导入”页生成或导入实例。", "warning");
      app.currentNav = "new-scene";
      app.currentPage = "new-scene";
      setActiveNav("new-scene");
      showPage("new-scene");
      return;
    }
  }
  app.currentNav = navKey;
  app.currentPage = resolved.page;
  const navGroup = groupForNav(navKey);
  if (navGroup) expandSidebarGroup(navGroup, true);
  if (resolved.reviewTab) app.reviewTab = resolved.reviewTab;
  if (resolved.systemTab) app.systemTab = resolved.systemTab;
  if (resolved.workflowStep) app.workflowStep = resolved.workflowStep;
  setActiveNav(navKey);
  showPage(resolved.page);
  if (pushHash) window.location.hash = navKey;
  await renderCurrentPage();
}

function renderKpiCards(items) {
  return `
    <div class="kpi-grid">
      ${items.map((item) => `
        <article class="kpi-card">
          <span>${escapeHtml(item.label)}</span>
          <strong>${item.value}</strong>
          <small>${escapeHtml(item.hint || "")}</small>
        </article>
      `).join("")}
    </div>
  `;
}

function renderKeyValueGrid(items, className = "context-grid") {
  return `
    <div class="${className}">
      ${items.map((item) => `
        <div>
          <span>${escapeHtml(item.label)}</span>
          <strong>${item.value}</strong>
        </div>
      `).join("")}
    </div>
  `;
}

function renderSimpleTable(headers, rows, options = {}) {
  if (!rows.length) {
    return renderEmptyState("暂无数据", "当前区域还没有可展示的内容。");
  }
  return `
    <div class="table-shell">
      <table>
        <thead>
          <tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
      ${options.footer ? `<div class="table-footer">${options.footer}</div>` : ""}
    </div>
  `;
}

function renderDashboard() {
  const container = el("dashboard-content");
  if (!container) return;
  const summary = getSceneSummary();
  const selected = getSelectedReviewCandidate();
  const objectiveKeys = app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys || [];
  const flowSteps = [
    { index: "01", label: "实例准备", done: !!app.currentScene, nav: "new-scene" },
    { index: "02", label: "约束校验", done: Number(summary.operations || 0) > 0 && Number(summary.machines || 0) > 0, nav: "new-scene" },
    { index: "03", label: "基线仿真", done: !!app.simResult, nav: "simulate" },
    { index: "04", label: "优化求解", done: !!app.optimizeResult, nav: "optimize-launch" },
    { index: "05", label: "方案评审", done: !!selected, nav: "solution-review" },
  ];
  const activeStep = flowSteps.findIndex((item) => !item.done);
  const nextStep = flowSteps[activeStep < 0 ? flowSteps.length - 1 : activeStep];
  container.innerHTML = `
    <article class="surface-card executive-hero">
      <div class="executive-hero-copy">
        <span class="eyebrow">Current workflow</span>
        <h3>${activeStep < 0 ? "当前调度流程已完成" : `下一步：${escapeHtml(nextStep.label)}`}</h3>
        <p>${activeStep < 0 ? "实例、仿真和候选方案均已就绪，可继续复核并导出最终方案。" : "按业务顺序推进，每一步的结果会自动成为下一步的输入。"}</p>
        <div class="form-actions"><button class="btn btn-primary" type="button" data-nav-jump="${escapeHtml(nextStep.nav)}">${activeStep < 0 ? "返回方案评审" : `继续${escapeHtml(nextStep.label)}`}</button></div>
      </div>
      <div class="executive-hero-metrics">
        <div><span>\u5f53\u524d\u4e3b\u76ee\u6807</span><strong>${escapeHtml(objectiveShortList(objectiveKeys))}</strong></div>
        <div><span>\u5df2\u5f97\u65b9\u6848\u6570</span><strong>${formatInt(app.optimizeResult?.found_solution_count || app.optimizeResult?.solutions?.length || 0)}</strong></div>
        <div><span>\u5f53\u524d\u5173\u6ce8\u65b9\u6848</span><strong>${escapeHtml(selected?.name || "\u672a\u6307\u5b9a")}</strong></div>
      </div>
    </article>
    <nav class="workflow-overview" aria-label="调度流程进度">
      ${flowSteps.map((item, index) => `
        <button class="workflow-overview-step ${item.done ? "done" : index === activeStep ? "active" : "pending"}" type="button" data-nav-jump="${item.nav}">
          <span>${item.done ? "✓" : item.index}</span>
          <strong>${item.label}</strong>
          <small>${item.done ? "已完成" : index === activeStep ? "下一步" : "待开始"}</small>
        </button>
      `).join("")}
    </nav>
    ${renderKpiCards([
      { label: "\u8ba2\u5355", value: formatInt(summary.orders), hint: "\u5f53\u524d\u5b9e\u4f8b\u8ba2\u5355\u89c4\u6a21" },
      { label: "\u5de5\u5e8f", value: formatInt(summary.operations), hint: "\u5f53\u524d\u5b9e\u4f8b\u5de5\u5e8f\u603b\u91cf" },
      { label: "\u8d44\u6e90", value: `${formatInt(summary.machines)} / ${formatInt(summary.toolings)} / ${formatInt(summary.personnel)}`, hint: "\u673a\u5668 / \u5de5\u88c5 / \u4eba\u5458" },
      { label: "\u4f18\u5316\u72b6\u6001", value: escapeHtml(app.optimizeStatus?.status || (app.optimizeResult ? "done" : "not-started")), hint: "\u6df7\u5408\u4f18\u5316\u4efb\u52a1\u5f53\u524d\u72b6\u6001" },
    ])}
    <div class="three-column">
      <article class="surface-card">
        <div class="card-head"><h3>\u95ee\u9898\u89c4\u6a21</h3><p>\u5e2e\u52a9\u5feb\u901f\u5224\u65ad\u5f53\u524d\u5b9e\u4f8b\u662f\u8054\u8c03\u7ea7\u3001\u4e2d\u578b\u8fd8\u662f\u6f14\u793a\u7ea7\u5927\u5b9e\u4f8b\u3002</p></div>
        ${renderKeyValueGrid([
          { label: "\u4efb\u52a1\u6570", value: formatInt(summary.tasks) },
          { label: "\u505c\u673a\u8bb0\u5f55", value: formatInt(app.downtimes.length) },
          { label: "\u5728\u5236\u5de5\u5e8f", value: formatInt(summary.ops_in_progress || 0) },
          { label: "\u8ba1\u5212\u8d77\u70b9", value: formatDateTime(app.instanceDetails?.plan_start_at) },
        ])}
      </article>
      <article class="surface-card">
        <div class="card-head"><h3>\u4f18\u5316\u8fdb\u5c55</h3><p>\u5173\u6ce8\u524d\u671f\u8fd1\u4f3c\u5e7f\u641c\u548c\u540e\u671f\u7cbe\u786e\u8bc4\u4f30\u7684\u63a8\u8fdb\u60c5\u51b5\u3002</p></div>
        ${renderKeyValueGrid([
          { label: "\u8fd1\u4f3c\u8bc4\u4f30", value: formatInt(app.optimizeResult?.approximate_evaluations || app.optimizeStatus?.approximate_evaluations || 0) },
          { label: "\u7cbe\u786e\u8bc4\u4f30", value: formatInt(app.optimizeResult?.exact_evaluations || app.optimizeStatus?.exact_evaluations || 0) },
          { label: "\u603b\u8bc4\u4f30\u6b21\u6570", value: formatInt(app.optimizeResult?.total_evaluations || app.optimizeStatus?.total_evaluations || 0) },
          { label: "\u8017\u65f6", value: formatDurationSeconds(app.optimizeResult?.elapsed_s || app.optimizeStatus?.elapsed_s || 0) },
        ])}
      </article>
      <article class="surface-card">
        <div class="card-head"><h3>\u5f53\u524d\u63a8\u8350\u5173\u6ce8</h3><p>\u5982\u679c\u5df2\u7ecf\u6709\u65b9\u6848\u7ed3\u679c\uff0c\u4f18\u5148\u5173\u6ce8\u5f53\u524d\u8bc4\u5ba1\u9009\u4e2d\u7684\u5019\u9009\u65b9\u6848\u3002</p></div>
        ${selected ? renderKeyValueGrid([
          { label: "\u65b9\u6848\u540d\u79f0", value: escapeHtml(selected.name || "-") },
          { label: "\u603b\u5ef6\u8bef", value: metricDisplay(selected, "total_tardiness") },
          { label: "\u603b\u5468\u671f", value: metricDisplay(selected, "makespan") },
          { label: "\u51c0\u53ef\u7528\u5229\u7528\u7387", value: metricDisplay(selected, "avg_net_available_utilization") },
        ]) : renderEmptyState("\u8fd8\u6ca1\u6709\u5019\u9009\u65b9\u6848", "\u5148\u8fd0\u884c\u6df7\u5408\u4f18\u5316\uff0c\u6216\u751f\u6210\u542f\u53d1\u5f0f / \u7cbe\u786e\u53c2\u8003\u65b9\u6848\u540e\u518d\u6765\u8fd9\u91cc\u6c47\u603b\u67e5\u770b\u3002")}
      </article>
    </div>
  `;
}


function buildMachineOverlays(machine, horizonStart, horizonEnd) {
  const overlays = [];
  const totalSpan = Math.max(horizonEnd - horizonStart, 1e-6);
  const rawDowntimes = asArray(machine?.downtimes).length
    ? asArray(machine.downtimes)
    : asArray(app.downtimes).filter((item) => item.machine_id === (machine?.machine_id || machine?.id));

  rawDowntimes.forEach((item) => {
    const start = Number(item.start_time ?? item.start ?? item.start_hour);
    const end = Number(item.end_time ?? item.end ?? item.end_hour);
    if (Number.isNaN(start) || Number.isNaN(end) || end <= horizonStart || start >= horizonEnd) return;
    const clampedStart = Math.max(start, horizonStart);
    const clampedEnd = Math.min(end, horizonEnd);
    overlays.push({
      className: `timeline-overlay ${item.downtime_type === "unplanned" ? "unplanned" : "planned"}`,
      left: `${((clampedStart - horizonStart) / totalSpan) * 100}%`,
      width: `${((clampedEnd - clampedStart) / totalSpan) * 100}%`,
      startOffset: clampedStart,
      endOffset: clampedEnd,
      title: `${item.downtime_type === "unplanned" ? "非计划停机" : "计划停机"} · ${formatTimelineLabel(start)} ~ ${formatTimelineLabel(end)}`,
    });
  });

  const shifts = machineShiftWindows(machine);

  if (shifts.length) {
    const filtered = shifts
      .filter((item) => item.end > horizonStart && item.start < horizonEnd)
      .sort((a, b) => a.start - b.start);
    let cursor = horizonStart;
    filtered.forEach((item) => {
      if (item.start > cursor) {
        overlays.push({
          className: "timeline-overlay offshift",
          left: `${((cursor - horizonStart) / totalSpan) * 100}%`,
          width: `${((item.start - cursor) / totalSpan) * 100}%`,
          startOffset: cursor,
          endOffset: item.start,
          title: `班次外 · ${formatTimelineLabel(cursor)} ~ ${formatTimelineLabel(item.start)}`,
        });
      }
      cursor = Math.max(cursor, item.end);
    });
    if (cursor < horizonEnd) {
      overlays.push({
        className: "timeline-overlay offshift",
        left: `${((cursor - horizonStart) / totalSpan) * 100}%`,
        width: `${((horizonEnd - cursor) / totalSpan) * 100}%`,
        startOffset: cursor,
        endOffset: horizonEnd,
        title: `班次外 · ${formatTimelineLabel(cursor)} ~ ${formatTimelineLabel(horizonEnd)}`,
      });
    }
  }
  return overlays;
}

const GANTT_FALLBACK_BASE = "2000-01-01T00:00:00.000Z";

function ganttOffsetToISO(offset, base) {
  return new Date(new Date(base).getTime() + Number(offset) * 3600 * 1000).toISOString();
}

function ganttWindowPayload(horizonStart, horizonEnd, base, nowOffset, viewKey) {
  const fullWindow = {
    start: ganttOffsetToISO(horizonStart, base),
    end: ganttOffsetToISO(horizonEnd, base),
  };
  const effectiveNow = nowOffset !== null && nowOffset >= horizonStart && nowOffset <= horizonEnd
    ? ganttOffsetToISO(nowOffset, base)
    : fullWindow.start;
  return {
    fullWindow,
    initialWindow: ReviewRuntime.computeInitialWindow(fullWindow.start, fullWindow.end, effectiveNow),
    nowISO: effectiveNow,
    viewKey,
  };
}

const GANTT_MACHINE_FILTER_DEFAULT = Object.freeze({ type: "__all__", downtimeOnly: false, query: "", page: 1 });
const GANTT_ENTRY_FILTER_DEFAULT = Object.freeze({ status: "__all__", query: "", from: "", to: "" });

function getGanttMachineFilter(canvasId) {
  return { ...GANTT_MACHINE_FILTER_DEFAULT, ...(app.ganttMachineFilter[canvasId] || {}) };
}

function getGanttEntryFilter(canvasId) {
  return { ...GANTT_ENTRY_FILTER_DEFAULT, ...(app.ganttEntryFilter[canvasId] || {}) };
}

// 机器维度信息：类型名 / 是否含停机 / 工序数，供筛选、排序与分页使用
function describeGanttMachine(machineId, machineName, machineMap, opCount) {
  const machine = machineMap.get(machineId);
  const typeName = machine?.type_name || machine?.type || machine?.machine_type || "未分组";
  // machineDowntimeRows 双数据源（machine.downtimes 优先，回退 /api/downtime 的 app.downtimes）
  const hasDowntime = machineDowntimeRows(machine || { machine_id: machineId }).length > 0;
  return { id: machineId, name: machineName, typeName, hasDowntime, opCount };
}

function filterGanttMachineRows(rows, filter) {
  const query = String(filter.query || "").trim().toLowerCase();
  return rows.filter((row) => {
    if (filter.type !== "__all__" && row.typeName !== filter.type) return false;
    if (filter.downtimeOnly && !row.hasDowntime) return false;
    if (query && !`${row.id} ${row.name}`.toLowerCase().includes(query)) return false;
    return true;
  });
}

// 含停机的机器优先，其次工序数降序——资源异常先被看到，而不是“最忙的前 N 台”
function sortGanttMachineRows(rows) {
  return rows.slice().sort((a, b) =>
    (Number(b.hasDowntime) - Number(a.hasDowntime))
    || (b.opCount - a.opCount)
    || String(a.id).localeCompare(String(b.id), "zh-CN", { numeric: true }));
}

// 订单内二级筛选：状态 / 关键字（工序号·任务·机器） / 时间窗（偏移小时）
function filterGanttEntries(entries, filter) {
  const query = String(filter.query || "").trim().toLowerCase();
  const from = filter.from === "" || filter.from === null || filter.from === undefined ? null : Number(filter.from);
  const to = filter.to === "" || filter.to === null || filter.to === undefined ? null : Number(filter.to);
  return entries.filter((item) => {
    if (filter.status !== "__all__" && normalizeScheduleStatus(item.status) !== filter.status) return false;
    if (query) {
      const haystack = `${item.op_id || item.operation_id || item.id || ""} ${item.task_id || ""} ${item.machine_id || ""} ${item.machine_name || ""}`.toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    const start = Number(item.start ?? item.start_time ?? 0);
    const end = Number(item.end ?? item.end_time ?? 0);
    if (from !== null && !Number.isNaN(from) && end < from) return false;
    if (to !== null && !Number.isNaN(to) && start > to) return false;
    return true;
  });
}

// 甘特"现在"线：进度分界线 = 进行中工序最早开始；无进行中则取已完成最晚结束
function ganttProgressNowOffset(normalized) {
  const processingStarts = normalized.filter((item) => item.status === "processing").map((item) => item.start);
  if (processingStarts.length) return Math.min(...processingStarts);
  const completedEnds = normalized.filter((item) => item.status === "completed").map((item) => item.end);
  if (completedEnds.length) return Math.max(...completedEnds);
  return null;
}

function buildGanttData(entries, options = {}) {
  const planStartAt = tryParseDate(app.instanceDetails?.plan_start_at);
  const hasRealBase = Boolean(planStartAt);
  const base = hasRealBase ? planStartAt.toISOString() : GANTT_FALLBACK_BASE;
  const groupMode = options.groupMode === "order" ? "order" : "machine";
  const viewKey = JSON.stringify({
    canvasId: options.canvasId || "",
    selectedOrder: options.selectedOrder || "",
    solutionIds: asArray(options.solutionIds).map(String).sort(),
    groupMode,
  });

  const normalized = asArray(entries)
    .map((item) => ({
      machineId: item.machine_id || item.machine_name || item.resource_id || "unknown",
      machineName: item.machine_name || item.machine_id || item.resource_name || "未知资源",
      opId: item.op_id || item.operation_id || item.id || "-",
      orderId: item.order_id || "-",
      orderName: item.order_name || "",
      taskId: item.task_id || "-",
      start: Number(item.start ?? item.start_time ?? 0),
      end: Number(item.end ?? item.end_time ?? 0),
      status: normalizeScheduleStatus(item.status),
      statusLabel: item.status_label || (normalizeScheduleStatus(item.status) === "completed" ? "已完成" : normalizeScheduleStatus(item.status) === "processing" ? "进行中" : "未来排产"),
    }))
    .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start);

  if (!normalized.length) return null;

  const machineMap = getMachineMap();
  const countByMachine = new Map();
  const nameByMachine = new Map();
  normalized.forEach((item) => {
    countByMachine.set(item.machineId, (countByMachine.get(item.machineId) || 0) + 1);
    if (!nameByMachine.has(item.machineId)) nameByMachine.set(item.machineId, item.machineName);
  });

  // 机器维度信息（两种分组模式共用）：类型 / 仅含停机 / 关键字 筛选
  const machineFilter = getGanttMachineFilter(options.canvasId);
  const allRows = sortGanttMachineRows(
    Array.from(countByMachine.keys(), (machineId) =>
      describeGanttMachine(machineId, nameByMachine.get(machineId), machineMap, countByMachine.get(machineId)))
  );
  const typeOptions = Array.from(new Set(allRows.map((row) => row.typeName)))
    .sort((a, b) => String(a).localeCompare(String(b), "zh-CN", { numeric: true }));
  const filteredRows = filterGanttMachineRows(allRows, machineFilter);
  const downtimeMachineIds = new Set(allRows.filter((row) => row.hasDowntime).map((row) => row.id));
  const downtimeGroups = downtimeMachineIds.size;
  const machineTypeName = (machineId) => allRows.find((row) => row.id === machineId)?.typeName || "未分组";

  const ganttItemTitle = (item) => `${item.statusLabel} · ${item.opId}\n订单:${item.orderId} 任务:${item.taskId}\n机器:${item.machineName}（${machineTypeName(item.machineId)}）\n${hasRealBase ? `${formatDateTime(ganttOffsetToISO(item.start, base))} ~ ${formatDateTime(ganttOffsetToISO(item.end, base))}` : `相对 ${item.start}h ~ ${item.end}h`}`;
  const overlayClass = (ov) => (ov.className.includes("unplanned") ? "unplanned" : ov.className.includes("planned") ? "planned" : "offshift");
  const nowOffset = ganttProgressNowOffset(normalized);

  // —— 模式一：按订单层级（订单 ▸ 任务令 ▸ 工序行，工序行标机器），按工序行分页 ——
  if (groupMode === "order") {
    const passingMachines = new Set(filteredRows.map((row) => row.id));
    const machineFiltered = normalized.filter((item) => passingMachines.has(item.machineId));
    const pageSize = CONFIG.GANTT_ORDER_PAGE_SIZE || 60;
    const pageCount = Math.max(1, Math.ceil(machineFiltered.length / pageSize));
    const page = Math.min(Math.max(1, Number(machineFilter.page) || 1), pageCount);
    const visible = machineFiltered
      .slice()
      .sort((a, b) => String(a.orderId).localeCompare(String(b.orderId), "zh-CN", { numeric: true })
        || String(a.taskId).localeCompare(String(b.taskId), "zh-CN", { numeric: true })
        || (a.start - b.start)
        || String(a.opId).localeCompare(String(b.opId), "zh-CN", { numeric: true }))
      .slice((page - 1) * pageSize, page * pageSize);
    const facet = {
      mode: "order",
      totalGroups: machineFiltered.length,
      filteredGroups: machineFiltered.length,
      machineGroups: filteredRows.length,
      downtimeGroups,
      page, pageCount, pageSize, typeOptions, filter: machineFilter,
    };
    if (!visible.length) {
      return {
        groups: [],
        items: [],
        hasRealBase,
        machineFacet: facet,
        fullWindow: null,
        initialWindow: null,
        nowISO: null,
        viewKey,
        mode: "order",
      };
    }
    const horizonStart = Math.min(...visible.map((i) => i.start));
    const horizonEnd = Math.max(...visible.map((i) => i.end));

    const orderBuckets = new Map();
    visible.forEach((item) => {
      if (!orderBuckets.has(item.orderId)) orderBuckets.set(item.orderId, { name: item.orderName, tasks: new Map(), opCount: 0 });
      const bucket = orderBuckets.get(item.orderId);
      if (!bucket.tasks.has(item.taskId)) bucket.tasks.set(item.taskId, []);
      bucket.tasks.get(item.taskId).push(item);
      bucket.opCount += 1;
    });

    const groups = [];
    const items = [];
    const opGroupIdOf = new Map();
    let seq = 0;
    orderBuckets.forEach((bucket, orderId) => {
      const taskGroupIds = Array.from(bucket.tasks.keys()).map((taskId) => `gtask|${orderId}|${taskId}`);
      groups.push({
        id: `gorder|${orderId}`,
        content: `<span class="gantt-order-dot" style="background:${orderColorFor(orderId)}"></span>${escapeHtml(orderId)}${bucket.name && bucket.name !== orderId ? ` · ${escapeHtml(bucket.name)}` : ""}<span class="gantt-group-count">${formatInt(bucket.opCount)} 道工序</span>`,
        className: "gantt-group-order",
        nestedGroups: taskGroupIds,
        showNested: true,
        seq: (seq += 1),
      });
      bucket.tasks.forEach((ops, taskId) => {
        const taskGroupId = `gtask|${orderId}|${taskId}`;
        const opIds = ops.map((op, index) => `${taskGroupId}|op${index}`);
        groups.push({
          id: taskGroupId,
          content: `${escapeHtml(taskId)}<span class="gantt-group-count">${formatInt(ops.length)} 道工序</span>`,
          className: "gantt-group-task",
          nestedGroups: opIds,
          showNested: true,
          seq: (seq += 1),
        });
        ops.forEach((op, index) => {
          groups.push({
            id: opIds[index],
            content: `<span class="gantt-op-label">${escapeHtml(op.opId)}</span><span class="gantt-op-machine">${escapeHtml(op.machineName)}${downtimeMachineIds.has(op.machineId) ? ' <span class="gantt-group-downtime" title="该机台含停机时段">⚠</span>' : ""}</span>`,
            className: "gantt-group-op",
            seq: (seq += 1),
          });
          opGroupIdOf.set(op, opIds[index]);
        });
      });
    });

    visible.forEach((item) => {
      const groupId = opGroupIdOf.get(item);
      items.push({
        id: `op-${groupId}`,
        group: groupId,
        start: ganttOffsetToISO(item.start, base),
        end: ganttOffsetToISO(item.end, base),
        content: escapeHtml(item.opId),
        className: `status-${item.status}`,
        style: `background:${ganttStatusBackground(orderColorFor(item.orderId), item.status)};`,
        title: ganttItemTitle(item),
      });
      // 工序行即该机台的时间切片：叠加该机台的班次外/停机遮罩，解释行内空档
      buildMachineOverlays(machineMap.get(item.machineId), horizonStart, horizonEnd).forEach((ov, i) => {
        items.push({
          id: `bg-${groupId}-${i}`,
          group: groupId,
          start: ganttOffsetToISO(ov.startOffset, base),
          end: ganttOffsetToISO(ov.endOffset, base),
          type: "background",
          className: overlayClass(ov),
          title: ov.title,
        });
      });
    });

    return {
      groups,
      items,
      hasRealBase,
      machineFacet: facet,
      ...ganttWindowPayload(horizonStart, horizonEnd, base, nowOffset, viewKey),
      mode: "order",
    };
  }

  // —— 模式二：按机器资源（原逻辑），大实例按筛选+分页控制单页行数，vis 不会锁死 ——
  const pageSize = CONFIG.GANTT_PAGE_SIZE || CONFIG.GANTT_MAX_GROUPS;
  const pageCount = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const page = Math.min(Math.max(1, Number(machineFilter.page) || 1), pageCount);
  const pageRows = filteredRows.slice((page - 1) * pageSize, page * pageSize);
  const keep = new Set(pageRows.map((row) => row.id));

  const visible = normalized.filter((item) => keep.has(item.machineId));
  const groups = pageRows.map((row, index) => ({
    id: row.id,
    content: `${escapeHtml(row.name)}${row.hasDowntime ? ' <span class="gantt-group-downtime" title="该机台含停机时段">⚠</span>' : ""}`,
    className: "gantt-group-machine",
    seq: index + 1,
  }));
  const machineFacet = {
    mode: "machine",
    totalGroups: allRows.length,
    filteredGroups: filteredRows.length,
    downtimeGroups,
    page, pageCount, pageSize, typeOptions, filter: machineFilter,
  };

  if (!visible.length) {
    // 筛选/翻页后本页无条目（如关键字无命中）：保留 facet 供工具条渲染
    return {
      groups,
      items: [],
      hasRealBase,
      machineFacet,
      fullWindow: null,
      initialWindow: null,
      nowISO: null,
      viewKey,
      mode: "machine",
    };
  }

  const horizonStart = Math.min(...visible.map((i) => i.start));
  const horizonEnd = Math.max(...visible.map((i) => i.end));

  const items = [];
  visible.forEach((item, index) => {
    items.push({
      id: `op-${index}`,
      group: item.machineId,
      start: ganttOffsetToISO(item.start, base),
      end: ganttOffsetToISO(item.end, base),
      content: escapeHtml(item.opId),
      className: `status-${item.status}`,
      style: `background:${ganttStatusBackground(orderColorFor(item.orderId), item.status)};`,
      title: ganttItemTitle(item),
    });
  });

  // 遮罩：班次外 / 停机 -> background 项；title 供悬停显示起止时刻与类型（vis 对 background 项同样出 tooltip）
  groups.forEach((g) => {
    const overlays = buildMachineOverlays(machineMap.get(g.id), horizonStart, horizonEnd);
    overlays.forEach((ov, i) => {
      items.push({
        id: `bg-${g.id}-${i}`,
        group: g.id,
        start: ganttOffsetToISO(ov.startOffset, base),
        end: ganttOffsetToISO(ov.endOffset, base),
        type: "background",
        className: overlayClass(ov),
        title: ov.title,
      });
    });
  });

  return {
    groups,
    items,
    hasRealBase,
    machineFacet,
    ...ganttWindowPayload(horizonStart, horizonEnd, base, nowOffset, viewKey),
    mode: "machine",
  };
}

function renderTimeline(entries, options = {}) {
  const id = options.canvasId || `gantt-${(options.title || "t").replace(/[^a-zA-Z0-9]/g, "").slice(0, 24)}`;
  const allEntries = asArray(entries);

  // 服务端按订单取数模式：订单 facet + entries 由后端下发，前端不再全量过滤（见评审页方案详情甘特图）。
  const serverOrders = options.serverOrders || null;
  const serverMode = !!serverOrders;
  if (serverMode && serverOrders.loading) {
    return `
      <div class="surface-card">
        <div class="card-head"><h3>${escapeHtml(options.title || "资源甘特图")}</h3></div>
        <div class="empty-state"><p>正在加载订单排程…</p></div>
      </div>
    `;
  }
  if (serverMode && serverOrders.error) {
    return `
      <div class="surface-card">
        <div class="card-head"><h3>${escapeHtml(options.title || "资源甘特图")}</h3></div>
        <div class="empty-state">
          <p>加载订单排程失败：${escapeHtml(serverOrders.error)}</p>
          <button class="btn-ghost" data-action="retry-plan-gantt">重试</button>
        </div>
      </div>
    `;
  }

  // 订单筛选（同图谱页的按订单聚焦逻辑）：大实例整图渲染会锁死主线程，
  // 一次只展示一个订单及其工序、涉及的机器；小实例才提供"全部订单"。
  const orderNames = new Map();
  if (serverMode) {
    asArray(serverOrders.orders).forEach((o) => orderNames.set(o.order_id, o.order_name || ""));
  } else {
    allEntries.forEach((item) => {
      const key = item.order_id || "-";
      if (!orderNames.has(key)) orderNames.set(key, item.order_name || "");
    });
  }
  const orderOptions = Array.from(orderNames.keys())
    .sort((a, b) => String(a).localeCompare(String(b), "zh-CN", { numeric: true }));
  const allowAll = serverMode ? false : allEntries.length <= CONFIG.GANTT_ALL_ORDERS_MAX_OPS;
  let selectedOrder;
  if (serverMode) {
    selectedOrder = serverOrders.selectedOrder || orderOptions[0];
  } else {
    selectedOrder = app.ganttOrderFilter[id];
    if (selectedOrder === "__all__" && !allowAll) selectedOrder = null;
    if (selectedOrder !== "__all__" && !orderOptions.includes(selectedOrder)) selectedOrder = null;
    if (!selectedOrder) selectedOrder = allowAll ? "__all__" : orderOptions[0];
  }
  const orderEntries = serverMode
    ? allEntries
    : (selectedOrder === "__all__"
      ? allEntries
      : allEntries.filter((item) => (item.order_id || "-") === selectedOrder));

  // 订单内二级筛选：状态 / 关键字 / 时间窗（选中订单后仍可继续过滤，见问题 B）
  const entryFilter = getGanttEntryFilter(id);
  const visibleEntries = filterGanttEntries(orderEntries, entryFilter);

  // 分组方式：按订单层级（订单▸任务令▸工序行）/ 按机器资源；"全部订单"时订单层级过重，强制机器模式
  const allowOrderMode = selectedOrder !== "__all__";
  const storedMode = app.ganttGroupMode[id];
  const groupMode = allowOrderMode ? (storedMode === "machine" ? "machine" : "order") : "machine";

  const data = buildGanttData(visibleEntries, {
    ...options,
    canvasId: id,
    groupMode,
    selectedOrder,
    solutionIds: asArray(options.solutionIds),
  });
  if (!data) {
    return renderEmptyState("暂无甘特数据", "当前方案还没有可显示的资源排程。");
  }
  // 连同已计算的 data 一起暂存：mountGantts 直接复用，避免大实例下重复构建（A3）
  app.pendingGantts.set(id, { entries: visibleEntries, options: { ...options, canvasId: id, groupMode }, data });

  const selectedOrderColor = selectedOrder === "__all__" ? null : orderColorFor(selectedOrder);
  const orderSearchItems = orderOptions.map((orderId) => ({
    order_id: orderId,
    order_name: orderNames.get(orderId) || "",
  }));
  if (allowAll) {
    orderSearchItems.unshift({
      order_id: "__all__",
      order_name: `全部订单（${formatInt(allEntries.length)} 道工序）`,
    });
  }
  const selectedOrderItem = orderSearchItems.find((order) => order.order_id === selectedOrder) || null;
  const orderComboboxId = `${id}-order`;
  const taskId = serverOrders?.taskId || app.planGantt.taskId;
  const solutionId = serverOrders?.solutionId || app.planGantt.solutionId;
  const orderSelector = orderOptions.length > 1 || !allowAll ? `
    <div class="field-inline">
      <span>订单</span>
      ${renderOrderCombobox({
        id: orderComboboxId,
        selected: selectedOrderItem,
        search: serverMode
          ? async (query, signal) => {
            const payload = await api.searchReviewOrders(taskId, [solutionId], query, signal);
            return asArray(payload?.orders);
          }
          : (query) => ReviewRuntime.rankOrders(orderSearchItems, query, ORDER_SEARCH_LIMIT),
        select: serverMode
          ? (order) => loadPlanGantt(taskId, solutionId, order.order_id)
          : (order) => {
            app.ganttOrderFilter[id] = order.order_id;
            return renderCurrentPage();
          },
      })}
      <span>分组</span>
      <select data-gantt-group-mode data-canvas="${escapeHtml(id)}">
        <option value="order" ${groupMode === "order" ? "selected" : ""} ${allowOrderMode ? "" : "disabled"}>按订单层级</option>
        <option value="machine" ${groupMode === "machine" ? "selected" : ""}>按机器资源</option>
      </select>
    </div>
  ` : "";

  const facet = data.machineFacet;
  const machineFilterBar = facet ? `
    <div class="field-inline gantt-filter-bar">
      <span>机器类型</span>
      <select data-gantt-machine-type data-canvas="${escapeHtml(id)}">
        <option value="__all__" ${facet.filter.type === "__all__" ? "selected" : ""}>全部类型（${formatInt(groupMode === "order" ? facet.machineGroups : facet.totalGroups)} 台）</option>
        ${facet.typeOptions.map((t) => `<option value="${escapeHtml(t)}" ${t === facet.filter.type ? "selected" : ""}>${escapeHtml(t)}</option>`).join("")}
      </select>
      <label class="gantt-filter-check"><input type="checkbox" data-gantt-downtime-only data-canvas="${escapeHtml(id)}" ${facet.filter.downtimeOnly ? "checked" : ""}> 仅含停机（${formatInt(facet.downtimeGroups)} 台）</label>
      <input type="search" class="gantt-filter-query" placeholder="搜索机器号…" value="${escapeHtml(facet.filter.query)}" data-gantt-machine-query data-canvas="${escapeHtml(id)}">
    </div>
  ` : "";
  const entryFilterBar = `
    <div class="field-inline gantt-filter-bar">
      <span>工序状态</span>
      <select data-gantt-status-filter data-canvas="${escapeHtml(id)}">
        <option value="__all__" ${entryFilter.status === "__all__" ? "selected" : ""}>全部状态</option>
        <option value="completed" ${entryFilter.status === "completed" ? "selected" : ""}>已完成</option>
        <option value="processing" ${entryFilter.status === "processing" ? "selected" : ""}>进行中</option>
        <option value="future" ${entryFilter.status === "future" ? "selected" : ""}>未来排产</option>
      </select>
      <input type="search" class="gantt-filter-query" placeholder="搜索工序/任务/机器…" value="${escapeHtml(entryFilter.query)}" data-gantt-entry-query data-canvas="${escapeHtml(id)}">
      <span>时间窗(h)</span>
      <input type="number" class="gantt-filter-num" placeholder="起" value="${escapeHtml(String(entryFilter.from))}" data-gantt-time-from data-canvas="${escapeHtml(id)}">
      <span>~</span>
      <input type="number" class="gantt-filter-num" placeholder="止" value="${escapeHtml(String(entryFilter.to))}" data-gantt-time-to data-canvas="${escapeHtml(id)}">
    </div>
  `;
  const pager = facet && facet.pageCount > 1 ? `
    <div class="gantt-pager">
      <button class="btn-ghost" data-gantt-page="${facet.page - 1}" data-canvas="${escapeHtml(id)}" ${facet.page <= 1 ? "disabled" : ""}>上一页</button>
      <span>第 ${formatInt(facet.page)} / ${formatInt(facet.pageCount)} 页 · 命中 ${formatInt(facet.filteredGroups)} / ${formatInt(facet.totalGroups)} ${groupMode === "order" ? "道工序行" : "台机器"}</span>
      <button class="btn-ghost" data-gantt-page="${facet.page + 1}" data-canvas="${escapeHtml(id)}" ${facet.page >= facet.pageCount ? "disabled" : ""}>下一页</button>
    </div>
  ` : "";

  const statusCounts = data.items.reduce((acc, it) => {
    if (it.type === "background") return acc;
    const key = (it.className || "").replace("status-", "");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, { completed: 0, processing: 0, future: 0 });

  // 图例：订单标识色（与图谱订单节点外圈一致）+ 状态质感 + 遮罩 + 现在线
  const legendOrderIds = Array.from(new Set(visibleEntries.map((item) => item.order_id || "-"))).slice(0, 6);
  const repColor = orderColorFor(legendOrderIds[0] || selectedOrder || "-");
  const legendHtml = `
    <div class="legend">
      ${legendOrderIds.map((orderId) => `<span class="legend-item"><span class="legend-swatch order-dot" style="background:${orderColorFor(orderId)}"></span>${escapeHtml(orderId === "-" ? "未指定订单" : orderId)}</span>`).join("")}
      <span class="legend-item"><span class="legend-swatch" style="background:${ganttStatusBackground(repColor, "completed")}"></span>已完成</span>
      <span class="legend-item"><span class="legend-swatch" style="background:${ganttStatusBackground(repColor, "processing")}"></span>进行中</span>
      <span class="legend-item"><span class="legend-swatch" style="background:${ganttStatusBackground(repColor, "future")}"></span>未来排产</span>
      <span class="legend-item"><span class="legend-swatch offshift"></span>班次外</span>
      <span class="legend-item"><span class="legend-swatch planned"></span>计划停机</span>
      <span class="legend-item"><span class="legend-swatch unplanned"></span>非计划停机</span>
      ${data.nowISO ? '<span class="legend-item"><span class="legend-swatch now-line"></span>现在（进度分界）</span>' : ""}
    </div>
  `;

  return `
    <div class="surface-card">
      <div class="card-head">
        <h3>${escapeHtml(options.title || "资源甘特图")}</h3>
        <p>条块颜色=订单标识色（与图谱订单节点外圈一致），质感区分已完成 / 进行中 / 未来排产；红色竖线为"现在"进度分界；斜纹遮罩显示班次外与停机占用（悬停可见起止与类型）。</p>
      </div>
      ${orderSelector}
      ${machineFilterBar}
      ${entryFilterBar}
      <div class="timeline-summary-strip">
        <div class="timeline-summary-card"><span>当前展示</span><strong>${selectedOrderColor ? `<i class="gantt-order-dot" style="background:${selectedOrderColor}"></i>` : ""}${selectedOrder === "__all__" ? "全部订单" : escapeHtml(selectedOrder)}（共 ${formatInt(orderOptions.length)} 个订单${serverMode ? ` / ${formatInt(serverOrders.totalOperations || 0)} 道工序` : ""}）</strong></div>
        <div class="timeline-summary-card"><span>${groupMode === "order" ? "工序行 / 涉及机器" : "工序 / 资源行数"}</span><strong>${groupMode === "order" ? `${formatInt(visibleEntries.length)} / ${formatInt(facet ? facet.machineGroups : 0)} 台` : `${formatInt(visibleEntries.length)} / ${formatInt(data.groups.length)}${facet ? `（共 ${formatInt(facet.totalGroups)} 台）` : ""}`}</strong></div>
        <div class="timeline-summary-card"><span>已完成 / 进行中 / 未来</span><strong>${formatInt(statusCounts.completed)} / ${formatInt(statusCounts.processing)} / ${formatInt(statusCounts.future)}</strong></div>
        <div class="timeline-summary-card"><span>时间基准</span><strong>${data.hasRealBase ? "计划起始时间" : "相对小时（无 plan_start_at）"}</strong></div>
      </div>
      ${pager}
      ${legendHtml}
      <div class="gantt-canvas" id="${escapeHtml(id)}"></div>
    </div>
  `;
}

// —— 图谱视觉编码：配色（token 派生）+ 形状 + 尺寸 三重编码 ——
// 规划链冷色（订单/任务/工序），资源暖色（机器/工装/人员），与 design-system token 对齐
const GRAPH_TYPE_COLORS = {
  order: "#1d53c0",      // 靛蓝（--primary-strong）
  task: "#0f6e56",       // 蓝绿——与订单靛蓝拉开色相，避免规划链上下游同蓝易混
  operation: "#3b6d11",  // 绿——规划链冷色梯度收束在绿，区别于任务的蓝绿
  machine: "#c2620a",    // --accent 琥珀
  tooling: "#bf3f8c",    // 暖品红
  personnel: "#7048e8",  // 暖紫
  other: "#7a8795",
};
// 节点形状尺寸（半径基准，选中时 +4）
const GRAPH_NODE_SIZES = { order: 24, task: 20, operation: 12, machine: 19, tooling: 17, personnel: 17, other: 15 };
// 节点内单字标签
const GRAPH_TYPE_CHARS = { order: "订", task: "任", operation: "工", machine: "机", tooling: "装", personnel: "员", other: "·" };
// 订单标识色板：固定的一套 12 色，按订单序号（自然排序后的位置）确定性分配，
// 不哈希、不随机——同一实例任何时刻、任何视图（图谱订单节点外圈 / 面包屑色点 /
// 甘特条块 / 图例色点）拿到的都是同一套颜色。色板与 design-system token 协调，
// 按相邻对比度排序，均保证白字可读。
const ORDER_COLOR_PALETTE = [
  "#2f6feb", // 蓝（--primary）
  "#c2620a", // 琥珀（--accent）
  "#0e8e7f", // 青（--info）
  "#7048e8", // 暖紫
  "#b45309", // 赭橙（--warning）
  "#bf3f8c", // 品红
  "#12a150", // 绿（--success）
  "#0b5f8a", // 深青蓝
  "#8d6e63", // 棕灰
  "#c2410c", // 橙红
  "#4d7c0f", // 橄榄
  "#64748b", // 石板灰
];
const ORDER_COLOR_NEUTRAL = "#64748b";

function graphTypeColor(type) {
  return GRAPH_TYPE_COLORS[String(type || "").toLowerCase()] || GRAPH_TYPE_COLORS.other;
}

// 已加载实例的订单全集（图谱下拉选项），作为颜色分配的固定序号来源
function knownOrderIds() {
  const ids = new Set();
  asArray(app.graphOrderOptions).forEach((raw) => {
    const entity = raw.entity_id || entityIdFromGraphId(raw.node_id || raw.id || "");
    if (entity) ids.add(String(entity));
  });
  return ids;
}

function orderColorFor(orderId) {
  const key = String(orderId || "").trim();
  if (!key || key === "-") return ORDER_COLOR_NEUTRAL;
  const ids = knownOrderIds();
  ids.add(key);
  const sorted = Array.from(ids).sort((a, b) => a.localeCompare(b, "zh-CN", { numeric: true }));
  const idx = Math.max(0, sorted.indexOf(key));
  const base = ORDER_COLOR_PALETTE[idx % ORDER_COLOR_PALETTE.length];
  // 超过 12 个订单时确定性派生：每循环一轮交替加深/减淡，仍属同一套固定规则
  const cycle = Math.floor(idx / ORDER_COLOR_PALETTE.length);
  if (!cycle) return base;
  return mixHex(base, cycle % 2 === 1 ? "#0e0f11" : "#fbfbfc", Math.min(0.22 * cycle, 0.45));
}

function hexToRgb(hex) {
  const value = String(hex || "").replace("#", "");
  const full = value.length === 3 ? value.split("").map((c) => c + c).join("") : value;
  const num = parseInt(full, 16);
  if (Number.isNaN(num)) return { r: 122, g: 135, b: 149 };
  return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
}

// 两色按比例混合（t=0 取 a，t=1 取 b），用于甘特状态质感与泳道/图例派生色
function mixHex(a, b, t) {
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  const mix = (x, y) => Math.round(x + (y - x) * Math.max(0, Math.min(1, t)));
  return `#${[mix(ca.r, cb.r), mix(ca.g, cb.g), mix(ca.b, cb.b)].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

// 甘特条块底色：订单标识色 × 状态质感（未来=实心，进行中=斜纹，已完成=降饱和）
function ganttStatusBackground(orderColor, status) {
  if (status === "processing") {
    const light = mixHex(orderColor, "#ffffff", 0.34);
    return `repeating-linear-gradient(135deg, ${orderColor}, ${orderColor} 7px, ${light} 7px, ${light} 13px)`;
  }
  if (status === "completed") return mixHex(orderColor, "#8a94a6", 0.55);
  return orderColor;
}

function graphPolygonPoints(sides, r, rotateDeg = -90) {
  return Array.from({ length: sides }, (_, i) => {
    const angle = ((rotateDeg + (360 * i) / sides) * Math.PI) / 180;
    return `${(r * Math.cos(angle)).toFixed(2)},${(r * Math.sin(angle)).toFixed(2)}`;
  }).join(" ");
}

// 类型形状：订单=圆角矩形、任务=菱形、工序=圆形、机器=六边形、工装=三角、人员=五边形
function graphNodeShapeSVG(type, r, attrs = "") {
  switch (String(type || "").toLowerCase()) {
    case "order":
      return `<rect x="${(-r * 1.25).toFixed(1)}" y="${(-r * 0.85).toFixed(1)}" width="${(r * 2.5).toFixed(1)}" height="${(r * 1.7).toFixed(1)}" rx="${(r * 0.42).toFixed(1)}" ${attrs}></rect>`;
    case "task":
      return `<polygon points="0,${-r} ${(r * 1.2).toFixed(1)},0 0,${r} ${(-r * 1.2).toFixed(1)},0" ${attrs}></polygon>`;
    case "machine":
      return `<polygon points="${graphPolygonPoints(6, r, 0)}" ${attrs}></polygon>`;
    case "tooling":
      return `<polygon points="${graphPolygonPoints(3, r, -90)}" ${attrs}></polygon>`;
    case "personnel":
      return `<polygon points="${graphPolygonPoints(5, r, -90)}" ${attrs}></polygon>`;
    default:
      return `<circle r="${r}" ${attrs}></circle>`;
  }
}

function graphTypeLabel(type) {
  return GRAPH_TYPE_LABELS[String(type || "").toLowerCase()] || GRAPH_TYPE_LABELS.other;
}

function graphEdgeTypeLabel(edgeType) {
  return GRAPH_EDGE_TYPE_LABELS[String(edgeType || "").toLowerCase()] || humanizeCodeLabel(edgeType || "-");
}

function graphEdgeGroupForType(type) {
  const value = String(type || "").toLowerCase();
  if (["order_has_task", "task_predecessor", "task_has_operation", "operation_sequence", "op_depends_task"].includes(value)) return "structure";
  if (["machine_eligible", "tooling_eligible", "personnel_eligible"].includes(value)) return "resource";
  return "other";
}

function ensureGraphViewState(nodes = []) {
  const next = app.graphView || defaultGraphView();
  const nodeTypes = { ...defaultGraphView().nodeTypes, ...(next.nodeTypes || {}) };
  nodes.forEach((node) => {
    const type = String(node.node_type || node.type || "other").toLowerCase();
    if (!(type in nodeTypes)) nodeTypes[type] = true;
  });
  app.graphView = {
    ...defaultGraphView(),
    ...next,
    nodeTypes,
    edgeGroups: { ...defaultGraphView().edgeGroups, ...(next.edgeGroups || {}) },
    positions: { ...(next.positions || {}) },
  };
}

function resetGraphView(options = {}) {
  const preserveFilters = options.preserveFilters !== false;
  const next = defaultGraphView();
  if (preserveFilters && app.graphView) {
    next.mode = app.graphView.mode || next.mode;
    next.search = app.graphView.search || "";
    next.nodeTypes = { ...next.nodeTypes, ...(app.graphView.nodeTypes || {}) };
    next.edgeGroups = { ...next.edgeGroups, ...(app.graphView.edgeGroups || {}) };
  }
  app.graphView = next;
}

function normalizeGraphNode(node) {
  const id = node.node_id || node.id;
  const type = String(node.node_type || node.type || "other").toLowerCase();
  const attrs = node.attrs || {};
  const entityId = node.entity_id || attrs.entity_id || entityIdFromGraphId(id);
  return {
    ...attrs,
    ...node,
    attrs,
    id,
    type,
    entity_id: entityId,
    label: node.label || attrs.label || node.name || attrs.name || entityId || id,
    order_id: node.order_id ?? attrs.order_id ?? null,
    task_id: node.task_id ?? attrs.task_id ?? null,
    process_type: node.process_type ?? attrs.process_type ?? null,
    machine_id: node.machine_id ?? attrs.machine_id ?? null,
    tooling_id: node.tooling_id ?? attrs.tooling_id ?? null,
    personnel_id: node.personnel_id ?? attrs.personnel_id ?? attrs.person_id ?? null,
  };
}

function normalizeGraphEdge(edge) {
  const source = edge.src || edge.source;
  const target = edge.dst || edge.target;
  const edgeType = edge.edge_type || edge.type || "unknown";
  const attrs = edge.attrs || {};
  return { ...attrs, ...edge, attrs, source, target, edgeType, group: graphEdgeGroupForType(edgeType) };
}

function graphNodeMatchesSearch(node, term) {
  if (!term) return true;
  const haystack = [
    node.id,
    node.label,
    node.type,
    node.order_id,
    node.task_id,
    node.operation_id,
    node.machine_id,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(term);
}

function isResourceNodeType(type) {
  return ["machine", "tooling", "personnel"].includes(String(type || "").toLowerCase());
}

function buildOrderClusterContext(nodes, edges) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const orderEntityToNodeId = new Map();
  const taskToOrderNodeId = new Map();
  const opToOrderNodeId = new Map();

  nodes.forEach((node) => {
    if (node.type === "order") {
      orderEntityToNodeId.set(entityIdFromGraphId(node.id), node.id);
    }
  });

  nodes.forEach((node) => {
    if (node.type === "task") {
      const orderNodeId = orderEntityToNodeId.get(String(node.order_id || ""));
      if (orderNodeId) {
        // 节点 ID（T:xxx）与实体 ID（xxx）都建立映射：工序节点 attrs.task_id 存的是实体 ID
        taskToOrderNodeId.set(node.id, orderNodeId);
        if (node.entity_id) taskToOrderNodeId.set(node.entity_id, orderNodeId);
      }
    }
  });

  nodes.forEach((node) => {
    if (node.type === "operation") {
      const orderNodeId = orderEntityToNodeId.get(String(node.order_id || "")) || taskToOrderNodeId.get(String(node.task_id || ""));
      if (orderNodeId) opToOrderNodeId.set(node.id, orderNodeId);
    }
  });

  const orderClusters = new Map();
  nodes.forEach((node) => {
    let orderNodeId = null;
    if (node.type === "order") orderNodeId = node.id;
    else if (node.type === "task") orderNodeId = taskToOrderNodeId.get(node.id) || null;
    else if (node.type === "operation") orderNodeId = opToOrderNodeId.get(node.id) || null;
    if (!orderNodeId) return;
    if (!orderClusters.has(orderNodeId)) orderClusters.set(orderNodeId, new Set([orderNodeId]));
    orderClusters.get(orderNodeId).add(node.id);
  });

  edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return;
    const orderNodeId = opToOrderNodeId.get(source.id) || opToOrderNodeId.get(target.id) || taskToOrderNodeId.get(source.id) || taskToOrderNodeId.get(target.id);
    if (!orderNodeId || !orderClusters.has(orderNodeId)) return;
    if (isResourceNodeType(source.type) || isResourceNodeType(target.type)) {
      orderClusters.get(orderNodeId).add(source.id);
      orderClusters.get(orderNodeId).add(target.id);
    }
  });

  const ordersFromNode = (nodeId) => {
    const node = nodeMap.get(nodeId);
    if (!node) return [];
    if (node.type === "order") return [node.id];
    if (node.type === "task") return taskToOrderNodeId.get(node.id) ? [taskToOrderNodeId.get(node.id)] : [];
    if (node.type === "operation") return opToOrderNodeId.get(node.id) ? [opToOrderNodeId.get(node.id)] : [];
    if (!isResourceNodeType(node.type)) return [];
    const result = new Set();
    edges.forEach((edge) => {
      if (edge.source !== node.id && edge.target !== node.id) return;
      const otherId = edge.source === node.id ? edge.target : edge.source;
      const orderNodeId = opToOrderNodeId.get(otherId) || taskToOrderNodeId.get(otherId);
      if (orderNodeId) result.add(orderNodeId);
    });
    return Array.from(result);
  };

  return {
    nodeMap,
    orderClusters,
    ordersFromNode,
  };
}

function buildOrderScopedNodeSet(nodes, edges, selectedId, matchedIds, nodeLimit, maxOrders = 6) {
  const { orderClusters, ordersFromNode } = buildOrderClusterContext(nodes, edges);

  const orderedIds = [];
  const pushOrder = (orderId) => {
    if (!orderId || orderedIds.includes(orderId) || !orderClusters.has(orderId)) return;
    orderedIds.push(orderId);
  };

  ordersFromNode(selectedId).forEach(pushOrder);
  matchedIds.forEach((id) => ordersFromNode(id).forEach(pushOrder));
  nodes.filter((node) => node.type === "order").forEach((node) => pushOrder(node.id));

  if (!orderedIds.length) return null;

  const keepIds = new Set();
  let keptOrders = 0;
  for (const orderId of orderedIds) {
    const cluster = orderClusters.get(orderId);
    if (!cluster?.size) continue;
    if (keptOrders >= Math.max(1, maxOrders || 1)) break;
    if (keepIds.size && keepIds.size + cluster.size > nodeLimit) break;
    cluster.forEach((id) => keepIds.add(id));
    keptOrders += 1;
    if (keepIds.size >= nodeLimit) break;
  }

  if (!keepIds.size && orderedIds[0]) {
    orderClusters.get(orderedIds[0])?.forEach((id) => keepIds.add(id));
  }

  return keepIds.size ? keepIds : null;
}

function graphNodeDetailRows(node, relatedEdges = []) {
  if (!node) return [];
  const rows = [
    { label: "节点 ID", value: escapeHtml(node.id || "-") },
    { label: "实体 ID", value: escapeHtml(node.entity_id || "-") },
    { label: "节点类型", value: escapeHtml(graphTypeLabel(node.type)) },
    { label: "显示标签", value: escapeHtml(node.label || "-") },
    { label: "关联关系", value: formatInt(relatedEdges.length) },
  ];

  const preferred = [
    ["order_id", "订单"],
    ["task_id", "任务"],
    ["process_type", "工艺类型"],
    ["type_name", "资源类型"],
    ["status", "状态"],
    ["priority", "优先级"],
    ["release_at", "释放时间"],
    ["due_at", "交期"],
    ["derived_start_at", "理想最晚开始"],
    ["derived_due_at", "理想最晚完成"],
    ["critical_slack", "关键余量"],
    ["critical_path_time", "关键路径时长"],
    ["processing_time", "标准工时"],
    ["required_tooling_types", "工装需求"],
    ["required_personnel_skills", "人员技能"],
    ["is_critical", "关键资源"],
    ["skills", "技能"],
  ];

  preferred.forEach(([key, label]) => {
    const value = node[key];
    if (value === null || value === undefined || value === "") return;
    if (Array.isArray(value) && !value.length) return;
    rows.push({
      label,
      value: escapeHtml(Array.isArray(value) ? value.join(", ") : String(value)),
    });
  });

  return rows;
}

// 层级泳道布局：订单→任务→工序→机器/工装/人员 六条纵向泳道（左冷右暖），
// 列内按"子列网格"紧凑排布，避免单列过高；返回泳道几何供背景渲染。
const GRAPH_LANE_CONFIG = {
  order: { rows: 12, rowGap: 86, colGap: 132 },
  task: { rows: 16, rowGap: 74, colGap: 128 },
  operation: { rows: 22, rowGap: 54, colGap: 92 },
  machine: { rows: 16, rowGap: 74, colGap: 128 },
  tooling: { rows: 14, rowGap: 74, colGap: 128 },
  personnel: { rows: 14, rowGap: 74, colGap: 128 },
  other: { rows: 14, rowGap: 74, colGap: 128 },
};
const GRAPH_LANE_PAD_X = 24;
const GRAPH_LANE_GAP = 30;
const GRAPH_LANE_TOP = 60;
const GRAPH_LANE_OUTER_PAD = 24;

function graphLaneFamily(type) {
  if (["order", "task", "operation"].includes(type)) return "plan";
  if (isResourceNodeType(type)) return "resource";
  return "other";
}

function layoutGraph(nodes, edges, selectedId) {
  const groups = new Map();
  GRAPH_NODE_ORDER.forEach((type) => groups.set(type, []));
  nodes.forEach((node) => {
    if (!groups.has(node.type)) groups.set(node.type, []);
    groups.get(node.type).push(node);
  });

  const degree = new Map(nodes.map((node) => [node.id, 0]));
  edges.forEach((edge) => {
    degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
  });

  const directNeighbors = new Set([selectedId]);
  edges.forEach((edge) => {
    if (edge.source === selectedId || edge.target === selectedId) {
      directNeighbors.add(edge.source);
      directNeighbors.add(edge.target);
    }
  });

  const usedGroups = Array.from(groups.entries()).filter(([, items]) => items.length);
  const placed = new Map();
  const lanes = [];
  let laneX = GRAPH_LANE_OUTER_PAD;
  let maxContentBottom = GRAPH_LANE_TOP + 40;

  usedGroups.forEach(([type, items]) => {
    const cfg = GRAPH_LANE_CONFIG[type] || GRAPH_LANE_CONFIG.other;
    items.sort((a, b) => {
      const aRank = a.id === selectedId ? 0 : directNeighbors.has(a.id) ? 1 : 2;
      const bRank = b.id === selectedId ? 0 : directNeighbors.has(b.id) ? 1 : 2;
      if (aRank !== bRank) return aRank - bRank;
      const degreeGap = (degree.get(b.id) || 0) - (degree.get(a.id) || 0);
      if (degreeGap !== 0) return degreeGap;
      return String(a.label).localeCompare(String(b.label), "zh-CN");
    });
    const subCols = Math.max(1, Math.ceil(items.length / cfg.rows));
    const usedRows = Math.max(1, Math.ceil(items.length / subCols));
    maxContentBottom = Math.max(maxContentBottom, GRAPH_LANE_TOP + 40 + (usedRows - 1) * cfg.rowGap);
    const laneWidth = GRAPH_LANE_PAD_X * 2 + (subCols - 1) * cfg.colGap + 64;
    items.forEach((node, index) => {
      const col = Math.floor(index / usedRows);
      const row = index % usedRows;
      placed.set(node.id, {
        x: laneX + GRAPH_LANE_PAD_X + 32 + col * cfg.colGap,
        y: GRAPH_LANE_TOP + 40 + row * cfg.rowGap,
        r: GRAPH_NODE_SIZES[node.type] || GRAPH_NODE_SIZES.other,
        node,
        type,
      });
    });
    lanes.push({
      type,
      family: graphLaneFamily(type),
      label: graphTypeLabel(type),
      count: items.length,
      x: laneX,
      width: laneWidth,
    });
    laneX += laneWidth + GRAPH_LANE_GAP;
  });

  const width = Math.max(960, laneX - GRAPH_LANE_GAP + GRAPH_LANE_OUTER_PAD);
  const height = Math.max(460, maxContentBottom + 96);
  lanes.forEach((lane) => {
    lane.y = GRAPH_LANE_TOP - 12;
    lane.height = height - GRAPH_LANE_TOP + 12 - 16;
  });

  // 泳道族（规划链/资源）连续区段，用于顶部族标
  const families = [];
  lanes.forEach((lane) => {
    const last = families[families.length - 1];
    if (last && last.family === lane.family) {
      last.width = lane.x + lane.width - last.x;
    } else {
      families.push({ family: lane.family, x: lane.x, width: lane.width });
    }
  });

  return { width, height, placed, lanes, families };
}

// 边几何：跨列走直线（按节点半径做边界吸附），同列顺序边向右弓形绕开节点
function graphEdgePathD(a, b) {
  const ra = a.r || 18;
  const rb = b.r || 18;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (Math.abs(dx) < 48) {
    const dir = dy >= 0 ? 1 : -1;
    const bow = 68;
    const sy = a.y + dir * ra;
    const ty = b.y - dir * rb;
    return `M ${a.x} ${sy} C ${a.x + bow} ${sy}, ${b.x + bow} ${ty}, ${b.x} ${ty}`;
  }
  const dir = dx > 0 ? 1 : -1;
  return `M ${a.x + dir * ra} ${a.y} L ${b.x - dir * rb} ${b.y}`;
}

function graphViewportTransform() {
  return `translate(${app.graphView.panX} ${app.graphView.panY}) scale(${app.graphView.zoom})`;
}

function applyGraphViewportState(root) {
  const container = root || document;
  const viewport = container.querySelector("[data-graph-viewport]");
  if (viewport) viewport.setAttribute("transform", graphViewportTransform());
  const badge = container.querySelector("[data-graph-zoom]");
  if (badge) badge.textContent = `${Math.round(app.graphView.zoom * 100)}%`;
}

function applyGraphNodePositions(root) {
  const container = root || document;
  const positions = app.graphView.positions || {};
  container.querySelectorAll("[data-graph-node]").forEach((nodeEl) => {
    const pos = positions[nodeEl.dataset.graphNode];
    if (!pos) return;
    nodeEl.setAttribute("transform", `translate(${pos.x} ${pos.y})`);
  });
  container.querySelectorAll("[data-graph-link]").forEach((pathEl) => {
    const source = positions[pathEl.dataset.source];
    const target = positions[pathEl.dataset.target];
    if (!source || !target) return;
    pathEl.setAttribute("d", graphEdgePathD(source, target));
  });
}

function fitGraphViewport(bounds) {
  if (!bounds) return;
  const width = Math.max(120, (bounds.maxX - bounds.minX) + 120);
  const height = Math.max(120, (bounds.maxY - bounds.minY) + 120);
  const canvasWidth = bounds.canvasWidth || 1220;
  const canvasHeight = bounds.canvasHeight || 640;
  const zoom = Math.max(0.32, Math.min(1.6, Math.min((canvasWidth - 120) / width, (canvasHeight - 120) / height)));
  app.graphView.zoom = zoom;
  app.graphView.panX = 60 - bounds.minX * zoom;
  app.graphView.panY = 80 - bounds.minY * zoom;
}

function renderGraphSection() {
  const graphMetaGrid = app.graphMeta
    ? renderKeyValueGrid([
      { label: "节点", value: formatInt(app.graphMeta.total_nodes) },
      { label: "边", value: formatInt(app.graphMeta.total_edges) },
      { label: "创建时间", value: formatDateTime(app.graphMeta.created_at) },
      { label: "节点类型", value: formatInt(Object.keys(app.graphMeta.node_type_counts || {}).length) },
    ], "context-grid cols-4")
    : graphBuildIsRunning()
      ? ""
      : renderEmptyState("尚未构建图谱", "数据强校验通过后会自动构建图谱，也可以点击下方按钮手动构建。");
  return `
    <div class="workflow-stage-stack">
      <article class="surface-card">
        <div class="card-head"><h3>图谱构建</h3></div>
        ${renderGraphBuildStatus()}
        ${graphMetaGrid}
        <div class="form-actions form-actions--gap">
          <button class="btn btn-primary" type="button" data-action="build-graph">构建图谱</button>
        </div>
      </article>
      ${app.graphMeta ? renderInteractiveGraph() : ""}
    </div>
  `;
}

function renderWorkflowStep3() {
  const simMetrics = app.simResult?.metrics || {};
  const simulationPanel = `
    <article class="surface-card">
      <div class="card-head"><h3>规则仿真</h3><p>先用规则基线验证数据、班次、停机和初始在制状态是否合理，再决定是否进入优化。</p></div>
      <div class="field-inline">
        <span>规则</span>
        <select id="workflow-sim-rule">
          ${CONFIG.HEURISTIC_RULES.map((rule) => `<option value="${rule}" ${rule === app.simRule ? "selected" : ""}>${rule}</option>`).join("")}
        </select>
      </div>
      <div class="form-actions form-actions--gap">
        <button class="btn btn-primary" type="button" data-action="run-simulate">运行仿真</button>
      </div>
      <div id="sim-status">${renderSimStatusInner(app.simStatus)}</div>
      ${app.simResult && simMetrics.feasible === false ? `
        <div class="sim-infeasible-banner" role="alert">
          <strong>仿真结果不完整，下方指标不可用于决策</strong>
          <span>${escapeHtml(app.simResult.diagnosis || `仅完成 ${formatInt(simMetrics.completed_operations)} / ${formatInt(simMetrics.total_operations)} 道工序，请到“数据导入”页运行数据校验。`)}</span>
          ${renderInfeasibleDetail(app.simResult.diagnosis_detail)}
          <div class="form-actions"><button class="btn btn-secondary" type="button" data-nav-jump="new-scene">去查看校验结果</button></div>
        </div>
      ` : ""}
      ${app.simResult ? renderKeyValueGrid([
        { label: "总延误", value: formatDurationHours(simMetrics.total_tardiness) },
        { label: "总周期", value: formatDurationHours(simMetrics.makespan) },
        { label: "净可用利用率", value: formatPercent(simMetrics.avg_net_available_utilization) },
        { label: "关键资源净可用利用率", value: formatPercent(simMetrics.critical_net_available_utilization) },
        { label: "总等待时间", value: formatDurationHours(simMetrics.total_wait_time) },
        { label: "完成工序", value: `${formatInt(simMetrics.completed_operations)} / ${formatInt(simMetrics.total_operations)}` },
      ]) : renderEmptyState("尚未运行仿真", "运行一次规则仿真后，这里会展示完整的时间轴、状态分布和停机遮罩。")}
    </article>
  `;

  const simulationDetail = app.simResult ? `
    ${renderTimeline(app.simResult.gantt, {
      title: `规则仿真甘特图 · ${app.simRule}`,
      canvasId: "gantt-sim",
      solutionIds: [app.simResult.solution_id || `RULE:${app.simRule}`],
    })}
    <article class="surface-card">
      <div class="card-head"><h3>仿真明细预览</h3><p>核查开始/结束时间、状态、订单和资源分配是否符合业务直觉。</p></div>
      ${renderSimpleTable(
        ["工序", "订单", "机器", "状态", "开始", "结束"],
        asArray(app.simResult.gantt).slice(0, 20).map((item) => [
          escapeHtml(item.op_id || "-"),
          escapeHtml(item.order_name || item.order_id || "-"),
          escapeHtml(item.machine_name || item.machine_id || "-"),
          statusChip(item.status_label || (normalizeScheduleStatus(item.status) === "completed" ? "已完成" : normalizeScheduleStatus(item.status) === "processing" ? "进行中" : "未来排产"), normalizeScheduleStatus(item.status) === "completed" ? "info" : normalizeScheduleStatus(item.status) === "processing" ? "warning" : "success"),
          escapeHtml(formatDateTime(item.start_at || offsetToDateTime(item.start))),
          escapeHtml(formatDateTime(item.end_at || offsetToDateTime(item.end))),
        ]),
        { footer: `当前仅展示前 ${Math.min(20, asArray(app.simResult.gantt).length)} 条工序记录。` },
      )}
    </article>
  ` : "";

  return `
    <div class="workflow-stage-stack">
      ${simulationPanel}
      ${simulationDetail}
    </div>
  `;
}

function renderObjectiveSelectors() {
  const catalog = asArray(app.optimizeObjectiveCatalog);
  return `
    <div class="objective-grid">
      ${catalog.map((item) => `
        <label class="objective-pill">
          <input type="checkbox" data-objective-key="${escapeHtml(item.key)}" ${app.optimizeForm.objectiveKeys.includes(item.key) ? "checked" : ""}>
          <strong>${escapeHtml(item.label || getObjectiveLabel(item.key))}</strong>
          <span>${escapeHtml(item.description || "")}</span>
        </label>
      `).join("")}
    </div>
  `;
}

function renderWorkflowStep4() {
  const status = app.optimizeStatus;
  const result = app.optimizeResult;
  const recommendedBudget = refreshOptimizeBudgetRecommendation({ preserveManual: true });
  return `
    <article class="surface-card">
      <div class="card-head"><h3>多目标优化配置</h3><p>前期做近似广搜，后期做精确精修，最终重算 Pareto 前沿。</p></div>
      ${renderObjectiveSelectors()}
      <div class="form-grid">
        <label><span>目标方案数</span><input id="opt-target-count" type="number" min="2" value="${app.optimizeForm.targetSolutionCount}"></label>
        <label><span>总预算秒数</span><input id="opt-time-limit" type="number" min="5" value="${app.optimizeForm.timeLimitS}"></label>
        <label><span>种群规模</span><input id="opt-population" type="number" min="4" value="${app.optimizeForm.populationSize}"></label>
        <label><span>代数</span><input id="opt-generations" type="number" min="1" value="${app.optimizeForm.generations}"></label>
        <label><span>前期时间占比</span><input id="opt-coarse-ratio" type="number" min="0.2" max="0.95" step="0.05" value="${app.optimizeForm.coarseTimeRatio}"></label>
        <label><span>精修轮数</span><input id="opt-refine-rounds" type="number" min="1" value="${app.optimizeForm.refineRounds}"></label>
        <label><span>ALNS 强度</span><input id="opt-alns-aggression" type="number" min="0.5" max="3" step="0.1" value="${app.optimizeForm.alnsAggression}"></label>
        <label><span>基线规则</span>
          <select id="opt-baseline-rule">
            ${CONFIG.HEURISTIC_RULES.map((rule) => `<option value="${rule}" ${rule === app.optimizeForm.baselineRuleName ? "selected" : ""}>${rule}</option>`).join("")}
          </select>
        </label>
      </div>
      <div class="subtle-note" id="opt-budget-hint">${app.optimizeForm.timeLimitTouched
        ? `建议约 ${recommendedBudget} 秒。当前保留手动值 ${app.optimizeForm.timeLimitS} 秒，可随时恢复建议值。`
        : `已按当前规模与参数自动推荐 ${recommendedBudget} 秒，可继续手动修改。`}</div>
      <div class="form-actions">
        <button class="btn btn-primary" type="button" data-action="start-hybrid-optimize" ${optimizeIsRunning() ? "disabled aria-busy=\"true\"" : ""}>${optimizeIsRunning() ? "优化运行中…" : "启动优化"}</button>
        <button class="btn btn-secondary" type="button" data-action="apply-budget-recommendation">采用建议预算</button>
        <button class="btn btn-ghost" type="button" data-nav-jump="pareto-library">进入方案库</button>
      </div>
    </article>
    ${renderOptimizeStatus()}
    ${(status || result) && !["error", "failed"].includes(String(status?.status || "").toLowerCase()) ? `
      <article class="surface-card">
        <div class="card-head"><h3>优化进度与结果</h3><p>自动显示近似评估、精确评估与候选池规模。</p></div>
        ${renderKpiCards([
          { label: "状态", value: escapeHtml(status?.status || result?.status || "unknown"), hint: "实时轮询状态" },
          { label: "近似评估", value: formatInt(status?.approximate_evaluations || result?.approximate_evaluations || 0), hint: "前期广搜吞吐" },
          { label: "精确评估", value: formatInt(status?.exact_evaluations || result?.exact_evaluations || 0), hint: "后期高质量验证" },
          { label: "候选方案", value: formatInt(result?.found_solution_count || result?.solutions?.length || 0), hint: `已请求 ${formatInt(app.optimizeForm.targetSolutionCount)} 个` },
        ])}
      </article>
    ` : ""}
  `;
}

function renderWorkflow() {
  const container = el("workflow-content");
  let html = "";
  if (app.workflowStep === 3) html = renderWorkflowStep3();
  if (app.workflowStep === 4) html = renderWorkflowStep4();
  container.innerHTML = html;
  requestAnimationFrame(() => mountGantts());
}

function renderNewScene() {
  const validationBox = el("new-scene-validation");
  if (validationBox) validationBox.innerHTML = app.currentScene ? renderValidationPanel() : "";
  const graphBox = el("new-scene-graph");
  if (graphBox) graphBox.innerHTML = app.currentScene ? renderGraphSection() : "";
  if (app.currentScene && app.graphMeta) requestAnimationFrame(() => mountInteractiveGraph());
}

function renderValidationPanel() {
  const validation = app.validation;
  if (app.validationBusy) {
    return `
      <article class="surface-card validation-panel">
        <div class="card-head"><h3>数据强校验</h3><p>正在对数据完整性、关联关系和约束条件做后台校验…</p></div>
        <div class="import-progress-track indeterminate"><i></i></div>
      </article>
    `;
  }
  if (!validation) {
    return `
      <article class="surface-card validation-panel">
        <div class="card-head"><h3>数据强校验</h3><p>导入或生成实例后会自动校验；也可以手动重新运行。</p></div>
        <div class="form-actions"><button class="btn btn-primary" type="button" data-action="run-validation">运行数据校验</button></div>
      </article>
    `;
  }
  const failed = validation.status === "failed";
  const tone = failed ? "danger" : validation.status === "warning" ? "warning" : "success";
  const label = failed ? "校验未通过" : validation.status === "warning" ? "校验通过（有警告）" : "校验通过";
  const issues = [...asArray(validation.errors), ...asArray(validation.warnings)];
  const collapsed = !!app.validationCollapsed;
  return `
    <article class="surface-card validation-panel ${tone} ${collapsed ? "is-collapsed" : ""}">
      <div class="card-head">
        <div class="validation-head-main">
          <h3>数据强校验</h3>
          ${statusChip(label, tone === "danger" ? "danger" : tone)}
        </div>
        ${renderCollapseButton("toggle-validation-collapse", collapsed, "数据强校验详情")}
      </div>
      <div class="validation-body collapsible-body">
        <p class="subtle-note">覆盖数据完整性、关联关系与约束条件；错误级问题会导致仿真/优化静默失败。</p>
        ${renderKeyValueGrid([
          { label: "错误", value: formatInt(validation.error_count || 0) },
          { label: "警告", value: formatInt(validation.warning_count || 0) },
          { label: "校验时间", value: formatDateTime(validation.checked_at) },
          { label: "日历天数", value: `${formatInt(validation.stats?.calendar?.final_days || 0)} 天` },
        ])}
        ${issues.length ? renderSimpleTable(
          ["级别", "问题 Sheet", "类别", "实体", "问题明细"],
          issues.slice(0, 50).map((item) => [
            statusChip(item.severity === "error" ? "错误" : "警告", item.severity === "error" ? "danger" : "warning"),
            escapeHtml(item.sheet || "-"),
            escapeHtml(item.category || "-"),
            escapeHtml(item.entity || "-"),
            escapeHtml(item.message || "-"),
          ]),
          { footer: issues.length > 50 ? `共 ${issues.length} 条问题，仅展示前 50 条，导出 Excel 可查看全部。` : `共 ${issues.length} 条问题。` },
        ) : '<div class="subtle-note">未发现脏数据或格式问题，可以进入仿真与优化。</div>'}
        <div class="form-actions">
          <button class="btn btn-ghost" type="button" data-action="run-validation">重新校验</button>
          ${issues.length ? '<button class="btn btn-ghost" type="button" data-action="export-validation">导出校验结果 Excel</button>' : ""}
          ${failed ? '<span class="subtle-note">请先修复上述错误（可在下方各标签页直接编辑数据），否则仿真指标会显示为 0。</span>' : ""}
        </div>
      </div>
    </article>
  `;
}

const HEURISTIC_RULE_BLURB = {
  ATC: "综合考虑交期紧迫度与处理时间，平衡延误与利用率",
  EDD: "优先安排交期最早的任务，降低最大延误",
  SPT: "优先处理耗时最短的工序，缩短平均周期",
  CR: "按交期剩余时间与剩余加工时间之比排序",
  FIFO: "按到达顺序先到先做，规则简单稳健",
  LPT: "优先处理长工序，规避长尾延误与资源空闲",
};

function objectiveDirection(key) {
  const fromResult = asArray(app.optimizeResult?.objective_catalog).find((item) => item.key === key)?.direction;
  if (fromResult) return fromResult;
  return asArray(app.optimizeObjectiveCatalog).find((item) => item.key === key)?.direction || null;
}

// 该 KPI 列在 preview 各方案中的最优数值；方向未知（目录未覆盖）则不判优，返回 null。
function bestMetricValue(preview, key) {
  const direction = objectiveDirection(key);
  if (!direction) return null;
  const values = preview.map((item) => metricValue(item, key)).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value)));
  if (!values.length) return null;
  const nums = values.map(Number);
  return direction === "max" ? Math.max(...nums) : Math.min(...nums);
}

// 方案对比表：单一决策工作台。左列勾选（共享上限 4），中列方案名+来源·模式副标题，
// 随后主目标 KPI + 其他 KPI（每列最佳值加粗），末列逐行操作。聚焦方案置顶，勾选行按方案色高亮。
function renderReviewCandidateComparison() {
  const candidates = getReviewCandidates();
  if (!candidates.length) return "";
  const primaryKeys = activePrimaryObjectiveKeys();
  const extraKeys = REVIEW_KPI_KEYS.filter((key) => !primaryKeys.includes(key));
  const allKeys = [...primaryKeys, ...extraKeys];
  // 聚焦方案置顶，其余保持原顺序（基线 → 方案 → 参照 → 精确冠军）
  const focusedId = app.reviewDetailId;
  const ordered = focusedId && candidates.some((item) => item.id === focusedId)
    ? [candidates.find((item) => item.id === focusedId), ...candidates.filter((item) => item.id !== focusedId)]
    : candidates;
  const bestByKey = Object.fromEntries(allKeys.map((key) => [key, bestMetricValue(candidates, key)]));
  const cell = (item, key) => {
    const value = metricValue(item, key);
    const best = bestByKey[key];
    const isBest = best !== null && value !== null && value !== undefined && Number.isFinite(Number(value)) && Number(value) === best;
    return `<td class="${isBest ? "is-best" : ""}">${isBest ? `<strong>${metricDisplay(item, key)}</strong>` : metricDisplay(item, key)}</td>`;
  };
  const rows = ordered.map((item) => {
    const checked = app.reviewSelection.includes(item.id);
    const colorIdx = schemeColorIndex(item.id);
    const rowClass = [checked ? "is-selected" : "", colorIdx >= 0 ? `scheme-c-${colorIdx}` : ""].filter(Boolean).join(" ");
    return `<tr class="${rowClass}">
      <td class="compare-check"><input type="checkbox" data-action="toggle-candidate" data-id="${escapeHtml(item.id)}" ${checked ? "checked" : ""} aria-label="勾选 ${escapeHtml(item.name)}"></td>
      <td class="compare-name">
        <strong>${colorIdx >= 0 ? `<span class="scheme-dot" style="background:${schemeColorToken(colorIdx)}"></span>` : ""}${escapeHtml(item.name)}</strong>
        <small>${escapeHtml(candidateSourceLabel(item))} · ${escapeHtml(candidateModeLabel(item))}</small>
      </td>
      ${allKeys.map((key) => cell(item, key)).join("")}
      <td class="compare-ops">
        <button class="op-btn" type="button" data-action="focus-candidate" data-id="${escapeHtml(item.id)}" title="查看详情" aria-label="查看详情">◎</button>
        <button class="op-btn" type="button" data-action="send-candidate-to-ai" data-id="${escapeHtml(item.id)}" title="送入 AI 评审" aria-label="送入 AI 评审">✦</button>
        <button class="op-btn" type="button" data-action="export-selected-solution" data-id="${escapeHtml(item.id)}" title="导出该方案" aria-label="导出该方案">⬇</button>
      </td>
    </tr>`;
  }).join("");
  const headers = ["选", "方案", ...allKeys.map((key) => getObjectiveLabel(key)), "操作"];
  return `
    <article class="surface-card">
      <div class="card-head">
        <h3>方案对比</h3>
        <p>勾选方案即同时驱动利用率对比、甘特联动与 AI 评审（共享上限 4）；「查看详情」将该行置顶。主目标在前、其他 KPI 在后，每列最佳值加粗，横向滚动查看全部。</p>
      </div>
      ${renderPrimaryObjectiveBadges(primaryKeys)}
      <div class="table-shell">
        <table class="data-table compare-table">
          <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </article>
  `;
}

function renderReviewTypeUtilization() {
  const columns = getSelectedReviewCandidates();
  if (!columns.length) {
    return `
      <article class="surface-card">
        <div class="card-head"><h3>机器分类利用率对比</h3></div>
        <div class="empty-state"><p>请在上方对比表勾选方案，以对比各机台类型的利用率。</p></div>
      </article>
    `;
  }
  const state = app.reviewRead;

  let inner;
  if (state.loading && !state.utilization) {
    inner = `<div class="empty-state"><p>正在计算机器分类利用率…</p></div>`;
  } else if (state.error && !state.utilization) {
    inner = `<div class="empty-state"><p>加载机器分类利用率失败：${escapeHtml(state.error)}</p><button class="btn-ghost" data-action="retry-type-utilization">重试</button></div>`;
  } else {
    const types = asArray(state.utilization?.types).filter((type) => Object.keys(type.per_solution || {}).length);
    if (!types.length) {
      inner = `<div class="empty-state"><p>当前对比方案的排程未覆盖任何机台类型。</p></div>`;
    } else {
      const rows = types.map((type) => {
        const cellUtils = columns.map((c) => type.per_solution?.[c.id]?.utilization).filter((u) => u !== undefined && u !== null);
        const best = cellUtils.length ? Math.max(...cellUtils) : null;
        const cells = columns.map((c) => {
          const entry = type.per_solution?.[c.id];
          if (!entry) return "<td>-</td>";
          const idx = schemeColorIndex(c.id);
          const isBest = best !== null && entry.utilization === best;
          const pct = isBest ? `<strong>${formatPercent(entry.utilization)}</strong>` : formatPercent(entry.utilization);
          const pctWidth = Math.max(0, Math.min(100, Math.round((Number(entry.utilization) || 0) * 100)));
          return `<td class="${isBest ? "is-best" : ""}">${pct}<span class="cell-sub">${formatInt(entry.used_machines)}/${formatInt(type.machines_total)} 台</span><span class="util-bar"><span class="util-bar-fill" style="width:${pctWidth}%;background:${schemeColorToken(idx)}"></span></span></td>`;
        }).join("");
        return `<tr>
          <td><strong>${escapeHtml(type.type_name)}</strong>（${escapeHtml(type.type_id)}）<span class="cell-sub">共 ${formatInt(type.machines_total)} 台${type.is_critical ? " · 关键" : ""}</span></td>
          ${cells}
        </tr>`;
      }).join("");
      inner = `
        <div class="table-shell">
          <table class="data-table util-table">
            <colgroup><col class="util-col-type">${columns.map(() => '<col class="util-col-plan">').join("")}</colgroup>
            <thead><tr><th>机器类型</th>${columns.map((c) => {
              const idx = schemeColorIndex(c.id);
              return `<th><span class="scheme-dot" style="background:${schemeColorToken(idx)}"></span><span class="util-plan-name" title="${escapeHtml(c.name)}">${escapeHtml(c.name)}</span></th>`;
            }).join("")}</tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }
  }
  return `
    <article class="surface-card">
      <div class="card-head">
        <h3>机器分类利用率对比</h3>
        <p>单机利用率 = 有排产时长 / 排产窗口（首个排产开始 ~ 最后排产结束）；按机器类型对有排产机器取算术平均，每行最佳值加粗，小字为该类"有排产 / 总台数"，色条长度即利用率。</p>
      </div>
      ${inner}
    </article>
  `;
}

// 勾选集甘特联动：按机器分组，机器行内每个勾选方案一条子行（vis nestedGroups：机器 → 方案），
// 每方案用勾选集稳定色板，直观对比同一订单在不同方案下各机台的排产差异。
function buildReviewGanttData(selected, schemes, canvasId) {
  const planStartAt = tryParseDate(app.instanceDetails?.plan_start_at);
  const hasRealBase = Boolean(planStartAt);
  const base = hasRealBase ? planStartAt.toISOString() : GANTT_FALLBACK_BASE;

  // 每个方案归一化后的条目，附带其配色序号（= 勾选顺序）
  const schemeRows = selected.map((item, index) => {
    const entries = asArray(schemes[item.id])
      .map((e) => ({
        machineId: e.machine_id || e.machine_name || e.resource_id || "unknown",
        machineName: e.machine_name || e.machine_id || e.resource_name || "未知资源",
        opId: e.op_id || e.operation_id || e.id || "-",
        orderId: e.order_id || "-",
        taskId: e.task_id || "-",
        start: Number(e.start ?? e.start_time ?? 0),
        end: Number(e.end ?? e.end_time ?? 0),
        status: normalizeScheduleStatus(e.status),
      }))
      .filter((e) => !Number.isNaN(e.start) && !Number.isNaN(e.end) && e.end > e.start);
    return { id: item.id, name: item.name, colorIdx: index % SCHEME_COLOR_TOKENS.length, entries };
  }).filter((row) => row.entries.length);

  const all = schemeRows.flatMap((row) => row.entries);
  if (!all.length) return null;

  // 机器行集合（跨方案取并集），按机器名自然排序
  const machineNames = new Map();
  all.forEach((e) => { if (!machineNames.has(e.machineId)) machineNames.set(e.machineId, e.machineName); });
  const machineIds = Array.from(machineNames.keys())
    .sort((a, b) => String(machineNames.get(a)).localeCompare(String(machineNames.get(b)), "zh-CN", { numeric: true }));

  const horizonStart = Math.min(...all.map((e) => e.start));
  const horizonEnd = Math.max(...all.map((e) => e.end));
  const viewKey = JSON.stringify({
    canvasId,
    selectedOrder: app.reviewRead.orderId || "",
    solutionIds: selected.map((item) => item.id).sort(),
    groupMode: "scheme",
  });

  const groups = [];
  const items = [];
  let seq = 0;
  machineIds.forEach((machineId) => {
    const rowsHere = schemeRows.filter((row) => row.entries.some((e) => e.machineId === machineId));
    const nestedIds = rowsHere.map((row) => `rm|${machineId}|s|${row.id}`);
    groups.push({
      id: `rm|${machineId}`,
      content: `${escapeHtml(machineNames.get(machineId))}<span class="gantt-group-count">${formatInt(rowsHere.length)} 方案</span>`,
      className: "gantt-group-machine",
      nestedGroups: nestedIds,
      showNested: true,
      seq: (seq += 1),
    });
    rowsHere.forEach((row) => {
      groups.push({
        id: `rm|${machineId}|s|${row.id}`,
        content: `<span class="scheme-dot scheme-c-${row.colorIdx}"></span><span class="gantt-op-machine">${escapeHtml(row.name)}</span>`,
        className: `gantt-group-op scheme-c-${row.colorIdx}`,
        seq: (seq += 1),
      });
    });
  });

  schemeRows.forEach((row) => {
    row.entries.forEach((e, i) => {
      items.push({
        id: `ri-${row.id}-${i}`,
        group: `rm|${e.machineId}|s|${row.id}`,
        start: ganttOffsetToISO(e.start, base),
        end: ganttOffsetToISO(e.end, base),
        content: escapeHtml(e.opId),
        className: `scheme-c-${row.colorIdx}`,
        title: `${escapeHtml(row.name)} · ${escapeHtml(e.opId)}\n订单:${escapeHtml(e.orderId)} 任务:${escapeHtml(e.taskId)}\n机器:${escapeHtml(e.machineName)}\n${hasRealBase ? `${formatDateTime(ganttOffsetToISO(e.start, base))} ~ ${formatDateTime(ganttOffsetToISO(e.end, base))}` : `相对 ${e.start}h ~ ${e.end}h`}`,
      });
    });
  });

  const nowOffset = ganttProgressNowOffset(all);
  return {
    groups,
    items,
    hasRealBase,
    machineFacet: null,
    ...ganttWindowPayload(horizonStart, horizonEnd, base, nowOffset, viewKey),
    mode: "scheme",
  };
}

function reviewGanttControlsHtml() {
  const selectedOrder = app.reviewRead.orderId;
  if (!selectedOrder) return "";
  const selected = getSelectedReviewCandidates();
  const taskId = app.optimizeResult?.task_id;
  const ids = selected.map((item) => item.id).filter(Boolean);
  return `
    <div class="field-inline">
      <span>订单</span>
      ${renderOrderCombobox({
        id: "gantt-review-compare-order",
        selected: { order_id: selectedOrder, order_name: "" },
        search: async (query) => {
          const result = await reviewDataClient.searchOrders({ taskId, ids, query });
          return result.cancelled ? [] : result.orders;
        },
        select: (order) => loadReviewData(getSelectedReviewCandidates(), order.order_id, false),
      })}
    </div>
  `;
}

function reviewGanttLegendHtml(selected) {
  return selected.map((item, index) => `
    <span class="legend-item"><span class="legend-swatch scheme-c-${index % SCHEME_COLOR_TOKENS.length}"></span>${escapeHtml(item.name)}</span>
  `).join("");
}

function currentReviewGanttData(selected) {
  if (app.reviewRead.loading) return null;
  const taskId = app.optimizeResult?.task_id;
  const ids = selected.map((item) => item.id).filter(Boolean);
  const selectionKey = taskId && ids.length ? ReviewRuntime.selectionKey(taskId, ids) : null;
  if (!selectionKey || app.reviewRead.selectionKey !== selectionKey) return null;
  return buildReviewGanttData(selected, app.reviewRead.schemes, "gantt-review-compare");
}

function reviewGanttStatusHtml(selected, data) {
  const state = app.reviewRead;
  if (state.loading) return `<p class="gantt-note">正在加载勾选方案的排产…</p>`;
  if (state.error) {
    return `<div class="empty-state"><p>加载排产失败：${escapeHtml(state.error)}</p><button class="btn-ghost" data-action="retry-review-gantt">重试</button></div>`;
  }
  const failedNames = asArray(state.failedIds)
    .map((failedId) => selected.find((item) => item.id === failedId)?.name || failedId)
    .filter(Boolean);
  const failedNote = failedNames.length
    ? `<p class="gantt-note">${escapeHtml(failedNames.join("、"))} 在该订单下无可回放的排产，未在下方甘特显示。</p>`
    : "";
  if (data) return failedNote;
  return `${failedNote}<p class="gantt-note">所选方案在当前订单下暂无可展示的排产（参照方案可能不支持排产回放）。</p>`;
}

function renderReviewGantt(selected) {
  const cardHead = `<div class="card-head"><h3>勾选方案甘特联动</h3><p>按机器分组，机器行内每个勾选方案独立配色（与对比表、利用率列头一致）；一次聚焦一个订单，切换订单可比较各方案在该订单上的排产差异。</p></div>`;
  if (!selected.length) {
    return `<article class="surface-card">${cardHead}<div class="empty-state"><p>请在上方对比表勾选方案，以联动甘特对比。</p></div></article>`;
  }
  if (!app.optimizeResult?.task_id) {
    return `<article class="surface-card">${cardHead}<div class="empty-state"><p>运行混合优化后，可在此查看勾选方案的排产甘特联动。</p></div></article>`;
  }
  const data = currentReviewGanttData(selected);
  return `
    <article class="surface-card">
      ${cardHead}
      <div class="review-gantt-controls">${reviewGanttControlsHtml()}</div>
      <div class="legend review-gantt-legend">${reviewGanttLegendHtml(selected)}</div>
      <div class="review-gantt-status">${reviewGanttStatusHtml(selected, data)}</div>
      <div class="gantt-canvas" id="gantt-review-compare"></div>
    </article>
  `;
}

function renderReviewLibraryTab() {
  ensureReviewSelection();
  ensureReferenceSolutions();
  const candidates = getReviewCandidates();
  const selected = getSelectedReviewCandidates();
  const exactCount = candidates.filter((item) => String(item.source || "").includes("exact")).length;
  const heuristicCount = candidates.filter((item) => String(item.source || "").includes("heuristic") || item.heuristicRuleName).length;
  const paretoCount = Math.max(0, candidates.length - exactCount - heuristicCount);
  return `
    <article class="surface-card decision-band decision-band--inline">
      <h3>把 Pareto、启发式和精确冠军放到同一个决策工作台里</h3>
      <div class="decision-band-stats">
        <div><span>Pareto</span><strong>${formatInt(paretoCount)}</strong></div>
        <div><span>启发式</span><strong>${formatInt(heuristicCount)}</strong></div>
        <div><span>精确冠军</span><strong>${formatInt(exactCount)}</strong></div>
        <div><span>已选方案</span><strong id="review-selected-count">${formatInt(selected.length)}</strong></div>
      </div>
    </article>
    <article class="surface-card">
      <div class="card-head"><h3>方案池工具</h3><p>汇总当前方案池，并可加载经典调度规则作为对照锚点，一并纳入比较与 AI 评审。</p></div>
      <div class="two-column">
        <div>
          ${renderPrimaryObjectiveBadges()}
          ${renderKeyValueGrid([
            { label: "总方案数", value: formatInt(candidates.length) },
            { label: "已选方案", value: formatInt(selected.length) },
            { label: "主目标", value: objectiveShortList(app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys) },
            { label: "最近优化", value: app.optimizeResult ? "已完成" : "未运行" },
          ])}
        </div>
        <div>
          <div class="field-inline"><span>添加启发式参照</span></div>
          <p class="pool-hint">教科书基准锚点：衡量优化器相对经典派工规则的提升。已就绪的规则即刻纳入比较与 AI 评审；「未计算」规则首次点击约需 1~2 分钟仿真。</p>
          <div class="chip-row">
            ${CONFIG.HEURISTIC_RULES.map((rule) => {
              const isBaseline = rule === app.optimizeResult?.baseline?.rule_name;
              const computing = app.referenceSolutionsState.computing.includes(rule);
              const cached = !isBaseline && ruleIsCached(rule);
              const cls = isBaseline ? "is-baseline" : computing ? "is-computing" : cached ? "is-ready" : "is-uncached";
              const badge = isBaseline ? "基线" : computing ? "计算中…" : cached ? "" : "未计算";
              const title = isBaseline
                ? "该规则已作为基线方案纳入对比"
                : computing ? "正在仿真计算…"
                : cached ? `${HEURISTIC_RULE_BLURB[rule] || "规则参考方案"}（已纳入对比）`
                : `${HEURISTIC_RULE_BLURB[rule] || "规则参考方案"}（首次计算约需 1~2 分钟）`;
              return `<button type="button" class="chip ${cls}" data-action="load-heuristic-rule" data-rule="${escapeHtml(rule)}" title="${escapeHtml(title)}" ${isBaseline || computing ? "disabled" : ""}>${escapeHtml(rule)}${badge ? `<span class="chip-badge">${escapeHtml(badge)}</span>` : ""}</button>`;
            }).join("")}
          </div>
        </div>
      </div>
    </article>
    <div id="review-comparison-region">
      ${candidates.length ? renderReviewCandidateComparison() : renderEmptyState(
        "暂无方案池",
        "先运行混合优化，或点击上方启发式规则加载参照方案。",
        '<button class="btn btn-primary" type="button" data-nav-jump="optimize-launch">去启动优化</button>',
      )}
    </div>
    <div id="review-utilization-region">${candidates.length ? renderReviewTypeUtilization() : ""}</div>
    <div id="review-gantt-region">${candidates.length ? renderReviewGantt(selected) : ""}</div>
  `;
}

function renderExactObjectiveOptions() {
  return asArray(app.exactObjectiveCatalog)
    .map((item) => `<option value="${escapeHtml(item.key)}" ${item.key === app.exactForm.objectiveKey ? "selected" : ""}>${escapeHtml(item.label || item.key)}</option>`)
    .join("");
}

function renderReviewExactTab() {
  const objectiveOptions = asArray(app.optimizeObjectiveCatalog).map((item) => `
    <label class="objective-pill">
      <span>${escapeHtml(item.label || item.key)}</span>
      <input type="number" min="0" step="0.1" data-weight-key="${escapeHtml(item.key)}" value="${app.exactForm.weights[item.key] ?? (app.optimizeForm.objectiveKeys.includes(item.key) ? 1 : 0)}">
    </label>
  `).join("");
  return `
    <div class="two-column">
      <article class="surface-card">
        <div class="card-head"><h3>单目标精确冠军</h3><p>用于在某一个业务指标上给出高置信冠军方案。</p></div>
        <div class="form-grid">
          <label class="span-2"><span>目标</span><select id="exact-single-objective">${renderExactObjectiveOptions()}</select></label>
          <label><span>时间预算</span><input id="exact-time-limit" type="number" min="5" value="${app.exactForm.timeLimitS}"></label>
        </div>
        <div class="form-actions"><button class="btn btn-primary" type="button" data-action="generate-exact-single">生成单目标冠军</button></div>
      </article>
      <article class="surface-card">
        <div class="card-head"><h3>加权单目标精确冠军</h3><p>用于把业务偏好转成一个加权目标，再生成一个冠军方案。</p></div>
        <div class="objective-grid">${objectiveOptions || renderEmptyState("暂无目标目录", "请先等待目标目录加载完成。")}</div>
        <div class="form-actions"><button class="btn btn-primary" type="button" data-action="generate-exact-weighted">生成加权冠军</button></div>
      </article>
    </div>
    ${app.exactReference ? `
      <article class="surface-card">
        <div class="card-head"><h3>最新精确冠军参考</h3><p>该方案会自动纳入方案库、AI 评审和导出流程。</p></div>
        ${renderKeyValueGrid([
          { label: "方案", value: escapeHtml(app.exactReference.exact_info?.label || app.exactReference.solution_id || "精确冠军参考") },
          { label: "模式", value: escapeHtml(app.exactReference.exact_info?.mode || app.exactReference.source || "-") },
          { label: "总延误", value: metricDisplay(app.exactReference, "total_tardiness") },
          { label: "总周期", value: metricDisplay(app.exactReference, "makespan") },
          { label: "净可用利用率", value: metricDisplay(app.exactReference, "avg_net_available_utilization") },
          { label: "求解信息", value: escapeHtml(app.exactReference.exact_info?.solver_status || app.exactReference.evaluationMode || "-") },
        ])}
      </article>
      ${renderTimeline(app.exactReference.schedule, {
        title: "精确冠军参考甘特图",
        canvasId: "gantt-exact",
        solutionIds: [app.exactReference.solution_id || "EXACT"],
      })}
    ` : ""}
  `;
}

function renderAiConversation() {
  if (!app.aiConversation.length) {
    return `
      <div class="chat-stream">
        <div class="chat-bubble assistant">
          <strong>AI 方案助手</strong>
          <p>请选择 1-4 个方案后，可以让我比较、推荐，或者追问某个方案为什么这样排。</p>
        </div>
      </div>
    `;
  }
  return `
    <div class="chat-stream">
      ${app.aiConversation.map((item) => `
        <div class="chat-bubble ${item.role === "user" ? "user" : "assistant"}">
          <strong>${item.role === "user" ? "你" : "AI 方案助手"}</strong>
          <p>${escapeHtml(item.content)}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function renderReviewAiTab() {
  ensureReviewSelection();
  const selected = getSelectedReviewCandidates();
  const options = getReviewCandidates().map((item) => `
    <option value="${escapeHtml(item.id)}" ${item.id === (app.reviewDetailId || app.aiLastRecommendedId) ? "selected" : ""}>${escapeHtml(item.name)}</option>
  `).join("");
  return `
    <article class="surface-card decision-band">
      <div>
        <span class="eyebrow">AI Decision Copilot</span>
        <h3>让 AI 在主目标、全量 KPI 和风险解释三层口径下给出建议</h3>
        <p>这里不是单纯聊天，而是面向业务评审会的方案协同区。建议先勾选候选，再输入真实业务诉求。</p>
      </div>
      <div class="decision-band-stats">
        <div><span>已纳入 AI</span><strong>${formatInt(selected.length)}</strong></div>
        <div><span>最近推荐</span><strong>${escapeHtml(app.aiLastRecommendedId || "-")}</strong></div>
      </div>
    </article>
    <div class="two-column">
      <article class="surface-card">
        <div class="card-head"><h3>当前纳入 AI 评审的方案</h3><p>AI 默认从主目标 + 全量 KPI + 风险解释三层给出意见。</p></div>
        ${renderPrimaryObjectiveBadges()}
        ${selected.length ? renderSimpleTable(
          ["方案", "来源", "总延误", "总周期", "净可用利用率"],
          selected.map((item) => [
            escapeHtml(item.name),
            escapeHtml(item.source),
            metricDisplay(item, "total_tardiness"),
            metricDisplay(item, "makespan"),
            metricDisplay(item, "avg_net_available_utilization"),
          ]),
        ) : renderEmptyState("暂无已选方案", "先在方案库里勾选 1-4 个方案，再来让 AI 分析。")}
        ${selected.length ? renderCandidateMetricMatrix(selected, "AI 评审输入方案全量指标") : ""}
      </article>
      <article class="surface-card">
        <div class="card-head"><h3>对话式评审</h3><p>输入后会直接在聊天流中回复，并显示“正在分析”反馈。</p></div>
        ${renderAiConversation()}
        <div class="form-grid">
          <label class="span-2"><span>当前追问方案</span><select id="ai-solution-select">${options}</select></label>
          <label class="span-2"><span>问题 / 诉求</span><textarea id="ai-input" rows="4" placeholder="例如：我们更看重主订单准交，其次看净可用利用率，哪个方案更适合？"></textarea></label>
        </div>
        <div class="form-actions">
          <button class="btn btn-ghost" type="button" data-action="ai-compare" ${app.aiBusy ? "disabled" : ""}>比较已勾选方案</button>
          <button class="btn btn-primary" type="button" data-action="ai-recommend" ${app.aiBusy ? "disabled" : ""}>按诉求推荐方案</button>
          <button class="btn btn-ghost" type="button" data-action="ai-ask" ${app.aiBusy ? "disabled" : ""}>追问当前方案</button>
        </div>
      </article>
    </div>
  `;
}

function clearReviewTimeline() {
  const entry = app.ganttInstances.find((item) => item.canvasId === "gantt-review-compare");
  if (!entry) return;
  entry.items.clear();
  entry.groups.clear();
  try { entry.timeline.removeCustomTime("sched-now"); } catch (_) {}
}

function upsertReviewTimeline(data) {
  if (!data || typeof window.vis === "undefined" || typeof window.vis.Timeline !== "function") return;
  const canvasId = "gantt-review-compare";
  const entry = app.ganttInstances.find((item) => item.canvasId === canvasId && item.el?.isConnected);
  if (!entry) {
    app.pendingGantts.set(canvasId, { entries: [], options: { canvasId }, data });
    requestAnimationFrame(() => mountGantts());
    return;
  }
  entry.items.clear();
  entry.groups.clear();
  if (data.groups.length) entry.groups.add(data.groups);
  if (data.items.length) entry.items.add(data.items);
  entry.data = data;
  entry.timeline.setOptions({
    min: data.fullWindow?.start,
    max: data.fullWindow?.end,
  });
  const stored = app.ganttViewWindows[canvasId];
  const selectedWindow = stored?.viewKey === data.viewKey ? stored.window : data.initialWindow;
  if (selectedWindow) {
    entry.timeline.setWindow(selectedWindow.start, selectedWindow.end, { animation: false });
  }
  if (data.nowISO) {
    try {
      entry.timeline.setCustomTime(data.nowISO, "sched-now");
    } catch (_) {
      try { entry.timeline.addCustomTime(data.nowISO, "sched-now"); } catch (_) {}
    }
  } else {
    try { entry.timeline.removeCustomTime("sched-now"); } catch (_) {}
  }
}

function refreshReviewDynamicRegions() {
  const candidates = getReviewCandidates();
  const selected = getSelectedReviewCandidates();
  const comparisonRegion = el("review-comparison-region");
  if (comparisonRegion) {
    comparisonRegion.innerHTML = candidates.length ? renderReviewCandidateComparison() : "";
  }
  const selectedCount = el("review-selected-count");
  if (selectedCount) selectedCount.textContent = formatInt(selected.length);

  const utilizationRegion = el("review-utilization-region");
  if (utilizationRegion) {
    utilizationRegion.innerHTML = candidates.length ? renderReviewTypeUtilization() : "";
  }

  const ganttRegion = el("review-gantt-region");
  if (!ganttRegion) return;
  const existingCanvas = ganttRegion.querySelector("#gantt-review-compare");
  if (!existingCanvas || !selected.length || !app.optimizeResult?.task_id) {
    ganttRegion.innerHTML = candidates.length ? renderReviewGantt(selected) : "";
  } else {
    const data = currentReviewGanttData(selected);
    const controls = ganttRegion.querySelector(".review-gantt-controls");
    const legend = ganttRegion.querySelector(".review-gantt-legend");
    const status = ganttRegion.querySelector(".review-gantt-status");
    if (controls) controls.innerHTML = reviewGanttControlsHtml();
    if (legend) legend.innerHTML = reviewGanttLegendHtml(selected);
    if (status) status.innerHTML = reviewGanttStatusHtml(selected, data);
  }

  const data = currentReviewGanttData(selected);
  if (data) {
    upsertReviewTimeline(data);
  } else if (!app.reviewRead.loading) {
    clearReviewTimeline();
  }
  mountOrderComboboxes();
  requestAnimationFrame(() => mountGantts());
}

function renderReview() {
  const container = el("review-content");
  syncTabButtons("data-review-tab", app.reviewTab);
  if (app.reviewTab === "library") {
    container.innerHTML = renderReviewLibraryTab();
    refreshReviewDynamicRegions();
    ensureReviewData(getSelectedReviewCandidates());
  }
  if (app.reviewTab === "exact") container.innerHTML = renderReviewExactTab();
  if (app.reviewTab === "ai") container.innerHTML = renderReviewAiTab();
  requestAnimationFrame(() => mountGantts());
}

function renderSystem() {
  const container = el("system-content");
  syncTabButtons("data-system-tab", app.systemTab);
  if (app.systemTab === "llm") {
    container.innerHTML = `
      <article class="surface-card">
        <div class="card-head"><h3>大模型连接</h3><p>用于 Pareto 方案比较、推荐和问答解释。</p></div>
        <div class="form-grid">
          <label class="span-2"><span>Base URL</span><input id="llm-base-url" type="text" value="${escapeHtml(app.llmConfig?.base_url || "")}"></label>
          <label class="span-2"><span>模型名称</span><input id="llm-model" type="text" value="${escapeHtml(app.llmConfig?.model || "")}"></label>
          <label class="span-2"><span>API Key</span><input id="llm-api-key" type="password" placeholder="${app.llmConfig?.has_key ? "已配置，如需更新请重新输入" : "请输入新的 API Key"}"></label>
        </div>
        <div class="form-actions">
          <button class="btn btn-primary" type="button" data-action="save-llm-config">保存配置</button>
          <button class="btn btn-ghost" type="button" data-action="test-llm-config">测试连接</button>
        </div>
      </article>
    `;
    return;
  }
  if (app.systemTab === "export") {
    const focused = getSelectedReviewCandidate();
    container.innerHTML = `
      <article class="surface-card">
        <div class="card-head"><h3>导出与交付</h3><p>支持导出模板、CSV 和当前已选方案 Excel。</p></div>
        <div class="form-actions">
          <button class="btn btn-ghost" type="button" data-action="download-template">下载模板</button>
          <button class="btn btn-ghost" type="button" data-action="export-csv">导出当前实例 CSV</button>
          <button class="btn btn-primary" type="button" data-action="export-selected-solution" ${focused ? `data-id="${escapeHtml(focused.id)}"` : ""}>导出当前方案</button>
        </div>
      </article>
    `;
    return;
  }
  container.innerHTML = `
    <article class="surface-card">
      <div class="card-head"><h3>系统状态</h3><p>查看接口连通性与当前工作上下文。</p></div>
      ${renderKeyValueGrid([
        { label: "健康状态", value: escapeHtml(app.health?.status || "未知") },
        { label: "当前页面", value: escapeHtml(app.currentPage) },
        { label: "当前导航", value: escapeHtml(app.currentNav) },
        { label: "当前任务", value: escapeHtml(app.optimizeTaskId || "-") },
      ])}
    </article>
  `;
}


function buildGraphSelectionScopeV2(selectedId, allNodes, allEdges) {
  const nodeMap = new Map(allNodes.map((node) => [node.id, node]));
  const edgesByNode = new Map(allNodes.map((node) => [node.id, []]));
  const outgoingByNode = new Map(allNodes.map((node) => [node.id, []]));
  const incomingByNode = new Map(allNodes.map((node) => [node.id, []]));
  allEdges.forEach((edge) => {
    edgesByNode.get(edge.source)?.push(edge);
    edgesByNode.get(edge.target)?.push(edge);
    outgoingByNode.get(edge.source)?.push(edge);
    incomingByNode.get(edge.target)?.push(edge);
  });

  const clusterContext = buildOrderClusterContext(allNodes, allEdges);
  const selectedNode = nodeMap.get(selectedId) || null;
  const scopeNodeIds = new Set();
  const scopeEdgeIds = new Set();
  const directNeighborIds = new Set();

  const addNode = (id) => {
    if (id && nodeMap.has(id)) scopeNodeIds.add(id);
  };
  const edgeKey = (edge) => `${edge.source}|${edge.target}|${edge.edgeType || edge.group || 'unknown'}`;
  const addEdge = (edge) => {
    if (!edge) return;
    scopeEdgeIds.add(edgeKey(edge));
    addNode(edge.source);
    addNode(edge.target);
  };
  const collectOutgoing = (id, predicate = () => true) => {
    (outgoingByNode.get(id) || []).forEach((edge) => {
      if (!predicate(edge)) return;
      addEdge(edge);
    });
  };
  const collectIncoming = (id, predicate = () => true) => {
    (incomingByNode.get(id) || []).forEach((edge) => {
      if (!predicate(edge)) return;
      addEdge(edge);
    });
  };
  const includeOrder = (orderId) => {
    if (!orderId) return;
    clusterContext.orderClusters.get(orderId)?.forEach((nodeId) => addNode(nodeId));
  };
  const addParentOrdersFor = (nodeId) => {
    clusterContext.ordersFromNode(nodeId).forEach((orderId) => addNode(orderId));
  };
  const includeOperationContext = (opId, options = {}) => {
    const { includeTaskChain = false, includeResourceNeighbors = true } = options;
    addNode(opId);
    collectIncoming(opId, (edge) => edge.edgeType === 'task_has_operation');
    collectIncoming(opId, (edge) => edge.edgeType === 'op_depends_task');
    collectIncoming(opId, (edge) => edge.edgeType === 'operation_sequence');
    collectOutgoing(opId, (edge) => edge.edgeType === 'operation_sequence');
    if (includeResourceNeighbors) {
      collectOutgoing(opId, (edge) => edge.group === 'resource');
    }
    if (includeTaskChain) {
      (incomingByNode.get(opId) || [])
        .filter((edge) => edge.edgeType === 'task_has_operation')
        .forEach((edge) => includeTaskContext(edge.source, { includeSiblingOps: true }));
    }
  };
  const includeTaskContext = (taskId, options = {}) => {
    const { includeSiblingOps = true } = options;
    addNode(taskId);
    addParentOrdersFor(taskId);
    collectIncoming(taskId, (edge) => edge.edgeType === 'task_predecessor');
    collectOutgoing(taskId, (edge) => edge.edgeType === 'task_predecessor');
    const opIds = new Set();
    (outgoingByNode.get(taskId) || []).forEach((edge) => {
      if (edge.edgeType === 'task_has_operation') {
        opIds.add(edge.target);
        addEdge(edge);
      }
      if (edge.edgeType === 'op_depends_task') addEdge(edge);
    });
    if (includeSiblingOps) {
      opIds.forEach((opId) => includeOperationContext(opId, { includeTaskChain: false, includeResourceNeighbors: true }));
    }
  };
  const includeResourceContext = (resourceId) => {
    addNode(resourceId);
    const opIds = new Set();
    collectIncoming(resourceId, (edge) => {
      if (edge.group !== 'resource') return false;
      opIds.add(edge.source);
      return true;
    });
    opIds.forEach((opId) => {
      includeOperationContext(opId, { includeTaskChain: false, includeResourceNeighbors: false });
      collectIncoming(opId, (edge) => edge.edgeType === 'task_has_operation');
      (incomingByNode.get(opId) || [])
        .filter((edge) => edge.edgeType === 'task_has_operation')
        .forEach((edge) => addParentOrdersFor(edge.source));
    });
  };

  addNode(selectedId);
  (edgesByNode.get(selectedId) || []).forEach((edge) => {
    const otherId = edge.source === selectedId ? edge.target : edge.source;
    if (otherId && nodeMap.has(otherId)) directNeighborIds.add(otherId);
  });

  if (!selectedNode) {
    const scopeNodes = [];
    return {
      selectedNode: null,
      nodeMap,
      clusterContext,
      scopeNodeIds,
      scopeNodes,
      scopeEdges: [],
      relatedNodes: [],
      relatedByType: {},
      countsByType: {},
      orderIds: new Set(),
      directNeighborIds,
      edgeGroupCounts: { structure: 0, resource: 0, other: 0 },
      directEdgeGroupCounts: { structure: 0, resource: 0, other: 0 },
      edgeTypeCounts: {},
      upstreamCount: 0,
      downstreamCount: 0,
      upstreamNodeIds: [],
      downstreamNodeIds: [],
      scopeHighlights: [],
    };
  }

  if (selectedNode.type === 'order') {
    const primaryOrders = new Set(clusterContext.ordersFromNode(selectedId));
    primaryOrders.add(selectedId);
    primaryOrders.forEach((orderId) => includeOrder(orderId));
  } else if (selectedNode.type === 'task') {
    includeTaskContext(selectedId, { includeSiblingOps: true });
  } else if (selectedNode.type === 'operation') {
    includeOperationContext(selectedId, { includeTaskChain: false, includeResourceNeighbors: true });
    (incomingByNode.get(selectedId) || [])
      .filter((edge) => edge.edgeType === 'task_has_operation')
      .forEach((edge) => {
        addNode(edge.source);
        addParentOrdersFor(edge.source);
      });
  } else if (isResourceNodeType(selectedNode.type)) {
    includeResourceContext(selectedId);
  } else {
    directNeighborIds.forEach((nodeId) => addNode(nodeId));
    addParentOrdersFor(selectedId);
  }

  directNeighborIds.forEach((nodeId) => addNode(nodeId));

  const scopeEdges = allEdges.filter((edge) => scopeNodeIds.has(edge.source) && scopeNodeIds.has(edge.target));
  scopeEdges.forEach((edge) => scopeEdgeIds.add(edgeKey(edge)));
  const scopeNodes = allNodes.filter((node) => scopeNodeIds.has(node.id));
  const relatedNodes = scopeNodes.filter((node) => node.id !== selectedId);
  const relatedByType = {};
  relatedNodes
    .slice()
    .sort((a, b) => String(a.label || a.id).localeCompare(String(b.label || b.id), 'zh-CN'))
    .forEach((node) => {
      if (!relatedByType[node.type]) relatedByType[node.type] = [];
      relatedByType[node.type].push(node);
    });

  const countsByType = {};
  scopeNodes.forEach((node) => {
    countsByType[node.type] = (countsByType[node.type] || 0) + 1;
  });

  const edgeGroupCounts = { structure: 0, resource: 0, other: 0 };
  scopeEdges.forEach((edge) => {
    edgeGroupCounts[edge.group] = (edgeGroupCounts[edge.group] || 0) + 1;
  });
  const directEdgeGroupCounts = { structure: 0, resource: 0, other: 0 };
  (edgesByNode.get(selectedId) || []).forEach((edge) => {
    directEdgeGroupCounts[edge.group] = (directEdgeGroupCounts[edge.group] || 0) + 1;
  });

  const upstreamNodeIds = new Set();
  const downstreamNodeIds = new Set();
  scopeEdges.forEach((edge) => {
    if (edge.target === selectedId) upstreamNodeIds.add(edge.source);
    if (edge.source === selectedId) downstreamNodeIds.add(edge.target);
  });

  // 关系分解：按边类型统计（结构/资源/其他），供右侧"关系分解"面板与图例核对
  const edgeTypeCounts = {};
  scopeEdges.forEach((edge) => {
    const key = edge.edgeType || edge.group || "unknown";
    edgeTypeCounts[key] = (edgeTypeCounts[key] || 0) + 1;
  });

  const orderIds = new Set(scopeNodes.filter((node) => node.type === 'order').map((node) => node.id));
  const scopeHighlights = [];
  if (selectedNode.type === 'order') {
    scopeHighlights.push(
      { label: '\u5b8c\u6574\u8ba2\u5355\u4efb\u52a1', value: formatInt(Math.max(0, countsByType.task || 0)) },
      { label: '\u5b8c\u6574\u8ba2\u5355\u5de5\u5e8f', value: formatInt(Math.max(0, countsByType.operation || 0)) },
      { label: '\u76f8\u5173\u673a\u5668 / \u5de5\u88c5 / \u4eba\u5458', value: `${formatInt(countsByType.machine || 0)} / ${formatInt(countsByType.tooling || 0)} / ${formatInt(countsByType.personnel || 0)}` },
      { label: '\u7ed3\u6784\u94fe\u8def / \u8d44\u6e90\u8fb9', value: `${formatInt(edgeGroupCounts.structure || 0)} / ${formatInt(edgeGroupCounts.resource || 0)}` },
    );
  } else if (selectedNode.type === 'task') {
    scopeHighlights.push(
      { label: '\u76f4\u63a5\u4e0a\u6e38 / \u4e0b\u6e38\u4efb\u52a1', value: `${formatInt(upstreamNodeIds.size)} / ${formatInt(downstreamNodeIds.size)}` },
      { label: '\u6240\u5c5e\u8ba2\u5355', value: formatInt(orderIds.size || 0) },
      { label: '\u5173\u8054\u5de5\u5e8f', value: formatInt(countsByType.operation || 0) },
      { label: '\u652f\u6491\u8d44\u6e90', value: `${formatInt(countsByType.machine || 0)} / ${formatInt(countsByType.tooling || 0)} / ${formatInt(countsByType.personnel || 0)}` },
    );
  } else if (selectedNode.type === 'operation') {
    scopeHighlights.push(
      { label: '\u5de5\u5e8f\u4e0a\u6e38 / \u4e0b\u6e38', value: `${formatInt(upstreamNodeIds.size)} / ${formatInt(downstreamNodeIds.size)}` },
      { label: '\u7236\u7ea7\u4efb\u52a1 / \u8ba2\u5355', value: `${formatInt(countsByType.task || 0)} / ${formatInt(orderIds.size || 0)}` },
      { label: '\u53ef\u7528\u673a\u5668 / \u5de5\u88c5 / \u4eba\u5458', value: `${formatInt(countsByType.machine || 0)} / ${formatInt(countsByType.tooling || 0)} / ${formatInt(countsByType.personnel || 0)}` },
      { label: '\u76f4\u63a5\u8d44\u6e90\u8fb9', value: formatInt(directEdgeGroupCounts.resource || 0) },
    );
  } else if (isResourceNodeType(selectedNode.type)) {
    scopeHighlights.push(
      { label: '\u5f71\u54cd\u8ba2\u5355', value: formatInt(orderIds.size || 0) },
      { label: '\u5173\u8054\u4efb\u52a1 / \u5de5\u5e8f', value: `${formatInt(countsByType.task || 0)} / ${formatInt(countsByType.operation || 0)}` },
      { label: '\u76f4\u63a5\u8d44\u6e90\u5173\u7cfb', value: formatInt(directEdgeGroupCounts.resource || 0) },
      { label: '\u76f8\u5173\u7ed3\u6784\u8fb9', value: formatInt(edgeGroupCounts.structure || 0) },
    );
  } else {
    scopeHighlights.push(
      { label: '\u76f4\u63a5\u5173\u8054\u8282\u70b9', value: formatInt(directNeighborIds.size) },
      { label: '\u5b8c\u6574\u76f8\u5173\u8282\u70b9', value: formatInt(Math.max(0, scopeNodes.length - 1)) },
      { label: '\u5b8c\u6574\u76f8\u5173\u8fb9', value: formatInt(scopeEdges.length) },
    );
  }

  return {
    selectedNode,
    nodeMap,
    clusterContext,
    scopeNodeIds,
    scopeNodes,
    scopeEdges,
    relatedNodes,
    relatedByType,
    countsByType,
    orderIds,
    directNeighborIds,
    edgeGroupCounts,
    directEdgeGroupCounts,
    edgeTypeCounts,
    upstreamCount: upstreamNodeIds.size,
    downstreamCount: downstreamNodeIds.size,
    upstreamNodeIds: Array.from(upstreamNodeIds),
    downstreamNodeIds: Array.from(downstreamNodeIds),
    scopeHighlights,
  };
}

function buildGraphSelectionHighlightsV2(graph) {
  if (!graph?.selectedNode) return [];
  const rows = asArray(graph.selectionScope?.scopeHighlights).map((item) => ({ ...item }));
  rows.push({
    label: '\u5de6\u4fa7\u5df2\u5c55\u793a\u76f8\u5173\u8282\u70b9',
    value: formatInt(graph.selectedStats.displayedRelatedNodeCount),
  });
  rows.push({
    label: '\u5de6\u4fa7\u5df2\u5c55\u793a\u76f8\u5173\u8ba2\u5355',
    value: `${formatInt(graph.selectedStats.displayedOrderCount)} / ${formatInt(graph.selectedStats.orderCount)}`,
  });
  return rows;
}

function buildGraphViewModel() {
  const allNodes = asArray(app.graphNodes).map(normalizeGraphNode).filter((node) => node.id);
  const allEdges = asArray(app.graphEdges).map(normalizeGraphEdge).filter((edge) => edge.source && edge.target);
  if (!allNodes.length) return null;

  ensureGraphViewState(allNodes);
  const visibleTypes = new Set(Object.entries(app.graphView.nodeTypes || {}).filter(([, enabled]) => enabled).map(([type]) => type));
  const visibleGroups = new Set(Object.entries(app.graphView.edgeGroups || {}).filter(([, enabled]) => enabled).map(([group]) => group));

  const eligibleNodes = allNodes.filter((node) => visibleTypes.has(node.type));
  const eligibleNodeIds = new Set(eligibleNodes.map((node) => node.id));
  const eligibleEdges = allEdges.filter((edge) => visibleGroups.has(edge.group) && eligibleNodeIds.has(edge.source) && eligibleNodeIds.has(edge.target));
  const visibleClusterContext = buildOrderClusterContext(eligibleNodes, eligibleEdges);

  const adjacency = new Map(eligibleNodes.map((node) => [node.id, []]));
  eligibleEdges.forEach((edge) => {
    adjacency.get(edge.source)?.push({ id: edge.target, edge });
    adjacency.get(edge.target)?.push({ id: edge.source, edge });
  });

  const term = (app.graphView.search || "").trim().toLowerCase();
  let selectedId = app.selectedGraphNodeId;
  if (!eligibleNodeIds.has(selectedId)) selectedId = eligibleNodes[0]?.id || null;

  let focusIds = new Set(eligibleNodeIds);
  const matchedIds = eligibleNodes.filter((node) => graphNodeMatchesSearch(node, term)).map((node) => node.id);
  const expandVisibleOrders = (ids, targetSet) => {
    asArray(ids).forEach((id) => {
      visibleClusterContext.ordersFromNode(id).forEach((orderId) => {
        visibleClusterContext.orderClusters.get(orderId)?.forEach((nodeId) => targetSet.add(nodeId));
      });
    });
  };

  if (term && matchedIds.length) {
    focusIds = new Set(matchedIds);
    matchedIds.forEach((id) => {
      (adjacency.get(id) || []).forEach((neighbor) => focusIds.add(neighbor.id));
    });
    expandVisibleOrders(matchedIds, focusIds);
    if (!focusIds.has(selectedId)) selectedId = matchedIds[0];
  }

  if (app.graphView.mode === "focus" && selectedId) {
    const scopedIds = new Set([selectedId]);
    (adjacency.get(selectedId) || []).forEach((neighbor) => {
      scopedIds.add(neighbor.id);
      if (neighbor.edge.group === "structure") {
        (adjacency.get(neighbor.id) || []).forEach((secondary) => {
          if (secondary.edge.group === "structure") scopedIds.add(secondary.id);
        });
      }
    });
    expandVisibleOrders([selectedId, ...matchedIds], scopedIds);
    focusIds.forEach((id) => scopedIds.add(id));
    focusIds = scopedIds;
  }

  let visibleNodes = eligibleNodes.filter((node) => focusIds.has(node.id));
  let visibleEdges = eligibleEdges.filter((edge) => focusIds.has(edge.source) && focusIds.has(edge.target));

  const nodeLimit = app.graphView.mode === "focus" ? CONFIG.GRAPH_FOCUS_NODE_LIMIT : CONFIG.GRAPH_ALL_NODE_LIMIT;
  const maxOrders = Math.max(1, Number(app.graphView.maxOrders || 6));
  let orderScoped = false;
  const visibleOrderCount = new Set(visibleNodes.filter((node) => node.type === "order").map((node) => node.id)).size;

  if (visibleNodes.length > nodeLimit || visibleOrderCount > maxOrders) {
    let keepIds = buildOrderScopedNodeSet(visibleNodes, visibleEdges, selectedId, matchedIds, Number.MAX_SAFE_INTEGER, maxOrders);
    orderScoped = !!keepIds;
    if (!keepIds) {
      const priorityIds = new Set([selectedId, ...matchedIds]);
      const degree = new Map(visibleNodes.map((node) => [node.id, 0]));
      visibleEdges.forEach((edge) => {
        degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
        degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
      });
      keepIds = new Set(
        visibleNodes
          .slice()
          .sort((a, b) => {
            const aRank = priorityIds.has(a.id) ? 0 : 1;
            const bRank = priorityIds.has(b.id) ? 0 : 1;
            if (aRank !== bRank) return aRank - bRank;
            return (degree.get(b.id) || 0) - (degree.get(a.id) || 0);
          })
          .slice(0, nodeLimit)
          .map((node) => node.id),
      );
    }
    visibleNodes = visibleNodes.filter((node) => keepIds.has(node.id));
    visibleEdges = visibleEdges.filter((edge) => keepIds.has(edge.source) && keepIds.has(edge.target));
  }

  if (!orderScoped && visibleEdges.length > CONFIG.GRAPH_ALL_EDGE_LIMIT) {
    const priorityIds = new Set([selectedId, ...matchedIds]);
    visibleEdges = visibleEdges
      .slice()
      .sort((a, b) => {
        const aRank = priorityIds.has(a.source) || priorityIds.has(a.target) ? 0 : 1;
        const bRank = priorityIds.has(b.source) || priorityIds.has(b.target) ? 0 : 1;
        if (aRank !== bRank) return aRank - bRank;
        if (a.group !== b.group) return a.group.localeCompare(b.group);
        return String(a.edgeType).localeCompare(String(b.edgeType));
      })
      .slice(0, CONFIG.GRAPH_ALL_EDGE_LIMIT);
  }

  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  visibleEdges = visibleEdges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));

  const connectedNodeIds = new Set();
  visibleEdges.forEach((edge) => {
    connectedNodeIds.add(edge.source);
    connectedNodeIds.add(edge.target);
  });
  if (selectedId && !connectedNodeIds.has(selectedId)) {
    selectedId = visibleNodes.find((node) => connectedNodeIds.has(node.id))?.id || selectedId;
  }
  visibleNodes = visibleNodes.filter((node) => connectedNodeIds.has(node.id) || node.id === selectedId);
  const filteredVisibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  visibleEdges = visibleEdges.filter((edge) => filteredVisibleNodeIds.has(edge.source) && filteredVisibleNodeIds.has(edge.target));
  if (!filteredVisibleNodeIds.has(selectedId)) {
    selectedId = visibleNodes.find((node) => connectedNodeIds.has(node.id))?.id || visibleNodes[0]?.id || null;
  }

  const selectionScope = buildGraphSelectionScopeV2(selectedId, allNodes, allEdges);
  const relatedVisibleNodeIds = new Set(Array.from(selectionScope.scopeNodeIds).filter((id) => filteredVisibleNodeIds.has(id)));
  const neighborIds = new Set([selectedId, ...Array.from(selectionScope.directNeighborIds).filter((id) => filteredVisibleNodeIds.has(id))]);
  const highlightedEdges = visibleEdges.filter((edge) => relatedVisibleNodeIds.has(edge.source) && relatedVisibleNodeIds.has(edge.target));

  const layout = layoutGraph(visibleNodes, visibleEdges, selectedId);
  const previousPositions = app.graphView.positions || {};
  const positions = {};
  visibleNodes.forEach((node) => {
    const base = layout.placed.get(node.id);
    const prev = previousPositions[node.id];
    positions[node.id] = prev
      ? { x: prev.x, y: prev.y, r: base?.r || 18 }
      : { x: base?.x || 0, y: base?.y || 0, r: base?.r || 18 };
  });
  app.graphView.positions = positions;
  app.selectedGraphNodeId = selectedId;

  const typeCounts = {};
  visibleNodes.forEach((node) => {
    typeCounts[node.type] = (typeCounts[node.type] || 0) + 1;
  });
  const edgeGroupCounts = {};
  visibleEdges.forEach((edge) => {
    edgeGroupCounts[edge.group] = (edgeGroupCounts[edge.group] || 0) + 1;
  });

  const displayedScopeNodeCount = Math.max(0, relatedVisibleNodeIds.size - (relatedVisibleNodeIds.has(selectedId) ? 1 : 0));
  const displayedOrderCount = visibleNodes.filter((node) => relatedVisibleNodeIds.has(node.id) && node.type === "order").length;
  const scopeCounts = selectionScope.countsByType || {};
  const selectedStats = {
    directNeighborCount: selectionScope.directNeighborIds.size,
    relatedNodeCount: Math.max(0, selectionScope.scopeNodes.length - 1),
    relatedEdgeCount: selectionScope.scopeEdges.length,
    orderCount: scopeCounts.order || 0,
    taskCount: scopeCounts.task || 0,
    operationCount: scopeCounts.operation || 0,
    machineCount: scopeCounts.machine || 0,
    toolingCount: scopeCounts.tooling || 0,
    personnelCount: scopeCounts.personnel || 0,
    structureEdgeCount: selectionScope.edgeGroupCounts?.structure || 0,
    resourceEdgeCount: selectionScope.edgeGroupCounts?.resource || 0,
    otherEdgeCount: selectionScope.edgeGroupCounts?.other || 0,
    directStructureEdgeCount: selectionScope.directEdgeGroupCounts?.structure || 0,
    directResourceEdgeCount: selectionScope.directEdgeGroupCounts?.resource || 0,
    upstreamCount: selectionScope.upstreamCount || 0,
    downstreamCount: selectionScope.downstreamCount || 0,
    displayedRelatedNodeCount: displayedScopeNodeCount,
    displayedOrderCount,
  };

  return {
    selectedId,
    selectedNode: selectionScope.selectedNode,
    nodeMap: selectionScope.nodeMap,
    visibleNodes,
    visibleEdges,
    highlightedEdges,
    relatedEdges: selectionScope.scopeEdges,
    relatedByType: selectionScope.relatedByType,
    neighborIds,
    highlightedNodeIds: relatedVisibleNodeIds,
    positions,
    layout,
    typeCounts,
    edgeGroupCounts,
    totalNodeCount: allNodes.length,
    totalEdgeCount: allEdges.length,
    culledNodeCount: Math.max(0, eligibleNodes.length - visibleNodes.length),
    culledEdgeCount: Math.max(0, eligibleEdges.length - visibleEdges.length),
    orderScoped,
    maxOrders,
    visibleOrderCount,
    selectedStats,
    selectionScope,
  };
}

// —— 图谱 V2：泳道布局 + 三重编码 + 订单过滤 + 图例/统计/详情 ——
function renderInteractiveGraph() {
  const graph = buildGraphViewModel();
  if (!graph || !graph.visibleNodes.length) {
    return renderEmptyState("暂无图谱节点", "请先构建图谱，或确认当前实例已正确加载。");
  }

  const nodeDegree = new Map(graph.visibleNodes.map((node) => [node.id, 0]));
  graph.visibleEdges.forEach((edge) => {
    nodeDegree.set(edge.source, (nodeDegree.get(edge.source) || 0) + 1);
    nodeDegree.set(edge.target, (nodeDegree.get(edge.target) || 0) + 1);
  });

  const selectedNode = graph.selectedNode;
  const selectedId = graph.selectedId;
  const scope = graph.selectionScope || {};
  const orderOption = selectedGraphOrderOption();
  const orderEntityId = orderOption?.entity_id || entityIdFromGraphId(app.selectedGraphOrderId || "");
  const orderColor = orderColorFor(orderEntityId);

  // 面包屑统计基于当前已加载的订单簇（全量），而非可见局部图
  const loadedCounts = {};
  asArray(app.graphNodes).forEach((raw) => {
    const type = String(raw.node_type || raw.type || "other").toLowerCase();
    loadedCounts[type] = (loadedCounts[type] || 0) + 1;
  });
  const loadedEdgeCount = asArray(app.graphEdges).length;

  const edgeMarkers = { structure: "graph-arrow-structure", resource: "graph-arrow-resource", other: "graph-arrow-other" };
  const familyLabels = { plan: "规划链 · 冷色", resource: "资源 · 暖色", other: "其他关系" };

  const svg = `
    <svg viewBox="0 0 ${graph.layout.width} ${graph.layout.height}" class="graph-svg interactive" data-graph-canvas role="img" aria-label="可交互有向异构图">
      <defs>
        <pattern id="graph-grid-pattern" width="28" height="28" patternUnits="userSpaceOnUse">
          <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(163,178,193,0.22)" stroke-width="1"></path>
        </pattern>
        <marker id="graph-arrow-structure" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#1d53c0"></path>
        </marker>
        <marker id="graph-arrow-resource" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#c2620a"></path>
        </marker>
        <marker id="graph-arrow-other" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa5b1"></path>
        </marker>
      </defs>
      <rect x="0" y="0" width="${graph.layout.width}" height="${graph.layout.height}" rx="26" fill="url(#graph-grid-pattern)" class="graph-canvas-bg"></rect>
      <g data-graph-viewport transform="${graphViewportTransform()}">
        <g data-graph-lanes>
          ${(graph.layout.families || []).map((fam) => `<text class="graph-family-label graph-family-${fam.family}" x="${fam.x + 6}" y="24">${escapeHtml(familyLabels[fam.family] || fam.family)}</text>`).join("")}
          ${(graph.layout.lanes || []).map((lane) => `
            <rect class="graph-lane graph-lane-${lane.family}" x="${lane.x}" y="${lane.y}" width="${lane.width}" height="${lane.height}" rx="16"></rect>
            <text class="graph-lane-label" x="${lane.x + 16}" y="${lane.y + 26}">${escapeHtml(lane.label)}</text>
            <text class="graph-lane-count" x="${lane.x + 16}" y="${lane.y + 42}">${formatInt(lane.count)} 个节点</text>
          `).join("")}
        </g>
        <g data-graph-links>
          ${graph.visibleEdges.map((edge) => {
            const source = graph.positions[edge.source];
            const target = graph.positions[edge.target];
            if (!source || !target) return "";
            const highlighted = graph.highlightedNodeIds.has(edge.source) && graph.highlightedNodeIds.has(edge.target);
            const group = ["structure", "resource", "other"].includes(edge.group) ? edge.group : "other";
            return `
              <path
                class="graph-link graph-link-${group} ${highlighted ? "highlighted" : ""}"
                data-graph-link
                data-source="${escapeHtml(edge.source)}"
                data-target="${escapeHtml(edge.target)}"
                d="${graphEdgePathD(source, target)}"
                marker-end="url(#${edgeMarkers[group]})"
              ></path>
            `;
          }).join("")}
        </g>
        <g data-graph-nodes>
          ${graph.visibleNodes.map((node) => {
            const pos = graph.positions[node.id];
            if (!pos) return "";
            const baseR = pos.r || GRAPH_NODE_SIZES[node.type] || GRAPH_NODE_SIZES.other;
            const isSelected = node.id === selectedId;
            const isNeighbor = graph.neighborIds.has(node.id) && !isSelected;
            const isScoped = graph.highlightedNodeIds.has(node.id);
            const r = isSelected ? baseR + 4 : baseR;
            const charFontSize = node.type === "operation" ? 9 : node.type === "order" ? 13 : 11;
            const charDy = node.type === "tooling" ? Math.round(r * 0.3 + 4) : 4;
            const labelY = Math.round((node.type === "order" ? r * 0.85 : r) + 15);
            const orderRing = node.type === "order"
              ? `<circle class="graph-node-order-ring" r="${(r * 1.25 + 7).toFixed(1)}" style="stroke:${orderColorFor(node.entity_id || entityIdFromGraphId(node.id))}"></circle>`
              : "";
            return `
              <g
                class="graph-node ${isSelected ? "selected" : isNeighbor ? "neighbor" : isScoped ? "scoped" : ""}"
                data-type="${escapeHtml(node.type)}"
                data-action="focus-graph-node"
                data-id="${escapeHtml(node.id)}"
                data-graph-node="${escapeHtml(node.id)}"
                data-node-label="${escapeHtml(node.label || node.id)}"
                data-node-type-label="${escapeHtml(graphTypeLabel(node.type))}"
                data-node-degree="${formatInt(nodeDegree.get(node.id) || 0)}"
                data-node-entity-id="${escapeHtml(node.entity_id || entityIdFromGraphId(node.id))}"
                transform="translate(${pos.x} ${pos.y})"
                style="cursor:pointer"
              >
                <title>${escapeHtml(`${graphTypeLabel(node.type)}\n${node.label || node.id}\nID: ${node.id}\n关联关系: ${formatInt(nodeDegree.get(node.id) || 0)}`)}</title>
                <circle class="graph-node-hitbox" r="${Math.max(r + 12, 28)}"></circle>
                ${orderRing}
                ${graphNodeShapeSVG(node.type, r, `class="graph-node-shape" fill="${graphTypeColor(node.type)}"`)}
                <text class="graph-node-char" y="${charDy}" text-anchor="middle" font-size="${charFontSize}">${escapeHtml(GRAPH_TYPE_CHARS[node.type] || "·")}</text>
                <text class="graph-node-label" y="${labelY}" text-anchor="middle">${escapeHtml(String(node.label).slice(0, 18))}</text>
              </g>
            `;
          }).join("")}
        </g>
      </g>
    </svg>
  `;

  const nodeTypeFilters = Object.entries(app.graphView.nodeTypes || {}).map(([type, enabled]) => `
      <button class="filter-chip ${enabled ? "active" : ""}" type="button" data-action="toggle-graph-node-type" data-key="${escapeHtml(type)}">
        <span class="filter-chip-dot" style="background:${graphTypeColor(type)}"></span>
        ${escapeHtml(graphTypeLabel(type))}
        <span class="filter-chip-count">${formatInt(graph.typeCounts[type] || 0)}</span>
      </button>
    `).join("");
  const edgeGroupFilters = Object.entries(app.graphView.edgeGroups || {}).map(([group, enabled]) => `
      <button class="filter-chip ${enabled ? "active" : ""}" type="button" data-action="toggle-graph-edge-group" data-key="${escapeHtml(group)}">
        ${escapeHtml(GRAPH_EDGE_GROUP_LABELS[group] || group)}
        <span class="filter-chip-count">${formatInt(graph.edgeGroupCounts[group] || 0)}</span>
      </button>
    `).join("");

  const bounds = Object.values(graph.positions).reduce((acc, item) => ({
    minX: Math.min(acc.minX, item.x),
    maxX: Math.max(acc.maxX, item.x),
    minY: Math.min(acc.minY, item.y),
    maxY: Math.max(acc.maxY, item.y),
    canvasWidth: graph.layout.width,
    canvasHeight: graph.layout.height,
  }), { minX: Infinity, maxX: 0, minY: Infinity, maxY: 0, canvasWidth: graph.layout.width, canvasHeight: graph.layout.height });
  app.graphView.bounds = bounds;

  const orderOptionsHtml = asArray(app.graphOrderOptions).map((raw) => {
    const order = normalizeGraphNode(raw);
    const entity = order.entity_id || "";
    const label = order.label && order.label !== entity ? `${entity} · ${order.label}` : entity;
    return `<option value="${escapeHtml(entity)}" ${entity === orderEntityId ? "selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
  const orderFilterBar = `
    <label class="graph-search graph-order-select">
      <span>按订单过滤</span>
      <select data-graph-order-select>
        ${orderOptionsHtml || `<option value="">${orderEntityId ? escapeHtml(orderEntityId) : "暂无订单"}</option>`}
      </select>
    </label>
    <label class="graph-search graph-order-jump">
      <span>其他订单</span>
      <span class="graph-order-jump-row">
        <input type="search" list="graph-order-options" data-graph-order-input placeholder="输入订单号，回车跳转" autocomplete="off">
        <button class="btn btn-ghost" type="button" data-action="graph-order-search">跳转</button>
      </span>
    </label>
    <datalist id="graph-order-options">
      ${asArray(app.graphOrderOptions).map((raw) => {
        const order = normalizeGraphNode(raw);
        return `<option value="${escapeHtml(order.entity_id || "")}">${escapeHtml(order.label || order.entity_id || "")}</option>`;
      }).join("")}
    </datalist>
  `;

  const breadcrumb = `
    <div class="graph-breadcrumb" aria-label="订单簇统计">
      <span class="crumb crumb-order"><i style="background:${orderColor}"></i>订单 ${escapeHtml(orderEntityId || "未选择")}</span>
      <span class="crumb-sep" aria-hidden="true">▸</span>
      <span class="crumb">任务 ${formatInt(loadedCounts.task || 0)}</span>
      <span class="crumb-sep" aria-hidden="true">▸</span>
      <span class="crumb">工序 ${formatInt(loadedCounts.operation || 0)}</span>
      <span class="crumb-sep" aria-hidden="true">▸</span>
      <span class="crumb">机器/工装/人员 ${formatInt(loadedCounts.machine || 0)} / ${formatInt(loadedCounts.tooling || 0)} / ${formatInt(loadedCounts.personnel || 0)}</span>
      <span class="crumb-sep" aria-hidden="true">▸</span>
      <span class="crumb">关系 ${formatInt(loadedEdgeCount)}</span>
    </div>
  `;

  const stageNarrative = graph.orderScoped
    ? `当前图中展示 ${formatInt(graph.selectedStats.displayedOrderCount)} / ${formatInt(graph.selectedStats.orderCount || graph.visibleOrderCount)} 个相关订单，右侧统计已按全部相关关系计算`
    : "当前视图支持滚轮缩放、拖拽平移和节点微调布局";
  const culled = graph.culledNodeCount > 0 || graph.culledEdgeCount > 0;
  const canvasStats = `
    <div class="graph-canvas-stats">
      <span>展示节点 <b>${formatInt(graph.visibleNodes.length)}</b> / ${formatInt(graph.totalNodeCount)}</span>
      <span>展示边 <b>${formatInt(graph.visibleEdges.length)}</b> / ${formatInt(graph.totalEdgeCount)}</span>
      <span>层级 <b>${formatInt((graph.layout.lanes || []).length)}</b> 条</span>
      <span>当前订单 <b>${escapeHtml(orderEntityId || "-")}</b></span>
      ${culled
        ? `<span class="graph-canvas-stats-warn">已按焦点/上限收敛，省略 ${formatInt(graph.culledNodeCount)} 节点 · ${formatInt(graph.culledEdgeCount)} 边</span>`
        : `<span class="graph-canvas-stats-note">${stageNarrative}</span>`}
    </div>
  `;

  const legendTypes = GRAPH_NODE_ORDER.map((type) => `
      <span class="legend-item"><svg class="legend-shape" viewBox="-12 -12 24 24" aria-hidden="true">${graphNodeShapeSVG(type, 8, `fill="${graphTypeColor(type)}"`)}</svg>${escapeHtml(graphTypeLabel(type))}</span>
    `).join("");
  const legendDock = `
    <div class="graph-legend-dock">
      <div class="legend-block">
        <span class="legend-title">节点 · 颜色/形状/层级</span>
        <div class="legend">${legendTypes}</div>
      </div>
      <div class="legend-block">
        <span class="legend-title">关系 · 线型</span>
        <div class="legend">
          <span class="legend-item"><span class="legend-line legend-line-structure"></span>结构链路（实线）</span>
          <span class="legend-item"><span class="legend-line legend-line-resource"></span>资源可行（虚线）</span>
          <span class="legend-item"><span class="legend-line legend-line-other"></span>其他关系（点线）</span>
        </div>
      </div>
      <div class="legend-block">
        <span class="legend-title">跨视图联动</span>
        <div class="legend">
          <span class="legend-item"><span class="legend-order-ring" style="border-color:${orderColor}"></span>订单标识色与甘特条块同色</span>
        </div>
      </div>
    </div>
  `;

  const selectionHighlights = buildGraphSelectionHighlightsV2(graph);
  const selectedSummary = selectedNode ? `
    <div class="graph-selected-summary" data-graph-selected-summary>
      <div class="graph-selected-main">
        <span class="graph-selected-badge-svg" aria-hidden="true">
          <svg viewBox="-14 -14 28 28">${graphNodeShapeSVG(selectedNode.type, 10, `fill="${graphTypeColor(selectedNode.type)}"`)}</svg>
        </span>
        <div>
          <div class="graph-selected-title">${escapeHtml(selectedNode.label || selectedNode.id)}</div>
          <div class="graph-selected-meta">
            <span class="graph-type-badge" style="background:${graphTypeColor(selectedNode.type)}">${escapeHtml(graphTypeLabel(selectedNode.type))}</span>
            <span>${escapeHtml(selectedNode.entity_id || selectedNode.id || "-")}</span>
            ${selectedNode.type === "order" ? `<span class="graph-order-tag"><i class="graph-order-dot" style="background:${orderColorFor(selectedNode.entity_id || entityIdFromGraphId(selectedNode.id))}"></i>订单标识色</span>` : ""}
          </div>
        </div>
      </div>
      <div class="graph-selected-stats">
        <span>直接邻居 ${formatInt(graph.selectedStats.directNeighborCount)}</span>
        <span>上游 / 下游 ${formatInt(graph.selectedStats.upstreamCount)} / ${formatInt(graph.selectedStats.downstreamCount)}</span>
        <span>全相关节点 ${formatInt(graph.selectedStats.relatedNodeCount)}</span>
        <span>全相关边 ${formatInt(graph.selectedStats.relatedEdgeCount)}</span>
        <span>结构边 ${formatInt(graph.selectedStats.structureEdgeCount)}</span>
        <span>资源边 ${formatInt(graph.selectedStats.resourceEdgeCount)}</span>
      </div>
    </div>
  ` : renderEmptyState("未选中节点", "点击左侧图中的节点后，这里会显示该节点相关的完整统计与关系说明。");

  const scopeOverview = selectedNode ? renderKeyValueGrid([
    { label: "全相关订单", value: formatInt(graph.selectedStats.orderCount) },
    { label: "全相关任务 / 工序", value: `${formatInt(graph.selectedStats.taskCount)} / ${formatInt(graph.selectedStats.operationCount)}` },
    { label: "全相关机器 / 工装 / 人员", value: `${formatInt(graph.selectedStats.machineCount)} / ${formatInt(graph.selectedStats.toolingCount)} / ${formatInt(graph.selectedStats.personnelCount)}` },
    { label: "当前图中展示订单", value: `${formatInt(graph.selectedStats.displayedOrderCount)} / ${formatInt(graph.selectedStats.orderCount)}` },
  ]) : "";

  const updownList = (ids, emptyText) => {
    const nodes = asArray(ids)
      .map((id) => graph.nodeMap.get(id))
      .filter(Boolean)
      .sort((a, b) => String(a.label || a.id).localeCompare(String(b.label || b.id), "zh-CN"));
    if (!nodes.length) return `<div class="graph-updown-empty">${emptyText}</div>`;
    return `
      ${nodes.slice(0, 6).map((node) => `
        <button class="pill graph-node-pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(node.id)}">
          <i style="background:${graphTypeColor(node.type)}"></i>${escapeHtml(String(node.label || node.id).slice(0, 14))}
        </button>
      `).join("")}
      ${nodes.length > 6 ? `<span class="graph-updown-more">+${formatInt(nodes.length - 6)}</span>` : ""}
    `;
  };
  const updownBlock = selectedNode ? `
    <div class="graph-updown">
      <section class="graph-updown-col">
        <header>上游（入边） · ${formatInt(scope.upstreamCount || 0)}</header>
        <div class="graph-updown-list">${updownList(scope.upstreamNodeIds, "无上游节点")}</div>
      </section>
      <section class="graph-updown-col">
        <header>下游（出边） · ${formatInt(scope.downstreamCount || 0)}</header>
        <div class="graph-updown-list">${updownList(scope.downstreamNodeIds, "无下游节点")}</div>
      </section>
    </div>
  ` : "";

  const relationRows = Object.entries(scope.edgeTypeCounts || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([edgeType, count]) => {
      const group = graphEdgeGroupForType(edgeType);
      return `
        <div class="graph-relation-row">
          <span class="legend-line legend-line-${group}"></span>
          <span class="graph-relation-name">${escapeHtml(graphEdgeTypeLabel(edgeType))}</span>
          <span class="graph-relation-group">${escapeHtml(GRAPH_EDGE_GROUP_LABELS[group] || group)}</span>
          <b>${formatInt(count)}</b>
        </div>
      `;
    }).join("");
  const relationBlock = selectedNode && relationRows ? `
    <div class="graph-relation-breakdown">
      <div class="graph-filter-title">关系分解（按边类型）</div>
      ${relationRows}
    </div>
  ` : "";

  return `
    <div class="surface-card graph-workbench">
      <div class="card-head">
        <h3>可交互有向异构图</h3>
        <p>六条层级泳道（规划链冷色 → 资源暖色），节点按颜色+形状+尺寸三重编码；订单标识色与甘特条块跨视图联动。</p>
      </div>
      <div class="graph-toolbar">
        <div class="inline-actions">
          <button class="btn ${app.graphView.mode === "focus" ? "btn-primary" : "btn-ghost"}" type="button" data-action="set-graph-mode" data-mode="focus">焦点邻域</button>
          <button class="btn ${app.graphView.mode === "all" ? "btn-primary" : "btn-ghost"}" type="button" data-action="set-graph-mode" data-mode="all">全关系视图</button>
          <button class="btn btn-ghost" type="button" data-action="fit-graph-view">适配视图</button>
          <button class="btn btn-ghost" type="button" data-action="reset-graph-view">重置视图</button>
          <button class="btn btn-ghost" type="button" data-action="zoom-graph-out">-</button>
          <button class="btn btn-ghost" type="button" data-action="toggle-graph-fullscreen">全屏查看</button>
          <span class="graph-zoom-pill" data-graph-zoom>${Math.round(app.graphView.zoom * 100)}%</span>
          <button class="btn btn-ghost" type="button" data-action="zoom-graph-in">+</button>
        </div>
        ${orderFilterBar}
        <label class="graph-search">
          <span>搜索节点</span>
          <input type="search" value="${escapeHtml(app.graphView.search || "")}" data-graph-search placeholder="搜索订单、任务、工序或资源">
        </label>
      </div>
      ${breadcrumb}
      <div class="graph-filter-grid">
        <section class="graph-filter-group">
          <div class="graph-filter-title">节点层级</div>
          <div class="filter-chip-row">${nodeTypeFilters}</div>
        </section>
        <section class="graph-filter-group">
          <div class="graph-filter-title">关系层级</div>
          <div class="filter-chip-row">${edgeGroupFilters}</div>
        </section>
      </div>
      <div class="split-panel graph-split">
        <article class="surface-card graph-stage-card">
          ${canvasStats}
          <div class="graph-hover-preview" data-graph-hover-preview>
            ${selectedNode ? `当前选中：${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.label || selectedNode.id)} / 关联关系 ${formatInt(nodeDegree.get(selectedNode.id) || 0)}` : "将鼠标悬浮到节点上，可快速预览该节点名称、类型与关联关系强度。"}
          </div>
          <div class="graph-shell">${svg}</div>
          ${legendDock}
        </article>
        <article class="surface-card graph-detail-card">
          <div class="card-head">
            <h3>节点详情与关系解释</h3>
            <p>右侧所有统计都基于“选中节点的完整相关作用域”计算，而不是只基于左侧可见局部图。</p>
          </div>
          ${selectedSummary}
          ${selectedNode ? renderKeyValueGrid(graphNodeDetailRows(selectedNode, graph.relatedEdges)) : renderEmptyState("未选中节点", "请先在左侧图中点击一个节点。")}
          ${updownBlock}
          ${relationBlock}
          ${scopeOverview}
          ${selectionHighlights.length ? renderKeyValueGrid(selectionHighlights) : ""}
          ${Object.keys(graph.relatedByType).length ? `
            <div class="graph-neighbor-groups">
              ${Object.entries(graph.relatedByType).map(([type, items]) => `
                <section class="graph-neighbor-group">
                  <header>
                    <strong><i class="graph-order-dot" style="background:${graphTypeColor(type)}"></i>${escapeHtml(graphTypeLabel(type))}</strong>
                    <span>${formatInt(items.length)} 个</span>
                  </header>
                  <div class="graph-neighbor-pills">
                    ${items.slice(0, 10).map((item) => `<button class="pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(item.id)}">${escapeHtml(item.label || item.id)}</button>`).join("")}
                  </div>
                </section>
              `).join("")}
            </div>
          ` : ""}
          ${graph.relatedEdges.length ? renderSimpleTable(
            ["关系", "来源", "目标"],
            graph.relatedEdges.slice(0, 16).map((edge) => [
              escapeHtml(graphEdgeTypeLabel(edge.edgeType || edge.group || "-")),
              escapeHtml(graph.nodeMap.get(edge.source)?.label || edge.source),
              escapeHtml(graph.nodeMap.get(edge.target)?.label || edge.target),
            ]),
            { footer: graph.relatedEdges.length > 16 ? `当前只展示前 16 条关联边，共 ${graph.relatedEdges.length} 条。` : "" },
          ) : renderEmptyState("暂无关联边", "当前节点在全相关作用域中没有可展示的关系边。")}
        </article>
      </div>
    </div>
  `;
}

function mountInteractiveGraph() {
  const svg = document.querySelector("[data-graph-canvas]");
  if (!svg || svg.dataset.bound === "1") return;
  svg.dataset.bound = "1";

  const getSvgScale = () => {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    return {
      x: viewBox.width / Math.max(rect.width, 1),
      y: viewBox.height / Math.max(rect.height, 1),
    };
  };

  const root = svg.closest(".graph-workbench") || document;
  const hoverPreview = root.querySelector("[data-graph-hover-preview]") || root.querySelector("#graph-hover-preview");
  applyGraphViewportState(root);
  applyGraphNodePositions(root);

  const updateHoverPreview = (nodeEl) => {
    if (!hoverPreview) return;
    if (!nodeEl) {
      const selected = Array.from(root.querySelectorAll("[data-graph-node]")).find((node) => node.dataset.graphNode === (app.selectedGraphNodeId || ""));
      if (selected) {
        hoverPreview.textContent = `\u5f53\u524d\u9009\u4e2d\uff1a${selected.dataset.nodeTypeLabel || "-"} / ${selected.dataset.nodeLabel || "-"} / \u5173\u8054\u5173\u7cfb ${selected.dataset.nodeDegree || "0"}`;
      } else {
        hoverPreview.textContent = "\u60ac\u6d6e\u8282\u70b9\u53ef\u5feb\u901f\u9884\u89c8\u5176\u7c7b\u578b\u3001\u6807\u7b7e\u548c\u5173\u7cfb\u6570\uff0c\u70b9\u51fb\u540e\u53f3\u4fa7\u4f1a\u8054\u52a8\u5237\u65b0\u5b8c\u6574\u8be6\u60c5\u3002";
      }
      return;
    }
    hoverPreview.textContent = `\u60ac\u6d6e\u9884\u89c8\uff1a${nodeEl.dataset.nodeTypeLabel || "-"} / ${nodeEl.dataset.nodeLabel || "-"} / \u5173\u8054\u5173\u7cfb ${nodeEl.dataset.nodeDegree || "0"}`;
  };

  let drag = null;
  let dragMoved = false;

  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const mouseX = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * viewBox.width;
    const mouseY = ((event.clientY - rect.top) / Math.max(rect.height, 1)) * viewBox.height;
    const currentZoom = app.graphView.zoom;
    const nextZoom = Math.max(0.45, Math.min(2.4, currentZoom * (event.deltaY < 0 ? 1.12 : 0.88)));
    const ratio = nextZoom / currentZoom;
    app.graphView.panX = mouseX - (mouseX - app.graphView.panX) * ratio;
    app.graphView.panY = mouseY - (mouseY - app.graphView.panY) * ratio;
    app.graphView.zoom = nextZoom;
    applyGraphViewportState(root);
  }, { passive: false });

  svg.addEventListener("pointerdown", (event) => {
    const nodeEl = event.target.closest("[data-graph-node]");
    const scales = getSvgScale();
    if (nodeEl) {
      const id = nodeEl.dataset.graphNode;
      const origin = app.graphView.positions[id];
      if (!origin) return;
      drag = {
        type: "node",
        id,
        startX: event.clientX,
        startY: event.clientY,
        origin: { ...origin },
        scales,
      };
      dragMoved = false;
      svg.classList.add("dragging-node");
    } else {
      drag = {
        type: "pan",
        startX: event.clientX,
        startY: event.clientY,
        originPanX: app.graphView.panX,
        originPanY: app.graphView.panY,
        scales,
      };
      dragMoved = false;
      svg.classList.add("panning");
    }
    // 注意：这里不做 setPointerCapture——按下即捕获会让随后 click 事件
    // 重定向到 svg 本身（Chrome 行为），导致"点击节点选中"永远落空；
    // 改为首次位移超阈值时再捕获（见 pointermove），纯点击不捕获。
  });

  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    if (Math.abs(event.clientX - drag.startX) > 2 || Math.abs(event.clientY - drag.startY) > 2) {
      dragMoved = true;
      if (event.pointerId != null && !svg.hasPointerCapture(event.pointerId)) {
        try { svg.setPointerCapture(event.pointerId); } catch (_) { /* 忽略捕获失败 */ }
      }
    }
    if (drag.type === "pan") {
      app.graphView.panX = drag.originPanX + (event.clientX - drag.startX) * drag.scales.x;
      app.graphView.panY = drag.originPanY + (event.clientY - drag.startY) * drag.scales.y;
      applyGraphViewportState(root);
      return;
    }
    const position = app.graphView.positions[drag.id];
    if (!position) return;
    position.x = drag.origin.x + ((event.clientX - drag.startX) * drag.scales.x) / app.graphView.zoom;
    position.y = drag.origin.y + ((event.clientY - drag.startY) * drag.scales.y) / app.graphView.zoom;
    applyGraphNodePositions(root);
  });

  const release = (event) => {
    if (!drag) return;
    if (dragMoved) app.graphSuppressClickUntil = Date.now() + 180;
    drag = null;
    dragMoved = false;
    svg.classList.remove("panning");
    svg.classList.remove("dragging-node");
    if (event?.pointerId != null && svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
  };

  svg.addEventListener("pointerup", release);
  svg.addEventListener("pointercancel", release);
  svg.addEventListener("pointerleave", (event) => {
    if (drag?.type === "pan") release(event);
    updateHoverPreview(null);
  });

  svg.addEventListener("click", (event) => {
    const nodeEl = event.target.closest("[data-graph-node]");
    if (!nodeEl) return;
    if (app.graphSuppressClickUntil && Date.now() < app.graphSuppressClickUntil) return;
    app.selectedGraphNodeId = nodeEl.dataset.graphNode || nodeEl.dataset.id || null;
    renderCurrentPage();
  });

  svg.addEventListener("pointermove", (event) => {
    const nodeEl = event.target.closest("[data-graph-node]");
    updateHoverPreview(nodeEl || null);
  });
}

function selectedGraphOrderOption() {
  return asArray(app.graphOrderOptions).find((node) => {
    return normalizeGraphNode(node).id === app.selectedGraphOrderId;
  }) || null;
}

function mountGantts() {
  mountOrderComboboxes();
  if (typeof window.vis === "undefined" || typeof window.vis.Timeline !== "function") return;
  const liveCanvasIds = new Set(Array.from(document.querySelectorAll(".page.active .gantt-canvas")).map((el) => el.id));
  // Destroy orphaned instances: canvas id no longer in the active DOM, or the id exists but
  // the bound element was replaced by a re-render (entry.el detached). Keep bound, still-visible
  // instances intact so a redundant mountGantts call never blanks a live canvas.
  app.ganttInstances = app.ganttInstances.filter((entry) => {
    const stillLive = liveCanvasIds.has(entry.canvasId) && (!entry.el || entry.el.isConnected);
    if (stillLive) return true;
    try { entry.timeline.destroy(); } catch (_) {}
    return false;
  });
  document.querySelectorAll(".page.active .gantt-canvas:not([data-bound='1'])").forEach((el) => {
    const payload = app.pendingGantts.get(el.id);
    if (!payload) return;
    // renderTimeline 已构建过 data（暂存在 payload），大实例下避免重复归一化/遮罩计算
    const data = payload.data || buildGanttData(payload.entries, payload.options);
    if (!data) return;
    const stored = app.ganttViewWindows[el.id];
    const selectedWindow = stored?.viewKey === data.viewKey ? stored.window : data.initialWindow;
    el.dataset.bound = "1";
    const items = new vis.DataSet(data.items);
    const groups = new vis.DataSet(data.groups);
    const timeline = new vis.Timeline(
      el,
      items,
      groups,
      {
        editable: false,
        selectable: false,
        zoomable: true,
        moveable: true,
        horizontalScroll: true,
        stack: false,
        // 行高加大：内容留白 + 条目纵向间距，条块圆角/轻投影由 CSS 承担
        margin: { item: { horizontal: 4, vertical: 8 }, axis: 10 },
        orientation: { axis: "top" },
        zoomMin: 1000 * 60 * 30,
        zoomMax: 1000 * 60 * 60 * 24 * 90,
        // 关闭内置真实当前时间线（相对小时基准下无意义），"现在"线由 data.nowISO 按进度分界注入
        showCurrentTime: false,
        groupOrder: (a, b) => (a.seq || 0) - (b.seq || 0),
        // 无数据时范围为空，交给 vis 自适应空视图。
        start: selectedWindow ? selectedWindow.start : undefined,
        end: selectedWindow ? selectedWindow.end : undefined,
        min: data.fullWindow ? data.fullWindow.start : undefined,
        max: data.fullWindow ? data.fullWindow.end : undefined,
        showTooltips: true,
        // 内置 moment 只打包了 de/en/ja/ru/uk 语料且全局 locale 被 uk 覆盖，
        // 因此各刻度一律用与语言无关的数字格式，不能依赖 ddd / MMMM。
        format: {
          minorLabels: { minute: "HH:mm", hour: "HH:mm", weekday: "D日", day: "D日", week: "w周", month: "M月", year: "YYYY" },
          majorLabels: { minute: "M月D日", hour: "M月D日", weekday: "YYYY年M月", day: "YYYY年M月", week: "YYYY年M月", month: "YYYY年", year: "" },
        },
      }
    );
    if (data.nowISO) {
      try { timeline.addCustomTime(data.nowISO, "sched-now"); } catch (_) { /* 忽略现在线注入失败 */ }
    }
    const entry = { canvasId: el.id, el, timeline, items, groups, data };
    timeline.on("rangechanged", (props) => {
      app.ganttViewWindows[el.id] = {
        viewKey: entry.data.viewKey,
        window: { start: props.start.toISOString(), end: props.end.toISOString() },
      };
    });
    app.ganttInstances.push(entry);
  });
}

async function renderCurrentPage() {
  ensureReviewSelection();
  updateShell();
  if (app.currentPage === "new-scene") {
    renderNewScene();
    if (!app.validation && !app.validationBusy && app.currentScene) handleRunValidation(true);
  }
  if (app.currentPage === "dashboard") renderDashboard();
  if (app.currentPage === "workflow") renderWorkflow();
  if (app.currentPage === "review") renderReview();
  if (app.currentPage === "system") renderSystem();
  syncGraphBuildControls();
}

async function loadCatalogs() {
  try {
    const [optRes, exactRes, llmRes] = await Promise.all([
      api.getOptimizeObjectives().catch(() => ({ objectives: [] })),
      api.getExactObjectives().catch(() => ({ objectives: [] })),
      api.getLlmConfig().catch(() => null),
    ]);
    app.optimizeObjectiveCatalog = asArray(optRes?.objectives);
    app.exactObjectiveCatalog = asArray(exactRes?.objectives);
    app.llmConfig = llmRes;
    if (!app.optimizeForm.objectiveKeys.length && app.optimizeObjectiveCatalog.length) {
      app.optimizeForm.objectiveKeys = app.optimizeObjectiveCatalog.slice(0, 3).map((item) => item.key);
    }
  } catch (error) {
    toast(`加载目录失败：${error.message}`, "warning");
  }
}

// 规则参照方案候选池（除基线规则外）；用于 chip 状态标记与自动加载 key。
function reviewHeuristicRules() {
  const baselineRule = app.optimizeResult?.baseline?.rule_name;
  return CONFIG.HEURISTIC_RULES.filter((rule) => rule !== baselineRule);
}

function referenceSolutionsKey(rules, objectiveKeys) {
  return `${rules.slice().sort().join(",")}|${asArray(objectiveKeys).slice().sort().join(",")}`;
}

function ruleIsCached(rule) {
  return app.referenceSolutionsState.cachedRules.includes(rule)
    || asArray(app.referenceSolutions).some((item) => item.rule_name === rule);
}

// 进评审页自动加载：只取缓存命中的规则参照（only_cached=true），不触发任何新仿真。
// 仿照 ensureTypeUtilization 的 key-based fire-and-forget，key 未变不重复请求。
function ensureReferenceSolutions() {
  const rules = reviewHeuristicRules();
  if (!rules.length) return;
  const objectiveKeys = app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys;
  const key = referenceSolutionsKey(rules, objectiveKeys);
  const state = app.referenceSolutionsState;
  if (state.key === key && (state.loading || state.cachedRules.length || state.missingRules.length || state.error)) return;
  if (state.key !== key) loadReferenceSolutionsCached(rules, objectiveKeys, key);
}

async function loadReferenceSolutionsCached(rules, objectiveKeys, key) {
  app.referenceSolutionsState = { ...app.referenceSolutionsState, key, loading: true, error: null };
  try {
    const result = await api.simulateReferenceSolutions(rules, objectiveKeys, true);
    if (app.referenceSolutionsState.key !== key) return; // 竞态：期间已切规则集/目标，丢弃过期响应
    app.referenceSolutions = asArray(result?.solutions);
    app.referenceSolutionsState = {
      key,
      loading: false,
      error: null,
      cachedRules: asArray(result?.cached_rules),
      missingRules: asArray(result?.missing_rules),
      computing: [],
    };
    ensureReviewSelection();
    await renderCurrentPage();
  } catch (error) {
    if (app.referenceSolutionsState.key === key) {
      app.referenceSolutionsState = { ...app.referenceSolutionsState, loading: false, error: error.message || String(error) };
    }
  }
}

// 显式触发未缓存规则的仿真（100s 量级）：chip 进入计算中，算完增量插入对比表，不阻塞页面。
async function computeReferenceSolution(rule) {
  const state = app.referenceSolutionsState;
  if (state.computing.includes(rule)) return;
  const objectiveKeys = app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys;
  // 竞态守卫：记录当前规则集/目标键，~100s 后响应回来若已变化则丢弃过期结果（同 loadReferenceSolutionsCached）
  const guardKey = referenceSolutionsKey(reviewHeuristicRules(), objectiveKeys);
  app.referenceSolutionsState = { ...state, computing: [...state.computing, rule] };
  await renderCurrentPage();
  try {
    const result = await api.simulateReferenceSolutions([rule], objectiveKeys, false);
    const stillCurrent = referenceSolutionsKey(reviewHeuristicRules(), app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys) === guardKey;
    const cur = app.referenceSolutionsState;
    if (!stillCurrent) {
      // 期间实例/目标已变，过期结果不并入对比表，仅清理该规则的"计算中"标记
      app.referenceSolutionsState = { ...cur, computing: cur.computing.filter((item) => item !== rule) };
      await renderCurrentPage();
      return;
    }
    const merged = new Map(asArray(app.referenceSolutions).map((item) => [item.solution_id, item]));
    asArray(result?.solutions).forEach((item) => merged.set(item.solution_id, item));
    app.referenceSolutions = Array.from(merged.values());
    app.referenceSolutionsState = {
      ...cur,
      cachedRules: Array.from(new Set([...cur.cachedRules, ...asArray(result?.cached_rules), rule])),
      missingRules: cur.missingRules.filter((item) => item !== rule),
      computing: cur.computing.filter((item) => item !== rule),
    };
    ensureReviewSelection();
    await renderCurrentPage();
  } catch (error) {
    const cur = app.referenceSolutionsState;
    app.referenceSolutionsState = { ...cur, computing: cur.computing.filter((item) => item !== rule) };
    toast(`计算规则 ${rule} 失败：${error.message || error}`, "warning");
    await renderCurrentPage();
  }
}

function collectOptimizeForm() {
  const selected = Array.from(document.querySelectorAll("[data-objective-key]"))
    .filter((node) => node.checked)
    .map((node) => node.dataset.objectiveKey);
  if (selected.length) app.optimizeForm.objectiveKeys = selected;
  app.optimizeForm.targetSolutionCount = Number(el("opt-target-count")?.value || app.optimizeForm.targetSolutionCount);
  app.optimizeForm.timeLimitS = Number(el("opt-time-limit")?.value || app.optimizeForm.timeLimitS);
  app.optimizeForm.populationSize = Number(el("opt-population")?.value || app.optimizeForm.populationSize);
  app.optimizeForm.generations = Number(el("opt-generations")?.value || app.optimizeForm.generations);
  app.optimizeForm.coarseTimeRatio = Number(el("opt-coarse-ratio")?.value || app.optimizeForm.coarseTimeRatio);
  app.optimizeForm.refineRounds = Number(el("opt-refine-rounds")?.value || app.optimizeForm.refineRounds);
  app.optimizeForm.alnsAggression = Number(el("opt-alns-aggression")?.value || app.optimizeForm.alnsAggression);
  app.optimizeForm.baselineRuleName = el("opt-baseline-rule")?.value || app.optimizeForm.baselineRuleName;
  refreshOptimizeBudgetRecommendation({ preserveManual: true });
  return {
    objective_keys: app.optimizeForm.objectiveKeys,
    target_solution_count: app.optimizeForm.targetSolutionCount,
    time_limit_s: app.optimizeForm.timeLimitS,
    population_size: app.optimizeForm.populationSize,
    generations: app.optimizeForm.generations,
    coarse_time_ratio: app.optimizeForm.coarseTimeRatio,
    refine_rounds: app.optimizeForm.refineRounds,
    alns_aggression: app.optimizeForm.alnsAggression,
    baseline_rule_name: app.optimizeForm.baselineRuleName,
  };
}

async function pollOptimizeStatus() {
  if (!app.optimizeTaskId) return;
  try {
    app.optimizeStatus = {
      ...await api.getOptimizeStatus(app.optimizeTaskId),
      received_at: Date.now() / 1000,
    };
    app.optimizePollFailures = 0;
    updateShell();
    if (["workflow", "dashboard", "review", "system"].includes(app.currentPage)) await renderCurrentPage();
    const lowered = String(app.optimizeStatus?.status || "").toLowerCase();
    if (["failed", "error"].includes(lowered)) {
      window.clearInterval(app.pollTimer);
      app.pollTimer = null;
      updateShell();
      await renderCurrentPage();
      toast(`优化失败：${app.optimizeStatus?.error || app.optimizeStatus?.message || "未收到具体原因"}`, "error");
      showErrorModal(
        `优化失败${app.optimizeStatus?.error_type ? `（${app.optimizeStatus.error_type}）` : ""}`,
        app.optimizeStatus?.error || app.optimizeStatus?.message || "后端未返回具体原因，请查看服务日志。",
        app.optimizeStatus?.technical_detail || "",
      );
      return;
    }
    if (["done", "completed", "success"].includes(lowered)) {
      window.clearInterval(app.pollTimer);
      app.pollTimer = null;
      app.optimizeResult = await api.getOptimizeResult(app.optimizeTaskId);
      app.referenceSolutions = asArray(app.optimizeResult.reference_solutions);
      ensureReviewSelection();
      persistReviewProgress();
      updateShell();
      await renderCurrentPage();
      toast("混合优化已完成。", "success");
    }
  } catch (error) {
    app.optimizePollFailures += 1;
    if (app.optimizePollFailures < 3) return;
    window.clearInterval(app.pollTimer);
    app.pollTimer = null;
    app.optimizeStatus = {
      ...(app.optimizeStatus || {}),
      status: "error",
      phase: "connection",
      message: "连续 3 次无法获取优化进度",
      error: `无法连接到优化状态服务：${error.message}`,
      error_type: "StatusConnectionError",
      updated_at: Date.now() / 1000,
    };
    updateShell();
    await renderCurrentPage();
    toast(app.optimizeStatus.error, "error");
    showErrorModal("无法获取优化进度", app.optimizeStatus.error, "请确认后端服务仍在运行，然后重新进入本页查看任务状态。");
  }
}

function startOptimizePolling() {
  if (app.pollTimer) window.clearInterval(app.pollTimer);
  app.pollTimer = window.setInterval(() => {
    pollOptimizeStatus();
  }, CONFIG.OPT_POLL_MS);
}

function setImportProgress(state) {
  const panel = el("import-progress");
  if (!panel) return;
  if (!state) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  panel.classList.toggle("is-error", state.tone === "error");
  panel.classList.toggle("is-success", state.tone === "success");
  const label = el("import-progress-label");
  const bar = el("import-progress-bar");
  const note = el("import-progress-note");
  if (label) label.textContent = state.label || "";
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, Number(state.percent || 0)))}%`;
  if (note) note.textContent = state.note || "";
  document.querySelectorAll('[data-action="trigger-import"]').forEach((button) => {
    button.disabled = !!state.busy;
    button.setAttribute("aria-busy", state.busy ? "true" : "false");
  });
}

function resetInstanceDerivedState() {
  invalidateReviewReadRequest();
  reviewDataClient.reset();
  app.reviewRead = emptyReviewRead();
  app.pendingGantts.delete("gantt-review-compare");
  app.ganttViewWindows = {};
  app.graphMeta = null;
  app.graphNodes = [];
  app.graphEdges = [];
  app.graphOrderOptions = [];
  app.selectedGraphOrderId = null;
  app.selectedGraphNodeId = null;
  stopGraphBuildPolling();
  app.graphBuildTaskId = null;
  app.graphBuildStatus = null;
  resetGraphView({ preserveFilters: false });
  app.simResult = null;
  app.optimizeResult = null;
  app.optimizeStatus = null;
  app.optimizeTaskId = null;
  app.referenceSolutions = [];
  app.exactReference = null;
}

async function handleRunValidation(silent = false) {
  if (app.validationBusy) return;
  app.validationBusy = true;
  try {
    // 用户点“运行/重新校验”才重算并覆盖；启动时的静默调用直接取库里已有的结论。
    app.validation = await api.validateInstance(!silent);
    if (!silent) {
      const failed = app.validation.status === "failed";
      toast(
        failed ? `校验发现 ${formatInt(app.validation.error_count)} 个错误，请查看明细。` : "数据校验完成。",
        failed ? "error" : "success",
      );
    }
  } catch (error) {
    if (!silent) toast(`数据校验失败：${error.message}`, "warning");
  } finally {
    app.validationBusy = false;
  }
  // 校验通过后自动折叠面板，只有存在错误时才保持展开
  app.validationCollapsed = app.validation ? app.validation.status !== "failed" : false;
  // Re-render validation panel on new-scene page after validation completes
  if (app.currentPage === "new-scene") {
    const box = el("new-scene-validation");
    if (box) box.innerHTML = app.currentScene ? renderValidationPanel() : "";
  }
}

async function handleImportFile(file) {
  if (!file) return;
  if (app.importBusy) {
    toast("正在导入中，请等待当前任务完成。", "warning");
    return;
  }
  app.importBusy = true;
  setImportProgress({ busy: true, percent: 0, label: "正在上传 Excel…", note: `文件：${file.name}` });
  try {
    const result = await api.importExcel(file, (percent) => {
      setImportProgress({
        busy: true,
        percent: Math.min(90, percent * 0.9),
        label: percent >= 100 ? "上传完成，正在解析并校验数据…" : `正在上传 Excel… ${percent}%`,
        note: `文件：${file.name}`,
      });
    });
    setImportProgress({ busy: true, percent: 95, label: "正在刷新实例数据…", note: "即将完成" });
    app.validation = result?.validation || null;
    await syncCurrentScene(true);
    resetInstanceDerivedState();
    const validation = app.validation;
    const failed = validation?.status === "failed";
    app.validationCollapsed = !!validation && !failed;
    setImportProgress({
      busy: false,
      percent: 100,
      tone: failed ? "error" : "success",
      label: failed ? "导入完成，但数据校验发现问题" : "导入成功",
      note: failed
        ? `发现 ${formatInt(validation.error_count)} 个错误、${formatInt(validation.warning_count)} 个警告，校验结果已在本页显示。`
        : `已加载 ${formatInt(getSceneSummary().orders)} 个订单 / ${formatInt(getSceneSummary().operations)} 道工序。`,
    });
    if (failed) {
      toast(`导入完成，但校验发现 ${formatInt(validation.error_count)} 个错误，请先修复数据。`, "error");
    } else if (validation?.status === "warning") {
      toast(`Excel 导入成功，但有 ${formatInt(validation.warning_count)} 条校验警告。`, "warning");
    } else {
      toast("Excel 导入成功，数据校验通过。", "success");
    }
    if (app.currentPage !== "new-scene") await navigate("new-scene");
    else await renderCurrentPage();
    // 强校验通过后自动构建图谱，构建进度显示在本页“图谱构建”卡片中
    if (!failed) await handleBuildGraph();
  } catch (error) {
    setImportProgress({ busy: false, percent: 100, tone: "error", label: "导入失败", note: error.message });
    toast(`导入失败：${error.message}`, "error");
  } finally {
    app.importBusy = false;
  }
}

async function loadGraphOrder(orderId) {
  const payload = await api.getGraphOrder(entityIdFromGraphId(orderId));
  app.graphNodes = asArray(payload?.nodes);
  app.graphEdges = asArray(payload?.edges);
  app.selectedGraphOrderId = payload?.order_id || `O:${entityIdFromGraphId(orderId)}`;
  app.selectedGraphNodeId = app.selectedGraphOrderId;
  resetGraphView({ preserveFilters: true });
}

async function loadPlanGantt(taskId, solutionId, orderId = null) {
  const key = `${taskId}::${solutionId}`;
  app.planGantt = { key, taskId, solutionId, orders: app.planGantt.key === key ? app.planGantt.orders : [], orderId, entries: [], totalOperations: 0, loading: true, error: null };
  await renderCurrentPage();
  try {
    const payload = await api.getOptimizeSolutionSchedule(taskId, solutionId, orderId);
    if (app.planGantt.key !== key) return; // 竞态：期间已切方案，丢弃过期响应
    app.planGantt = { key, taskId, solutionId, orders: asArray(payload?.orders), orderId: payload?.order_id || null, entries: asArray(payload?.entries), totalOperations: Number(payload?.total_operations || 0), loading: false, error: null };
  } catch (error) {
    if (app.planGantt.key === key) app.planGantt = { ...app.planGantt, loading: false, error: error.message || String(error) };
  }
  await renderCurrentPage();
}

async function loadReviewData(selected, orderId = null, includeUtilization = true) {
  const taskId = app.optimizeResult?.task_id;
  const ids = asArray(selected).map((item) => item.id).filter(Boolean);
  if (!taskId || !ids.length) return;
  const selectionKey = ReviewRuntime.selectionKey(taskId, ids);
  const requestedScheduleKey = ReviewRuntime.scheduleKey(taskId, ids, orderId);
  const requestGeneration = ++reviewReadRequestGeneration;
  pendingReviewScheduleKey = requestedScheduleKey;
  const previous = app.reviewRead;
  const selectionChanged = previous.selectionKey !== selectionKey;
  if (selectionChanged) app.pendingGantts.delete("gantt-review-compare");
  app.reviewRead = {
    ...previous,
    selectionKey,
    scheduleKey: selectionChanged ? null : previous.scheduleKey,
    orderId: selectionChanged ? null : previous.orderId,
    schemes: selectionChanged ? {} : previous.schemes,
    utilization: selectionChanged ? null : previous.utilization,
    failedIds: selectionChanged ? [] : previous.failedIds,
    failureMessages: selectionChanged ? {} : previous.failureMessages,
    loading: true,
    error: null,
  };
  if (selectionChanged) clearReviewTimeline();
  refreshReviewDynamicRegions();
  try {
    const result = await reviewDataClient.loadData({
      taskId,
      ids,
      orderId,
      includeUtilization,
    });
    if (result.cancelled) return;
    if (!isCurrentReviewReadRequest(requestGeneration, selectionKey, requestedScheduleKey)) return;
    const payload = result.payload;
    app.reviewRead = {
      selectionKey,
      scheduleKey: ReviewRuntime.scheduleKey(taskId, ids, payload.order_id),
      orderId: payload.order_id,
      schemes: payload.schemes || {},
      utilization: payload.type_utilization ?? app.reviewRead.utilization,
      failedIds: payload.failed_solution_ids || [],
      failureMessages: payload.failure_messages || {},
      loading: false,
      error: null,
    };
    pendingReviewScheduleKey = null;
  } catch (error) {
    if (isCurrentReviewReadRequest(requestGeneration, selectionKey, requestedScheduleKey)) {
      app.reviewRead = {
        ...app.reviewRead,
        loading: false,
        error: error.message || String(error),
      };
      pendingReviewScheduleKey = null;
    }
  }
  refreshReviewDynamicRegions();
}

function ensureReviewData(selected) {
  const taskId = app.optimizeResult?.task_id;
  const ids = asArray(selected).map((item) => item.id).filter(Boolean);
  if (!taskId || !ids.length) {
    if (app.reviewRead.selectionKey) {
      invalidateReviewReadRequest();
      reviewDataClient.reset();
      app.reviewRead = emptyReviewRead();
      app.pendingGantts.delete("gantt-review-compare");
      refreshReviewDynamicRegions();
    }
    return app.reviewRead;
  }
  const selectionKey = ReviewRuntime.selectionKey(taskId, ids);
  if (
    app.reviewRead.selectionKey === selectionKey &&
    (app.reviewRead.loading || app.reviewRead.scheduleKey)
  ) {
    return app.reviewRead;
  }
  loadReviewData(selected, null, true);
  return app.reviewRead;
}

// 订单过滤：下拉仅覆盖前 200 条订单，其余订单通过搜索按后端解析加载并聚焦该订单簇
async function searchGraphOrderAndRender(query) {
  const value = String(query || "").trim();
  if (!value) {
    toast("请输入或选择订单号", "warning");
    return;
  }
  const currentEntityId = entityIdFromGraphId(app.selectedGraphOrderId || "");
  if (value === app.selectedGraphOrderId || value === currentEntityId) return;
  try {
    const payload = await api.searchGraphOrder(value);
    app.graphNodes = asArray(payload?.nodes);
    app.graphEdges = asArray(payload?.edges);
    app.selectedGraphOrderId = payload?.order_id || `O:${value}`;
    app.selectedGraphNodeId = app.selectedGraphOrderId;
    resetGraphView({ preserveFilters: true });
    await renderCurrentPage();
  } catch (error) {
    toast(error.message || `没有找到该订单：${value}`, "warning");
  }
}

async function initializeGraphOrderView(meta, preferredOrderId = null) {
  // 仅拉取有限条订单用于下拉选择；任意订单都通过搜索按钮走后端解析，不在前端全量加载/筛选。
  const payload = await api.getGraphNodes(200, 0, "order");
  app.graphOrderOptions = asArray(payload?.nodes || payload)
    .sort((a, b) => String(a.entity_id || a.node_id).localeCompare(String(b.entity_id || b.node_id), "zh-CN", { numeric: true }));
  if (!app.graphOrderOptions.length) {
    app.graphNodes = [];
    app.graphEdges = [];
    app.selectedGraphOrderId = null;
    app.selectedGraphNodeId = null;
    return;
  }
  const preferred = entityIdFromGraphId(preferredOrderId || "");
  const firstOrder = app.graphOrderOptions.find((node) => node.entity_id === preferred || node.node_id === preferredOrderId)
    || app.graphOrderOptions[0];
  await loadGraphOrder(firstOrder.entity_id || firstOrder.node_id);
}

async function loadExistingGraph() {
  try {
    const meta = await api.getGraphMeta();
    if (meta?.cache_ready === false) {
      const reason = meta.invalid_reason || "实例或图上下文已变化";
      app.graphBuildStatus = {
        status: "error",
        stage: "invalid",
        message: "当前图谱已失效，请重新构建",
        error: reason,
      };
      throw new Error(`当前图谱已失效：${reason}`);
    }
    app.graphMeta = {
      ...meta,
      created_at: tryParseDate(meta?.created_at) ? meta.created_at : new Date().toISOString(),
    };
    await initializeGraphOrderView(meta, app.selectedGraphOrderId);
    return true;
  } catch (error) {
    app.graphMeta = null;
    app.graphNodes = [];
    app.graphEdges = [];
    app.graphOrderOptions = [];
    app.selectedGraphOrderId = null;
    app.selectedGraphNodeId = null;
    return false;
  }
}

// 启动时把已完成步骤的结果从后端读回来，让刷新或重启后停在原来的进度上，
// 而不是把导入→仿真→优化→评审整条流程重跑一遍。后端只返回与当前库内实例
// 匹配的快照：实例改过的话这里拿到的就是 null，对应步骤自然退回“待开始”。
async function restoreWorkflowProgress() {
  let progress;
  try {
    progress = await api.getWorkflowProgress();
  } catch (error) {
    console.warn("恢复流程进度失败，按未开始处理", error);
    return;
  }

  if (progress.validation) {
    app.validation = progress.validation;
    app.validationCollapsed = progress.validation.status !== "failed";
  }

  if (progress.simulation) {
    app.simResult = progress.simulation;
    app.simRule = progress.simulation.rule || app.simRule;
    app.simStatus = {
      phase: progress.simulation.diagnosis ? "done-warn" : "done",
      message: progress.simulation.diagnosis
        ? `已恢复上次仿真结果（${app.simRule}），但结果不完整：${progress.simulation.diagnosis}`
        : `已恢复上次仿真结果（${app.simRule}）。`,
      elapsedMs: 0,
    };
  }

  if (progress.optimization) {
    app.optimizeResult = progress.optimization;
    app.optimizeTaskId = progress.optimization.task_id || null;
    app.referenceSolutions = asArray(progress.optimization.reference_solutions);
    app.optimizeStatus = {
      status: "done",
      phase: "done",
      message: "已恢复上次优化结果，可直接进入方案评审。",
      archive_size: progress.optimization.archive_size || 0,
      current_generation: progress.optimization.generations_completed || 0,
      elapsed_s: progress.optimization.elapsed_s || 0,
      total_evaluations: progress.optimization.total_evaluations || 0,
      received_at: Date.now() / 1000,
    };
    if (Array.isArray(progress.optimization.objective_keys) && progress.optimization.objective_keys.length) {
      app.optimizeForm.objectiveKeys = progress.optimization.objective_keys;
    }
  }

  if (progress.review) {
    app.reviewSelection = asArray(progress.review.selection);
    app.reviewDetailId = progress.review.detail_id || null;
    app.aiLastRecommendedId = progress.review.ai_recommended_id || null;
  }
  // 恢复的选择可能指向已不存在的方案（比如只恢复了评审、优化结果已失效），
  // ensureReviewSelection 会剔除这些 id 并补上默认选择。
  ensureReviewSelection();
}

// 评审页的选择本身就是一步进度，改动即存，重启后回到同样的视图。
function persistReviewProgress() {
  api.saveReviewProgress({
    selection: app.reviewSelection,
    detail_id: app.reviewDetailId,
    ai_recommended_id: app.aiLastRecommendedId,
  }).catch((error) => console.warn("保存评审选择失败", error));
}

function stopGraphBuildPolling() {
  if (app.graphBuildPollTimer) window.clearTimeout(app.graphBuildPollTimer);
  app.graphBuildPollTimer = null;
}

async function refreshGraphBuildFeedback() {
  const panel = el("graph-build-status-panel");
  if (panel) panel.outerHTML = renderGraphBuildStatus();
  else if (app.currentPage === "new-scene") await renderCurrentPage();
  syncGraphBuildControls();
}

async function finishGraphBuild(status) {
  app.graphBuildStatus = {
    ...status,
    status: "running",
    stage: "loading",
    progress: 98,
    message: "后端构建成功，正在加载首个订单",
  };
  await refreshGraphBuildFeedback();
  const meta = await api.getGraphMeta();
  if (meta?.cache_ready === false) {
    throw new Error(meta.invalid_reason || "图谱缓存未就绪");
  }
  app.graphMeta = {
    ...meta,
    created_at: tryParseDate(meta?.created_at) ? meta.created_at : new Date().toISOString(),
  };
  await initializeGraphOrderView(meta);
  app.graphBuildStatus = { ...status, status: "done", stage: "done", progress: 100, message: "图谱构建并加载成功" };
  toast(`图谱构建完成：${formatInt(meta.total_nodes)} 个节点、${formatInt(meta.total_edges)} 条边。`, "success");
  await renderCurrentPage();
}

async function pollGraphBuildStatus() {
  if (!app.graphBuildTaskId) return;
  try {
    const status = await api.getGraphBuildStatus(app.graphBuildTaskId);
    app.graphBuildStatus = status;
    app.graphBuildPollFailures = 0;
    await refreshGraphBuildFeedback();
    if (status.status === "done") {
      stopGraphBuildPolling();
      try {
        await finishGraphBuild(status);
      } catch (error) {
        app.graphBuildStatus = { ...status, status: "error", stage: "loading", error: `图谱已构建，但加载可视化样本失败：${error.message}` };
        toast(app.graphBuildStatus.error, "warning");
        await renderCurrentPage();
      }
      return;
    }
    if (status.status === "error") {
      stopGraphBuildPolling();
      toast(`构建图谱失败：${status.error || status.message || "未知错误"}`, "warning");
      await renderCurrentPage();
      return;
    }
    const clientLimit = Number(status.timeout_s || 180) + 30;
    if (Number(status.elapsed_s || 0) > clientLimit) {
      stopGraphBuildPolling();
      app.graphBuildStatus = { ...status, status: "error", stage: "timeout", error: `超过 ${clientLimit} 秒仍未收到完成结果，请检查服务日志后重试` };
      toast(app.graphBuildStatus.error, "warning");
      await renderCurrentPage();
      return;
    }
  } catch (error) {
    app.graphBuildPollFailures += 1;
    if (app.graphBuildPollFailures >= 3) {
      stopGraphBuildPolling();
      app.graphBuildStatus = { ...(app.graphBuildStatus || {}), status: "error", stage: "connection", error: `连续 3 次无法获取构建进度：${error.message}` };
      toast(app.graphBuildStatus.error, "warning");
      await renderCurrentPage();
      return;
    }
  }
  app.graphBuildPollTimer = window.setTimeout(pollGraphBuildStatus, 1000);
}

async function handleBuildGraph() {
  if (graphBuildIsRunning()) {
    toast("图谱正在构建，请等待当前任务完成。", "warning");
    return;
  }
  try {
    const result = await api.buildGraph();
    app.graphBuildTaskId = result.task_id;
    app.graphBuildStatus = {
      status: result.status || "queued",
      stage: "queued",
      progress: 0,
      message: result.message || "图谱构建任务已提交",
      elapsed_s: 0,
      timeout_s: result.timeout_s || 180,
    };
    app.graphBuildPollFailures = 0;
    toast("图谱构建任务已提交，页面将持续显示详细进度。", "info");
    await renderCurrentPage();
    stopGraphBuildPolling();
    await pollGraphBuildStatus();
  } catch (error) {
    app.graphBuildStatus = { status: "error", stage: "submit", progress: 0, error: `无法启动图谱构建：${error.message}` };
    toast(app.graphBuildStatus.error, "warning");
    await renderCurrentPage();
  }
}

function syncSimulateControls() {
  document.querySelectorAll('[data-action="run-simulate"]').forEach((button) => {
    button.disabled = app.simBusy;
    button.setAttribute("aria-busy", app.simBusy ? "true" : "false");
    button.textContent = app.simBusy ? "仿真计算中…" : "运行仿真";
  });
}

// 仿真运行状态区：运行中显示进度条 + 实时计时，完成/失败/不完整都明确留在界面上，
// 不再只依赖一闪而过的 toast，避免用户“没有任何反应、不知道成败”。
function renderSimStatusInner(status) {
  if (!status || status.phase === "idle") return "";
  const isRunning = status.phase === "running";
  const cls = isRunning ? "" : status.phase === "done" ? "is-success" : status.phase === "done-warn" ? "is-warning" : "is-error";
  const icon = isRunning
    ? `<span class="import-spinner"></span>`
    : status.phase === "done" ? `<span class="sim-status-icon">✓</span>`
    : status.phase === "done-warn" ? `<span class="sim-status-icon">⚠</span>`
    : `<span class="sim-status-icon">✗</span>`;
  const title = isRunning ? "仿真运行中" : status.phase === "done" ? "仿真完成" : status.phase === "done-warn" ? "仿真完成（结果不完整）" : "仿真失败";
  const elapsed = `<span class="sim-elapsed" id="sim-elapsed">${formatDurationMs(status.elapsedMs || 0)}</span>`;
  const track = `<div class="import-progress-track ${isRunning ? "indeterminate" : ""}"><i${isRunning ? "" : ' style="width:100%"'}></i></div>`;
  return `
    <article class="import-progress ${cls}">
      <div class="import-progress-head">
        ${icon}
        <strong>${title}</strong>
        ${elapsed}
      </div>
      ${track}
      <p class="import-progress-note" id="sim-status-note">${escapeHtml(status.message || "")}</p>
      ${app.simResult && !isRunning ? `<div class="form-actions" style="margin-top:10px"><button class="btn btn-secondary" type="button" data-action="export-sim-excel">导出仿真结果 Excel</button></div>` : ""}
    </article>`;
}

function paintSimStatus() {
  const node = el("sim-status");
  if (!node) return;
  node.innerHTML = renderSimStatusInner(app.simStatus);
  node.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
}

async function handleSimulate() {
  if (app.simBusy) return;
  app.simBusy = true;
  const startedAt = Date.now();
  app.simStatus = {
    phase: "running",
    message: `正在运行规则仿真（${app.simRule}）：提交请求并初始化排程上下文…`,
    elapsedMs: 0,
  };
  syncSimulateControls();
  paintSimStatus();
  const stageTimer = window.setInterval(() => {
    if (app.simStatus && app.simStatus.phase === "running") {
      app.simStatus.elapsedMs = Date.now() - startedAt;
      const span = el("sim-elapsed");
      if (span) span.textContent = formatDurationMs(app.simStatus.elapsedMs);
    }
  }, 300);
  app.simElapsedTimer = stageTimer;
  try {
    app.simRule = el("workflow-sim-rule")?.value || app.simRule;
    app.simStatus.message = `规则引擎计算中（${app.simRule}）…`;
    const note = el("sim-status-note");
    if (note) note.textContent = app.simStatus.message;
    app.simResult = await api.simulate(app.simRule);
    const elapsedMs = Date.now() - startedAt;
    const diagnosis = app.simResult?.diagnosis;
    // 调试输出：确认后端计算真实执行、关键中间值是否为 0
    const metrics = app.simResult?.metrics || {};
    console.info("[simulate]", app.simRule, {
      scheduled: asArray(app.simResult?.gantt).length,
      completed_operations: metrics.completed_operations,
      total_operations: metrics.total_operations,
      makespan: metrics.makespan,
      total_tardiness: metrics.total_tardiness,
      avg_net_available_utilization: metrics.avg_net_available_utilization,
      feasible: metrics.feasible,
      diagnosis,
    });
    app.simStatus = diagnosis
      ? { phase: "done-warn", message: `仿真完成，但结果不完整：${diagnosis}`, elapsedMs }
      : { phase: "done", message: `规则仿真已完成（${app.simRule}），耗时 ${formatDurationMs(elapsedMs)}。`, elapsedMs };
    if (diagnosis) toast(`仿真完成，但结果不完整：${diagnosis}`, "error");
    else toast(`规则仿真已完成：${app.simRule}`, "success");
    await renderCurrentPage();
  } catch (error) {
    app.simStatus = {
      phase: "error",
      message: `运行仿真失败：${error.message}。请确认已生成/导入实例，或查看浏览器控制台与后端日志。`,
      elapsedMs: Date.now() - startedAt,
    };
    toast(`运行仿真失败：${error.message}`, "error");
    await renderCurrentPage();
  } finally {
    window.clearInterval(stageTimer);
    app.simElapsedTimer = null;
    app.simBusy = false;
    syncSimulateControls();
  }
}

async function handleStartOptimize() {
  const payload = collectOptimizeForm();
  if (!payload.objective_keys.length) {
    toast("请至少选择 1 个优化目标。", "warning");
    return;
  }
  app.optimizeStatus = {
    status: "submitting",
    phase: "submitting",
    message: "正在提交参数并创建优化任务",
    elapsed_s: 0,
    updated_at: Date.now() / 1000,
  };
  app.optimizeTaskId = null;
  app.optimizePollFailures = 0;
  await renderCurrentPage();
  toast("正在提交优化任务，请稍候…", "info");
  try {
    const result = await api.startHybridOptimize(payload);
    app.optimizeTaskId = result.task_id;
    app.optimizeStatus = {
      status: "running",
      phase: "initializing",
      message: "任务已创建，正在初始化优化器",
      config: result.config || payload,
      elapsed_s: 0,
      updated_at: Date.now() / 1000,
    };
    app.optimizeResult = null;
    app.referenceSolutions = [];
    app.exactReference = null;
    startOptimizePolling();
    await renderCurrentPage();
    toast(`优化任务 ${result.task_id} 已启动，页面将持续更新运行状态。`, "success");
    await pollOptimizeStatus();
  } catch (error) {
    app.optimizeStatus = {
      status: "error",
      phase: "submit",
      message: "优化任务未能启动",
      error: error.message,
      error_type: "SubmitError",
      updated_at: Date.now() / 1000,
    };
    await renderCurrentPage();
    toast(`启动优化失败：${error.message}`, "error");
    showErrorModal("启动优化失败", error.message);
  }
}

async function handleGenerateExact(mode) {
  if (!app.optimizeTaskId) {
    toast("请先运行一次混合优化，再生成精确冠军参考方案。", "warning");
    return;
  }
  app.exactForm.timeLimitS = Number(el("exact-time-limit")?.value || app.exactForm.timeLimitS);
  app.exactForm.objectiveKey = el("exact-single-objective")?.value || app.exactForm.objectiveKey;
  const weights = {};
  document.querySelectorAll("[data-weight-key]").forEach((node) => {
    weights[node.dataset.weightKey] = Number(node.value || 0);
  });
  app.exactForm.weights = weights;
  try {
    const result = await api.createExactReference({
      task_id: app.optimizeTaskId,
      mode,
      objective_key: app.exactForm.objectiveKey,
      objective_weights: mode === "weighted" ? weights : {},
      time_limit_s: app.exactForm.timeLimitS,
    });
    app.exactReference = result.solution;
    ensureReviewSelection();
    toast("精确冠军参考方案已生成。", "success");
    await renderCurrentPage();
  } catch (error) {
    toast(`生成精确冠军失败：${error.message}`, "warning");
  }
}

async function handleExportSolution(solutionId) {
  const candidate = getReviewCandidates().find((item) => item.id === solutionId) || getSelectedReviewCandidate();
  if (!candidate) {
    toast("请先选择一个方案。", "warning");
    return;
  }
  if (!app.optimizeTaskId) {
    toast("当前导出依赖优化任务上下文，请先运行优化。", "warning");
    return;
  }
  try {
    const blob = await api.exportOptimizeSolution(app.optimizeTaskId, candidate.id);
    downloadBlob(blob, `${candidate.id || "solution"}.xlsx`);
    toast("方案导出成功。", "success");
  } catch (error) {
    toast(`导出失败：${error.message}`, "warning");
  }
}

function pushAiMessage(role, content) {
  app.aiConversation.push({ role, content });
}

async function handleAiAction(mode) {
  if (!app.optimizeTaskId) {
    toast("请先完成一次优化，再使用 AI 方案评审。", "warning");
    return;
  }
  const input = el("ai-input");
  const question = input?.value?.trim() || "";
  const solutionSelect = el("ai-solution-select");
  const selection = buildAiSelection();
  if (!selection.solution_ids.length && !selection.heuristic_rule_names.length) {
    toast("请先在方案库中勾选 1-4 个方案。", "warning");
    return;
  }
  if (mode === "compare" && selection.solution_ids.length + selection.heuristic_rule_names.length < 2) {
    toast("比较方案时至少需要选择 2 个方案。", "warning");
    return;
  }
  if (mode !== "compare" && !question) {
    toast("请先输入诉求或问题。", "warning");
    return;
  }
  app.aiBusy = true;
  pushAiMessage("user", mode === "compare" ? "请比较当前已勾选方案。" : question);
  pushAiMessage("assistant", mode === "compare" ? "正在比较已勾选方案，请稍候……" : "正在分析你的诉求，请稍候……");
  await renderCurrentPage();
  try {
    let result;
    if (mode === "compare") {
      result = await api.aiCompare({
        task_id: app.optimizeTaskId,
        solution_ids: selection.solution_ids,
        heuristic_rule_names: selection.heuristic_rule_names,
        requirement: question,
        conversation: app.aiConversation,
      });
    } else if (mode === "recommend") {
      result = await api.aiRecommend({
        task_id: app.optimizeTaskId,
        solution_ids: selection.solution_ids,
        heuristic_rule_names: selection.heuristic_rule_names,
        requirement: question,
        conversation: app.aiConversation,
      });
      app.aiLastRecommendedId = result.analysis?.recommended_solution_id || app.aiLastRecommendedId;
      if (app.aiLastRecommendedId) app.reviewDetailId = app.aiLastRecommendedId;
      persistReviewProgress();
    } else {
      result = await api.aiAsk({
        task_id: app.optimizeTaskId,
        solution_id: solutionSelect?.value,
        heuristic_rule_names: selection.heuristic_rule_names,
        question,
        conversation: app.aiConversation,
      });
    }
    app.aiConversation.pop();
    pushAiMessage("assistant", result.display_text || "AI 已返回结果。");
    input.value = "";
  } catch (error) {
    app.aiConversation.pop();
    pushAiMessage("assistant", `分析失败：${error.message}`);
  } finally {
    app.aiBusy = false;
    await renderCurrentPage();
  }
}

async function handleSaveLlmConfig() {
  try {
    await api.setLlmConfig({
      base_url: el("llm-base-url").value || null,
      api_key: el("llm-api-key").value || null,
      model: el("llm-model").value || null,
    });
    app.llmConfig = await api.getLlmConfig();
    toast("大模型配置已保存。", "success");
    renderSystem();
  } catch (error) {
    toast(`保存配置失败：${error.message}`, "warning");
  }
}

async function handleTestLlmConfig() {
  try {
    const result = await api.testLlmConfig();
    toast(result?.message || "大模型连接测试通过。", "success");
  } catch (error) {
    toast(`连接测试失败：${error.message}`, "warning");
  }
}

async function handleAction(action, target) {
  if (action === "retry-plan-gantt") return loadPlanGantt(app.planGantt.taskId, app.planGantt.solutionId, app.planGantt.orderId);
  if (action === "goto-new-scene") return navigate("new-scene");
  if (action === "goto-dashboard") return navigate("dashboard");
  if (action === "goto-review") return navigate("solution-review");
  if (action === "graph-order-search") {
    const root = target.closest(".graph-workbench");
    const input = root?.querySelector("[data-graph-order-input]");
    await searchGraphOrderAndRender(input?.value);
    return;
  }
  if (action === "focus-graph-node") {
    if (app.graphSuppressClickUntil && Date.now() < app.graphSuppressClickUntil) return;
    app.selectedGraphNodeId = target.dataset.id;
    return renderCurrentPage();
  }
  if (action === "set-graph-mode") {
    app.graphView.mode = target.dataset.mode || "focus";
    return renderCurrentPage();
  }
  if (action === "toggle-graph-node-type") {
    const key = target.dataset.key;
    app.graphView.nodeTypes[key] = !app.graphView.nodeTypes[key];
    return renderCurrentPage();
  }
  if (action === "toggle-graph-edge-group") {
    const key = target.dataset.key;
    app.graphView.edgeGroups[key] = !app.graphView.edgeGroups[key];
    return renderCurrentPage();
  }
  if (action === "zoom-graph-in") {
    app.graphView.zoom = Math.min(2.4, app.graphView.zoom * 1.15);
    return renderCurrentPage();
  }
  if (action === "zoom-graph-out") {
    app.graphView.zoom = Math.max(0.45, app.graphView.zoom * 0.87);
    return renderCurrentPage();
  }
  if (action === "reset-graph-view") {
    const mode = app.graphView.mode;
    const search = app.graphView.search;
    const nodeTypes = { ...(app.graphView.nodeTypes || {}) };
    const edgeGroups = { ...(app.graphView.edgeGroups || {}) };
    resetGraphView({ preserveFilters: false });
    app.graphView.mode = mode;
    app.graphView.search = search;
    app.graphView.nodeTypes = nodeTypes;
    app.graphView.edgeGroups = edgeGroups;
    return renderCurrentPage();
  }
  if (action === "fit-graph-view") {
    fitGraphViewport(app.graphView.bounds);
    return renderCurrentPage();
  }
  if (action === "sync-current-scene") {
    await syncCurrentScene();
    return renderCurrentPage();
  }
  if (action === "trigger-import") return el("import-file").click();
  if (action === "download-template") {
    const blob = await api.downloadTemplate();
    downloadBlob(blob, "instance_template_v2.xlsx");
    toast("模板已下载。", "success");
    return;
  }
  if (action === "export-csv") {
    const blob = await api.exportCsv();
    downloadBlob(blob, "instance.csv");
    toast("CSV 已导出。", "success");
    return;
  }
  if (action === "run-validation") return handleRunValidation();
  if (action === "toggle-validation-collapse") {
    app.validationCollapsed = !app.validationCollapsed;
    const box = el("new-scene-validation");
    if (box) box.innerHTML = app.currentScene ? renderValidationPanel() : "";
    return;
  }
  if (action === "export-validation") {
    const blob = await api.exportValidation();
    downloadBlob(blob, `validation_result_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}.xlsx`);
    toast("校验结果 Excel 已导出。", "success");
    return;
  }
  if (action === "export-sim-excel") {
    try {
      const blob = await api.exportSimExcel();
      downloadBlob(blob, `sim_result_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}.xlsx`);
      toast("仿真结果 Excel 已开始下载。", "success");
    } catch (error) {
      toast(`导出失败：${error.message}`, "error");
    }
    return;
  }
  if (action === "build-graph") return handleBuildGraph();
  if (action === "run-simulate") return handleSimulate();
  if (action === "start-hybrid-optimize") return handleStartOptimize();
  if (action === "apply-budget-recommendation") {
    app.optimizeForm.timeLimitTouched = false;
    app.optimizeForm.timeLimitS = refreshOptimizeBudgetRecommendation({ preserveManual: false });
    if (el("opt-time-limit")) el("opt-time-limit").value = app.optimizeForm.timeLimitS;
    updateOptimizeBudgetHint();
    toast(`已恢复建议预算 ${app.optimizeForm.timeLimitS} 秒。`, "success");
    return;
  }
  if (action === "load-heuristic-rule") {
    const rule = target.dataset.rule;
    if (!rule) return;
    if (rule === app.optimizeResult?.baseline?.rule_name) {
      toast(`规则 ${rule} 已作为基线方案纳入对比。`, "info");
      return;
    }
    if (ruleIsCached(rule) || app.referenceSolutionsState.computing.includes(rule)) return; // 已在池中或计算中
    return computeReferenceSolution(rule);
  }
  if (action === "toggle-candidate") {
    const id = target.dataset.id;
    if (!id) return;
    if (app.reviewSelection.includes(id)) {
      app.reviewSelection = app.reviewSelection.filter((item) => item !== id);
      if (app.reviewDetailId === id) app.reviewDetailId = app.reviewSelection[0] || null;
    } else {
      if (app.reviewSelection.length >= 4) {
        toast("AI 评审最多同时选择 4 个方案。", "warning");
        return;
      }
      app.reviewSelection.push(id);
      app.reviewDetailId = id;
    }
    persistReviewProgress();
    updateShell();
    const comparisonRegion = el("review-comparison-region");
    if (comparisonRegion) comparisonRegion.innerHTML = renderReviewCandidateComparison();
    const selectedCount = el("review-selected-count");
    if (selectedCount) selectedCount.textContent = formatInt(app.reviewSelection.length);
    return ensureReviewData(getSelectedReviewCandidates());
  }
  if (action === "retry-type-utilization") {
    return loadReviewData(getSelectedReviewCandidates(), app.reviewRead.orderId, true);
  }
  if (action === "retry-review-gantt") {
    return loadReviewData(getSelectedReviewCandidates(), app.reviewRead.orderId, true);
  }
  if (action === "generate-exact-single") return handleGenerateExact("single");
  if (action === "generate-exact-weighted") return handleGenerateExact("weighted");
  if (action === "export-selected-solution") return handleExportSolution(target?.dataset.id || getSelectedReviewCandidate()?.id);
  if (action === "focus-candidate") {
    app.reviewDetailId = target.dataset.id;
    persistReviewProgress();
    updateShell();
    return renderCurrentPage();
  }
  if (action === "send-candidate-to-ai") {
    const id = target.dataset.id;
    if (id) {
      const nextSelection = [id, ...app.reviewSelection.filter((item) => item !== id)].slice(0, 4);
      app.reviewSelection = nextSelection;
    }
    app.reviewDetailId = id;
    app.reviewTab = "ai";
    persistReviewProgress();
    updateShell();
    return navigate("ai-review");
  }
  if (action === "ai-compare") return handleAiAction("compare");
  if (action === "ai-recommend") return handleAiAction("recommend");
  if (action === "ai-ask") return handleAiAction("ask");
  if (action === "save-llm-config") return handleSaveLlmConfig();
  if (action === "test-llm-config") return handleTestLlmConfig();
  if (action === "goto-workflow-step") {
    const step = Number(target.dataset.step || 1);
    const stepNavKey = { 3: "simulate", 4: "optimize-launch" }[step];
    if (stepNavKey) return navigate(stepNavKey);
    app.workflowStep = step;
    return navigate("workflow");
  }
  if (action === "toggle-graph-fullscreen") {
    const shell = target.closest(".graph-workbench") || document.querySelector(".graph-workbench");
    if (!shell || !shell.requestFullscreen) {
      toast("当前浏览器不支持图谱全屏显示。", "warning");
      return;
    }
    if (document.fullscreenElement === shell) {
      if (document.exitFullscreen) await document.exitFullscreen();
      return;
    }
    if (document.fullscreenElement && document.exitFullscreen) await document.exitFullscreen();
    await shell.requestFullscreen();
    return;
  }
}

function bindGlobalEvents() {
  document.addEventListener("fullscreenchange", () => {
    const fullscreenGraph = document.fullscreenElement?.classList.contains("graph-workbench")
      ? document.fullscreenElement
      : null;
    document.querySelectorAll('[data-action="toggle-graph-fullscreen"]').forEach((button) => {
      const active = !!fullscreenGraph && button.closest(".graph-workbench") === fullscreenGraph;
      button.textContent = active ? "退出全屏" : "全屏查看";
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
  });

  document.addEventListener("click", async (event) => {
    const ganttPageTarget = event.target.closest("[data-gantt-page]");
    if (ganttPageTarget && !ganttPageTarget.disabled) {
      event.preventDefault();
      const canvasId = ganttPageTarget.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.page = Number(ganttPageTarget.dataset.ganttPage) || 1;
      app.ganttMachineFilter[canvasId] = filter;
      renderCurrentPage();
      return;
    }
    const reviewTabTarget = event.target.closest("[data-review-tab]");
    if (reviewTabTarget) {
      event.preventDefault();
      app.reviewTab = reviewTabTarget.dataset.reviewTab;
      renderReview();
      return;
    }
    const systemTabTarget = event.target.closest("[data-system-tab]");
    if (systemTabTarget) {
      event.preventDefault();
      app.systemTab = systemTabTarget.dataset.systemTab;
      renderSystem();
      return;
    }
    const navParentTarget = event.target.closest(".nav-parent");
    if (navParentTarget) {
      event.preventDefault();
      const navKey = navParentTarget.dataset.nav;
      const groupKey = "optimize";
      if (app.currentNav === navKey) {
        const nextExpanded = !app.sidebarExpanded[groupKey];
        expandSidebarGroup(groupKey, nextExpanded);
        syncSidebarHierarchy();
      } else {
        expandSidebarGroup(groupKey, true);
        await navigate(navKey);
      }
      return;
    }
    const actionTarget = event.target.closest("[data-action]");
    if (actionTarget) {
      event.preventDefault();
      await handleAction(actionTarget.dataset.action, actionTarget);
      return;
    }
    const navTarget = event.target.closest("[data-nav]");
    if (navTarget) {
      event.preventDefault();
      await navigate(navTarget.dataset.nav);
      return;
    }
    const jumpTarget = event.target.closest("[data-nav-jump]");
    if (jumpTarget) {
      event.preventDefault();
      await navigate(jumpTarget.dataset.navJump);
    }
  });

  document.addEventListener("change", async (event) => {
    const target = event.target;
    if (target.matches("#import-file")) {
      await handleImportFile(target.files?.[0]);
      target.value = "";
      return;
    }
    if (target.matches("[data-objective-key]")) {
      const selected = Array.from(document.querySelectorAll("[data-objective-key]"))
        .filter((node) => node.checked)
        .map((node) => node.dataset.objectiveKey);
      if (selected.length > 5) {
        target.checked = false;
        toast("优化目标最多选择 5 个。", "warning");
        return;
      }
      app.optimizeForm.objectiveKeys = selected;
      updateOptimizeBudgetHint();
      return;
    }
    if (target.matches("[data-gantt-group-mode]")) {
      app.ganttGroupMode[target.dataset.canvas] = target.value;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-machine-type]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.type = target.value;
      filter.page = 1;
      app.ganttMachineFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-downtime-only]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.downtimeOnly = target.checked;
      filter.page = 1;
      app.ganttMachineFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-machine-query]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttMachineFilter(canvasId);
      filter.query = target.value;
      filter.page = 1;
      app.ganttMachineFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-status-filter]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttEntryFilter(canvasId);
      filter.status = target.value;
      app.ganttEntryFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-entry-query]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttEntryFilter(canvasId);
      filter.query = target.value;
      app.ganttEntryFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-gantt-time-from], [data-gantt-time-to]")) {
      const canvasId = target.dataset.canvas;
      const filter = getGanttEntryFilter(canvasId);
      if (target.hasAttribute("data-gantt-time-from")) filter.from = target.value;
      else filter.to = target.value;
      app.ganttEntryFilter[canvasId] = filter;
      return renderCurrentPage();
    }
    if (target.matches("[data-graph-order-select]")) {
      await loadGraphOrder(target.value);
      return renderCurrentPage();
    }
    if (target.matches("[data-graph-order-input]")) {
      const value = String(target.value || "").trim();
      if (value) await searchGraphOrderAndRender(value);
      return;
    }
    if (target.matches("#workflow-sim-rule")) app.simRule = target.value;
    if (target.matches("#ai-solution-select")) {
      app.reviewDetailId = target.value;
      persistReviewProgress();
      updateShell();
    }
    if (target.matches("#opt-target-count, #opt-population, #opt-generations, #opt-coarse-ratio, #opt-refine-rounds, #opt-alns-aggression, #opt-baseline-rule")) {
      collectOptimizeForm();
      updateOptimizeBudgetHint();
      return;
    }
    if (target.matches("#opt-time-limit")) {
      app.optimizeForm.timeLimitTouched = true;
      app.optimizeForm.timeLimitS = Number(target.value || app.optimizeForm.timeLimitS);
      updateOptimizeBudgetHint();
    }
  });

  document.addEventListener("input", (event) => {
    const target = event.target;
    if (target.matches("[data-graph-search]")) {
      app.graphView.search = target.value || "";
      renderCurrentPage();
      window.setTimeout(() => {
        const searchInput = document.querySelector("[data-graph-search]");
        if (searchInput) {
          searchInput.focus();
          const end = searchInput.value.length;
          searchInput.setSelectionRange(end, end);
        }
      }, 0);
    }
    if (target.matches("#opt-target-count, #opt-population, #opt-generations, #opt-coarse-ratio, #opt-refine-rounds, #opt-alns-aggression")) {
      collectOptimizeForm();
      updateOptimizeBudgetHint();
      return;
    }
    if (target.matches("#opt-time-limit")) {
      app.optimizeForm.timeLimitTouched = true;
      app.optimizeForm.timeLimitS = Number(target.value || app.optimizeForm.timeLimitS);
      updateOptimizeBudgetHint();
    }
  });

  window.addEventListener("hashchange", async () => {
    const navKey = window.location.hash.replace("#", "") || (app.currentScene ? "dashboard" : "new-scene");
    await navigate(navKey, false);
  });
}

async function init() {
  loadSceneHistory();
  bindGlobalEvents();
  await loadCatalogs();
  await syncCurrentScene(true);
  // 用户可能在初始化尚未完成时就开始导入 Excel：此时不再抢占页面，
  // 导入流程会自行渲染“数据导入”页，避免导入途中被切到“工作概览”。
  if (app.importBusy) return;
  // 先恢复进度再导航：仿真/优化结果决定了首屏该停在流程的哪一步。
  if (app.currentScene) await restoreWorkflowProgress();
  // 先落到目标页面再加载图谱：图谱加载较慢，放在导航之前会推迟首屏。
  const navKey = window.location.hash.replace("#", "") || (app.currentScene ? "dashboard" : "new-scene");
  await navigate(navKey, false);
  if (!app.currentScene || app.importBusy) return;
  // 库里没有可用的校验结论（如实例刚被改过）时才补算一次。
  if (!app.validation) await handleRunValidation(true);
  if (await loadExistingGraph() && !app.importBusy) await renderCurrentPage();
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    console.error(error);
    toast(`应用初始化失败：${error.message}`, "warning");
  });
});
