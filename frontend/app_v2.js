const CONFIG = {
  API_BASE: "/api",
  HISTORY_KEY: "llm4drd_v2_scene_history",
  SIDEBAR_COLLAPSED_KEY: "llm4drd_v2_sidebar_collapsed",
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
const ORDER_RECENT_LIMIT = 8;

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
  // 重构 v3：9 个一级页面；工作流只由顶部流程进度条表达，侧栏无步骤编号。
  import: { page: "import" },
  graph: { page: "graph", requiresScene: true },
  simulate: { page: "simulate", requiresScene: true },
  optimize: { page: "optimize", requiresScene: true },
  review: { page: "review", reviewTab: "library", requiresScene: true },
  dashboard: { page: "dashboard", requiresScene: true },
  export: { page: "export", requiresScene: true },
  llm: { page: "llm" },
  system: { page: "system" },
  // 旧书签 hash 兼容映射
  "new-scene": { page: "import" },
  "scene-library": { page: "import" },
  workflow: { page: "simulate", requiresScene: true },
  "optimize-config": { page: "optimize", requiresScene: true },
  "optimize-launch": { page: "optimize", requiresScene: true },
  "solution-review": { page: "review", reviewTab: "library", requiresScene: true },
  "pareto-library": { page: "review", reviewTab: "library", requiresScene: true },
  "exact-reference": { page: "review", reviewTab: "exact", requiresScene: true },
  "ai-review": { page: "review", reviewTab: "ai", requiresScene: true },
  "llm-config": { page: "llm" },
  "export-data": { page: "export", requiresScene: true },
  settings: { page: "system" },
};

const app = {
  currentPage: "import",
  currentNav: "import",
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
  // 流程进度条聚合状态：来自 GET /api/workflow/overview，带节流时间戳
  workflowOverview: null,
  workflowOverviewAt: 0,
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
  cyGraphInstance: null,
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
  // LLM 测试连接的最近一次结果（顶栏/大模型页状态条使用）
  llmTestResult: null,
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
  // 甘特图订单多选筛选：canvasId -> 选中的订单 id 数组（空数组 = 全部订单）
  ganttOrderMulti: {},
  // Excel 风格多选订单筛选组件的已注册数据源：id -> config
  multiSelectSources: new Map(),
  // 通用单选可搜索下拉筛选组件的已注册数据源：id -> config
  singleSelectSources: new Map(),
  // 评审甘特：当前选中的单个方案 id（与勾选集 reviewSelection 解耦）
  reviewGanttSchemeId: null,
  // 方案对比「列配置」面板展开态（跨区域重渲染保持）
  reviewColConfigOpen: false,
  // 评审甘特：单方案完整排程缓存 schemeId -> { loading, error, entries, orders }
  reviewSchemeCache: {},
  // 评审：某方案每日机器分类利用率缓存 schemeId -> { loading, error, days, types }
  reviewDailyUtilCache: {},
  // 方案每日利用率趋势详情：当前展开的方案 id（null = 未展开）
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
  orderComboboxRecent: ReviewRuntime.createRecentOrderStore({
    contextLimit: 24,
    itemLimit: ORDER_RECENT_LIMIT,
  }),
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
  getWorkflowOverview() { return this.json("/workflow/overview"); },
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
  exportSimExcel(payload = null) {
    // 回传前端已持有的排产明细（app.simResult 或某方案的 schedule），后端据此生成 Excel，
    // 避免依赖易失效的后端 last_sim_payload。payload 为空时后端回退到最近一次仿真结果。
    return this.request("/simulate/export-excel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload ? JSON.stringify(payload) : undefined,
    });
  },
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
  getMachineTypeDailyUtilization(taskId, solutionId) {
    const params = new URLSearchParams({ solution_id: solutionId });
    return this.json(`/optimize/hybrid/result/${encodeURIComponent(taskId)}/machine-type-daily-utilization?${params.toString()}`);
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
  // 订单一律按订单号展示，不再拼接订单名。
  return order.order_id || "";
}

function rememberOrderComboboxSelection(recentKey, order) {
  app.orderComboboxRecent.record(recentKey, order);
}

function renderOrderCombobox(config) {
  const recentKey = `${config.id}::${config.contextKey}`;
  rememberOrderComboboxSelection(recentKey, config.selected);
  app.orderComboboxSources.set(config.id, { ...config, recentKey });
  const listId = `${config.id}-list`;
  const labelOf = config.label || orderComboboxLabel;
  return `
    <div class="order-combobox" data-order-combobox="${escapeHtml(config.id)}">
      <input type="search" role="combobox" aria-autocomplete="list"
        aria-expanded="false" aria-controls="${escapeHtml(listId)}"
        value="${escapeHtml(labelOf(config.selected))}" placeholder="输入订单号模糊搜索">
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

    const labelOf = source.label || orderComboboxLabel;
    let controller;
    const renderState = (state) => {
      list.innerHTML = state.results.map((order, index) => {
        const optionId = `${source.id}-option-${index}`;
        const active = index === state.activeIndex;
        return `<button type="button" id="${escapeHtml(optionId)}" class="order-combobox-option${active ? " is-active" : ""}" role="option" aria-selected="${active ? "true" : "false"}" data-order-result="${index}">${escapeHtml(labelOf(order))}</button>`;
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
      current: source.selected,
      recent: app.orderComboboxRecent.read(source.recentKey),
      select: async (order) => {
        input.value = labelOf(order);
        rememberOrderComboboxSelection(source.recentKey, order);
        await source.select(order);
      },
      delay: ORDER_SEARCH_DEBOUNCE_MS,
      limit: ORDER_SEARCH_LIMIT,
      onState: renderState,
    });
    app.orderComboboxMounts.set(container, controller);

    input.addEventListener("keydown", async (event) => {
      await ReviewRuntime.handleOrderComboboxKey(controller, event);
    });

    input.addEventListener("input", () => {
      controller.input(input.value);
    });
    ReviewRuntime.bindOrderComboboxOpen(input, controller);
  });
}

// —— 通用「可搜索 + 多选」下拉筛选框（客户端）——
// 订单与机器类型两个甘特筛选共用同一组件，交互与外观一致。
// config: { id, options:[{id, label}], selectedIds:string[], onChange:(ids)=>void, noun, unit? }
// 语义：选中 0 个 = 全部；顶部「全选」勾选即选中全部；输入框按 label 模糊过滤。
// 交互性能：面板内改动只维护本地工作集，关闭面板时一次性提交（避免每次勾选都重建甘特）。
const MULTI_SELECT_RENDER_CAP = 300;

function multiSelectSummary(options, selectedIds, noun, unit = "") {
  const total = options.length;
  if (!selectedIds.length || selectedIds.length >= total) return `全部${noun}（${formatInt(total)}${unit}）`;
  if (selectedIds.length === 1) {
    const only = options.find((o) => o.id === selectedIds[0]);
    return only ? only.label : String(selectedIds[0]);
  }
  return `已选 ${formatInt(selectedIds.length)} / ${formatInt(total)} 个${noun}`;
}

function renderMultiSelectFilter(config) {
  const noun = config.noun || "项";
  const unit = config.unit || "";
  app.multiSelectSources.set(config.id, { ...config, noun, unit });
  const selected = new Set(config.selectedIds || []);
  return `
    <div class="multi-order-filter" data-multi-select-filter="${escapeHtml(config.id)}">
      <button type="button" class="multi-order-summary" data-multi-select-toggle aria-expanded="false">
        <span class="multi-order-summary-text">${escapeHtml(multiSelectSummary(config.options, config.selectedIds || [], noun, unit))}</span>
        <span class="multi-order-caret" aria-hidden="true">▾</span>
      </button>
      <div class="multi-order-panel" hidden>
        <input type="search" class="multi-order-search" placeholder="输入${escapeHtml(noun)}模糊搜索" aria-label="搜索${escapeHtml(noun)}">
        <label class="multi-order-all"><input type="checkbox" data-multi-select-all> 全选</label>
        <div class="multi-order-list" role="group" aria-label="${escapeHtml(noun)}列表">
          ${config.options.slice(0, MULTI_SELECT_RENDER_CAP).map((opt) => `
            <label class="multi-order-option"><input type="checkbox" data-multi-select-id="${escapeHtml(opt.id)}" ${selected.has(opt.id) ? "checked" : ""}> ${escapeHtml(opt.label)}</label>
          `).join("")}
        </div>
        <div class="multi-order-foot">
          ${config.options.length > MULTI_SELECT_RENDER_CAP ? `<span class="multi-order-more">共 ${formatInt(config.options.length)} 项，输入关键字可搜索全部</span>` : ""}
        </div>
      </div>
    </div>
  `;
}

function mountMultiSelectFilters() {
  document.querySelectorAll(".page.active [data-multi-select-filter]:not([data-multi-select-bound='1'])").forEach((container) => {
    const source = app.multiSelectSources.get(container.dataset.multiSelectFilter);
    if (!source) return;
    container.dataset.multiSelectBound = "1";
    const toggle = container.querySelector("[data-multi-select-toggle]");
    const panel = container.querySelector(".multi-order-panel");
    const search = container.querySelector(".multi-order-search");
    const list = container.querySelector(".multi-order-list");
    const allBox = container.querySelector("[data-multi-select-all]");
    const summaryText = container.querySelector(".multi-order-summary-text");
    if (!toggle || !panel || !list || !allBox) return;

    const working = new Set(source.selectedIds || []);
    const total = source.options.length;

    const syncAllBox = () => {
      // 仅当显式勾选了全部选项时才自动勾上全选框；空集（语义上=全部）不强制勾选，
      // 避免用户取消全选后被 syncAllBox 立即重新勾上。
      const allSelected = total > 0 && working.size >= total;
      if (allBox.checked !== allSelected) allBox.checked = allSelected;
    };
    // 初始：空集 = 全部，全选框默认勾上
    allBox.checked = working.size === 0 || working.size >= total;
    const updateSummary = () => {
      summaryText.textContent = multiSelectSummary(source.options, Array.from(working), source.noun, source.unit);
    };

    const commit = () => {
      // 空集合与全选都表示"全部"，统一提交为空数组，让上游按无筛选处理
      const ids = working.size >= total ? [] : Array.from(working);
      source.onChange(ids);
    };

    const open = () => {
      panel.hidden = false;
      toggle.setAttribute("aria-expanded", "true");
      if (search) search.focus();
    };
    const close = ({ commitChanges = true } = {}) => {
      if (panel.hidden) return;
      panel.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
      if (commitChanges) commit();
    };

    toggle.addEventListener("click", () => {
      if (panel.hidden) open(); else close();
    });

    // 从全量 source.options 重渲染选项（勾选态取自本地 working 集合），供搜索时使用
    const renderOptions = (opts) => {
      list.innerHTML = opts.map((opt) => `<label class="multi-order-option"><input type="checkbox" data-multi-select-id="${escapeHtml(opt.id)}" ${working.has(opt.id) ? "checked" : ""}> ${escapeHtml(opt.label)}</label>`).join("");
    };
    if (search) {
      search.addEventListener("input", () => {
        const q = search.value.trim().toLowerCase();
        // 在全量 source.options 上搜索（而非仅已渲染的前 N 条），大实例下第 N 名之后的项也能搜到并勾选
        const matched = q ? source.options.filter((o) => String(o.label).toLowerCase().includes(q)) : source.options;
        renderOptions(matched.slice(0, MULTI_SELECT_RENDER_CAP));
      });
    }

    allBox.addEventListener("change", () => {
      if (allBox.checked) {
        source.options.forEach((o) => working.add(o.id));
      } else {
        working.clear();
      }
      // 重渲染选项列表以反映最新勾选态（而非仅操作当前已渲染的 DOM）
      const q = search ? search.value.trim().toLowerCase() : "";
      const matched = q ? source.options.filter((o) => String(o.label).toLowerCase().includes(q)) : source.options;
      renderOptions(matched.slice(0, MULTI_SELECT_RENDER_CAP));
      syncAllBox();
      updateSummary();
    });

    list.addEventListener("change", (event) => {
      const box = event.target.closest("[data-multi-select-id]");
      if (!box) return;
      if (box.checked) working.add(box.dataset.multiSelectId); else working.delete(box.dataset.multiSelectId);
      syncAllBox();
      updateSummary();
    });

    // 点击组件外部：关闭并提交
    const onDocClick = (event) => {
      if (!container.isConnected) { document.removeEventListener("click", onDocClick, true); return; }
      if (!container.contains(event.target)) close();
    };
    document.addEventListener("click", onDocClick, true);
  });
}

// —— 通用「可搜索 + 单选」下拉筛选框（客户端）——
// 复用甘特图 multi-order-* 的视觉样式，行为改单选：按钮显示当前选中项 →
// 点击展开面板 → 搜索框模糊过滤 → 点选项即选中并关闭面板。
// config: { id, options:[{id, label}], selectedId:string|null, onChange:(id)=>void, noun }
// 语义：always 有且仅有一个选中项（无选中时按钮显示"全部{noun}"占位）。
const SINGLE_SELECT_RENDER_CAP = 300;

function singleSelectSummary(options, selectedId, noun) {
  if (!selectedId) return `选择${noun}…`;
  const found = options.find((o) => o.id === selectedId);
  return found ? found.label : String(selectedId);
}

function renderSingleSelectFilter(config) {
  const noun = config.noun || "项";
  app.singleSelectSources.set(config.id, { ...config, noun });
  return `
    <div class="multi-order-filter single-select-filter" data-single-select-filter="${escapeHtml(config.id)}">
      <button type="button" class="multi-order-summary" data-single-select-toggle aria-expanded="false">
        <span class="multi-order-summary-text">${escapeHtml(singleSelectSummary(config.options, config.selectedId, noun))}</span>
        <span class="multi-order-caret" aria-hidden="true">▾</span>
      </button>
      <div class="multi-order-panel" hidden>
        <input type="search" class="multi-order-search" placeholder="输入${escapeHtml(noun)}模糊搜索" aria-label="搜索${escapeHtml(noun)}">
        <div class="multi-order-list" role="listbox" aria-label="${escapeHtml(noun)}列表">
          ${config.options.slice(0, SINGLE_SELECT_RENDER_CAP).map((opt) => `
            <label class="multi-order-option${opt.id === config.selectedId ? " is-checked" : ""}" data-single-select-id="${escapeHtml(opt.id)}">${escapeHtml(opt.label)}</label>
          `).join("")}
        </div>
        <div class="multi-order-foot">
          ${config.options.length > SINGLE_SELECT_RENDER_CAP ? `<span class="multi-order-more">共 ${formatInt(config.options.length)} 项，输入关键字可搜索全部</span>` : ""}
        </div>
      </div>
    </div>
  `;
}

function mountSingleSelectFilters() {
  document.querySelectorAll(".page.active [data-single-select-filter]:not([data-single-select-bound='1'])").forEach((container) => {
    const source = app.singleSelectSources.get(container.dataset.singleSelectFilter);
    if (!source) return;
    container.dataset.singleSelectBound = "1";
    const toggle = container.querySelector("[data-single-select-toggle]");
    const panel = container.querySelector(".multi-order-panel");
    const search = container.querySelector(".multi-order-search");
    const list = container.querySelector(".multi-order-list");
    const summaryText = container.querySelector(".multi-order-summary-text");
    if (!toggle || !panel || !list) return;

    const updateSummary = () => {
      summaryText.textContent = singleSelectSummary(source.options, source.selectedId, source.noun);
    };

    const renderOptions = (opts) => {
      list.innerHTML = opts.map((opt) => `
        <label class="multi-order-option${opt.id === source.selectedId ? " is-checked" : ""}" data-single-select-id="${escapeHtml(opt.id)}">${escapeHtml(opt.label)}</label>
      `).join("");
    };

    const open = () => {
      panel.hidden = false;
      toggle.setAttribute("aria-expanded", "true");
      if (search) { search.value = ""; renderOptions(source.options.slice(0, SINGLE_SELECT_RENDER_CAP)); }
      search?.focus();
    };
    const close = () => {
      if (panel.hidden) return;
      panel.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
    };

    toggle.addEventListener("click", () => {
      if (panel.hidden) open(); else close();
    });

    if (search) {
      search.addEventListener("input", () => {
        const q = search.value.trim().toLowerCase();
        const matched = q ? source.options.filter((o) => String(o.label).toLowerCase().includes(q)) : source.options;
        renderOptions(matched.slice(0, SINGLE_SELECT_RENDER_CAP));
      });
    }

    list.addEventListener("click", (event) => {
      const opt = event.target.closest("[data-single-select-id]");
      if (!opt) return;
      const id = opt.dataset.singleSelectId;
      source.selectedId = id;
      updateSummary();
      source.onChange(id);
      close();
    });

    const onDocClick = (event) => {
      if (!container.isConnected) { document.removeEventListener("click", onDocClick, true); return; }
      if (!container.contains(event.target)) close();
    };
    document.addEventListener("click", onDocClick, true);
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

// 优化三阶段映射：coarse→近似广搜；exact_promotion/elite_refine→精确精修；finalize→Pareto 前沿重算
const OPT_PHASE_STAGES = [
  { key: "coarse", label: "近似广搜", phases: ["coarse"] },
  { key: "refine", label: "精确精修", phases: ["exact_promotion", "elite_refine"] },
  { key: "pareto", label: "Pareto 前沿重算", phases: ["finalize"] },
];

function optimizeStageIndex(status) {
  const state = String(status?.status || "").toLowerCase();
  if (state === "done" || state === "completed" || state === "success") return OPT_PHASE_STAGES.length;
  const phase = String(status?.phase || "").toLowerCase();
  const idx = OPT_PHASE_STAGES.findIndex((stage) => stage.phases.includes(phase));
  return idx; // -1 = 尚未进入第一阶段（初始化/图上下文）
}

function renderOptimizeStatus() {
  const status = app.optimizeStatus;
  const result = app.optimizeResult;
  if (!status && !result) return "";
  const state = String(status?.status || (result ? "done" : "running")).toLowerCase();
  const failed = state === "error" || state === "failed";
  const done = state === "done" || state === "completed" || state === "success";
  const activity = window.OptimizeProgress.optimizeActivity(status || {});
  const tone = failed ? "danger" : done ? "success" : activity.stalled ? "warning" : "info";
  const label = failed ? "优化失败" : done ? "优化完成" : state === "submitting" ? "正在提交优化任务" : "优化运行中";
  const progress = done ? 100 : optimizeProgress(status || {});
  const CIRC = 276.5;
  const stageIdx = optimizeStageIndex(status);
  const phaseChip = failed
    ? `<span class="chip err">执行失败</span>`
    : done
      ? `<span class="chip ok">全部阶段完成</span>`
      : stageIdx >= 0
        ? `<span class="chip blue">阶段 ${stageIdx + 1}/3 · ${OPT_PHASE_STAGES[stageIdx].label}</span>`
        : `<span class="chip blue">初始化 · ${escapeHtml(optimizePhaseLabel(status?.phase || state))}</span>`;
  const stageStat = (stage, idx) => {
    if (idx === 0) return `${formatInt(status?.approximate_evaluations || result?.approximate_evaluations || 0)} 评估`;
    if (idx === 1) return `${formatInt(status?.exact_evaluations || result?.exact_evaluations || 0)} 评估`;
    return done ? "已完成" : "待执行";
  };
  const phaseLine = OPT_PHASE_STAGES.map((stage, idx) => {
    const cls = done || idx < stageIdx ? "done" : idx === stageIdx ? "on" : "";
    const mark = done || idx < stageIdx ? "✓" : idx === stageIdx ? "●" : "";
    return `<div class="phase ${cls}">${mark ? `${mark} ` : ""}${stage.label} · ${stageStat(stage, idx)}</div>`;
  }).join("");
  const message = failed
    ? (status?.error || status?.message || "未收到具体错误说明")
    : done
      ? (status?.message || "优化完成，方案已可用于评审")
      : (activity.message || status?.message || "任务已提交，正在等待优化器返回进度");
  const bestObjective = (() => {
    if (!done || !result) return null;
    const key = asArray(result.objective_keys)[0];
    const best = bestMetricValue(asArray(result.solutions), key);
    if (!key || best === null || best === undefined) return null;
    return { key, value: metricDisplay({ objectives: { [key]: best } }, key) };
  })();
  return `
    <div class="card card-pad state-card ${done ? "success" : failed ? "danger" : ""}" id="optimize-run-status" role="status" aria-live="polite">
      <div class="card-head" style="border:0;margin:0;padding:0">
        <div><h3>${escapeHtml(label)}</h3><p>任务 ${escapeHtml(app.optimizeTaskId || "待分配")} · 已运行 ${formatDurationSeconds(status?.elapsed_s || result?.elapsed_s || 0)}${activity.stalled && !failed && !done ? ` · 真实进度静止 ${formatDurationSeconds(activity.secondsSinceRealProgress)}` : ""}</p></div>
        ${phaseChip}
      </div>
      <div class="progress-ring-wrap mt-16">
        <div class="ring">
          <svg width="104" height="104">
            <circle cx="52" cy="52" r="44" fill="none" stroke="var(--inset)" stroke-width="9"/>
            <circle cx="52" cy="52" r="44" fill="none" stroke="${failed ? "var(--danger)" : "var(--primary)"}" stroke-width="9" stroke-linecap="round" stroke-dasharray="${CIRC}" stroke-dashoffset="${(CIRC * (1 - progress / 100)).toFixed(1)}"/>
          </svg>
          <span class="pct">${progress}%</span>
        </div>
        <div style="flex:1">
          <div class="phase-line">${phaseLine}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${progress}%"></div></div>
          <p class="subtle mt-16" style="margin-top:10px">${escapeHtml(message)}</p>
        </div>
      </div>
      <div class="grid-4 mt-16">
        <div class="kpi-card"><span>近似评估</span><strong>${formatInt(status?.approximate_evaluations || result?.approximate_evaluations || 0)}</strong><small>前期广搜吞吐</small></div>
        <div class="kpi-card"><span>精确评估</span><strong>${formatInt(status?.exact_evaluations || result?.exact_evaluations || 0)}</strong><small>后期高质量验证</small></div>
        <div class="kpi-card"><span>候选方案</span><strong>${formatInt(result?.found_solution_count || result?.solutions?.length || status?.archive_size || status?.coarse_pool_size || 0)}</strong><small>目标 ${formatInt(app.optimizeForm.targetSolutionCount)} 个</small></div>
        <div class="kpi-card"><span>${bestObjective ? `当前最优${escapeHtml(getObjectiveLabel(bestObjective.key))}` : "可行率"}</span><strong>${bestObjective ? bestObjective.value : formatPercent(status?.feasible_ratio || 0)}</strong><small>${bestObjective ? "Pareto 前沿判优" : "候选池可行占比"}</small></div>
      </div>
      ${failed ? `
        <div class="optimize-error-detail mt-16">
          <strong>${escapeHtml(status?.error_type ? `错误类型：${status.error_type}` : "失败原因")}</strong>
          <span>请根据上方原因检查实例数据与优化参数后重试；若仍失败，可将下方技术详情提供给开发人员。</span>
          ${status?.technical_detail ? `<details><summary>查看技术详情</summary><pre>${escapeHtml(status.technical_detail)}</pre></details>` : ""}
        </div>
      ` : ""}
    </div>
  `;
}

// 运行日志流：后端 hybrid status 的 events 字段（无该字段的旧后端回退到当前活动文案）
function renderOptimizeLogCard() {
  const status = app.optimizeStatus;
  if (!status && !app.optimizeResult) return "";
  const events = asArray(status?.events);
  let lines;
  if (events.length) {
    lines = events.map((event) => {
      const ts = event.ts ? new Date(Number(event.ts) * 1000) : null;
      const stamp = ts && !Number.isNaN(ts.getTime())
        ? `${String(ts.getHours()).padStart(2, "0")}:${String(ts.getMinutes()).padStart(2, "0")}:${String(ts.getSeconds()).padStart(2, "0")}`
        : "--:--:--";
      return `[${stamp}] ${event.text || ""}`;
    }).join("\n");
  } else {
    const fallback = status?.message || window.OptimizeProgress.optimizeActivity(status || {}).message || "等待求解器事件…";
    lines = `[--] ${fallback}`;
  }
  return `
    <div class="card card-pad">
      <div class="card-head"><div><h3>运行日志</h3><p>求解器关键事件流。</p></div></div>
      <div class="log-box" id="optimize-log-box">${escapeHtml(lines)}</div>
    </div>
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

// 方案显示命名：一律「方案一 / 方案二 / 方案三…」（中文序号，按候选顺序分配），
// 界面任何位置不得出现后端 S-xxxxxxxx 式 solution_id；id/solutionId 字段本身不动。
const CHINESE_NUMERALS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
function schemeDisplayName(index) {
  const n = index + 1;
  let num;
  if (n <= 10) num = CHINESE_NUMERALS[n];
  else if (n < 20) num = `十${CHINESE_NUMERALS[n - 10]}`;
  else num = `${CHINESE_NUMERALS[Math.floor(n / 10)]}十${n % 10 ? CHINESE_NUMERALS[n % 10] : ""}`;
  return `方案${num}`;
}

// 来源徽标（对比表方案名旁的小 chip）：基线 ATC / Pareto / 启发式 EDD / 精确冠军
function candidateSourceTag(item) {
  const source = String(item?.source || "");
  if (source === "baseline") return { label: `基线 ${item.raw?.rule_name || item.heuristicRuleName || "ATC"}`, cls: "info" };
  if (source === "exact_reference") return { label: "精确冠军", cls: "ok" };
  if (source === "reference" || source === "heuristic") return { label: `启发式 ${item.heuristicRuleName || ""}`.trim(), cls: "neutral" };
  return { label: "Pareto", cls: "blue" };
}

function getReviewCandidates() {
  const items = [];
  if (app.optimizeResult?.baseline) {
    items.push(normalizeCandidate(app.optimizeResult.baseline, { source: "baseline" }));
  }
  asArray(app.optimizeResult?.solutions).forEach((item) => {
    items.push(normalizeCandidate(item, { source: item.source || "pareto" }));
  });
  asArray(app.optimizeResult?.reference_solutions).forEach((item) => {
    items.push(normalizeCandidate(item, { source: item.source || "reference" }));
  });
  asArray(app.referenceSolutions).forEach((item) => {
    items.push(normalizeCandidate(item, { source: item.source || "heuristic" }));
  });
  if (app.exactReference) {
    items.push(normalizeCandidate(app.exactReference, { source: "exact_reference" }));
  }
  const uniq = new Map();
  items.filter(Boolean).forEach((item) => uniq.set(item.id, item));
  // 去重后按顺序统一分配中文序号显示名
  return Array.from(uniq.values()).map((item, index) => ({ ...item, name: schemeDisplayName(index) }));
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
          escapeHtml(candidateSourceTag(item).label),
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
  el("topbar-scene-name").textContent = hasScene ? app.currentScene.name : "未加载实例";
  const sceneMeta = el("topbar-scene-meta");
  if (sceneMeta) sceneMeta.textContent = hasScene ? `${formatInt(summary.orders)}订单 · ${formatInt(summary.operations)}工序` : "";
  const sceneDot = document.querySelector(".scene-pill .dot");
  if (sceneDot) sceneDot.classList.toggle("missing", !hasScene);
  el("topbar-orders").textContent = hasScene ? formatInt(summary.orders) : "-";
  el("topbar-tasks").textContent = hasScene ? formatInt(summary.tasks) : "-";
  el("topbar-operations").textContent = hasScene ? formatInt(summary.operations) : "-";
  el("topbar-machines").textContent = hasScene ? formatInt(summary.machines) : "-";
  el("topbar-toolings").textContent = hasScene ? formatInt(summary.toolings) : "-";
  el("topbar-personnel").textContent = hasScene ? formatInt(summary.personnel) : "-";

  // LLM 状态 pill：已配置 Key → 已连接（测试成功后带延迟信息），未配置 → 警告态
  const llmPill = el("llm-status-pill");
  const llmText = el("llm-status-text");
  if (llmPill && llmText) {
    const configured = !!app.llmConfig?.has_key;
    const tested = app.llmTestResult?.status === "ok";
    llmText.textContent = tested ? "LLM 已连接" : configured ? "LLM 已配置" : "LLM 未配置";
    llmPill.classList.toggle("warn", !configured);
  }

  document.querySelectorAll(".requires-scene").forEach((node) => {
    node.classList.toggle("is-disabled", !hasScene);
  });

  // 流程进度条 + 侧栏状态灯：优先用后端聚合接口，未返回前用前端状态推导兜底
  renderFlowbar();
  refreshWorkflowOverview();
}

// 前端兜底推导 5 步状态（与后端 /api/workflow/overview 同构；接口可用后以服务端为准）
function deriveWorkflowSteps() {
  const hasScene = !!app.currentScene;
  const summary = getSceneSummary();
  const validation = app.validation;
  const graphReady = !!app.graphMeta && app.graphMeta.cache_ready !== false;
  const sim = app.simResult;
  const simFeasible = sim?.metrics?.feasible !== false;
  const optStatus = String(app.optimizeStatus?.status || "").toLowerCase();
  const optRunning = optimizeIsRunning();
  const optDone = !!app.optimizeResult || optStatus === "done";
  const optError = ["error", "failed"].includes(optStatus);
  const selectedCount = app.reviewSelection.length;
  const candidateCount = getReviewCandidates().length;

  const steps = [];
  if (!hasScene) {
    steps.push({ key: "import", state: "current", tone: "none", label: "数据导入", detail: "未导入实例" });
  } else if (validation?.status === "failed") {
    steps.push({ key: "import", state: "done", tone: "err", label: "数据导入", detail: `校验失败 · ${formatInt(validation.error_count || 0)} 错误` });
  } else if (validation?.status === "warning") {
    steps.push({ key: "import", state: "done", tone: "warn", label: "数据导入", detail: `校验通过 · ${formatInt(validation.warning_count || 0)} 警告` });
  } else if (validation) {
    steps.push({ key: "import", state: "done", tone: "ok", label: "数据导入", detail: "校验通过" });
  } else {
    steps.push({ key: "import", state: "done", tone: "warn", label: "数据导入", detail: `${formatInt(summary.orders)} 订单 · 未校验` });
  }
  steps.push(!hasScene
    ? { key: "graph", state: "todo", tone: "none", label: "图谱构建", detail: "待导入" }
    : graphReady
      ? { key: "graph", state: "done", tone: "ok", label: "图谱构建", detail: `${formatInt(app.graphMeta.total_nodes)} 节点 · ${formatInt(app.graphMeta.total_edges)} 边` }
      : { key: "graph", state: "todo", tone: "none", label: "图谱构建", detail: "未构建" });
  steps.push(!hasScene
    ? { key: "simulate", state: "todo", tone: "none", label: "规则仿真", detail: "待导入" }
    : sim
      ? { key: "simulate", state: "done", tone: simFeasible ? "ok" : "warn", label: "规则仿真", detail: `${sim.rule || app.simRule} · ${simFeasible ? "可行" : "不完整"}` }
      : { key: "simulate", state: "todo", tone: "none", label: "规则仿真", detail: "未运行" });
  steps.push(optRunning
    ? { key: "optimize", state: "current", tone: "run", label: "优化求解", detail: `运行中 · ${optimizeProgress(app.optimizeStatus)}%` }
    : optError
      ? { key: "optimize", state: "blocked", tone: "err", label: "优化求解", detail: "优化失败" }
      : optDone
        ? { key: "optimize", state: "done", tone: "ok", label: "优化求解", detail: `${formatInt(app.optimizeResult?.found_solution_count || app.optimizeResult?.solutions?.length || 0)} 候选方案` }
        : { key: "optimize", state: "todo", tone: "none", label: "优化求解", detail: "未运行" });
  steps.push(selectedCount
    ? { key: "review", state: "done", tone: "ok", label: "方案评审", detail: `已选 ${selectedCount} 个方案`, badge: `${selectedCount}/4` }
    : candidateCount
      ? { key: "review", state: optDone ? "current" : "todo", tone: "none", label: "方案评审", detail: `已有 ${formatInt(candidateCount)} 候选`, badge: "0/4" }
      : { key: "review", state: "todo", tone: "none", label: "方案评审", detail: "暂无候选", badge: "0/4" });
  // current 唯一：第一个非 done 非 blocked 的步骤
  if (!steps.some((step) => step.state === "current")) {
    const next = steps.find((step) => step.state === "todo");
    if (next) next.state = "current";
  }
  return steps;
}

function workflowSteps() {
  const steps = app.workflowOverview?.steps;
  return Array.isArray(steps) && steps.length ? steps : deriveWorkflowSteps();
}

function renderFlowbar() {
  const steps = workflowSteps();
  // 侧栏状态灯 / 徽标与进度条同源
  const toneMap = { import: "lamp-import", graph: "lamp-graph", simulate: "lamp-simulate", optimize: "lamp-optimize" };
  steps.forEach((step) => {
    const lampId = toneMap[step.key];
    if (lampId) {
      const lamp = el(lampId);
      if (lamp) lamp.className = `lamp ${step.tone && step.tone !== "none" ? step.tone : ""}`;
    }
    if (step.key === "review") {
      const badge = el("badge-review");
      if (badge) {
        badge.hidden = !step.badge;
        badge.textContent = step.badge || "";
      }
    }
  });
  const llmLamp = el("lamp-llm");
  if (llmLamp) llmLamp.className = `lamp ${app.llmConfig?.has_key ? "ok" : "warn"}`;
  // 侧栏优化运行提示
  const note = el("nav-note");
  if (note) {
    if (optimizeIsRunning()) {
      note.hidden = false;
      note.innerHTML = `<strong>优化运行中</strong>${escapeHtml(optimizePhaseLabel(app.optimizeStatus?.phase || ""))} · 已运行 ${formatDurationSeconds(app.optimizeStatus?.elapsed_s || 0)}，完成后可进入方案评审。`;
    } else {
      note.hidden = true;
      note.innerHTML = "";
    }
  }
}

// 聚合状态节流刷新：默认 5s 一次；关键动作完成后可 force
async function refreshWorkflowOverview({ force = false } = {}) {
  const now = Date.now();
  if (!force && now - app.workflowOverviewAt < 5000) return;
  app.workflowOverviewAt = now;
  try {
    const payload = await api.getWorkflowOverview();
    app.workflowOverview = payload;
    renderFlowbar();
  } catch {
    // 接口不可用（旧后端）时保持前端推导，不打扰用户
  }
}

async function navigate(navKey, pushHash = true) {
  const resolved = NAV_MAP[navKey] || { page: navKey };
  if (resolved.requiresScene) {
    const ready = await ensureSceneLoaded();
    if (!ready) {
      toast("请先在“数据导入”页生成或导入实例。", "warning");
      app.currentNav = "import";
      app.currentPage = "import";
      setActiveNav("import");
      showPage("import");
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
  // 流程进度由顶部全局进度条表达，本页只保留 Hero + KPI + 三列详情
  const flowSteps = [
    { label: "数据导入", done: !!app.currentScene, nav: "import" },
    { label: "图谱构建", done: !!app.graphMeta, nav: "graph" },
    { label: "规则仿真", done: !!app.simResult, nav: "simulate" },
    { label: "优化求解", done: !!app.optimizeResult, nav: "optimize" },
    { label: "方案评审", done: !!selected, nav: "review" },
  ];
  const activeStep = flowSteps.findIndex((item) => !item.done);
  const nextStep = flowSteps[activeStep < 0 ? flowSteps.length - 1 : activeStep];
  const optState = app.optimizeStatus?.status || (app.optimizeResult ? "done" : "not-started");
  const heroTitle = optimizeIsRunning()
    ? "下一步：等待优化求解完成"
    : activeStep < 0 ? "当前调度流程已完成" : `下一步：${escapeHtml(nextStep.label)}`;
  const heroDesc = optimizeIsRunning()
    ? `优化任务正在${escapeHtml(optimizePhaseLabel(app.optimizeStatus?.phase || ""))}（${optimizeProgress(app.optimizeStatus)}%），完成后建议复核 Pareto 前沿并进入方案评审导出最终方案。`
    : activeStep < 0 ? "实例、仿真和候选方案均已就绪，可继续复核并导出最终方案。" : "按业务顺序推进，每一步的结果会自动成为下一步的输入。";
  container.innerHTML = `
    <div class="stack">
      <div class="hero">
        <div>
          <span style="font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:rgba(255,255,255,0.6)">Current workflow</span>
          <h3>${heroTitle}</h3>
          <p>${heroDesc}</p>
          <button class="btn btn-white" type="button" data-nav-jump="${escapeHtml(nextStep.nav)}">${activeStep < 0 ? "返回方案评审" : `继续${escapeHtml(nextStep.label)}`}</button>
        </div>
        <div class="cells">
          <div><span>当前主目标</span><strong>${escapeHtml(objectiveShortList(objectiveKeys))}</strong></div>
          <div><span>已得方案数</span><strong>${formatInt(app.optimizeResult?.found_solution_count || app.optimizeResult?.solutions?.length || 0)}</strong></div>
          <div><span>当前关注方案</span><strong>${escapeHtml(selected?.name || "未指定")}</strong></div>
        </div>
      </div>
      <div class="grid-4">
        <div class="kpi-card"><span>订单</span><strong>${formatInt(summary.orders)}</strong><small>当前实例订单规模</small></div>
        <div class="kpi-card"><span>工序</span><strong>${formatInt(summary.operations)}</strong><small>当前实例工序总量</small></div>
        <div class="kpi-card"><span>资源</span><strong>${formatInt(summary.machines)} / ${formatInt(summary.toolings)} / ${formatInt(summary.personnel)}</strong><small>机器 / 工装 / 人员</small></div>
        <div class="kpi-card"><span>优化状态</span><strong style="color:var(--primary)">${escapeHtml(optState)}</strong><small>混合优化任务当前状态</small></div>
      </div>
      <div class="grid-3">
        <article class="surface-card">
          <div class="card-head"><div><h3>问题规模</h3></div></div>
          <div class="kv-grid" style="grid-template-columns:1fr 1fr">
            <div><span>任务数</span><strong>${formatInt(summary.tasks)}</strong></div>
            <div><span>停机记录</span><strong>${formatInt(app.downtimes.length)}</strong></div>
            <div><span>在制工序</span><strong>${formatInt(summary.ops_in_progress || 0)}</strong></div>
            <div><span>计划起点</span><strong>${escapeHtml(formatDateTime(app.instanceDetails?.plan_start_at))}</strong></div>
          </div>
        </article>
        <article class="surface-card">
          <div class="card-head"><div><h3>优化进展</h3></div></div>
          <div class="kv-grid" style="grid-template-columns:1fr 1fr">
            <div><span>近似评估</span><strong>${formatInt(app.optimizeResult?.approximate_evaluations || app.optimizeStatus?.approximate_evaluations || 0)}</strong></div>
            <div><span>精确评估</span><strong>${formatInt(app.optimizeResult?.exact_evaluations || app.optimizeStatus?.exact_evaluations || 0)}</strong></div>
            <div><span>总评估次数</span><strong>${formatInt(app.optimizeResult?.total_evaluations || app.optimizeStatus?.total_evaluations || 0)}</strong></div>
            <div><span>耗时</span><strong>${formatDurationSeconds(app.optimizeResult?.elapsed_s || app.optimizeStatus?.elapsed_s || 0)}</strong></div>
          </div>
        </article>
        <article class="surface-card">
          <div class="card-head"><div><h3>当前推荐关注</h3></div></div>
          ${selected ? `
          <div class="kv-grid" style="grid-template-columns:1fr 1fr">
            <div><span>方案名称</span><strong>${escapeHtml(selected.name || "-")}</strong></div>
            <div><span>总延误</span><strong>${metricDisplay(selected, "total_tardiness")}</strong></div>
            <div><span>总周期</span><strong>${metricDisplay(selected, "makespan")}</strong></div>
            <div><span>净可用利用率</span><strong>${metricDisplay(selected, "avg_net_available_utilization")}</strong></div>
          </div>
          ` : renderEmptyState("还没有候选方案", "先运行混合优化，或生成启发式 / 精确参考方案后再来这里汇总查看。")}
        </article>
      </div>
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

const GANTT_MACHINE_FILTER_DEFAULT = Object.freeze({ types: [], downtimeOnly: false, query: "", page: 1 });
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
  const types = asArray(filter.types);
  return rows.filter((row) => {
    if (types.length && !types.includes(row.typeName)) return false;
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
  // 本地模式：订单多选集合（空数组 = 全部订单）。大实例默认聚焦首个订单，避免一次性全量渲染卡死。
  let localSelectedIds = [];
  if (!serverMode) {
    localSelectedIds = asArray(app.ganttOrderMulti[id]).filter((oid) => orderOptions.includes(oid));
    if (!localSelectedIds.length && !allowAll && orderOptions.length) localSelectedIds = [orderOptions[0]];
  }
  // selectedOrder 兼容下游（分组/配色/构建）：单选=该订单；多选或全部=特殊 "__all__"（走机器分组）
  const selectedOrder = serverMode
    ? (serverOrders.selectedOrder || orderOptions[0])
    : (localSelectedIds.length === 1 ? localSelectedIds[0] : "__all__");
  const orderEntries = serverMode
    ? allEntries
    : (localSelectedIds.length
      ? allEntries.filter((item) => localSelectedIds.includes(item.order_id || "-"))
      : allEntries);

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
  // 服务端取数模式（方案详情/评审对比）沿用异步单选；本地模式（规则仿真等）用 Excel 风格订单多选。
  const orderControl = serverMode
    ? `
      ${renderOrderCombobox({
        id: orderComboboxId,
        selected: selectedOrderItem,
        contextKey: `plan::${taskId}::${solutionId}`,
        search: async (query, signal) => {
          const payload = await api.searchReviewOrders(taskId, [solutionId], query, signal);
          return asArray(payload?.orders);
        },
        select: (order) => loadPlanGantt(taskId, solutionId, order.order_id),
      })}`
    : renderMultiSelectFilter({
      id: orderComboboxId,
      noun: "订单",
      options: orderOptions.map((orderId) => ({ id: orderId, label: orderId })),
      selectedIds: localSelectedIds,
      onChange: (ids) => {
        app.ganttOrderMulti[id] = ids;
        return renderCurrentPage();
      },
    });
  // 设计稿 v3：甘特筛选只保留 4 个并排成一排——订单（多选）、机器类型（多选）、工序状态、分组；
  // 已按反馈移除：仅含停机、机器号搜索、工序搜索、时间窗输入（底层筛选状态保持默认值，不再渲染控件）。
  const showOrderControl = orderOptions.length > 1 || !allowAll;
  const facet = data.machineFacet;
  const machineTypeControl = facet ? renderMultiSelectFilter({
    id: `${id}-mtype`,
    noun: "机器类型",
    options: facet.typeOptions.map((t) => ({ id: t, label: t })),
    selectedIds: asArray(facet.filter.types),
    onChange: (types) => {
      const f = getGanttMachineFilter(id);
      f.types = types;
      f.page = 1;
      app.ganttMachineFilter[id] = f;
      return renderCurrentPage();
    },
  }) : "";
  const hitText = facet
    ? `命中 ${formatInt(facet.filteredGroups)} / ${formatInt(facet.totalGroups)} ${groupMode === "order" ? "道工序行" : "台机器"}`
    : `共 ${formatInt(data.groups.length)} 行`;
  const filterRow = `
    <div class="gantt-filter-row nowrap">
      ${showOrderControl ? `<span class="flabel">订单</span>${orderControl}<span class="sep3"></span>` : ""}
      ${facet ? `<span class="flabel">机器类型</span>${machineTypeControl}<span class="sep3"></span>` : ""}
      <span class="flabel">工序状态</span>
      <select data-gantt-status-filter data-canvas="${escapeHtml(id)}">
        <option value="__all__" ${entryFilter.status === "__all__" ? "selected" : ""}>全部状态</option>
        <option value="completed" ${entryFilter.status === "completed" ? "selected" : ""}>已完成</option>
        <option value="processing" ${entryFilter.status === "processing" ? "selected" : ""}>进行中</option>
        <option value="future" ${entryFilter.status === "future" ? "selected" : ""}>未来排产</option>
      </select>
      <span class="sep3"></span>
      <span class="flabel">分组</span>
      <select data-gantt-group-mode data-canvas="${escapeHtml(id)}">
        <option value="order" ${groupMode === "order" ? "selected" : ""} ${allowOrderMode ? "" : "disabled"}>按订单层级</option>
        <option value="machine" ${groupMode === "machine" ? "selected" : ""}>按机器资源</option>
      </select>
      <span class="grow"></span>
      <span class="flabel">${escapeHtml(hitText)}</span>
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
        <div><h3>${escapeHtml(options.title || "资源甘特图")}</h3>
        <p>条块颜色=订单标识色，质感区分已完成 / 进行中 / 未来排产；红色竖线为"现在"进度分界；斜纹遮罩显示班次外与停机占用。</p></div>
        ${selectedOrder === "__all__" ? `<span class="chip info">${escapeHtml(hitText)}</span>` : `<span class="chip info">${escapeHtml(selectedOrder)}</span>`}
      </div>
      ${filterRow}
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

// 扁平网格布局（7月15号版风格）：不分泳道、不加家族带，节点按类型聚簇后从上到下网格平铺，
// 保留类型配色/形状/尺寸编码与自由拖动。lanes / families 返回空数组，渲染层不再绘制分隔带。
const GRAPH_GRID_PAD = 60;
const GRAPH_GRID_CELL = 132;
const GRAPH_GRID_ROW = 112;

function layoutGraph(nodes, edges, selectedId) {
  const degree = new Map(nodes.map((node) => [node.id, 0]));
  edges.forEach((edge) => {
    degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
  });

  // 按类型聚簇（沿用类型次序），类型内按度数排序，使强连接节点靠前；整体作为一条连续序列平铺。
  const typeRank = new Map(GRAPH_NODE_ORDER.map((type, index) => [type, index]));
  const ordered = nodes.slice().sort((a, b) => {
    const ta = typeRank.has(a.type) ? typeRank.get(a.type) : GRAPH_NODE_ORDER.length;
    const tb = typeRank.has(b.type) ? typeRank.get(b.type) : GRAPH_NODE_ORDER.length;
    if (ta !== tb) return ta - tb;
    const degreeGap = (degree.get(b.id) || 0) - (degree.get(a.id) || 0);
    if (degreeGap !== 0) return degreeGap;
    return String(a.label).localeCompare(String(b.label), "zh-CN");
  });

  // 网格列数：偏向纵向（从上到下）阅读，取略小于正方形的列数。
  const count = ordered.length;
  const cols = Math.max(1, Math.round(Math.sqrt(Math.max(1, count)) * 0.9));
  const rows = Math.max(1, Math.ceil(count / cols));

  const placed = new Map();
  ordered.forEach((node, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    placed.set(node.id, {
      x: GRAPH_GRID_PAD + col * GRAPH_GRID_CELL,
      y: GRAPH_GRID_PAD + row * GRAPH_GRID_ROW,
      r: GRAPH_NODE_SIZES[node.type] || GRAPH_NODE_SIZES.other,
      node,
      type: node.type,
    });
  });

  const width = Math.max(960, GRAPH_GRID_PAD * 2 + Math.max(0, cols - 1) * GRAPH_GRID_CELL + 80);
  const height = Math.max(460, GRAPH_GRID_PAD * 2 + Math.max(0, rows - 1) * GRAPH_GRID_ROW + 80);

  return { width, height, placed, lanes: [], families: [] };
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

function renderSimulatePage() {
  const container = el("simulate-content");
  if (!container) return;
  const simMetrics = app.simResult?.metrics || {};
  const completed = formatInt(simMetrics.completed_operations || 0);
  const total = formatInt(simMetrics.total_operations || 0);
  const infeasible = !!app.simResult && simMetrics.feasible === false;
  // 前置检查条
  if (infeasible) {
    renderPrecheck("simulate-precheck", "err", `⚠ 仿真不完整：仅完成 ${completed} / ${total} 道工序，下方指标不可用于决策。<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="import">去查看校验结果 →</button>`);
  } else if (app.simResult) {
    renderPrecheck("simulate-precheck", "ok", `✓ 基线仿真（${escapeHtml(app.simResult.rule || app.simRule)}）可行，${completed} / ${total} 工序已排产<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="optimize">下一步：优化求解 →</button>`);
  } else {
    renderPrecheck("simulate-precheck", "info", `选择派工规则后点击右侧「运行仿真」按钮，结果将作为优化求解的基线参照。`);
  }
  // 规则卡片选择（ATC/EDD/SPT/CR/FIFO/LPT + 一句话业务解释）
  const ruleCards = `
    <article class="surface-card">
      <div class="card-head">
        <div><h3>选择派工规则</h3><p>当前 ${escapeHtml(app.simRule)}。仿真结果作为优化求解的基线参照。</p></div>
        <button class="btn btn-primary" type="button" data-action="run-simulate" ${app.simBusy ? "disabled" : ""}>${app.simBusy ? "仿真运行中…" : "运行仿真"}</button>
      </div>
      <div class="rule-grid">
        ${CONFIG.HEURISTIC_RULES.map((rule) => `
          <button class="rule-card ${rule === app.simRule ? "selected" : ""}" type="button" data-action="select-sim-rule" data-rule="${escapeHtml(rule)}">
            <strong>${escapeHtml(rule)}</strong><p>${escapeHtml(HEURISTIC_RULE_BLURB[rule] || "经典派工规则")}</p>
          </button>
        `).join("")}
      </div>
      <div id="sim-status" class="mt-16">${renderSimStatusInner(app.simStatus)}</div>
    </article>
  `;
  const kpiCards = app.simResult ? `
    <div class="grid-4">
      <div class="kpi-card"><span>总延误</span><strong>${formatDurationHours(simMetrics.total_tardiness)}</strong><small>相对基线规则的参考值</small></div>
      <div class="kpi-card"><span>总周期 (Makespan)</span><strong>${formatDurationHours(simMetrics.makespan)}</strong><small>完整排产周期</small></div>
      <div class="kpi-card"><span>净可用利用率</span><strong>${formatPercent(simMetrics.avg_net_available_utilization)}</strong><small>机器加权平均</small></div>
      <div class="kpi-card"><span>完成工序</span><strong>${completed} / ${total}</strong><small style="color:${infeasible ? "var(--danger)" : "var(--success)"}">${infeasible ? "存在未排产工序" : "全部排产成功"}</small></div>
    </div>
  ` : "";
  const infeasibleBanner = infeasible ? `
    <div class="sim-infeasible-banner" role="alert">
      <strong>仿真结果不完整，指标不可用于决策</strong>
      <span>${escapeHtml(app.simResult.diagnosis || `仅完成 ${completed} / ${total} 道工序，请到“数据导入”页运行数据校验。`)}</span>
      ${renderInfeasibleDetail(app.simResult.diagnosis_detail)}
    </div>
  ` : "";
  const detail = app.simResult ? `
    ${renderTimeline(app.simResult.gantt, {
      title: `规则仿真甘特图 · ${app.simRule}`,
      canvasId: "gantt-sim",
      solutionIds: [app.simResult.solution_id || `RULE:${app.simRule}`],
    })}
    <article class="surface-card">
      <div class="card-head"><div><h3>仿真明细预览</h3><p>核查开始/结束时间、状态、订单和资源分配是否符合业务直觉。</p></div></div>
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
  ` : app.simBusy
    ? `<article class="surface-card">${renderEmptyState("仿真计算中…", `正在运行规则仿真（${escapeHtml(app.simRule)}），完成后将自动展示甘特图与指标。`)}</article>`
    : `<article class="surface-card">${renderEmptyState("尚未运行仿真", "选择派工规则后点击「运行仿真」按钮，这里会展示完整的时间轴、状态分布和停机遮罩。")}</article>`;
  container.innerHTML = `<div class="stack">${ruleCards}${infeasibleBanner}${kpiCards}${detail}</div>`;
  requestAnimationFrame(() => mountGantts());
}

// 多目标选择：紧凑 chip 流式网格（目标很多时自然换行），方向 min/max 小字标注
function renderObjectiveSelectors() {
  const catalog = asArray(app.optimizeObjectiveCatalog);
  return `
    <div class="obj-chip-grid">
      ${catalog.map((item) => {
        const on = app.optimizeForm.objectiveKeys.includes(item.key);
        return `
        <label class="obj-chip ${on ? "on" : ""}" title="${escapeHtml(item.description || "")}">
          <input type="checkbox" data-objective-key="${escapeHtml(item.key)}" ${on ? "checked" : ""} hidden>
          <span class="obox">✓</span>${escapeHtml(item.label || getObjectiveLabel(item.key))}<small>${escapeHtml(item.direction || "")}</small>
        </label>`;
      }).join("")}
    </div>
  `;
}

function renderOptimizePage() {
  const container = el("optimize-content");
  if (!container) return;
  const status = app.optimizeStatus;
  const result = app.optimizeResult;
  const recommendedBudget = refreshOptimizeBudgetRecommendation({ preserveManual: true });
  // 前置检查条
  const simFeasible = app.simResult ? app.simResult.metrics?.feasible !== false : null;
  if (optimizeIsRunning()) {
    renderPrecheck("optimize-precheck", "info", `● 优化正在运行（${optimizeProgress(status)}%），进度与日志在右侧实时更新。<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="review">查看已有候选方案 →</button>`);
  } else if (result) {
    renderPrecheck("optimize-precheck", "ok", `✓ 优化已完成 · ${formatInt(result.found_solution_count || result.solutions?.length || 0)} 个候选方案可评审<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="review">下一步：方案评审 →</button>`);
  } else if (simFeasible) {
    renderPrecheck("optimize-precheck", "ok", `✓ 前置就绪：基线仿真（${escapeHtml(app.simResult?.rule || app.simRule)}）可行，可启动优化求解`);
  } else if (app.simResult) {
    renderPrecheck("optimize-precheck", "warn", `⚠ 基线仿真不完整，建议先回「规则仿真」排查，再启动优化。<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="simulate">返回规则仿真 →</button>`);
  } else {
    renderPrecheck("optimize-precheck", "warn", `⚠ 尚未运行基线仿真，建议先完成「规则仿真」作为对照基线。<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="simulate">先去规则仿真 →</button>`);
  }
  const configCard = `
    <div class="sticky-col stack">
      <div class="card card-pad">
        <div class="card-head"><div><h3>多目标优化配置</h3><p id="opt-obj-count">已选 ${formatInt(app.optimizeForm.objectiveKeys.length)} / ${formatInt(app.optimizeObjectiveCatalog.length)} 个目标，参与 Pareto 前沿求解。</p></div></div>
        ${renderObjectiveSelectors()}
        <div class="form-grid mt-16">
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
        <div class="budget-hint" id="opt-budget-hint">${app.optimizeForm.timeLimitTouched
          ? `已按当前规模与参数自动推荐 <strong>${recommendedBudget} 秒</strong>。当前保留手动值 ${app.optimizeForm.timeLimitS} 秒，可随时恢复建议值。`
          : `已按当前规模与参数自动推荐 <strong>${recommendedBudget} 秒</strong>，可继续手动修改。`}</div>
        <div class="mt-16" style="display:flex;gap:9px">
          <button class="btn btn-primary" type="button" data-action="start-hybrid-optimize" ${optimizeIsRunning() ? "disabled aria-busy=\"true\"" : ""}>${optimizeIsRunning() ? "优化运行中…" : "启动优化"}</button>
          <button class="btn btn-ghost" type="button" data-action="apply-budget-recommendation">采用建议预算</button>
        </div>
      </div>
    </div>
  `;
  const statusStack = status || result
    ? `${renderOptimizeStatus()}${renderOptimizeLogCard()}`
    : `<div class="card card-pad">${renderEmptyState("尚未启动优化", "配置左侧目标与参数后点击「启动优化」，这里会实时显示进度环、阶段指示与运行日志。")}</div>`;
  container.innerHTML = `<div class="opt-layout">${configCard}<div class="stack">${statusStack}</div></div>`;
}

// 前置检查条：每页唯一的状态通栏（替代旧的大标题 Header 区）
function renderPrecheck(targetId, tone, innerHtml) {
  const box = el(targetId);
  if (!box) return;
  box.innerHTML = innerHtml ? `<div class="precheck ${tone}">${innerHtml}</div>` : "";
}

function renderImportPage() {
  const validationBox = el("new-scene-validation");
  if (validationBox) validationBox.innerHTML = app.currentScene ? renderValidationPanel() : "";
  if (!app.currentScene) {
    renderPrecheck("import-precheck", "info", `尚未加载实例：请先上传 Excel 或下载模板准备数据。`);
    return;
  }
  const v = app.validation;
  if (app.validationBusy) {
    renderPrecheck("import-precheck", "info", `正在对实例数据做强校验…`);
  } else if (v?.status === "failed") {
    renderPrecheck("import-precheck", "err", `✕ 数据强校验未通过（${formatInt(v.error_count || 0)} 条错误），仿真/优化结果不可用，请先修复。`);
  } else if (v?.status === "warning") {
    renderPrecheck("import-precheck", "ok", `✓ 数据强校验已通过（${formatInt(v.warning_count || 0)} 条警告不影响仿真）<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="graph">下一步：图谱构建 →</button>`);
  } else if (v) {
    renderPrecheck("import-precheck", "ok", `✓ 数据强校验已通过<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="graph">下一步：图谱构建 →</button>`);
  } else {
    renderPrecheck("import-precheck", "warn", `实例已加载，尚未运行数据强校验。`);
  }
}

function renderGraphPage() {
  const container = el("graph-content");
  if (!container) return;
  const meta = app.graphMeta;
  if (graphBuildIsRunning()) {
    renderPrecheck("graph-precheck", "info", `图谱正在构建中 · ${formatInt(app.graphBuildStatus?.progress || 0)}%，页面会自动更新。`);
  } else if (meta) {
    renderPrecheck("graph-precheck", "ok", `✓ 图谱已构建完成 · 构建于 ${escapeHtml(formatDateTime(meta.created_at))} · 耗时 ${escapeHtml(formatDurationSeconds((meta.build_time_ms || 0) / 1000))}<span class="grow"></span><button class="btn btn-ghost btn-xs" type="button" data-action="build-graph">重新构建</button><button class="link-btn" type="button" data-nav-jump="simulate">下一步：规则仿真 →</button>`);
  } else {
    renderPrecheck("graph-precheck", "warn", `图谱尚未构建：图谱是仿真与优化的数据准备阶段。<span class="grow"></span><button class="btn btn-primary btn-xs" type="button" data-action="build-graph">构建图谱</button>`);
  }
  let html = "";
  if (graphBuildIsRunning() || app.graphBuildStatus?.status === "error") html += renderGraphBuildStatus();
  if (meta) {
    html += renderInteractiveGraph();
  } else if (!graphBuildIsRunning()) {
    html += `<div class="card card-pad">${renderEmptyState("尚未构建图谱", "点击上方「构建图谱」按钮，或完成数据校验后自动构建。", '<button class="btn btn-primary" type="button" data-action="build-graph">构建图谱</button>')}</div>`;
  }
  container.innerHTML = html;
  if (meta) requestAnimationFrame(() => mountInteractiveGraph());
}

function renderNewScene() {
  renderImportPage();
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
  const label = failed ? "校验未通过" : validation.status === "warning" ? "校验通过 · 有警告" : "校验通过";
  const issues = [...asArray(validation.errors), ...asArray(validation.warnings)];
  const collapsed = !!app.validationCollapsed;
  // 三档大色块摘要：通过项/警告/错误。后端校验接口只返回错误/警告明细与实体统计，
  // 「通过项」按已校验实体总数（stats 各计数之和）扣除问题项近似（设计稿口径，后端无逐条通过计数）。
  const statsTotal = ["orders", "tasks", "operations", "machines", "toolings", "personnel"]
    .reduce((sum, key) => sum + Number(validation.stats?.[key] || 0), 0);
  const errCount = Number(validation.error_count || 0);
  const warnCount = Number(validation.warning_count || 0);
  const passCount = Math.max(0, statsTotal - errCount - warnCount);
  return `
    <article class="surface-card validation-panel ${tone} ${collapsed ? "is-collapsed" : ""}">
      <div class="card-head">
        <div><h3>数据强校验</h3><p>错误级问题会导致仿真/优化静默失败。</p></div>
        <div style="display:flex;gap:10px;align-items:center">
          ${statusChip(label, tone === "danger" ? "danger" : tone)}
          ${renderCollapseButton("toggle-validation-collapse", collapsed, "数据强校验详情")}
        </div>
      </div>
      <div class="validation-body collapsible-body">
        <div class="valid-summary">
          <div class="ok"><strong>${formatInt(passCount)}</strong><span>通过项</span></div>
          <div class="warn"><strong>${formatInt(warnCount)}</strong><span>警告</span></div>
          <div class="err"><strong>${formatInt(errCount)}</strong><span>错误</span></div>
        </div>
        ${issues.length ? `<div class="tbl-wrap mt-16"><table>
          <thead><tr><th>级别</th><th>问题 Sheet</th><th>类别</th><th>实体</th><th>问题明细</th></tr></thead>
          <tbody>
            ${issues.slice(0, 50).map((item) => `<tr>
              <td>${statusChip(item.severity === "error" ? "错误" : "警告", item.severity === "error" ? "danger" : "warning")}</td>
              <td class="mono">${escapeHtml(item.sheet || "-")}</td>
              <td>${escapeHtml(item.category || "-")}</td>
              <td class="mono">${escapeHtml(item.entity || "-")}</td>
              <td>${escapeHtml(item.message || "-")}</td>
            </tr>`).join("")}
          </tbody>
        </table></div>
        <p class="subtle mt-16">${issues.length > 50 ? `共 ${issues.length} 条问题，仅展示前 50 条，导出 Excel 可查看全部。` : `共 ${issues.length} 条问题。`}</p>` : '<p class="subtle mt-16">未发现脏数据或格式问题，可以进入仿真与优化。</p>'}
        <div class="mt-16" style="display:flex;gap:10px">
          <button class="btn btn-ghost" type="button" data-action="run-validation">重新校验</button>
          ${issues.length ? '<button class="btn btn-ghost" type="button" data-action="export-validation">导出校验结果 Excel</button>' : ""}
          ${failed ? '<span class="subtle">请先修复上述错误，否则仿真指标会显示为 0。</span>' : ""}
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

// —— 方案对比：可配置列 + 与基线 KPI 对比 ——
const REVIEW_COMPARE_HIDDEN_LS = "review-compare-hidden-cols-v1";

function reviewCompareAllKeys() {
  const primaryKeys = activePrimaryObjectiveKeys();
  return { primaryKeys, allKeys: [...primaryKeys, ...REVIEW_KPI_KEYS.filter((key) => !primaryKeys.includes(key))] };
}

// 默认显示：3 个主目标 + 2 个常用 KPI（共 5 列）；localStorage 无记录时应用默认
function reviewDefaultHiddenColumns(allKeys, primaryKeys) {
  const visible = new Set([...primaryKeys, ...allKeys.filter((key) => !primaryKeys.includes(key)).slice(0, 2)]);
  return new Set(allKeys.filter((key) => !visible.has(key)));
}

function getReviewHiddenColumns(allKeys, primaryKeys) {
  try {
    const raw = localStorage.getItem(REVIEW_COMPARE_HIDDEN_LS);
    if (raw === null) return reviewDefaultHiddenColumns(allKeys, primaryKeys);
    return new Set(asArray(JSON.parse(raw)));
  } catch {
    return reviewDefaultHiddenColumns(allKeys, primaryKeys);
  }
}

function setReviewColumnHidden(key, hidden) {
  const { allKeys, primaryKeys } = reviewCompareAllKeys();
  const set = getReviewHiddenColumns(allKeys, primaryKeys);
  if (hidden) set.add(key); else set.delete(key);
  try { localStorage.setItem(REVIEW_COMPARE_HIDDEN_LS, JSON.stringify(Array.from(set))); } catch { /* 隐私模式忽略 */ }
}

// Δ 绝对值：利用率/比率类用百分点(pp)，计数类用整数，其余按小时
function formatDeltaAbs(key, delta) {
  const sign = delta > 0 ? "+" : "−";
  const mag = Math.abs(delta);
  if (isPercentMetric(key)) return `${sign}${(mag * 100).toFixed(1)}pp`;
  if (isCountMetric(key)) return `${sign}${formatInt(Math.round(mag))}`;
  return `${sign}${mag.toFixed(1)}h`;
}

// 单元格相对基线的 Δ 标注：箭头随数值升降，配色按改善(绿)/变差(红)；方向未知不判优、基线自身不标注
function compareDeltaHtml(item, key, baseline) {
  if (!baseline || item.id === baseline.id) return "";
  const value = metricValue(item, key);
  const base = metricValue(baseline, key);
  if (value === null || value === undefined || base === null || base === undefined) return "";
  if (!Number.isFinite(Number(value)) || !Number.isFinite(Number(base))) return "";
  const delta = Number(value) - Number(base);
  if (delta === 0) return `<span class="cmp-delta cmp-flat">持平</span>`;
  const direction = objectiveDirection(key);
  const improved = direction === "min" ? delta < 0 : direction === "max" ? delta > 0 : null;
  const tone = improved === null ? "cmp-flat" : improved ? "cmp-good" : "cmp-bad";
  const arrow = delta > 0 ? "▲" : "▼";
  const absStr = formatDeltaAbs(key, delta);
  const baseAbs = Math.abs(Number(base));
  const pctStr = baseAbs > 0 ? `（${delta > 0 ? "+" : "−"}${Math.abs((delta / baseAbs) * 100).toFixed(1)}%）` : "";
  return `<span class="cmp-delta ${tone}"><span class="cmp-arrow" aria-hidden="true">${arrow}</span>${absStr}${pctStr}</span>`;
}

// 方案对比表：单一决策工作台。左列勾选（共享上限 4），方案名 + 来源 chip，
// 随后主目标 KPI（带标记）+ 其他 KPI（列配置勾选，默认 3 主目标 + 2 常用共 5 列），
// 每列与基线方案 Δ 对比、最优值加粗标「最优」；末列「◎ 详情 / ✦ AI 评审 / ⬇ 导出」。
// 方案多时表体纵向滚动 + 表头吸顶；行内最优方案整行淡绿。
function renderReviewCandidateComparison() {
  const candidates = getReviewCandidates();
  if (!candidates.length) return "";
  const { primaryKeys, allKeys } = reviewCompareAllKeys();
  const hidden = getReviewHiddenColumns(allKeys, primaryKeys);
  const visibleKeys = allKeys.filter((key) => !hidden.has(key));
  // 基线方案：对比的锚点，用于每个 KPI 单元格的 Δ 标注
  const baseline = candidates.find((item) => item.source === "baseline") || null;
  // 聚焦方案置顶，其余保持原顺序（基线 → 方案 → 参照 → 精确冠军）
  const focusedId = app.reviewDetailId;
  const ordered = focusedId && candidates.some((item) => item.id === focusedId)
    ? [candidates.find((item) => item.id === focusedId), ...candidates.filter((item) => item.id !== focusedId)]
    : candidates;
  const bestByKey = Object.fromEntries(visibleKeys.map((key) => [key, bestMetricValue(candidates, key)]));
  // 行内最优：在可见列上拿到最多「最优」的方案整行淡绿（is-best-row）
  const bestWinCount = new Map(candidates.map((item) => [item.id, 0]));
  candidates.forEach((item) => {
    visibleKeys.forEach((key) => {
      const best = bestByKey[key];
      const value = metricValue(item, key);
      if (best !== null && value !== null && value !== undefined && Number.isFinite(Number(value)) && Number(value) === best) {
        bestWinCount.set(item.id, (bestWinCount.get(item.id) || 0) + 1);
      }
    });
  });
  const maxWins = Math.max(0, ...bestWinCount.values());
  const bestRowId = maxWins > 0 ? candidates.find((item) => bestWinCount.get(item.id) === maxWins)?.id : null;
  const cell = (item, key) => {
    const value = metricValue(item, key);
    const best = bestByKey[key];
    const isBest = best !== null && value !== null && value !== undefined && Number.isFinite(Number(value)) && Number(value) === best;
    const disp = metricDisplay(item, key);
    const isBaselineRow = baseline && item.id === baseline.id;
    const delta = isBaselineRow ? `<span class="delta flat">基准</span>` : compareDeltaHtml(item, key, baseline);
    return `<td data-col="${escapeHtml(key)}" class="${isBest ? "is-best" : ""}"><div class="cmp-cell"><span class="mono cmp-value">${isBest ? `<strong>${disp}</strong><span class="best-tag">最优</span>` : disp}</span>${delta}</div></td>`;
  };
  const rows = ordered.map((item) => {
    const checked = app.reviewSelection.includes(item.id);
    const colorIdx = schemeColorIndex(item.id);
    const rowClass = [checked ? "is-selected" : "", item.id === bestRowId ? "is-best-row" : "", colorIdx >= 0 ? `scheme-c-${colorIdx}` : ""].filter(Boolean).join(" ");
    const tag = candidateSourceTag(item);
    const detailOn = item.id === app.reviewGanttSchemeId;
    return `<tr class="${rowClass}">
      <td class="compare-check"><input type="checkbox" data-action="toggle-candidate" data-id="${escapeHtml(item.id)}" ${checked ? "checked" : ""} aria-label="勾选 ${escapeHtml(item.name)}"></td>
      <td class="compare-name">
        <strong>${colorIdx >= 0 ? `<span class="scheme-dot" style="background:${schemeColorToken(colorIdx)}"></span>` : ""}${escapeHtml(item.name)}</strong>
        <span class="chip ${tag.cls}">${escapeHtml(tag.label)}</span>
      </td>
      ${visibleKeys.map((key) => cell(item, key)).join("")}
      <td class="compare-ops"><div class="row-ops">
        <button class="op-btn detail ${detailOn ? "on" : ""}" type="button" data-action="focus-candidate" data-id="${escapeHtml(item.id)}" title="联动下方利用率与排程甘特">◎ 详情</button>
        <button class="op-btn ai" type="button" data-action="send-candidate-to-ai" data-id="${escapeHtml(item.id)}" title="送入 AI 评审">✦ AI 评审</button>
        <button class="op-btn exp" type="button" data-action="export-selected-solution" data-id="${escapeHtml(item.id)}" title="导出该方案 Excel">⬇ 导出</button>
      </div></td>
    </tr>`;
  }).join("");
  const colConfig = `
    <details class="col-config" ${app.reviewColConfigOpen ? "open" : ""}>
      <summary data-col-config-summary>列配置（${formatInt(visibleKeys.length)}/${formatInt(allKeys.length)}）</summary>
      <div class="col-config-panel">
        <div class="cp-head"><span>勾选要展示的 KPI 列</span><button type="button" data-action="reset-compare-cols">恢复默认</button></div>
        ${allKeys.map((key) => `<label class="col-config-item"><input type="checkbox" data-compare-col="${escapeHtml(key)}" ${hidden.has(key) ? "" : "checked"}> ${escapeHtml(getObjectiveLabel(key))}${primaryKeys.includes(key) ? ' <span class="primary-tag">主目标</span>' : ""}</label>`).join("")}
      </div>
    </details>`;
  const primaryLabels = primaryKeys.map((key) => getObjectiveLabel(key)).join(" / ");
  const baselineHint = baseline ? `每个 KPI 与基线方案（${escapeHtml(baseline.name)}）对比，箭头随数值升降，<span style="color:var(--success)">绿=改善</span> / <span style="color:var(--danger)">红=变差</span>，括号内为相对基线百分比。方案较多时表格纵向滚动，表头吸顶。` : "未加载基线方案，暂不显示对比箭头。";
  return `
    <article class="surface-card">
      <div class="card-head" style="align-items:center">
        <div>
          <h3>方案对比</h3>
          <p>默认展示主目标（<span style="color:var(--primary)">${escapeHtml(primaryLabels)}</span>）+ 2 个常用 KPI；更多指标在列配置中勾选。</p>
        </div>
        ${colConfig}
      </div>
      <div class="tbl-wrap">
        <table class="data-table compare-table">
          <thead><tr><th class="compare-check" style="width:34px"><span class="sr-only">选择方案</span></th><th>方案</th>${visibleKeys.map((key) => `<th data-col="${escapeHtml(key)}">${escapeHtml(getObjectiveLabel(key))}</th>`).join("")}<th style="width:250px">操作</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="subtle mt-16">${baselineHint}</p>
    </article>
  `;
}

async function loadReviewDailyUtil(schemeId) {
  const taskId = app.optimizeResult?.task_id;
  if (!taskId || !schemeId) return;
  try {
    const payload = await api.getMachineTypeDailyUtilization(taskId, schemeId);
    app.reviewDailyUtilCache[schemeId] = { loading: false, error: null, days: asArray(payload?.days), types: asArray(payload?.types) };
  } catch (error) {
    app.reviewDailyUtilCache[schemeId] = { loading: false, error: error.message || String(error), days: [], types: [] };
  }
  refreshReviewDynamicRegions();
}

function ensureReviewDailyUtil(schemeId) {
  const st = app.reviewDailyUtilCache[schemeId];
  if (st && (st.loading || st.types || st.error)) return st;
  app.reviewDailyUtilCache[schemeId] = { loading: true, error: null, days: [], types: null };
  loadReviewDailyUtil(schemeId);
  return app.reviewDailyUtilCache[schemeId];
}

function renderReviewTypeUtilization() {
  ensureReviewGanttSchemeSelection();
  const candidates = getReviewCandidates();
  const schemeId = app.reviewGanttSchemeId;
  const schemeName = candidates.find((c) => c.id === schemeId)?.name || schemeId || "";
  const legend = `
    <div class="util-legend">
      <span><i style="background:rgba(220,38,38,0.55)"></i>≥90% 过载</span>
      <span><i style="background:rgba(180,83,9,0.45)"></i>&lt;60% 偏低</span>
      <span><i style="background:rgba(18,161,80,0.5)"></i>行内最佳</span>
    </div>`;
  const head = `
    <div class="card-head" style="align-items:center">
      <div><h3>机器分类利用率对比 · 当前方案 <span style="color:var(--primary)">${escapeHtml(String(schemeName))}</span></h3>
      <p>日利用率 = 该类型当日有排产时长 /（该类型机器数 × 当日 24h）；用于定位哪天、哪类机器过载或闲置。点击对比表「◎ 详情」可切换方案。</p></div>
      ${legend}
    </div>`;
  if (!candidates.length || !schemeId) {
    return `<article class="surface-card">${head}<div class="empty-state"><p>请先运行优化或加载参照方案。</p></div></article>`;
  }
  if (!app.optimizeResult?.task_id) {
    return `<article class="surface-card">${head}<div class="empty-state"><p>运行混合优化后，可查看每日机器分类利用率。</p></div></article>`;
  }
  const st = ensureReviewDailyUtil(schemeId);
  let inner;
  if (st.loading) {
    inner = `<div class="empty-state"><p>正在计算每日机器分类利用率…</p></div>`;
  } else if (st.error) {
    inner = `<div class="empty-state"><p>加载失败：${escapeHtml(st.error)}</p><button class="btn btn-ghost btn-xs" type="button" data-action="retry-daily-util">重试</button></div>`;
  } else if (!asArray(st.types).length) {
    inner = `<div class="empty-state"><p>该方案的排程未覆盖任何机台类型。</p></div>`;
  } else {
    const days = asArray(st.days);
    const rows = st.types.map((type) => {
      const vals = asArray(type.per_day).filter((v) => v !== null && v !== undefined);
      const best = vals.length ? Math.max(...vals) : null;
      const cells = asArray(type.per_day).map((v) => {
        if (v === null || v === undefined) return "<td class=\"pct\">-</td>";
        const isBest = best !== null && v === best;
        // ≥90% 过载红、<60% 偏低黄、行内最佳绿加粗（行内最佳优先于过载/偏低配色）
        const cls = isBest ? "pct is-best" : v >= 0.9 ? "pct hi" : v < 0.6 ? "pct lo" : "pct";
        return `<td class="${cls}">${formatPercent(v)}</td>`;
      }).join("");
      return `<tr><td><strong>${escapeHtml(type.type_name)}</strong></td>${cells}</tr>`;
    }).join("");
    inner = `
      <div class="tbl-wrap util-table-wrap">
        <table class="util-day-table">
          <thead><tr><th>机器类型</th>${days.map((d) => `<th>D${escapeHtml(String(d))}</th>`).join("")}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }
  return `<article class="surface-card">${head}${inner}</article>`;
}

// 评审方案排程甘特：单方案 + 订单多选，复用统一甘特（renderTimeline，按机器分组 + 4 天默认窗口）。
function ensureReviewGanttSchemeSelection() {
  const candidates = getReviewCandidates();
  if (!candidates.length) { app.reviewGanttSchemeId = null; return; }
  if (!app.reviewGanttSchemeId || !candidates.some((c) => c.id === app.reviewGanttSchemeId)) {
    app.reviewGanttSchemeId = app.reviewDetailId || app.reviewSelection[0] || candidates[0].id;
  }
}

async function loadReviewSchemeSchedule(schemeId) {
  const taskId = app.optimizeResult?.task_id;
  if (!taskId || !schemeId) return;
  try {
    const payload = await api.getOptimizeSolutionSchedule(taskId, schemeId, "__all__");
    app.reviewSchemeCache[schemeId] = { loading: false, error: null, entries: asArray(payload?.entries), orders: asArray(payload?.orders) };
  } catch (error) {
    app.reviewSchemeCache[schemeId] = { loading: false, error: error.message || String(error), entries: null, orders: [] };
  }
  refreshReviewDynamicRegions();
}

function ensureReviewSchemeSchedule(schemeId) {
  const st = app.reviewSchemeCache[schemeId];
  if (st && (st.loading || st.entries || st.error)) return st;
  app.reviewSchemeCache[schemeId] = { loading: true, error: null, entries: null, orders: [] };
  loadReviewSchemeSchedule(schemeId);
  return app.reviewSchemeCache[schemeId];
}

// 甘特展示的方案由「方案对比」操作列的「查看详情」按钮联动（focus-candidate 同步 reviewGanttSchemeId），此处不再单独提供方案选择器。
function renderReviewGantt() {
  ensureReviewGanttSchemeSelection();
  const candidates = getReviewCandidates();
  const emptyHead = `<div class="card-head"><div><h3>排程甘特 · 当前方案</h3><p>与「◎ 详情」联动；样式与筛选能力和规则仿真甘特一致；默认显示前 4 天，可滚动查看全部。</p></div></div>`;
  if (!candidates.length) {
    return `<article class="surface-card">${emptyHead}<div class="empty-state"><p>暂无方案，请先运行优化或加载参照方案。</p></div></article>`;
  }
  if (!app.optimizeResult?.task_id) {
    return `<article class="surface-card">${emptyHead}<div class="empty-state"><p>运行混合优化后，可在此查看方案排产甘特。</p></div></article>`;
  }
  const schemeId = app.reviewGanttSchemeId;
  const schemeName = candidates.find((c) => c.id === schemeId)?.name || schemeId;
  const st = ensureReviewSchemeSchedule(schemeId);
  if (st.loading) {
    return `<article class="surface-card"><div class="empty-state"><p>正在加载该方案排程…</p></div></article>`;
  }
  if (st.error) {
    return `<article class="surface-card"><div class="empty-state"><p>加载排程失败：${escapeHtml(st.error)}</p><button class="btn-ghost" data-action="retry-review-scheme">重试</button></div></article>`;
  }
  if (!asArray(st.entries).length) {
    return `<article class="surface-card"><div class="empty-state"><p>该方案没有可显示的排程。</p></div></article>`;
  }
  return renderTimeline(st.entries, {
    title: `排程甘特 · 当前方案 ${schemeName}`,
    canvasId: "gantt-review-scheme",
    solutionIds: [schemeId],
  });
}

function renderReviewLibraryTab() {
  ensureReviewSelection();
  ensureReferenceSolutions();
  const candidates = getReviewCandidates();
  return `
    <div class="stack">
      <div id="review-comparison-region">
        ${candidates.length ? renderReviewCandidateComparison() : renderEmptyState(
          "暂无方案池",
          "先运行混合优化，或点击上方启发式规则加载参照方案。",
          '<button class="btn btn-primary" type="button" data-nav-jump="optimize">去启动优化</button>',
        )}
      </div>
      <div id="review-utilization-region">${candidates.length ? renderReviewTypeUtilization() : ""}</div>
      <div id="review-gantt-region">${candidates.length ? renderReviewGantt() : ""}</div>
    </div>
  `;
}

// 启发式参照 chips（Tab 条右侧）：基线/计算中/未计算/已纳入 四种状态
function renderReviewRefChips() {
  return `
    <div class="review-ref-chips">
      <span class="subtle">启发式参照：</span>
      ${CONFIG.HEURISTIC_RULES.map((rule) => {
        const isBaseline = rule === app.optimizeResult?.baseline?.rule_name;
        const computing = app.referenceSolutionsState.computing.includes(rule);
        const cached = !isBaseline && ruleIsCached(rule);
        const cls = isBaseline || cached ? "ok" : computing ? "warn" : "neutral";
        const stateText = isBaseline ? "基线" : computing ? "计算中…" : cached ? "已纳入" : "";
        const title = isBaseline
          ? "该规则已作为基线方案纳入对比"
          : computing ? "正在仿真计算…"
          : cached ? `${HEURISTIC_RULE_BLURB[rule] || "规则参考方案"}（已纳入对比）`
          : `${HEURISTIC_RULE_BLURB[rule] || "规则参考方案"}（首次计算约需 1~2 分钟）`;
        return `<button type="button" class="chip ${cls}" data-action="load-heuristic-rule" data-rule="${escapeHtml(rule)}" title="${escapeHtml(title)}" ${isBaseline || computing ? "disabled" : ""}>${escapeHtml(rule)}${stateText ? ` · ${stateText}` : ""}</button>`;
      }).join("")}
    </div>
  `;
}

function renderReviewExactTab() {
  const objectiveOptions = asArray(app.optimizeObjectiveCatalog).map((item) => `
    <label class="obj-chip ${Number(app.exactForm.weights[item.key] ?? (app.optimizeForm.objectiveKeys.includes(item.key) ? 1 : 0)) > 0 ? "on" : ""}" style="cursor:default">
      <input type="number" min="0" step="0.1" data-weight-key="${escapeHtml(item.key)}" value="${app.exactForm.weights[item.key] ?? (app.optimizeForm.objectiveKeys.includes(item.key) ? 1 : 0)}" style="width:56px;min-height:24px;padding:2px 6px">
      ${escapeHtml(item.label || item.key)}
    </label>
  `).join("");
  const exactName = app.exactReference
    ? (getReviewCandidates().find((item) => item.id === app.exactReference.solution_id)?.name || "精确冠军参考")
    : "";
  return `
    <div class="grid-2">
      <article class="surface-card">
        <div class="card-head"><div><h3>单目标精确冠军</h3><p>在某一个业务指标上给出高置信冠军方案。</p></div></div>
        <div class="form-grid">
          <label style="grid-column:span 2"><span>目标</span><select id="exact-single-objective">${renderExactObjectiveOptions()}</select></label>
          <label><span>时间预算（秒）</span><input id="exact-time-limit" type="number" min="5" value="${app.exactForm.timeLimitS}"></label>
        </div>
        <div class="mt-16"><button class="btn btn-primary" type="button" data-action="generate-exact-single">生成单目标冠军</button></div>
      </article>
      <article class="surface-card">
        <div class="card-head"><div><h3>加权单目标精确冠军</h3><p>把业务偏好转成加权目标，生成一个冠军方案。</p></div></div>
        <div class="obj-chip-grid">${objectiveOptions || renderEmptyState("暂无目标目录", "请先等待目标目录加载完成。")}</div>
        <div class="mt-16"><button class="btn btn-primary" type="button" data-action="generate-exact-weighted">生成加权冠军</button></div>
      </article>
    </div>
    ${app.exactReference ? `
      <article class="surface-card">
        <div class="card-head"><div><h3>最新精确冠军参考</h3><p>自动纳入方案库、AI 评审和导出流程。</p></div><span class="chip ok">已纳入方案库</span></div>
        <div class="kv-grid" style="grid-template-columns:repeat(3,1fr)">
          <div><span>方案</span><strong>${escapeHtml(exactName)}</strong></div>
          <div><span>模式</span><strong>${escapeHtml(app.exactReference.exact_info?.mode || app.exactReference.source || "-")}</strong></div>
          <div><span>求解状态</span><strong>${escapeHtml(app.exactReference.exact_info?.solver_status || app.exactReference.evaluationMode || "-")}</strong></div>
          <div><span>总延误</span><strong>${metricDisplay(app.exactReference, "total_tardiness")}</strong></div>
          <div><span>总周期</span><strong>${metricDisplay(app.exactReference, "makespan")}</strong></div>
          <div><span>净可用利用率</span><strong>${metricDisplay(app.exactReference, "avg_net_available_utilization")}</strong></div>
        </div>
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
      <div class="chat-stream" id="chat-stream">
        <div class="bubble ai"><strong>AI 方案助手</strong>请选择 1-4 个方案后，可以让我比较、按诉求推荐，或者追问某个方案为什么这样排。</div>
      </div>
    `;
  }
  return `
    <div class="chat-stream" id="chat-stream">
      ${app.aiConversation.map((item) => `
        <div class="bubble ${item.role === "user" ? "me" : "ai"}"><strong>${item.role === "user" ? "你" : "AI 方案助手"}</strong>${escapeHtml(item.content)}</div>
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
    <div class="grid-2">
      <article class="surface-card">
        <div class="card-head"><div><h3>当前纳入 AI 评审的方案</h3><p>AI 从主目标 + 全量 KPI + 风险解释三层给出意见。</p></div><span class="chip blue">${formatInt(selected.length)} 个方案</span></div>
        ${selected.length ? renderSimpleTable(
          ["方案", "来源", "总延误", "总周期", "净可用利用率"],
          selected.map((item) => [
            escapeHtml(item.name),
            escapeHtml(candidateSourceTag(item).label),
            metricDisplay(item, "total_tardiness"),
            metricDisplay(item, "makespan"),
            metricDisplay(item, "avg_net_available_utilization"),
          ]),
        ) : renderEmptyState("暂无已选方案", "先在方案库里勾选 1-4 个方案，再来让 AI 分析。")}
        ${selected.length ? renderCandidateMetricMatrix(selected, "AI 评审输入方案全量指标") : ""}
        <p class="subtle mt-16">在「方案库」勾选/取消勾选后这里同步更新；也可用操作列「✦ AI 评审」把单个方案直接送入。</p>
      </article>
      <article class="surface-card">
        <div class="card-head"><div><h3>对话式评审</h3><p>输入真实业务诉求，AI 在聊天流中直接回复。</p></div></div>
        ${renderAiConversation()}
        <div class="quick-acts">
          <button class="btn btn-ghost btn-xs" type="button" data-action="ai-compare" ${app.aiBusy ? "disabled" : ""}>比较已勾选方案</button>
          <button class="btn btn-ghost btn-xs" type="button" data-action="ai-recommend" ${app.aiBusy ? "disabled" : ""}>按诉求推荐方案</button>
          <button class="btn btn-ghost btn-xs" type="button" data-action="ai-ask" ${app.aiBusy ? "disabled" : ""}>追问当前方案</button>
        </div>
        <div class="form-grid mt-16">
          <label style="grid-column:span 2"><span>当前追问方案</span><select id="ai-solution-select">${options}</select></label>
        </div>
        <div class="chat-input">
          <textarea rows="2" id="ai-input" placeholder="例如：我们更看重主订单准交，其次看净可用利用率，哪个方案更适合？"></textarea>
          <button class="btn btn-primary" type="button" data-action="ai-recommend" ${app.aiBusy ? "disabled" : ""}>发送</button>
        </div>
        ${app.aiBusy ? `<p class="subtle" style="margin-top:6px">AI 正在分析…</p>` : ""}
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
  if (ganttRegion) {
    // 单方案甘特由 renderTimeline 走统一的 pendingGantts/mountGantts 渲染路径。
    ganttRegion.innerHTML = candidates.length ? renderReviewGantt() : "";
  }
  mountOrderComboboxes();
  requestAnimationFrame(() => mountGantts());
}

function renderReview() {
  const container = el("review-content");
  ensureReviewSelection();
  const candidates = getReviewCandidates();
  const selected = getSelectedReviewCandidates();
  const exactCount = candidates.filter((item) => String(item.source || "").includes("exact")).length;
  const heuristicCount = candidates.filter((item) => String(item.source || "").includes("heuristic") || String(item.source || "").includes("reference") || item.heuristicRuleName).length;
  const paretoCount = Math.max(0, candidates.length - exactCount - heuristicCount - (candidates.some((item) => item.source === "baseline") ? 1 : 0));
  // 前置检查条
  if (optimizeIsRunning()) {
    renderPrecheck("review-precheck", "warn", `⚠ 优化仍在运行（${optimizeProgress(app.optimizeStatus)}%），当前方案池为中间结果，完成后建议复核<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="optimize">查看优化进度 →</button>`);
  } else if (app.optimizeResult) {
    renderPrecheck("review-precheck", "ok", `✓ 优化已完成，可勾选方案对比、送入 AI 评审或导出交付<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="export">下一步：导出与交付 →</button>`);
  } else {
    renderPrecheck("review-precheck", "warn", `⚠ 尚无优化结果：可先加载启发式参照方案，或先运行混合优化<span class="grow"></span><button class="link-btn" type="button" data-nav-jump="optimize">去启动优化 →</button>`);
  }
  const statChips = `
    <div class="review-stat-chips" title="勾选 1–4 个方案驱动利用率、甘特与 AI 评审；「◎ 详情」联动下方分析区。">
      <span><i>Pareto</i><b>${formatInt(paretoCount)}</b></span>
      <span><i>启发式</i><b>${formatInt(heuristicCount)}</b></span>
      <span><i>精确冠军</i><b>${formatInt(exactCount)}</b></span>
      <span><i>已选</i><b id="review-selected-count">${formatInt(selected.length)}</b></span>
    </div>
  `;
  const tabRow = `
    <div class="review-tab-row">
      <div class="tabs">
        <button class="tab ${app.reviewTab === "library" ? "active" : ""}" type="button" data-review-tab="library">方案库</button>
        <button class="tab ${app.reviewTab === "exact" ? "active" : ""}" type="button" data-review-tab="exact">精确冠军参考</button>
        <button class="tab ${app.reviewTab === "ai" ? "active" : ""}" type="button" data-review-tab="ai">AI 评审助手</button>
      </div>
      <div class="review-tab-aside">
        ${statChips}
        ${renderReviewRefChips()}
      </div>
    </div>
  `;
  let pane = "";
  if (app.reviewTab === "library") pane = renderReviewLibraryTab();
  if (app.reviewTab === "exact") pane = `<div class="stack">${renderReviewExactTab()}</div>`;
  if (app.reviewTab === "ai") pane = renderReviewAiTab();
  container.innerHTML = `
    <div class="stack">
      ${tabRow}
      <div class="rtab-scroll" id="review-tab-scroll">${pane}</div>
    </div>
  `;
  if (app.reviewTab === "library") {
    refreshReviewDynamicRegions();
    ensureReviewData(getSelectedReviewCandidates());
  }
  requestAnimationFrame(() => mountGantts());
}

function renderExactObjectiveOptions() {
  return asArray(app.exactObjectiveCatalog)
    .map((item) => `<option value="${escapeHtml(item.key)}" ${item.key === app.exactForm.objectiveKey ? "selected" : ""}>${escapeHtml(item.label || item.key)}</option>`)
    .join("");
}

function renderExportPage() {
  const container = el("export-content");
  if (!container) return;
  const focused = getSelectedReviewCandidate();
  const focusedMetric = focused ? metricDisplay(focused, "total_tardiness") : "";
  container.innerHTML = `
    <div class="stack">
      <div class="export-card">
        <div class="eicon"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 3v12m0 0 4-4m-4 4-4-4M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/></svg></div>
        <div><h4>下载导入模板</h4><p>包含 9 类 Sheet 的标准模板与字段说明。</p></div>
        <button class="btn btn-ghost" type="button" data-action="download-template">下载模板</button>
      </div>
      <div class="export-card">
        <div class="eicon green"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M4 4h16v16H4zM4 9h16M9 4v16"/></svg></div>
        <div><h4>导出当前实例 CSV</h4><p>订单、工序、资源等全量数据打包，用于存档或外部分析。</p></div>
        <button class="btn btn-ghost" type="button" data-action="export-csv">导出 CSV</button>
      </div>
      <div class="export-card highlight">
        <div class="eicon orange"><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="m12 2 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 17l9 5 9-5"/></svg></div>
        <div><h4>导出当前方案 Excel</h4><p>当前聚焦：<strong>${escapeHtml(focused?.name || "未选择")}</strong>${focused && focusedMetric !== "-" ? `（总延误 ${focusedMetric}）` : ""}。含甘特排产明细、资源分配与 KPI 摘要。</p></div>
        <button class="btn btn-primary" type="button" data-action="export-selected-solution" ${focused ? `data-id="${escapeHtml(focused.id)}"` : "disabled"}>${focused ? `导出${escapeHtml(focused.name)}` : "暂无可导出方案"}</button>
      </div>
    </div>
  `;
}

function renderLlmPage() {
  const container = el("llm-content");
  if (!container) return;
  const configured = !!app.llmConfig?.has_key;
  const tested = app.llmTestResult?.status === "ok";
  const testMsg = app.llmTestResult?.msg || "";
  if (tested) {
    renderPrecheck("llm-precheck", "ok", `● 连接测试成功${testMsg ? ` · ${escapeHtml(testMsg)}` : ""}<span class="grow"></span><button class="link-btn" type="button" data-action="test-llm-config">重新测试</button>`);
  } else if (configured) {
    renderPrecheck("llm-precheck", "warn", `已配置 API Key，尚未验证连通性<span class="grow"></span><button class="link-btn" type="button" data-action="test-llm-config">测试连接</button>`);
  } else {
    renderPrecheck("llm-precheck", "warn", `未配置大模型：AI 评审助手将使用内置启发式回退。<span class="grow"></span><button class="link-btn" type="button" data-action="test-llm-config">测试连接</button>`);
  }
  container.innerHTML = `
    <div class="card card-pad">
      <div class="card-head"><div><h3>连接配置</h3><p>支持 OpenAI 兼容接口；未配置时 AI 评审助手使用启发式回退。</p></div></div>
      <div class="form-grid" style="grid-template-columns:1fr">
        <label><span>Base URL</span><input id="llm-base-url" type="text" value="${escapeHtml(app.llmConfig?.base_url || "")}"></label>
        <label><span>模型名称</span><input id="llm-model" type="text" value="${escapeHtml(app.llmConfig?.model || "")}"></label>
        <label><span>API Key</span><input id="llm-api-key" type="password" placeholder="${app.llmConfig?.has_key ? "已配置，如需更新请重新输入" : "请输入新的 API Key"}"></label>
      </div>
      <div class="mt-16" style="display:flex;gap:10px">
        <button class="btn btn-primary" type="button" data-action="save-llm-config">保存配置</button>
        <button class="btn btn-ghost" type="button" data-action="test-llm-config">测试连接</button>
      </div>
    </div>
  `;
}

function renderSystemPage() {
  const container = el("system-content");
  if (!container) return;
  const healthy = String(app.health?.status || "").toLowerCase() === "ok";
  const pageLabel = { import: "数据导入", graph: "图谱构建", simulate: "规则仿真", optimize: "优化求解", review: "方案评审", dashboard: "工作概览", export: "导出与交付", llm: "大模型连接", system: "系统状态" }[app.currentPage] || app.currentPage;
  container.innerHTML = `
    <div class="stack">
      <div class="grid-4">
        <div class="kpi-card"><span>后端健康</span><strong style="color:${healthy ? "var(--success)" : "var(--danger)"}">${healthy ? "healthy" : escapeHtml(app.health?.status || "未知")}</strong><small>/api/health · ${app.health ? "200 OK" : "未连通"}</small></div>
        <div class="kpi-card"><span>数据库实例</span><strong>${app.health?.has_instance ? "已加载" : "空"}</strong><small>llm4drd.db</small></div>
        <div class="kpi-card"><span>当前任务</span><strong class="mono" style="font-size:13px">${escapeHtml(app.optimizeTaskId || "-")}</strong><small>混合优化 · ${escapeHtml(app.optimizeStatus?.status || (app.optimizeResult ? "done" : "未运行"))}</small></div>
        <div class="kpi-card"><span>前端版本</span><strong>v3.0</strong><small>app_v2.js 20260720-7</small></div>
      </div>
      <div class="card card-pad">
        <div class="card-head"><div><h3>工作上下文</h3></div></div>
        <div class="kv-grid" style="grid-template-columns:repeat(4,1fr)">
          <div><span>当前实例</span><strong style="font-size:11px">${escapeHtml(app.currentScene?.name || "未加载")}</strong></div>
          <div><span>当前页面</span><strong>${escapeHtml(pageLabel)}</strong></div>
          <div><span>已选方案</span><strong>${formatInt(app.reviewSelection.length)} / 4</strong></div>
          <div><span>LLM 状态</span><strong style="color:${app.llmConfig?.has_key ? "var(--success)" : "var(--warning)"}">${app.llmConfig?.has_key ? "已配置" : "未配置"}</strong></div>
        </div>
      </div>
    </div>
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

  // 绘图层改用 cytoscape（力导向/层次布局）：仍复用 buildGraphViewModel 的筛选结果，
  // 但忽略其算出的 SVG lane 坐标；节点/边由 mountInteractiveGraph() 挂载后交给 cytoscape 布局。
  const svg = `<div class="graph-cytoscape-canvas" data-graph-canvas></div>`;

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


  const breadcrumb = `
    <div class="graph-breadcrumb" aria-label="订单簇统计">
      <span class="crumb-order"><i class="odot" style="background:${orderColor}"></i>订单 ${escapeHtml(orderEntityId || "未选择")}</span>
      <span class="sep2" aria-hidden="true">▸</span>
      <span>任务 <b>${formatInt(loadedCounts.task || 0)}</b></span>
      <span class="sep2" aria-hidden="true">▸</span>
      <span>工序 <b>${formatInt(loadedCounts.operation || 0)}</b></span>
      <span class="sep2" aria-hidden="true">▸</span>
      <span>机器/工装/人员 <b>${formatInt(loadedCounts.machine || 0)} / ${formatInt(loadedCounts.tooling || 0)} / ${formatInt(loadedCounts.personnel || 0)}</b></span>
      <span class="sep2" aria-hidden="true">▸</span>
      <span>关系 <b>${formatInt(loadedEdgeCount)}</b></span>
      <span class="spacer"></span>
      <span class="cstat">展示节点 <b>${formatInt(graph.visibleNodes.length)}</b>/${formatInt(graph.totalNodeCount)} · 边 <b>${formatInt(graph.visibleEdges.length)}</b>/${formatInt(graph.totalEdgeCount)}${graph.culledNodeCount > 0 || graph.culledEdgeCount > 0 ? ` · 已按焦点收敛，省略 ${formatInt(graph.culledNodeCount)} 节点 ${formatInt(graph.culledEdgeCount)} 边` : ""}</span>
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
    <div class="surface-card graph-workbench graph-v3">
      <div class="graph-toolbar">
        <div class="graph-mode">
          <button class="${app.graphView.mode === "focus" ? "on" : ""}" type="button" data-action="set-graph-mode" data-mode="focus">焦点邻域</button>
          <button class="${app.graphView.mode === "all" ? "on" : ""}" type="button" data-action="set-graph-mode" data-mode="all">全关系视图</button>
        </div>
        <span class="sep"></span>
        <label class="graph-order-filter">按订单过滤
          ${renderSingleSelectFilter({
            id: "graph-order",
            noun: "订单",
            options: asArray(app.graphOrderOptions).map(normalizeGraphNode).map((o) => ({
              id: o.entity_id || o.id,
              label: (() => {
                const e = o.entity_id || "";
                const n = o.label && o.label !== e ? o.label : "";
                return n ? `${e} · ${n}` : e;
              })(),
            })),
            selectedId: orderEntityId || null,
            onChange: async (id) => { await loadGraphOrder(id); return renderCurrentPage(); },
          })}
        </label>
      </div>
      ${breadcrumb}
      <div class="graph-filters">
        <span class="ft">节点层级</span>
        ${Object.entries(app.graphView.nodeTypes || {}).map(([type, enabled]) => `
          <button class="fchip ${enabled ? "on" : "off"}" type="button" data-action="toggle-graph-node-type" data-key="${escapeHtml(type)}">
            <span class="fdot" style="background:${graphTypeColor(type)}"></span>${escapeHtml(graphTypeLabel(type))}<span class="fc">${formatInt(graph.typeCounts[type] || 0)}</span>
          </button>
        `).join("")}
        <span class="ft" style="margin-left:14px">关系层级</span>
        ${Object.entries(app.graphView.edgeGroups || {}).map(([group, enabled]) => `
          <button class="fchip ${enabled ? "on" : "off"}" type="button" data-action="toggle-graph-edge-group" data-key="${escapeHtml(group)}">
            ${escapeHtml(GRAPH_EDGE_GROUP_LABELS[group] || group)}<span class="fc">${formatInt(graph.edgeGroupCounts[group] || 0)}</span>
          </button>
        `).join("")}
      </div>
      <div class="graph-main">
        <div class="graph-canvas-wrap">
          <div class="graph-cytoscape-canvas" data-graph-canvas></div>
          <div class="graph-canvas-overlay">
            ${GRAPH_NODE_ORDER.map((type) => `<span><i style="display:inline-block;width:9px;height:9px;border-radius:${type === "machine" ? "3px" : "50%"};background:${graphTypeColor(type)};margin-right:4px"></i>${escapeHtml(graphTypeLabel(type))}</span>`).join("")}
            <span style="color:var(--text-faint)">实线=结构 · 虚线=资源可行</span>
          </div>
          <div class="graph-zoom-ctl">
            <button class="btn btn-ghost btn-xs" type="button" data-action="fit-graph-view">适配</button>
            <button class="btn btn-ghost btn-xs" type="button" data-action="reset-graph-view">重置</button>
            <button class="btn btn-ghost btn-xs" type="button" data-action="zoom-graph-out">−</button>
            <span class="zval" data-graph-zoom>${Math.round(app.graphView.zoom * 100)}%</span>
            <button class="btn btn-ghost btn-xs" type="button" data-action="zoom-graph-in">＋</button>
            <button class="btn btn-ghost btn-xs" type="button" data-action="toggle-graph-fullscreen">全屏</button>
          </div>
        </div>
        <aside class="graph-detail-v3">
          <div class="dh">节点详情与关系解释</div>
          <div class="dbody">
            ${selectedSummary}
            ${selectedNode ? renderKeyValueGrid(graphNodeDetailRows(selectedNode, graph.relatedEdges)) : ""}
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
            ) : ""}
            <p class="subtle">统计基于选中节点的完整相关作用域，而非左侧可见局部图。点击 pill 可将对应节点居中聚焦。</p>
          </div>
        </aside>
      </div>
    </div>
  `;
}

function mountInteractiveGraph() {
  mountSingleSelectFilters();
  const container = document.querySelector(".page.active [data-graph-canvas]") || document.querySelector("[data-graph-canvas]");
  if (!container || container.dataset.bound === "1") return;
  container.dataset.bound = "1";

  if (typeof window.cytoscape !== "function") {
    container.innerHTML = '<div class="empty-state"><h3>图谱组件加载失败</h3><p>请确认 /static/vendor 下的 Cytoscape 运行库完整。</p></div>';
    return;
  }

  if (app.cyGraphInstance) {
    try { app.cyGraphInstance.destroy(); } catch (_) { /* 忽略销毁失败 */ }
    app.cyGraphInstance = null;
  }

  const root = container.closest(".graph-workbench") || document;
  const hoverPreview = root.querySelector("[data-graph-hover-preview]");

  // 复用 buildGraphViewModel 的筛选结果（节点类型/关系层级/搜索/焦点模式），
  // 但忽略其算出的 SVG lane 坐标，节点位置改由 cytoscape 的 dagre 层次布局决定。
  const graph = buildGraphViewModel();
  const nodes = asArray(graph?.visibleNodes);
  const edges = asArray(graph?.visibleEdges);
  const nodeIds = new Set(nodes.map((node) => node.id));

  const nodeDegree = new Map(nodes.map((node) => [node.id, 0]));
  edges.forEach((edge) => {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return;
    nodeDegree.set(edge.source, (nodeDegree.get(edge.source) || 0) + 1);
    nodeDegree.set(edge.target, (nodeDegree.get(edge.target) || 0) + 1);
  });

  const elements = [
    ...nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.label || node.entity_id || node.id,
        node_type: node.type,
        entity_id: node.entity_id || "",
        type_label: graphTypeLabel(node.type),
        degree: nodeDegree.get(node.id) || 0,
      },
    })),
    ...edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)).map((edge, index) => ({
      data: {
        id: `cy-edge-${edge.id || index}`,
        source: edge.source,
        target: edge.target,
        edge_type: edge.edgeType,
        edge_group: ["structure", "resource", "other"].includes(edge.group) ? edge.group : "other",
      },
    })),
  ];

  const cy = window.cytoscape({
    container,
    elements,
    // 节点形状/配色严格对齐 8545cc0 的 cytoscape 原版样式（各 node_type 对应 shape + background-color）
    style: [
      { selector: "node", style: { "label": "data(label)", "font-size": 9, "color": "#32485a", "text-valign": "bottom", "text-margin-y": 7, "text-max-width": 100, "text-wrap": "ellipsis", "width": 24, "height": 24, "border-width": 2, "border-color": "#ffffff" } },
      { selector: 'node[node_type="order"]', style: { "background-color": graphTypeColor("order"), "shape": "diamond", "width": 38, "height": 38, "font-size": 11, "font-weight": 700 } },
      { selector: 'node[node_type="task"]', style: { "background-color": graphTypeColor("task"), "shape": "round-rectangle", "width": 32, "height": 22 } },
      { selector: 'node[node_type="operation"]', style: { "background-color": graphTypeColor("operation"), "shape": "ellipse", "width": 19, "height": 19, "font-size": 8 } },
      { selector: 'node[node_type="machine"]', style: { "background-color": graphTypeColor("machine"), "shape": "hexagon", "width": 32, "height": 32 } },
      { selector: 'node[node_type="tooling"]', style: { "background-color": graphTypeColor("tooling"), "shape": "round-diamond", "width": 29, "height": 29 } },
      { selector: 'node[node_type="personnel"]', style: { "background-color": graphTypeColor("personnel"), "shape": "pentagon", "width": 29, "height": 29 } },
      { selector: "edge", style: { "width": 1.2, "line-color": "#9caebe", "target-arrow-color": "#9caebe", "target-arrow-shape": "triangle", "curve-style": "bezier", "arrow-scale": 0.7, "opacity": 0.62 } },
      { selector: 'edge[edge_group="structure"]', style: { "line-color": "#0f4c81", "target-arrow-color": "#0f4c81", "opacity": 0.72 } },
      { selector: 'edge[edge_group="resource"]', style: { "line-style": "dashed", "line-color": "#b76800", "target-arrow-color": "#b76800", "opacity": 0.58 } },
      { selector: ".cy-selected", style: { "border-width": 5, "border-color": "#102f4c", "width": 46, "height": 46, "font-size": 12, "font-weight": 700, "z-index": 9999 } },
      { selector: ".cy-neighbor", style: { "border-width": 3, "border-color": "#0f4c81", "opacity": 1, "z-index": 100 } },
      { selector: ".cy-neighbor-edge", style: { "width": 3, "opacity": 1, "z-index": 100 } },
      { selector: ".cy-dimmed", style: { "opacity": 0.55 } },
    ],
    minZoom: 0.05,
    maxZoom: 5,
    boxSelectionEnabled: false,
  });
  app.cyGraphInstance = cy;

  const runLayout = () => {
    try {
      cy.layout({ name: "dagre", rankDir: "TB", rankSep: 150, nodeSep: 28, edgeSep: 10, ranker: "tight-tree", padding: 40, fit: true, animate: false }).run();
    } catch (error) {
      console.warn("Dagre layout unavailable, falling back to breadthfirst", error);
      cy.layout({ name: "breadthfirst", directed: true, spacingFactor: 1.2, padding: 40, fit: true }).run();
    }
  };
  runLayout();

  // 高亮当前选中节点及其邻域，与右侧详情面板的选中态保持一致
  const selected = cy.getElementById(app.selectedGraphNodeId || "");
  if (selected.length) {
    cy.elements().addClass("cy-dimmed");
    const neighborhood = selected.neighborhood();
    selected.removeClass("cy-dimmed").addClass("cy-selected");
    neighborhood.nodes().removeClass("cy-dimmed").addClass("cy-neighbor");
    selected.connectedEdges().removeClass("cy-dimmed").addClass("cy-neighbor-edge");
    // 不再缩放到选中邻域：保持布局的 fit:true 全览，让全部节点默认呈现
  }

  const zoomPill = root.querySelector("[data-graph-zoom]");
  const syncZoomPill = () => { if (zoomPill) zoomPill.textContent = `${Math.round(cy.zoom() * 100)}%`; };
  cy.on("zoom", syncZoomPill);
  syncZoomPill();

  if (hoverPreview) {
    cy.on("mouseover", "node", (event) => {
      const data = event.target.data();
      hoverPreview.textContent = `悬浮预览：${data.type_label || "-"} / ${data.label || "-"} / 关联关系 ${data.degree || 0}`;
    });
    cy.on("mouseout", "node", () => {
      const sel = cy.getElementById(app.selectedGraphNodeId || "");
      if (sel.length) {
        const d = sel.data();
        hoverPreview.textContent = `当前选中：${d.type_label || "-"} / ${d.label || "-"} / 关联关系 ${d.degree || 0}`;
      } else {
        hoverPreview.textContent = "将鼠标悬浮到节点上，可快速预览该节点名称、类型与关联关系强度。";
      }
    });
  }

  // 点节点 → 复用现有选中逻辑：更新 selectedGraphNodeId 并整页重渲染，
  // 右侧详情面板与关系解释照常由 renderInteractiveGraph 更新。
  cy.on("tap", "node", (event) => {
    app.selectedGraphNodeId = event.target.id();
    renderCurrentPage();
  });
}

function selectedGraphOrderOption() {
  return asArray(app.graphOrderOptions).find((node) => {
    return normalizeGraphNode(node).id === app.selectedGraphOrderId;
  }) || null;
}

function mountGantts() {
  mountOrderComboboxes();
  mountMultiSelectFilters();
  mountSingleSelectFilters();
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
        // 关闭 vis 内置 XSS 净化：分组标签内容为本地调度数据（订单号/机器名等，非外部 HTML），
        // 净化会剥掉 .gantt-order-dot / .gantt-group-count 等 span 的 class/style，导致订单色点丢失、标签样式失效
        xss: { disabled: true },
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
  if (app.currentPage === "import") {
    renderImportPage();
    if (!app.validation && !app.validationBusy && app.currentScene) handleRunValidation(true);
  }
  if (app.currentPage === "graph") renderGraphPage();
  if (app.currentPage === "simulate") renderSimulatePage();
  if (app.currentPage === "optimize") renderOptimizePage();
  if (app.currentPage === "dashboard") renderDashboard();
  if (app.currentPage === "review") renderReview();
  if (app.currentPage === "export") renderExportPage();
  if (app.currentPage === "llm") renderLlmPage();
  if (app.currentPage === "system") renderSystemPage();
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
    if (["optimize", "review", "dashboard", "system"].includes(app.currentPage)) await renderCurrentPage();
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
      refreshWorkflowOverview({ force: true });
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
  app.orderComboboxRecent.reset();
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
  // 校验面板默认展开（三档大色块摘要是设计稿的主视觉）；用户可手动折叠
  app.validationCollapsed = false;
  // Re-render validation panel on import page after validation completes
  if (app.currentPage === "import") {
    const box = el("new-scene-validation");
    if (box) box.innerHTML = app.currentScene ? renderValidationPanel() : "";
  }
  refreshWorkflowOverview({ force: true });
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
    app.validationCollapsed = false;
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
    if (app.currentPage !== "import") await navigate("import");
    else await renderCurrentPage();
    syncTopbarAndNav();
    refreshWorkflowOverview({ force: true });
    // 强校验通过后自动构建图谱，构建进度显示在「图谱构建」页
    if (!failed) {
      await handleBuildGraph();
      // 图谱构建完成后再次刷新侧栏状态灯
      syncTopbarAndNav();
      refreshWorkflowOverview({ force: true });
    }
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
    app.validationCollapsed = false;
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
  else if (app.currentPage === "graph") await renderCurrentPage();
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
  refreshWorkflowOverview({ force: true });
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
    refreshWorkflowOverview({ force: true });
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
    refreshWorkflowOverview({ force: true });
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
  // 文件名用方案显示名（如「方案1」「基线方案 · ATC」），替换掉不合法的文件名字符。
  const filename = `${String(candidate.name || candidate.id || "solution").replace(/[\\/:*?"<>|]+/g, "_")}_排产.xlsx`;
  // 混合优化任务上下文：对比表与甘特都用 app.optimizeResult.task_id 拉取排产，导出端点与其共用
  // _resolve_export_solution，故凡甘特能渲染的候选（基线/启发式参照/pareto/references）都能导出。
  const taskId = app.optimizeResult?.task_id || app.optimizeTaskId;
  // 精确冠军参考不在 hybrid task 内、但自带 schedule；无 taskId 时也退回客户端明细导出。
  const canUseTask = taskId && candidate.source !== "exact_reference";
  const exportFromSchedule = async () => {
    if (!candidate.schedule?.length) throw new Error("该方案暂无排产明细可导出。");
    return api.exportSimExcel({ gantt: candidate.schedule, rule: candidate.name, metrics: candidate.metrics || {} });
  };
  try {
    const blob = canUseTask
      ? await api.exportOptimizeSolution(taskId, candidate.id)
      : await exportFromSchedule();
    downloadBlob(blob, filename);
    toast("方案导出成功。", "success");
  } catch (error) {
    // 优化任务解析失败时，若候选自带排产明细，退回客户端明细导出兜底。
    if (canUseTask && candidate.schedule?.length) {
      try {
        downloadBlob(await exportFromSchedule(), filename);
        toast("方案导出成功。", "success");
        return;
      } catch (_) {
        // 兜底也失败则落到下方统一提示
      }
    }
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
    await renderCurrentPage();
  } catch (error) {
    toast(`保存配置失败：${error.message}`, "warning");
  }
}

async function handleTestLlmConfig() {
  try {
    const result = await api.testLlmConfig();
    app.llmTestResult = { status: "ok", msg: result?.msg || result?.message || "" };
    toast(result?.message || "大模型连接测试通过。", "success");
  } catch (error) {
    app.llmTestResult = { status: "error", msg: error.message || "" };
    toast(`连接测试失败：${error.message}`, "warning");
  }
  await renderCurrentPage();
}

async function handleAction(action, target) {
  if (action === "toggle-sidebar") {
    const collapsed = document.querySelector(".app")?.classList.toggle("is-sidebar-collapsed");
    try { localStorage.setItem(CONFIG.SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch { /* 隐私模式忽略 */ }
    return;
  }
  if (action === "retry-plan-gantt") return loadPlanGantt(app.planGantt.taskId, app.planGantt.solutionId, app.planGantt.orderId);
  if (action === "goto-new-scene") return navigate("import");
  if (action === "goto-dashboard") return navigate("dashboard");
  if (action === "goto-review") return navigate("review");
  if (action === "select-sim-rule") {
    app.simRule = target.dataset.rule || app.simRule;
    return renderCurrentPage();
  }
  if (action === "reset-compare-cols") {
    try { localStorage.removeItem(REVIEW_COMPARE_HIDDEN_LS); } catch { /* 忽略 */ }
    app.reviewColConfigOpen = true;
    refreshReviewDynamicRegions();
    return;
  }
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
    const cy = app.cyGraphInstance;
    if (cy) { cy.zoom({ level: Math.min(cy.maxZoom(), cy.zoom() * 1.15), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }); return; }
    app.graphView.zoom = Math.min(2.4, app.graphView.zoom * 1.15);
    return renderCurrentPage();
  }
  if (action === "zoom-graph-out") {
    const cy = app.cyGraphInstance;
    if (cy) { cy.zoom({ level: Math.max(cy.minZoom(), cy.zoom() * 0.87), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }); return; }
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
    const cy = app.cyGraphInstance;
    if (cy) { cy.fit(undefined, 40); return; }
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
    if (!app.simResult?.gantt?.length) {
      toast("当前没有可导出的仿真排程，请先运行一次规则仿真。", "warning");
      return;
    }
    try {
      const blob = await api.exportSimExcel(app.simResult);
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
  if (action === "generate-exact-single") return handleGenerateExact("single");
  if (action === "generate-exact-weighted") return handleGenerateExact("weighted");
  if (action === "export-selected-solution") return handleExportSolution(target?.dataset.id || getSelectedReviewCandidate()?.id);
  if (action === "focus-candidate") {
    // ◎ 详情：联动下方利用率对比与排程甘特切换到该方案
    const id = target.dataset.id;
    app.reviewDetailId = id;
    app.reviewGanttSchemeId = id;
    persistReviewProgress();
    updateShell();
    return renderCurrentPage();
  }
  if (action === "retry-review-scheme") {
    const id = app.reviewGanttSchemeId;
    if (id) { delete app.reviewSchemeCache[id]; refreshReviewDynamicRegions(); }
    return;
  }
  if (action === "retry-daily-util") {
    const id = app.reviewGanttSchemeId;
    if (id) { delete app.reviewDailyUtilCache[id]; refreshReviewDynamicRegions(); }
    return;
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
    const stepNavKey = { 1: "import", 2: "graph", 3: "simulate", 4: "optimize", 5: "review" }[step] || "import";
    return navigate(stepNavKey);
  }
  if (action === "toggle-graph-fullscreen") {
    // 只全屏图谱画布区（含图例悬浮层与缩放控制），不含工具栏/筛选/右侧详情栏
    const workbench = target.closest(".graph-workbench") || document.querySelector(".graph-workbench");
    const shell = workbench?.querySelector(".graph-canvas-wrap");
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
    const fullscreenGraph = document.fullscreenElement?.classList.contains("graph-canvas-wrap")
      ? document.fullscreenElement
      : null;
    document.querySelectorAll('[data-action="toggle-graph-fullscreen"]').forEach((button) => {
      const active = !!fullscreenGraph && button.closest(".graph-canvas-wrap") === fullscreenGraph;
      button.textContent = active ? "退出全屏" : "全屏";
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
    // 全屏进出会改变 cytoscape 容器尺寸，需要 resize + 重新适配视图
    if (app.cyGraphInstance) {
      requestAnimationFrame(() => {
        try { app.cyGraphInstance.resize(); app.cyGraphInstance.fit(undefined, 40); } catch (_) { /* 忽略 */ }
      });
    }
  });

  document.addEventListener("click", async (event) => {
    // 列配置面板开合：镜像 <details> 原生状态，供跨区域重渲染后保持展开
    const colCfgSummary = event.target.closest("[data-col-config-summary]");
    if (colCfgSummary) {
      app.reviewColConfigOpen = !app.reviewColConfigOpen;
      return;
    }
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
      // chip 网格：同步高亮态与计数文案，不整页重渲
      document.querySelectorAll("[data-objective-key]").forEach((node) => {
        node.closest(".obj-chip")?.classList.toggle("on", node.checked);
      });
      const countEl = el("opt-obj-count");
      if (countEl) countEl.textContent = `已选 ${selected.length} / ${app.optimizeObjectiveCatalog.length} 个目标，参与 Pareto 前沿求解。`;
      updateOptimizeBudgetHint();
      return;
    }
    if (target.matches("[data-compare-col]")) {
      app.reviewColConfigOpen = true;
      setReviewColumnHidden(target.dataset.compareCol, !target.checked);
      refreshReviewDynamicRegions();
      return;
    }
    if (target.matches("[data-gantt-group-mode]")) {
      app.ganttGroupMode[target.dataset.canvas] = target.value;
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
    if (target.matches("[data-graph-order-input]")) {
      const value = String(target.value || "").trim();
      if (value) await searchGraphOrderAndRender(value);
      return;
    }
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
    const navKey = window.location.hash.replace("#", "") || (app.currentScene ? "dashboard" : "import");
    await navigate(navKey, false);
  });

  // 拖拽导入：Excel 文件可直接拖到上传 Hero 区
  const dropHero = document.getElementById("drop-hero");
  if (dropHero) {
    ["dragenter", "dragover"].forEach((type) => dropHero.addEventListener(type, (event) => {
      event.preventDefault();
      dropHero.classList.add("dragging");
    }));
    ["dragleave", "drop"].forEach((type) => dropHero.addEventListener(type, (event) => {
      event.preventDefault();
      dropHero.classList.remove("dragging");
    }));
    dropHero.addEventListener("drop", async (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) await handleImportFile(file);
    });
  }
}

async function init() {
  try {
    if (localStorage.getItem(CONFIG.SIDEBAR_COLLAPSED_KEY) === "1") {
      document.querySelector(".app")?.classList.add("is-sidebar-collapsed");
    }
  } catch { /* 隐私模式忽略 */ }
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
  const navKey = window.location.hash.replace("#", "") || (app.currentScene ? "dashboard" : "import");
  await navigate(navKey, false);
  if (!app.currentScene || app.importBusy) return;
  // 库里没有可用的校验结论（如实例刚被改过）时才补算一次。
  if (!app.validation) await handleRunValidation(true);
  if (await loadExistingGraph() && !app.importBusy) await renderCurrentPage();
  refreshWorkflowOverview({ force: true });
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    console.error(error);
    toast(`应用初始化失败：${error.message}`, "warning");
  });
});
