const CONFIG = {
  API_BASE: "/api",
  HISTORY_KEY: "llm4drd_v2_scene_history",
  OPT_POLL_MS: 1500,
  TABLE_LIMIT: 40,
  HEURISTIC_RULES: ["ATC", "EDD", "SPT", "CR", "FIFO", "LPT"],
  GRAPH_FOCUS_NODE_LIMIT: 80,
  GRAPH_ALL_NODE_LIMIT: 150,
  GRAPH_ALL_EDGE_LIMIT: 320,
};

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
  // 旧的通用 "workflow" 书签 hash（原 rail 导航已移除）统一落到「仿真与洞察」图谱视图。
  workflow: { page: "workflow", workflowStep: 3, workflowFocus: "graph", requiresScene: true },
  graph: { page: "workflow", workflowStep: 3, workflowFocus: "graph", requiresScene: true },
  simulate: { page: "workflow", workflowStep: 3, workflowFocus: "graph", requiresScene: true },
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
  importBusy: false,
  simBusy: false,
  instanceDb: null,
  downtimes: [],
  graphMeta: null,
  graphNodes: [],
  graphEdges: [],
  cyGraphInstance: null,
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
  workflowFocus: null,
  filters: { orders: "", operations: "", resources: "", downtime: "" },
  reviewSelection: [],
  reviewDetailId: null,
  heuristicSelection: ["ATC", "EDD"],
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
};

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
      throw new Error(message || `请求失败（HTTP ${response.status}）`);
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
  getInstanceDetails() { return this.json("/instance/details"); },
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
  validateInstance() { return this.json("/instance/validate"); },
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
  simulate(ruleName) { return this.json("/simulate", "POST", { rule_name: ruleName }); },
  exportSimExcel() { return this.request("/simulate/export-excel"); },
  simulateReferenceSolutions(ruleNames, objectiveKeys) {
    return this.json("/simulate/reference-solutions", "POST", {
      rule_names: ruleNames,
      objective_keys: objectiveKeys,
    });
  },
  getOptimizeObjectives() { return this.json("/optimize/objectives"); },
  startHybridOptimize(payload) { return this.json("/optimize/hybrid", "POST", payload); },
  getOptimizeStatus(taskId) { return this.json(`/optimize/hybrid/status/${taskId}`); },
  getOptimizeResult(taskId) { return this.json(`/optimize/hybrid/result/${taskId}`); },
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
  const state = String(status?.status || "").toLowerCase();
  if (state === "done") return 100;
  if (state === "error") return Math.min(99, Math.max(2, Number(status?.progress || 0)));
  if (state === "submitting") return 2;
  const phase = String(status?.phase || "initializing");
  const generationRatio = Number(status?.current_generation || 0) / Math.max(1, Number(status?.config?.generations || app.optimizeForm.generations || 1));
  const timeRatio = Number(status?.elapsed_s || 0) / Math.max(1, Number(status?.config?.time_limit_s || app.optimizeForm.timeLimitS || 1));
  if (phase === "coarse") return Math.round(8 + Math.min(1, Math.max(generationRatio, timeRatio)) * 57);
  if (phase === "exact_promotion") return 72;
  if (phase === "elite_refine") return 84;
  if (phase === "finalize") return 94;
  return 5;
}

function renderOptimizeStatus() {
  const status = app.optimizeStatus;
  if (!status) return "";
  const state = String(status.status || "running").toLowerCase();
  const failed = state === "error" || state === "failed";
  const done = state === "done" || state === "completed" || state === "success";
  const tone = failed ? "danger" : done ? "success" : "info";
  const label = failed ? "优化失败" : done ? "优化完成" : state === "submitting" ? "正在提交优化任务" : "优化正在运行";
  const progress = optimizeProgress(status);
  const message = failed
    ? (status.error || status.message || "未收到具体错误说明")
    : done
      ? (status.message || "优化完成，方案已可用于评审")
      : (status.message || "任务已提交，正在等待优化器返回进度");
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
        <span>最近更新：${status.updated_at || status.received_at ? escapeHtml(formatDateTime(status.updated_at || status.received_at)) : "等待首次进度"}</span>
      </div>
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
  return PRIMARY_KPI_LABELS[key] || key;
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
      name: item.solution_id || `Pareto-${index + 1}`,
    }));
  });
  asArray(app.optimizeResult?.reference_solutions).forEach((item) => {
    items.push(normalizeCandidate(item, {
      source: item.source || "reference",
      name: item.rule_name ? `启发式参考 · ${item.rule_name}` : item.solution_id,
    }));
  });
  asArray(app.referenceSolutions).forEach((item) => {
    items.push(normalizeCandidate(item, {
      source: item.source || "heuristic",
      name: item.rule_name ? `启发式参考 · ${item.rule_name}` : item.solution_id,
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

function machineShiftWindows(machine) {
  if (asArray(machine?.shift_windows).length) {
    return asArray(machine.shift_windows)
      .map((item) => ({
        start: Number(item.start),
        end: Number(item.end),
      }))
      .filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start)
      .sort((a, b) => a.start - b.start);
  }
  return asArray(machine?.shifts)
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
  const [details, db, downtimes] = await Promise.all([
    api.getInstanceDetails(),
    api.getInstanceDb(),
    api.getDowntimes().catch(() => []),
  ]);
  app.instanceDetails = details;
  app.instanceDb = db;
  app.downtimes = Array.isArray(downtimes?.downtimes)
    ? downtimes.downtimes
    : (Array.isArray(downtimes) ? downtimes : []);
  refreshOptimizeBudgetRecommendation({ preserveManual: true });
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
    if (!silent) toast("当前没有可用实例，请先生成或导入。", "warning");
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
  el("topbar-scene-meta").textContent = hasScene ? "查看实例摘要" : "请新建或导入实例";
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
  if (resolved.workflowFocus) app.workflowFocus = resolved.workflowFocus;
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

  const shifts = asArray(machine?.shift_windows).length
    ? asArray(machine.shift_windows).map((item) => ({
        start: Number(item.start),
        end: Number(item.end),
      }))
    : asArray(machine?.shifts).map((item) => ({
        start: Number(item.day || 0) * 24 + Number(item.start_hour ?? item.start ?? 0),
        end: Number(item.day || 0) * 24 + Number(item.end_hour ?? item.end ?? 0),
      })).filter((item) => !Number.isNaN(item.start) && !Number.isNaN(item.end) && item.end > item.start);

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
  const groups = Array.from(groupsMap, ([id, content]) => ({ id, content: escapeHtml(content) }));

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

function graphTypeColor(type) {
  const palette = {
    order: "#0f4c81",
    task: "#2c7fb0",
    operation: "#4d908e",
    machine: "#b76800",
    tooling: "#a23b72",
    personnel: "#6c5ce7",
  };
  return palette[String(type || "").toLowerCase()] || "#7a8795";
}

function graphTypeLabel(type) {
  return GRAPH_TYPE_LABELS[String(type || "").toLowerCase()] || GRAPH_TYPE_LABELS.other;
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
      if (orderNodeId) taskToOrderNodeId.set(node.id, orderNodeId);
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
  const columnGap = usedGroups.length > 4 ? 188 : 210;
  const width = Math.max(1240, 160 + usedGroups.length * columnGap);
  const rowGap = 78;
  const maxRows = Math.max(...usedGroups.map(([, items]) => items.length), 1);
  const height = Math.max(560, 150 + maxRows * rowGap);
  const placed = new Map();

  usedGroups.forEach(([type, items], colIndex) => {
    items.sort((a, b) => {
      const aRank = a.id === selectedId ? 0 : directNeighbors.has(a.id) ? 1 : 2;
      const bRank = b.id === selectedId ? 0 : directNeighbors.has(b.id) ? 1 : 2;
      if (aRank !== bRank) return aRank - bRank;
      const degreeGap = (degree.get(b.id) || 0) - (degree.get(a.id) || 0);
      if (degreeGap !== 0) return degreeGap;
      return String(a.label).localeCompare(String(b.label), "zh-CN");
    });
    const totalHeight = (items.length - 1) * rowGap;
    const startY = Math.max(88, (height - totalHeight) / 2);
    items.forEach((node, rowIndex) => {
      placed.set(node.id, {
        x: 110 + colIndex * columnGap,
        y: startY + rowIndex * rowGap,
        node,
        type,
      });
    });
  });

  return { width, height, placed };
}

function renderInteractiveGraph() {
  const nodes = asArray(app.graphNodes).slice(0, 80);
  const edges = asArray(app.graphEdges).slice(0, 140);
  if (!nodes.length) {
    return renderEmptyState("暂无图谱节点", "请先构建图谱，或确认当前实例已正确加载。");
  }

  const layout = layoutGraph(nodes, edges);
  const selectedId = app.selectedGraphNodeId || nodes[0]?.node_id || nodes[0]?.id;
  const selected = layout.placed.get(selectedId) || Array.from(layout.placed.values())[0];
  const neighborIds = new Set();
  layout.edges.forEach((edge) => {
    const src = edge.src || edge.source;
    const dst = edge.dst || edge.target;
    if (src === selected?.node?.node_id || src === selected?.node?.id || dst === selected?.node?.node_id || dst === selected?.node?.id) {
      neighborIds.add(src);
      neighborIds.add(dst);
    }
  });

  const svg = `
    <svg viewBox="0 0 ${layout.width} ${layout.height}" class="graph-svg" role="img" aria-label="异构图预览">
      <defs>
        <marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8ca0b5"></path>
        </marker>
      </defs>
      ${layout.edges.map((edge) => {
        const srcId = edge.src || edge.source;
        const dstId = edge.dst || edge.target;
        const src = layout.placed.get(srcId);
        const dst = layout.placed.get(dstId);
        const highlighted = neighborIds.has(srcId) && neighborIds.has(dstId);
        return `
          <line
            x1="${src.x + 26}" y1="${src.y + 18}"
            x2="${dst.x - 26}" y2="${dst.y + 18}"
            stroke="${highlighted ? "#0f4c81" : "#c4d0db"}"
            stroke-width="${highlighted ? 2.4 : 1.2}"
            opacity="${highlighted ? 0.92 : 0.58}"
            marker-end="url(#graph-arrow)"
          ></line>
        `;
      }).join("")}
      ${Array.from(layout.placed.values()).map((item) => {
        const id = item.node.node_id || item.node.id;
        const isSelected = id === (selected?.node?.node_id || selected?.node?.id);
        const isNeighbor = neighborIds.has(id);
        const fill = graphTypeColor(item.type);
        return `
          <g class="graph-node ${isSelected ? "selected" : isNeighbor ? "neighbor" : ""}" data-action="focus-graph-node" data-id="${escapeHtml(id)}" style="cursor:pointer">
            <circle cx="${item.x}" cy="${item.y}" r="${isSelected ? 20 : 16}" fill="${fill}" opacity="${isSelected ? 1 : isNeighbor ? 0.95 : 0.82}"></circle>
            <text x="${item.x}" y="${item.y + 4}" text-anchor="middle" fill="#fff" font-size="9" font-weight="700">${escapeHtml(String(item.type).slice(0, 3).toUpperCase())}</text>
            <text x="${item.x}" y="${item.y + 36}" text-anchor="middle" fill="#32485a" font-size="10">${escapeHtml((item.node.label || id).slice(0, 16))}</text>
          </g>
        `;
      }).join("")}
    </svg>
  `;

  const selectedNode = selected?.node || null;
  const selectedEdges = layout.edges.filter((edge) => {
    const src = edge.src || edge.source;
    const dst = edge.dst || edge.target;
    const currentId = selectedNode?.node_id || selectedNode?.id;
    return src === currentId || dst === currentId;
  });

  return `
    <div class="split-panel">
      <article class="surface-card">
        <div class="card-head">
          <h3>有向异构图</h3>
          <p>展示订单、任务、工序与资源之间的依赖和可行关系。点击节点可高亮相关边与邻居。</p>
        </div>
        <div class="graph-shell">${svg}</div>
      </article>
      <article class="surface-card">
        <div class="card-head">
          <h3>节点详情</h3>
          <p>用于解释问题复杂性与当前节点的上下游依赖。</p>
        </div>
        ${selectedNode ? renderKeyValueGrid([
          { label: "节点 ID", value: escapeHtml(selectedNode.node_id || selectedNode.id || "-") },
          { label: "节点类型", value: escapeHtml(selectedNode.node_type || selectedNode.type || "-") },
          { label: "显示标签", value: escapeHtml(selectedNode.label || selectedNode.name || "-") },
          { label: "关联边数", value: formatInt(selectedEdges.length) },
        ]) : renderEmptyState("未选中节点", "点击左侧图中的节点查看详情。")}
        ${selectedEdges.length ? renderSimpleTable(
          ["关系", "源", "目标"],
          selectedEdges.slice(0, 12).map((edge) => [
            escapeHtml(edge.edge_type || edge.type || "-"),
            escapeHtml(edge.src || edge.source || "-"),
            escapeHtml(edge.dst || edge.target || "-"),
          ]),
          { footer: selectedEdges.length > 12 ? `仅展示前 12 条关联边，共 ${selectedEdges.length} 条。` : "" },
        ) : ""}
      </article>
    </div>
  `;
}

function buildGraphViewModel() {
  const allNodes = asArray(app.graphNodes).map(normalizeGraphNode).filter((node) => node.id);
  const allEdges = asArray(app.graphEdges).map(normalizeGraphEdge).filter((edge) => edge.source && edge.target);
  if (!allNodes.length) return null;

  ensureGraphViewState(allNodes);
  const nodeMap = new Map(allNodes.map((node) => [node.id, node]));
  const visibleTypes = new Set(Object.entries(app.graphView.nodeTypes || {}).filter(([, enabled]) => enabled).map(([type]) => type));
  const visibleGroups = new Set(Object.entries(app.graphView.edgeGroups || {}).filter(([, enabled]) => enabled).map(([group]) => group));

  const eligibleNodes = allNodes.filter((node) => visibleTypes.has(node.type));
  const eligibleNodeIds = new Set(eligibleNodes.map((node) => node.id));
  const eligibleEdges = allEdges.filter((edge) => visibleGroups.has(edge.group) && eligibleNodeIds.has(edge.source) && eligibleNodeIds.has(edge.target));
  const clusterContext = buildOrderClusterContext(eligibleNodes, eligibleEdges);

  const adjacency = new Map(eligibleNodes.map((node) => [node.id, []]));
  eligibleEdges.forEach((edge) => {
    adjacency.get(edge.source)?.push({ id: edge.target, edge, direction: "out" });
    adjacency.get(edge.target)?.push({ id: edge.source, edge, direction: "in" });
  });

  const term = (app.graphView.search || "").trim().toLowerCase();
  let selectedId = app.selectedGraphNodeId;
  if (!eligibleNodeIds.has(selectedId)) selectedId = eligibleNodes[0]?.id || null;

  let focusIds = new Set(eligibleNodeIds);
  const matchedIds = eligibleNodes.filter((node) => graphNodeMatchesSearch(node, term)).map((node) => node.id);
  const expandToWholeOrders = (ids, targetSet) => {
    asArray(ids).forEach((id) => {
      clusterContext.ordersFromNode(id).forEach((orderId) => {
        clusterContext.orderClusters.get(orderId)?.forEach((nodeId) => targetSet.add(nodeId));
      });
    });
  };
  if (term && matchedIds.length) {
    focusIds = new Set(matchedIds);
    matchedIds.forEach((id) => {
      (adjacency.get(id) || []).forEach((neighbor) => {
        focusIds.add(neighbor.id);
      });
    });
    expandToWholeOrders(matchedIds, focusIds);
    if (!focusIds.has(selectedId)) selectedId = matchedIds[0];
  }

  if (app.graphView.mode === "focus" && selectedId) {
    const localIds = new Set([selectedId]);
    (adjacency.get(selectedId) || []).forEach((neighbor) => {
      localIds.add(neighbor.id);
      if (neighbor.edge.group === "structure") {
        (adjacency.get(neighbor.id) || []).forEach((secondary) => {
          if (secondary.edge.group === "structure") localIds.add(secondary.id);
        });
      }
    });
    expandToWholeOrders([selectedId, ...matchedIds], localIds);
    focusIds.forEach((id) => localIds.add(id));
    focusIds = localIds;
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

  const layout = layoutGraph(visibleNodes, visibleEdges, selectedId);
  const previousPositions = app.graphView.positions || {};
  const positions = {};
  visibleNodes.forEach((node) => {
    const base = layout.placed.get(node.id);
    positions[node.id] = previousPositions[node.id] ? { ...previousPositions[node.id] } : { x: base?.x || 0, y: base?.y || 0 };
  });
  app.graphView.positions = positions;
  app.selectedGraphNodeId = selectedId;

  const selectedNode = nodeMap.get(selectedId) || visibleNodes[0] || null;
  const visibleAdjacency = new Map(visibleNodes.map((node) => [node.id, []]));
  visibleEdges.forEach((edge) => {
    visibleAdjacency.get(edge.source)?.push(edge.target);
    visibleAdjacency.get(edge.target)?.push(edge.source);
  });
  const relatedFocusIds = new Set(selectedId ? [selectedId] : []);
  (visibleAdjacency.get(selectedId) || []).forEach((neighborId) => relatedFocusIds.add(neighborId));
  clusterContext.ordersFromNode(selectedId).forEach((orderId) => {
    clusterContext.orderClusters.get(orderId)?.forEach((nodeId) => {
      if (visibleAdjacency.has(nodeId)) relatedFocusIds.add(nodeId);
    });
  });
  const neighborIds = new Set(selectedId ? [selectedId] : []);
  (visibleAdjacency.get(selectedId) || []).forEach((neighborId) => neighborIds.add(neighborId));
  const relatedEdges = visibleEdges.filter((edge) => relatedFocusIds.has(edge.source) || relatedFocusIds.has(edge.target));
  const relatedNodeIds = new Set();
  relatedEdges.forEach((edge) => {
    relatedNodeIds.add(edge.source);
    relatedNodeIds.add(edge.target);
  });
  const relatedNodes = visibleNodes.filter((node) => relatedNodeIds.has(node.id) && node.id !== selectedId);
  const relatedByType = {};
  relatedNodes.forEach((node) => {
    if (!relatedByType[node.type]) relatedByType[node.type] = [];
    relatedByType[node.type].push(node);
  });

  const relatedNodeCountByType = {};
  relatedNodes.forEach((node) => {
    relatedNodeCountByType[node.type] = (relatedNodeCountByType[node.type] || 0) + 1;
  });

  const selectedStats = {
    directNeighborCount: Math.max(0, neighborIds.size - (selectedId ? 1 : 0)),
    relatedNodeCount: relatedNodes.length,
    relatedEdgeCount: relatedEdges.length,
    orderCount: new Set([
      ...(clusterContext.ordersFromNode(selectedId) || []),
      ...relatedNodes.filter((node) => node.type === "order").map((node) => node.id),
      ...(selectedNode?.type === "order" ? [selectedNode.id] : []),
    ]).size,
    taskCount: relatedNodeCountByType.task || 0,
    operationCount: relatedNodeCountByType.operation || 0,
    machineCount: relatedNodeCountByType.machine || 0,
    toolingCount: relatedNodeCountByType.tooling || 0,
    personnelCount: relatedNodeCountByType.personnel || 0,
  };

  const typeCounts = {};
  visibleNodes.forEach((node) => {
    typeCounts[node.type] = (typeCounts[node.type] || 0) + 1;
  });
  const edgeGroupCounts = {};
  visibleEdges.forEach((edge) => {
    edgeGroupCounts[edge.group] = (edgeGroupCounts[edge.group] || 0) + 1;
  });

  return {
    selectedId,
    selectedNode,
    nodeMap,
    visibleNodes,
    visibleEdges,
    relatedEdges,
    relatedByType,
    neighborIds,
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
  };
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
  container.querySelectorAll("[data-graph-link]").forEach((lineEl) => {
    const source = positions[lineEl.dataset.source];
    const target = positions[lineEl.dataset.target];
    if (!source || !target) return;
    lineEl.setAttribute("x1", String(source.x + 28));
    lineEl.setAttribute("y1", String(source.y));
    lineEl.setAttribute("x2", String(target.x - 28));
    lineEl.setAttribute("y2", String(target.y));
  });
}

function fitGraphViewport(bounds) {
  if (!bounds) return;
  const width = Math.max(120, (bounds.maxX - bounds.minX) + 120);
  const height = Math.max(120, (bounds.maxY - bounds.minY) + 120);
  const canvasWidth = bounds.canvasWidth || 1220;
  const canvasHeight = bounds.canvasHeight || 640;
  const zoom = Math.max(0.62, Math.min(1.6, Math.min((canvasWidth - 120) / width, (canvasHeight - 120) / height)));
  app.graphView.zoom = zoom;
  app.graphView.panX = 60 - bounds.minX * zoom;
  app.graphView.panY = 80 - bounds.minY * zoom;
}

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
  const svg = `
    <svg viewBox="0 0 ${graph.layout.width} ${graph.layout.height}" class="graph-svg interactive" data-graph-canvas role="img" aria-label="Graph Interactive View">
      <defs>
        <pattern id="graph-grid-pattern" width="28" height="28" patternUnits="userSpaceOnUse">
          <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(163,178,193,0.22)" stroke-width="1"></path>
        </pattern>
        <marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8ca0b5"></path>
        </marker>
      </defs>
      <rect x="0" y="0" width="${graph.layout.width}" height="${graph.layout.height}" rx="26" fill="url(#graph-grid-pattern)" class="graph-canvas-bg"></rect>
      <g data-graph-viewport transform="${graphViewportTransform()}">
        <g data-graph-links>
          ${graph.visibleEdges.map((edge) => {
            const source = graph.positions[edge.source];
            const target = graph.positions[edge.target];
            const highlighted = graph.neighborIds.has(edge.source) && graph.neighborIds.has(edge.target);
            return `
              <line
                class="graph-link ${highlighted ? "highlighted" : ""} graph-link-${escapeHtml(edge.group)}"
                data-graph-link
                data-source="${escapeHtml(edge.source)}"
                data-target="${escapeHtml(edge.target)}"
                x1="${source.x + 28}" y1="${source.y}"
                x2="${target.x - 28}" y2="${target.y}"
                marker-end="url(#graph-arrow)"
              ></line>
            `;
          }).join("")}
        </g>
        <g data-graph-nodes>
          ${graph.visibleNodes.map((node) => {
            const pos = graph.positions[node.id];
            const isSelected = node.id === selectedId;
            const isNeighbor = graph.neighborIds.has(node.id) && !isSelected;
            return `
              <g
                class="graph-node ${isSelected ? "selected" : isNeighbor ? "neighbor" : ""}"
                data-action="focus-graph-node"
                data-id="${escapeHtml(node.id)}"
                data-graph-node="${escapeHtml(node.id)}"
                data-node-label="${escapeHtml(node.label || node.id)}"
                data-node-type-label="${escapeHtml(graphTypeLabel(node.type))}"
                data-node-degree="${formatInt(nodeDegree.get(node.id) || 0)}"
                transform="translate(${pos.x} ${pos.y})"
                style="cursor:pointer"
              >
                <title>${escapeHtml(`${graphTypeLabel(node.type)}\n${node.label || node.id}\nID: ${node.id}\n关联关系: ${formatInt(nodeDegree.get(node.id) || 0)}`)}</title>
                <circle class="graph-node-hitbox" r="28" fill="transparent"></circle>
                <circle r="${isSelected ? 22 : 18}" fill="${graphTypeColor(node.type)}"></circle>
                <text x="0" y="4" text-anchor="middle" fill="#fff" font-size="9" font-weight="700">${escapeHtml(String(node.type).slice(0, 3).toUpperCase())}</text>
                <text class="graph-node-label" x="0" y="34" text-anchor="middle">${escapeHtml(String(node.label).slice(0, 18))}</text>
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

  return `
    <div class="surface-card graph-workbench">
      <div class="card-head">
        <h3>可交互有向异构图</h3>
        <p>保留所有关系类型，并通过图层筛选、焦点邻域、拖拽缩放和节点联动，让复杂结构可讲清、可分析、可展示。</p>
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
        <label class="graph-search">
          <span>搜索</span>
          <input type="search" value="${escapeHtml(app.graphView.search || "")}" data-graph-search placeholder="搜索订单、任务、工序或资源">
        </label>
        <label class="graph-search graph-order-limit">
          <span>鏈€澶氬睍绀鸿鍗?</span>
          <input type="number" min="1" max="20" step="1" value="${escapeHtml(String(app.graphView.maxOrders || 6))}" id="graph-max-orders">
        </label>
      </div>
        <label class="graph-search graph-order-limit">
          <span>鏈€澶氬睍绀鸿鍗?</span>
          <input type="number" min="1" max="20" step="1" value="${escapeHtml(String(app.graphView.maxOrders || 6))}" id="graph-max-orders">
        </label>
      <div class="graph-stage-meta">
        <span>展示节点 ${formatInt(graph.visibleNodes.length)} / ${formatInt(graph.totalNodeCount)}</span>
        <span>展示边 ${formatInt(graph.visibleEdges.length)} / ${formatInt(graph.totalEdgeCount)}</span>
        <span>${graph.orderScoped ? "大图已按完整订单子图保留全部关系" : "支持滚轮缩放、拖拽空白平移、拖拽节点微调布局"}</span>
      </div>
      <div class="graph-filter-grid">
        <section class="graph-filter-group">
          <div class="graph-filter-title">节点层级</div>
          <div class="filter-chip-row">${nodeTypeFilters}</div>
        </section>
        <section class="graph-filter-group">
          <div class="graph-filter-title">关系图层</div>
          <div class="filter-chip-row">${edgeGroupFilters}</div>
        </section>
      </div>
      <div class="split-panel graph-split">
        <article class="surface-card graph-stage-card">
          <div class="graph-shell">${svg}</div>
          <div class="legend graph-legend">
            ${GRAPH_NODE_ORDER.filter((type) => graph.typeCounts[type] || app.graphView.nodeTypes[type]).map((type) => `
              <span class="legend-item">
                <span class="legend-swatch" style="background:${graphTypeColor(type)}"></span>
                ${escapeHtml(graphTypeLabel(type))}
              </span>
            `).join("")}
            ${Object.keys(app.graphView.edgeGroups || {}).map((group) => `
              <span class="legend-item">
                <span class="legend-line legend-line-${escapeHtml(group)}"></span>
                ${escapeHtml(GRAPH_EDGE_GROUP_LABELS[group] || group)}
              </span>
            `).join("")}
          </div>
          ${(graph.culledNodeCount || graph.culledEdgeCount) ? `
            <div class="graph-footnote">
              当前为了保证大实例下交互流畅，已按优先级收敛部分节点/边，但所有关系类型都已纳入可视化逻辑。
              ${graph.orderScoped ? "当前优先按完整订单子图进行保留，保证已展示订单的关联关系不被截断。" : ""}
              省略节点 ${formatInt(graph.culledNodeCount)}，省略边 ${formatInt(graph.culledEdgeCount)}。
            </div>
          ` : ""}
        </article>
        <article class="surface-card graph-detail-card">
          <div class="card-head">
            <h3>节点详情与关系解释</h3>
            <p>从当前节点切入解释复杂性，帮助业务理解关键路径、资源耦合和装配结构。</p>
          </div>
          <div class="graph-hover-preview" id="graph-hover-preview">
            ${selectedNode ? `当前选中：${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.label || selectedNode.id)} / 关联关系 ${formatInt(nodeDegree.get(selectedNode.id) || 0)}` : "悬浮节点可快速预览其类型、标签和关系数，点击后会在右侧锁定完整详情。"}
          </div>
          ${selectedNode ? renderKeyValueGrid([
            { label: "节点 ID", value: escapeHtml(selectedNode.id || "-") },
            { label: "节点类型", value: escapeHtml(graphTypeLabel(selectedNode.type)) },
            { label: "显示标签", value: escapeHtml(selectedNode.label || "-") },
            { label: "关联边数", value: formatInt(graph.relatedEdges.length) },
          ]) : renderEmptyState("未选中节点", "点击图中的节点查看详细关系。")}
          ${selectedNode ? `
            <div class="graph-selected-summary" data-graph-selected-summary>
              <div class="graph-selected-main">
                <span class="graph-selected-badge" style="background:${graphTypeColor(selectedNode.type)}"></span>
                <div>
                  <div class="graph-selected-title">${escapeHtml(selectedNode.label || selectedNode.id)}</div>
                  <div class="graph-selected-meta">${escapeHtml(graphTypeLabel(selectedNode.type))} · ${escapeHtml(selectedNode.entity_id || selectedNode.id || "-")}</div>
                </div>
              </div>
              <div class="graph-selected-stats">
                <span>鍏宠仈鍏崇郴 ${formatInt(nodeDegree.get(selectedNode.id) || 0)}</span>
                <span>閭诲眳鑺傜偣 ${formatInt(graph.neighborIds.size ? graph.neighborIds.size - 1 : 0)}</span>
                <span>鍙杈?${formatInt(graph.relatedEdges.length)}</span>
              </div>
            </div>
          ` : ""}
          ${Object.keys(graph.relatedByType).length ? `
            <div class="graph-neighbor-groups">
              ${Object.entries(graph.relatedByType).map(([type, items]) => `
                <section class="graph-neighbor-group">
                  <div class="graph-filter-title">${escapeHtml(graphTypeLabel(type))}</div>
                  <div class="neighbor-pill-row">
                    ${items.slice(0, 10).map((item) => `
                      <button class="neighbor-pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(item.id)}">${escapeHtml(String(item.label).slice(0, 18))}</button>
                    `).join("")}
                  </div>
                </section>
              `).join("")}
            </div>
          ` : ""}
          ${graph.relatedEdges.length ? renderSimpleTable(
            ["关系", "起点", "终点"],
            graph.relatedEdges.slice(0, 14).map((edge) => [
              escapeHtml(humanizeCodeLabel(edge.edgeType)),
              escapeHtml(edge.source),
              escapeHtml(edge.target),
            ]),
            { footer: graph.relatedEdges.length > 14 ? `当前节点共关联 ${graph.relatedEdges.length} 条边，这里展示前 14 条。` : "" },
          ) : ""}
        </article>
      </div>
    </div>
  `;
}

function renderInteractiveGraph() {
  const graph = buildGraphViewModel();
  if (!graph || !graph.visibleNodes.length) {
    return renderEmptyState("暂无图谱节点", "请先构建图谱，或确认当前实例已经正确加载。");
  }

  const nodeDegree = new Map(graph.visibleNodes.map((node) => [node.id, 0]));
  graph.visibleEdges.forEach((edge) => {
    nodeDegree.set(edge.source, (nodeDegree.get(edge.source) || 0) + 1);
    nodeDegree.set(edge.target, (nodeDegree.get(edge.target) || 0) + 1);
  });

  const selectedNode = graph.selectedNode;
  const selectedId = graph.selectedId;
  const svg = `
    <svg viewBox="0 0 ${graph.layout.width} ${graph.layout.height}" class="graph-svg interactive" data-graph-canvas role="img" aria-label="Graph Interactive View">
      <defs>
        <pattern id="graph-grid-pattern" width="28" height="28" patternUnits="userSpaceOnUse">
          <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(163,178,193,0.22)" stroke-width="1"></path>
        </pattern>
        <marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8ca0b5"></path>
        </marker>
      </defs>
      <rect x="0" y="0" width="${graph.layout.width}" height="${graph.layout.height}" rx="26" fill="url(#graph-grid-pattern)" class="graph-canvas-bg"></rect>
      <g data-graph-viewport transform="${graphViewportTransform()}">
        <g data-graph-links>
          ${graph.visibleEdges.map((edge) => {
            const source = graph.positions[edge.source];
            const target = graph.positions[edge.target];
            const highlighted = graph.neighborIds.has(edge.source) && graph.neighborIds.has(edge.target);
            return `
              <line
                class="graph-link ${highlighted ? "highlighted" : ""} graph-link-${escapeHtml(edge.group)}"
                data-graph-link
                data-source="${escapeHtml(edge.source)}"
                data-target="${escapeHtml(edge.target)}"
                x1="${source.x + 28}" y1="${source.y}"
                x2="${target.x - 28}" y2="${target.y}"
                marker-end="url(#graph-arrow)"
              ></line>
            `;
          }).join("")}
        </g>
        <g data-graph-nodes>
          ${graph.visibleNodes.map((node) => {
            const pos = graph.positions[node.id];
            const isSelected = node.id === selectedId;
            const isNeighbor = graph.neighborIds.has(node.id) && !isSelected;
            return `
              <g
                class="graph-node ${isSelected ? "selected" : isNeighbor ? "neighbor" : ""}"
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
                <circle class="graph-node-hitbox" r="28" fill="transparent"></circle>
                <circle r="${isSelected ? 22 : 18}" fill="${graphTypeColor(node.type)}"></circle>
                <text x="0" y="4" text-anchor="middle" fill="#fff" font-size="9" font-weight="700">${escapeHtml(String(node.type).slice(0, 3).toUpperCase())}</text>
                <text class="graph-node-label" x="0" y="34" text-anchor="middle">${escapeHtml(String(node.label).slice(0, 18))}</text>
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

  return `
    <div class="surface-card graph-workbench">
      <div class="card-head">
        <h3>可交互有向异构图</h3>
        <p>保留订单、任务、工序、机器、工装、人员等全部关系类型，并通过图层筛选、焦点邻域、拖拽缩放和节点联动，让复杂结构可解释、可钻取、可展示。</p>
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
        <label class="graph-search">
          <span>搜索</span>
          <input type="search" value="${escapeHtml(app.graphView.search || "")}" data-graph-search placeholder="搜索订单、任务、工序或资源">
        </label>
      </div>
      <div class="graph-stage-meta">
        <span>展示节点 ${formatInt(graph.visibleNodes.length)} / ${formatInt(graph.totalNodeCount)}</span>
        <span>展示边 ${formatInt(graph.visibleEdges.length)} / ${formatInt(graph.totalEdgeCount)}</span>
        <span>${graph.orderScoped ? "大图已按完整订单子图保留全部关联" : "支持滚轮缩放、拖拽平移和节点微调布局"}</span>
      </div>
      <div class="graph-filter-grid">
        <section class="graph-filter-group">
          <div class="graph-filter-title">节点层级</div>
          <div class="filter-chip-row">${nodeTypeFilters}</div>
        </section>
        <section class="graph-filter-group">
          <div class="graph-filter-title">关系图层</div>
          <div class="filter-chip-row">${edgeGroupFilters}</div>
        </section>
      </div>
      <div class="split-panel graph-split">
        <article class="surface-card graph-stage-card">
          <div class="graph-shell">${svg}</div>
          <div class="legend graph-legend">
            ${GRAPH_NODE_ORDER.filter((type) => graph.typeCounts[type] || app.graphView.nodeTypes[type]).map((type) => `
              <span class="legend-item">
                <span class="legend-swatch" style="background:${graphTypeColor(type)}"></span>
                ${escapeHtml(graphTypeLabel(type))}
              </span>
            `).join("")}
            ${Object.keys(app.graphView.edgeGroups || {}).map((group) => `
              <span class="legend-item">
                <span class="legend-line legend-line-${escapeHtml(group)}"></span>
                ${escapeHtml(GRAPH_EDGE_GROUP_LABELS[group] || group)}
              </span>
            `).join("")}
          </div>
          ${(graph.culledNodeCount || graph.culledEdgeCount) ? `
            <div class="graph-footnote">
              当前为保证大实例下交互流畅，已按优先级收敛部分节点和边，但所有关系类型都纳入了可视化逻辑。
              ${graph.orderScoped ? "当前优先按完整订单子图保留，已展示订单不会被拆断。" : ""}
              省略节点 ${formatInt(graph.culledNodeCount)}，省略边 ${formatInt(graph.culledEdgeCount)}。
            </div>
          ` : ""}
        </article>
        <article class="surface-card graph-detail-card">
          <div class="card-head">
            <h3>节点详情与关系解释</h3>
            <p>点击左侧节点后，右侧立即展示完整属性、关联对象与关系证据，便于业务快速理解复杂性。</p>
          </div>
          <div class="graph-hover-preview" id="graph-hover-preview">
            ${selectedNode ? `当前选中：${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.label || selectedNode.id)} / 关联关系 ${formatInt(nodeDegree.get(selectedNode.id) || 0)}` : "悬浮节点可快速预览其类型、标签和关系数，点击后会在右侧锁定完整详情。"}
          </div>
          ${selectedNode ? renderKeyValueGrid(graphNodeDetailRows(selectedNode, graph.relatedEdges)) : renderEmptyState("未选中节点", "点击图中的节点查看详细关系。")}
          ${Object.keys(graph.relatedByType).length ? `
            <div class="graph-neighbor-groups">
              ${Object.entries(graph.relatedByType).map(([type, items]) => `
                <section class="graph-neighbor-group">
                  <div class="graph-filter-title">${escapeHtml(graphTypeLabel(type))}</div>
                  <div class="neighbor-pill-row">
                    ${items.slice(0, 10).map((item) => `
                      <button class="neighbor-pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(item.id)}">${escapeHtml(String(item.label).slice(0, 18))}</button>
                    `).join("")}
                  </div>
                </section>
              `).join("")}
            </div>
          ` : ""}
          ${graph.relatedEdges.length ? renderSimpleTable(
            ["关系", "起点", "终点"],
            graph.relatedEdges.slice(0, 14).map((edge) => [
              escapeHtml(humanizeCodeLabel(edge.edgeType)),
              escapeHtml(edge.source),
              escapeHtml(edge.target),
            ]),
            { footer: graph.relatedEdges.length > 14 ? `当前节点共关联 ${graph.relatedEdges.length} 条边，这里展示前 14 条。` : "" },
          ) : ""}
        </article>
      </div>
    </div>
  `;
}

function renderInteractiveGraph() {
  const graph = buildGraphViewModel();
  if (!graph || !graph.visibleNodes.length) {
    return renderEmptyState("\u6682\u65e0\u56fe\u8c31\u8282\u70b9", "\u8bf7\u5148\u6784\u5efa\u56fe\u8c31\uff0c\u6216\u786e\u8ba4\u5f53\u524d\u5b9e\u4f8b\u5df2\u7ecf\u6b63\u786e\u52a0\u8f7d\u3002");
  }

  const nodeDegree = new Map(graph.visibleNodes.map((node) => [node.id, 0]));
  graph.visibleEdges.forEach((edge) => {
    nodeDegree.set(edge.source, (nodeDegree.get(edge.source) || 0) + 1);
    nodeDegree.set(edge.target, (nodeDegree.get(edge.target) || 0) + 1);
  });

  const selectedNode = graph.selectedNode;
  const selectedId = graph.selectedId;
  const svg = `
    <svg viewBox="0 0 ${graph.layout.width} ${graph.layout.height}" class="graph-svg interactive" data-graph-canvas role="img" aria-label="\u53ef\u4ea4\u4e92\u6709\u5411\u5f02\u6784\u56fe">
      <defs>
        <pattern id="graph-grid-pattern" width="28" height="28" patternUnits="userSpaceOnUse">
          <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(163,178,193,0.22)" stroke-width="1"></path>
        </pattern>
        <marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8ca0b5"></path>
        </marker>
      </defs>
      <rect x="0" y="0" width="${graph.layout.width}" height="${graph.layout.height}" rx="26" fill="url(#graph-grid-pattern)" class="graph-canvas-bg"></rect>
      <g data-graph-viewport transform="${graphViewportTransform()}">
        <g data-graph-links>
          ${graph.visibleEdges.map((edge) => {
            const source = graph.positions[edge.source];
            const target = graph.positions[edge.target];
            const highlighted = graph.neighborIds.has(edge.source) || graph.neighborIds.has(edge.target);
            return `
              <line
                class="graph-link ${highlighted ? "highlighted" : ""} graph-link-${escapeHtml(edge.group)}"
                data-graph-link
                data-source="${escapeHtml(edge.source)}"
                data-target="${escapeHtml(edge.target)}"
                x1="${source.x + 28}" y1="${source.y}"
                x2="${target.x - 28}" y2="${target.y}"
                marker-end="url(#graph-arrow)"
              ></line>
            `;
          }).join("")}
        </g>
        <g data-graph-nodes>
          ${graph.visibleNodes.map((node) => {
            const pos = graph.positions[node.id];
            const isSelected = node.id === selectedId;
            const isNeighbor = graph.neighborIds.has(node.id) && !isSelected;
            return `
              <g
                class="graph-node ${isSelected ? "selected" : isNeighbor ? "neighbor" : ""}"
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
                <title>${escapeHtml(`${graphTypeLabel(node.type)}\n${node.label || node.id}\nID: ${node.id}\n\u5173\u8054\u5173\u7cfb: ${formatInt(nodeDegree.get(node.id) || 0)}`)}</title>
                <circle class="graph-node-hitbox" r="28" fill="transparent"></circle>
                <circle r="${isSelected ? 22 : 18}" fill="${graphTypeColor(node.type)}"></circle>
                <text x="0" y="4" text-anchor="middle" fill="#fff" font-size="9" font-weight="700">${escapeHtml(String(node.type).slice(0, 3).toUpperCase())}</text>
                <text class="graph-node-label" x="0" y="34" text-anchor="middle">${escapeHtml(String(node.label).slice(0, 18))}</text>
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

  const selectedSummary = selectedNode ? `
    <div class="graph-selected-summary" data-graph-selected-summary>
      <div class="graph-selected-main">
        <span class="graph-selected-badge" style="background:${graphTypeColor(selectedNode.type)}"></span>
        <div>
          <div class="graph-selected-title">${escapeHtml(selectedNode.label || selectedNode.id)}</div>
          <div class="graph-selected-meta">${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.entity_id || selectedNode.id || "-")}</div>
        </div>
      </div>
      <div class="graph-selected-stats">
        <span>\u76f4\u63a5\u90bb\u5c45 ${formatInt(graph.selectedStats.directNeighborCount)}</span>
        <span>\u8ba2\u5355 ${formatInt(graph.selectedStats.orderCount)}</span>
        <span>\u4efb\u52a1 ${formatInt(graph.selectedStats.taskCount)}</span>
        <span>\u5de5\u5e8f ${formatInt(graph.selectedStats.operationCount)}</span>
        <span>\u673a\u5668 ${formatInt(graph.selectedStats.machineCount)}</span>
        <span>\u5de5\u88c5 ${formatInt(graph.selectedStats.toolingCount)}</span>
        <span>\u4eba\u5458 ${formatInt(graph.selectedStats.personnelCount)}</span>
        <span>\u5173\u8054\u8fb9 ${formatInt(graph.selectedStats.relatedEdgeCount)}</span>
      </div>
    </div>
  ` : renderEmptyState("\u672a\u9009\u4e2d\u8282\u70b9", "\u70b9\u51fb\u5de6\u4fa7\u56fe\u4e2d\u7684\u8282\u70b9\u540e\uff0c\u8fd9\u91cc\u4f1a\u663e\u793a\u8be5\u8282\u70b9\u76f8\u5173\u7684\u5b8c\u6574\u7edf\u8ba1\u4e0e\u5173\u7cfb\u8bf4\u660e\u3002");

  return `
    <div class="surface-card graph-workbench">
      <div class="card-head">
        <h3>\u53ef\u4ea4\u4e92\u6709\u5411\u5f02\u6784\u56fe</h3>
        <p>\u4fdd\u7559\u8ba2\u5355\u3001\u4efb\u52a1\u3001\u5de5\u5e8f\u3001\u673a\u5668\u3001\u5de5\u88c5\u3001\u4eba\u5458\u7b49\u5168\u90e8\u5173\u7cfb\u7c7b\u578b\uff0c\u5e76\u901a\u8fc7\u56fe\u5c42\u7b5b\u9009\u3001\u7126\u70b9\u90bb\u57df\u3001\u62d6\u62fd\u7f29\u653e\u548c\u8282\u70b9\u8054\u52a8\uff0c\u8ba9\u590d\u6742\u7ed3\u6784\u53ef\u89e3\u91ca\u3001\u53ef\u94bb\u53d6\u3001\u53ef\u5c55\u793a\u3002</p>
      </div>
      <div class="graph-toolbar">
        <div class="inline-actions">
          <button class="btn ${app.graphView.mode === "focus" ? "btn-primary" : "btn-ghost"}" type="button" data-action="set-graph-mode" data-mode="focus">\u7126\u70b9\u90bb\u57df</button>
          <button class="btn ${app.graphView.mode === "all" ? "btn-primary" : "btn-ghost"}" type="button" data-action="set-graph-mode" data-mode="all">\u5168\u5173\u7cfb\u89c6\u56fe</button>
          <button class="btn btn-ghost" type="button" data-action="fit-graph-view">\u9002\u914d\u89c6\u56fe</button>
          <button class="btn btn-ghost" type="button" data-action="reset-graph-view">\u91cd\u7f6e\u89c6\u56fe</button>
          <button class="btn btn-ghost" type="button" data-action="zoom-graph-out">-</button>
          <button class="btn btn-ghost" type="button" data-action="toggle-graph-fullscreen">\u5168\u5c4f\u67e5\u770b</button>
          <span class="graph-zoom-pill" data-graph-zoom>${Math.round(app.graphView.zoom * 100)}%</span>
          <button class="btn btn-ghost" type="button" data-action="zoom-graph-in">+</button>
        </div>
        <label class="graph-search">
          <span>\u641c\u7d22\u8282\u70b9</span>
          <input type="search" value="${escapeHtml(app.graphView.search || "")}" data-graph-search placeholder="\u641c\u7d22\u8ba2\u5355\u3001\u4efb\u52a1\u3001\u5de5\u5e8f\u6216\u8d44\u6e90">
        </label>
        <label class="graph-search graph-order-limit">
          <span>\u6700\u591a\u5c55\u793a\u8ba2\u5355\u6570</span>
          <input type="number" min="1" max="20" step="1" value="${escapeHtml(String(app.graphView.maxOrders || 6))}" id="graph-max-orders">
        </label>
      </div>
      <div class="graph-stage-meta">
        <span>\u5c55\u793a\u8282\u70b9 ${formatInt(graph.visibleNodes.length)} / ${formatInt(graph.totalNodeCount)}</span>
        <span>\u5c55\u793a\u8fb9 ${formatInt(graph.visibleEdges.length)} / ${formatInt(graph.totalEdgeCount)}</span>
        <span>${graph.orderScoped ? `\u5927\u56fe\u5df2\u6309\u5b8c\u6574\u8ba2\u5355\u5b50\u56fe\u4fdd\u7559\uff0c\u5f53\u524d\u6700\u591a\u5c55\u793a ${formatInt(graph.maxOrders)} \u4e2a\u8ba2\u5355` : "\u5f53\u524d\u89c6\u56fe\u652f\u6301\u6eda\u8f6e\u7f29\u653e\u3001\u62d6\u62fd\u5e73\u79fb\u548c\u8282\u70b9\u5fae\u8c03\u5e03\u5c40"}</span>
      </div>
      <div class="graph-filter-grid">
        <section class="graph-filter-group">
          <div class="graph-filter-title">\u8282\u70b9\u5c42\u7ea7</div>
          <div class="filter-chip-row">${nodeTypeFilters}</div>
        </section>
        <section class="graph-filter-group">
          <div class="graph-filter-title">\u5173\u7cfb\u5c42\u7ea7</div>
          <div class="filter-chip-row">${edgeGroupFilters}</div>
        </section>
      </div>
      <div class="split-panel graph-split">
        <article class="surface-card graph-stage-card">
          <div class="graph-hover-preview" data-graph-hover-preview>
            ${selectedNode ? `\u5f53\u524d\u9009\u4e2d\uff1a${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.label || selectedNode.id)} / \u5173\u8054\u5173\u7cfb ${formatInt(nodeDegree.get(selectedNode.id) || 0)}` : "\u5c06\u9f20\u6807\u60ac\u6d6e\u5230\u8282\u70b9\u4e0a\uff0c\u53ef\u5feb\u901f\u9884\u89c8\u8be5\u8282\u70b9\u540d\u79f0\u3001\u7c7b\u578b\u4e0e\u5173\u8054\u5173\u7cfb\u5f3a\u5ea6\u3002"}
          </div>
          <div class="graph-shell">${svg}</div>
        </article>
        <article class="surface-card graph-detail-card">
          <div class="card-head">
            <h3>\u8282\u70b9\u8be6\u60c5\u4e0e\u5173\u7cfb\u89e3\u91ca</h3>
            <p>\u70b9\u51fb\u5de6\u4fa7\u4efb\u610f\u8282\u70b9\u540e\uff0c\u53f3\u4fa7\u4f1a\u8054\u52a8\u5237\u65b0\u8be5\u8282\u70b9\u7684\u5173\u8054\u7edf\u8ba1\u3001\u5c5e\u6027\u5b57\u6bb5\u3001\u5173\u7cfb\u6e05\u5355\u4e0e\u5c40\u90e8\u90bb\u57df\u5206\u5e03\u3002</p>
          </div>
          ${selectedSummary}
          ${selectedNode ? renderKeyValueGrid(graphNodeDetailRows(selectedNode, graph.relatedEdges)) : renderEmptyState("\u672a\u9009\u4e2d\u8282\u70b9", "\u8bf7\u5148\u5728\u5de6\u4fa7\u56fe\u4e2d\u70b9\u51fb\u4e00\u4e2a\u8282\u70b9\u3002")}
          ${Object.keys(graph.relatedByType).length ? `
            <div class="graph-neighbor-groups">
              ${Object.entries(graph.relatedByType).map(([type, items]) => `
                <section class="graph-neighbor-group">
                  <header>
                    <strong>${escapeHtml(graphTypeLabel(type))}</strong>
                    <span>${formatInt(items.length)} \u4e2a</span>
                  </header>
                  <div class="graph-neighbor-pills">
                    ${items.slice(0, 10).map((item) => `<button class="pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(item.id)}">${escapeHtml(item.label || item.id)}</button>`).join("")}
                  </div>
                </section>
              `).join("")}
            </div>
          ` : ""}
          ${graph.relatedEdges.length ? renderSimpleTable(
            ["\u5173\u7cfb", "\u6765\u6e90", "\u76ee\u6807"],
            graph.relatedEdges.slice(0, 12).map((edge) => [
              escapeHtml(humanizeCodeLabel(edge.edgeType || edge.group || "-")),
              escapeHtml(graph.nodeMap.get(edge.source)?.label || edge.source),
              escapeHtml(graph.nodeMap.get(edge.target)?.label || edge.target),
            ]),
            { footer: graph.relatedEdges.length > 12 ? `\u5f53\u524d\u53ea\u5c55\u793a\u524d 12 \u6761\u5173\u8054\u8fb9\uff0c\u5171 ${graph.relatedEdges.length} \u6761\u3002` : "" },
          ) : renderEmptyState("\u6682\u65e0\u5173\u8054\u8fb9", "\u5f53\u524d\u8282\u70b9\u5728\u53ef\u89c1\u56fe\u4e2d\u6ca1\u6709\u5173\u8054\u8fb9\u3002")}
        </article>
      </div>
    </div>
  `;
}

function renderWorkflowStep3() {
  const focus = app.workflowFocus || "graph";
  const simMetrics = app.simResult?.metrics || {};
  const graphPanel = `
    <div class="workflow-stage-stack">
      <article class="surface-card">
      <div class="card-head"><h3>图谱构建</h3></div>
      ${renderGraphBuildStatus()}
      ${app.graphMeta ? renderKeyValueGrid([
        { label: "节点", value: formatInt(app.graphMeta.total_nodes) },
        { label: "边", value: formatInt(app.graphMeta.total_edges) },
        { label: "创建时间", value: formatDateTime(app.graphMeta.created_at) },
        { label: "节点类型", value: formatInt(Object.keys(app.graphMeta.node_type_counts || {}).length) },
      ], "context-grid cols-4") : renderEmptyState("尚未构建图谱", "点击下方按钮即可根据当前实例构建图谱。")}
      <div class="form-actions form-actions--gap">
        <button class="btn btn-primary" type="button" data-action="build-graph">构建图谱</button>
      </div>
      </article>
      ${app.graphMeta ? renderLegacyCytoscapeGraph() : ""}
    </div>
  `;

  const simulationPanel = `
    <article class="surface-card">
      <div class="card-head"><h3>规则仿真</h3><p>先用规则基线验证数据、班次、停机和初始在制状态是否合理，再决定是否进入优化。</p></div>
      <div class="field-inline">
        <span>规则</span>
        <select id="workflow-sim-rule">
          ${CONFIG.HEURISTIC_RULES.map((rule) => `<option value="${rule}" ${rule === app.simRule ? "selected" : ""}>${rule}</option>`).join("")}
        </select>
      </div>
      <div class="form-actions">
        <button class="btn btn-primary" type="button" data-action="run-simulate">运行仿真</button>
        <button class="btn btn-ghost" type="button" data-action="set-workflow-focus" data-focus="simulate">打开完整仿真页</button>
        <button class="btn btn-ghost" type="button" data-action="set-workflow-focus" data-focus="graph">返回图谱视图</button>
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
    ${renderTimeline(app.simResult.gantt, { title: `规则仿真甘特图 · ${app.simRule}`, canvasId: "gantt-sim" })}
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
    <div class="workflow-focus-tabs">
      <button class="focus-tab ${focus === "graph" ? "active" : ""}" type="button" data-action="set-workflow-focus" data-focus="graph">图谱视图</button>
      <button class="focus-tab ${focus === "simulate" ? "active" : ""}" type="button" data-action="set-workflow-focus" data-focus="simulate">完整仿真页</button>
    </div>
    ${focus === "simulate" ? `
      <div class="workflow-stage-stack">
        ${simulationPanel}
        ${simulationDetail || ""}
      </div>
    ` : `
      <div class="workflow-stage-stack">
        ${graphPanel}
        ${simulationPanel}
      </div>
    `}
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
  if (app.workflowStep === 3 && (app.workflowFocus || "graph") === "graph" && app.graphMeta) {
    requestAnimationFrame(() => mountLegacyCytoscapeGraph());
  }
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
  return `
    <article class="surface-card validation-panel ${tone}">
      <div class="card-head">
        <div><h3>数据强校验</h3><p>覆盖数据完整性、关联关系与约束条件；错误级问题会导致仿真/优化静默失败。</p></div>
        ${statusChip(label, tone === "danger" ? "danger" : tone)}
      </div>
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
    </article>
  `;
}

function renderCandidateCards(candidates) {
  return `
    <div class="candidate-layout">
      ${candidates.map((item) => `
        <article class="surface-card ${app.reviewDetailId === item.id ? "is-selected" : ""}">
          <div class="candidate-name">
            <div>
              <strong>${escapeHtml(item.name)}</strong>
              <small>${escapeHtml(item.source)} · ${escapeHtml(item.evaluationMode)}</small>
              ${renderPrimaryObjectiveBadges()}
            </div>
            <input type="checkbox" data-action="toggle-candidate" data-id="${escapeHtml(item.id)}" ${app.reviewSelection.includes(item.id) ? "checked" : ""}>
          </div>
          <div class="form-actions">
            <button class="btn btn-ghost" type="button" data-action="focus-candidate" data-id="${escapeHtml(item.id)}">查看详情</button>
            <button class="btn btn-primary" type="button" data-action="send-candidate-to-ai" data-id="${escapeHtml(item.id)}">送入 AI 评审</button>
            <button class="btn btn-ghost" type="button" data-action="export-selected-solution" data-id="${escapeHtml(item.id)}">导出</button>
          </div>
        </article>
      `).join("")}
    </div>
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

function renderReviewCandidateComparison() {
  const candidates = getReviewCandidates();
  if (!candidates.length) return "";
  const focusedId = app.reviewDetailId;
  let preview;
  if (focusedId && candidates.some((item) => item.id === focusedId)) {
    const focused = candidates.find((item) => item.id === focusedId);
    preview = [focused, ...candidates.filter((item) => item.id !== focusedId)].slice(0, 5);
  } else {
    preview = candidates.slice(0, 5);
  }
  const primaryKeys = activePrimaryObjectiveKeys();
  const extraKeys = REVIEW_KPI_KEYS.filter((key) => !primaryKeys.includes(key));
  const headers = ["方案", "来源", ...primaryKeys.map((key) => getObjectiveLabel(key)), ...extraKeys.map((key) => getObjectiveLabel(key))];
  const rows = preview.map((item) => {
    const isFocused = app.reviewDetailId === item.id;
    return `<tr class="${isFocused ? "is-selected" : ""}">
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.source)}</td>
      ${primaryKeys.map((key) => `<td>${metricDisplay(item, key)}</td>`).join("")}
      ${extraKeys.map((key) => `<td>${metricDisplay(item, key)}</td>`).join("")}
    </tr>`;
  }).join("");
  return `
    <article class="surface-card">
      <div class="card-head">
        <h3>主目标 + 全量 KPI 对比</h3>
        <p>点击上方方案卡片的「查看详情」可置顶并高亮对应行；默认展示前 ${preview.length} 个方案，横向滚动可查看全部 KPI。</p>
      </div>
      ${renderPrimaryObjectiveBadges(primaryKeys)}
      <div class="table-shell">
        <table class="data-table">
          <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </article>
  `;
}

function renderReviewLibraryTab() {
  ensureReviewSelection();
  const candidates = getReviewCandidates();
  const selected = getSelectedReviewCandidates();
  const focused = getSelectedReviewCandidate();
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
        <div><span>已选方案</span><strong>${formatInt(selected.length)}</strong></div>
      </div>
    </article>
    <div class="two-column">
      <article class="surface-card">
        <div class="card-head"><h3>启发式参考方案</h3><p>将常用启发式规则与 Pareto 解一起带入比较和 AI 评审。</p></div>
        <div class="objective-grid">
          ${CONFIG.HEURISTIC_RULES.map((rule) => `
            <label class="objective-pill">
              <input type="checkbox" data-heuristic-rule="${escapeHtml(rule)}" ${app.heuristicSelection.includes(rule) ? "checked" : ""}>
              <strong>${escapeHtml(rule)}</strong>
              <span>${escapeHtml(HEURISTIC_RULE_BLURB[rule] || "用于快速加载规则参考方案")}</span>
            </label>
          `).join("")}
        </div>
        <div class="form-actions">
          <button class="btn btn-primary" type="button" data-action="load-heuristic-references">加载参考方案</button>
        </div>
      </article>
      <article class="surface-card">
        <div class="card-head"><h3>方案池概况</h3><p>将 Pareto 解、基线、启发式和精确冠军统一纳入评审。</p></div>
        ${renderPrimaryObjectiveBadges()}
        ${renderKeyValueGrid([
          { label: "总方案数", value: formatInt(candidates.length) },
          { label: "已选方案", value: formatInt(selected.length) },
          { label: "主目标", value: objectiveShortList(app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys) },
          { label: "最近优化", value: app.optimizeResult ? "已完成" : "未运行" },
        ])}
      </article>
    </div>
    ${candidates.length ? renderCandidateCards(candidates) : renderEmptyState(
      "暂无方案池",
      "先运行混合优化，或先加载启发式参考方案。",
      '<button class="btn btn-primary" type="button" data-nav-jump="optimize-launch">去启动优化</button>',
    )}
    ${candidates.length ? renderReviewCandidateComparison() : ""}
    ${focused ? renderTimeline(focused.schedule, { title: `方案详情甘特图 · ${focused.name}`, canvasId: `gantt-plan-${focused.id}` }) : ""}
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
      ${renderTimeline(app.exactReference.schedule, { title: "精确冠军参考甘特图", canvasId: "gantt-exact" })}
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

function renderReview() {
  const container = el("review-content");
  syncTabButtons("data-review-tab", app.reviewTab);
  if (app.reviewTab === "library") container.innerHTML = renderReviewLibraryTab();
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
      upstreamCount: 0,
      downstreamCount: 0,
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
    upstreamCount: upstreamNodeIds.size,
    downstreamCount: downstreamNodeIds.size,
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
    positions[node.id] = previousPositions[node.id] ? { ...previousPositions[node.id] } : { x: base?.x || 0, y: base?.y || 0 };
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

function renderInteractiveGraph() {
  const graph = buildGraphViewModel();
  if (!graph || !graph.visibleNodes.length) {
    return renderEmptyState("\u6682\u65e0\u56fe\u8c31\u8282\u70b9", "\u8bf7\u5148\u6784\u5efa\u56fe\u8c31\uff0c\u6216\u786e\u8ba4\u5f53\u524d\u5b9e\u4f8b\u5df2\u7ecf\u6b63\u786e\u52a0\u8f7d\u3002");
  }

  const nodeDegree = new Map(graph.visibleNodes.map((node) => [node.id, 0]));
  graph.visibleEdges.forEach((edge) => {
    nodeDegree.set(edge.source, (nodeDegree.get(edge.source) || 0) + 1);
    nodeDegree.set(edge.target, (nodeDegree.get(edge.target) || 0) + 1);
  });

  const selectedNode = graph.selectedNode;
  const selectedId = graph.selectedId;
  const svg = `
    <svg viewBox="0 0 ${graph.layout.width} ${graph.layout.height}" class="graph-svg interactive" data-graph-canvas role="img" aria-label="\u53ef\u4ea4\u4e92\u6709\u5411\u5f02\u6784\u56fe">
      <defs>
        <pattern id="graph-grid-pattern" width="28" height="28" patternUnits="userSpaceOnUse">
          <path d="M 28 0 L 0 0 0 28" fill="none" stroke="rgba(163,178,193,0.22)" stroke-width="1"></path>
        </pattern>
        <marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#8ca0b5"></path>
        </marker>
      </defs>
      <rect x="0" y="0" width="${graph.layout.width}" height="${graph.layout.height}" rx="26" fill="url(#graph-grid-pattern)" class="graph-canvas-bg"></rect>
      <g data-graph-viewport transform="${graphViewportTransform()}">
        <g data-graph-links>
          ${graph.visibleEdges.map((edge) => {
            const source = graph.positions[edge.source];
            const target = graph.positions[edge.target];
            const highlighted = graph.highlightedNodeIds.has(edge.source) && graph.highlightedNodeIds.has(edge.target);
            return `
              <line
                class="graph-link ${highlighted ? "highlighted" : ""} graph-link-${escapeHtml(edge.group)}"
                data-graph-link
                data-source="${escapeHtml(edge.source)}"
                data-target="${escapeHtml(edge.target)}"
                x1="${source.x + 28}" y1="${source.y}"
                x2="${target.x - 28}" y2="${target.y}"
                marker-end="url(#graph-arrow)"
              ></line>
            `;
          }).join("")}
        </g>
        <g data-graph-nodes>
          ${graph.visibleNodes.map((node) => {
            const pos = graph.positions[node.id];
            const isSelected = node.id === selectedId;
            const isNeighbor = graph.neighborIds.has(node.id) && !isSelected;
            const isScoped = graph.highlightedNodeIds.has(node.id);
            return `
              <g
                class="graph-node ${isSelected ? "selected" : isNeighbor ? "neighbor" : isScoped ? "scoped" : ""}"
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
                <title>${escapeHtml(`${graphTypeLabel(node.type)}\n${node.label || node.id}\nID: ${node.id}\n\u5173\u8054\u5173\u7cfb: ${formatInt(nodeDegree.get(node.id) || 0)}`)}</title>
                <circle class="graph-node-hitbox" r="28" fill="transparent"></circle>
                <circle r="${isSelected ? 22 : 18}" fill="${graphTypeColor(node.type)}"></circle>
                <text x="0" y="4" text-anchor="middle" fill="#fff" font-size="9" font-weight="700">${escapeHtml(String(node.type).slice(0, 3).toUpperCase())}</text>
                <text class="graph-node-label" x="0" y="34" text-anchor="middle">${escapeHtml(String(node.label).slice(0, 18))}</text>
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

  const stageNarrative = graph.orderScoped
    ? `\u5f53\u524d\u56fe\u4e2d\u5c55\u793a ${formatInt(graph.selectedStats.displayedOrderCount)} / ${formatInt(graph.selectedStats.orderCount || graph.visibleOrderCount)} \u4e2a\u76f8\u5173\u8ba2\u5355\uff0c\u53f3\u4fa7\u7edf\u8ba1\u5df2\u6309\u5168\u90e8\u76f8\u5173\u5173\u7cfb\u8ba1\u7b97`
    : "\u5f53\u524d\u89c6\u56fe\u652f\u6301\u6eda\u8f6e\u7f29\u653e\u3001\u62d6\u62fd\u5e73\u79fb\u548c\u8282\u70b9\u5fae\u8c03\u5e03\u5c40";

  const selectionHighlights = buildGraphSelectionHighlightsV2(graph);
  const selectedSummary = selectedNode ? `
    <div class="graph-selected-summary" data-graph-selected-summary>
      <div class="graph-selected-main">
        <span class="graph-selected-badge" style="background:${graphTypeColor(selectedNode.type)}"></span>
        <div>
          <div class="graph-selected-title">${escapeHtml(selectedNode.label || selectedNode.id)}</div>
          <div class="graph-selected-meta">${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.entity_id || selectedNode.id || "-")}</div>
        </div>
      </div>
      <div class="graph-selected-stats">
        <span>\u76f4\u63a5\u90bb\u5c45 ${formatInt(graph.selectedStats.directNeighborCount)}</span>
        <span>\u4e0a\u6e38 / \u4e0b\u6e38 ${formatInt(graph.selectedStats.upstreamCount)} / ${formatInt(graph.selectedStats.downstreamCount)}</span>
        <span>\u5168\u76f8\u5173\u8282\u70b9 ${formatInt(graph.selectedStats.relatedNodeCount)}</span>
        <span>\u5168\u76f8\u5173\u8fb9 ${formatInt(graph.selectedStats.relatedEdgeCount)}</span>
        <span>\u7ed3\u6784\u8fb9 ${formatInt(graph.selectedStats.structureEdgeCount)}</span>
        <span>\u8d44\u6e90\u8fb9 ${formatInt(graph.selectedStats.resourceEdgeCount)}</span>
      </div>
    </div>
  ` : renderEmptyState("\u672a\u9009\u4e2d\u8282\u70b9", "\u70b9\u51fb\u5de6\u4fa7\u56fe\u4e2d\u7684\u8282\u70b9\u540e\uff0c\u8fd9\u91cc\u4f1a\u663e\u793a\u8be5\u8282\u70b9\u76f8\u5173\u7684\u5b8c\u6574\u7edf\u8ba1\u4e0e\u5173\u7cfb\u8bf4\u660e\u3002");

  const scopeOverview = selectedNode ? renderKeyValueGrid([
    { label: "\u5168\u76f8\u5173\u8ba2\u5355", value: formatInt(graph.selectedStats.orderCount) },
    { label: "\u5168\u76f8\u5173\u4efb\u52a1 / \u5de5\u5e8f", value: `${formatInt(graph.selectedStats.taskCount)} / ${formatInt(graph.selectedStats.operationCount)}` },
    { label: "\u5168\u76f8\u5173\u673a\u5668 / \u5de5\u88c5 / \u4eba\u5458", value: `${formatInt(graph.selectedStats.machineCount)} / ${formatInt(graph.selectedStats.toolingCount)} / ${formatInt(graph.selectedStats.personnelCount)}` },
    { label: "\u5f53\u524d\u56fe\u4e2d\u5c55\u793a\u8ba2\u5355", value: `${formatInt(graph.selectedStats.displayedOrderCount)} / ${formatInt(graph.selectedStats.orderCount)}` },
  ]) : "";

  return `
    <div class="surface-card graph-workbench">
      <div class="card-head">
        <h3>\u53ef\u4ea4\u4e92\u6709\u5411\u5f02\u6784\u56fe</h3>
        <p>\u4fdd\u7559\u8ba2\u5355\u3001\u4efb\u52a1\u3001\u5de5\u5e8f\u3001\u673a\u5668\u3001\u5de5\u88c5\u3001\u4eba\u5458\u7b49\u5168\u90e8\u5173\u7cfb\u7c7b\u578b\uff0c\u5de6\u4fa7\u56fe\u8d1f\u8d23\u5c55\u793a\uff0c\u53f3\u4fa7\u9762\u677f\u4e13\u6ce8\u8be5\u8282\u70b9\u7684\u5b8c\u6574\u5173\u8054\u7edf\u8ba1\u4e0e\u89e3\u91ca\u3002</p>
      </div>
      <div class="graph-toolbar">
        <div class="inline-actions">
          <button class="btn ${app.graphView.mode === "focus" ? "btn-primary" : "btn-ghost"}" type="button" data-action="set-graph-mode" data-mode="focus">\u7126\u70b9\u90bb\u57df</button>
          <button class="btn ${app.graphView.mode === "all" ? "btn-primary" : "btn-ghost"}" type="button" data-action="set-graph-mode" data-mode="all">\u5168\u5173\u7cfb\u89c6\u56fe</button>
          <button class="btn btn-ghost" type="button" data-action="fit-graph-view">\u9002\u914d\u89c6\u56fe</button>
          <button class="btn btn-ghost" type="button" data-action="reset-graph-view">\u91cd\u7f6e\u89c6\u56fe</button>
          <button class="btn btn-ghost" type="button" data-action="zoom-graph-out">-</button>
          <button class="btn btn-ghost" type="button" data-action="toggle-graph-fullscreen">\u5168\u5c4f\u67e5\u770b</button>
          <span class="graph-zoom-pill" data-graph-zoom>${Math.round(app.graphView.zoom * 100)}%</span>
          <button class="btn btn-ghost" type="button" data-action="zoom-graph-in">+</button>
        </div>
        <label class="graph-search">
          <span>\u641c\u7d22\u8282\u70b9</span>
          <input type="search" value="${escapeHtml(app.graphView.search || "")}" data-graph-search placeholder="\u641c\u7d22\u8ba2\u5355\u3001\u4efb\u52a1\u3001\u5de5\u5e8f\u6216\u8d44\u6e90">
        </label>
        <label class="graph-search graph-order-limit">
          <span>\u6700\u591a\u5c55\u793a\u8ba2\u5355\u6570</span>
          <input type="number" min="1" max="20" step="1" value="${escapeHtml(String(app.graphView.maxOrders || 6))}" id="graph-max-orders">
        </label>
      </div>
      <div class="graph-stage-meta">
        <span>\u5c55\u793a\u8282\u70b9 ${formatInt(graph.visibleNodes.length)} / ${formatInt(graph.totalNodeCount)}</span>
        <span>\u5c55\u793a\u8fb9 ${formatInt(graph.visibleEdges.length)} / ${formatInt(graph.totalEdgeCount)}</span>
        <span>${stageNarrative}</span>
      </div>
      <div class="graph-filter-grid">
        <section class="graph-filter-group">
          <div class="graph-filter-title">\u8282\u70b9\u5c42\u7ea7</div>
          <div class="filter-chip-row">${nodeTypeFilters}</div>
        </section>
        <section class="graph-filter-group">
          <div class="graph-filter-title">\u5173\u7cfb\u5c42\u7ea7</div>
          <div class="filter-chip-row">${edgeGroupFilters}</div>
        </section>
      </div>
      <div class="split-panel graph-split">
        <article class="surface-card graph-stage-card">
          <div class="graph-hover-preview" data-graph-hover-preview>
            ${selectedNode ? `\u5f53\u524d\u9009\u4e2d\uff1a${escapeHtml(graphTypeLabel(selectedNode.type))} / ${escapeHtml(selectedNode.label || selectedNode.id)} / \u5173\u8054\u5173\u7cfb ${formatInt(nodeDegree.get(selectedNode.id) || 0)}` : "\u5c06\u9f20\u6807\u60ac\u6d6e\u5230\u8282\u70b9\u4e0a\uff0c\u53ef\u5feb\u901f\u9884\u89c8\u8be5\u8282\u70b9\u540d\u79f0\u3001\u7c7b\u578b\u4e0e\u5173\u8054\u5173\u7cfb\u5f3a\u5ea6\u3002"}
          </div>
          <div class="graph-shell">${svg}</div>
        </article>
        <article class="surface-card graph-detail-card">
          <div class="card-head">
            <h3>\u8282\u70b9\u8be6\u60c5\u4e0e\u5173\u7cfb\u89e3\u91ca</h3>
            <p>\u53f3\u4fa7\u6240\u6709\u7edf\u8ba1\u90fd\u57fa\u4e8e\u201c\u9009\u4e2d\u8282\u70b9\u7684\u5b8c\u6574\u76f8\u5173\u4f5c\u7528\u57df\u201d\u8ba1\u7b97\uff0c\u800c\u4e0d\u662f\u53ea\u57fa\u4e8e\u5de6\u4fa7\u53ef\u89c1\u5c40\u90e8\u56fe\u3002</p>
          </div>
          ${selectedSummary}
          ${scopeOverview}
          ${selectionHighlights.length ? renderKeyValueGrid(selectionHighlights) : ""}
          ${selectedNode ? renderKeyValueGrid(graphNodeDetailRows(selectedNode, graph.relatedEdges)) : renderEmptyState("\u672a\u9009\u4e2d\u8282\u70b9", "\u8bf7\u5148\u5728\u5de6\u4fa7\u56fe\u4e2d\u70b9\u51fb\u4e00\u4e2a\u8282\u70b9\u3002")}
          ${Object.keys(graph.relatedByType).length ? `
            <div class="graph-neighbor-groups">
              ${Object.entries(graph.relatedByType).map(([type, items]) => `
                <section class="graph-neighbor-group">
                  <header>
                    <strong>${escapeHtml(graphTypeLabel(type))}</strong>
                    <span>${formatInt(items.length)} \u4e2a</span>
                  </header>
                  <div class="graph-neighbor-pills">
                    ${items.slice(0, 10).map((item) => `<button class="pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(item.id)}">${escapeHtml(item.label || item.id)}</button>`).join("")}
                  </div>
                </section>
              `).join("")}
            </div>
          ` : ""}
          ${graph.relatedEdges.length ? renderSimpleTable(
            ["\u5173\u7cfb", "\u6765\u6e90", "\u76ee\u6807"],
            graph.relatedEdges.slice(0, 16).map((edge) => [
              escapeHtml(humanizeCodeLabel(edge.edgeType || edge.group || "-")),
              escapeHtml(graph.nodeMap.get(edge.source)?.label || edge.source),
              escapeHtml(graph.nodeMap.get(edge.target)?.label || edge.target),
            ]),
            { footer: graph.relatedEdges.length > 16 ? `\u5f53\u524d\u53ea\u5c55\u793a\u524d 16 \u6761\u5173\u8054\u8fb9\uff0c\u5171 ${graph.relatedEdges.length} \u6761\u3002` : "" },
          ) : renderEmptyState("\u6682\u65e0\u5173\u8054\u8fb9", "\u5f53\u524d\u8282\u70b9\u5728\u5168\u76f8\u5173\u4f5c\u7528\u57df\u4e2d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u5173\u7cfb\u8fb9\u3002")}
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
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    if (drag.type === "pan") {
      if (Math.abs(event.clientX - drag.startX) > 2 || Math.abs(event.clientY - drag.startY) > 2) dragMoved = true;
      app.graphView.panX = drag.originPanX + (event.clientX - drag.startX) * drag.scales.x;
      app.graphView.panY = drag.originPanY + (event.clientY - drag.startY) * drag.scales.y;
      applyGraphViewportState(root);
      return;
    }
    const position = app.graphView.positions[drag.id];
    if (!position) return;
    if (Math.abs(event.clientX - drag.startX) > 2 || Math.abs(event.clientY - drag.startY) > 2) dragMoved = true;
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

function legacyGraphDataset() {
  const allNodes = asArray(app.graphNodes).map(normalizeGraphNode).filter((node) => node.id);
  const allEdges = asArray(app.graphEdges).map(normalizeGraphEdge).filter((edge) => edge.source && edge.target);
  const nodeIds = new Set(allNodes.map((node) => node.id));
  return {
    nodes: allNodes,
    edges: allEdges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)),
  };
}

function selectedGraphOrderOption() {
  return asArray(app.graphOrderOptions).find((node) => {
    return normalizeGraphNode(node).id === app.selectedGraphOrderId;
  }) || null;
}

function legacyGraphDetailContent(selectedNode, selectedEdges) {
  return `
    <div class="card-head"><h3>节点详情</h3><p>点击左侧节点后，这里会同步显示属性与直接关系。</p></div>
    ${selectedNode ? renderKeyValueGrid(graphNodeDetailRows(selectedNode, selectedEdges)) : ""}
    ${selectedEdges.length ? renderSimpleTable(
      ["关系", "来源", "目标"],
      selectedEdges.slice(0, 16).map((edge) => [
        escapeHtml(humanizeCodeLabel(edge.edgeType || "-")),
        escapeHtml(edge.source),
        escapeHtml(edge.target),
      ]),
      { footer: selectedEdges.length > 16 ? `当前节点共 ${selectedEdges.length} 条直接关系，仅展示前 16 条。` : "" },
    ) : renderEmptyState("暂无直接关系", "当前节点在已加载样本中没有直接关系。")}
  `;
}

function legacyGraphSelectionContent(selectedNode, selectedEdges, nodes) {
  if (!selectedNode) return '<span class="subtle-note">尚未选中节点</span>';

  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const relationButtons = selectedEdges.slice(0, 12).map((edge) => {
    const isOutgoing = edge.source === selectedNode.id;
    const neighborId = isOutgoing ? edge.target : edge.source;
    const neighbor = nodeById.get(neighborId);
    const neighborLabel = neighbor?.label || neighbor?.entity_id || neighborId;
    const direction = isOutgoing ? "→" : "←";
    return `
      <button class="pill" type="button" data-action="focus-graph-node" data-id="${escapeHtml(neighborId)}" title="选中关联节点 ${escapeHtml(neighborLabel)}">
        ${escapeHtml(humanizeCodeLabel(edge.edgeType || "关系"))} ${direction} ${escapeHtml(neighborLabel)}
      </button>
    `;
  }).join("");

  return `
    <span class="pill active" title="当前选中节点">
      当前：${escapeHtml(graphTypeLabel(selectedNode.type))} · ${escapeHtml(selectedNode.label || selectedNode.entity_id || selectedNode.id)}
    </span>
    ${relationButtons || '<span class="subtle-note">当前节点没有直接关系</span>'}
    ${selectedEdges.length > 12 ? `<span class="subtle-note">另有 ${formatInt(selectedEdges.length - 12)} 条直接关系</span>` : ""}
  `;
}

function focusLegacyCytoscapeNode(nodeId, options = {}) {
  const cy = app.cyGraphInstance;
  const node = cy?.getElementById(nodeId || "");
  if (!node?.length) return false;

  app.selectedGraphNodeId = node.id();
  cy.elements().removeClass("cy-selected cy-neighbor cy-neighbor-edge cy-dimmed cy-search-match");
  cy.elements().addClass("cy-dimmed");
  const neighborhood = node.neighborhood();
  node.removeClass("cy-dimmed").addClass("cy-selected");
  neighborhood.nodes().removeClass("cy-dimmed").addClass("cy-neighbor");
  node.connectedEdges().removeClass("cy-dimmed").addClass("cy-neighbor-edge");

  const { nodes, edges } = legacyGraphDataset();
  const selectedNode = nodes.find((item) => item.id === node.id()) || null;
  const selectedEdges = edges.filter((edge) => edge.source === node.id() || edge.target === node.id());
  const root = document.fullscreenElement?.classList.contains("legacy-graph-workbench")
    ? document.fullscreenElement
    : document.querySelector(".page.active .legacy-graph-workbench");
  const detail = root?.querySelector(".graph-detail-card");
  if (detail) detail.innerHTML = legacyGraphDetailContent(selectedNode, selectedEdges);
  const selection = root?.querySelector("[data-graph-selection]");
  if (selection) selection.innerHTML = legacyGraphSelectionContent(selectedNode, selectedEdges, nodes);
  if (options.fit) cy.animate({ fit: { eles: node.union(neighborhood), padding: 70 } }, { duration: 250 });
  return true;
}

function renderLegacyCytoscapeGraph() {
  const { nodes, edges } = legacyGraphDataset();
  if (!nodes.length) {
    return renderEmptyState("暂无图谱节点", "请先构建图谱，或确认当前实例已正确加载。");
  }

  const selectedId = nodes.some((node) => node.id === app.selectedGraphNodeId)
    ? app.selectedGraphNodeId
    : nodes[0].id;
  app.selectedGraphNodeId = selectedId;
  const selectedNode = nodes.find((node) => node.id === selectedId) || null;
  const selectedEdges = edges.filter((edge) => edge.source === selectedId || edge.target === selectedId);
  const selectedOrder = selectedGraphOrderOption();
  const selectedOrderValue = selectedOrder
    ? normalizeGraphNode(selectedOrder).entity_id
    : entityIdFromGraphId(app.selectedGraphOrderId || "");

  return `
    <div class="surface-card graph-workbench legacy-graph-workbench">
      <div class="card-head">
        <h3>订单关联图谱</h3>
        <p>展示当前订单的任务、工序依赖与可用资源。</p>
      </div>
      <div class="graph-toolbar legacy-graph-toolbar">
        <label class="graph-search graph-order-filter">
          <span>筛选订单</span>
          <input type="search" data-cy-order-filter list="graph-order-options" value="${escapeHtml(selectedOrderValue)}" placeholder="输入订单 ID 或名称" autocomplete="off">
          <datalist id="graph-order-options">
            ${asArray(app.graphOrderOptions).map((node) => {
              const order = normalizeGraphNode(node);
              return `<option value="${escapeHtml(order.entity_id)}">${escapeHtml(order.label || order.entity_id)}</option>`;
            }).join("")}
          </datalist>
        </label>
        <span class="subtle-note">从上到下展开 · 仅展示当前订单完整关联</span>
        <label class="legacy-graph-switch"><input type="checkbox" data-cy-resource-edges checked> 显示资源可行边</label>
        <button class="btn btn-ghost" type="button" data-cy-fit>适配视图</button>
        <button class="btn btn-ghost" type="button" data-action="toggle-graph-fullscreen" aria-pressed="false">全屏查看</button>
      </div>
      <div class="graph-stage-meta">
        <span>当前订单 ${escapeHtml(selectedOrderValue || "-")}</span>
        <span>关联节点 ${formatInt(nodes.length)} · 关联边 ${formatInt(edges.length)}</span>
        <span>单击节点查看邻域，双击节点聚焦</span>
      </div>
      <div class="graph-neighbor-pills legacy-graph-shortcuts" data-graph-selection aria-label="当前选中节点与直接关系">
        ${legacyGraphSelectionContent(selectedNode, selectedEdges, nodes)}
      </div>
      <div class="split-panel graph-split">
        <article class="surface-card graph-stage-card">
          <div class="graph-shell legacy-graph-shell"><div class="cy-graph" data-cy-graph></div></div>
          <div class="legend graph-legend">
            ${GRAPH_NODE_ORDER.map((type) => `
              <span class="legend-item"><span class="legend-swatch" style="background:${graphTypeColor(type)}"></span>${escapeHtml(graphTypeLabel(type))}</span>
            `).join("")}
          </div>
        </article>
        <article class="surface-card graph-detail-card">
          ${legacyGraphDetailContent(selectedNode, selectedEdges)}
        </article>
      </div>
      <div class="cy-graph-accessibility" aria-hidden="true">
        ${nodes.map((node) => `<button type="button" data-action="focus-graph-node" data-id="${escapeHtml(node.id)}" data-graph-node="${escapeHtml(node.id)}" data-node-label="${escapeHtml(node.label || node.id)}" data-node-type-label="${escapeHtml(graphTypeLabel(node.type))}"></button>`).join("")}
        ${edges.map((edge) => `<span data-graph-link data-source="${escapeHtml(edge.source)}" data-target="${escapeHtml(edge.target)}"></span>`).join("")}
      </div>
    </div>
  `;
}

function mountGantts() {
  if (typeof window.vis === "undefined" || typeof window.vis.Timeline !== "function") return;
  const liveCanvasIds = new Set(Array.from(document.querySelectorAll(".page.active .gantt-canvas")).map((el) => el.id));
  // Destroy only orphaned instances (canvas no longer in the active DOM); keep bound, still-visible instances intact so a redundant mountGantts call never blanks a live canvas.
  app.ganttInstances = app.ganttInstances.filter((entry) => {
    if (liveCanvasIds.has(entry.canvasId)) return true;
    try { entry.timeline.destroy(); } catch (_) {}
    return false;
  });
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
        format: {
          minorLabels: { minute: "HH:mm", hour: "HH:mm", day: "D日", week: "w周", month: "M月", year: "YYYY" },
          majorLabels: { minute: "M月D日", hour: "M月D日", day: "YYYY年M月", week: "YYYY年M月", month: "YYYY年", year: "" },
        },
      }
    );
    app.ganttInstances.push({ canvasId: el.id, timeline });
  });
}

function mountLegacyCytoscapeGraph() {
  const container = document.querySelector(".page.active [data-cy-graph]");
  if (!container || container.dataset.bound === "1") return;
  container.dataset.bound = "1";

  if (typeof window.cytoscape !== "function") {
    container.innerHTML = '<div class="empty-state"><h3>图谱组件加载失败</h3><p>请确认 /static/vendor 下的 Cytoscape 运行库完整。</p></div>';
    return;
  }

  if (app.cyGraphInstance) app.cyGraphInstance.destroy();
  const root = container.closest(".legacy-graph-workbench");
  const { nodes, edges } = legacyGraphDataset();
  const nodeIds = new Set(nodes.map((node) => node.id));
  const elements = [
    ...nodes.map((node) => ({
      data: {
        id: node.id,
        label: node.label || node.entity_id || node.id,
        node_type: node.type,
        entity_id: node.entity_id || "",
      },
    })),
    ...edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target)).map((edge, index) => ({
      data: {
        id: `cy-edge-${edge.id || index}`,
        source: edge.source,
        target: edge.target,
        edge_type: edge.edgeType,
        edge_group: edge.group,
      },
    })),
  ];

  const cy = window.cytoscape({
    container,
    elements,
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
      { selector: 'edge[edge_type="operation_sequence"]', style: { "width": 2.4, "line-color": "#0b5f8a", "target-arrow-color": "#0b5f8a", "opacity": 0.95 } },
      { selector: 'edge[edge_group="resource"]', style: { "line-style": "dashed", "line-color": "#b76800", "target-arrow-color": "#b76800", "opacity": 0.58 } },
      { selector: ".cy-selected", style: { "border-width": 5, "border-color": "#102f4c", "width": 46, "height": 46, "font-size": 12, "font-weight": 700, "z-index": 9999 } },
      { selector: ".cy-neighbor", style: { "border-width": 3, "border-color": "#0f4c81", "opacity": 1, "z-index": 100 } },
      { selector: ".cy-neighbor-edge", style: { "width": 3, "opacity": 1, "z-index": 100 } },
      { selector: ".cy-dimmed", style: { "opacity": 0.1 } },
      { selector: ".cy-search-match", style: { "border-width": 5, "border-color": "#d97706", "width": 42, "height": 42, "z-index": 9999 } },
    ],
    minZoom: 0.05,
    maxZoom: 5,
    boxSelectionEnabled: false,
  });
  app.cyGraphInstance = cy;

  const enforceSerialTopToBottom = () => {
    const sequenceEdges = cy.edges('[edge_type="operation_sequence"]');
    for (let pass = 0; pass < cy.nodes('[node_type="operation"]').length; pass += 1) {
      let moved = false;
      sequenceEdges.forEach((edge) => {
        const source = edge.source();
        const target = edge.target();
        if (target.position("y") > source.position("y")) return;
        target.position("y", source.position("y") + 150);
        moved = true;
      });
      if (!moved) break;
    }
  };

  const runLayout = () => {
    try {
      cy.layout({ name: "dagre", rankDir: "TB", rankSep: 150, nodeSep: 28, edgeSep: 14, ranker: "longest-path", padding: 40, fit: true, animate: false }).run();
    } catch (error) {
      console.warn("Dagre layout unavailable, falling back to breadthfirst", error);
      cy.layout({ name: "breadthfirst", directed: true, spacingFactor: 1.2, padding: 40, fit: true }).run();
    }
    enforceSerialTopToBottom();
    cy.fit(undefined, 40);
  };
  runLayout();
  const initial = cy.getElementById(app.selectedGraphNodeId || "");
  if (initial.length) {
    focusLegacyCytoscapeNode(initial.id());
    cy.fit(initial.union(initial.neighborhood()), 70);
  }

  cy.on("tap", "node", (event) => {
    focusLegacyCytoscapeNode(event.target.id());
  });
  cy.on("tap", "edge", (event) => {
    const edge = event.target;
    const source = edge.source();
    focusLegacyCytoscapeNode(source.id());
    edge.removeClass("cy-dimmed").addClass("cy-neighbor-edge");
    edge.target().removeClass("cy-dimmed").addClass("cy-neighbor");
  });
  cy.on("dbltap", "node", (event) => {
    cy.animate({ fit: { eles: event.target.union(event.target.neighborhood()), padding: 70 } }, { duration: 350 });
  });
  cy.on("tap", (event) => {
    if (event.target !== cy) return;
    cy.elements().removeClass("cy-selected cy-neighbor cy-neighbor-edge cy-dimmed cy-search-match");
  });

  const orderFilter = root?.querySelector("[data-cy-order-filter]");
  const applyOrderFilter = async () => {
    const value = String(orderFilter?.value || "").trim().toLowerCase();
    if (!value) return;
    const orders = asArray(app.graphOrderOptions).map(normalizeGraphNode);
    const fields = (node) => [node.id, node.entity_id, node.label].map((item) => String(item || "").toLowerCase());
    const match = orders.find((node) => fields(node).includes(value))
      || orders.find((node) => fields(node).some((item) => item.includes(value)));
    if (!match) {
      toast(`未找到订单：${orderFilter.value}`, "warning");
      return;
    }
    if (match.id === app.selectedGraphOrderId) return;
    try {
      await loadGraphOrder(match.entity_id);
      await renderCurrentPage();
    } catch (error) {
      toast(`加载订单图谱失败：${error.message}`, "warning");
    }
  };
  orderFilter?.addEventListener("change", applyOrderFilter);
  orderFilter?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    applyOrderFilter();
  });
  root?.querySelector("[data-cy-fit]")?.addEventListener("click", () => cy.fit(undefined, 40));
  root?.querySelector("[data-cy-resource-edges]")?.addEventListener("change", (event) => {
    cy.edges('[edge_group="resource"]').style("display", event.target.checked ? "element" : "none");
    runLayout();
  });
}


async function renderCurrentPage() {
  ensureReviewSelection();
  updateShell();
  if (app.currentPage === "new-scene") {
    const box = el("new-scene-validation");
    if (box) box.innerHTML = app.currentScene ? renderValidationPanel() : "";
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

async function refreshReferenceSolutions() {
  if (!app.heuristicSelection.length) {
    app.referenceSolutions = [];
    return;
  }
  try {
    const result = await api.simulateReferenceSolutions(
      app.heuristicSelection,
      app.optimizeResult?.objective_keys || app.optimizeForm.objectiveKeys,
    );
    app.referenceSolutions = asArray(result?.solutions);
    ensureReviewSelection();
  } catch (error) {
    toast(`加载启发式参考方案失败：${error.message}`, "warning");
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
    app.validation = await api.validateInstance();
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

async function initializeGraphOrderView(meta, preferredOrderId = null) {
  const total = Number(meta?.node_type_counts?.order || 0);
  const requests = [];
  for (let offset = 0; offset < total; offset += 1000) {
    requests.push(api.getGraphNodes(Math.min(1000, total - offset), offset, "order"));
  }
  const pages = requests.length ? await Promise.all(requests) : [];
  app.graphOrderOptions = pages
    .flatMap((page) => asArray(page?.nodes || page))
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

function stopGraphBuildPolling() {
  if (app.graphBuildPollTimer) window.clearTimeout(app.graphBuildPollTimer);
  app.graphBuildPollTimer = null;
}

async function refreshGraphBuildFeedback() {
  const panel = el("graph-build-status-panel");
  if (panel) panel.outerHTML = renderGraphBuildStatus();
  else if (app.currentPage === "workflow") await renderCurrentPage();
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
  if (action === "goto-new-scene") return navigate("new-scene");
  if (action === "goto-dashboard") return navigate("dashboard");
  if (action === "goto-review") return navigate("solution-review");
  if (action === "focus-graph-node") {
    if (app.graphSuppressClickUntil && Date.now() < app.graphSuppressClickUntil) return;
    if (target.closest(".legacy-graph-workbench") && focusLegacyCytoscapeNode(target.dataset.id, { fit: true })) return;
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
  if (action === "load-heuristic-references") {
    await refreshReferenceSolutions();
    return renderCurrentPage();
  }
  if (action === "generate-exact-single") return handleGenerateExact("single");
  if (action === "generate-exact-weighted") return handleGenerateExact("weighted");
  if (action === "export-selected-solution") return handleExportSolution(target?.dataset.id || getSelectedReviewCandidate()?.id);
  if (action === "focus-candidate") {
    app.reviewDetailId = target.dataset.id;
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
  if (action === "set-workflow-focus") {
    app.workflowFocus = target.dataset.focus || "graph";
    return renderWorkflow();
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
    const resizeGraph = () => {
      const cy = app.cyGraphInstance;
      if (!cy) return;
      cy.resize();
      const selected = cy.getElementById(app.selectedGraphNodeId || "");
      if (selected.length) cy.fit(selected.union(selected.neighborhood()), fullscreenGraph ? 110 : 70);
    };
    window.requestAnimationFrame(resizeGraph);
    window.setTimeout(resizeGraph, 120);
  });

  document.addEventListener("click", async (event) => {
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
    if (target.matches("[data-heuristic-rule]")) {
      app.heuristicSelection = Array.from(document.querySelectorAll("[data-heuristic-rule]"))
        .filter((node) => node.checked)
        .map((node) => node.dataset.heuristicRule);
      return;
    }
    if (target.matches('[data-action="toggle-candidate"]')) {
      const id = target.dataset.id;
      if (target.checked) {
        if (!app.reviewSelection.includes(id) && app.reviewSelection.length >= 4) {
          target.checked = false;
          toast("AI 评审最多同时选择 4 个方案。", "warning");
          return;
        }
        if (!app.reviewSelection.includes(id)) app.reviewSelection.push(id);
        app.reviewDetailId = id;
      } else {
        app.reviewSelection = app.reviewSelection.filter((item) => item !== id);
        if (app.reviewDetailId === id) app.reviewDetailId = app.reviewSelection[0] || null;
      }
      updateShell();
      return renderCurrentPage();
    }
    if (target.matches("#workflow-sim-rule")) app.simRule = target.value;
    if (target.matches("#ai-solution-select")) {
      app.reviewDetailId = target.value;
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
    if (target.matches("#graph-max-orders")) {
      app.graphView.maxOrders = Math.max(1, Math.min(20, Number(target.value || app.graphView.maxOrders || 6)));
      renderCurrentPage();
      return;
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
  if (app.currentScene) await loadExistingGraph();
  const navKey = window.location.hash.replace("#", "") || (app.currentScene ? "dashboard" : "new-scene");
  await navigate(navKey, false);
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    console.error(error);
    toast(`应用初始化失败：${error.message}`, "warning");
  });
});
