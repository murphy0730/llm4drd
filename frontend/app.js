const RULES = ["ATC", "EDD", "SPT", "LPT", "CR", "FIFO", "MST", "PRIORITY", "KIT_AWARE", "BOTTLENECK", "COMPOSITE"];
const NODE_TYPE_ORDER = ["order", "task", "operation", "machine", "tooling", "personnel"];
const NODE_COLORS = { order: "#9b4d17", task: "#c7681f", operation: "#3d5a80", machine: "#2a9d8f", tooling: "#a44a3f", personnel: "#6c5b7b" };
const EDGE_COLORS = { order_has_task: "#9b4d17", task_predecessor: "#bb3e03", task_has_operation: "#c7681f", op_depends_task: "#6c757d", operation_sequence: "#3d5a80", machine_eligible: "#2a9d8f", tooling_eligible: "#a44a3f", personnel_eligible: "#6c5b7b" };
const TAB_SEQUENCE = ["config", "graph", "simulate", "optimize", "llm"];
const REFERENCE_RULE_LIMIT = 4;
const LLM_PRESETS = [
  { name: "DeepSeek Chat", provider: "DeepSeek", model: "deepseek-chat", base_url: "https://api.deepseek.com/v1", description: "通用对话与方案说明。" },
  { name: "DeepSeek Reasoner", provider: "DeepSeek", model: "deepseek-reasoner", base_url: "https://api.deepseek.com/v1", description: "适合复杂推理与解释比较。" },
  { name: "OpenAI GPT-4.1-mini", provider: "OpenAI", model: "gpt-4.1-mini", base_url: "https://api.openai.com/v1", description: "快速响应，适合方案摘要。" },
  { name: "OpenAI GPT-4o-mini", provider: "OpenAI", model: "gpt-4o-mini", base_url: "https://api.openai.com/v1", description: "多模态与通用任务均衡。" },
  { name: "Qwen Max", provider: "阿里云百炼", model: "qwen-max", base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1", description: "中文业务理解能力较强。" },
  { name: "Moonshot Kimi", provider: "Moonshot", model: "moonshot-v1-8k", base_url: "https://api.moonshot.cn/v1", description: "适合长文本方案比较。" },
  { name: "GLM-4-Flash", provider: "智谱", model: "glm-4-flash", base_url: "https://open.bigmodel.cn/api/paas/v4", description: "轻量快速，适合交互分析。" },
  { name: "SiliconFlow Qwen", provider: "SiliconFlow", model: "Qwen/Qwen2.5-72B-Instruct", base_url: "https://api.siliconflow.cn/v1", description: "兼容 OpenAI 风格接口。" },
  { name: "Custom OpenAI Compatible", provider: "Custom", model: "", base_url: "", description: "自定义兼容 OpenAI 的网关地址与模型名。" },
];

const OPTIMIZE_PRESETS = {
  fast: {
    time_limit_s: 30,
    population_size: 12,
    generations: 5,
    alns_iterations_per_candidate: 2,
    coarse_time_ratio: 0.78,
    promotion_pool_multiplier: 2,
    random_promotion_ratio: 0.08,
    refine_rounds: 1,
    alns_aggression: 0.85,
  },
  balanced: {
    time_limit_s: 90,
    population_size: 24,
    generations: 12,
    alns_iterations_per_candidate: 6,
    coarse_time_ratio: 0.68,
    promotion_pool_multiplier: 3,
    random_promotion_ratio: 0.12,
    refine_rounds: 1,
    alns_aggression: 1.0,
  },
  deep: {
    time_limit_s: 180,
    population_size: 36,
    generations: 18,
    alns_iterations_per_candidate: 10,
    coarse_time_ratio: 0.58,
    promotion_pool_multiplier: 4,
    random_promotion_ratio: 0.16,
    refine_rounds: 2,
    alns_aggression: 1.2,
  },
};

const state = {
  instanceDb: null,
  instanceDetails: null,
  downtimes: [],
  graphMeta: null,
  graphNodes: [],
  graphEdges: [],
  graphNodeFilter: "",
  graphEdgeFilter: "",
  selectedNodeId: null,
  selectedNodeNeighbors: null,
  simResult: null,
  objectiveCatalog: [],
  selectedObjectiveKeys: new Set(["total_tardiness", "makespan", "avg_utilization"]),
  optimizeTaskId: null,
  optimizeStatus: null,
  optimizeResult: null,
  optimizePollTimer: null,
  currentTab: "config",
  selectedReferenceRules: new Set(["ATC", "EDD"]),
  referenceRuleSolutions: [],
  exactObjectiveCatalog: [],
  exactReferenceSolutions: [],
  selectedExactReferenceIds: new Set(),
  llmConfig: null,
  paretoAssistant: null,
  paretoConversation: [],
  paretoPending: null,
};

function $(id) { return document.getElementById(id); }
function pad(v) { return String(v).padStart(2, "0"); }
function escapeHtml(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;"); }
function formatNumber(value, digits = 2) { if (value === null || value === undefined || value === "") return "-"; const n = Number(value); return Number.isNaN(n) ? String(value) : n.toFixed(digits); }
function formatValue(value) { if (value === null || value === undefined || value === "") return "-"; if (Array.isArray(value)) return value.length ? value.join(", ") : "-"; if (typeof value === "number") return Number.isInteger(value) ? String(value) : formatNumber(value, 2); return String(value); }
function toLocalInput(value) { if (!value) return ""; const d = new Date(value); if (Number.isNaN(d.getTime())) return ""; return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`; }
function localInputToIso(value) { if (!value) return null; const d = new Date(value); if (Number.isNaN(d.getTime())) return null; const offset = -d.getTimezoneOffset(); const sign = offset >= 0 ? "+" : "-"; const abs = Math.abs(offset); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:00${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`; }
function parseIsoDate(value) { if (!value) return null; const d = new Date(value); return Number.isNaN(d.getTime()) ? null : d; }
function offsetToDate(offsetHours) {
  const planStart = parseIsoDate(state.instanceDetails?.plan_start_at);
  if (!planStart || offsetHours === null || offsetHours === undefined || Number.isNaN(Number(offsetHours))) return null;
  return new Date(planStart.getTime() + Number(offsetHours) * 3600 * 1000);
}
function formatTimelineLabel(offsetHours, compact = false) {
  const date = offsetToDate(offsetHours);
  if (!date) return `${formatNumber(offsetHours, 1)}h`;
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hour = pad(date.getHours());
  const minute = pad(date.getMinutes());
  return compact ? `${month}/${day} ${hour}:${minute}` : `${month}-${day} ${hour}:${minute}`;
}
function formatTimelineRange(startHours, endHours) {
  const startText = formatTimelineLabel(startHours, false);
  const endText = formatTimelineLabel(endHours, false);
  return `${startText} -> ${endText}`;
}
function normalizeList(value) { return String(value ?? "").split(/[;,，\n]+/).map((item) => item.trim()).filter(Boolean); }
function serializeList(value) { return normalizeList(value).join(";"); }
function asArray(value) { return Array.isArray(value) ? value : normalizeList(value); }
function getNodeColor(type) { return NODE_COLORS[type] || "#6b7280"; }
function getEdgeColor(type) { return EDGE_COLORS[type] || "#8d99ae"; }

async function api(path, options = {}) {
  const fetchOptions = { method: "GET", ...options, headers: { ...(options.headers || {}) } };
  if (options.json !== undefined) {
    fetchOptions.body = JSON.stringify(options.json);
    fetchOptions.headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, fetchOptions);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
}

async function downloadTemplate() {
  const response = await fetch(`/api/instance/template?ts=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/i);
  const filename = match?.[1] || `instance_template_${Date.now()}.xlsx`;
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  $("instance-action-status").textContent = `已下载最新模板: ${filename}`;
  toast(`模板已更新下载: ${filename}`, "info");
}

async function downloadApiBlob(path, body, fallbackName) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await response.text() || `HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/i);
  const filename = match?.[1] || fallbackName;
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  return filename;
}

function toast(message, type = "info") {
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  $("toast-wrap").appendChild(node);
  window.setTimeout(() => node.remove(), 3600);
}

function setHeroStatus() {
  const summary = state.instanceDetails?.summary;
  $("hero-instance-status").textContent = summary ? `${summary.orders}单 / ${summary.operations}工序` : "未加载";
  $("hero-instance-hint").textContent = summary ? `计划起点 ${state.instanceDetails.plan_start_at}` : "先生成实例或导入 Excel";
  $("hero-graph-status").textContent = state.graphMeta ? `${state.graphMeta.total_nodes} 节点` : "未构建";
  $("hero-graph-hint").textContent = state.graphMeta ? `${state.graphMeta.total_edges} 条边` : "订单-任务-工序-机器-工装-人员";
  const status = state.optimizeStatus?.status || "待启动";
  $("hero-opt-status").textContent = status === "done" ? "已完成" : status === "running" ? "进行中" : status === "error" ? "失败" : "待启动";
  $("hero-opt-hint").textContent = state.optimizeResult ? `返回 ${state.optimizeResult.found_solution_count} 个方案` : "支持选择 1-5 个业务指标";
  $("hero-llm-status").textContent = state.llmConfig?.model || "待加载";
  $("hero-llm-hint").textContent = state.llmConfig ? `${state.llmConfig.base_url || "未设置 Base URL"} / ${state.llmConfig.has_key ? "已配置Key" : "未配置Key"}` : "含快捷预设与连通性测试";
}

function setDefaultPlanStart() {
  const now = new Date();
  now.setSeconds(0, 0);
  now.setHours(8, 0, 0, 0);
  $("cfg-plan-start").value = toLocalInput(now.toISOString());
}

function populateRuleOptions() {
  $("simulate-rule").innerHTML = RULES.map((rule) => `<option value="${escapeHtml(rule)}">${escapeHtml(rule)}</option>`).join("");
  $("opt-baseline-rule").innerHTML = RULES.map((rule) => `<option value="${escapeHtml(rule)}">${escapeHtml(rule)}</option>`).join("");
  $("simulate-rule").value = "ATC";
  $("opt-baseline-rule").value = "ATC";
}

function ensureOptimizeAdvancedControls() {
  if ($("opt-search-preset")) return;
  const anchorField = $("opt-alns")?.closest(".field");
  if (!anchorField) return;
  anchorField.insertAdjacentHTML("afterend", `
    <div class="field"><label for="opt-search-preset">Search Mode</label><select id="opt-search-preset"><option value="fast">Fast</option><option value="balanced" selected>Balanced</option><option value="deep">Deep</option><option value="custom">Custom</option></select></div>
    <div class="field"><label for="opt-coarse-time-ratio">Coarse Time Ratio</label><input id="opt-coarse-time-ratio" type="number" min="0.2" max="0.9" step="0.01" value="0.68"></div>
    <div class="field"><label for="opt-promotion-multiplier">Promotion Multiplier</label><input id="opt-promotion-multiplier" type="number" min="1" value="3"></div>
    <div class="field"><label for="opt-random-promotion-ratio">Random Promotion Ratio</label><input id="opt-random-promotion-ratio" type="number" min="0" max="0.5" step="0.01" value="0.12"></div>
    <div class="field"><label for="opt-refine-rounds">Refine Rounds</label><input id="opt-refine-rounds" type="number" min="1" value="1"></div>
    <div class="field"><label for="opt-alns-aggression">ALNS Aggression</label><input id="opt-alns-aggression" type="number" min="0.4" max="2.0" step="0.05" value="1.0"></div>
  `);
  $("opt-search-preset")?.addEventListener("change", applyOptimizePreset);
  ["opt-time-limit", "opt-population", "opt-generations", "opt-alns", "opt-coarse-time-ratio", "opt-promotion-multiplier", "opt-random-promotion-ratio", "opt-refine-rounds", "opt-alns-aggression"].forEach((id) => {
    $(id)?.addEventListener("input", () => {
      if ($("opt-search-preset") && $("opt-search-preset").value !== "custom") {
        $("opt-search-preset").value = "custom";
      }
    });
  });
  applyOptimizePreset();
}

function applyOptimizePreset() {
  const presetName = $("opt-search-preset")?.value || "balanced";
  if (presetName === "custom") return;
  const preset = OPTIMIZE_PRESETS[presetName] || OPTIMIZE_PRESETS.balanced;
  $("opt-time-limit").value = preset.time_limit_s;
  $("opt-population").value = preset.population_size;
  $("opt-generations").value = preset.generations;
  $("opt-alns").value = preset.alns_iterations_per_candidate;
  $("opt-coarse-time-ratio").value = preset.coarse_time_ratio;
  $("opt-promotion-multiplier").value = preset.promotion_pool_multiplier;
  $("opt-random-promotion-ratio").value = preset.random_promotion_ratio;
  $("opt-refine-rounds").value = preset.refine_rounds;
  $("opt-alns-aggression").value = preset.alns_aggression;
}

function renderMetricCards(containerId, metrics, order = []) {
  const entries = order.length ? order.filter((key) => metrics[key] !== undefined).map((key) => [key, metrics[key]]) : Object.entries(metrics || {});
  $(containerId).innerHTML = entries.length ? entries.map(([label, value]) => `<div class="metric-card"><div class="label">${escapeHtml(metricLabel(label))}</div><div class="value">${escapeHtml(formatValue(value))}</div><div class="hint">${escapeHtml(label)}</div></div>`).join("") : `<div class="empty-state">暂无指标</div>`;
}

function buildMaps() {
  const taskMap = new Map();
  const opMap = new Map();
  (state.instanceDetails?.orders || []).forEach((order) => (order.tasks || []).forEach((task) => {
    taskMap.set(task.id, task);
    (task.ops || []).forEach((op) => opMap.set(op.id, op));
  }));
  return { taskMap, opMap };
}

function renderInstance() {
  const details = state.instanceDetails;
  const summary = details?.summary;
  $("instance-summary-cards").innerHTML = summary ? [
    ["计划起点", details.plan_start_at], ["订单数", summary.orders], ["任务数", summary.tasks], ["工序数", summary.operations],
    ["进行中工序", summary.ops_in_progress], ["已完成工序", summary.ops_completed], ["机器数", summary.machines], ["工装数", summary.toolings], ["人员数", summary.personnel], ["日历天数", summary.calendar_days],
  ].map(([label, value]) => `<div class="metric-card"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(formatValue(value))}</div></div>`).join("") : `<div class="empty-state">当前还没有加载实例。</div>`;
  $("summary-pill-holder").innerHTML = summary ? `<span class="pill ok">实例已加载</span>` : "";
  renderResourceList("machine-type-list", details?.machine_types || [], (item) => `<div class="resource-item"><div><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.id)}</span></div><div class="tag-row"><span class="tag">${item.count} 台</span>${item.is_critical ? `<span class="tag">关键</span>` : ""}</div></div>`);
  renderResourceList("tooling-type-list", details?.tooling_types || [], (item) => `<div class="resource-item"><div><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.id)}</span></div><div class="tag-row"><span class="tag">${item.count} 套</span></div></div>`);
  renderResourceList("personnel-list", details?.personnel || [], (item) => `<div class="resource-item"><div><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.id)}</span></div><div class="tag-row">${(item.skills || []).map((skill) => `<span class="tag">${escapeHtml(skill)}</span>`).join("") || `<span class="tag">无技能标签</span>`}</div></div>`);
  renderEditors();
  renderDowntime();
}

function renderResourceList(containerId, items, formatter) {
  $(containerId).innerHTML = items.length ? `<div class="resource-list">${items.map((item) => formatter(item)).join("")}</div>` : `<div class="empty-state">暂无数据</div>`;
}

function renderEditors() {
  const db = state.instanceDb;
  const details = state.instanceDetails;
  if (!db || !details) {
    ["orders-table-wrap", "tasks-table-wrap", "operations-table-wrap", "machines-table-wrap", "initial-state-table-wrap"].forEach((id) => { $(id).innerHTML = `<div class="empty-state">暂无数据</div>`; });
    return;
  }
  const { taskMap, opMap } = buildMaps();
  $("orders-table-wrap").innerHTML = `<table><thead><tr><th>订单ID</th><th>订单名称</th><th>释放时间</th><th>交付时间</th><th>优先级</th><th>任务数</th><th>操作</th></tr></thead><tbody>${(details.orders || []).map((order) => `<tr data-id="${escapeHtml(order.id)}"><td><code>${escapeHtml(order.id)}</code></td><td><input class="table-input" data-field="order_name" value="${escapeHtml(order.name || "")}"></td><td><input class="table-input" data-field="release_at" type="datetime-local" value="${escapeHtml(toLocalInput(order.release_at))}"></td><td><input class="table-input" data-field="due_at" type="datetime-local" value="${escapeHtml(toLocalInput(order.due_at))}"></td><td><input class="table-input" data-field="priority" type="number" min="1" max="9" value="${escapeHtml(order.priority)}"></td><td>${order.tasks?.length || 0}</td><td><button class="mini-btn" data-action="save-order">保存</button></td></tr>`).join("")}</tbody></table>`;
  $("tasks-table-wrap").innerHTML = `<table><thead><tr><th>任务ID</th><th>订单ID</th><th>任务名称</th><th>主任务</th><th>前置任务</th><th>释放时间</th><th>交付时间</th><th>理想最晚开始</th><th>理想最晚完成</th><th>关键余量(h)</th><th>操作</th></tr></thead><tbody>${(db.tasks || []).map((task) => { const info = taskMap.get(task.task_id); return `<tr data-id="${escapeHtml(task.task_id)}"><td><code>${escapeHtml(task.task_id)}</code></td><td><input class="table-input" data-field="order_id" value="${escapeHtml(task.order_id)}"></td><td><input class="table-input" data-field="task_name" value="${escapeHtml(task.task_name || "")}"></td><td><select class="table-select" data-field="is_main"><option value="1" ${Number(task.is_main) ? "selected" : ""}>是</option><option value="0" ${Number(task.is_main) ? "" : "selected"}>否</option></select></td><td><input class="table-input" data-field="predecessor_task_ids" value="${escapeHtml(task.predecessor_task_ids || "")}"></td><td><input class="table-input" data-field="release_at" type="datetime-local" value="${escapeHtml(toLocalInput(info?.release_at))}"></td><td><input class="table-input" data-field="due_at" type="datetime-local" value="${escapeHtml(toLocalInput(info?.due_at))}"></td><td>${escapeHtml(formatValue(info?.derived_start_at || "-"))}</td><td>${escapeHtml(formatValue(info?.derived_due_at || "-"))}</td><td>${escapeHtml(formatValue(info?.critical_slack ?? "-"))}</td><td><button class="mini-btn" data-action="save-task">保存</button></td></tr>`; }).join("")}</tbody></table>`;
  $("operations-table-wrap").innerHTML = `<table><thead><tr><th>工序ID</th><th>任务ID</th><th>名称</th><th>类型</th><th>时长(h)</th><th>前置工序</th><th>前置任务</th><th>可选机器</th><th>工装类型</th><th>人员技能</th><th>理想最晚开始</th><th>理想最晚完成</th><th>关键余量(h)</th><th>操作</th></tr></thead><tbody>${(db.operations || []).map((op) => { const info = opMap.get(op.op_id); return `<tr data-id="${escapeHtml(op.op_id)}"><td><code>${escapeHtml(op.op_id)}</code></td><td><input class="table-input" data-field="task_id" value="${escapeHtml(op.task_id)}"></td><td><input class="table-input" data-field="op_name" value="${escapeHtml(op.op_name || "")}"></td><td><input class="table-input" data-field="process_type" value="${escapeHtml(op.process_type || "")}"></td><td><input class="table-input" data-field="processing_time" type="number" min="0" step="0.01" value="${escapeHtml(op.processing_time)}"></td><td><input class="table-input" data-field="predecessor_ops" value="${escapeHtml(op.predecessor_ops || "")}"></td><td><input class="table-input" data-field="predecessor_tasks" value="${escapeHtml(op.predecessor_tasks || "")}"></td><td><input class="table-input" data-field="eligible_machine_ids" value="${escapeHtml(op.eligible_machine_ids || "")}"></td><td><input class="table-input" data-field="required_tooling_types" value="${escapeHtml(op.required_tooling_types || "")}"></td><td><input class="table-input" data-field="required_personnel_skills" value="${escapeHtml(op.required_personnel_skills || "")}"></td><td>${escapeHtml(formatValue(info?.derived_start_at || "-"))}</td><td>${escapeHtml(formatValue(info?.derived_due_at || "-"))}</td><td>${escapeHtml(formatValue(info?.critical_slack ?? "-"))}</td><td><button class="mini-btn" data-action="save-operation">保存</button></td></tr>`; }).join("")}</tbody></table>`;
  $("machines-table-wrap").innerHTML = `<table><thead><tr><th>设备ID</th><th>设备名称</th><th>设备类型</th><th>班次定义</th><th>操作</th></tr></thead><tbody>${(db.machines || []).map((machine) => `<tr data-id="${escapeHtml(machine.machine_id)}"><td><code>${escapeHtml(machine.machine_id)}</code></td><td><input class="table-input" data-field="machine_name" value="${escapeHtml(machine.machine_name || "")}"></td><td><input class="table-input" data-field="type_id" value="${escapeHtml(machine.type_id || "")}"></td><td><input class="table-input" data-field="shifts" value="${escapeHtml(machine.shifts || "")}" placeholder="0/8/10;1/8/10"></td><td><button class="mini-btn" data-action="save-machine">保存</button></td></tr>`).join("")}</tbody></table>`;
  $("initial-state-table-wrap").innerHTML = `<table><thead><tr><th>工序ID</th><th>任务ID</th><th>初始状态</th><th>开始时间</th><th>占用到</th><th>剩余工时(h)</th><th>机器</th><th>工装</th><th>人员</th><th>操作</th></tr></thead><tbody>${(db.operations || []).map((op) => { const info = opMap.get(op.op_id); return `<tr data-id="${escapeHtml(op.op_id)}"><td><code>${escapeHtml(op.op_id)}</code></td><td>${escapeHtml(op.task_id)}</td><td><select class="table-select" data-field="initial_status"><option value="" ${!(info?.initial_status) || info.initial_status === "pending" ? "selected" : ""}>pending</option><option value="ready" ${info?.initial_status === "ready" ? "selected" : ""}>ready</option><option value="processing" ${info?.initial_status === "processing" ? "selected" : ""}>processing</option><option value="completed" ${info?.initial_status === "completed" ? "selected" : ""}>completed</option></select></td><td><input class="table-input" data-field="initial_start_at" type="datetime-local" value="${escapeHtml(toLocalInput(info?.initial_start_at))}"></td><td><input class="table-input" data-field="initial_end_at" type="datetime-local" value="${escapeHtml(toLocalInput(info?.initial_end_at))}"></td><td><input class="table-input" data-field="initial_remaining_processing_time" type="number" min="0" step="0.01" value="${escapeHtml(info?.initial_remaining_processing_time ?? "")}"></td><td><input class="table-input" data-field="initial_assigned_machine_id" value="${escapeHtml(info?.initial_assigned_machine_id || "")}" placeholder="assembly_1"></td><td><input class="table-input" data-field="initial_assigned_tooling_ids" value="${escapeHtml(asArray(info?.initial_assigned_tooling_ids).join(";"))}" placeholder="TL-assembly-01"></td><td><input class="table-input" data-field="initial_assigned_personnel_ids" value="${escapeHtml(asArray(info?.initial_assigned_personnel_ids).join(";"))}" placeholder="PS-assembly-01"></td><td><button class="mini-btn" data-action="save-operation-state">保存</button></td></tr>`; }).join("")}</tbody></table>`;
}

function renderDowntime() {
  const machines = state.instanceDetails?.machines || [];
  $("downtime-machine").innerHTML = machines.length ? machines.map((machine) => `<option value="${escapeHtml(machine.id)}">${escapeHtml(machine.name)} (${escapeHtml(machine.id)})</option>`).join("") : `<option value="">请先生成实例</option>`;
  if (!state.downtimes.length) { $("downtime-table-wrap").innerHTML = `<div class="empty-state">暂无停机记录</div>`; return; }
  const machineOptions = machines.map((machine) => `<option value="${escapeHtml(machine.id)}">${escapeHtml(machine.name)}</option>`).join("");
  $("downtime-table-wrap").innerHTML = `<table><thead><tr><th>ID</th><th>设备</th><th>类型</th><th>开始</th><th>结束</th><th>操作</th></tr></thead><tbody>${state.downtimes.map((row) => `<tr data-id="${escapeHtml(row.id)}"><td><code>${escapeHtml(row.id)}</code></td><td><select class="table-select" data-field="machine_id">${machineOptions.replace(`value="${escapeHtml(row.machine_id)}"`, `value="${escapeHtml(row.machine_id)}" selected`)}</select></td><td><select class="table-select" data-field="downtime_type"><option value="planned" ${row.downtime_type === "planned" ? "selected" : ""}>计划停机</option><option value="unplanned" ${row.downtime_type === "unplanned" ? "selected" : ""}>非计划停机</option></select></td><td><input class="table-input" data-field="start_time" type="datetime-local" value="${escapeHtml(toLocalInput(row.start_at))}"></td><td><input class="table-input" data-field="end_time" type="datetime-local" value="${escapeHtml(toLocalInput(row.end_at))}"></td><td class="nowrap"><button class="mini-btn" data-action="update-downtime">保存</button><button class="mini-btn danger" data-action="delete-downtime">删除</button></td></tr>`).join("")}</tbody></table>`;
}

function renderGraphMeta() {
  const meta = state.graphMeta;
  if (!meta) {
    $("graph-meta-cards").innerHTML = `<div class="empty-state">暂无图统计信息</div>`;
    $("graph-legend").innerHTML = "";
    $("graph-edge-summary").innerHTML = `<div class="empty-state">暂无边分布信息</div>`;
    return;
  }
  renderMetricCards("graph-meta-cards", { total_nodes: meta.total_nodes, total_edges: meta.total_edges, node_types: Object.keys(meta.node_type_counts || {}).length, edge_types: Object.keys(meta.edge_type_counts || {}).length });
  $("graph-node-type").innerHTML = `<option value="">全部节点</option>` + Object.entries(meta.node_type_counts || {}).map(([type, count]) => `<option value="${escapeHtml(type)}" ${state.graphNodeFilter === type ? "selected" : ""}>${escapeHtml(type)} (${count})</option>`).join("");
  $("graph-edge-type").innerHTML = `<option value="">全部边</option>` + Object.entries(meta.edge_type_counts || {}).map(([type, count]) => `<option value="${escapeHtml(type)}" ${state.graphEdgeFilter === type ? "selected" : ""}>${escapeHtml(type)} (${count})</option>`).join("");
  $("graph-legend").innerHTML = NODE_TYPE_ORDER.filter((type) => meta.node_type_counts?.[type]).map((type) => `<span class="tag"><span class="swatch" style="background:${getNodeColor(type)}"></span>${escapeHtml(type)}</span>`).join("");
  $("graph-edge-summary").innerHTML = `<div class="tag-row">${Object.entries(meta.edge_type_counts || {}).map(([type, count]) => `<span class="tag"><span class="swatch" style="background:${getEdgeColor(type)}"></span>${escapeHtml(type)} · ${count}</span>`).join("")}</div>`;
}

function renderGraphVisual() {
  const svg = $("graph-svg");
  const nodes = state.graphNodes || [];
  const nodeMap = new Map(nodes.map((node) => [node.node_id, node]));
  const edges = (state.graphEdges || []).filter((edge) => nodeMap.has(edge.source) && nodeMap.has(edge.target));
  if (!nodes.length) {
    svg.setAttribute("viewBox", "0 0 1200 420");
    svg.innerHTML = `<text x="600" y="210" text-anchor="middle" fill="#6b7280" font-size="20">暂无图数据，请先构建异构图</text>`;
    $("graph-nodes-table-wrap").innerHTML = `<div class="empty-state">暂无节点数据</div>`;
    $("graph-edges-table-wrap").innerHTML = `<div class="empty-state">暂无边数据</div>`;
    return;
  }
  const orderedTypes = [...NODE_TYPE_ORDER.filter((type) => nodes.some((node) => node.node_type === type)), ...Array.from(new Set(nodes.map((node) => node.node_type))).filter((type) => !NODE_TYPE_ORDER.includes(type))];
  const grouped = new Map(orderedTypes.map((type) => [type, nodes.filter((node) => node.node_type === type)]));
  const width = 1280, left = 100, right = 100, top = 70, rowGap = 86, maxRows = Math.max(...Array.from(grouped.values()).map((items) => items.length), 1), height = Math.max(440, top + maxRows * rowGap + 70), colGap = orderedTypes.length > 1 ? (width - left - right) / (orderedTypes.length - 1) : 0;
  const pos = new Map();
  orderedTypes.forEach((type, colIndex) => { const items = grouped.get(type) || []; const x = left + colIndex * colGap; const offset = (height - top - 70 - (items.length - 1) * rowGap) / 2; items.forEach((node, rowIndex) => pos.set(node.node_id, { x, y: top + offset + rowIndex * rowGap })); });
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${orderedTypes.map((type, index) => { const x = left + index * colGap; return `<text x="${x}" y="36" text-anchor="middle" fill="#5f6c7b" font-size="14" font-weight="700">${escapeHtml(type)}</text><line x1="${x}" y1="52" x2="${x}" y2="${height - 28}" stroke="rgba(155,77,23,0.08)" stroke-dasharray="4 6"></line>`; }).join("")}${edges.map((edge) => { const s = pos.get(edge.source); const t = pos.get(edge.target); const mid = (s.x + t.x) / 2; return `<path d="M ${s.x + 28} ${s.y} C ${mid} ${s.y}, ${mid} ${t.y}, ${t.x - 28} ${t.y}" fill="none" stroke="${getEdgeColor(edge.edge_type)}" stroke-width="2" opacity="0.32"></path>`; }).join("")}${nodes.map((node) => { const p = pos.get(node.node_id); const color = getNodeColor(node.node_type); const label = node.attrs?.label || node.entity_id || node.node_id; const selected = node.node_id === state.selectedNodeId; return `<g class="graph-node" data-node-id="${escapeHtml(node.node_id)}" style="cursor:pointer"><rect x="${p.x - 50}" y="${p.y - 20}" width="100" height="40" rx="14" fill="${selected ? color : "white"}" stroke="${color}" stroke-width="${selected ? 3 : 2}"></rect><text x="${p.x}" y="${p.y - 3}" text-anchor="middle" fill="${selected ? "white" : color}" font-size="13" font-weight="700">${escapeHtml(String(label).slice(0, 16))}</text><text x="${p.x}" y="${p.y + 12}" text-anchor="middle" fill="${selected ? "rgba(255,255,255,0.88)" : "#5f6c7b"}" font-size="11">${escapeHtml(node.entity_id)}</text></g>`; }).join("")}`;
  $("graph-nodes-table-wrap").innerHTML = `<table><thead><tr><th>节点ID</th><th>类型</th><th>实体ID</th><th>标签</th></tr></thead><tbody>${nodes.map((node) => `<tr><td><code>${escapeHtml(node.node_id)}</code></td><td>${escapeHtml(node.node_type)}</td><td>${escapeHtml(node.entity_id)}</td><td>${escapeHtml(node.attrs?.label || "-")}</td></tr>`).join("")}</tbody></table>`;
  $("graph-edges-table-wrap").innerHTML = `<table><thead><tr><th>源节点</th><th>目标节点</th><th>边类型</th></tr></thead><tbody>${edges.map((edge) => `<tr><td><code>${escapeHtml(edge.source)}</code></td><td><code>${escapeHtml(edge.target)}</code></td><td>${escapeHtml(edge.edge_type)}</td></tr>`).join("")}</tbody></table>`;
}

function renderNodeDetail() {
  const node = (state.graphNodes || []).find((item) => item.node_id === state.selectedNodeId);
  if (!node) { $("graph-node-detail").innerHTML = `<div class="empty-state">请选择一个节点</div>`; return; }
  const outgoing = state.selectedNodeNeighbors?.outgoing || [];
  const incoming = state.selectedNodeNeighbors?.incoming || [];
  $("graph-node-detail").innerHTML = `<div class="detail-block"><h4>${escapeHtml(node.attrs?.label || node.node_id)}</h4><dl><dt>节点ID</dt><dd><code>${escapeHtml(node.node_id)}</code></dd><dt>节点类型</dt><dd>${escapeHtml(node.node_type)}</dd><dt>实体ID</dt><dd>${escapeHtml(node.entity_id)}</dd>${Object.entries(node.attrs || {}).map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(formatValue(value))}</dd>`).join("")}</dl></div><div class="detail-block"><h4>邻接关系</h4><div class="tag-row">${outgoing.map((edge) => `<span class="tag"><span class="swatch" style="background:${getEdgeColor(edge.edge_type)}"></span>出边 ${escapeHtml(edge.edge_type)}</span>`).join("") || "<span class='muted'>暂无出边</span>"}</div><div class="tag-row" style="margin-top:10px">${incoming.map((edge) => `<span class="tag"><span class="swatch" style="background:${getEdgeColor(edge.edge_type)}"></span>入边 ${escapeHtml(edge.edge_type)}</span>`).join("") || "<span class='muted'>暂无入边</span>"}</div></div>`;
}

function getMachineCalendarMap() {
  return new Map((state.instanceDetails?.machines || []).map((machine) => [machine.id, machine]));
}

function getOperationDetailMap() {
  const map = new Map();
  (state.instanceDetails?.orders || []).forEach((order) => {
    (order.tasks || []).forEach((task) => {
      (task.ops || []).forEach((op) => {
        map.set(op.id, {
          order_id: order.id,
          order_name: order.name,
          task_id: task.id,
          task_name: task.name,
          is_main: task.is_main,
          due_at: task.due_at,
          op_name: op.name,
        });
      });
    });
  });
  return map;
}

function getInstanceOperationDbMap() {
  return new Map((state.instanceDb?.operations || []).map((op) => [op.op_id, op]));
}

function scheduleStatusMeta(status) {
  const normalized = String(status || "scheduled").toLowerCase();
  if (normalized === "completed") return { key: "completed", label: "已完成", fill: "#4b7bec", stroke: "#2450b2", text: "#ffffff" };
  if (normalized === "in_progress") return { key: "in_progress", label: "进行中", fill: "#c7681f", stroke: "#8f4a13", text: "#ffffff" };
  return { key: "scheduled", label: "未来排产", fill: "#1c7c54", stroke: "#14573b", text: "#ffffff" };
}

function buildCompletedInitialEntries() {
  const opMap = getOperationDetailMap();
  const dbMap = getInstanceOperationDbMap();
  const machineMap = getMachineCalendarMap();
  const rows = [];
  dbMap.forEach((row, opId) => {
    if (String(row.initial_status || "").toLowerCase() !== "completed") return;
    if (row.initial_end_time === null || row.initial_end_time === undefined || row.initial_end_time === "") return;
    const detail = opMap.get(opId) || {};
    const end = Number(row.initial_end_time);
    const processing = Number(row.processing_time || 0);
    const start = row.initial_start_time !== null && row.initial_start_time !== undefined && row.initial_start_time !== ""
      ? Number(row.initial_start_time)
      : Math.max(0, end - (Number.isFinite(processing) ? processing : 0));
    const machineId = row.initial_assigned_machine_id || "";
    const machine = machineMap.get(machineId);
    rows.push({
      order_id: detail.order_id || row.order_id,
      order_name: detail.order_name,
      task_id: row.task_id,
      op_id: opId,
      op_name: detail.op_name || row.op_name,
      machine_id: machineId || null,
      machine_name: machine?.name || machineId || "Unknown",
      tooling_ids: asArray(row.initial_assigned_tooling_ids),
      personnel_ids: asArray(row.initial_assigned_personnel_ids),
      start,
      end,
      start_at: offsetToDate(start) ? formatTimelineLabel(start, false) : formatNumber(start, 2),
      end_at: offsetToDate(end) ? formatTimelineLabel(end, false) : formatNumber(end, 2),
      duration: Number.isFinite(processing) ? processing : Math.max(0, end - start),
      elapsed_duration: Math.max(0, end - start),
      is_main: Boolean(detail.is_main),
      due_at: detail.due_at || null,
      status: "completed",
      status_label: "已完成",
    });
  });
  return rows;
}

function scheduleForDisplay(schedule) {
  const entries = (schedule || []).map((entry) => {
    const status = entry.status || "scheduled";
    return { ...entry, status, status_label: entry.status_label || scheduleStatusMeta(status).label };
  });
  const presentOpIds = new Set(entries.map((entry) => entry.op_id).filter(Boolean));
  const completed = buildCompletedInitialEntries().filter((entry) => !presentOpIds.has(entry.op_id));
  return [...completed, ...entries].sort((left, right) => (Number(left.start) || 0) - (Number(right.start) || 0));
}

function clipWindow(start, end, minStart, maxEnd) {
  const clippedStart = Math.max(minStart, Number(start) || 0);
  const clippedEnd = Math.min(maxEnd, Number(end) || 0);
  return clippedEnd > clippedStart + 1e-9 ? { start: clippedStart, end: clippedEnd } : null;
}

function buildMachineUnavailableBlocks(machine, minStart, maxEnd) {
  if (!machine) return [];
  const blocks = [];
  const shiftWindows = [...(machine.shift_windows || [])]
    .map((window) => ({ start: Number(window.start) || 0, end: Number(window.end) || 0 }))
    .filter((window) => window.end > window.start)
    .sort((left, right) => left.start - right.start);
  if (shiftWindows.length) {
    let cursor = minStart;
    shiftWindows.forEach((window) => {
      if (window.end <= minStart || window.start >= maxEnd) return;
      const visible = clipWindow(window.start, window.end, minStart, maxEnd);
      if (!visible) return;
      if (cursor < visible.start - 1e-9) {
        blocks.push({ start: cursor, end: visible.start, type: "off_shift" });
      }
      cursor = Math.max(cursor, visible.end);
    });
    if (cursor < maxEnd - 1e-9) {
      blocks.push({ start: cursor, end: maxEnd, type: "off_shift" });
    }
  }
  (machine.downtimes || []).forEach((downtime) => {
    const visible = clipWindow(downtime.start, downtime.end, minStart, maxEnd);
    if (!visible) return;
    blocks.push({
      start: visible.start,
      end: visible.end,
      type: downtime.downtime_type === "planned" ? "planned_downtime" : "unplanned_downtime",
    });
  });
  return blocks.sort((left, right) => left.start - right.start);
}

function ganttBlockStyle(type) {
  if (type === "planned_downtime") return { fill: "rgba(168,92,0,0.22)", stroke: "rgba(168,92,0,0.38)", label: "计划停机" };
  if (type === "unplanned_downtime") return { fill: "rgba(179,38,30,0.22)", stroke: "rgba(179,38,30,0.42)", label: "非计划停机" };
  return { fill: "rgba(31,41,51,0.08)", stroke: "rgba(31,41,51,0.14)", label: "班次外" };
}

function renderGantt(svgId, schedule, compact = false) {
  const svg = $(svgId);
  const displaySchedule = scheduleForDisplay(schedule);
  if (!displaySchedule?.length) { svg.setAttribute("viewBox", "0 0 1200 320"); svg.innerHTML = `<text x="600" y="160" text-anchor="middle" fill="#6b7280" font-size="20">暂无排程数据</text>`; return; }
  const machineCalendarMap = getMachineCalendarMap();
  const rowsByMachine = new Map();
  displaySchedule.forEach((entry) => { const key = entry.machine_name || entry.machine_id || "Unknown"; if (!rowsByMachine.has(key)) rowsByMachine.set(key, []); rowsByMachine.get(key).push(entry); });
  const lanes = Array.from(rowsByMachine.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  const minStart = Math.min(...displaySchedule.map((entry) => Number(entry.start) || 0));
  const maxEnd = Math.max(...displaySchedule.map((entry) => Number(entry.end) || 0));
  const width = compact ? 1280 : 1480, left = 180, right = 60, top = 60, rowHeight = compact ? 42 : 58, barHeight = compact ? 24 : 30, height = Math.max(220, top + lanes.length * rowHeight + 44), scale = (width - left - right) / Math.max(maxEnd - minStart, 1);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const axis = Array.from({ length: 7 }, (_, tick) => {
    const value = minStart + ((maxEnd - minStart) * tick / 6);
    const x = left + (value - minStart) * scale;
    return `<line x1="${x}" y1="42" x2="${x}" y2="${height - 28}" stroke="rgba(155,77,23,0.08)"></line><text x="${x}" y="28" text-anchor="middle" fill="#5f6c7b" font-size="12">${escapeHtml(formatTimelineLabel(value, compact))}</text>`;
  }).join("");
  const statusLegend = ["completed", "in_progress", "scheduled"].map((status, index) => {
    const meta = scheduleStatusMeta(status);
    const x = left + index * 104;
    return `<rect x="${x}" y="10" width="14" height="14" rx="4" fill="${meta.fill}" stroke="${meta.stroke}"></rect><text x="${x + 20}" y="22" fill="#5f6c7b" font-size="11">${meta.label}</text>`;
  }).join("");
  const legendItems = ["off_shift", "planned_downtime", "unplanned_downtime"].map((type, index) => {
    const style = ganttBlockStyle(type);
    const x = width - 260 + index * 82;
    return `<rect x="${x}" y="10" width="14" height="14" rx="4" fill="${style.fill}" stroke="${style.stroke}"></rect><text x="${x + 20}" y="22" fill="#5f6c7b" font-size="11">${style.label}</text>`;
  }).join("");
  const lanesMarkup = lanes.map(([machineName, entries], laneIndex) => {
    const y = top + laneIndex * rowHeight;
    const machine = machineCalendarMap.get(entries[0]?.machine_id);
    const unavailable = buildMachineUnavailableBlocks(machine, minStart, maxEnd)
      .map((block) => {
        const x = left + (block.start - minStart) * scale;
        const w = Math.max((block.end - block.start) * scale, 2);
        const style = ganttBlockStyle(block.type);
        return `<g><rect x="${x}" y="${y - rowHeight * 0.34}" width="${w}" height="${rowHeight * 0.68}" rx="8" fill="${style.fill}" stroke="${style.stroke}" stroke-width="1"></rect><title>${escapeHtml(`${style.label}: ${formatTimelineRange(block.start, block.end)}`)}</title></g>`;
      })
      .join("");
    const bars = entries.sort((a, b) => a.start - b.start).map((entry) => {
      const x = left + (entry.start - minStart) * scale;
      const w = Math.max((entry.end - entry.start) * scale, 6);
      const label = entry.order_name ? `${entry.order_name}/${entry.op_id}` : entry.op_id;
      const stateMeta = scheduleStatusMeta(entry.status);
      const title = `${label}\n状态: ${entry.status_label || stateMeta.label}\n${formatTimelineRange(entry.start, entry.end)}\n设备: ${entry.machine_name || entry.machine_id || "-"}`;
      const stroke = entry.is_tardy ? "#b3261e" : stateMeta.stroke;
      return `<g><rect x="${x}" y="${y - barHeight / 2}" width="${w}" height="${barHeight}" rx="8" fill="${stateMeta.fill}" stroke="${stroke}" stroke-width="1.4" opacity="0.86"></rect><text x="${x + Math.min(w / 2, 120)}" y="${y + 4}" text-anchor="middle" fill="${stateMeta.text}" font-size="11" font-weight="700">${escapeHtml(String(label).slice(0, Math.max(8, Math.floor(w / 8))))}</text><title>${escapeHtml(title)}</title></g>`;
    }).join("");
    return `<text x="${left - 14}" y="${y + 4}" text-anchor="end" fill="#1f2933" font-size="12" font-weight="700">${escapeHtml(machineName)}</text><line x1="${left}" y1="${y + rowHeight / 2}" x2="${width - right}" y2="${y + rowHeight / 2}" stroke="rgba(155,77,23,0.05)"></line>${unavailable}${bars}`;
  }).join("");
  svg.innerHTML = `${axis}${statusLegend}${legendItems}${lanesMarkup}`;
}

function renderSimulation() {
  if (!state.simResult) { $("simulate-metrics").innerHTML = `<div class="empty-state">请先运行一次仿真。</div>`; $("simulate-table-wrap").innerHTML = `<div class="empty-state">暂无仿真明细</div>`; renderGantt("simulate-gantt", []); return; }
  renderMetricCards("simulate-metrics", state.simResult.metrics, ["makespan", "total_tardiness", "avg_tardiness", "max_tardiness", "tardy_job_count", "avg_flowtime", "total_wait_time", "avg_utilization", "avg_active_window_utilization", "avg_net_available_utilization", "critical_utilization", "critical_active_window_utilization", "critical_net_available_utilization"]);
  renderGantt("simulate-gantt", state.simResult.gantt);
  const displaySchedule = scheduleForDisplay(state.simResult.gantt || []);
  $("simulate-table-wrap").innerHTML = `<table><thead><tr><th>状态</th><th>订单</th><th>任务</th><th>工序</th><th>设备</th><th>工装</th><th>人员</th><th>开始</th><th>结束</th><th>持续(h)</th><th>超期</th></tr></thead><tbody>${displaySchedule.map((entry) => `<tr><td><span class="tag"><span class="swatch" style="background:${scheduleStatusMeta(entry.status).fill}"></span>${escapeHtml(entry.status_label || scheduleStatusMeta(entry.status).label)}</span></td><td>${escapeHtml(entry.order_name || entry.order_id || "-")}</td><td>${escapeHtml(entry.task_id)}</td><td>${escapeHtml(entry.op_id)}</td><td>${escapeHtml(entry.machine_name || entry.machine_id || "-")}</td><td>${escapeHtml(asArray(entry.tooling_ids).join(", ") || "-")}</td><td>${escapeHtml(asArray(entry.personnel_ids).join(", ") || "-")}</td><td>${escapeHtml(entry.start_at || formatNumber(entry.start, 2))}</td><td>${escapeHtml(entry.end_at || formatNumber(entry.end, 2))}</td><td>${escapeHtml(formatNumber(entry.duration, 2))}</td><td>${entry.is_tardy ? `<span class="pill danger">是</span>` : `<span class="pill ok">否</span>`}</td></tr>`).join("")}</tbody></table>`;
}

function renderObjectives() {
  const items = state.objectiveCatalog || [];
  const selectedList = Array.from(state.selectedObjectiveKeys);
  const summaryNode = $("objective-selection-summary");
  if (summaryNode) {
    summaryNode.textContent = items.length
      ? `已选择 ${selectedList.length} / 5 个目标${selectedList.length ? `：${selectedList.join("、")}` : "。"}`
      : "暂无可选指标。";
  }
  $("objective-grid").innerHTML = items.length ? items.map((item) => {
    const selected = state.selectedObjectiveKeys.has(item.key);
    const disabled = item.available === false;
    return `<div class="objective-card ${selected ? "selected" : ""}">
      <header>
        <div>
          <h4>${escapeHtml(item.label)}</h4>
          <div class="provider">${escapeHtml(item.key)}</div>
        </div>
        <span class="pill ${selected ? "ok" : "warn"}">${selected ? "已选择" : "未选择"}</span>
      </header>
      <p>${escapeHtml(item.description || "")}</p>
      <div class="tag-row" style="margin-top:10px">
        <span class="tag">${item.direction === "max" ? "最大化" : "最小化"}</span>
        ${disabled ? `<span class="tag">当前不可用</span>` : `<span class="tag">可选</span>`}
      </div>
      <div class="btn-row" style="margin-top:14px">
        <button class="mini-btn" type="button" data-action="toggle-objective" data-objective-key="${escapeHtml(item.key)}" ${disabled ? "disabled" : ""}>${selected ? "取消选择" : "选择目标"}</button>
      </div>
    </div>`;
  }).join("") : `<div class="empty-state">暂无指标定义</div>`;
}

function renderOptimizeStatus() {
  const status = state.optimizeStatus;
  if (!status) {
    $("opt-status-pill").className = "pill";
    $("opt-status-pill").textContent = "待启动";
    $("opt-status-cards").innerHTML = `<div class="status-card"><div class="label">状态</div><div class="value">待启动</div></div>`;
    $("opt-history-table-wrap").innerHTML = `<div class="empty-state">暂无优化历史</div>`;
    $("opt-log").textContent = "等待优化任务启动...";
    return;
  }
  const label = status.status === "running" ? "运行中" : status.status === "done" ? "已完成" : status.status === "error" ? "失败" : status.status;
  $("opt-status-pill").className = `pill ${status.status === "done" ? "ok" : status.status === "error" ? "danger" : "warn"}`;
  $("opt-status-pill").textContent = label;
  $("opt-status-cards").innerHTML = [["状态", label], ["代数", status.current_generation ?? 0], ["归档规模", status.archive_size ?? 0], ["粗筛池", status.coarse_pool_size ?? 0], ["可行率", `${formatNumber((status.feasible_ratio || 0) * 100, 1)}%`], ["超体积", formatNumber(status.hypervolume || 0, 4)], ["耗时(秒)", formatNumber(status.elapsed_s || 0, 2)], ["总评估", status.total_evaluations ?? 0], ["近似评估", status.approximate_evaluations ?? 0], ["精确评估", status.exact_evaluations ?? 0], ["瓶颈设备", (status.bottleneck_machine_ids || []).join(", ") || "-"]].map(([k, v]) => `<div class="status-card"><div class="label">${escapeHtml(k)}</div><div class="value">${escapeHtml(formatValue(v))}</div></div>`).join("");
  const history = status.history || [];
  $("opt-history-table-wrap").innerHTML = history.length ? `<table><thead><tr><th>代数</th><th>归档规模</th><th>可行率</th><th>超体积</th><th>耗时(s)</th><th>评估次数</th></tr></thead><tbody>${history.map((row) => `<tr><td>${row.generation}</td><td>${row.archive_size}</td><td>${formatNumber((row.feasible_ratio || 0) * 100, 1)}%</td><td>${formatNumber(row.hypervolume || 0, 4)}</td><td>${formatNumber(row.elapsed_s || 0, 2)}</td><td>${row.total_evaluations || 0}</td></tr>`).join("")}</tbody></table>` : `<div class="empty-state">暂无优化历史</div>`;
  const lines = [`task_id: ${state.optimizeTaskId || "-"}`, `status: ${status.status}`, `current_generation: ${status.current_generation ?? 0}`, `archive_size: ${status.archive_size ?? 0}`, `coarse_pool_size: ${status.coarse_pool_size ?? 0}`, `approximate_evaluations: ${status.approximate_evaluations ?? 0}`, `exact_evaluations: ${status.exact_evaluations ?? 0}`, `feasible_ratio: ${formatNumber(status.feasible_ratio || 0, 4)}`, `hypervolume: ${formatNumber(status.hypervolume || 0, 4)}`, `elapsed_s: ${formatNumber(status.elapsed_s || 0, 2)}`];
  if (status.error) lines.push(`error: ${status.error}`);
  $("opt-log").textContent = lines.join("\n");
}

function renderReferenceRuleSelector() {
  const selected = state.selectedReferenceRules;
  const holder = $("opt-reference-rules");
  if (!holder) return;
  holder.innerHTML = RULES.map((rule) => {
    const active = selected.has(rule);
    return `<button type="button" class="mini-btn ${active ? "" : "btn-lite"}" data-action="toggle-reference-rule" data-rule-name="${escapeHtml(rule)}">${active ? "已选" : "选择"} ${escapeHtml(rule)}</button>`;
  }).join("");
  if ($("opt-reference-status")) {
    $("opt-reference-status").textContent = selected.size
      ? `已选择 ${selected.size} 个启发式参考规则：${Array.from(selected).join(", ")}`
      : "尚未选择启发式参考规则，当前对比只显示基线和帕累托解。";
  }
}

function exactObjectiveLabel(key) {
  const spec = (state.exactObjectiveCatalog || []).find((item) => item.key === key);
  return spec?.label || metricLabel(key);
}

function metricLabel(key) {
  const spec = (state.objectiveCatalog || []).find((item) => item.key === key) || (state.exactObjectiveCatalog || []).find((item) => item.key === key);
  return spec?.label || key;
}

function solutionMetricValue(solution, key) {
  if (!solution) return undefined;
  if (solution.objectives && solution.objectives[key] !== undefined) return solution.objectives[key];
  if (solution.metrics && solution.metrics[key] !== undefined) return solution.metrics[key];
  if (solution.summary && solution.summary[key] !== undefined) return solution.summary[key];
  return undefined;
}

function exactReferenceTitle(solution) {
  const info = solution?.exact_info || {};
  if (info.mode === "single") {
    return `精确单目标 · ${exactObjectiveLabel(info.objective_key || "makespan")}`;
  }
  const parts = Object.entries(info.objective_weights || {})
    .filter(([, value]) => Number(value) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 3)
    .map(([key, value]) => `${exactObjectiveLabel(key)}×${formatNumber(value, 2)}`);
  return `精确加权 · ${parts.join(" / ") || solution.solution_id}`;
}

function setExactModeVisibility() {
  const mode = $("opt-exact-mode")?.value || "single";
  const singleField = $("opt-exact-single-objective")?.closest(".field");
  const weightedGrid = $("opt-exact-weighted-grid");
  if (singleField) singleField.style.display = mode === "single" ? "" : "none";
  if (weightedGrid) weightedGrid.style.opacity = mode === "weighted" ? "1" : "0.7";
}

function renderExactObjectiveControls() {
  const catalog = state.exactObjectiveCatalog || [];
  const singleSelect = $("opt-exact-single-objective");
  if (singleSelect) {
    const current = singleSelect.value || "makespan";
    singleSelect.innerHTML = catalog.length
      ? catalog.map((item) => `<option value="${escapeHtml(item.key)}">${escapeHtml(item.label)} · ${escapeHtml(item.support_mode === "direct" ? "精确" : "代理")}</option>`).join("")
      : `<option value="makespan">Makespan</option>`;
    if (catalog.some((item) => item.key === current)) singleSelect.value = current;
  }
  const grid = $("opt-exact-weighted-grid");
  if (grid) {
    grid.innerHTML = catalog.length
      ? catalog.map((item) => `
        <div class="field">
          <label for="opt-exact-weight-${escapeHtml(item.key)}">${escapeHtml(item.label)}</label>
          <input id="opt-exact-weight-${escapeHtml(item.key)}" data-exact-weight-key="${escapeHtml(item.key)}" type="number" min="0" step="0.05" value="${state.selectedObjectiveKeys.has(item.key) ? "1" : "0"}">
          <div class="subtle">${escapeHtml(item.support_mode === "direct" ? "精确目标" : `代理到 ${item.solver_key}`)}</div>
        </div>
      `).join("")
      : `<div class="empty-state">暂无可用精确目标。</div>`;
  }
  setExactModeVisibility();
}

function renderExactReferenceList() {
  const wrap = $("opt-exact-reference-wrap");
  if (!wrap) return;
  const solutions = state.exactReferenceSolutions || [];
  if (!solutions.length) {
    wrap.innerHTML = `<div class="empty-state">尚未生成精确参考方案。</div>`;
    if ($("opt-exact-status")) $("opt-exact-status").textContent = "可把精确参考方案加入最终方案比较和 AI 决策。";
    return;
  }
  wrap.innerHTML = `<table><thead><tr><th>纳入比较</th><th>方案ID</th><th>模式</th><th>目标说明</th><th>求解状态</th><th>求解耗时</th><th>关键指标</th></tr></thead><tbody>${solutions.map((solution) => {
    const selected = state.selectedExactReferenceIds.has(solution.solution_id);
    const info = solution.exact_info || {};
    const objectiveText = info.mode === "single"
      ? exactObjectiveLabel(info.objective_key || "makespan")
      : Object.entries(info.objective_weights || {}).filter(([, value]) => Number(value) > 0).map(([key, value]) => `${exactObjectiveLabel(key)}×${formatNumber(value, 2)}`).join(" / ");
    const metrics = [
      `makespan ${formatValue(solution.objectives?.makespan)}`,
      `延误 ${formatValue(solution.objectives?.total_tardiness)}`,
      solution.objectives?.weighted_score !== undefined ? `weighted ${formatValue(solution.objectives?.weighted_score)}` : "",
    ].filter(Boolean).join(" | ");
    return `<tr>
      <td><button type="button" class="mini-btn ${selected ? "" : "btn-lite"}" data-action="toggle-exact-reference" data-solution-id="${escapeHtml(solution.solution_id)}">${selected ? "已选" : "选择"}</button></td>
      <td><code>${escapeHtml(solution.solution_id)}</code></td>
      <td>${escapeHtml(info.mode === "weighted" ? "加权" : "单目标")}</td>
      <td>${escapeHtml(objectiveText || "-")}</td>
      <td>${escapeHtml(info.status || solution.metrics?.status || "-")}</td>
      <td>${escapeHtml(formatValue(info.solve_time_s || solution.metrics?.solve_time_s))}s</td>
      <td>${escapeHtml(metrics || "-")}</td>
    </tr>`;
  }).join("")}</tbody></table>`;
  if ($("opt-exact-status")) {
    const selected = Array.from(state.selectedExactReferenceIds);
    $("opt-exact-status").textContent = selected.length
      ? `已选择 ${selected.length} 个精确参考方案：${selected.join(", ")}`
      : `已生成 ${solutions.length} 个精确参考方案，可选择加入方案对比与 AI 决策。`;
  }
}

function getSelectedExactReferenceIds(limit = 6) {
  return (state.exactReferenceSolutions || [])
    .filter((solution) => state.selectedExactReferenceIds.has(solution.solution_id))
    .map((solution) => solution.solution_id)
    .slice(0, limit);
}

function getSelectedCompareSolutionIds(limit = 6) {
  return Array.from(new Set([...getCheckedSolutionIds(limit), ...getSelectedExactReferenceIds(limit)])).slice(0, limit);
}

function collectExactObjectiveWeights() {
  const weights = {};
  document.querySelectorAll("[data-exact-weight-key]").forEach((input) => {
    const key = input.dataset.exactWeightKey;
    const value = Number(input.value || 0);
    if (key && Number.isFinite(value) && value > 0) weights[key] = value;
  });
  return weights;
}

function fillExactWeightedFromSelected() {
  const active = Array.from(state.selectedObjectiveKeys).filter((key) => (state.exactObjectiveCatalog || []).some((item) => item.key === key));
  document.querySelectorAll("[data-exact-weight-key]").forEach((input) => {
    input.value = active.includes(input.dataset.exactWeightKey) ? "1" : "0";
  });
  $("opt-exact-mode").value = "weighted";
  setExactModeVisibility();
  const text = active.length ? active.map((key) => exactObjectiveLabel(key)).join(", ") : "暂无可直接映射的已选目标";
  if ($("opt-exact-status")) $("opt-exact-status").textContent = `已按当前优化目标等权填充：${text}`;
}

async function loadExactObjectiveCatalog() {
  const response = await api("/api/exact/objectives");
  state.exactObjectiveCatalog = response.objectives || [];
  renderExactObjectiveControls();
}

async function generateExactReferenceSolution() {
  if (!state.optimizeTaskId || !state.optimizeResult) throw new Error("请先完成一次混合优化");
  const mode = $("opt-exact-mode")?.value || "single";
  const payload = {
    task_id: state.optimizeTaskId,
    mode,
    time_limit_s: Number($("opt-exact-time-limit")?.value || 60),
  };
  if (mode === "single") {
    payload.objective_key = $("opt-exact-single-objective")?.value || "makespan";
  } else {
    payload.objective_weights = collectExactObjectiveWeights();
    if (!Object.keys(payload.objective_weights).length) throw new Error("请至少为一个精确目标设置大于 0 的权重");
  }
  if ($("opt-exact-status")) $("opt-exact-status").textContent = "正在求解精确冠军参考方案，请稍候…";
  const response = await api("/api/optimize/exact-reference", { method: "POST", json: payload });
  const latest = response.solution;
  const next = new Map((state.exactReferenceSolutions || []).map((item) => [item.solution_id, item]));
  next.set(latest.solution_id, latest);
  state.exactReferenceSolutions = Array.from(next.values());
  state.selectedExactReferenceIds.add(latest.solution_id);
  if (state.optimizeResult) state.optimizeResult.reference_solutions = state.exactReferenceSolutions;
  renderExactReferenceList();
  refreshCompare();
  renderParetoAssistant();
  toast(`已生成精确参考方案 ${latest.solution_id}`, "success");
}

function toggleExactReference(solutionId) {
  if (!solutionId) return;
  if (state.selectedExactReferenceIds.has(solutionId)) {
    state.selectedExactReferenceIds.delete(solutionId);
  } else {
    state.selectedExactReferenceIds.add(solutionId);
  }
  renderExactReferenceList();
  refreshCompare();
  renderParetoAssistant();
}

async function loadReferenceRuleSolutions() {
  if (!state.optimizeResult || !state.selectedReferenceRules.size) {
    state.referenceRuleSolutions = [];
    return;
  }
  const response = await api("/api/simulate/reference-solutions", {
    method: "POST",
    json: {
      rule_names: Array.from(state.selectedReferenceRules),
      objective_keys: state.optimizeResult.objective_keys || [],
    },
  });
  state.referenceRuleSolutions = response.solutions || [];
}

function renderOptimizeResult() {
  const result = state.optimizeResult;
  if (!result) {
    $("opt-baseline-wrap").innerHTML = `<div class="empty-state">暂无基线方案</div>`;
    $("opt-solutions-table-wrap").innerHTML = `<div class="empty-state">暂无优化结果</div>`;
    $("opt-compare-wrap").innerHTML = `<div class="empty-state">先勾选至少一个优化方案。</div>`;
    state.referenceRuleSolutions = [];
    state.exactReferenceSolutions = [];
    state.selectedExactReferenceIds = new Set();
    renderReferenceRuleSelector();
    renderExactObjectiveControls();
    renderExactReferenceList();
    renderParetoAssistant();
    return;
  }
  state.exactReferenceSolutions = result.reference_solutions || state.exactReferenceSolutions || [];
  const baseline = result.baseline;
  $("opt-baseline-wrap").innerHTML = `<div class="section-stack"><div class="tag-row"><span class="tag">基线规则 ${escapeHtml(baseline.rule_name || "-")}</span><span class="tag">方案ID ${escapeHtml(baseline.solution_id || "-")}</span></div><div class="metrics-grid">${Object.entries(baseline.objectives || {}).map(([k, v]) => `<div class="metric-card"><div class="label">${escapeHtml(k)}</div><div class="value">${escapeHtml(formatValue(v))}</div></div>`).join("")}</div><div class="footer-note">已请求 ${result.requested_solution_count} 个方案，当前返回 ${result.found_solution_count} 个非支配方案。</div></div>`;
  const solutions = result.solutions || [];
  $("opt-solutions-table-wrap").innerHTML = solutions.length ? `<table><thead><tr><th>对比</th><th>方案ID</th><th>来源</th><th>Rank</th><th>可行</th>${(result.objective_keys || []).map((key) => `<th>${escapeHtml(key)}</th>`).join("")}<th>关键摘要</th></tr></thead><tbody>${solutions.map((solution) => `<tr data-solution-id="${escapeHtml(solution.solution_id)}"><td><input type="checkbox" class="solution-compare" value="${escapeHtml(solution.solution_id)}"></td><td><code>${escapeHtml(solution.solution_id)}</code></td><td>${escapeHtml(solution.source || "-")}</td><td>${escapeHtml(formatValue(solution.rank))}</td><td>${solution.feasible ? `<span class="pill ok">可行</span>` : `<span class="pill danger">不可行</span>`}</td>${(result.objective_keys || []).map((key) => `<td>${escapeHtml(formatValue(solution.objectives?.[key]))}</td>`).join("")}<td>${escapeHtml((solution.summary?.bottleneck_machine_ids || []).join(", ") || "-")}</td></tr>`).join("")}</tbody></table>` : `<div class="empty-state">在当前预算内还没有找到非支配方案。</div>`;
  renderReferenceRuleSelector();
  renderExactReferenceList();
  loadReferenceRuleSolutions()
    .then(() => { refreshCompare(); renderParetoAssistant(); })
    .catch((error) => { state.referenceRuleSolutions = []; refreshCompare(); renderParetoAssistant(); toast(`加载启发式参考失败: ${error.message}`, "warn"); });
}

function refreshCompare() {
  const result = state.optimizeResult;
  if (!result) { $("opt-compare-wrap").innerHTML = `<div class="empty-state">先勾选至少一个优化方案。</div>`; return; }
  const selectedIds = Array.from(document.querySelectorAll(".solution-compare:checked")).map((item) => item.value).slice(0, 4);
  const selected = (result.solutions || []).filter((item) => selectedIds.includes(item.solution_id));
  const referenceRules = state.referenceRuleSolutions || [];
  const exactReferences = (state.exactReferenceSolutions || []).filter((item) => state.selectedExactReferenceIds.has(item.solution_id));
  if (!selected.length && !referenceRules.length && !exactReferences.length) { $("opt-compare-wrap").innerHTML = `<div class="empty-state">先勾选至少一个优化方案，或启用启发式规则 / 精确冠军参考方案。</div>`; return; }
  const baseline = result.baseline;
  const compareTargets = [...referenceRules, ...exactReferences, ...selected];
  const compareMetricKeys = [];
  const priorityMetricKeys = [
    ...(result.objective_keys || []),
    "avg_utilization",
    "avg_active_window_utilization",
    "avg_net_available_utilization",
    "critical_utilization",
    "critical_active_window_utilization",
    "critical_net_available_utilization",
    "total_wait_time",
    "avg_flowtime",
    "weighted_score",
  ];
  const allSolutions = [baseline, ...compareTargets];
  priorityMetricKeys.forEach((key) => {
    if (!compareMetricKeys.includes(key) && allSolutions.some((item) => solutionMetricValue(item, key) !== undefined)) compareMetricKeys.push(key);
  });
  allSolutions.forEach((item) => {
    [item.objectives || {}, item.metrics || {}, item.summary || {}].forEach((bucket) => {
      Object.keys(bucket).forEach((key) => {
        if (!compareMetricKeys.includes(key) && solutionMetricValue(item, key) !== undefined && key !== "completed_operations" && key !== "total_operations") compareMetricKeys.push(key);
      });
    });
  });
  $("opt-compare-wrap").innerHTML = `<div class="table-wrap"><table><thead><tr><th>指标</th><th>基线</th>${compareTargets.map((item) => `<th>${escapeHtml(item.solution_id)}</th>`).join("")}</tr></thead><tbody>${compareMetricKeys.map((key) => `<tr><td>${escapeHtml(metricLabel(key))}<div class="provider">${escapeHtml(key)}</div></td><td>${escapeHtml(formatValue(solutionMetricValue(baseline, key)))}</td>${compareTargets.map((item) => `<td>${escapeHtml(formatValue(solutionMetricValue(item, key)))}</td>`).join("")}</tr>`).join("")}</tbody></table></div><div class="grid-2" id="compare-cards"></div>`;
  const all = [
    { title: `基线方案 · ${baseline.rule_name || "-"}`, summary: baseline.summary || {}, schedule: baseline.schedule || [] },
    ...referenceRules.map((solution) => ({ title: `启发式 · ${solution.rule_name || solution.solution_id}`, summary: solution.summary || {}, schedule: solution.schedule || [] })),
    ...exactReferences.map((solution) => ({ title: exactReferenceTitle(solution), summary: solution.summary || {}, schedule: solution.schedule || [] })),
    ...selected.map((solution) => ({ title: solution.solution_id, summary: solution.summary || {}, schedule: solution.schedule || [] })),
  ];
  $("compare-cards").innerHTML = all.map((card, idx) => `<div class="compare-card"><h4>${escapeHtml(card.title)}</h4><p>${escapeHtml((card.summary?.bottleneck_machine_ids || []).join(", ") || "暂无瓶颈摘要")}</p><div class="tag-row" style="margin-top:10px">${Object.entries(card.summary || {}).slice(0, 4).map(([k, v]) => `<span class="tag">${escapeHtml(k)}: ${escapeHtml(formatValue(v))}</span>`).join("")}</div><div class="gantt-stage" style="margin-top:14px"><svg id="compare-gantt-${idx}" viewBox="0 0 1200 260"></svg></div></div>`).join("");
  all.forEach((card, idx) => renderGantt(`compare-gantt-${idx}`, card.schedule || [], true));
}

function getCheckedSolutionIds(limit = 4) {
  return Array.from(document.querySelectorAll(".solution-compare:checked")).map((item) => item.value).slice(0, limit);
}

function updateParetoSelectionHint() {
  const selectedIds = getCheckedSolutionIds(6);
  const referenceRules = Array.from(state.selectedReferenceRules);
  const exactRefs = getSelectedExactReferenceIds(6);
  if ($("pareto-selected-hint")) {
    const parts = [];
    if (selectedIds.length) parts.push(`方案: ${selectedIds.join(", ")}`);
    if (exactRefs.length) parts.push(`精确: ${exactRefs.join(", ")}`);
    if (referenceRules.length) parts.push(`规则: ${referenceRules.join(", ")}`);
    $("pareto-selected-hint").value = parts.length ? parts.join(" | ") : "尚未勾选方案";
  }
}

function renderParetoAssistant() {
  const result = state.optimizeResult;
  const assistant = state.paretoAssistant;
  const select = $("pareto-solution-select");
  if (select) {
    const paretoOptions = (result?.solutions || []).map((solution) => `<option value="${escapeHtml(solution.solution_id)}">${escapeHtml(solution.solution_id)} · ${escapeHtml(solution.source || "-")}</option>`).join("");
    const heuristicOptions = (state.referenceRuleSolutions || []).map((solution) => `<option value="${escapeHtml(solution.solution_id)}">${escapeHtml(solution.solution_id)} · 启发式规则</option>`).join("");
    const exactOptions = (state.exactReferenceSolutions || []).map((solution) => `<option value="${escapeHtml(solution.solution_id)}">${escapeHtml(solution.solution_id)} · ${escapeHtml(exactReferenceTitle(solution))}</option>`).join("");
    const options = `${paretoOptions}${heuristicOptions}${exactOptions}`;
    select.innerHTML = options ? `<option value="">请选择一个方案</option>${options}` : `<option value="">请先完成优化</option>`;
    const knownIds = new Set([...(result?.solutions || []).map((item) => item.solution_id), ...(state.referenceRuleSolutions || []).map((item) => item.solution_id), ...(state.exactReferenceSolutions || []).map((item) => item.solution_id)]);
    if (assistant?.analysis?.solution_id && knownIds.has(assistant.analysis.solution_id)) {
      select.value = assistant.analysis.solution_id;
    }
  }
  updateParetoSelectionHint();
  $("pareto-ai-meta").innerHTML = assistant ? [
    `<span class="tag">模式 ${escapeHtml(assistant.mode || "-")}</span>`,
    `<span class="tag">模型 ${escapeHtml(assistant.used_model || "-")}</span>`,
    `<span class="tag">任务 ${escapeHtml(assistant.task_id || state.optimizeTaskId || "-")}</span>`,
    assistant.analysis?.recommended_solution_id ? `<span class="tag">推荐 ${escapeHtml(assistant.analysis.recommended_solution_id)}</span>` : "",
    state.paretoPending ? `<span class="tag">处理中</span>` : "",
  ].filter(Boolean).join("") : `${state.paretoPending ? `<span class="tag">处理中</span>` : `<span class="tag">等待方案结果</span>`}`;
  const historyMarkup = state.paretoConversation.length ? state.paretoConversation.slice(-12).map((item) => `<div class="chat-item ${escapeHtml(item.role)}"><div class="role">${item.role === "assistant" ? "AI" : "User"}</div><div class="content">${escapeHtml(item.content)}</div></div>`).join("") : `<div class="empty-state">请先完成一次混合优化，然后勾选方案或输入业务诉求开始交流。</div>`;
  const pendingMarkup = state.paretoPending ? `<div class="chat-item assistant pending"><div class="role">AI</div><div class="content">${escapeHtml(state.paretoPending)}<span class="thinking-dots"><span></span><span></span><span></span></span></div></div>` : "";
  $("pareto-chat-history").innerHTML = `${historyMarkup}${pendingMarkup}`;
  const historyNode = $("pareto-chat-history");
  if (historyNode) historyNode.scrollTop = historyNode.scrollHeight;
}

function rememberParetoMessage(role, content) {
  const text = String(content || "").trim();
  if (!text) return;
  state.paretoConversation.push({ role, content: text });
  state.paretoConversation = state.paretoConversation.slice(-12);
}

function paretoConversationPayload() {
  return state.paretoConversation.slice(-8).map((item) => ({ role: item.role, content: item.content }));
}

async function toggleReferenceRule(ruleName) {
  if (state.selectedReferenceRules.has(ruleName)) {
    state.selectedReferenceRules.delete(ruleName);
  } else {
    if (state.selectedReferenceRules.size >= REFERENCE_RULE_LIMIT) {
      toast(`最多选择 ${REFERENCE_RULE_LIMIT} 个启发式参考规则`, "warn");
      return;
    }
    state.selectedReferenceRules.add(ruleName);
  }
  renderReferenceRuleSelector();
  if (state.optimizeResult) {
    await loadReferenceRuleSolutions();
    refreshCompare();
    renderParetoAssistant();
  }
}

function resolveExportFallbackReferenceId() {
  const exactIds = getSelectedExactReferenceIds(4);
  if (exactIds.length === 1) return exactIds[0];
  if (exactIds.length === 0 && state.selectedReferenceRules.size === 1) {
    return `RULE:${Array.from(state.selectedReferenceRules)[0]}`;
  }
  return null;
}

function resolveExportSolutionId() {
  const checked = getCheckedSolutionIds(4);
  if (checked.length === 1) return checked[0];
  const currentAsk = $("pareto-solution-select")?.value;
  if (currentAsk) return currentAsk;
  const recommended = state.paretoAssistant?.analysis?.recommended_solution_id;
  if (recommended) return recommended;
  const fallbackReference = resolveExportFallbackReferenceId();
  if (fallbackReference) return fallbackReference;
  throw new Error("请先明确选择一个要导出的方案，可以勾选单个方案，或在追问方案下拉框中指定。");
}

async function exportSelectedSolution() {
  if (!state.optimizeTaskId || !state.optimizeResult) throw new Error("请先完成一次混合优化");
  const solutionId = resolveExportSolutionId();
  const filename = await downloadApiBlob(
    "/api/optimize/hybrid/export-solution",
    { task_id: state.optimizeTaskId, solution_id: solutionId },
    `solution_export_${Date.now()}.xlsx`,
  );
  toast(`方案已导出: ${filename}`, "success");
}

async function compareParetoSolutions() {
  if (state.paretoPending) { toast("AI 还在处理中，请稍候", "warn"); return; }
  if (!state.optimizeTaskId || !state.optimizeResult) { toast("请先完成一次混合优化", "warn"); return; }
  const selectedIds = getSelectedCompareSolutionIds(6);
  const heuristicRuleNames = Array.from(state.selectedReferenceRules);
  if (selectedIds.length + heuristicRuleNames.length < 2) { toast("请至少选择 2 个候选（帕累托方案、精确参考方案或启发式规则）再做比较", "warn"); return; }
  const requirement = $("pareto-message").value.trim();
  rememberParetoMessage("user", `比较候选: ${[...selectedIds, ...heuristicRuleNames.map((name) => `RULE:${name}`)].join(", ")}${requirement ? `\n诉求: ${requirement}` : ""}`);
  state.paretoPending = `正在比较 ${selectedIds.length + heuristicRuleNames.length} 个候选`;
  $("pareto-ai-status").value = "AI 正在分析已勾选方案，请稍候…";
  renderParetoAssistant();
  try {
    const response = await api("/api/ai/pareto/compare", {
      method: "POST",
      json: {
        task_id: state.optimizeTaskId,
        solution_ids: selectedIds,
        heuristic_rule_names: heuristicRuleNames,
        requirement,
        conversation: paretoConversationPayload(),
      },
    });
    state.paretoAssistant = response;
    rememberParetoMessage("assistant", response.display_text || "");
    $("pareto-ai-status").value = `已完成 ${selectedIds.length + heuristicRuleNames.length} 个候选的自然语言比较`;
  } finally {
    state.paretoPending = null;
    renderParetoAssistant();
  }
}

async function recommendParetoSolution() {
  if (state.paretoPending) { toast("AI 还在处理中，请稍候", "warn"); return; }
  if (!state.optimizeTaskId || !state.optimizeResult) { toast("请先完成一次混合优化", "warn"); return; }
  const requirement = $("pareto-message").value.trim();
  if (!requirement) { toast("请先输入业务诉求，再让 AI 推荐方案", "warn"); return; }
  const selectedIds = getSelectedCompareSolutionIds(6);
  rememberParetoMessage("user", `请按以下要求推荐方案:\n${requirement}`);
  state.paretoPending = "正在结合你的诉求推荐方案";
  $("pareto-ai-status").value = "AI 正在根据业务诉求推理最匹配的方案…";
  renderParetoAssistant();
  try {
    const response = await api("/api/ai/pareto/recommend", {
      method: "POST",
      json: {
        task_id: state.optimizeTaskId,
        solution_ids: selectedIds,
        heuristic_rule_names: Array.from(state.selectedReferenceRules),
        requirement,
        conversation: paretoConversationPayload(),
      },
    });
    state.paretoAssistant = response;
    rememberParetoMessage("assistant", response.display_text || "");
    $("pareto-ai-status").value = `已基于当前诉求推荐 ${response.analysis?.recommended_solution_id || "目标方案"}`;
  } finally {
    state.paretoPending = null;
    renderParetoAssistant();
  }
}

async function askParetoSolution() {
  if (state.paretoPending) { toast("AI 还在处理中，请稍候", "warn"); return; }
  if (!state.optimizeTaskId || !state.optimizeResult) { toast("请先完成一次混合优化", "warn"); return; }
  const solutionId = $("pareto-solution-select").value;
  const question = $("pareto-message").value.trim();
  if (!solutionId) { toast("请先选择一个要追问的方案", "warn"); return; }
  if (!question) { toast("请先输入你的问题", "warn"); return; }
  rememberParetoMessage("user", `关于方案 ${solutionId}：${question}`);
  state.paretoPending = `正在分析方案 ${solutionId}`;
  $("pareto-ai-status").value = `AI 正在解读方案 ${solutionId} 的规则与排程过程…`;
  renderParetoAssistant();
  try {
    const response = await api("/api/ai/pareto/ask", {
      method: "POST",
      json: {
        task_id: state.optimizeTaskId,
        solution_id: solutionId,
        heuristic_rule_names: Array.from(state.selectedReferenceRules),
        question,
        conversation: paretoConversationPayload(),
      },
    });
    state.paretoAssistant = response;
    rememberParetoMessage("assistant", response.display_text || "");
    $("pareto-ai-status").value = `已回答关于方案 ${solutionId} 的问题`;
  } finally {
    state.paretoPending = null;
    renderParetoAssistant();
  }
}

function clearParetoConversation() {
  state.paretoAssistant = null;
  state.paretoConversation = [];
  state.paretoPending = null;
  $("pareto-ai-status").value = "对话已清空，可以重新比较、推荐或追问。";
  renderParetoAssistant();
}

function renderLLMPrests() {
  $("llm-presets").innerHTML = LLM_PRESETS.map((preset, index) => `<div class="preset-card"><header><div><div class="provider">${escapeHtml(preset.provider)}</div><h4>${escapeHtml(preset.name)}</h4></div></header><p>${escapeHtml(preset.description)}</p><div class="tag-row" style="margin-top:10px"><span class="tag">${escapeHtml(preset.model || "自定义模型")}</span><span class="tag">${escapeHtml(preset.base_url || "自定义地址")}</span></div><div class="btn-row" style="margin-top:14px"><button class="mini-btn" data-action="apply-llm-preset" data-preset-index="${index}">套用预设</button></div></div>`).join("");
}

function renderLLMConfig() {
  const cfg = state.llmConfig;
  if (!cfg) { $("llm-current-cards").innerHTML = `<div class="empty-state">请先读取当前配置。</div>`; setHeroStatus(); return; }
  $("llm-base-url").value = cfg.base_url || "";
  $("llm-model").value = cfg.model || "";
  $("llm-api-key").value = "";
  $("llm-config-status").textContent = cfg.has_key ? `当前已配置 API Key，预览 ${cfg.preview || ""}` : "当前未配置 API Key";
  $("llm-current-cards").innerHTML = `<div class="metric-card"><div class="label">Base URL</div><div class="value">${escapeHtml(cfg.base_url || "-")}</div></div><div class="metric-card"><div class="label">模型</div><div class="value">${escapeHtml(cfg.model || "-")}</div></div><div class="metric-card"><div class="label">Key 状态</div><div class="value">${cfg.has_key ? "已配置" : "未配置"}</div><div class="hint">${escapeHtml(cfg.preview || "")}</div></div>`;
  setHeroStatus();
}

async function loadInstance(silent = false) {
  try {
    const [db, details, downtimes] = await Promise.all([api("/api/instance/db"), api("/api/instance/details"), api("/api/downtime").catch(() => ({ downtimes: [] }))]);
    state.instanceDb = db; state.instanceDetails = details; state.downtimes = downtimes.downtimes || [];
    renderInstance(); setHeroStatus(); if (!silent) toast("实例已刷新", "success");
  } catch (error) {
    state.instanceDb = null; state.instanceDetails = null; state.downtimes = [];
    renderInstance(); setHeroStatus(); if (!silent) toast(`未能读取实例: ${error.message}`, "warn");
  }
}

async function generateInstance() {
  await api("/api/instance/generate", { method: "POST", json: { plan_start_at: localInputToIso($("cfg-plan-start").value), num_orders: Number($("cfg-num-orders").value), tasks_per_order_min: Number($("cfg-task-min").value), tasks_per_order_max: Number($("cfg-task-max").value), ops_per_task_min: Number($("cfg-op-min").value), ops_per_task_max: Number($("cfg-op-max").value), machines_per_type: Number($("cfg-machines-per-type").value), due_date_factor: Number($("cfg-due-factor").value), arrival_spread: Number($("cfg-arrival-spread").value), seed: Number($("cfg-seed").value), day_shift_hours: Number($("cfg-day-shift-hours").value), night_shift_hours: Number($("cfg-night-shift-hours").value), schedule_days: Number($("cfg-schedule-days").value), maintenance_prob: Number($("cfg-maintenance-prob").value), toolings_per_type: Number($("cfg-toolings-per-type").value), personnel_per_skill: Number($("cfg-personnel-per-skill").value) } });
  await loadInstance(true); toast("实例生成成功", "success"); $("instance-action-status").textContent = "实例已重新生成并写入数据库。"; state.graphMeta = null; state.graphNodes = []; state.graphEdges = []; renderGraphMeta(); renderGraphVisual();
}

async function saveRow(button, kind) {
  const row = button.closest("tr"), id = row.dataset.id;
  const opSource = (state.instanceDb?.operations || []).find((item) => item.op_id === id);
  const paths = { order: `/api/instance/order/${encodeURIComponent(id)}`, task: `/api/instance/task/${encodeURIComponent(id)}`, operation: `/api/instance/operation/${encodeURIComponent(id)}`, "operation-state": `/api/instance/operation/${encodeURIComponent(id)}`, machine: `/api/instance/machine/${encodeURIComponent(id)}` };
  const payloads = {
    order: { order_name: row.querySelector('[data-field="order_name"]').value, release_at: localInputToIso(row.querySelector('[data-field="release_at"]').value), due_at: localInputToIso(row.querySelector('[data-field="due_at"]').value), priority: Number(row.querySelector('[data-field="priority"]').value || 1) },
    task: { order_id: row.querySelector('[data-field="order_id"]').value, task_name: row.querySelector('[data-field="task_name"]').value, is_main: row.querySelector('[data-field="is_main"]').value === "1", predecessor_task_ids: serializeList(row.querySelector('[data-field="predecessor_task_ids"]').value), release_at: localInputToIso(row.querySelector('[data-field="release_at"]').value), due_at: localInputToIso(row.querySelector('[data-field="due_at"]').value) },
    operation: {
      task_id: row.querySelector('[data-field="task_id"]').value,
      op_name: row.querySelector('[data-field="op_name"]').value,
      process_type: row.querySelector('[data-field="process_type"]').value,
      processing_time: Number(row.querySelector('[data-field="processing_time"]').value || 0),
      predecessor_ops: serializeList(row.querySelector('[data-field="predecessor_ops"]').value),
      predecessor_tasks: serializeList(row.querySelector('[data-field="predecessor_tasks"]').value),
      eligible_machine_ids: serializeList(row.querySelector('[data-field="eligible_machine_ids"]').value),
      required_tooling_types: serializeList(row.querySelector('[data-field="required_tooling_types"]').value),
      required_personnel_skills: serializeList(row.querySelector('[data-field="required_personnel_skills"]').value),
      initial_status: opSource?.initial_status || "",
      initial_start_time: opSource?.initial_start_time ?? null,
      initial_end_time: opSource?.initial_end_time ?? null,
      initial_remaining_processing_time: opSource?.initial_remaining_processing_time ?? null,
      initial_assigned_machine_id: opSource?.initial_assigned_machine_id || "",
      initial_assigned_tooling_ids: opSource?.initial_assigned_tooling_ids || "",
      initial_assigned_personnel_ids: opSource?.initial_assigned_personnel_ids || "",
    },
    "operation-state": {
      task_id: opSource?.task_id || "",
      op_name: opSource?.op_name || "",
      process_type: opSource?.process_type || "",
      processing_time: Number(opSource?.processing_time || 0),
      predecessor_ops: opSource?.predecessor_ops || "",
      predecessor_tasks: opSource?.predecessor_tasks || "",
      eligible_machine_ids: opSource?.eligible_machine_ids || "",
      required_tooling_types: opSource?.required_tooling_types || "",
      required_personnel_skills: opSource?.required_personnel_skills || "",
      initial_status: row.querySelector('[data-field="initial_status"]').value,
      initial_start_at: localInputToIso(row.querySelector('[data-field="initial_start_at"]').value),
      initial_end_at: localInputToIso(row.querySelector('[data-field="initial_end_at"]').value),
      initial_remaining_processing_time: row.querySelector('[data-field="initial_remaining_processing_time"]').value,
      initial_assigned_machine_id: row.querySelector('[data-field="initial_assigned_machine_id"]').value.trim(),
      initial_assigned_tooling_ids: serializeList(row.querySelector('[data-field="initial_assigned_tooling_ids"]').value),
      initial_assigned_personnel_ids: serializeList(row.querySelector('[data-field="initial_assigned_personnel_ids"]').value),
    },
    machine: { machine_name: row.querySelector('[data-field="machine_name"]').value, type_id: row.querySelector('[data-field="type_id"]').value, shifts: row.querySelector('[data-field="shifts"]').value },
  };
  await api(paths[kind], { method: "PUT", json: payloads[kind] }); await loadInstance(true); toast(`${id} 已保存`, "success");
}

async function refreshDowntime(silent = false) { const response = await api("/api/downtime"); state.downtimes = response.downtimes || []; renderDowntime(); if (!silent) toast("停机列表已刷新", "success"); }
async function addDowntime() { if (!$("downtime-machine").value) { toast("请先生成实例并选择设备", "warn"); return; } await api("/api/downtime", { method: "POST", json: { machine_id: $("downtime-machine").value, downtime_type: $("downtime-type").value, start_time: localInputToIso($("downtime-start").value), end_time: localInputToIso($("downtime-end").value) } }); await loadInstance(true); await refreshDowntime(true); toast("停机记录已新增", "success"); }
async function updateDowntimeRow(button) { const row = button.closest("tr"), id = row.dataset.id; await api(`/api/downtime/${encodeURIComponent(id)}`, { method: "PUT", json: { machine_id: row.querySelector('[data-field="machine_id"]').value, downtime_type: row.querySelector('[data-field="downtime_type"]').value, start_time: localInputToIso(row.querySelector('[data-field="start_time"]').value), end_time: localInputToIso(row.querySelector('[data-field="end_time"]').value) } }); await loadInstance(true); await refreshDowntime(true); toast(`停机 ${id} 已更新`, "success"); }
async function deleteDowntimeRow(button) { const id = button.closest("tr").dataset.id; await api(`/api/downtime/${encodeURIComponent(id)}`, { method: "DELETE" }); await loadInstance(true); await refreshDowntime(true); toast(`停机 ${id} 已删除`, "success"); }
async function importExcel(file) { const formData = new FormData(); formData.append("file", file); await api("/api/instance/import-excel", { method: "POST", body: formData }); await loadInstance(true); toast("Excel 导入成功", "success"); }
async function runSimulation() { state.simResult = await api("/api/simulate", { method: "POST", json: { rule_name: $("simulate-rule").value } }); renderSimulation(); toast(`仿真完成: ${$("simulate-rule").value}`, "success"); }

async function loadGraph() {
  try {
    state.graphNodeFilter = $("graph-node-type").value; state.graphEdgeFilter = $("graph-edge-type").value;
    const search = $("graph-search").value.trim();
    const nodeParams = new URLSearchParams({ limit: "1200" }); const edgeParams = new URLSearchParams({ limit: "2400" });
    if (state.graphNodeFilter) nodeParams.set("node_type", state.graphNodeFilter);
    if (state.graphEdgeFilter) edgeParams.set("edge_type", state.graphEdgeFilter);
    if (search) { nodeParams.set("search", search); edgeParams.set("search", search); }
    const [meta, nodes, edges] = await Promise.all([api("/api/graph/meta"), api(`/api/graph/nodes?${nodeParams.toString()}`), api(`/api/graph/edges?${edgeParams.toString()}`)]);
    state.graphMeta = meta; state.graphNodes = nodes.nodes || []; state.graphEdges = edges.edges || [];
    renderGraphMeta(); renderGraphVisual(); renderNodeDetail(); $("graph-status-line").textContent = `已加载 ${state.graphNodes.length} 个节点，${state.graphEdges.length} 条边。`; setHeroStatus();
  } catch (error) { state.graphMeta = null; state.graphNodes = []; state.graphEdges = []; renderGraphMeta(); renderGraphVisual(); renderNodeDetail(); toast(`图数据加载失败: ${error.message}`, "warn"); }
}

async function buildGraph() { await api("/api/graph/build", { method: "POST" }); await loadGraph(); toast("异构图已构建", "success"); }
async function selectGraphNode(nodeId) { state.selectedNodeId = nodeId; try { state.selectedNodeNeighbors = await api(`/api/graph/node/${encodeURIComponent(nodeId)}/neighbors`); } catch { state.selectedNodeNeighbors = null; } renderGraphVisual(); renderNodeDetail(); }
async function loadObjectiveCatalog() {
  const response = await api("/api/optimize/objectives");
  state.objectiveCatalog = response.objectives || [];
  const available = new Set(state.objectiveCatalog.filter((item) => item.available !== false).map((item) => item.key));
  state.selectedObjectiveKeys = new Set(Array.from(state.selectedObjectiveKeys).filter((key) => available.has(key)));
  if (!state.selectedObjectiveKeys.size) ["total_tardiness", "makespan", "avg_utilization"].forEach((key) => { if (available.has(key)) state.selectedObjectiveKeys.add(key); });
  renderObjectives();
}
async function loadLLM(silent = false) { state.llmConfig = await api("/api/config/llm"); renderLLMConfig(); if (!silent) toast("已读取当前大模型配置", "success"); }
async function saveLLM() { const payload = { base_url: $("llm-base-url").value.trim(), model: $("llm-model").value.trim() }; const apiKey = $("llm-api-key").value.trim(); if (apiKey) payload.api_key = apiKey; await api("/api/config/llm", { method: "PUT", json: payload }); await loadLLM(true); toast("大模型配置已保存", "success"); }
async function testLLM() { const result = await api("/api/config/llm/test", { method: "POST" }); $("llm-test-log").textContent = JSON.stringify(result, null, 2); toast(result.status === "ok" ? "模型连通性测试成功" : "模型连通性测试返回错误", result.status === "ok" ? "success" : "warn"); }
function applyPreset(index) { const preset = LLM_PRESETS[index]; $("llm-base-url").value = preset.base_url || ""; $("llm-model").value = preset.model || ""; $("llm-config-status").textContent = `已套用预设: ${preset.name}。保存后生效。`; toast(`已套用预设 ${preset.name}`, "success"); }

function getSelectedObjectiveKeys() { return Array.from(document.querySelectorAll(".objective-check:checked")).map((node) => node.value); }
function syncSelectedObjectives() { state.selectedObjectiveKeys = new Set(getSelectedObjectiveKeys()); document.querySelectorAll(".objective-card").forEach((card) => card.classList.toggle("selected", !!card.querySelector(".objective-check")?.checked)); }
function toggleObjective(key) {
  if (state.selectedObjectiveKeys.has(key)) {
    state.selectedObjectiveKeys.delete(key);
  } else {
    if (state.selectedObjectiveKeys.size >= 5) {
      toast("最多只能选择 5 个目标", "warn");
      return;
    }
    state.selectedObjectiveKeys.add(key);
  }
  renderObjectives();
}
async function startOptimization() {
  const objectiveKeys = Array.from(state.selectedObjectiveKeys);
  if (objectiveKeys.length < 1 || objectiveKeys.length > 5) { toast("请在 1 到 5 个目标之间进行选择", "warn"); return; }
  const response = await api("/api/optimize/hybrid", {
    method: "POST",
    json: {
      objective_keys: objectiveKeys,
      target_solution_count: Number($("opt-target-count").value),
      time_limit_s: Number($("opt-time-limit").value),
      population_size: Number($("opt-population").value),
      generations: Number($("opt-generations").value),
      alns_iterations_per_candidate: Number($("opt-alns").value),
      coarse_time_ratio: Number($("opt-coarse-time-ratio")?.value || 0.68),
      promotion_pool_multiplier: Number($("opt-promotion-multiplier")?.value || 3),
      random_promotion_ratio: Number($("opt-random-promotion-ratio")?.value || 0.12),
      refine_rounds: Number($("opt-refine-rounds")?.value || 1),
      alns_aggression: Number($("opt-alns-aggression")?.value || 1.0),
      seed: Number($("opt-seed").value),
      baseline_rule_name: $("opt-baseline-rule").value,
    },
  });
  state.optimizeTaskId = response.task_id;
  state.optimizeStatus = { status: "running", ...response };
  state.optimizeResult = null;
  state.referenceRuleSolutions = [];
  state.exactReferenceSolutions = [];
  state.selectedExactReferenceIds = new Set();
  state.paretoAssistant = null;
  state.paretoConversation = [];
  renderOptimizeStatus();
  renderOptimizeResult();
  setHeroStatus();
  toast(`优化任务已启动: ${response.task_id}`, "success");
  pollOptimization(response.task_id);
}
async function pollOptimization(taskId) {
  if (state.optimizePollTimer) window.clearInterval(state.optimizePollTimer);
  const tick = async () => {
    try {
      state.optimizeStatus = await api(`/api/optimize/hybrid/status/${encodeURIComponent(taskId)}`);
      renderOptimizeStatus(); setHeroStatus();
      if (state.optimizeStatus.status === "done") { window.clearInterval(state.optimizePollTimer); state.optimizePollTimer = null; state.optimizeResult = await api(`/api/optimize/hybrid/result/${encodeURIComponent(taskId)}`); renderOptimizeResult(); setHeroStatus(); toast("优化完成", "success"); }
      if (state.optimizeStatus.status === "error") { window.clearInterval(state.optimizePollTimer); state.optimizePollTimer = null; toast(`优化失败: ${state.optimizeStatus.error || "未知错误"}`, "error"); }
    } catch (error) { window.clearInterval(state.optimizePollTimer); state.optimizePollTimer = null; toast(`轮询优化任务失败: ${error.message}`, "error"); }
  };
  await tick(); state.optimizePollTimer = window.setInterval(tick, 1800);
}

function switchTab(name, updateHash = true) {
  if (!TAB_SEQUENCE.includes(name)) return;
  state.currentTab = name;
  document.querySelectorAll(".tab-btn").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.toggle("active", pane.id === `tab-${name}`));
  if (updateHash) window.history.replaceState(null, "", `#${name}`);
  document.querySelector(`.tab-btn[data-tab="${name}"]`)?.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
}

function switchRelativeTab(step) {
  const currentIndex = Math.max(0, TAB_SEQUENCE.indexOf(state.currentTab));
  const nextIndex = Math.min(TAB_SEQUENCE.length - 1, Math.max(0, currentIndex + step));
  switchTab(TAB_SEQUENCE[nextIndex]);
}

function initTabStateFromHash() {
  const hashTab = window.location.hash.replace("#", "").trim();
  switchTab(TAB_SEQUENCE.includes(hashTab) ? hashTab : state.currentTab, false);
}

function attachEvents() {
  $("tab-bar").addEventListener("click", (event) => { const button = event.target.closest(".tab-btn"); if (button) switchTab(button.dataset.tab); });
  document.body.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-action]"); if (!button) return;
    try {
      const action = button.dataset.action;
      if (action === "generate-instance") await generateInstance();
      if (action === "reload-instance") await loadInstance();
      if (action === "open-import") $("excel-file").click();
      if (action === "download-template") await downloadTemplate();
      if (action === "export-csv") window.open("/api/instance/csv", "_blank");
      if (action === "prev-tab") switchRelativeTab(-1);
      if (action === "next-tab") switchRelativeTab(1);
      if (action === "refresh-downtime") await refreshDowntime();
      if (action === "add-downtime") await addDowntime();
      if (action === "save-order") await saveRow(button, "order");
      if (action === "save-task") await saveRow(button, "task");
      if (action === "save-operation") await saveRow(button, "operation");
      if (action === "save-operation-state") await saveRow(button, "operation-state");
      if (action === "save-machine") await saveRow(button, "machine");
      if (action === "update-downtime") await updateDowntimeRow(button);
      if (action === "delete-downtime") await deleteDowntimeRow(button);
      if (action === "build-graph") await buildGraph();
      if (action === "reload-graph" || action === "apply-graph-filter") await loadGraph();
      if (action === "run-simulate") await runSimulation();
      if (action === "refresh-objectives") await loadObjectiveCatalog();
      if (action === "toggle-objective") toggleObjective(button.dataset.objectiveKey);
      if (action === "toggle-reference-rule") await toggleReferenceRule(button.dataset.ruleName);
      if (action === "generate-exact-reference") await generateExactReferenceSolution();
      if (action === "toggle-exact-reference") toggleExactReference(button.dataset.solutionId);
      if (action === "fill-exact-weighted-from-selected") fillExactWeightedFromSelected();
      if (action === "start-optimization") await startOptimization();
      if (action === "refresh-compare") refreshCompare();
      if (action === "export-selected-solution") await exportSelectedSolution();
      if (action === "pareto-compare") await compareParetoSolutions();
      if (action === "pareto-recommend") await recommendParetoSolution();
      if (action === "pareto-ask") await askParetoSolution();
      if (action === "clear-pareto-chat") clearParetoConversation();
      if (action === "load-llm") await loadLLM();
      if (action === "save-llm") await saveLLM();
      if (action === "test-llm") await testLLM();
      if (action === "apply-llm-preset") applyPreset(Number(button.dataset.presetIndex));
    } catch (error) { toast(error.message || "操作失败", "error"); }
  });
  $("excel-file").addEventListener("change", async (event) => { const file = event.target.files?.[0]; if (!file) return; try { await importExcel(file); } catch (error) { toast(`导入失败: ${error.message}`, "error"); } finally { event.target.value = ""; } });
  $("objective-grid").addEventListener("change", (event) => { if (!event.target.classList.contains("objective-check")) return; if (document.querySelectorAll(".objective-check:checked").length > 5) { event.target.checked = false; toast("最多只能选择 5 个目标", "warn"); } syncSelectedObjectives(); });
  $("opt-exact-mode")?.addEventListener("change", () => setExactModeVisibility());
  $("graph-svg").addEventListener("click", (event) => { const node = event.target.closest(".graph-node"); if (node) selectGraphNode(node.dataset.nodeId); });
  document.body.addEventListener("change", (event) => { if (event.target.classList.contains("solution-compare")) { if (document.querySelectorAll(".solution-compare:checked").length > 4) { event.target.checked = false; toast("最多同时对比 4 个方案", "warn"); return; } refreshCompare(); renderParetoAssistant(); } });
  window.addEventListener("hashchange", () => initTabStateFromHash());
  window.addEventListener("keydown", (event) => {
    const targetTag = event.target?.tagName?.toLowerCase();
    if (["input", "textarea", "select"].includes(targetTag)) return;
    if (event.altKey && event.key === "ArrowLeft") { event.preventDefault(); switchRelativeTab(-1); }
    if (event.altKey && event.key === "ArrowRight") { event.preventDefault(); switchRelativeTab(1); }
  });
}

async function init() {
  setDefaultPlanStart();
  populateRuleOptions();
  ensureOptimizeAdvancedControls();
  renderReferenceRuleSelector();
  renderExactObjectiveControls();
  renderExactReferenceList();
  renderLLMPrests();
  attachEvents();
  initTabStateFromHash();
  renderInstance();
  renderGraphMeta();
  renderGraphVisual();
  renderNodeDetail();
  renderSimulation();
  renderObjectives();
  renderOptimizeStatus();
  renderOptimizeResult();
  renderParetoAssistant();
  renderLLMConfig();
  setHeroStatus();
  await Promise.allSettled([loadInstance(true), loadObjectiveCatalog(), loadExactObjectiveCatalog(), loadLLM(true)]);
  setHeroStatus();
}

window.addEventListener("load", init);
