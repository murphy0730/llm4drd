/* ============ LLM4DRD V2 - JavaScript 核心框架 ============ */

/**
 * 智能调度决策中枢 - 核心应用框架
 * 
 * 职责分工：
 * - app 对象：全局状态与 API 接口
 * - UI 控制器：页面导航、事件绑定、DOM 操作
 * - 工作流引擎：步骤管理、状态追踪、结果缓存
 */

// ========== 全局配置 ==========

const CONFIG = {
  API_BASE: '/api',
  TIMEOUT: 30000,
  CACHE_KEY_PREFIX: 'llm4drd_',
};

// ========== 业务数据状态 ==========

const app = {
  // 当前场景
  currentScene: null,
  currentSceneId: null,
  scenesCache: {},

  // 实例数据
  instanceDetails: null,
  instanceSummary: null,

  // 图数据
  graphMeta: null,
  graphNodes: [],
  graphEdges: [],

  // 仿真结果
  simResult: null,
  simResults: {}, // { rule: result }

  // 优化结果
  optimizeTaskId: null,
  optimizeStatus: null,
  optimizeResult: null,
  optimizeHistory: [],

  // LLM 配置
  llmConfig: null,

  // UI 状态
  currentPage: 'scene-library',
  sidebarCollapsed: false,
  infoPanelCollapsed: false,

  // 工作流进度
  workflowProgress: {
    step1: 'not-started',
    step2: 'not-started',
    step3: 'not-started',
    step4: 'not-started',
    step5: 'not-started',
  },
};

// ========== API 接口封装 ==========

const api = {
  /**
   * 通用 fetch 包装
   */
  async request(endpoint, options = {}) {
    const url = `${CONFIG.API_BASE}${endpoint}`;
    const fetchOptions = {
      method: 'GET',
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
    };

    if (options.body && typeof options.body === 'object') {
      fetchOptions.body = JSON.stringify(options.body);
    }

    try {
      const response = await fetch(url, fetchOptions);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `HTTP ${response.status}`);
      }

      const contentType = response.headers.get('Content-Type') || '';
      if (contentType.includes('application/json')) {
        return await response.json();
      } else if (contentType.includes('application/octet-stream') || 
                 contentType.includes('application/vnd.openxmlformats')) {
        return await response.blob();
      }
      return await response.text();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error.message);
      throw error;
    }
  },

  // ===== 实例 API =====

  async generateInstance(config) {
    return this.request('/instance/generate', {
      method: 'POST',
      body: config,
    });
  },

  async getInstanceDetails() {
    return this.request('/instance/details');
  },

  async getInstanceDb() {
    return this.request('/instance/db');
  },

  async importExcel(file) {
    const formData = new FormData();
    formData.append('file', file);
    return fetch(`${CONFIG.API_BASE}/instance/import-excel`, {
      method: 'POST',
      body: formData,
    }).then(r => r.json());
  },

  async downloadTemplate() {
    return this.request('/instance/template');
  },

  async exportCsv() {
    return this.request('/instance/csv');
  },

  // ===== 图 API =====

  async buildGraph() {
    return this.request('/graph/build', { method: 'POST' });
  },

  async getGraphMeta() {
    return this.request('/graph/meta');
  },

  async getGraphNodes() {
    return this.request('/graph/nodes');
  },

  async getGraphEdges() {
    return this.request('/graph/edges');
  },

  // ===== 仿真 API =====

  async simulate(ruleName) {
    return this.request('/simulate', {
      method: 'POST',
      body: { rule_name: ruleName },
    });
  },

  // ===== 优化 API =====

  async startHybridOptimize(config) {
    return this.request('/optimize/hybrid', {
      method: 'POST',
      body: config,
    });
  },

  async getOptimizeStatus(taskId) {
    return this.request(`/optimize/${taskId}`);
  },

  async getOptimizeResult(taskId) {
    return this.request(`/optimize/${taskId}/result`);
  },

  // ===== LLM API =====

  async getLlmConfig() {
    return this.request('/llm-config');
  },

  async setLlmConfig(config) {
    return this.request('/llm-config', {
      method: 'POST',
      body: config,
    });
  },

  async testLlmConnection(config) {
    return this.request('/llm-config/test', {
      method: 'POST',
      body: config,
    });
  },
};

// ========== UI 控制器 ==========

const ui = {
  /**
   * 初始化事件监听
   */
  init() {
    this.bindNavigationEvents();
    this.bindToolbarEvents();
    this.bindPageEvents();
    this.loadInitialState();
  },

  /**
   * 绑定导航事件
   */
  bindNavigationEvents() {
    // 侧边栏导航项
    document.querySelectorAll('[data-nav]').forEach(button => {
      button.addEventListener('click', (e) => {
        const nav = button.dataset.nav;
        this.navigateTo(nav);
      });
    });

    // 子菜单展开/收起
    document.querySelectorAll('.nav-submenu-toggle').forEach(toggle => {
      toggle.parentElement.addEventListener('click', (e) => {
        if (e.target === toggle || e.target.parentElement === toggle) {
          const submenu = toggle.parentElement.nextElementSibling;
          if (submenu?.classList.contains('nav-submenu')) {
            submenu.classList.toggle('show');
            toggle.parentElement.dataset.navExpanded = 
              submenu.classList.contains('show') ? 'true' : 'false';
          }
        }
      });
    });

    // 右侧信息栏折叠按钮
    const panelToggle = document.getElementById('panel-toggle');
    if (panelToggle) {
      panelToggle.addEventListener('click', () => {
        this.toggleInfoPanel();
      });
    }
  },

  /**
   * 绑定顶部栏事件
   */
  bindToolbarEvents() {
    const refreshBtn = document.querySelector('[data-action="refresh-all"]');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        this.refreshAllData();
      });
    }

    const helpBtn = document.querySelector('[data-action="toggle-help"]');
    if (helpBtn) {
      helpBtn.addEventListener('click', () => {
        this.toggleHelp();
      });
    }

    const switchSceneBtn = document.querySelector('[data-action="switch-scene"]');
    if (switchSceneBtn) {
      switchSceneBtn.addEventListener('click', () => {
        this.navigateTo('scene-library');
      });
    }
  },

  /**
   * 绑定页面内事件
   */
  bindPageEvents() {
    // 场景库
    const createScenarioBtn = document.querySelector('[data-action="create-scenario"]');
    if (createScenarioBtn) {
      createScenarioBtn.addEventListener('click', () => {
        this.navigateTo('new-scene');
      });
    }

    // 新建场景
    const doGenerateBtn = document.querySelector('[data-action="do-generate"]');
    if (doGenerateBtn) {
      doGenerateBtn.addEventListener('click', () => {
        this.doGenerateInstance();
      });
    }

    // 优化相关
    const jumpToOptBtn = document.querySelector('[data-action="jump-to-optimize"]');
    if (jumpToOptBtn) {
      jumpToOptBtn.addEventListener('click', () => {
        this.navigateTo('optimize-launch');
      });
    }

    // 工作台步骤
    document.querySelectorAll('[data-action="view-step"]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const step = e.target.dataset.step;
        workflow.nextStep(step);
      });
    });
  },

  /**
   * 导航到指定页面
   */
  navigateTo(pageName) {
    // 隐藏所有页面
    document.querySelectorAll('.page').forEach(page => {
      page.classList.remove('active');
    });

    // 更新导航项状态
    document.querySelectorAll('[data-nav]').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.nav === pageName) {
        btn.classList.add('active');
      }
    });

    // 显示目标页面
    const pageMap = {
      'scene-library': 'page-scene-library',
      'new-scene': 'page-new-scene',
      'dashboard': 'page-dashboard',
      'workflow': 'page-workflow',
      'config': 'page-config',
    };

    const pageId = pageMap[pageName];
    if (pageId) {
      const page = document.getElementById(pageId);
      if (page) {
        page.classList.add('active');
        app.currentPage = pageName;

        // 页面特定的初始化
        this.onPageLoad(pageName);
      }
    }
  },

  /**
   * 页面加载时的钩子
   */
  onPageLoad(pageName) {
    switch (pageName) {
      case 'dashboard':
        this.renderDashboard();
        break;
      case 'scene-library':
        this.renderSceneLibrary();
        break;
      case 'workflow':
        this.renderWorkflow();
        break;
    }
  },

  /**
   * 渲染智能仪表板
   */
  async renderDashboard() {
    try {
      // 加载必要数据
      if (!app.instanceDetails) {
        app.instanceDetails = await api.getInstanceDetails();
      }

      // 更新 KPI 卡片
      const summary = app.instanceDetails.summary;
      document.getElementById('dash-orders').textContent = summary?.orders || '0';
      document.getElementById('dash-operations').textContent = summary?.operations || '0';
      document.getElementById('dash-machines').textContent = summary?.machines || '0';
      document.getElementById('dash-toolings').textContent = summary?.toolings || '0';

      // 渲染其他内容（延迟加载）
      this.renderDashboardSolutions();
    } catch (error) {
      this.showNotification('仪表板加载失败: ' + error.message, 'error');
    }
  },

  /**
   * 渲染方案表
   */
  renderDashboardSolutions() {
    const tbody = document.getElementById('dashboard-solution-tbody');
    if (!app.optimizeResult || !app.optimizeResult.solutions) {
      return;
    }

    tbody.innerHTML = app.optimizeResult.solutions
      .slice(0, 5)
      .map((sol, idx) => `
        <tr>
          <td>${sol.is_recommended ? '✓' : ''}</td>
          <td>方案 ${idx + 1}</td>
          <td>${(sol.objectives?.total_tardiness || 0).toFixed(1)}</td>
          <td>${(sol.objectives?.avg_utilization * 100 || 0).toFixed(1)}%</td>
          <td>${sol.score ? '⭐'.repeat(Math.round(sol.score * 5)) : '—'}</td>
          <td><button class="btn btn-sm" data-action="view-solution">查看</button></td>
        </tr>
      `).join('');
  },

  /**
   * 渲染场景库
   */
  async renderSceneLibrary() {
    try {
      const grid = document.getElementById('scenario-grid');
      if (!grid) return;

      // 模拟获取场景列表（实际应从后端获取）
      const scenarios = this.getMockScenarios();

      grid.innerHTML = scenarios
        .map(scene => `
          <div class="scenario-card" data-scene-id="${scene.id}">
            <div class="card-header">
              <h3>${scene.name}</h3>
            </div>
            <div class="card-body">
              <div class="scenario-info">
                <p>📦 ${scene.orders} 个订单 / ${scene.operations} 个工序</p>
                <p>⚙️ ${scene.machines} 台机器 / ${scene.toolings} 套工装</p>
                <p>📅 ${scene.createTime}</p>
              </div>
            </div>
            <div class="card-footer">
              <button class="btn btn-sm btn-primary" data-action="load-scenario" data-scenario-id="${scene.id}">
                加载场景
              </button>
            </div>
          </div>
        `).join('');

      // 绑定加载事件
      grid.querySelectorAll('[data-action="load-scenario"]').forEach(btn => {
        btn.addEventListener('click', (e) => {
          const sceneId = e.target.dataset.sceneId;
          this.loadScene(sceneId);
        });
      });
    } catch (error) {
      this.showNotification('场景库加载失败: ' + error.message, 'error');
    }
  },

  /**
   * 渲染工作台
   */
  renderWorkflow() {
    // 初始化步骤状态
    const steps = [
      { id: 1, status: app.workflowProgress.step1 || 'not-started' },
      { id: 2, status: app.workflowProgress.step2 || 'not-started' },
      { id: 3, status: app.workflowProgress.step3 || 'not-started' },
      { id: 4, status: app.workflowProgress.step4 || 'not-started' },
      { id: 5, status: app.workflowProgress.step5 || 'not-started' },
    ];

    steps.forEach(step => {
      const stepEl = document.getElementById(`step-${step.id}`);
      if (stepEl) {
        const statusEl = stepEl.querySelector('[id^="step-"][id$="-status"]');
        if (statusEl) {
          statusEl.textContent = this.getStatusLabel(step.status);
        }
        if (step.status === 'in-progress') {
          stepEl.classList.add('active');
        }
      }
    });
  },

  /**
   * 生成实例
   */
  async doGenerateInstance() {
    try {
      const form = document.getElementById('form-generate');
      if (!form) return;

      const config = {
        num_orders: parseInt(form.querySelector('#gen-orders').value),
        machines_per_type: parseInt(form.querySelector('#gen-machines').value),
        due_date_factor: parseFloat(form.querySelector('#gen-due-factor').value),
        seed: parseInt(form.querySelector('#gen-seed').value),
        plan_start_at: form.querySelector('#gen-start-time').value,
      };

      this.showNotification('生成中...', 'info');
      const result = await api.generateInstance(config);

      app.instanceDetails = result;
      this.showNotification('实例已生成', 'success');

      // 自动导航到仪表板
      setTimeout(() => {
        this.navigateTo('dashboard');
        this.updateTopbar();
      }, 500);
    } catch (error) {
      this.showNotification('生成失败: ' + error.message, 'error');
    }
  },

  /**
   * 加载场景
   */
  async loadScene(sceneId) {
    try {
      app.currentSceneId = sceneId;
      app.currentScene = app.scenesCache[sceneId];

      this.updateTopbar();
      this.updateInfoPanel();
      this.navigateTo('dashboard');

      // 显示决策路径导航
      document.getElementById('decision-path-section').style.display = 'block';
      document.getElementById('workflow-section').style.display = 'block';
      document.getElementById('optimize-section').style.display = 'block';

      this.showNotification(`已加载场景: ${app.currentScene.name}`, 'success');
    } catch (error) {
      this.showNotification('场景加载失败: ' + error.message, 'error');
    }
  },

  /**
   * 更新顶部栏状态
   */
  updateTopbar() {
    if (app.currentScene) {
      document.getElementById('topbar-scene-name').textContent = app.currentScene.name;
      const summary = app.currentScene.summary;
      document.getElementById('topbar-orders-ops').textContent = 
        `${summary.orders}/${summary.operations}`;
      document.getElementById('topbar-resources').textContent = 
        `${summary.machines}×${summary.toolings}`;
    } else {
      document.getElementById('topbar-scene-name').textContent = '未加载场景';
      document.getElementById('topbar-orders-ops').textContent = '—';
      document.getElementById('topbar-resources').textContent = '—';
    }
  },

  /**
   * 更新右侧信息栏
   */
  updateInfoPanel() {
    if (!app.currentScene) return;

    const scene = app.currentScene;
    document.getElementById('panel-scene-name').textContent = scene.name;
    document.getElementById('panel-orders').textContent = scene.summary.orders;
    document.getElementById('panel-operations').textContent = scene.summary.operations;
    document.getElementById('panel-machines').textContent = scene.summary.machines;

    const statusEl = document.getElementById('panel-status');
    if (scene.readyForOptimize) {
      statusEl.textContent = '已准备';
      statusEl.className = 'status-badge ready';
    } else {
      statusEl.textContent = '未准备';
      statusEl.className = 'status-badge';
    }
  },

  /**
   * 显示通知消息
   */
  showNotification(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => toast.remove(), 3000);
  },

  /**
   * 切换信息栏折叠
   */
  toggleInfoPanel() {
    const panel = document.getElementById('info-panel');
    panel?.classList.toggle('collapsed');
    app.infoPanelCollapsed = panel?.classList.contains('collapsed');
  },

  /**
   * 刷新所有数据
   */
  async refreshAllData() {
    try {
      if (app.currentSceneId) {
        app.instanceDetails = await api.getInstanceDetails();
        this.updateTopbar();
        this.updateInfoPanel();
        this.showNotification('数据已刷新', 'success');
      }
    } catch (error) {
      this.showNotification('刷新失败: ' + error.message, 'error');
    }
  },

  /**
   * 工具方法：获取状态标签
   */
  getStatusLabel(status) {
    const labels = {
      'not-started': '未开始',
      'in-progress': '进行中',
      'completed': '已完成',
      'error': '失败',
    };
    return labels[status] || '—';
  },

  /**
   * 工具方法：获取模拟场景数据
   */
  getMockScenarios() {
    const now = new Date();
    return [
      {
        id: 'demo-001',
        name: 'Demo_20250327',
        orders: 10,
        operations: 45,
        machines: 3,
        toolings: 2,
        createTime: now.toLocaleString('zh-CN'),
        summary: { orders: 10, operations: 45, machines: 3, toolings: 2 },
        readyForOptimize: true,
      },
      {
        id: 'demo-002',
        name: 'Demo_20250326',
        orders: 15,
        operations: 62,
        machines: 4,
        toolings: 2,
        createTime: new Date(now.getTime() - 86400000).toLocaleString('zh-CN'),
        summary: { orders: 15, operations: 62, machines: 4, toolings: 2 },
        readyForOptimize: false,
      },
    ];
  },

  /**
   * 初始化场景缓存
   */
  loadInitialState() {
    const scenarios = this.getMockScenarios();
    scenarios.forEach(s => {
      app.scenesCache[s.id] = s;
    });

    // 显示场景库
    this.navigateTo('scene-library');
  },

  /**
   * 切换帮助面板
   */
  toggleHelp() {
    this.showNotification('帮助功能开发中...', 'info');
  },

  /**
   * 刷新场景 UI
   */
  refreshSceneLibrary() {
    if (app.currentPage === 'scene-library') {
      this.renderSceneLibrary();
    }
  },
};

// ========== 工作流引擎 ==========

const workflow = {
  currentStep: 0,

  async nextStep(stepNumber) {
    this.currentStep = parseInt(stepNumber);
    const stepKey = `step${stepNumber}`;

    // 根据步骤加载不同的工作区内容
    await this.loadStepContent(stepNumber);

    // 更新步骤状态
    document.querySelectorAll('.step-item').forEach((el, idx) => {
      if (idx + 1 === this.currentStep) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    });
  },

  async loadStepContent(stepNumber) {
    const container = document.getElementById('work-area-content');
    if (!container) return;

    let content = '';

    switch (stepNumber) {
      case 1:
        content = this.renderStep1();
        break;
      case 2:
        content = this.renderStep2();
        break;
      case 3:
        content = await this.renderStep3();
        break;
      case 4:
        content = this.renderStep4();
        break;
      case 5:
        content = this.renderStep5();
        break;
    }

    container.innerHTML = content;
    this.bindStepEvents(stepNumber);
  },

  renderStep1() {
    return `
      <div class="step-panel">
        <h3>第 1 步：问题设计</h3>
        <p>定义调度问题的基本参数：订单、任务、工序、资源等</p>
        <form id="form-step1">
          <div class="form-group">
            <label>订单数量</label>
            <input type="number" value="10" min="1">
          </div>
          <div class="form-group">
            <label>计划周期（天）</label>
            <input type="number" value="7" min="1">
          </div>
          <button class="btn btn-primary" type="button" data-action="step-next">下一步</button>
        </form>
      </div>
    `;
  },

  renderStep2() {
    return `
      <div class="step-panel">
        <h3>第 2 步：约束校准</h3>
        <p>维护机器班次、停机时间、工装需求和在制状态</p>
        <div class="grid-2">
          <div class="card">
            <div class="card-body">
              <h4>班次管理</h4>
              <button class="btn btn-ghost" data-action="manage-shifts">管理班次</button>
            </div>
          </div>
          <div class="card">
            <div class="card-body">
              <h4>停机维护</h4>
              <button class="btn btn-ghost" data-action="manage-downtime">管理停机</button>
            </div>
          </div>
        </div>
        <button class="btn btn-primary" type="button" data-action="step-complete">完成并继续</button>
      </div>
    `;
  },

  async renderStep3() {
    return `
      <div class="step-panel">
        <h3>第 3 步：结构与仿真</h3>
        <p>构建异构图、运行规则仿真，校验约束可行性</p>
        <div class="grid-2">
          <div class="card">
            <div class="card-header">异构图分析</div>
            <div class="card-body">
              <p id="graph-status" class="muted">待构图</p>
              <button class="btn btn-primary" data-action="build-graph">构建图</button>
            </div>
          </div>
          <div class="card">
            <div class="card-header">规则仿真</div>
            <div class="card-body">
              <select id="simulate-rule">
                <option value="ATC">ATC 规则</option>
                <option value="EDD">EDD 规则</option>
                <option value="SPT">SPT 规则</option>
              </select>
              <button class="btn btn-primary" data-action="run-simulate">运行仿真</button>
            </div>
          </div>
        </div>
        <button class="btn btn-primary" type="button" data-action="step-complete">完成并继续</button>
      </div>
    `;
  },

  renderStep4() {
    return `
      <div class="step-panel">
        <h3>第 4 步：多目标优化</h3>
        <p>配置优化目标和参数，启动混合优化求解</p>
        <form id="form-optimize">
          <div class="form-group">
            <label>优化目标</label>
            <input type="checkbox" value="total_tardiness" checked> 最小延期
            <input type="checkbox" value="makespan" checked> 最小完工时间
            <input type="checkbox" value="avg_utilization"> 最大利用率
          </div>
          <div class="form-group">
            <label>运行时间（秒）</label>
            <input type="number" value="90" min="10">
          </div>
          <button class="btn btn-primary" type="button" data-action="start-optimize">启动优化</button>
        </form>
        <div id="optimize-progress"></div>
        <button class="btn btn-primary" type="button" data-action="step-complete" style="display:none;">完成并继续</button>
      </div>
    `;
  },

  renderStep5() {
    return `
      <div class="step-panel">
        <h3>第 5 步：方案评审</h3>
        <p>评审多个候选方案，选择最优方案进行部署</p>
        <div id="solution-review-container">
          <p class="muted">运行优化以获得方案</p>
        </div>
        <button class="btn btn-primary" type="button" data-action="export-results">导出结果</button>
      </div>
    `;
  },

  bindStepEvents(stepNumber) {
    const container = document.getElementById('work-area-content');

    // 下一步按钮
    const nextBtn = container?.querySelector('[data-action="step-next"]');
    if (nextBtn) {
      nextBtn.addEventListener('click', () => {
        if (stepNumber < 5) this.nextStep(stepNumber + 1);
      });
    }

    // 完成按钮
    const completeBtn = container?.querySelector('[data-action="step-complete"]');
    if (completeBtn) {
      completeBtn.addEventListener('click', () => {
        if (stepNumber < 5) this.nextStep(stepNumber + 1);
      });
    }

    // 仿真按钮
    const simBtn = container?.querySelector('[data-action="run-simulate"]');
    if (simBtn) {
      simBtn.addEventListener('click', async () => {
        const rule = container.querySelector('#simulate-rule').value;
        try {
          ui.showNotification('仿真中...', 'info');
          const result = await api.simulate(rule);
          app.simResult = result;
          ui.showNotification('仿真完成', 'success');
        } catch (error) {
          ui.showNotification('仿真失败: ' + error.message, 'error');
        }
      });
    }

    // 构建图按钮
    const graphBtn = container?.querySelector('[data-action="build-graph"]');
    if (graphBtn) {
      graphBtn.addEventListener('click', async () => {
        try {
          ui.showNotification('构建图中...', 'info');
          const meta = await api.buildGraph();
          app.graphMeta = meta;
          container.querySelector('#graph-status').textContent = 
            `已构建: ${meta.total_nodes} 节点, ${meta.total_edges} 条边`;
          ui.showNotification('图已构建', 'success');
        } catch (error) {
          ui.showNotification('构建失败: ' + error.message, 'error');
        }
      });
    }

    // 启动优化按钮
    const optBtn = container?.querySelector('[data-action="start-optimize"]');
    if (optBtn) {
      optBtn.addEventListener('click', async () => {
        try {
          ui.showNotification('优化启动中...', 'info');
          const config = {
            objective_keys: ['total_tardiness', 'makespan'],
            time_limit_s: parseInt(container.querySelector('input[type="number"]').value),
          };
          const result = await api.startHybridOptimize(config);
          app.optimizeTaskId = result.task_id;
          ui.showNotification('优化已启动', 'success');
          
          // 显示完成按钮
          container.querySelector('[data-action="step-complete"]').style.display = 'block';
        } catch (error) {
          ui.showNotification('优化启动失败: ' + error.message, 'error');
        }
      });
    }
  },
};

// ========== 主程序入口 ==========

document.addEventListener('DOMContentLoaded', () => {
  ui.init();
  console.log('LLM4DRD V2 应用已启动');
});
