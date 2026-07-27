import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "frontend" / "app_v2.js").read_text(encoding="utf-8")
CSS = (ROOT / "frontend" / "app_v2.css").read_text(encoding="utf-8")
RUNTIME_JS = (ROOT / "frontend" / "review_runtime.js").read_text(encoding="utf-8")


class ReviewFrontendContractTests(unittest.TestCase):
    def test_all_gantts_use_shared_initial_window(self):
        self.assertIn("function ganttWindowPayload(", JS)
        self.assertGreaterEqual(JS.count("ganttWindowPayload("), 3)
        self.assertIn("ReviewRuntime.computeInitialWindow", JS)
        self.assertIn("fullWindow", JS)
        self.assertIn("initialWindow", JS)
        self.assertIn("data.initialWindow", JS)
        self.assertNotIn("data.window ?", JS)

    def test_gantt_ranges_are_bound_to_logical_views(self):
        self.assertIn("ganttViewWindows: {}", JS)
        self.assertIn("stored?.viewKey === data.viewKey", JS)
        self.assertIn('timeline.on("rangechanged"', JS)
        self.assertIn("min: data.fullWindow", JS)
        self.assertIn("max: data.fullWindow", JS)
        self.assertIn("app.ganttViewWindows = {}", JS)

    def test_view_keys_include_logical_dimensions(self):
        self.assertIn("options.selectedOrder", JS)
        self.assertIn("options.solutionIds", JS)
        self.assertIn("groupMode", JS)
        # solutionIds 归一化后参与 viewKey：顺序变化不应误判为新视图
        self.assertIn("solutionIds: asArray(options.solutionIds).map(String).sort()", JS)

    def test_review_api_accepts_abort_signals(self):
        self.assertIn("getReviewData(taskId, solutionIds, orderId, includeUtilization, signal)", JS)
        self.assertIn("searchReviewOrders(taskId, solutionIds, query, signal)", JS)
        self.assertGreaterEqual(JS.count("{ signal }"), 2)

    def test_review_linkage_is_batched_and_locally_rendered(self):
        self.assertIn("getReviewData(", JS)
        self.assertNotIn("Promise.all(ids.map((id) =>", JS)
        self.assertIn('id="review-comparison-region"', JS)
        self.assertIn('id="review-utilization-region"', JS)
        self.assertIn('id="review-gantt-region"', JS)
        self.assertIn("refreshReviewDynamicRegions()", JS)
        self.assertIn(".items.clear()", JS)
        self.assertIn(".groups.clear()", JS)

    def test_review_loads_do_not_render_the_whole_page(self):
        start = JS.index("async function loadReviewData(")
        end = JS.index("\nfunction ", start)
        self.assertNotIn("renderCurrentPage()", JS[start:end])

    def test_review_selection_drops_stale_pending_canvas_without_full_render(self):
        self.assertIn('app.pendingGantts.delete("gantt-review-compare")', JS)
        start = JS.index('if (action === "toggle-candidate")')
        end = JS.index('if (action === "generate-exact-single")', start)
        self.assertNotIn("renderCurrentPage()", JS[start:end])
        self.assertIn("ensureReviewData(getSelectedReviewCandidates())", JS[start:end])

    def test_order_selectors_use_accessible_fuzzy_combobox(self):
        for token in (
            'role="combobox"',
            'role="listbox"',
            "aria-expanded",
            "ORDER_SEARCH_DEBOUNCE_MS = 200",
            "ORDER_SEARCH_LIMIT = 50",
        ):
            self.assertIn(token, JS)
        for key in ('"ArrowDown"', '"ArrowUp"', '"Enter"', '"Escape"'):
            self.assertIn(key, RUNTIME_JS)
        self.assertNotIn("data-review-gantt-order", JS)
        self.assertNotIn("data-gantt-order-select", JS)

    def test_combobox_selection_paths_preserve_gantt_loading_contracts(self):
        self.assertIn("ReviewRuntime.createOrderComboboxController(", JS)
        self.assertIn("ReviewRuntime.handleOrderComboboxKey(controller, event)", JS)
        # 模糊排序内聚在 controller 内部，不再由页面侧直接调用
        self.assertIn("rankOrders(matches, query, limit, { current, recent })", RUNTIME_JS)
        # 服务端取数模式：搜索走可取消的评审订单接口，选中后按订单重载甘特
        self.assertIn("api.searchReviewOrders(taskId, [solutionId], query, signal)", JS)
        self.assertIn("loadPlanGantt(taskId, solutionId, order.order_id)", JS)

    def test_comboboxes_mount_after_timeline_and_review_region_rendering(self):
        gantt_start = JS.index("function mountGantts()")
        gantt_end = JS.index("\nasync function renderCurrentPage(", gantt_start)
        self.assertIn("mountOrderComboboxes()", JS[gantt_start:gantt_end])
        review_start = JS.index("function refreshReviewDynamicRegions()")
        review_end = JS.index("\nfunction renderReview()", review_start)
        self.assertIn("mountOrderComboboxes()", JS[review_start:review_end])

    def test_combobox_focus_click_and_recent_order_contract(self):
        mount_start = JS.index("function mountOrderComboboxes()")
        mount_end = JS.index("\nfunction formatNumber", mount_start)
        mount_source = JS[mount_start:mount_end]
        self.assertIn("ReviewRuntime.bindOrderComboboxOpen(input, controller)", mount_source)
        self.assertIn(
            "orderComboboxRecent: ReviewRuntime.createRecentOrderStore({",
            JS,
        )
        self.assertIn("contextLimit: 24", JS)
        self.assertIn("itemLimit: ORDER_RECENT_LIMIT", JS)
        self.assertIn("app.orderComboboxRecent.record(recentKey, order)", JS)
        self.assertIn("app.orderComboboxRecent.read(source.recentKey)", mount_source)
        reset_start = JS.index("function resetInstanceDerivedState()")
        reset_end = JS.index("\nfunction ", reset_start)
        self.assertIn("app.orderComboboxRecent.reset()", JS[reset_start:reset_end])

    def test_order_combobox_recent_key_is_scoped_by_data_context(self):
        # 甘特已统一到 renderTimeline：服务端取数模式用单选 combobox（plan 上下文），
        # 本地模式改用 renderMultiSelectFilter，因此只剩这一个 combobox 上下文。
        self.assertIn("contextKey: `plan::${taskId}::${solutionId}`", JS)
        self.assertIn("const recentKey = `${config.id}::${config.contextKey}`", JS)
        self.assertIn("source.recentKey", JS)
        # 本地模式不再挂 combobox，避免跨实例串用最近订单
        self.assertIn("renderMultiSelectFilter({", JS)

    def test_review_gantt_preserves_escaped_backend_failure_messages(self):
        # 评审甘特已改为单方案：后端失败信息就地转义展示，并提供重试入口
        start = JS.index("function renderReviewGantt(")
        end = JS.index("\nfunction renderReviewLibraryTab(", start)
        source = JS[start:end]
        self.assertIn("escapeHtml(st.error)", source)
        self.assertIn('data-action="retry-review-scheme"', source)
        self.assertIn("if (st.loading)", source)

    def test_review_commits_are_guarded_by_request_generation_and_schedule(self):
        self.assertIn("let reviewReadRequestGeneration = 0", JS)
        self.assertIn("function isCurrentReviewReadRequest(", JS)
        start = JS.index("async function loadReviewData(")
        end = JS.index("\nfunction ensureReviewData(", start)
        self.assertGreaterEqual(JS[start:end].count("isCurrentReviewReadRequest("), 2)

    def test_pending_order_does_not_relabel_or_upsert_committed_schemes(self):
        load_start = JS.index("async function loadReviewData(")
        load_end = JS.index("\nfunction ensureReviewData(", load_start)
        self.assertIn("orderId: selectionChanged ? null : previous.orderId", JS[load_start:load_end])
        self.assertIn("scheduleKey: selectionChanged ? null : previous.scheduleKey", JS[load_start:load_end])

    def test_review_tables_use_compact_approved_content(self):
        comparison_start = JS.index("function renderReviewCandidateComparison")
        comparison_end = JS.index("function renderReviewTypeUtilization")
        comparison_source = JS[comparison_start:comparison_end]
        self.assertIn('<th class="compare-check"', comparison_source)
        self.assertIn('<span class="sr-only">选择方案</span>', comparison_source)
        self.assertNotIn('const headers = ["选"', comparison_source)
        self.assertIn('aria-label="勾选 ${escapeHtml(item.name)}"', comparison_source)

        # 利用率区块已改为「按天 × 机器类型」表：无进度条/副标题装饰，行内最佳高亮，
        # 并标注该类型「已排产/总台数」以说明日利用率分母口径。
        utilization_start = JS.index("function renderReviewTypeUtilization")
        utilization_end = JS.index("function ensureReviewGanttSchemeSelection(")
        utilization_source = JS[utilization_start:utilization_end]
        for token in ("util-bar", "cell-sub"):
            self.assertNotIn(token, utilization_source)
        self.assertIn("<strong>${escapeHtml(type.type_name)}</strong>", utilization_source)
        self.assertIn("formatPercent(v)", utilization_source)
        self.assertIn('"pct is-best"', utilization_source)
        self.assertIn("type.machines_used", utilization_source)
        self.assertIn("type.machines_total", utilization_source)

        self.assertIn(".util-col-type { width: 132px; }", CSS)
        util_css_start = CSS.index(".util-table")
        util_css = CSS[util_css_start:CSS.index(".table-shell", util_css_start)]
        self.assertIn("white-space: normal", util_css)
        self.assertIn("overflow: visible", util_css)
        self.assertIn("word-break: break-word", util_css)
        self.assertNotIn("text-overflow: ellipsis", util_css)
        self.assertNotIn("max-width", util_css)
        self.assertNotIn(".util-bar", util_css)

        sr_only_start = CSS.index(".sr-only")
        sr_only_css = CSS[sr_only_start:CSS.index("}", sr_only_start)]
        self.assertIn("position: absolute", sr_only_css)
        self.assertIn("clip:", sr_only_css)

    def test_multi_select_commit_keeps_explicit_selection(self):
        # 全选必须原样提交全量 id：折叠成空数组会与订单筛选下游的「未选择」语义撞车，
        # 大实例下被首单兜底覆盖，表现为「全选后点空白处只剩第一个订单」。
        self.assertIn("source.onChange(Array.from(working));", JS)
        self.assertNotIn("working.size >= total ? [] : Array.from(working)", JS)
        # 大实例首单兜底仍保留（空数组 = 未选择）
        self.assertIn(
            "if (!localSelectedIds.length && !allowAll && orderOptions.length) localSelectedIds = [orderOptions[0]];",
            JS,
        )
        # 全选后的性能保护：机器行分页额外受单页条目数上限约束
        self.assertIn("function paginateGanttMachineRows(", JS)
        self.assertIn("GANTT_PAGE_MAX_ITEMS", JS)
        self.assertIn("pageItems", JS)
        # 机器类型侧仍把「全选」折叠成空数组，保持 filterGanttMachineRows 的不过滤语义
        self.assertIn("types.length >= facet.typeOptions.length ? [] : types", JS)
        # 全选后按订单集合过滤必须走 Set，避免 O(条目数 × 选中数)
        self.assertIn("selectedOrderSet.has(item.order_id", JS)

    def test_graph_layout_is_wide(self):
        # 图谱过高的来源是 rankSep 过大（150 → fit 后仅 51% 缩放）；压到 80 让整图变宽扁。
        # 方向保持 TB：末层资源节点多，TB 下横向铺开才是宽的方向（改 LR 实测反而更高）。
        self.assertIn(
            'name: "dagre", rankDir: "TB", rankSep: 80, nodeSep: 32',
            JS,
        )
        self.assertNotIn("rankSep: 150", JS)

    def test_optimize_layout_is_single_column(self):
        # 优化求解页：多目标优化配置在上、运行状态与日志在下
        opt_start = CSS.index(".opt-layout {")
        opt_css = CSS[opt_start:CSS.index("\n", opt_start)]
        self.assertNotIn("400px", opt_css)
        self.assertIn(".opt-layout .sticky-col { position: static; }", CSS)
        self.assertIn("配置上方目标与参数", JS)

    def test_optimize_cold_start_switch_defaults_to_warm_and_is_submitted(self):
        self.assertIn("coldStart: false", JS)
        self.assertIn('id="opt-cold-start"', JS)
        self.assertIn('role="switch"', JS)
        self.assertIn("cold_start: app.optimizeForm.coldStart", JS)
        self.assertIn('target.matches("#opt-cold-start")', JS)
        self.assertIn("app.optimizeForm.coldStart = target.checked", JS)
        self.assertIn(".cold-start-control", CSS)
        # 不落 localStorage：刷新页面后必须恢复默认热启动。
        self.assertNotIn("COLD_START", JS)
        self.assertNotIn('localStorage.setItem("coldStart"', JS)


if __name__ == "__main__":
    unittest.main()
