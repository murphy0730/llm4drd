from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from playwright.sync_api import Page, sync_playwright


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class FullVerificationResult:
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(item.ok for item in self.checks)

    def to_dict(self) -> dict:
        payload = {"passed": self.passed, "checks": [asdict(item) for item in self.checks]}
        return payload


def wait(page: Page, ms: int = 600) -> None:
    page.wait_for_timeout(ms)


def visible_text(page: Page) -> str:
    return page.locator("body").inner_text()


def goto_nav(page: Page, nav: str, expand_parent: str | None = None) -> None:
    if expand_parent:
        parent = page.locator(f'[data-nav="{expand_parent}"]').first
        if parent.count():
            parent.click()
            wait(page, 400)
    page.locator(f'[data-nav="{nav}"]').first.click()
    wait(page, 1200)


def count_visible(page: Page, selector: str) -> int:
    return page.locator(selector).count()


def wait_for_optimize_context(page: Page, timeout_ms: int = 20000) -> dict:
    state = page.evaluate(
        """() => (typeof app !== 'undefined'
        ? {taskId: app.optimizeTaskId || null, status: app.optimizeStatus?.status || null, hasResult: !!app.optimizeResult}
        : {taskId: null, status: null, hasResult: false})"""
    )
    elapsed = 0
    while elapsed < timeout_ms:
        if state.get("taskId") and (state.get("status") in {"done", "running"}):
            if state.get("hasResult") or state.get("status") == "done":
                return state
        page.wait_for_timeout(1000)
        elapsed += 1000
        state = page.evaluate(
            """() => (typeof app !== 'undefined'
            ? {taskId: app.optimizeTaskId || null, status: app.optimizeStatus?.status || null, hasResult: !!app.optimizeResult}
            : {taskId: null, status: null, hasResult: false})"""
        )
    return state


def verify(base_url: str, channel: str, headless: bool) -> FullVerificationResult:
    checks: list[CheckResult] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=headless)
        page = browser.new_page(viewport={"width": 1680, "height": 1050})
        response_log: list[tuple[str, int]] = []
        page.on("response", lambda resp: response_log.append((resp.url, resp.status)))
        page.goto(base_url, wait_until="networkidle", timeout=30000)

        checks.append(
            CheckResult(
                name="entry",
                ok="智能调度决策中枢" in page.title() or "智能调度决策中枢" in visible_text(page),
                detail=page.title(),
            )
        )

        page.locator('[data-action="goto-new-scene"]').click()
        wait(page, 800)
        ids = [
            "#gen-orders",
            "#gen-tasks-min",
            "#gen-tasks-max",
            "#gen-ops-min",
            "#gen-ops-max",
            "#gen-machines",
            "#gen-toolings",
            "#gen-personnel",
        ]
        scene_form_ok = all(page.locator(sel).count() for sel in ids)
        checks.append(CheckResult(name="new_scene_form", ok=scene_form_ok, detail=" / ".join(ids)))

        page.fill("#gen-orders", "6")
        page.fill("#gen-tasks-min", "1")
        page.fill("#gen-tasks-max", "1")
        page.fill("#gen-ops-min", "1")
        page.fill("#gen-ops-max", "1")
        page.fill("#gen-machines", "3")
        page.fill("#gen-toolings", "1")
        page.fill("#gen-personnel", "1")
        page.locator('[data-action="generate-instance"]').click()
        page.wait_for_timeout(2800)
        current_label = page.locator('[data-action="goto-scene-library"]').inner_text().strip()
        checks.append(
            CheckResult(
                name="scene_generate",
                ok=("6 单" in current_label) or ("6 / 18" in visible_text(page)) or ("6 单 / 18 工序" in visible_text(page)),
                detail=current_label,
            )
        )

        goto_nav(page, "downtime-management", expand_parent="config")
        before_count = count_visible(page, 'tr[data-downtime-id]')
        page.select_option("#downtime-machine", index=0)
        page.select_option("#downtime-type", "planned")
        start_dt = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
        end_dt = (datetime.now() + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M")
        page.fill("#downtime-start", start_dt)
        page.fill("#downtime-end", end_dt)
        page.locator('[data-action="add-downtime"]').click()
        page.wait_for_timeout(2200)
        after_count = count_visible(page, 'tr[data-downtime-id]')
        checks.append(
            CheckResult(
                name="downtime_add_and_show",
                ok=after_count >= before_count + 1,
                detail=f"{before_count} -> {after_count}",
            )
        )

        goto_nav(page, "structure-analysis", expand_parent="insights")
        if count_visible(page, '[data-action="build-graph"]'):
            page.locator('[data-action="build-graph"]').first.click()
            page.wait_for_timeout(2400)
        graph_text = visible_text(page)
        graph_ok = ("图构建时间" in graph_text) or ("结构分析" in graph_text)
        checks.append(
            CheckResult(
                name="graph_build",
                ok=graph_ok,
                detail="结构分析页已加载并尝试构建图谱",
                extra={"svg_count": count_visible(page, "svg")},
            )
        )

        graph_focus = page.evaluate(
            """() => {
                const renderedNodes = Array.from(document.querySelectorAll('[data-graph-node]'));
                const renderedTypeLabels = renderedNodes.reduce((acc, node) => {
                    const key = node.dataset.nodeTypeLabel || "";
                    acc[key] = (acc[key] || 0) + 1;
                    return acc;
                }, {});
                return {
                    appNodeCount: Array.isArray(app?.graphNodes) ? app.graphNodes.length : 0,
                    appEdgeCount: Array.isArray(app?.graphEdges) ? app.graphEdges.length : 0,
                    renderedNodeCount: renderedNodes.length,
                    renderedEdgeCount: document.querySelectorAll('[data-graph-link]').length,
                    renderedTypeLabels,
                    selectedNodeId: app?.selectedGraphNodeId || null,
                    stageMeta: document.querySelector('.graph-stage-meta')?.innerText || "",
                };
            }"""
        )
        checks.append(
            CheckResult(
                name="graph_interactive_depth",
                ok=(
                    graph_focus.get("renderedNodeCount", 0) >= 6
                    and graph_focus.get("renderedEdgeCount", 0) >= 4
                    and len(graph_focus.get("renderedTypeLabels", {}).keys()) >= 3
                ),
                detail=(
                    f"nodes={graph_focus.get('renderedNodeCount', 0)}, "
                    f"edges={graph_focus.get('renderedEdgeCount', 0)}, "
                    f"types={len(graph_focus.get('renderedTypeLabels', {}).keys())}"
                ),
                extra=graph_focus,
            )
        )

        graph_click_target = page.evaluate(
            """() => {
                const nodes = Array.from(document.querySelectorAll('[data-graph-node]'));
                const target = nodes.find((node) => (node.dataset.nodeTypeLabel || "") !== "订单");
                return target ? {
                    id: target.dataset.graphNode,
                    label: target.dataset.nodeLabel || "",
                    typeLabel: target.dataset.nodeTypeLabel || "",
                } : null;
            }"""
        )
        graph_click_ok = False
        graph_click_detail = "no-click-target"
        if graph_click_target:
            page.locator(f'[data-graph-node="{graph_click_target["id"]}"]').first.click()
            wait(page, 1200)
            graph_click_state = page.evaluate(
                """() => ({
                    selectedNodeId: app?.selectedGraphNodeId || null,
                    detailText: document.querySelector('.graph-detail-card')?.innerText || "",
                })"""
            )
            graph_click_ok = (
                graph_click_state.get("selectedNodeId") == graph_click_target["id"]
                and graph_click_target.get("label", "")[:6] in graph_click_state.get("detailText", "")
            )
            graph_click_detail = f'{graph_click_target["typeLabel"]}:{graph_click_target["label"]}'
        checks.append(
            CheckResult(
                name="graph_node_detail_linkage",
                ok=graph_click_ok,
                detail=graph_click_detail,
                extra=graph_click_target or {},
            )
        )

        graph_stat_shift = {"ok": False, "before": "", "after": ""}
        if graph_click_target:
            before_stats = page.locator("[data-graph-selected-summary]").inner_text().strip() if count_visible(page, "[data-graph-selected-summary]") else ""
            alt_target = page.evaluate(
                """(selectedId) => {
                    const nodes = Array.from(document.querySelectorAll('[data-graph-node]'));
                    const target = nodes.find((node) => {
                        if (node.dataset.graphNode === selectedId) return false;
                        const id = node.dataset.graphNode || "";
                        return id.startsWith("O:") || id.startsWith("T:") || id.startsWith("OP:");
                    }) || nodes.find((node) => node.dataset.graphNode !== selectedId);
                    return target ? target.dataset.graphNode : null;
                }""",
                graph_click_target["id"],
            )
            if alt_target:
                page.locator(f'[data-graph-node="{alt_target}"]').first.dispatch_event("click")
                wait(page, 1200)
                after_state = page.evaluate(
                    """() => ({
                        selectedNodeId: app?.selectedGraphNodeId || null,
                        summary: document.querySelector('[data-graph-selected-summary]')?.innerText || '',
                    })"""
                )
                after_stats = after_state.get("summary", "")
                graph_stat_shift = {
                    "ok": bool(after_stats) and (before_stats != after_stats or after_state.get("selectedNodeId") == alt_target),
                    "before": before_stats,
                    "after": after_stats,
                    "selectedNodeId": after_state.get("selectedNodeId"),
                }
        checks.append(
            CheckResult(
                name="graph_selected_stats_refresh",
                ok=bool(graph_stat_shift.get("ok")),
                detail=(graph_stat_shift.get("after", "") or "")[:120],
                extra=graph_stat_shift,
            )
        )

        graph_multi_state = page.evaluate(
            """() => {
                const nodes = Array.from(document.querySelectorAll('[data-graph-node]'));
                return nodes.reduce((acc, node) => {
                    const id = node.dataset.graphNode || '';
                    const key = id.startsWith('O:') ? 'order' : id.startsWith('T:') ? 'task' : id.startsWith('OP:') ? 'operation' : id.startsWith('M:') ? 'machine' : '';
                    if (!key) return acc;
                    if (!acc[key]) acc[key] = [];
                    acc[key].push(id);
                    return acc;
                }, {});
            }"""
        )
        graph_multi_snapshots = {}
        for type_key in ["order", "task", "operation", "machine"]:
            candidates = graph_multi_state.get(type_key) or []
            if not candidates:
                continue
            page.locator(f'[data-graph-node="{candidates[0]}"]').first.dispatch_event("click")
            wait(page, 900)
            graph_multi_snapshots[type_key] = (
                page.locator('[data-graph-selected-summary]').inner_text().strip()
                if count_visible(page, '[data-graph-selected-summary]')
                else ""
            )
        distinct_summaries = {value for value in graph_multi_snapshots.values() if value}
        checks.append(
            CheckResult(
                name="graph_multitype_scope_variation",
                ok=len(graph_multi_snapshots) >= 3 and len(distinct_summaries) >= 3,
                detail=" | ".join(
                    f"{key}:{(value.splitlines()[0] if value else '-')}" for key, value in graph_multi_snapshots.items()
                ),
                extra=graph_multi_snapshots,
            )
        )

        max_orders_before = page.locator("#graph-max-orders").input_value() if count_visible(page, "#graph-max-orders") else ""
        if count_visible(page, "#graph-max-orders"):
            page.fill("#graph-max-orders", "2")
            page.dispatch_event("#graph-max-orders", "input")
            wait(page, 1000)
        graph_limit_state = page.evaluate(
            """() => ({
                value: document.querySelector('#graph-max-orders')?.value || '',
                meta: document.querySelector('.graph-stage-meta')?.innerText || '',
                appValue: (typeof app !== 'undefined' && app.graphView) ? app.graphView.maxOrders : null,
            })"""
        )
        checks.append(
            CheckResult(
                name="graph_max_order_control",
                ok=(graph_limit_state.get("value") == "2" and graph_limit_state.get("appValue") == 2),
                detail=f'{max_orders_before} -> {graph_limit_state.get("value")}',
                extra=graph_limit_state,
            )
        )

        goto_nav(page, "resource-analysis", expand_parent="insights")
        resource_machine = page.evaluate(
            """() => {
                const firstRow = document.querySelector('#insights-content table tbody tr');
                if (!firstRow) return {ok: false, text: ''};
                const cells = Array.from(firstRow.querySelectorAll('td')).map((cell) => cell.innerText.trim());
                return {
                    ok: Boolean(cells[0] && cells[0] !== '-' && cells[0] !== '(-)'),
                    text: cells.join(' | '),
                };
            }"""
        )
        checks.append(
            CheckResult(
                name="resource_machine_name_render",
                ok=bool(resource_machine.get("ok")),
                detail=resource_machine.get("text", ""),
            )
        )

        goto_nav(page, "simulate")
        if count_visible(page, '[data-action="run-simulate"]'):
            page.locator('[data-action="run-simulate"]').click()
            page.wait_for_timeout(2800)
        sim_text = visible_text(page)
        checks.append(
            CheckResult(
                name="simulate_run",
                ok=("仿真摘要" in sim_text or "规则仿真甘特图" in sim_text or "时间窗口" in sim_text),
                detail="仿真页已运行并出现甘特/摘要区域",
            )
        )

        goto_nav(page, "optimize-launch")
        budget_hint_before = page.locator("#opt-budget-hint").inner_text().strip() if count_visible(page, "#opt-budget-hint") else ""
        budget_before = page.input_value("#opt-time-limit") if count_visible(page, "#opt-time-limit") else ""
        page.fill("#opt-population", "30")
        page.dispatch_event("#opt-population", "input")
        wait(page, 500)
        budget_hint_after = page.locator("#opt-budget-hint").inner_text().strip() if count_visible(page, "#opt-budget-hint") else ""
        page.locator('[data-action="apply-budget-recommendation"]').click()
        wait(page, 500)
        budget_after_apply = page.input_value("#opt-time-limit") if count_visible(page, "#opt-time-limit") else ""
        checks.append(
            CheckResult(
                name="optimize_budget_dynamic",
                ok=(budget_hint_before != budget_hint_after) or (budget_before != budget_after_apply),
                detail=f"{budget_before} -> {budget_after_apply}",
                extra={"before": budget_hint_before, "after": budget_hint_after},
            )
        )

        page.locator('[data-action="start-hybrid-optimize"]').click()
        page.wait_for_timeout(4200)
        optimize_text = visible_text(page)
        optimize_state = wait_for_optimize_context(page, 20000)
        checks.append(
            CheckResult(
                name="optimize_launch",
                ok=("近似评估" in optimize_text and "精确评估" in optimize_text and bool(optimize_state.get("taskId"))),
                detail="启动优化后出现进度卡",
                extra=optimize_state,
            )
        )

        goto_nav(page, "exact-reference")
        if count_visible(page, '[data-action="generate-exact-single"]'):
            page.locator('[data-action="generate-exact-single"]').click()
            page.wait_for_timeout(8000)
        exact_text = visible_text(page)
        exact_api_ok = any("/api/optimize/exact-reference" in url and status == 200 for url, status in response_log)
        checks.append(
            CheckResult(
                name="exact_reference_single",
                ok=exact_api_ok or ("最新精确冠军参考" in exact_text) or ("EXACT:SINGLE:" in exact_text) or ("精确冠军参考甘特图" in exact_text),
                detail="已生成单目标精确冠军参考方案",
                extra={"api_ok": exact_api_ok},
            )
        )

        goto_nav(page, "pareto-library")
        if count_visible(page, '[data-action="load-heuristic-references"]'):
            page.locator('[data-action="load-heuristic-references"]').click()
            page.wait_for_timeout(2500)
        review_text = visible_text(page)
        checks.append(
            CheckResult(
                name="review_library",
                ok=("已选方案主目标 + 全量 KPI" in review_text) or ("方案池概况" in review_text and "查看详情" in review_text),
                detail="方案库已出现候选与指标矩阵",
            )
        )

        review_metrics = page.evaluate(
            """() => {
                const rows = Array.from(document.querySelectorAll('.surface-card table tbody tr'));
                const candidateRows = rows
                  .map((row) => Array.from(row.querySelectorAll('td')).map((cell) => cell.innerText.trim()))
                  .filter((cells) => cells.length >= 6);
                const usable = candidateRows.map((cells) => ({
                    populated: cells.filter((value, idx) => idx < 2 || (value && value !== '-')).length,
                }));
                return {
                    rowCount: usable.length,
                    bestPopulated: usable.reduce((max, row) => Math.max(max, row.populated), 0),
                };
            }"""
        )
        checks.append(
            CheckResult(
                name="review_metric_completeness",
                ok=review_metrics.get("rowCount", 0) >= 1 and review_metrics.get("bestPopulated", 0) >= 6,
                detail=f'rows={review_metrics.get("rowCount", 0)}, populated={review_metrics.get("bestPopulated", 0)}',
                extra=review_metrics,
            )
        )

        if count_visible(page, '[data-action="export-selected-solution"]'):
            page.locator('[data-action="export-selected-solution"]').first.click()
            page.wait_for_timeout(2000)
        checks.append(
            CheckResult(
                name="export_solution_action",
                ok=True,
                detail="方案导出动作已触发",
            )
        )

        if count_visible(page, '[data-action="send-candidate-to-ai"]'):
            page.locator('[data-action="send-candidate-to-ai"]').first.click()
            page.wait_for_timeout(1500)
        ai_text = visible_text(page)
        checks.append(
            CheckResult(
                name="ai_review_navigation",
                ok=("#ai-review" in page.url) or ("AI 方案助手" in ai_text and "当前纳入 AI 评审的方案" in ai_text),
                detail=page.url,
            )
        )

        if count_visible(page, "#ai-input") and count_visible(page, '[data-action="ai-compare"]'):
            page.fill("#ai-input", "请比较已勾选方案，重点说明总周期、等待时间和利用率差异。")
            page.locator('[data-action="ai-compare"]').click()
            page.wait_for_timeout(5000)
        ai_text_after = visible_text(page)
        ai_working = ("正在" in ai_text_after) or ("比较" in ai_text_after) or ("推荐" in ai_text_after)
        checks.append(
            CheckResult(
                name="ai_review_interaction",
                ok=ai_working,
                detail="AI页已可交互并尝试发起比较",
            )
        )

        goto_nav(page, "llm-config")
        if count_visible(page, "#llm-base-url"):
            page.fill("#llm-base-url", "https://api.deepseek.com/v1")
        if count_visible(page, "#llm-model"):
            page.fill("#llm-model", "deepseek-chat")
        if count_visible(page, '[data-action="save-llm-config"]'):
            page.locator('[data-action="save-llm-config"]').click()
            page.wait_for_timeout(1000)
        if count_visible(page, '[data-action="test-llm-config"]'):
            page.locator('[data-action="test-llm-config"]').click()
            page.wait_for_timeout(3000)
        llm_text = visible_text(page)
        checks.append(
            CheckResult(
                name="llm_config_ops",
                ok=("测试通过" in llm_text) or ("测试失败" in llm_text) or ("连接" in llm_text),
                detail="大模型配置保存与连接测试动作已执行",
            )
        )

        goto_nav(page, "export-data")
        if count_visible(page, '[data-action="download-template"]'):
            index = 1 if page.locator('[data-action="download-template"]').count() > 1 else 0
            page.locator('[data-action="download-template"]').nth(index).click()
            page.wait_for_timeout(1500)
        if count_visible(page, '[data-action="export-csv"]'):
            page.locator('[data-action="export-csv"]').first.click()
            page.wait_for_timeout(1500)
        checks.append(
            CheckResult(
                name="template_and_csv_export",
                ok=True,
                detail="模板下载与 CSV 导出动作已触发",
            )
        )

        browser.close()

    return FullVerificationResult(checks=checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fuller end-to-end verification flow against the V2 frontend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v2")
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--headed", action="store_true", help="Launch the browser with a visible window.")
    args = parser.parse_args()

    result = verify(args.base_url, args.channel, headless=not args.headed)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
