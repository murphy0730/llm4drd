from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from playwright.async_api import Page, async_playwright

from llm4drd.data.db import InstanceStore, WorkflowProgressStore, init_db
from llm4drd.tests.shop_fixtures import make_graph_context_shop


REVIEW_DATA_PATH = "/review-data"
ORDER_IDS = ("ORD-ALPHA-100", "ORD-BETA-200", "ORD-GAMMA-300")
PERCENT_RE = re.compile(r"^-?\d+(?:\.\d+)?%$|^-$")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    evidence: dict = field(default_factory=dict)


def _schedule(solution_index: int) -> list[dict]:
    schedule = []
    for order_index, order_id in enumerate(ORDER_IDS):
        for entry_index, start in enumerate((0.0, 30.0, 80.0, 140.0)):
            offset = solution_index * 0.75 + order_index * 0.2
            machine_id = ("M-C1", "M-C2", "M-A1")[entry_index % 3]
            schedule.append({
                "op_id": f"S{solution_index + 1}-{order_id}-OP{entry_index + 1}",
                "task_id": f"{order_id}-TASK",
                "order_id": order_id,
                "order_name": f"验收订单 {order_id}",
                "machine_id": machine_id,
                "machine_name": f"验收机器 {machine_id}",
                "start": start + offset,
                "end": start + offset + 8.0 + entry_index,
                "status": "scheduled",
            })
    return schedule


def prepare_fixture_database(db_path: Path) -> None:
    shop = make_graph_context_shop()
    init_db(str(db_path))
    InstanceStore(str(db_path)).save_from_shopfloor(shop)

    solutions = []
    export_solutions = []
    for index in range(4):
        solution_id = f"VERIFY-S-{index + 1}"
        objectives = {
            "total_tardiness": float(12 - index),
            "makespan": float(160 + index),
            "avg_net_available_utilization": 0.71 + index * 0.02,
            "critical_net_available_utilization": 0.68 + index * 0.02,
            "avg_utilization": 0.73 + index * 0.02,
            "critical_utilization": 0.69 + index * 0.02,
            "tooling_utilization": 0.55 + index * 0.01,
            "personnel_utilization": 0.61 + index * 0.01,
        }
        summary = {
            "total_operations": 12,
            "completed_operations": 12,
            "tardy_order_ids": [],
            "tardy_task_ids": [],
            "bottleneck_machine_ids": ["M-C1"],
        }
        candidate = {
            "solution_id": solution_id,
            "source": "pareto",
            "feasible": True,
            "evaluation_mode": "exact",
            "objectives": objectives,
            "metrics": objectives,
            "summary": summary,
        }
        solutions.append(candidate)
        export_solutions.append({**candidate, "schedule": _schedule(index)})

    task_id = "review-ui-verification"
    task = {
        "status": "done",
        "phase": "done",
        "result": {
            "objective_keys": [
                "total_tardiness",
                "makespan",
                "avg_net_available_utilization",
            ],
            "solutions": solutions,
            "archive_size": 4,
            "found_solution_count": 4,
            "generations_completed": 1,
            "elapsed_s": 0.1,
        },
        "export_result": {"solutions": export_solutions},
        "reference_solutions": [],
    }
    workflow = WorkflowProgressStore(str(db_path))
    workflow.save("validation", {
        "status": "passed",
        "error_count": 0,
        "warning_count": 0,
        "errors": [],
        "warnings": [],
        "stats": {"calendar": {"final_days": 14}},
    })
    workflow.save("optimization", {"task_id": task_id, "task": task})
    workflow.save("review", {
        "selection": [item["solution_id"] for item in solutions],
        "detail_id": solutions[0]["solution_id"],
        "ai_recommended_id": None,
    })


def _server_command(host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "llm4drd.api.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]


def _wait_for_server(base_url: str, process: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"fixture server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(base_url, timeout=0.5) as response:
                if response.status < 500:
                    return
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
        time.sleep(0.1)
    raise RuntimeError(f"fixture server did not become ready: {last_error}")


def _base_host_port(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("--fixture-server requires an http:// base URL")
    return parsed.hostname, parsed.port or 80


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


async def _select_order_with_keyboard(page: Page, query: str) -> None:
    selector = '[data-order-combobox="gantt-review-compare-order"] input'
    input_box = page.locator(selector)
    await input_box.wait_for(state="visible")
    await input_box.fill(query)
    option = page.locator(
        '[data-order-combobox="gantt-review-compare-order"] [role="option"]'
    ).first
    await option.wait_for(state="visible", timeout=5_000)
    await input_box.press("ArrowDown")
    await input_box.press("Enter")


async def _choose_order(page: Page, query: str, expected_order_id: str) -> None:
    await _select_order_with_keyboard(page, query)
    await page.wait_for_function(
        """expected => (
            app.reviewRead.orderId === expected &&
            !app.reviewRead.loading
        )""",
        arg=expected_order_id,
        timeout=10_000,
    )


async def verify(base_url: str, channel: str | None, headed: bool) -> list[Check]:
    checks: list[Check] = []
    console_errors: list[str] = []
    page_errors: list[str] = []
    review_data_requests: list[str] = []

    async with async_playwright() as playwright:
        launch_options = {"headless": not headed}
        if channel:
            launch_options["channel"] = channel
        browser = await playwright.chromium.launch(**launch_options)
        page = await browser.new_page(viewport={"width": 1680, "height": 1050})
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error" else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "request",
            lambda request: review_data_requests.append(request.url)
            if urllib.parse.urlparse(request.url).path.endswith(REVIEW_DATA_PATH)
            else None,
        )

        try:
            url = f"{base_url.rstrip('/')}/#solution-review"
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.locator("#review-comparison-region .compare-table").wait_for(
                state="visible", timeout=20_000
            )
            await page.wait_for_function(
                """() => (
                    app.reviewRead.orderId &&
                    !app.reviewRead.loading &&
                    app.ganttInstances.some(
                        item => item.canvasId === "gantt-review-compare" &&
                        item.el?.isConnected
                    )
                )""",
                timeout=20_000,
            )

            header_evidence = await page.locator(
                "#review-comparison-region .compare-table thead th"
            ).first.evaluate(
                """header => {
                    const visible = [];
                    const walk = node => {
                        if (node.nodeType === Node.TEXT_NODE) {
                            const text = node.textContent.trim();
                            if (text) visible.push(text);
                            return;
                        }
                        if (!(node instanceof Element)) return;
                        const style = getComputedStyle(node);
                        const rect = node.getBoundingClientRect();
                        const hidden = (
                            style.display === "none" ||
                            style.visibility === "hidden" ||
                            Number(style.opacity) === 0 ||
                            node.classList.contains("sr-only") ||
                            (rect.width <= 1 && rect.height <= 1)
                        );
                        if (!hidden) node.childNodes.forEach(walk);
                    };
                    header.childNodes.forEach(walk);
                    return {
                        visibleText: visible.join(" ").trim(),
                        accessibleText: header.textContent.trim(),
                    };
                }"""
            )
            checks.append(Check(
                "comparison_first_header_blank",
                header_evidence["visibleText"] == "",
                f'visible="{header_evidence["visibleText"]}"',
                header_evidence,
            ))

            utilization = await page.locator(
                "#review-utilization-region .util-table"
            ).evaluate(
                """table => {
                    const headers = Array.from(table.querySelectorAll("thead th"));
                    const cells = Array.from(
                        table.querySelectorAll("tbody tr td:not(:first-child)")
                    ).map(cell => cell.innerText.trim());
                    const headerEvidence = headers.map(header => {
                        const rect = header.getBoundingClientRect();
                        const name = header.querySelector(".util-plan-name");
                        return {
                            text: header.innerText.trim(),
                            width: rect.width,
                            scrollWidth: header.scrollWidth,
                            clientWidth: header.clientWidth,
                            titleMatches: !name || name.title === name.textContent.trim(),
                        };
                    });
                    return {
                        headerEvidence,
                        cells,
                        machineTypeWidth: headers[0]?.getBoundingClientRect().width || 0,
                    };
                }"""
            )
            percent_ok = bool(utilization["cells"]) and all(
                PERCENT_RE.fullmatch(value) for value in utilization["cells"]
            )
            checks.append(Check(
                "utilization_percentage_only",
                percent_ok,
                f'{len(utilization["cells"])} cells',
                {"cells": utilization["cells"]},
            ))
            headers_ok = bool(utilization["headerEvidence"]) and all(
                item["scrollWidth"] <= item["clientWidth"] + 1
                and item["titleMatches"]
                for item in utilization["headerEvidence"]
            )
            checks.append(Check(
                "utilization_headers_unclipped",
                headers_ok,
                f'machine type width={utilization["machineTypeWidth"]:.1f}px',
                {
                    "headers": utilization["headerEvidence"],
                    "machineTypeWidth": utilization["machineTypeWidth"],
                },
            ))

            review_data_requests.clear()
            await _choose_order(page, "BETA", "ORD-BETA-200")
            single_order_requests = list(review_data_requests)
            checks.append(Check(
                "keyboard_fuzzy_order_single_request",
                len(single_order_requests) == 1,
                f"{len(single_order_requests)} review-data request(s)",
                {
                    "selectedOrder": await page.evaluate(
                        "() => app.reviewRead.orderId"
                    ),
                    "requests": single_order_requests,
                },
            ))

            timeline = await page.evaluate(
                """() => {
                    const entry = app.ganttInstances.find(
                        item => item.canvasId === "gantt-review-compare" &&
                        item.el?.isConnected
                    );
                    const active = entry.timeline.getWindow();
                    const full = entry.data.fullWindow;
                    return {
                        activeStart: active.start.toISOString(),
                        activeEnd: active.end.toISOString(),
                        activeSpanMs: active.end - active.start,
                        fullStart: full.start,
                        fullEnd: full.end,
                        fullSpanMs: new Date(full.end) - new Date(full.start),
                    };
                }"""
            )
            max_active_ms = (96 * 60 + 1) * 60 * 1000
            timeline_ok = (
                timeline["fullSpanMs"] > 96 * 60 * 60 * 1000
                and timeline["activeSpanMs"] <= max_active_ms
            )
            checks.append(Check(
                "timeline_initial_window_at_most_96h",
                timeline_ok,
                (
                    f'active={timeline["activeSpanMs"] / 3_600_000:.3f}h, '
                    f'full={timeline["fullSpanMs"] / 3_600_000:.3f}h'
                ),
                timeline,
            ))

            first_delayed = True

            async def delay_first_order(route, request):
                nonlocal first_delayed
                query = urllib.parse.parse_qs(urllib.parse.urlparse(request.url).query)
                if (
                    first_delayed
                    and query.get("order_id") == ["ORD-ALPHA-100"]
                ):
                    first_delayed = False
                    await asyncio.sleep(0.75)
                try:
                    await route.continue_()
                except Exception:
                    # A second selection intentionally aborts the delayed request.
                    pass

            await page.route("**/review-data?*", delay_first_order)
            console_errors.clear()
            page_errors.clear()
            rapid_started = time.perf_counter()
            await _select_order_with_keyboard(page, "ALPHA")
            await page.wait_for_function(
                "() => app.reviewRead.loading === true",
                timeout=5_000,
            )
            await _choose_order(page, "GAMMA", "ORD-GAMMA-300")
            rapid_elapsed_ms = (time.perf_counter() - rapid_started) * 1000
            final_state = await page.evaluate(
                """() => ({
                    orderId: app.reviewRead.orderId,
                    inputValue: document.querySelector(
                        '[data-order-combobox="gantt-review-compare-order"] input'
                    )?.value || "",
                })"""
            )
            rapid_ok = (
                final_state["orderId"] == "ORD-GAMMA-300"
                and "ORD-GAMMA-300" in final_state["inputValue"]
                and not console_errors
                and not page_errors
            )
            checks.append(Check(
                "rapid_order_race_keeps_second",
                rapid_ok,
                (
                    f'final={final_state["orderId"]}, '
                    f"elapsed={rapid_elapsed_ms:.1f}ms, "
                    f"errors={len(console_errors) + len(page_errors)}"
                ),
                {
                    **final_state,
                    "elapsedMs": round(rapid_elapsed_ms, 3),
                    "consoleErrors": console_errors,
                    "pageErrors": page_errors,
                },
            ))
        except Exception as error:  # noqa: BLE001
            checks.append(Check(
                "verifier_execution",
                False,
                f"{type(error).__name__}: {error}",
                {"consoleErrors": console_errors, "pageErrors": page_errors},
            ))
        finally:
            await browser.close()
    return checks


def _run_with_fixture_server(
    base_url: str,
    channel: str | None,
    headed: bool,
) -> list[Check]:
    host, port = _base_host_port(base_url)
    if not _port_is_free(host, port):
        return [Check(
            "fixture_server_start",
            False,
            f"{host}:{port} is already in use; refusing to stop or overwrite it",
        )]
    repo_root = Path(__file__).resolve().parents[1]
    package_parent = repo_root.parent
    with tempfile.TemporaryDirectory(prefix="llm4drd-review-ui-") as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "review-ui.db"
        log_path = temp_path / "uvicorn.log"
        prepare_fixture_database(db_path)
        env = os.environ.copy()
        env["LLM4DRD_DB"] = str(db_path)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(package_parent), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                _server_command(host, port),
                cwd=package_parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_for_server(base_url, process)
                return asyncio.run(verify(base_url, channel, headed))
            except Exception as error:  # noqa: BLE001
                log.flush()
                tail = log_path.read_text(encoding="utf-8")[-4000:]
                return [Check(
                    "fixture_server_start",
                    False,
                    f"{type(error).__name__}: {error}",
                    {"serverLogTail": tail},
                )]
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify approved solution-review UI contracts in a real browser."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8888/")
    parser.add_argument(
        "--channel",
        default=None,
        help="Optional Playwright Chromium channel, for example chrome.",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument(
        "--fixture-server",
        action="store_true",
        help=(
            "Prepare an isolated temporary database, start uvicorn at --base-url, "
            "run verification, then stop it."
        ),
    )
    args = parser.parse_args()

    checks = (
        _run_with_fixture_server(args.base_url, args.channel, args.headed)
        if args.fixture_server
        else asyncio.run(verify(args.base_url, args.channel, args.headed))
    )
    payload = {
        "passed": all(check.ok for check in checks),
        "baseUrl": args.base_url,
        "fixtureServer": args.fixture_server,
        "checks": [asdict(check) for check in checks],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
