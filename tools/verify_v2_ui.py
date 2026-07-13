from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from playwright.sync_api import sync_playwright


@dataclass
class V2SmokeResult:
    scene_generate_label: str
    scene_generate_ok: bool
    downtime_before: int
    downtime_after: int
    downtime_add_visible: bool
    budget_hint_before: str
    budget_before: str
    budget_hint_after_tune: str
    budget_after_apply: str
    budget_dynamic_changed: bool
    optimize_status_text: str
    optimize_status_card_present: bool


def run_v2_smoke(base_url: str, headless: bool = True, channel: str = "msedge") -> V2SmokeResult:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=headless)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(base_url, wait_until="networkidle", timeout=30000)

        page.locator('[data-action="goto-new-scene"]').click()
        page.wait_for_timeout(600)
        page.fill("#gen-orders", "6")
        page.fill("#gen-tasks-min", "1")
        page.fill("#gen-tasks-max", "1")
        page.fill("#gen-ops-min", "1")
        page.fill("#gen-ops-max", "1")
        page.fill("#gen-machines", "3")
        page.fill("#gen-toolings", "1")
        page.fill("#gen-personnel", "1")
        page.locator('[data-action="generate-instance"]').click()
        page.wait_for_timeout(2500)
        current_label = page.locator('[data-action="goto-scene-library"]').inner_text().strip()

        page.locator('[data-nav="config"]').click()
        page.wait_for_timeout(300)
        page.locator('[data-nav="downtime-management"]').click()
        page.wait_for_timeout(1200)
        before_count = page.locator("tr[data-downtime-id]").count()
        page.select_option("#downtime-machine", index=0)
        page.select_option("#downtime-type", "planned")
        start_dt = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
        end_dt = (datetime.now() + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M")
        page.fill("#downtime-start", start_dt)
        page.fill("#downtime-end", end_dt)
        page.locator('[data-action="add-downtime"]').click()
        page.wait_for_timeout(2000)
        after_count = page.locator("tr[data-downtime-id]").count()

        page.locator('[data-nav="workflow"]').click()
        page.wait_for_timeout(300)
        page.locator('[data-nav="optimize-launch"]').click()
        page.wait_for_timeout(1200)
        hint_before = page.locator("#opt-budget-hint").inner_text().strip()
        budget_before = page.input_value("#opt-time-limit")
        page.fill("#opt-population", "30")
        page.dispatch_event("#opt-population", "input")
        page.wait_for_timeout(400)
        hint_after_tune = page.locator("#opt-budget-hint").inner_text().strip()
        page.locator('[data-action="apply-budget-recommendation"]').click()
        page.wait_for_timeout(400)
        budget_after_apply = page.input_value("#opt-time-limit")

        page.locator('[data-action="start-hybrid-optimize"]').click()
        page.wait_for_timeout(4000)
        status_card_present = page.locator("text=近似评估").count() > 0 and page.locator("text=精确评估").count() > 0
        optimize_status_text = ""
        if page.locator("text=running").count():
            optimize_status_text = page.locator("text=running").first.inner_text().strip()
        elif page.locator("text=done").count():
            optimize_status_text = page.locator("text=done").first.inner_text().strip()

        browser.close()

    return V2SmokeResult(
        scene_generate_label=current_label,
        scene_generate_ok=("6 单" in current_label) or ("6" in current_label),
        downtime_before=before_count,
        downtime_after=after_count,
        downtime_add_visible=after_count >= before_count + 1,
        budget_hint_before=hint_before,
        budget_before=budget_before,
        budget_hint_after_tune=hint_after_tune,
        budget_after_apply=budget_after_apply,
        budget_dynamic_changed=(hint_before != hint_after_tune) or (budget_before != budget_after_apply),
        optimize_status_text=optimize_status_text,
        optimize_status_card_present=status_card_present,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke test against the V2 frontend.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v2")
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--headed", action="store_true", help="Launch the browser with a visible window.")
    args = parser.parse_args()

    result = run_v2_smoke(args.base_url, headless=not args.headed, channel=args.channel)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
