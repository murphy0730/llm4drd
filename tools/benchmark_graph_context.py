from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..core.models import (
    Machine,
    MachineType,
    Operation,
    Order,
    Personnel,
    Shift,
    ShopFloor,
    Task,
    Tooling,
    ToolingType,
)
from ..data.db import init_db
from ..data.graph_artifact_store import GraphArtifactStore
from ..knowledge.context_service import GraphContextService
from ..optimization.hybrid_nsga3_alns import (
    HybridConfig,
    HybridNSGA3ALNSOptimizer,
)


SCHEMA_VERSION = 1
MIN_WARM_SPEEDUP = 2.0
MAX_RUNTIME_REGRESSION_PERCENT = 3.0
MAX_PEAK_MEMORY_RATIO = 1.25


def _parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must not contain duplicates")
    return sizes


def _make_shop(operation_count: int) -> ShopFloor:
    """Create a deterministic, bounded-depth synthetic scheduling instance."""
    shop = ShopFloor()
    shifts = [Shift(day=day, start_hour=0.0, hours=24.0) for day in range(21)]
    process_count = 8
    for process_index in range(process_count):
        process_id = f"P{process_index}"
        tooling_type_id = f"TL{process_index}"
        skill_id = f"SK{process_index}"
        shop.machine_types[process_id] = MachineType(
            process_id,
            f"Process {process_index}",
            is_critical=process_index == 0,
        )
        shop.tooling_types[tooling_type_id] = ToolingType(
            tooling_type_id, f"Tooling {process_index}"
        )
        for machine_index in range(2):
            machine_id = f"M{process_index}-{machine_index}"
            shop.machines[machine_id] = Machine(
                machine_id,
                f"Machine {process_index}-{machine_index}",
                process_id,
                shifts=list(shifts),
            )
            tooling_id = f"TL{process_index}-{machine_index}"
            shop.toolings[tooling_id] = Tooling(
                tooling_id,
                f"Tooling {process_index}-{machine_index}",
                tooling_type_id,
                shifts=list(shifts),
            )
            person_id = f"PERS{process_index}-{machine_index}"
            shop.personnel[person_id] = Personnel(
                person_id,
                f"Person {process_index}-{machine_index}",
                [skill_id],
                shifts=list(shifts),
            )

    width = max(3, len(str(operation_count)))
    operations_per_task = 10
    for task_number, start in enumerate(
        range(0, operation_count, operations_per_task)
    ):
        order_id = f"O{task_number:0{width}d}"
        task_id = f"T{task_number:0{width}d}"
        order = Order(
            order_id,
            f"Order {task_number}",
            due_date=float(operation_count * 4 + task_number),
            priority=1 + task_number % 3,
            task_ids=[task_id],
            main_task_id=task_id,
        )
        task = Task(
            task_id,
            order_id,
            f"Task {task_number}",
            is_main=True,
            due_date=order.due_date,
        )
        shop.orders[order_id] = order
        shop.tasks[task_id] = task

        previous_id = ""
        for operation_number in range(
            start, min(start + operations_per_task, operation_count)
        ):
            operation_id = f"OP{operation_number:0{width}d}"
            process_index = operation_number % process_count
            operation = Operation(
                operation_id,
                task_id,
                f"Operation {operation_number}",
                f"P{process_index}",
                1.0 + (operation_number % 5) * 0.25,
                predecessor_ops=[previous_id] if previous_id else [],
                eligible_machine_ids=(
                    [f"M{process_index}-{operation_number % 2}"]
                    if operation_number % 3 == 0
                    else []
                ),
                required_tooling_types=[f"TL{process_index}"],
                required_personnel_skills=[f"SK{process_index}"],
            )
            task.operations.append(operation)
            shop.operations[operation_id] = operation
            previous_id = operation_id

    shop.build_indexes()
    return shop


def _config(seed: int) -> HybridConfig:
    return HybridConfig(
        objective_keys=["total_tardiness", "makespan"],
        target_solution_count=2,
        population_size=4,
        generations=1,
        alns_iterations_per_candidate=0,
        time_limit_s=3600,
        parallel_workers=1,
        seed=seed,
    )


def _strip_wall_time(value) -> None:
    if isinstance(value, dict):
        value.pop("wall_time_ms", None)
        for nested in value.values():
            _strip_wall_time(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_wall_time(nested)


def _result_signature(result) -> str:
    payload = copy.deepcopy(result.to_export_dict())
    _strip_wall_time(payload["baseline"])
    _strip_wall_time(payload["solutions"])
    deterministic = {
        "baseline": payload["baseline"],
        "solutions": payload["solutions"],
        "archive_size": payload["archive_size"],
        "found_solution_count": payload["found_solution_count"],
        "generations_completed": payload["generations_completed"],
        "total_evaluations": payload["total_evaluations"],
        "approximate_evaluations": payload["approximate_evaluations"],
        "exact_evaluations": payload["exact_evaluations"],
    }
    encoded = json.dumps(
        deterministic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _measure(function: Callable):
    gc.collect()
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started

    gc.collect()
    tracemalloc.start()
    try:
        memory_result = function()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del memory_result
    return elapsed, peak, result


def _collect(
    function: Callable,
    runs: int,
    warmup: int,
    signature: Callable | None = None,
) -> dict:
    for _ in range(warmup):
        result = function()
        del result
        gc.collect()

    elapsed_samples: list[float] = []
    peak_samples: list[int] = []
    signatures: list[str] = []
    for _ in range(runs):
        elapsed, peak, result = _measure(function)
        elapsed_samples.append(elapsed)
        peak_samples.append(peak)
        if signature is not None:
            signatures.append(signature(result))
        del result

    return {
        "elapsed_seconds_samples": [round(value, 9) for value in elapsed_samples],
        "peak_memory_bytes_samples": peak_samples,
        "median_elapsed_seconds": round(statistics.median(elapsed_samples), 9),
        "median_peak_memory_bytes": int(statistics.median(peak_samples)),
        **({"signature_samples": signatures} if signature is not None else {}),
    }


def _legacy_initialization(shop: ShopFloor, config: HybridConfig):
    return HybridNSGA3ALNSOptimizer(shop, config, graph_context_mode="legacy")


def _legacy_end_to_end(shop: ShopFloor, config: HybridConfig):
    return _legacy_initialization(shop, config).run()


def _environment() -> dict:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit_hash = "unknown"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "commit_hash": commit_hash,
    }


def _base_payload(args, mode: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "parameters": {
            "sizes": args.sizes,
            "runs": args.runs,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "results": {},
    }


def _benchmark_baseline(args) -> dict:
    payload = _base_payload(args, "baseline")
    config = _config(args.seed)
    for size in args.sizes:
        print(f"baseline: {size} operations", flush=True)
        shop = _make_shop(size)
        payload["results"][str(size)] = {
            "operation_count": size,
            "legacy_initialization": _collect(
                lambda: _legacy_initialization(shop, config),
                args.runs,
                args.warmup,
            ),
            "legacy_end_to_end": _collect(
                lambda: _legacy_end_to_end(shop, config),
                args.runs,
                args.warmup,
                _result_signature,
            ),
        }
    return payload


def _benchmark_compare(args, baseline: dict | None) -> dict:
    payload = _base_payload(args, "compare")
    payload["thresholds"] = {
        "minimum_warm_initialization_speedup": MIN_WARM_SPEEDUP,
        "maximum_runtime_regression_percent": MAX_RUNTIME_REGRESSION_PERCENT,
        "maximum_peak_memory_ratio": MAX_PEAK_MEMORY_RATIO,
    }
    payload["baseline_reference"] = (
        {
            "generated_at": baseline.get("generated_at"),
            "commit_hash": baseline.get("environment", {}).get("commit_hash"),
        }
        if baseline
        else None
    )
    config = _config(args.seed)

    with tempfile.TemporaryDirectory(prefix="llm4drd-graph-benchmark-") as directory:
        for size in args.sizes:
            print(f"compare: {size} operations", flush=True)
            shop = _make_shop(size)
            db_path = str(Path(directory) / f"graph-{size}.db")
            init_db(db_path)
            store = GraphArtifactStore(db_path)

            legacy_initialization = _collect(
                lambda: _legacy_initialization(shop, config),
                args.runs,
                args.warmup,
            )

            cold_build = _collect(
                lambda: GraphContextService(store).get_or_build(
                    shop, force_rebuild=True
                )[0],
                args.runs,
                args.warmup,
            )

            def sqlite_initialization():
                context, diagnostics = GraphContextService(store).get_or_build(shop)
                if diagnostics.cache_level != "sqlite":
                    raise RuntimeError(
                        f"expected SQLite cache hit, got {diagnostics.cache_level}"
                    )
                return HybridNSGA3ALNSOptimizer(shop, config, context, "active")

            sqlite_optimizer_initialization = _collect(
                sqlite_initialization,
                args.runs,
                args.warmup,
            )

            l1_service = GraphContextService(store)
            l1_service.get_or_build(shop)

            def l1_initialization():
                context, diagnostics = l1_service.get_or_build(shop)
                if diagnostics.cache_level != "l1":
                    raise RuntimeError(
                        f"expected L1 cache hit, got {diagnostics.cache_level}"
                    )
                return HybridNSGA3ALNSOptimizer(shop, config, context, "active")

            l1_optimizer_initialization = _collect(
                l1_initialization,
                args.runs,
                args.warmup,
            )

            legacy_end_to_end = _collect(
                lambda: _legacy_end_to_end(shop, config),
                args.runs,
                args.warmup,
                _result_signature,
            )

            def active_end_to_end():
                context, diagnostics = l1_service.get_or_build(shop)
                if diagnostics.cache_level != "l1":
                    raise RuntimeError(
                        f"expected L1 cache hit, got {diagnostics.cache_level}"
                    )
                return HybridNSGA3ALNSOptimizer(
                    shop, config, context, "active"
                ).run()

            active_end_to_end_result = _collect(
                active_end_to_end,
                args.runs,
                args.warmup,
                _result_signature,
            )

            legacy_init_median = legacy_initialization["median_elapsed_seconds"]
            sqlite_init_median = sqlite_optimizer_initialization[
                "median_elapsed_seconds"
            ]
            l1_init_median = l1_optimizer_initialization["median_elapsed_seconds"]
            legacy_runtime = legacy_end_to_end["median_elapsed_seconds"]
            active_runtime = active_end_to_end_result["median_elapsed_seconds"]
            legacy_peak = legacy_end_to_end["median_peak_memory_bytes"]
            active_peak = active_end_to_end_result["median_peak_memory_bytes"]
            legacy_signatures = set(legacy_end_to_end["signature_samples"])
            active_signatures = set(active_end_to_end_result["signature_samples"])
            baseline_signatures = set(
                (baseline or {})
                .get("results", {})
                .get(str(size), {})
                .get("legacy_end_to_end", {})
                .get("signature_samples", [])
            )
            signatures_equal = (
                len(legacy_signatures) == 1
                and legacy_signatures == active_signatures
                and (not baseline_signatures or baseline_signatures == legacy_signatures)
            )
            derived = {
                "sqlite_warm_initialization_speedup": round(
                    legacy_init_median / sqlite_init_median, 4
                ),
                "l1_warm_initialization_speedup": round(
                    legacy_init_median / l1_init_median, 4
                ),
                "runtime_regression_percent": round(
                    (active_runtime / legacy_runtime - 1.0) * 100.0, 4
                ),
                "peak_memory_ratio": round(active_peak / legacy_peak, 4),
                "signatures_equal": signatures_equal,
            }
            payload["results"][str(size)] = {
                "operation_count": size,
                "legacy_initialization": legacy_initialization,
                "cold_build_and_sqlite_write": cold_build,
                "sqlite_load_and_optimizer_initialization": (
                    sqlite_optimizer_initialization
                ),
                "l1_hit_and_optimizer_initialization": l1_optimizer_initialization,
                "legacy_end_to_end": legacy_end_to_end,
                "active_end_to_end": active_end_to_end_result,
                "derived": derived,
            }

    acceptance_sizes = args.sizes[1:] if len(args.sizes) > 1 else args.sizes
    failures: list[str] = []
    for size in acceptance_sizes:
        derived = payload["results"][str(size)]["derived"]
        if derived["sqlite_warm_initialization_speedup"] < MIN_WARM_SPEEDUP:
            failures.append(
                f"{size}: SQLite warm initialization speedup "
                f"{derived['sqlite_warm_initialization_speedup']:.2f}x < "
                f"{MIN_WARM_SPEEDUP:.2f}x"
            )
        if derived["l1_warm_initialization_speedup"] < MIN_WARM_SPEEDUP:
            failures.append(
                f"{size}: L1 warm initialization speedup "
                f"{derived['l1_warm_initialization_speedup']:.2f}x < "
                f"{MIN_WARM_SPEEDUP:.2f}x"
            )
    for size in args.sizes:
        derived = payload["results"][str(size)]["derived"]
        if derived["runtime_regression_percent"] > MAX_RUNTIME_REGRESSION_PERCENT:
            failures.append(
                f"{size}: runtime regression "
                f"{derived['runtime_regression_percent']:.2f}% > "
                f"{MAX_RUNTIME_REGRESSION_PERCENT:.2f}%"
            )
        if not derived["signatures_equal"]:
            failures.append(f"{size}: optimization output signatures differ")
        if derived["peak_memory_ratio"] > MAX_PEAK_MEMORY_RATIO:
            failures.append(
                f"{size}: peak memory ratio {derived['peak_memory_ratio']:.3f} > "
                f"{MAX_PEAK_MEMORY_RATIO:.3f}"
            )
    payload["acceptance"] = {
        "passed": not failures,
        "speedup_sizes": acceptance_sizes,
        "failures": failures,
    }
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_report(baseline: dict | None, result: dict) -> str:
    environment = result["environment"]
    parameters = result["parameters"]
    lines = [
        "# Graph Context Benchmark Report",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "## Environment",
        "",
        f"- Python: `{environment['python_version']}`",
        f"- Platform: `{environment['platform']}`",
        f"- Processor: `{environment['processor'] or 'not reported'}`",
        f"- CPU count: `{environment['cpu_count']}`",
        f"- Commit: `{environment['commit_hash']}`",
        f"- Sizes: `{','.join(str(size) for size in parameters['sizes'])}`",
        f"- Runs / warmups: `{parameters['runs']} / {parameters['warmup']}`",
        f"- Seed: `{parameters['seed']}`",
        "",
        "## Results",
        "",
        "| Operations | Legacy init (ms) | Cold build + write (ms) | SQLite init (ms) | SQLite speedup | L1 init (ms) | L1 speedup | Runtime regression | Peak memory ratio | Signatures |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for size in parameters["sizes"]:
        item = result["results"][str(size)]
        derived = item["derived"]
        lines.append(
            "| {size} | {legacy:.3f} | {cold:.3f} | {sqlite:.3f} | "
            "{sqlite_speedup:.2f}x | {l1:.3f} | {l1_speedup:.2f}x | "
            "{regression:.2f}% | {memory:.3f}x | {signatures} |".format(
                size=size,
                legacy=item["legacy_initialization"]["median_elapsed_seconds"]
                * 1000,
                cold=item["cold_build_and_sqlite_write"][
                    "median_elapsed_seconds"
                ]
                * 1000,
                sqlite=item["sqlite_load_and_optimizer_initialization"][
                    "median_elapsed_seconds"
                ]
                * 1000,
                sqlite_speedup=derived["sqlite_warm_initialization_speedup"],
                l1=item["l1_hit_and_optimizer_initialization"][
                    "median_elapsed_seconds"
                ]
                * 1000,
                l1_speedup=derived["l1_warm_initialization_speedup"],
                regression=derived["runtime_regression_percent"],
                memory=derived["peak_memory_ratio"],
                signatures="match" if derived["signatures_equal"] else "DIFFER",
            )
        )

    acceptance = result["acceptance"]
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"Overall: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
            "",
            "The acceptance gates are at least 2.0x SQLite and L1 warm initialization speedup for all but the smallest requested fixture, no more than 3% end-to-end runtime regression, matching deterministic output signatures, and no more than 1.25x peak memory for every fixture.",
        ]
    )
    if acceptance["failures"]:
        lines.extend(["", "Failures:", ""])
        lines.extend(f"- {failure}" for failure in acceptance["failures"])
    if baseline:
        lines.extend(
            [
                "",
                "## Baseline Reference",
                "",
                f"- Generated: `{baseline.get('generated_at', 'unknown')}`",
                f"- Commit: `{baseline.get('environment', {}).get('commit_hash', 'unknown')}`",
            ]
        )
    lines.extend(
        [
            "",
            "Raw timing, memory, and signature samples are preserved in `graph-context-baseline.json` and `graph-context-result.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark legacy and cached graph-context optimizer paths."
    )
    parser.add_argument("--sizes", type=_parse_sizes, default=_parse_sizes("80,500,2500"))
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("baseline", "compare"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    output_dir = args.output_dir
    baseline_path = output_dir / "graph-context-baseline.json"
    if args.mode == "baseline":
        payload = _benchmark_baseline(args)
        _write_json(baseline_path, payload)
        print(f"wrote {baseline_path}")
        return 0

    baseline = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload = _benchmark_compare(args, baseline)
    result_path = output_dir / "graph-context-result.json"
    report_path = output_dir / "graph-context-report.md"
    _write_json(result_path, payload)
    report_path.write_text(_render_report(baseline, payload), encoding="utf-8")
    print(f"wrote {result_path}")
    print(f"wrote {report_path}")
    if not payload["acceptance"]["passed"]:
        for failure in payload["acceptance"]["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("acceptance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
