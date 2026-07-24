#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""夜间基准方案库构建（命令行入口）。

读当前实例 → 跑三维多目标大规模优化 → 抽取方案 → 质量门禁 → 落库（批次轮转，保留
最近两批）。夜间默认预算 3 小时；调试可用 --time-limit-s 缩短。

用法：
  # 生产（默认 3h）
  python tools/build_baseline.py --db llm4drd.db
  # 调试（60 秒）
  python tools/build_baseline.py --db llm4drd.db --time-limit-s 60
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# 允许 `python tools/build_baseline.py` 直接运行：把含 llm4drd 包的目录加入 sys.path。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from llm4drd.data.db import BaselineSolutionStore, InstanceStore, init_db  # noqa: E402
from llm4drd.optimization.baseline_build import (  # noqa: E402
    DEFAULT_OBJECTIVE_KEYS,
    NIGHTLY_COARSE_TIME_RATIO,
    NIGHTLY_GENERATIONS,
    NIGHTLY_STAGNATION_GENERATIONS,
    NIGHTLY_TIME_LIMIT_S,
    build_baseline_library,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="夜间基准方案库构建")
    parser.add_argument("--db", default="llm4drd.db", help="SQLite 数据库路径")
    parser.add_argument("--time-limit-s", type=int, default=NIGHTLY_TIME_LIMIT_S, help="总预算秒数（真正的搜索闸）")
    parser.add_argument("--generations", type=int, default=NIGHTLY_GENERATIONS, help="代数上限帽")
    parser.add_argument("--stagnation-generations", type=int, default=NIGHTLY_STAGNATION_GENERATIONS, help="连续无新解早停代数")
    parser.add_argument("--coarse-time-ratio", type=float, default=NIGHTLY_COARSE_TIME_RATIO, help="coarse 广搜时间占比")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective-keys",
        default=",".join(DEFAULT_OBJECTIVE_KEYS),
        help="逗号分隔的三个目标键（默认已定三维）",
    )
    args = parser.parse_args()

    init_db(args.db)
    store = InstanceStore(args.db)
    if not store.has_data():
        print("当前数据库没有实例数据，请先生成/导入实例。", file=sys.stderr)
        return 1
    shop = store.build_shopfloor()
    objective_keys = [key.strip() for key in args.objective_keys.split(",") if key.strip()]

    print(f"开始夜间构建：实例工序数={len(shop.operations)}，预算={args.time_limit_s}s，目标={objective_keys}")
    summary = build_baseline_library(
        shop,
        objective_keys=objective_keys,
        time_limit_s=args.time_limit_s,
        generations=args.generations,
        stagnation_generations=args.stagnation_generations,
        coarse_time_ratio=args.coarse_time_ratio,
        seed=args.seed,
        store=BaselineSolutionStore(args.db),
        db_path=args.db,
    )

    print(
        f"完成：批次 {summary['batch_id']}，抽取 {summary['extracted']} 个候选，"
        f"过门禁 {summary['passed']} 个入库。"
    )
    for row in summary["rows"]:
        improved = {
            key: round(info["improvement"], 3)
            for key, info in row["baseline_compare"]["objectives"].items()
        }
        print(f"  · {row['emphasis']:<40} 相对 ATC 改进比例 {improved}")
    if not summary["rows"]:
        print("  （本次无方案过门禁，库保持上一批不变）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
