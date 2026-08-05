# -*- coding: utf-8 -*-
"""
排产结果还原脚本
  输入:
    - 《方案*_排产.xlsx》  排产结果表 (* 为一~两个汉字, 可有多个方案)
    - 《20260710.xlsx》    原始数据表 (自动选取当前目录下日期最新的 YYYYMMDD.xlsx)
    - 《merge_report.xlsx》合并信息表 (工序还原依据)
  输出:
    - 《设备排产表_方案X.xlsx》 每个方案各生成一个

处理:
  步骤一: 依据 merge_report 将合并工序还原为多道原始工序, 按原工时依次顺排时间
  步骤二: 依据 任务令+work 从原始数据表补充承诺交期等字段
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================================================
# 全局参数
# ============================================================

TURNOVER_TIME_HRS = 2        # 默认周转时间(小时): 前工序完工后到下道工序可齐套
RELEASE_OFFSET_DAYS = 5      # 前工序为空时: 工艺排配时间 + 5 天
PLAN_FILE_PATTERN = r'^方案[\u4e00-\u9fa5]{1,2}_排产\.xlsx$'
SOURCE_FILE_PATTERN = r'^(\d{8})\.xlsx$'      # 原始数据表: 自动取日期最新的
DATETIME_FORMAT = 'YYYY-MM-DD HH:MM:SS'       # Excel 日期时间显示格式
DATETIME_COLUMNS = ['计划开工时间', '计划完工时间']   # 需统一为日期时间格式的列

# ---- 《方案*_排产.xlsx》列名 ----
P_PLAN = '计划号'
P_TASK = '任务令'
P_OP_ID = '工序ID'
P_OP_NAME = '工序'
P_PRED_OP = '前工序ID'
P_STATION = '计划工位'
P_MACHINE = '机器名称'
P_START_H = '开始(小时)'
P_END_H = '结束(小时)'
P_START_T = '计划开工时间'
P_END_T = '计划完工时间'
P_DUR = '时长(小时)'
P_OCCUPY = '占用时长(小时)'
P_DUE = '排产交期'

# ---- 原始数据表(YYYYMMDD.xlsx)列名 ----
S_TASK = '任务令'
S_WORK = 'WORK'
S_PLAN = '计划号'
S_MES_DUE = '【Mes+】交期*'
S_CHAIN = '齐套需求工序'
S_NEXT_PROC = '后工序'
S_WORKHOUR = '工时'
S_PART_QTY = '零件数量'
S_REMAIN = '剩余工时'
S_EARLIEST_START = '最晚开工时间'
S_PART_NO = '零件编号'
S_ORDER_TIME = '接单时间'
S_CRAFT_TIME = '工艺排配时间'
S_PRIORITY = '排产优先级'

# ---- 《merge_report.xlsx》列名 ----
M_TYPE = '类型'
M_TASK = 'task_id'
M_NEW_OP_ID = '合并后op_id'
M_NEW_OP_NAME = '合并后op_name'
M_OLD_OP_ID = '原op_id'
M_OLD_OP_NAME = '原op_name'
M_WORK_NUM = 'WORK编号'
M_OLD_TIME = '原processing_time_hrs'
M_NEW_TURNOVER = '合并后turnover_time_hrs'

# ---- 目标表列顺序(未列出的原有列排在最右) ----
TARGET_COLUMN_ORDER = [
    '计划号', '任务令', 'work', '承诺交期', '工序', '计划工位', '预计齐套时间',
    '计划开工时间', '计划完工时间', '排产优先级', '齐套需求工序', '后工序',
    '排产交期', '单件工时', '工位剩余总工时', '任务剩余总工时', '最晚开工时间',
    '零件号', '零件数量', '接单时间',
]


# ============================================================
# 工具函数
# ============================================================

def norm_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if pd.isna(v):
        return ""
    return str(v).strip()


def work_from_op_id(op_id):
    """工序ID中最后一个 '-' 后的数字 -> WORK{数字}"""
    s = norm_str(op_id)
    m = re.search(r'-(\d+)\s*$', s)
    return f"WORK{m.group(1)}" if m else ""


def round2(v):
    if v is None or pd.isna(v):
        return ""
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return ""


# ============================================================
# 步骤一: 工序还原
# ============================================================

def build_merge_map(merge_df):
    """构建 合并后op_id -> 该段的原工序明细(按WORK编号升序)

    返回:
      seg_map:  合并后op_id -> [ {原op_id, 原op_name, work_num, 原工时, 周转}, ... ]
      tail_map: 合并后op_id -> 段末原op_id (用于外部前工序ID重定向)
      turnover_map: 原op_id -> 合并后turnover_time_hrs (内部为0, 段末为2)
    """
    seg_map, tail_map, turnover_map = {}, {}, {}
    if merge_df is None or merge_df.empty:
        return seg_map, tail_map, turnover_map

    merged = merge_df[merge_df[M_TYPE].apply(norm_str) == '已合并'].copy()
    if merged.empty:
        return seg_map, tail_map, turnover_map

    merged[M_WORK_NUM] = pd.to_numeric(merged[M_WORK_NUM], errors='coerce')
    merged[M_OLD_TIME] = pd.to_numeric(merged[M_OLD_TIME], errors='coerce')
    merged[M_NEW_TURNOVER] = pd.to_numeric(merged[M_NEW_TURNOVER], errors='coerce')

    for new_op_id, grp in merged.groupby(merged[M_NEW_OP_ID].apply(norm_str)):
        if not new_op_id:
            continue
        grp = grp.sort_values(M_WORK_NUM, kind='mergesort')
        items = []
        for _, r in grp.iterrows():
            old_id = norm_str(r[M_OLD_OP_ID])
            turnover = r[M_NEW_TURNOVER]
            turnover = 0.0 if pd.isna(turnover) else float(turnover)
            items.append({
                'old_op_id': old_id,
                'old_op_name': norm_str(r[M_OLD_OP_NAME]),
                'work_num': r[M_WORK_NUM],
                'proc_time': 0.0 if pd.isna(r[M_OLD_TIME]) else float(r[M_OLD_TIME]),
                'turnover': turnover,
            })
            turnover_map[old_id] = turnover
        if items:
            seg_map[new_op_id] = items
            tail_map[new_op_id] = items[-1]['old_op_id']
    return seg_map, tail_map, turnover_map


def restore_operations(plan_df, seg_map, tail_map):
    """将合并工序还原为多道原始工序

    - 时间: 从合并行的计划开工时间起, 按各原工时依次顺排(段内无间隔)
    - 工序ID/工序: 取原op_id / 原op_name
    - 时长/占用时长: 取原工时
    - 前工序ID: 段首保留原值, 段内后续指向前一道原op_id
    - 外部指向合并后op_id 的前工序ID -> 重定向到段末原op_id
    """
    out_rows = []
    restored_seg_ids = set()   # 属于某个还原段内部的行(其前工序ID不参与重定向)
    expanded_count = 0

    for _, row in plan_df.iterrows():
        op_id = norm_str(row[P_OP_ID])
        seg = seg_map.get(op_id)
        if not seg or len(seg) < 2:
            out_rows.append(row.to_dict())
            continue

        expanded_count += 1
        start_t = pd.to_datetime(row.get(P_START_T), errors='coerce')
        start_h = pd.to_numeric(pd.Series([row.get(P_START_H)]), errors='coerce').iloc[0]
        cursor_t = start_t
        cursor_h = start_h
        prev_old_id = norm_str(row.get(P_PRED_OP))

        for k, item in enumerate(seg):
            d = row.to_dict()
            dur = item['proc_time']
            d[P_OP_ID] = item['old_op_id']
            d[P_OP_NAME] = item['old_op_name']
            d[P_DUR] = round2(dur)
            d[P_OCCUPY] = round2(dur)
            # 时间顺排
            if pd.notna(cursor_t):
                end_t = cursor_t + pd.Timedelta(hours=dur)
                d[P_START_T] = cursor_t
                d[P_END_T] = end_t
                cursor_t = end_t
            if pd.notna(cursor_h):
                end_h = cursor_h + dur
                d[P_START_H] = round2(cursor_h)
                d[P_END_H] = round2(end_h)
                cursor_h = end_h
            # 段内前工序链
            d[P_PRED_OP] = prev_old_id
            prev_old_id = item['old_op_id']
            if k > 0:
                restored_seg_ids.add(item['old_op_id'])
            out_rows.append(d)

    df = pd.DataFrame(out_rows)

    # 外部前工序ID重定向: 指向合并后op_id 的, 改指段末原op_id
    if tail_map and P_PRED_OP in df.columns:
        def _redirect(r):
            pred = norm_str(r[P_PRED_OP])
            # 段内后续行的前工序指向段首(=合并后op_id), 属于正常链条, 不重定向
            if norm_str(r[P_OP_ID]) in restored_seg_ids:
                return pred
            return tail_map.get(pred, pred)
        df[P_PRED_OP] = df.apply(_redirect, axis=1)

    print(f"[还原] 展开合并工序 {expanded_count} 段, 行数 {len(plan_df)} -> {len(df)}")
    return df


# ============================================================
# 步骤二: 补充字段
# ============================================================

def build_source_lookup(src_df):
    """构建 (任务令, WORK) -> 源表行 的查找表"""
    lookup = {}
    if S_TASK not in src_df.columns or S_WORK not in src_df.columns:
        sys.exit(f"[错误] 原始数据表缺少列 {S_TASK} 或 {S_WORK}\n实际列: {list(src_df.columns)}")
    for _, r in src_df.iterrows():
        key = (norm_str(r[S_TASK]), norm_str(r[S_WORK]).upper())
        if key not in lookup:      # 同键重复时保留首行
            lookup[key] = r
    return lookup


def enrich(df, lookup, turnover_map, src_columns):
    """补充目标表字段"""
    def get_src(task, work, col):
        r = lookup.get((norm_str(task), norm_str(work).upper()))
        if r is None or col not in src_columns:
            return None
        return r[col]

    # ---- work ----
    df['work'] = df[P_OP_ID].apply(work_from_op_id)

    # ---- 直接映射的字段 ----
    # 注: 计划号 会覆盖排产结果表中原有的值, 统一以原始数据表为准
    simple_map = [
        ('计划号', S_PLAN),
        ('承诺交期', S_MES_DUE),
        ('齐套需求工序', S_CHAIN),
        ('后工序', S_NEXT_PROC),
        ('工位剩余总工时', S_REMAIN),
        ('任务剩余总工时', S_REMAIN),
        ('最晚开工时间', S_EARLIEST_START),
        ('零件号', S_PART_NO),
        ('零件数量', S_PART_QTY),
        ('接单时间', S_ORDER_TIME),
        ('排产优先级', S_PRIORITY),
    ]
    missing_cols = []
    for tgt, src_col in simple_map:
        if src_col not in src_columns:
            if src_col not in missing_cols:
                missing_cols.append(src_col)
            df[tgt] = ""
            continue
        df[tgt] = [get_src(t, w, src_col) for t, w in zip(df[P_TASK], df['work'])]
    if missing_cols:
        print(f"[警告] 原始数据表缺少列 {missing_cols}, 对应目标字段留空")

    # ---- 单件工时 = 源表【工时】 ----
    if S_WORKHOUR in src_columns:
        df['单件工时'] = [get_src(t, w, S_WORKHOUR) for t, w in zip(df[P_TASK], df['work'])]
    else:
        print(f"[警告] 原始数据表缺少 {S_WORKHOUR} 列, 单件工时留空")
        df['单件工时'] = ""

    # ---- 预计齐套时间 ----
    # 前工序ID非空: 前工序的计划完工时间 + 该前工序的周转时间(合并段内部为0, 其余为2)
    # 前工序ID为空: 工艺排配时间 + 5 天
    end_by_op = {}
    for op_id, end_t in zip(df[P_OP_ID], pd.to_datetime(df[P_END_T], errors='coerce')):
        end_by_op[norm_str(op_id)] = end_t

    ready_vals = []
    for task, work, pred in zip(df[P_TASK], df['work'], df[P_PRED_OP]):
        pred_id = norm_str(pred)
        if pred_id:
            pred_end = end_by_op.get(pred_id)
            if pred_end is not None and pd.notna(pred_end):
                hrs = turnover_map.get(pred_id, TURNOVER_TIME_HRS)
                ready_vals.append(pred_end + pd.Timedelta(hours=float(hrs)))
            else:
                ready_vals.append(pd.NaT)
        else:
            craft = get_src(task, work, S_CRAFT_TIME)
            craft_t = pd.to_datetime(craft, errors='coerce')
            ready_vals.append(craft_t + pd.Timedelta(days=RELEASE_OFFSET_DAYS)
                              if pd.notna(craft_t) else pd.NaT)
    df['预计齐套时间'] = ready_vals

    # ---- 统一日期时间格式(转为真正的 datetime, 写出时套用 DATETIME_FORMAT) ----
    # 注: 必须用 format='mixed' 逐元素解析。pandas 2.0+ 默认会按首个元素推断统一格式,
    #     混用 "2026/7/29 8:00" 与 "2026-07-29 14:00:00" 时会把不匹配的静默置为 NaT
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')
            n_bad = int(df[col].isna().sum())
            if n_bad:
                print(f"[警告] {col} 有 {n_bad} 个值无法解析为日期时间, 已置空")

    return df


def find_latest_source(work_dir):
    """在目录下查找形如 YYYYMMDD.xlsx 的文件, 返回日期最新的那个"""
    candidates = []
    for p in work_dir.iterdir():
        if not p.is_file():
            continue
        m = re.match(SOURCE_FILE_PATTERN, p.name)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            continue
        candidates.append((dt, p))
    if not candidates:
        sys.exit(f"[错误] 在 {work_dir} 下未找到形如 20260729.xlsx 的原始数据表")
    candidates.sort(key=lambda x: x[0])
    dt, path = candidates[-1]
    others = [p.name for _, p in candidates[:-1]]
    print(f"[源表] 选用 {path.name} (共发现 {len(candidates)} 个候选"
          + (f", 其余: {others}" if others else "") + ")")
    return path


def reorder_columns(df):
    """按指定顺序排列, 未列出的原有列排在最右"""
    ordered = [c for c in TARGET_COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


# ============================================================
# 主流程
# ============================================================

def process_one(plan_path, src_df, merge_df, out_dir):
    print(f"\n===== 处理: {plan_path.name} =====")
    plan_df = pd.read_excel(plan_path)
    plan_df.columns = [norm_str(c) for c in plan_df.columns]
    print(f"[读取] 排产结果 {len(plan_df)} 行")

    for col in (P_TASK, P_OP_ID, P_START_T, P_END_T):
        if col not in plan_df.columns:
            sys.exit(f"[错误] {plan_path.name} 缺少列: {col}\n实际列: {list(plan_df.columns)}")

    seg_map, tail_map, turnover_map = build_merge_map(merge_df)
    print(f"[合并表] 可还原的合并段 {len(seg_map)} 个")

    df = restore_operations(plan_df, seg_map, tail_map)
    lookup = build_source_lookup(src_df)
    df = enrich(df, lookup, turnover_map, set(src_df.columns))
    df = reorder_columns(df)

    # 输出文件名: 设备排产表_方案X.xlsx
    m = re.match(r'^(方案[\u4e00-\u9fa5]{1,2})_排产\.xlsx$', plan_path.name)
    suffix = m.group(1) if m else plan_path.stem
    out_path = out_dir / f"设备排产表_{suffix}.xlsx"
    with pd.ExcelWriter(out_path, engine='openpyxl',
                        datetime_format=DATETIME_FORMAT) as writer:
        df.to_excel(writer, index=False, sheet_name="设备排产表")
    print(f"[完成] 已输出: {out_path} ({len(df)} 行, {len(df.columns)} 列)")
    return out_path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="排产结果工序还原与字段补充")
    ap.add_argument("--dir", default=str(here), help="输入文件所在目录(默认脚本同目录)")
    ap.add_argument("--source", default=None,
                    help="原始数据表路径(默认自动选取 <dir> 下日期最新的 YYYYMMDD.xlsx)")
    ap.add_argument("--merge", default=None, help="合并信息表路径(默认 <dir>/merge_report.xlsx)")
    ap.add_argument("--outdir", default=None, help="输出目录(默认同 dir)")
    args = ap.parse_args()

    work_dir = Path(args.dir).resolve()
    src_path = Path(args.source).resolve() if args.source else find_latest_source(work_dir)
    merge_path = Path(args.merge) if args.merge else work_dir / "merge_report.xlsx"
    out_dir = Path(args.outdir).resolve() if args.outdir else work_dir

    if not src_path.exists():
        sys.exit(f"[错误] 找不到原始数据表: {src_path}")
    src_df = pd.read_excel(src_path)
    src_df.columns = [norm_str(c) for c in src_df.columns]
    print(f"[读取] 原始数据表 {src_path.name}: {len(src_df)} 行")

    if merge_path.exists():
        merge_df = pd.read_excel(merge_path)
        merge_df.columns = [norm_str(c) for c in merge_df.columns]
        print(f"[读取] 合并信息表 {merge_path.name}: {len(merge_df)} 行")
    else:
        merge_df = None
        print(f"[警告] 未找到合并信息表 {merge_path}, 将跳过工序还原")

    plan_files = sorted(p for p in work_dir.iterdir()
                        if p.is_file() and re.match(PLAN_FILE_PATTERN, p.name))
    if not plan_files:
        sys.exit(f"[错误] 在 {work_dir} 下未找到《方案*_排产.xlsx》文件")
    print(f"[发现] 排产结果文件 {len(plan_files)} 个: {[p.name for p in plan_files]}")

    for p in plan_files:
        process_one(p, src_df, merge_df, out_dir)


if __name__ == "__main__":
    main()
