# -*- coding: utf-8 -*-
"""
排产结果还原脚本
  输入:
    - 《方案*_排产.xlsx》  排产结果表 (* 为一~两个汉字, 可有多个方案)
    - 《20260710.xlsx》    原始数据表 (自动选取当前目录下日期最新的 YYYYMMDD.xlsx)
    - 《merge_report.xlsx》合并/拆分信息表 (工序还原依据)
  输出:
    - 《设备排产表_方案X.xlsx》 每个方案各生成一个

处理:
  步骤一: 依据 merge_report 逆向还原拆分子批和合并工序, 按比例/原工时顺排时间
  步骤二: 依据 任务令+work 从原始数据表补充承诺交期等字段
"""

import argparse
import math
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

# ---- 《merge_report.xlsx》的「拆分清单」sheet 列名 ----
SP_TYPE = '类型'
SP_PARENT_OP_ID = '拆分前op_id'
SP_CHILD_OP_ID = '子op_id'
SP_WORK_NUM = 'WORK编号'
SP_INDEX = '拆分序号'
SP_COUNT = '拆分数量'
SP_TOTAL_QTY = '原零件数量'
SP_CHILD_QTY = '子批零件数量'
SP_CHILD_TIME = '子processing_time_hrs'
SP_TURNOVER = '最终turnover_time_hrs'

META_WORK_NUM = '__restore_work_num'
META_SPLIT_QTY = '__restore_split_qty'
META_TURNOVER = '__restore_turnover'
META_FIXED_PRED = '__restore_fixed_predecessor'

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


def split_op_ids(v):
    result, seen = [], set()
    for token in re.split(r'[;；]', norm_str(v)):
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def join_op_ids(values):
    result, seen = [], set()
    for value in values:
        value = norm_str(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return ';'.join(result)


def positive_int_or_none(v):
    if v is None or pd.isna(v):
        return None
    try:
        number = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        return None
    return int(number)


def finite_float_or_none(v):
    if v is None or pd.isna(v):
        return None
    try:
        number = float(v)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def allocate_weighted_hours(total_hours, weights):
    """按权重将总工时分配到0.01小时, 保证合计不变。"""
    total_units = int(round(float(total_hours) * 100))
    safe_weights = [
        max(0, int(round((finite_float_or_none(weight) or 0.0) * 100)))
        for weight in weights
    ]
    total_weight = sum(safe_weights)
    if total_weight <= 0:
        safe_weights = [1] * len(weights)
        total_weight = len(weights)
    numerators = [total_units * weight for weight in safe_weights]
    units = [int(value // total_weight) for value in numerators]
    remainder = total_units - sum(units)
    order = sorted(
        range(len(weights)),
        key=lambda i: (-(numerators[i] % total_weight), i),
    )
    for index in order[:remainder]:
        units[index] += 1
    return [round(value / 100.0, 2) for value in units]


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


def build_split_map(split_df):
    """构建 子op_id -> 拆分前工序/子批数量/工时/周转 的查找表。"""
    child_map = {}
    if split_df is None or split_df.empty or SP_TYPE not in split_df.columns:
        return child_map
    required = {SP_PARENT_OP_ID, SP_CHILD_OP_ID, SP_INDEX, SP_COUNT,
                SP_TOTAL_QTY, SP_CHILD_QTY, SP_CHILD_TIME, SP_TURNOVER}
    if not required.issubset(set(split_df.columns)):
        print(f"[警告] 拆分清单缺少列 {sorted(required - set(split_df.columns))}, 将跳过拆分还原")
        return child_map

    rows = split_df[split_df[SP_TYPE].apply(norm_str) == '已拆分']
    for _, row in rows.iterrows():
        child_id = norm_str(row.get(SP_CHILD_OP_ID))
        parent_id = norm_str(row.get(SP_PARENT_OP_ID))
        split_index = positive_int_or_none(row.get(SP_INDEX))
        split_count = positive_int_or_none(row.get(SP_COUNT))
        child_quantity = positive_int_or_none(row.get(SP_CHILD_QTY))
        total_quantity = positive_int_or_none(row.get(SP_TOTAL_QTY))
        if not child_id or not parent_id or split_index is None or split_count is None:
            continue
        child_time = finite_float_or_none(row.get(SP_CHILD_TIME))
        turnover = finite_float_or_none(row.get(SP_TURNOVER))
        work_num = pd.to_numeric(pd.Series([row.get(SP_WORK_NUM)]), errors='coerce').iloc[0] \
            if SP_WORK_NUM in split_df.columns else float('nan')
        child_map[child_id] = {
            'parent_op_id': parent_id,
            'split_index': split_index,
            'split_count': split_count,
            'total_quantity': total_quantity,
            'child_quantity': child_quantity,
            'child_time': child_time,
            'turnover': TURNOVER_TIME_HRS if turnover is None else turnover,
            'work_num': None if pd.isna(work_num) else int(work_num),
        }
    return child_map


def restore_operations(plan_df, seg_map, tail_map=None, split_map=None):
    """逆向还原“先合并、后拆分”的排产工序。

    拆分子批若来自合并工序, 则按原合并段工时权重展开, 并为每个
    原 op_id 保留子批后缀。外部多前置在全部行展开后再一次性重定向到
    各计划工序的还原段末, 段内链条保持不变。
    """
    del tail_map  # 新逻辑通过排程行的实际展开结果构建段末映射
    split_map = split_map or {}
    out_rows = []
    plan_tail_map = {}
    expanded_count = 0
    restored_split_children = set()

    for _, row in plan_df.iterrows():
        plan_op_id = norm_str(row[P_OP_ID])
        split_info = split_map.get(plan_op_id)
        parent_op_id = split_info['parent_op_id'] if split_info else plan_op_id
        seg = seg_map.get(parent_op_id)

        if not seg or len(seg) < 2:
            d = row.to_dict()
            d[META_WORK_NUM] = split_info.get('work_num') if split_info else None
            d[META_SPLIT_QTY] = split_info.get('child_quantity') if split_info else None
            d[META_TURNOVER] = split_info.get('turnover') if split_info else None
            d[META_FIXED_PRED] = False
            out_rows.append(d)
            plan_tail_map[plan_op_id] = plan_op_id
            if split_info:
                restored_split_children.add(plan_op_id)
            continue

        expanded_count += 1
        split_suffix = f"__S{split_info['split_index']:02d}" if split_info else ""
        if split_info:
            restored_split_children.add(plan_op_id)
            total_duration = split_info.get('child_time')
            if total_duration is None:
                total_duration = finite_float_or_none(row.get(P_DUR))
            if total_duration is None:
                total_duration = sum(item['proc_time'] for item in seg)
            durations = allocate_weighted_hours(
                float(total_duration), [item['proc_time'] for item in seg],
            )
        else:
            durations = [item['proc_time'] for item in seg]

        start_t = pd.to_datetime(row.get(P_START_T), errors='coerce')
        start_h = pd.to_numeric(pd.Series([row.get(P_START_H)]), errors='coerce').iloc[0]
        cursor_t = start_t
        cursor_h = start_h
        previous_restored_id = None

        for index, (item, duration) in enumerate(zip(seg, durations)):
            d = row.to_dict()
            restored_id = f"{item['old_op_id']}{split_suffix}"
            restored_name = item['old_op_name']
            if split_info:
                restored_name = (
                    f"{restored_name}[拆{split_info['split_index']}/{split_info['split_count']}]"
                )
            d[P_OP_ID] = restored_id
            d[P_OP_NAME] = restored_name
            d[P_DUR] = round2(duration)
            d[P_OCCUPY] = round2(duration)

            if pd.notna(cursor_t):
                end_t = cursor_t + pd.Timedelta(hours=duration)
                d[P_START_T] = cursor_t
                d[P_END_T] = end_t
                cursor_t = end_t
            if pd.notna(cursor_h):
                end_h = cursor_h + duration
                d[P_START_H] = round2(cursor_h)
                d[P_END_H] = round2(end_h)
                cursor_h = end_h

            if index == 0:
                d[P_PRED_OP] = norm_str(row.get(P_PRED_OP))
                d[META_FIXED_PRED] = False
            else:
                d[P_PRED_OP] = previous_restored_id
                d[META_FIXED_PRED] = True
            d[META_WORK_NUM] = item['work_num']
            d[META_SPLIT_QTY] = split_info.get('child_quantity') if split_info else None
            if index < len(seg) - 1:
                d[META_TURNOVER] = 0.0
            else:
                d[META_TURNOVER] = (
                    split_info.get('turnover') if split_info else item.get('turnover')
                )
            out_rows.append(d)
            previous_restored_id = restored_id

        plan_tail_map[plan_op_id] = previous_restored_id

    df = pd.DataFrame(out_rows)
    if P_PRED_OP not in df.columns:
        df[P_PRED_OP] = ""

    for index, row in df.iterrows():
        if bool(row.get(META_FIXED_PRED)):
            continue
        redirected = [
            plan_tail_map.get(predecessor_id, predecessor_id)
            for predecessor_id in split_op_ids(row.get(P_PRED_OP))
        ]
        df.at[index, P_PRED_OP] = join_op_ids(redirected)

    print(f"[还原] 识别拆分子工序 {len(restored_split_children)} 道, "
          f"展开合并工序 {expanded_count} 段, 行数 {len(plan_df)} -> {len(df)}")
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

    # ---- work: 拆分/合并还原行优先使用清单中的 WORK 编号 ----
    if META_WORK_NUM in df.columns:
        df['work'] = [
            f"WORK{work_num}" if (work_num := positive_int_or_none(meta_work)) is not None
            else work_from_op_id(op_id)
            for op_id, meta_work in zip(df[P_OP_ID], df[META_WORK_NUM])
        ]
    else:
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

    # ---- 零件数量: 拆分行写子批数量, 其余保持原表逻辑 ----
    if S_PART_QTY not in src_columns:
        print(f"[警告] 原始数据表缺少 {S_PART_QTY} 列, 非拆分行的零件数量留空")
    split_quantities = df[META_SPLIT_QTY] if META_SPLIT_QTY in df.columns else [None] * len(df)
    df['零件数量'] = [
        parsed_quantity
        if (parsed_quantity := positive_int_or_none(split_quantity)) is not None
        else (get_src(task, work, S_PART_QTY) if S_PART_QTY in src_columns else "")
        for task, work, split_quantity in zip(df[P_TASK], df['work'], split_quantities)
    ]

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

    effective_turnover_map = dict(turnover_map)
    if META_TURNOVER in df.columns:
        for op_id, value in zip(df[P_OP_ID], df[META_TURNOVER]):
            parsed = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
            if pd.notna(parsed):
                effective_turnover_map[norm_str(op_id)] = float(parsed)

    ready_vals = []
    for task, work, pred in zip(df[P_TASK], df['work'], df[P_PRED_OP]):
        predecessor_ids = split_op_ids(pred)
        if predecessor_ids:
            gates = []
            for predecessor_id in predecessor_ids:
                pred_end = end_by_op.get(predecessor_id)
                if pred_end is None or pd.isna(pred_end):
                    gates = []
                    break
                hours = effective_turnover_map.get(predecessor_id, TURNOVER_TIME_HRS)
                gates.append(pred_end + pd.Timedelta(hours=float(hours)))
            ready_vals.append(max(gates) if gates else pd.NaT)
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

    df = df.drop(columns=[META_WORK_NUM, META_SPLIT_QTY, META_TURNOVER, META_FIXED_PRED],
                 errors='ignore')

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

def process_one(plan_path, src_df, merge_df, out_dir, split_df=None):
    print(f"\n===== 处理: {plan_path.name} =====")
    plan_df = pd.read_excel(plan_path)
    plan_df.columns = [norm_str(c) for c in plan_df.columns]
    print(f"[读取] 排产结果 {len(plan_df)} 行")

    for col in (P_TASK, P_OP_ID, P_START_T, P_END_T):
        if col not in plan_df.columns:
            sys.exit(f"[错误] {plan_path.name} 缺少列: {col}\n实际列: {list(plan_df.columns)}")

    seg_map, tail_map, turnover_map = build_merge_map(merge_df)
    split_map = build_split_map(split_df)
    print(f"[合并表] 可还原的合并段 {len(seg_map)} 个")
    print(f"[拆分表] 可还原的拆分子工序 {len(split_map)} 道")

    df = restore_operations(plan_df, seg_map, tail_map, split_map)
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
    ap.add_argument("--merge", default=None,
                    help="合并/拆分信息表路径(默认 <dir>/merge_report.xlsx)")
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
        report_book = pd.ExcelFile(merge_path)
        merge_sheet = '合并清单' if '合并清单' in report_book.sheet_names else report_book.sheet_names[0]
        merge_df = pd.read_excel(merge_path, sheet_name=merge_sheet)
        merge_df.columns = [norm_str(c) for c in merge_df.columns]
        if '拆分清单' in report_book.sheet_names:
            split_df = pd.read_excel(merge_path, sheet_name='拆分清单')
            split_df.columns = [norm_str(c) for c in split_df.columns]
        else:
            split_df = None
        report_book.close()
        print(f"[读取] 合并/拆分信息表 {merge_path.name}: "
              f"合并 {len(merge_df)} 行, 拆分 {len(split_df) if split_df is not None else 0} 行")
    else:
        merge_df = None
        split_df = None
        print(f"[警告] 未找到合并/拆分信息表 {merge_path}, 将跳过工序还原")

    plan_files = sorted(p for p in work_dir.iterdir()
                        if p.is_file() and re.match(PLAN_FILE_PATTERN, p.name))
    if not plan_files:
        sys.exit(f"[错误] 在 {work_dir} 下未找到《方案*_排产.xlsx》文件")
    print(f"[发现] 排产结果文件 {len(plan_files)} 个: {[p.name for p in plan_files]}")

    for p in plan_files:
        process_one(p, src_df, merge_df, out_dir, split_df)


if __name__ == "__main__":
    main()
