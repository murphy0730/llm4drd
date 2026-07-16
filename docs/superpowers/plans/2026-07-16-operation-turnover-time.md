# Operation Turnover Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为工序引入 `turnover_time`（流转等待时间），使四条排产路径一致满足 `后继.start_time ≥ 前驱.end_time + 前驱.turnover_time`。

**Architecture:** 在 `ShopFloor` 上新增 `get_operation_flow_ready_time(op)` 作为唯一的时间闸门实现（与既有 `get_operation_release_time(op)` 并列）。仿真器复用既有的 `release_check` 事件机制，把闸门从「静态 release_time」泛化为「release_time 与前驱 turnover 的较大者」；滚动排产、近似评价、CP-SAT 各自接入同一语义。`_is_op_ready` / `check_op_ready` 的纯布尔语义保持不变。

**Tech Stack:** Python 3、dataclasses、openpyxl、SQLite（stdlib `sqlite3`）、OR-Tools CP-SAT、`unittest`（既有测试框架，见 `tests/test_simulator_robustness.py`）

**Spec:** `docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md`

## Global Constraints

- **实施顺序**：本计划**排在 `docs/superpowers/plans/2026-07-16-unified-graph-context.md`（Task 1–8）之后**。Task 11 专门偿付该顺序的代价。
- **代码基线**：本计划行号基于 commit `1188508`（graph 改造尚未落地）。graph 改造落地后，Task 10 的落点必然变化，Task 2 的 `data/db.py` 落点可能变化——**每个 Task 开工前必须先 `grep` 定位，不得盲信行号**。
- **零回归是硬约束**：`turnover_time = 0` 时全系统行为必须与改造前逐工序完全一致。Task 1 Step 1 先固化这条基线。
- **默认值一律为 0**：新字段在模型、DDL、Excel 导入、DB 读取四处的缺省值都是 `0.0`，容忍缺列与 NULL。
- **turnover 允许 `0`，拒绝负值**——与 `processing_time` 必须 `> 0` 的规则不同。
- **自然时间口径**：turnover **不得**经过 `_joint_compute_effective_end`（该函数在 `core/simulator.py:892`、`optimization/approx_eval.py:34`、`data/db.py:812` 有三份副本，均不涉及本改动）。
- **闸门单点实现**：只允许 `ShopFloor.get_operation_flow_ready_time` 一处实现。任何第二处手写 `end_time + turnover_time` 都是缺陷。
- **测试运行方式**：`python -m pytest tests/ -v`（仓库使用 `unittest` 风格类，pytest 可直接收集）。
- **包导入路径**：测试从 `llm4drd.core.models` 导入，非相对路径。

## File Structure

### Modified files

| 文件 | 职责变化 |
|---|---|
| `core/models.py` | `Operation.turnover_time` 字段；新增 `ShopFloor.get_operation_flow_ready_time()`；关键路径与 `derived_start_time` 计入 turnover |
| `data/template_builder.py` | `operations` headers 增列；`TEMPLATE_VERSION` 递增 |
| `docs/instance_template.xlsx` | 由构建器重新生成（不手工编辑） |
| `data/db.py` | DDL、幂等迁移、Excel 导入、`update_operation`、行→`Operation` 加载 |
| `core/simulator.py` | 闸门接入；`_queue_release_or_ready` 时钟倒流加固 |
| `scheduling/online.py` | 闸门接入 |
| `optimization/approx_eval.py` | `base_ready` 计入 turnover |
| `optimization/exact.py` | CP-SAT 前驱约束加 turnover |
| `api/server.py` | 校验、payload、返回体 |
| `knowledge/graph.py` **或** graph 改造后的 `CanonicalGraphBuilder` | OP 节点属性（Task 10） |
| `tests/test_simulator_robustness.py` | `_build_shop` helper 增加 turnover 维度 |

### New files

| 文件 | 职责 |
|---|---|
| `tests/test_turnover_time.py` | turnover 语义的全部专项测试（口径 1–4、兼容性、双引擎一致） |

**决策记录**：spec §5.3 建议把闸门提到 `core/models.py` 模块级函数或新建 `core/precedence.py`。本计划改为 **`ShopFloor` 的实例方法**，因为 `core/models.py:733` 已有同族的 `get_operation_release_time(op)`，闸门是它的天然同胞；两个调用方（`Simulator` 持 `shop` 参数、`OnlineScheduler` 持 `self.sim_shop`）都能直接拿到 `ShopFloor`。这比新增文件少引入一个概念，且严格遵循既有模式。

---

### Task 1: 冻结零回归基线并引入 `turnover_time` 字段与模板列

**Files:**
- Create: `tests/test_turnover_time.py`
- Modify: `core/models.py:340-345`（`Operation` dataclass）
- Modify: `data/template_builder.py:11`（`TEMPLATE_VERSION`）、`:63-83`（`operations` sheet）
- Regenerate: `docs/instance_template.xlsx`

**Interfaces:**
- Produces: `Operation.turnover_time: float = 0.0` —— Task 2–10 全部依赖此字段。
- Produces: `operations` sheet 含 `turnover_time_hrs` 列，位置固定在 `processing_time_hrs` 与 `predecessor_ops` 之间 —— Task 2 的 Excel 导入依赖此列名。

- [ ] **Step 1: 固化零回归基线**

在改任何代码前，先跑既有测试并把结果存档，作为「turnover=0 时行为不变」的比对基准。

Run:
```bash
python -m pytest tests/ -v 2>&1 | tee /tmp/turnover-baseline.txt
tail -5 /tmp/turnover-baseline.txt
```

Expected: 全部通过。记下通过数（如 `47 passed`）。若基线本身就有失败，**停止并上报**——不得在红灯基线上叠加改动。

- [ ] **Step 2: 写失败测试（字段存在且默认为 0）**

创建 `tests/test_turnover_time.py`：

```python
"""工序流转等待时间（turnover_time）语义测试。

对应设计：docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md
"""
import unittest

from llm4drd.core.models import Operation


class TestTurnoverField(unittest.TestCase):
    def test_turnover_time_defaults_to_zero(self):
        """既有构造点不传 turnover_time 时必须落 0，这是零回归的锚点。"""
        op = Operation(id="OP1", task_id="T1", name="OP1", process_type="turning", processing_time=5.0)
        self.assertEqual(op.turnover_time, 0.0)

    def test_turnover_time_is_settable(self):
        op = Operation(
            id="OP1", task_id="T1", name="OP1", process_type="turning",
            processing_time=5.0, turnover_time=3.5,
        )
        self.assertEqual(op.turnover_time, 3.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试，确认失败**

Run: `python -m pytest tests/test_turnover_time.py -v`
Expected: FAIL —— `TypeError: Operation.__init__() got an unexpected keyword argument 'turnover_time'`（第二个测试），第一个测试报 `AttributeError`。

- [ ] **Step 4: 加字段**

在 `core/models.py` 的 `Operation` dataclass 中，`processing_time` 之后插入一行：

```python
@dataclass
class Operation:
    id: str
    task_id: str
    name: str
    process_type: str
    processing_time: float
    turnover_time: float = 0.0
    predecessor_ops: list[str] = field(default_factory=list)
```

**注意**：`turnover_time` 必须有默认值且位于所有无默认值字段之后，否则 dataclass 报 `non-default argument follows default argument`。`processing_time` 无默认值、`predecessor_ops` 有默认值，插在两者之间恰好合法。

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_turnover_time.py -v`
Expected: PASS（2 passed）

- [ ] **Step 6: 确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 通过数 = Step 1 基线数 + 2。**任何既有测试转红都必须停下排查**。

- [ ] **Step 7: 改模板构建器**

`data/template_builder.py`，两处改动。

其一，`TEMPLATE_VERSION` 递增：

```python
TEMPLATE_VERSION = "2026.07.16.1"
```

其二，`operations` sheet 的 headers 与示例行：

```python
    _add_sheet(
        wb,
        "operations",
        [
            "op_id",
            "task_id",
            "op_name",
            "process_type",
            "processing_time_hrs",
            "turnover_time_hrs",
            "predecessor_ops",
            "predecessor_tasks",
            "eligible_machine_ids",
            "required_tooling_types",
            "required_personnel_skills",
        ],
        [
            ["OP-0001-01-01", "T-0001-01", "Turning", "turning", 5.5, 2, "", "", "turning_1;turning_2", "tool_turning", "skill_turning"],
            ["OP-0001-01-02", "T-0001-01", "Milling", "milling", 3.2, 0, "OP-0001-01-01", "", "milling_1", "tool_milling", "skill_milling"],
            ["OP-0001-ASM", "T-0001-MAIN", "Assembly", "assembly", 6, 0, "", "T-0001-01", "assembly_1", "tool_assembly", "skill_assembly"],
        ],
    )
```

`OP-0001-01-01` 取非零值 `2`，使模板本身即可演示该语义（车削完工后等 2h 才能流转到铣削）。

- [ ] **Step 8: 写模板列序测试**

追加到 `tests/test_turnover_time.py`：

```python
import io

import openpyxl

from llm4drd.data.template_builder import build_instance_template_bytes


class TestTemplateColumn(unittest.TestCase):
    def _operations_headers(self) -> list[str]:
        wb = openpyxl.load_workbook(io.BytesIO(build_instance_template_bytes()))
        ws = wb["operations"]
        return [cell.value for cell in ws[1]]

    def test_turnover_column_follows_processing_time(self):
        headers = self._operations_headers()
        self.assertIn("turnover_time_hrs", headers)
        self.assertEqual(
            headers.index("turnover_time_hrs"),
            headers.index("processing_time_hrs") + 1,
            "turnover_time_hrs 必须紧跟在 processing_time_hrs 之后",
        )

    def test_template_demonstrates_nonzero_turnover(self):
        wb = openpyxl.load_workbook(io.BytesIO(build_instance_template_bytes()))
        ws = wb["operations"]
        headers = [cell.value for cell in ws[1]]
        column = headers.index("turnover_time_hrs")
        values = [row[column].value for row in ws.iter_rows(min_row=2)]
        self.assertTrue(any(v for v in values), "模板应至少有一行非零 turnover 以演示语义")
```

- [ ] **Step 9: 运行测试**

Run: `python -m pytest tests/test_turnover_time.py -v`
Expected: PASS（4 passed）

- [ ] **Step 10: 重新生成 xlsx**

**不要手工编辑 xlsx。** 由构建器生成：

```bash
python -c "
from llm4drd.data.template_builder import build_instance_template_bytes
open('docs/instance_template.xlsx','wb').write(build_instance_template_bytes())
print('regenerated')
"
```

Expected: 输出 `regenerated`

验证列已落地：
```bash
python -c "
import openpyxl
ws = openpyxl.load_workbook('docs/instance_template.xlsx')['operations']
print([c.value for c in ws[1]])
"
```
Expected: 输出的列表中 `'turnover_time_hrs'` 紧跟 `'processing_time_hrs'`。

- [ ] **Step 11: Commit**

```bash
git add tests/test_turnover_time.py core/models.py data/template_builder.py docs/instance_template.xlsx
git commit -m "feat: add turnover_time field and template column

Operation.turnover_time defaults to 0.0 so every existing construction
site keeps its current behavior. The operations sheet gains
turnover_time_hrs directly after processing_time_hrs."
```

---

### Task 2: 持久化（DDL、迁移、导入、更新、加载）

**Files:**
- Modify: `data/db.py:144-162`（`inst_operations` DDL）、`:227-236`（幂等迁移区）、`:390-420`（save INSERT）、`:475-500`（Excel 导入 INSERT）、`:565-590`（`update_operation`）、`:623`（行→`Operation`）
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `Operation.turnover_time`（Task 1）
- Produces: `inst_operations.turnover_time REAL DEFAULT 0` —— Task 9 的 API 层依赖此列；round-trip 后 `Operation.turnover_time` 保值。

> **开工前先定位**：graph 改造已在 `data/db.py` 新增 `graph_context_*` 系列表（graph plan Task 4），行号必然偏移。先跑：
> ```bash
> grep -n "CREATE TABLE IF NOT EXISTS inst_operations" -A 20 data/db.py
> grep -n "_safe_add_column(conn, \"inst_operations\"" data/db.py
> grep -n "INSERT INTO inst_operations" data/db.py
> grep -n "def update_operation" data/db.py
> grep -n "op = Operation(id=row\[\"op_id\"\]" data/db.py
> ```

- [ ] **Step 1: 写失败测试（round-trip 与旧库兼容）**

追加到 `tests/test_turnover_time.py`：

```python
import sqlite3
import tempfile
from pathlib import Path

from llm4drd.data.db import init_db


class TestTurnoverPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "test.db")
        init_db(self.db_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ddl_has_turnover_column(self):
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inst_operations)")}
        self.assertIn("turnover_time", columns)

    def test_null_turnover_in_legacy_row_loads_as_zero(self):
        """模拟旧库：turnover_time 为 NULL 的行加载必须落 0 而非崩溃。"""
        from llm4drd.data.db import _float_or_default
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO inst_operations (op_id, task_id, op_name, process_type, processing_time, turnover_time) "
                "VALUES ('OP1','T1','OP1','turning',5.0,NULL)"
            )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = dict(conn.execute("SELECT * FROM inst_operations WHERE op_id='OP1'").fetchone())
        self.assertEqual(_float_or_default(row.get("turnover_time"), 0.0), 0.0)

    def test_migration_is_idempotent_on_legacy_table(self):
        """旧库缺列时 init_db 必须补列且可重复执行。"""
        legacy = str(Path(self._tmp.name) / "legacy.db")
        with sqlite3.connect(legacy) as conn:
            conn.execute(
                "CREATE TABLE inst_operations (op_id TEXT PRIMARY KEY, task_id TEXT, "
                "op_name TEXT, process_type TEXT, processing_time REAL)"
            )
        init_db(legacy)
        init_db(legacy)  # 第二次必须不报错
        with sqlite3.connect(legacy) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(inst_operations)")}
        self.assertIn("turnover_time", columns)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_turnover_time.py::TestTurnoverPersistence -v`
Expected: FAIL —— `test_ddl_has_turnover_column` 报 `AssertionError: 'turnover_time' not found`；NULL 测试报 `sqlite3.OperationalError: table inst_operations has no column named turnover_time`。

- [ ] **Step 3: 加 DDL 列**

`data/db.py` 的 `inst_operations` 建表语句，在 `processing_time REAL,` 之后插入一行：

```sql
            CREATE TABLE IF NOT EXISTS inst_operations (
                op_id TEXT PRIMARY KEY,
                task_id TEXT,
                op_name TEXT,
                process_type TEXT,
                processing_time REAL,
                turnover_time REAL DEFAULT 0,
                predecessor_ops TEXT DEFAULT '',
```

- [ ] **Step 4: 加幂等迁移**

在既有 `_safe_add_column(conn, "inst_operations", ...)` 调用序列中追加一行（与其他并列即可）：

```python
        _safe_add_column(conn, "inst_operations", "turnover_time", "REAL DEFAULT 0")
```

`_safe_add_column`（`data/db.py:58`）内部吞掉 `sqlite3.OperationalError`，重复执行安全——这正是 Step 1 第三个测试要验证的。

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m pytest tests/test_turnover_time.py::TestTurnoverPersistence -v`
Expected: PASS（3 passed）

- [ ] **Step 6: 接通 save INSERT**

`data/db.py` 中 `INSERT INTO inst_operations` 的**第一处**（save/持久化 `ShopFloor` 路径）。列清单加 `turnover_time`、占位符加一个 `?`、值元组加 `op.turnover_time`：

```python
                conn.execute(
                    """
                    INSERT INTO inst_operations
                    (
                        op_id, task_id, op_name, process_type, processing_time, turnover_time, predecessor_ops, predecessor_tasks,
                        eligible_machine_ids, required_tooling_types, required_personnel_skills, initial_status,
                        initial_start_time, initial_end_time, initial_remaining_processing_time, initial_assigned_machine_id,
                        initial_assigned_tooling_ids, initial_assigned_personnel_ids
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        op_id,
                        op.task_id,
                        op.name,
                        op.process_type,
                        op.processing_time,
                        op.turnover_time,
                        ";".join(op.predecessor_ops),
                        ";".join(op.predecessor_tasks),
                        ...
```

**务必核对占位符数量**：原为 17 个 `?`，现为 18 个。数量不符会抛 `sqlite3.ProgrammingError`。

- [ ] **Step 7: 接通 Excel 导入 INSERT**

`data/db.py` 中 `INSERT INTO inst_operations` 的**第二处**（Excel 导入路径，值用 `_clean_scalar` / `_float_or_default` 包裹）。同样加列、加 `?`、加值：

```python
                        _float_or_default(row.get("processing_time_hrs", row.get("processing_time", 0)), 0.0),
                        _float_or_default(row.get("turnover_time_hrs", row.get("turnover_time", 0)), 0.0),
                        _clean_scalar(row.get("predecessor_ops"), ""),
```

`_float_or_default`（`data/db.py:28`）在值为 `None` 或空串时返回默认值 —— 旧 xlsx 缺列时 `row.get(...)` 返回 `0`，落 `0.0`。同时兼容 `turnover_time_hrs`（模板列名）与 `turnover_time`（裸列名），与 `processing_time` 的双名容忍写法一致。

- [ ] **Step 8: 接通 `update_operation`**

`data/db.py` 的 `update_operation`，SET 子句加字段、值元组加取值：

```python
                UPDATE inst_operations
                SET task_id=?, op_name=?, process_type=?, processing_time=?, turnover_time=?, predecessor_ops=?, predecessor_tasks=?, eligible_machine_ids=?, required_tooling_types=?, required_personnel_skills=?, initial_status=?, initial_start_time=?, initial_end_time=?, initial_remaining_processing_time=?, initial_assigned_machine_id=?, initial_assigned_tooling_ids=?, initial_assigned_personnel_ids=?
                WHERE op_id=?
                """,
                (
                    data["task_id"],
                    data["op_name"],
                    data["process_type"],
                    float(data["processing_time"]),
                    _float_or_default(data.get("turnover_time"), 0.0),
                    data.get("predecessor_ops", ""),
```

用 `_float_or_default` 而非 `float(data["turnover_time"])`：调用方可能不传该键（老的编辑 payload），必须容忍。

- [ ] **Step 9: 接通行→`Operation` 加载**

`data/db.py` 中构造 `Operation` 的那行，加 `turnover_time`：

```python
            op = Operation(id=row["op_id"], task_id=row["task_id"], name=row["op_name"], process_type=row["process_type"], processing_time=float(row["processing_time"]), turnover_time=_float_or_default(row.get("turnover_time"), 0.0), predecessor_ops=[token.strip() for token in row["predecessor_ops"].split(";") if token.strip()], ...)
```

用 `_float_or_default(row.get("turnover_time"), 0.0)` 而非 `float(row["turnover_time"])`：旧库经迁移补列后该值为 NULL，`float(None)` 会抛 `TypeError`。

- [ ] **Step 10: 写导入取值测试**

追加到 `TestTurnoverPersistence`：

```python
    def test_excel_import_without_turnover_column_defaults_to_zero(self):
        """旧 xlsx 缺列时导入必须落 0。"""
        from llm4drd.data.db import _float_or_default
        legacy_row = {"op_id": "OP1", "task_id": "T1", "processing_time_hrs": 5.0}
        self.assertEqual(
            _float_or_default(legacy_row.get("turnover_time_hrs", legacy_row.get("turnover_time", 0)), 0.0),
            0.0,
        )

    def test_excel_import_reads_turnover_column(self):
        from llm4drd.data.db import _float_or_default
        row = {"op_id": "OP1", "task_id": "T1", "processing_time_hrs": 5.0, "turnover_time_hrs": 2.5}
        self.assertEqual(
            _float_or_default(row.get("turnover_time_hrs", row.get("turnover_time", 0)), 0.0),
            2.5,
        )
```

> **补充 round-trip 覆盖**：上述两条只验证取值表达式，**不足以证明 INSERT/加载已接通**。实施者必须先勘查既有的 store 测试基建：
> ```bash
> ls tests/
> grep -rn "InstanceStore" tests/ | head
> ```
> 据实补一条完整的 save→load round-trip 断言（构造带非零 turnover 的 `ShopFloor` → 存 → 读 → 断言 `turnover_time` 保值）。若仓库无 store 测试先例，则以 Task 9 的 API 集成测试兜底，并在此注明。

- [ ] **Step 11: 运行测试**

Run: `python -m pytest tests/test_turnover_time.py -v`
Expected: PASS（全部通过）

- [ ] **Step 12: 确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 无既有测试转红。

- [ ] **Step 13: Commit**

```bash
git add data/db.py tests/test_turnover_time.py
git commit -m "feat: persist turnover_time

Adds the column with an idempotent migration so legacy databases
upgrade in place. NULL and missing-column values both read as 0.0."
```

---

### Task 3: `ShopFloor` 时间闸门（单点实现）

**Files:**
- Modify: `core/models.py:733-738`（在 `get_operation_release_time` 之后新增方法）
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `Operation.turnover_time`（Task 1）
- Produces:
  ```python
  ShopFloor.get_operation_flow_ready_time(op: Operation, release_time: float | None = None) -> float
  ```
  Task 4（仿真器）与 Task 8（滚动排产）**必须**调用此方法，不得自行手写 `end_time + turnover_time`。`release_time` 形参供调用方传入已缓存的放行时刻以跳过重算；传 `None` 时内部调 `get_operation_release_time(op)`。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_turnover_time.py`：

```python
from llm4drd.core.models import Machine, MachineType, Order, Shift, ShopFloor, Task


def _shop_with_two_ops(turnover: float) -> ShopFloor:
    """OP1 -> OP2 串行，OP1 的 turnover 可调。OP1 已完工于 t=10。"""
    op1 = Operation(id="OP1", task_id="T1", name="OP1", process_type="turning",
                    processing_time=5.0, turnover_time=turnover)
    op2 = Operation(id="OP2", task_id="T1", name="OP2", process_type="milling",
                    processing_time=3.0, predecessor_ops=["OP1"])
    task = Task(id="T1", order_id="O1", name="T1", due_date=100.0, operations=[op1, op2])
    calendar = [Shift(day=d, start_hour=0.0, hours=24.0) for d in range(30)]
    shop = ShopFloor(
        machine_types={"turning": MachineType(id="turning", name="turning"),
                       "milling": MachineType(id="milling", name="milling")},
        machines={"m1": Machine(id="m1", name="m1", type_id="turning", shifts=calendar),
                  "m2": Machine(id="m2", name="m2", type_id="milling", shifts=calendar)},
        orders={"O1": Order(id="O1", name="O1", due_date=100.0, task_ids=["T1"], main_task_id="T1")},
        tasks={"T1": task},
        operations={"OP1": op1, "OP2": op2},
    )
    shop.build_indexes()
    op1.end_time = 10.0
    return shop


class TestFlowReadyGate(unittest.TestCase):
    def test_gate_adds_predecessor_turnover_to_its_end_time(self):
        shop = _shop_with_two_ops(turnover=4.0)
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 14.0)

    def test_zero_turnover_gate_equals_predecessor_end(self):
        """零回归锚点：turnover=0 时闸门退化为前驱完工时刻。"""
        shop = _shop_with_two_ops(turnover=0.0)
        op2 = shop.operations["OP2"]
        self.assertEqual(
            shop.get_operation_flow_ready_time(op2),
            max(shop.get_operation_release_time(op2), 10.0),
        )

    def test_gate_honors_release_time_when_larger(self):
        shop = _shop_with_two_ops(turnover=1.0)
        shop.tasks["T1"].release_time = 50.0
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 50.0)

    def test_gate_accepts_precomputed_release_time(self):
        """调用方传入缓存的 release_time 时，结果必须与内部计算一致。"""
        shop = _shop_with_two_ops(turnover=4.0)
        op2 = shop.operations["OP2"]
        self.assertEqual(
            shop.get_operation_flow_ready_time(op2, release_time=0.0),
            shop.get_operation_flow_ready_time(op2),
        )

    def test_gate_tolerates_predecessor_without_end_time(self):
        """前驱尚未完工（end_time is None）时不得崩溃。"""
        shop = _shop_with_two_ops(turnover=4.0)
        shop.operations["OP1"].end_time = None
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 0.0)

    def test_gate_uses_max_over_task_predecessor_operations(self):
        """口径 2：任务级前驱取 max(各工序 end + 各自 turnover)。

        刻意让「完工最晚的工序」(OP4, end=12) 与「流转最晚的工序」
        (OP3, end=5 但 turnover=9) 不是同一道——若实现错写成「取任务
        completion_time 再加末工序 turnover」，此测试会得 13 而非 14。
        """
        shop = _shop_with_two_ops(turnover=1.0)
        early = Operation(id="OP3", task_id="T2", name="OP3", process_type="turning",
                          processing_time=2.0, turnover_time=9.0)
        late = Operation(id="OP4", task_id="T2", name="OP4", process_type="turning",
                         processing_time=2.0, turnover_time=1.0)
        early.end_time = 5.0   # 5 + 9 = 14
        late.end_time = 12.0   # 12 + 1 = 13
        shop.tasks["T2"] = Task(id="T2", order_id="O1", name="T2", due_date=100.0,
                                operations=[early, late])
        shop.operations["OP3"] = early
        shop.operations["OP4"] = late
        shop.operations["OP2"].predecessor_tasks = ["T2"]
        # max(OP1: 10+1=11, OP3: 14, OP4: 13) = 14
        self.assertEqual(shop.get_operation_flow_ready_time(shop.operations["OP2"]), 14.0)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_turnover_time.py::TestFlowReadyGate -v`
Expected: FAIL —— `AttributeError: 'ShopFloor' object has no attribute 'get_operation_flow_ready_time'`

- [ ] **Step 3: 实现闸门**

`core/models.py`，紧接 `get_operation_release_time` 之后插入：

```python
    def get_operation_flow_ready_time(self, op: Operation, release_time: float | None = None) -> float:
        """工序可开工的最早时刻：任务/订单放行 与 前驱流转完成 的较大者。

        前驱工序完工后，工件需再等 turnover_time 才能流转到本工序，故
        本工序最早开工时刻不早于 前驱.end_time + 前驱.turnover_time。
        turnover 按自然时间流逝，不参与班次推进。

        仅在前驱全部完成时返回终值；前驱未完工(end_time is None)时该前驱
        不贡献约束，调用方应先用 check_op_ready / _is_op_ready 判断就绪。

        release_time: 调用方已缓存的放行时刻，传 None 则内部计算。
        """
        gate = release_time if release_time is not None else self.get_operation_release_time(op)
        for predecessor_id in op.predecessor_ops:
            predecessor = self.operations.get(predecessor_id)
            if predecessor is not None and predecessor.end_time is not None:
                gate = max(gate, predecessor.end_time + predecessor.turnover_time)
        for predecessor_task_id in op.predecessor_tasks:
            predecessor_task = self.tasks.get(predecessor_task_id)
            if predecessor_task is None:
                continue
            for task_op in predecessor_task.operations:
                if task_op.end_time is not None:
                    gate = max(gate, task_op.end_time + task_op.turnover_time)
        return gate
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_turnover_time.py::TestFlowReadyGate -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 无既有测试转红（本 Task 只加方法，未改任何调用点）。

- [ ] **Step 6: Commit**

```bash
git add core/models.py tests/test_turnover_time.py
git commit -m "feat: add ShopFloor.get_operation_flow_ready_time

Single implementation of the turnover gate, sibling to the existing
get_operation_release_time. Task-level predecessors take the max over
(end_time + turnover_time) across all their operations."
```

---

### Task 4: 仿真器接入闸门 + 时钟倒流加固

**Files:**
- Modify: `core/simulator.py:270-280`（`release_check` handler）、`:426-429`（完工触发）、`:477-482`（热启动复核）、`:591-605`（`_queue_release_or_ready`）、`:719`（派工 probe）、`:748`（`wait_time` 特征）
- Modify: `tests/test_simulator_robustness.py:33-58`（`_build_shop` helper）
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `ShopFloor.get_operation_flow_ready_time(op, release_time=None)`（Task 3）
- Produces: 仿真器输出的 `schedule` 满足 `后继.start ≥ 前驱.end + 前驱.turnover`

**背景（实施者必读）**：仿真器已有「算出时刻 → 压 `release_check` 事件 → 到点重查就绪」的完整链路，今天只用于 task/order 的 `release_time`。本 Task 把闸门泛化，事件机制原样复用，**不新增事件类型**。`_is_op_ready`（`core/simulator.py:582`）保持纯布尔语义**不变**——时间判定全部在闸门里。

- [ ] **Step 1: 扩展测试 helper**

`tests/test_simulator_robustness.py` 的 `_build_shop`，让 `ops_spec` 容忍可选的第 5 元（turnover），保持既有 4 元调用点全部不变：

```python
def _build_shop(ops_spec, machines_spec, due_date: float = 100.0) -> ShopFloor:
    """ops_spec: [(op_id, process_type, hours, predecessor_ops)]
    或 [(op_id, process_type, hours, predecessor_ops, turnover_hours)]
    """
    operations = {}
    task = Task(id="T1", order_id="O1", name="T1", due_date=due_date, operations=[])
    for spec in ops_spec:
        op_id, process_type, hours, preds = spec[:4]
        turnover = spec[4] if len(spec) > 4 else 0.0
        op = Operation(
            id=op_id, task_id="T1", name=op_id, process_type=process_type,
            processing_time=hours, turnover_time=turnover, predecessor_ops=list(preds),
        )
        operations[op_id] = op
        task.operations.append(op)
```

其余部分（`machines` 循环起）保持原样不动。

- [ ] **Step 2: 确认既有测试仍绿**

Run: `python -m pytest tests/test_simulator_robustness.py -v`
Expected: PASS —— helper 向后兼容，既有 4 元调用点不受影响。

- [ ] **Step 3: 写失败测试**

> **先核对 schedule 条目键名**：下面用 `entry["op_id"] / ["start"] / ["end"]`。开工前跑 `grep -n "schedule.append" -A 12 core/simulator.py` 确认实际键名，据实调整（`api/server.py:3505` 一带用的是 `"duration"`，说明该结构有自己的约定）。

追加到 `tests/test_turnover_time.py`：

```python
from llm4drd.core.rules import BUILTIN_RULES
from llm4drd.core.simulator import Simulator

from tests.test_simulator_robustness import _build_shop, _full_calendar


def _run(shop):
    result = Simulator().run(shop, BUILTIN_RULES["FIFO"])
    return {entry["op_id"]: entry for entry in result.schedule}


class TestSimulatorTurnover(unittest.TestCase):
    def test_successor_waits_for_predecessor_turnover(self):
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 3.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertGreaterEqual(
            entries["OP2"]["start"], entries["OP1"]["end"] + 3.0 - 1e-9,
            "后继必须等满前驱的 turnover",
        )

    def test_zero_turnover_is_unchanged(self):
        """零回归：turnover=0 时后继紧接前驱开工。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 0.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertAlmostEqual(entries["OP2"]["start"], entries["OP1"]["end"], places=6)

    def test_machine_is_free_during_turnover(self):
        """口径 3：turnover 期间机床可接别的活——这是本字段的核心动机。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 100.0),   # 超长 turnover
             ("OP2", "milling", 2.0, ["OP1"]),
             ("OP3", "turning", 4.0, [])],          # 无前驱，抢同一台车床
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertLess(
            entries["OP3"]["start"], entries["OP1"]["end"] + 100.0,
            "OP3 不该等 OP1 的 turnover——turnover 不占用机床",
        )
```

- [ ] **Step 4: 运行测试，确认失败**

Run: `python -m pytest tests/test_turnover_time.py::TestSimulatorTurnover -v`
Expected: `test_successor_waits_for_predecessor_turnover` FAIL（后继紧接前驱开工，未等 turnover）；另两个 PASS（turnover 尚未生效，行为即今天的行为）。

- [ ] **Step 5: 加固 `_queue_release_or_ready` 并接入闸门**

这是本 Task 的核心。`core/simulator.py` 的 `_queue_release_or_ready` 整体替换为：

```python
    def _queue_release_or_ready(
        self,
        shop: ShopFloor,
        op: Operation,
        ready_ops: set[str],
        event_queue: list[Event],
        now: float = 0.0,
    ) -> None:
        gate = self._flow_ready_time(shop, op)
        if gate <= now:
            self._mark_ready(op, ready_ops)
            return
        if op.id in self._release_checks_scheduled:
            return
        self._release_checks_scheduled.add(op.id)
        self._push(event_queue, max(gate, now), "release_check", op_id=op.id)
```

**为什么必须传 `now`**：原实现判断 `if release_time <= 0`——与 `0` 比而非与 `now` 比，因为旧的 `release_time` 是静态的、`<= 0` 即等价于「已过期」。换成动态闸门后，若闸门值小于 `now` 就会向事件队列压入**过去时刻**的事件，而 `:267` 的 `now = event.time` 会让仿真时钟倒流。虽然推演表明在 `:429` 调用点闸门恒 `≥ now`（max 里含当前刚完工的前驱），但该不变量是隐式且脆弱的，此处显式加固。`max(gate, now)` 是第二道保险。

同时新增闸门薄封装（放在 `_is_op_ready` 附近）：

```python
    def _flow_ready_time(self, shop: ShopFloor, op: Operation) -> float:
        """闸门取值，复用 _release_time_cache 避免重算放行时刻。

        实现单点在 ShopFloor.get_operation_flow_ready_time——此处只做缓存加速。
        """
        release_time = self._release_time_cache.get(op.id)
        return shop.get_operation_flow_ready_time(op, release_time=release_time)
```

`_release_time_cache.get(op.id)` 未命中时返回 `None`，恰好触发 `ShopFloor` 侧内部计算，语义与原 `self._release_time_cache.get(op.id, shop.get_operation_release_time(op))` 完全一致。

- [ ] **Step 6: 更新调用点传 `now`**

`core/simulator.py:257`（初始化路径）保持不变——该处 `now` 恒为 0，默认形参已覆盖。

`:429`（完工触发路径）必须传入当前事件时刻：

```python
                    for next_op_id in impacted_ops:
                        next_op = shop.operations.get(next_op_id)
                        if next_op and next_op.status == OpStatus.PENDING and self._is_op_ready(next_op):
                            self._queue_release_or_ready(shop, next_op, ready_ops, event_queue, now=now)
                            if next_op.status == OpStatus.READY:
                                newly_ready_types.update(
                                    self._op_dispatch_type_ids.get(next_op.id, {next_op.process_type})
                                )
```

- [ ] **Step 7: 替换其余四个闸门取值点**

`:275`（`release_check` handler 的到点判定）：

```python
                if op and op.status == OpStatus.PENDING and self._is_op_ready(op):
                    if self._flow_ready_time(shop, op) <= now:
```

`:479`（热启动复核）：

```python
        for op in shop.operations.values():
            if op.status == OpStatus.READY:
                if self._is_op_ready(op) and self._flow_ready_time(shop, op) <= 0:
                    self._mark_ready(op, ready_ops)
                else:
                    op.status = OpStatus.PENDING
```

`:719`（派工 probe）：

```python
        probe = max(not_before, self._flow_ready_time(shop, op), machine.current_finish_time)
```

`:748`（`wait_time` 特征）：

```python
        release_time = self._flow_ready_time(shop, op)
```

> **注意 `:719` 与 `:748` 的 `shop` 可见性**：先跑 `sed -n '700,760p' core/simulator.py` 确认这两处所在方法的签名是否已有 `shop` 形参。若无，需沿调用链传入。

- [ ] **Step 8: 运行测试，确认通过**

Run: `python -m pytest tests/test_turnover_time.py::TestSimulatorTurnover -v`
Expected: PASS（3 passed）

- [ ] **Step 9: 确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 既有测试**零转红**。仿真器是全系统最核心的部件，此处任何转红都必须停下排查，不得跳过。

- [ ] **Step 10: 写自然时间口径测试（口径 1）**

追加到 `TestSimulatorTurnover`：

```python
    def test_turnover_elapses_on_wall_clock_not_shift_time(self):
        """口径 1：turnover 跨越非排班时段时不被拉长。

        机床每天只排 0-8h。OP1 需 6h，从 0 开工、8h 前完工。turnover=10h
        跨越了当天的非排班时段。若 turnover 错误地只在班次内计时，OP2 会被
        推迟一整天；按自然时间则闸门落在次日，OP2 应在闸门开启后的第一个
        排班窗口内开工。
        """
        from llm4drd.core.models import Shift
        short_shifts = [Shift(day=d, start_hour=0.0, hours=8.0) for d in range(30)]
        shop = _build_shop(
            [("OP1", "turning", 6.0, [], 10.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", short_shifts), ("m2", "milling", short_shifts)],
        )
        entries = _run(shop)
        gate = entries["OP1"]["end"] + 10.0
        self.assertGreaterEqual(entries["OP2"]["start"], gate - 1e-9)
        self.assertLess(entries["OP2"]["start"], gate + 24.0,
                        "OP2 应在闸门后的首个排班窗口开工，而非因 turnover 被班次化再推一天")
```

- [ ] **Step 11: 运行测试**

Run: `python -m pytest tests/test_turnover_time.py -v`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add core/simulator.py tests/test_simulator_robustness.py tests/test_turnover_time.py
git commit -m "feat: enforce turnover gate in the simulator

Generalizes the existing release_check event path from a static
release_time to max(release_time, predecessor end + turnover).

Also hardens _queue_release_or_ready against clock rewind: it now
compares the gate against now rather than 0, and clamps the pushed
event time to max(gate, now)."
```

---

### Task 5: `ShopFloor` 派生时间计入 turnover

**Files:**
- Modify: `core/models.py:623-645`（任务内关键路径前推 + `task_meta`）、`:668-682`（任务级前推）、`:690-703`（`derived_start_time` 反推）
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `Operation.turnover_time`（Task 1）
- Produces: `task_meta[task_id]["critical_path_with_turnover"]` —— 仅 `core/models.py` 内部使用，不对外暴露。

> **本 Task 是全计划风险最高的一处**（spec §5.2 标注「设计中验证最薄的一环」）。它改的是关键路径语义，会波及 `derived_due_date`、`critical_slack` 及所有依赖 slack 的派工规则（`core/rules.py:26` 的 `slack`、`:50` 的 `urgency`）。**测试先行，不得跳过 Step 1–2。**

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_turnover_time.py`：

```python
class TestDerivedTimesWithTurnover(unittest.TestCase):
    def test_earliest_start_of_successor_includes_predecessor_turnover(self):
        """任务内前推：后继的 earliest_start 必须含前驱 turnover。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        # OP1: est=0, pt=5, turnover=4 -> OP2.est = 0+5+4 = 9
        self.assertAlmostEqual(shop.operations["OP2"].earliest_start_time, 9.0, places=6)

    def test_zero_turnover_leaves_earliest_start_unchanged(self):
        """零回归锚点。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 0.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        self.assertAlmostEqual(shop.operations["OP2"].earliest_start_time, 5.0, places=6)

    def test_derived_start_time_backs_off_by_own_turnover(self):
        """反推：后继要在 t 开工，则本工序须在 t - turnover 前完工。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
            due_date=100.0,
        )
        op1, op2 = shop.operations["OP1"], shop.operations["OP2"]
        # OP2.derived_start = 100 - 2 = 98；OP1 须在 98 - 4 = 94 前完工
        # 故 OP1.derived_due_date = 94, OP1.derived_start = 94 - 5 = 89
        self.assertAlmostEqual(op2.derived_start_time, 98.0, places=6)
        self.assertAlmostEqual(op1.derived_due_date, 94.0, places=6)
        self.assertAlmostEqual(op1.derived_start_time, 89.0, places=6)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_turnover_time.py::TestDerivedTimesWithTurnover -v`
Expected: FAIL —— `test_earliest_start_of_successor_includes_predecessor_turnover` 得 5.0（期望 9.0）；`test_derived_start_time_backs_off_by_own_turnover` 中 `op1.derived_due_date` 得 98.0（期望 94.0）。`test_zero_turnover_leaves_earliest_start_unchanged` 应 PASS。

- [ ] **Step 3: 前推计入 turnover**

`core/models.py` 的任务内关键路径循环，`offset` 加上前驱 turnover，并新增 `critical_path_with_turnover`：

```python
            topo = _topological_order(predecessors, op_ids)
            earliest_offsets: dict[str, float] = {}
            critical_path = 0.0
            critical_path_with_turnover = 0.0
            for op_id in topo:
                op = self.operations[op_id]
                # 环内工序(工序级前驱互锁)的前驱可能尚未计算(拓扑序对环不完整)，
                # 跳过避免 KeyError；环内工序的 earliest_offset 无实际意义，
                # 仿真器会单独做 SCC 环检测并标记 feasible=False。
                offset = max(
                    (earliest_offsets[predecessor_id]
                     + self.operations[predecessor_id].processing_time
                     + self.operations[predecessor_id].turnover_time
                     for predecessor_id in predecessors[op_id] if predecessor_id in earliest_offsets),
                    default=0.0,
                )
                earliest_offsets[op_id] = offset
                critical_path = max(critical_path, offset + op.processing_time)
                critical_path_with_turnover = max(
                    critical_path_with_turnover, offset + op.processing_time + op.turnover_time
                )
            task_meta[task_id] = {
                "op_ids": op_ids,
                "predecessors": predecessors,
                "successors": successors,
                "topo": topo,
                "earliest_offsets": earliest_offsets,
                "critical_path": critical_path,
                "critical_path_with_turnover": critical_path_with_turnover,
            }
```

**两个口径的分工**（本 Task 最容易搞错之处）：
- `critical_path` **不含**末工序 turnover —— 它是任务自身的工期，用于 `task.critical_path_time` 与 `derived_start_time` 反推。末工序的 turnover 不延长任务自己的完工。
- `critical_path_with_turnover` **含**末工序 turnover —— 用于跨任务前推（口径 2：后继任务要等前驱任务末工序流转完）。

- [ ] **Step 4: 任务级前推使用含 turnover 的口径**

`core/models.py` 的任务级前推循环。`task.earliest_finish_time` 保持不含 turnover（它是任务自身完工时刻，语义不能被污染），另算一个供后继取用的流转完成时刻：

```python
        task_flow_finish: dict[str, float] = {}
        for task_id in task_topo:
            task = self.tasks[task_id]
            order = self.orders.get(task.order_id)
            base_release = max(task.release_time, order.release_time if order else 0.0)
            predecessor_finish = max(
                (task_flow_finish[predecessor_id]
                 for predecessor_id in task_predecessors.get(task_id, set())
                 if predecessor_id in task_flow_finish),
                default=base_release,
            )
            task.earliest_start_time = max(base_release, predecessor_finish)
            task.earliest_finish_time = task.earliest_start_time + task.critical_path_time
            task_flow_finish[task_id] = task.earliest_start_time + task_meta.get(task_id, {}).get(
                "critical_path_with_turnover", task.critical_path_time
            )
            task.critical_slack = (
                task.derived_due_date - task.earliest_finish_time
                if math.isfinite(task.derived_due_date)
                else float("inf")
            )
```

**与原实现的差异**：原 `predecessor_finish` 取 `self.tasks[predecessor_id].earliest_finish_time`。改后取 `task_flow_finish[predecessor_id]`（含末工序 turnover）。`task_flow_finish` 在 `task_topo` 序下先于任何后继被写入，故 `if predecessor_id in task_flow_finish` 的守卫只在依赖环时才漏，与原实现对环的容忍度一致。

**前置条件**：`task.critical_path_time` 在反推循环（`:660` 一带）已赋值，该循环位于本前推循环之前，故此处可直接读。开工时确认此顺序未变。

- [ ] **Step 5: 反推扣掉自身 turnover**

`core/models.py` 的工序级反推循环：

```python
            for op_id in reversed(topo):
                op = self.operations[op_id]
                successor_starts = [
                    self.operations[successor_id].derived_start_time - op.turnover_time
                    for successor_id in successors.get(op_id, set())
                    if math.isfinite(self.operations[successor_id].derived_start_time)
                ]
                candidates = [value for value in [task.derived_due_date, *successor_starts] if math.isfinite(value)]
                op.derived_due_date = min(candidates) if candidates else float("inf")
                op.derived_start_time = (
                    op.derived_due_date - op.processing_time
                    if math.isfinite(op.derived_due_date)
                    else float("inf")
                )
```

turnover 是**前驱**的属性，故此处减的是 `op.turnover_time`（op 即前驱），不是后继的。

- [ ] **Step 6: 运行测试，确认通过**

Run: `python -m pytest tests/test_turnover_time.py::TestDerivedTimesWithTurnover -v`
Expected: PASS（3 passed）

- [ ] **Step 7: 确认零回归（本 Task 的重中之重）**

Run: `python -m pytest tests/ -v 2>&1 | tail -20`
Expected: 既有测试零转红。

本 Task 改了 `critical_slack` 的上游，而 `core/rules.py:26` 的 `slack`、`:50` 的 `urgency` 都依赖它。若 `tests/test_graph.py` 或规则相关测试转红，**不要改测试来迁就实现**——先判断是实现错了、还是测试断言本就编码了旧语义，并向用户上报。

- [ ] **Step 8: Commit**

```bash
git add core/models.py tests/test_turnover_time.py
git commit -m "feat: account for turnover in derived times

Forward pass adds predecessor turnover to earliest offsets. Task-level
forward pass uses a new critical_path_with_turnover so a successor task
waits for the predecessor task's last operation to finish flowing.
Backward pass backs derived_due_date off by the operation's own turnover.

task.earliest_finish_time keeps its meaning (the task's own completion)
and stays free of trailing turnover."
```

---

### Task 6: 近似评价接入 turnover

**Files:**
- Modify: `optimization/approx_eval.py:237-243`、`:323-329`
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `Operation.turnover_time`（Task 1）

> **开工前必读**：spec §5.4 把此处描述为「取值时加上 turnover」，**这低估了改动**。任务级前驱要满足口径 2（取 max(各工序 end + 各自 turnover)），而现有代码用的是 `task_completion`（任务整体完工时刻），该值**不足以还原口径 2**——必须改成遍历任务工序。
>
> 先勘查：
> ```bash
> sed -n '60,140p' optimization/approx_eval.py    # 预计算风格与 self.shop 可见性
> sed -n '230,260p' optimization/approx_eval.py   # 第一处 base_ready 及 task_ready 的合并语义
> sed -n '315,340p' optimization/approx_eval.py   # 第二处 base_ready
> ```
> 重点确认三件事：(a) 这两处所在方法能否访问 `self.shop`；(b) 第一处 `task_ready` 的 `float("inf")` 语义（表示前驱任务未完工？）如何与 `base_ready` 合并；(c) 该文件是否偏好预计算映射（见 `:78` 的 release_time 预计算）——若是，`{op_id: turnover_time}` 预计算比逐次查 `self.shop` 更贴合既有风格。

- [ ] **Step 1: 写测试**

追加到 `tests/test_turnover_time.py`：

```python
class TestApproxEvalTurnover(unittest.TestCase):
    def test_simulator_result_respects_turnover_as_reference(self):
        """参照锚点：仿真器的真实结果，供近似评价对齐。"""
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        entries = _run(shop)
        self.assertGreaterEqual(entries["OP2"]["start"], entries["OP1"]["end"] + 4.0 - 1e-9)
```

> **本条不足以证明 approx_eval 已接入**——它只断言仿真器。实施者**必须**据 Step 0 的勘查结果补一条直接调用 `ApproxEvaluator` 的断言，验证其内部 `base_ready` 含 turnover。这是本 Task 的主测试，不得省略。

- [ ] **Step 2: 改第一处工序级 `base_ready`**

`optimization/approx_eval.py` 第一处（`:239` 一带）：

```python
        if op.predecessor_ops:
            base_ready = max(base_ready, max(
                predecessor_completion.get(pred_id, 0.0) + self.shop.operations[pred_id].turnover_time
                for pred_id in op.predecessor_ops
                if pred_id in self.shop.operations
            ))
```

**必须补 `if pred_id in self.shop.operations` 守卫**：原式 `predecessor_completion.get(pred_id, 0.0)` 对不存在的前驱返回 0.0；加了 `self.shop.operations[pred_id]` 后，悬空前驱会抛 `KeyError`——而 `api/server.py:2260` 显式处理悬空前驱，证明这不是不可能情形。

- [ ] **Step 3: 改第一处任务级前驱**

```python
        if op.predecessor_tasks:
            task_ready = 0.0
            for task_id in op.predecessor_tasks:
                task = self.shop.tasks.get(task_id)
                if task is None:
                    task_ready = max(task_ready, task_completion.get(task_id, float("inf")))
                    continue
                for task_op in task.operations:
                    task_ready = max(
                        task_ready,
                        predecessor_completion.get(task_op.id, 0.0) + task_op.turnover_time,
                    )
            base_ready = max(base_ready, task_ready)
```

保留了 `task` 不存在时回落 `task_completion` 的原语义。**据 Step 0 勘查的 (b) 项据实调整**——若原 `task_ready` 与 `base_ready` 的合并方式不是 `max`，按实际改。

- [ ] **Step 4: 改第二处 `base_ready`**

第二处（`:325` 一带）结构不同（工序级与任务级都取 `.get(..., 0.0)`）：

```python
            if op.predecessor_ops:
                base_ready = max(base_ready, max(
                    predecessor_completion.get(pred_id, 0.0) + self.shop.operations[pred_id].turnover_time
                    for pred_id in op.predecessor_ops
                    if pred_id in self.shop.operations
                ))
            if op.predecessor_tasks:
                for task_id in op.predecessor_tasks:
                    task = self.shop.tasks.get(task_id)
                    if task is None:
                        base_ready = max(base_ready, task_completion.get(task_id, 0.0))
                        continue
                    for task_op in task.operations:
                        base_ready = max(
                            base_ready,
                            predecessor_completion.get(task_op.id, 0.0) + task_op.turnover_time,
                        )
```

- [ ] **Step 5: 运行测试并确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 零转红。

- [ ] **Step 6: Commit**

```bash
git add optimization/approx_eval.py tests/test_turnover_time.py
git commit -m "feat: account for turnover in approximate evaluation

Task-level predecessors now iterate their operations rather than using
the task completion time, so the max-over-(end + turnover) semantics
match the simulator."
```

---

### Task 7: CP-SAT 精确求解接入 turnover

**Files:**
- Modify: `optimization/exact.py:281-294`
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `Operation.turnover_time`（Task 1）

- [ ] **Step 1: 写失败测试（双引擎一致）**

> **`ExactSolver` 签名待核**：先跑 `grep -n "class ExactSolver" -A 15 optimization/exact.py` 与 `grep -rn "ExactSolver(" tests/ optimization/ api/` 确认构造与 `solve()` 的实际签名、返回结构及键名。`tests/test_simulator_robustness.py` 顶部注释提到「ExactSolver 对无交期任务不崩溃」，说明该文件已有调用范例可比照。

追加到 `tests/test_turnover_time.py`：

```python
class TestExactSolverTurnover(unittest.TestCase):
    def test_exact_respects_turnover(self):
        from llm4drd.optimization.exact import ExactSolver
        shop = _build_shop(
            [("OP1", "turning", 5.0, [], 4.0),
             ("OP2", "milling", 2.0, ["OP1"])],
            [("m1", "turning", _full_calendar()), ("m2", "milling", _full_calendar())],
        )
        result = ExactSolver(shop).solve()
        entries = {e["op_id"]: e for e in result.schedule}
        self.assertGreaterEqual(
            entries["OP2"]["start"], entries["OP1"]["end"] + 4.0 - 1e-6,
            "CP-SAT 必须与仿真器同口径地满足 turnover 约束",
        )
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `python -m pytest tests/test_turnover_time.py::TestExactSolverTurnover -v`
Expected: FAIL —— OP2 紧接 OP1 开工，未满足 +4。

- [ ] **Step 3: 改前驱约束**

`optimization/exact.py` 的前驱约束循环，整体替换：

```python
        for op in decision_ops:
            start_var = op_vars[op.id]["start"]
            for predecessor_id in op.predecessor_ops:
                predecessor_end = op_end_exprs.get(predecessor_id)
                predecessor = self.shop.operations.get(predecessor_id)
                if predecessor_end is not None and predecessor is not None:
                    model.Add(start_var >= predecessor_end + int(round(predecessor.turnover_time * scale)))
            for predecessor_task_id in op.predecessor_tasks:
                predecessor_task = self.shop.tasks.get(predecessor_task_id)
                if not predecessor_task:
                    continue
                for predecessor_op in predecessor_task.operations:
                    predecessor_end = op_end_exprs.get(predecessor_op.id)
                    if predecessor_end is not None:
                        model.Add(start_var >= predecessor_end + int(round(predecessor_op.turnover_time * scale)))
```

**取整口径**：用 `int(round(x * scale))`，与 `:165` 既有的 `model.Add(start >= max(0, int(round(self.shop.get_operation_release_time(op) * scale))))` 完全一致。**不得**用 `int()` 截断或 `math.ceil`——取整方向与仿真器不一致会让两引擎在边界算例上给出不同结论，Step 1 的测试正是为此设的。

**工序级循环加了 `predecessor is not None` 守卫**：`op_end_exprs.get(predecessor_id)` 非 None 不蕴含 `shop.operations` 里有该 op（前者可能来自 completed/fixed 工序的 `NewConstant`）。任务级循环无此问题——`predecessor_op` 直接来自 `predecessor_task.operations`。

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m pytest tests/test_turnover_time.py::TestExactSolverTurnover -v`
Expected: PASS

- [ ] **Step 5: 确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 零转红。

- [ ] **Step 6: Commit**

```bash
git add optimization/exact.py tests/test_turnover_time.py
git commit -m "feat: add turnover to CP-SAT precedence constraints

Uses int(round(t * scale)) to match the existing release_time rounding,
keeping the exact solver and the simulator in agreement on boundaries."
```

---

### Task 8: 滚动排产接入闸门

**Files:**
- Modify: `scheduling/online.py:116`、`:141`、`:186`、`:319`、`:472-473`（验证）
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `ShopFloor.get_operation_flow_ready_time(op)`（Task 3）

> **开工前先定位**：`grep -n "get_operation_release_time" scheduling/online.py`

- [ ] **Step 1: 替换四处闸门取值**

`scheduling/online.py:116`（probe）：

```python
        probe = max(not_before, self.sim_shop.get_operation_flow_ready_time(op), machine.current_finish_time)
```

`:141`（就绪筛选）：

```python
            if self.sim_shop.check_op_ready(op) and self.sim_shop.get_operation_flow_ready_time(op) <= now:
```

`:186`（`wait_time`，先跑 `sed -n '180,190p' scheduling/online.py` 确认上下文）：

```python
                    "wait_time": max(0.0, now - self.sim_shop.get_operation_flow_ready_time(op)),
```

`:319`：

```python
                    release_time = self.sim_shop.get_operation_flow_ready_time(op)
```

- [ ] **Step 2: 验证跨窗口前驱裁剪（未决问题，不得跳过）**

`scheduling/online.py:472-473` 在裁剪 remaining shop 时过滤掉不在窗口内的前驱引用：

```python
            op.predecessor_ops = [pred for pred in op.predecessor_ops if pred in remaining_shop.operations]
            op.predecessor_tasks = [pred for pred in op.predecessor_tasks if pred in remaining_shop.tasks]
```

**风险**：被过滤掉的前驱，其 turnover 约束会随之消失。若该前驱已在上一窗口完工、turnover 尚未走完，则下一窗口的后继可能被允许提前开工——turnover 跨窗口丢失。

先读 `sed -n '440,480p' scheduling/online.py`，判断被裁剪前驱的完工时刻如何传递到下一窗口（可能通过 `initial_state` 或 `not_before`）：

- 若已通过 `end_time` 传递且闸门能读到 → 无需改动，在此加一行注释说明并补一条跨窗口测试确认；
- 若确实丢失 → 需在裁剪前把 `max(pred.end_time + pred.turnover_time)` 固化到该工序的窗口内约束（如 `not_before`），并补跨窗口测试。

**此处不得凭猜测跳过。** 若判断困难，向用户上报再决定。

- [ ] **Step 3: 确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 零转红。

- [ ] **Step 4: Commit**

```bash
git add scheduling/online.py tests/test_turnover_time.py
git commit -m "feat: enforce turnover gate in online scheduling"
```

---

### Task 9: API 校验、payload 与返回体

**Files:**
- Modify: `api/server.py:1348`（payload 解析）、`:1409-1417`（批量导入 payload）、`:1496-1510`（返回体）、`:1630-1631`（校验）
- Modify: `tests/test_turnover_time.py`

**Interfaces:**
- Consumes: `inst_operations.turnover_time`（Task 2）

> **开工前先定位**：
> ```bash
> grep -n "加工时长非法" -B 5 -A 5 api/server.py    # 校验函数的签名与 err() 用法
> sed -n '1340,1355p' api/server.py                 # 单工序 payload 解析
> sed -n '1400,1420p' api/server.py                 # 批量导入 payload
> sed -n '1490,1515p' api/server.py                 # 返回体结构
> ```

- [ ] **Step 1: 写校验测试**

追加到 `tests/test_turnover_time.py`：

```python
class TestTurnoverValidation(unittest.TestCase):
    """turnover 允许 0、拒绝负值——与 processing_time 必须 > 0 的规则不同。"""

    def test_zero_turnover_is_valid(self):
        op = Operation(id="OP1", task_id="T1", name="OP1", process_type="turning",
                       processing_time=5.0, turnover_time=0.0)
        self.assertEqual(op.turnover_time, 0.0)
```

> **实施者必须补一条负值被拒的真断言**：按 Step 0 勘查到的校验函数签名（`api/server.py:1630` 一带 `err("数据完整性", op_id, ...)` 所在函数），构造 `turnover_time=-1` 的工序，断言校验产出对应错误项且 `sheet="operations"`。这是本 Task 的主测试，不得省略或以 `skipTest` 代替。

- [ ] **Step 2: 加校验**

`api/server.py` 的 `processing_time` 校验之后并列增加：

```python
        if op.processing_time is None or float(op.processing_time) <= 0:
            err("数据完整性", op_id, f"加工时长非法（{op.processing_time}），必须大于 0", sheet="operations")
        if op.turnover_time is not None and float(op.turnover_time) < 0:
            err("数据完整性", op_id, f"流转等待时长非法（{op.turnover_time}），不能为负", sheet="operations")
```

注意两条规则的差异：`processing_time` 要求 `> 0`，`turnover_time` 要求 `>= 0`。

- [ ] **Step 3: 接通 payload 解析**

`api/server.py:1348` 一带（比照该处 `initial_remaining_processing_time` 的写法），把 `turnover_time` 落进传给 `update_operation` 的 `data` dict：

```python
        "turnover_time": float(data["turnover_time"]) if str(data.get("turnover_time", "")).strip() else 0.0,
```

`:1409-1417` 的批量导入 payload 同样处理（比照该处 `remaining_raw` 的写法）。

- [ ] **Step 4: 接通返回体**

`api/server.py:1496` 一带的工序详情返回体，在 `"time": op.processing_time` 旁并列：

```python
                        "time": op.processing_time,
                        "turnover_time": op.turnover_time,
```

> **字段命名待核**：该返回体用 `"time"` 而非 `"processing_time"`，说明此处有自己的命名约定。据 Step 0 勘查的消费方（`frontend/app_v2.js`）据实命名。

- [ ] **Step 5: 运行测试并确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 零转红。

- [ ] **Step 6: Commit**

```bash
git add api/server.py tests/test_turnover_time.py
git commit -m "feat: expose and validate turnover_time in the API

turnover_time accepts 0 but rejects negatives, unlike processing_time
which must be strictly positive."
```

---

### Task 10: 图谱节点属性

**Files:**
- Modify: graph 改造后的 `CanonicalGraphBuilder`（**不是** `knowledge/graph.py`——见下）

**Interfaces:**
- Consumes: `Operation.turnover_time`（Task 1）

> **本 Task 的落点必然已变**。本计划排在 graph 改造之后，届时 `knowledge/graph.py` 已被降级为兼容适配器（graph plan Task 2 Step 5），关系与属性的定义已迁入 `CanonicalGraphBuilder`（graph design §4.2 规定「NetworkX 适配器只能消费 `CanonicalGraph`，不能再次解释 `ShopFloor`」）。

- [ ] **Step 1: 定位 OP 节点属性定义处**

```bash
ls knowledge/
grep -rn "processing_time" knowledge/ | head
grep -rln "CanonicalGraphBuilder\|CanonicalNode" .
```

今天该属性在 `knowledge/graph.py:162`（`processing_time=op.processing_time`）。找到改造后的实际定义处。

**若 `CanonicalGraphBuilder` 尚不存在**（graph 改造未如期落地），则退回改 `knowledge/graph.py:162`，并在提交信息中注明该属性未来需随 graph 改造迁移。

- [ ] **Step 2: 加节点属性**

在 `processing_time=op.processing_time,` 之后并列：

```python
                turnover_time=op.turnover_time,
```

- [ ] **Step 3: 写测试**

比照既有构图测试的形态（`grep -rn "processing_time" tests/test_graph.py`）写一条同形态断言：OP 节点属性含 `turnover_time` 且值正确。

- [ ] **Step 4: 运行测试并确认零回归**

Run: `python -m pytest tests/ -v 2>&1 | tail -5`
Expected: 零转红。

- [ ] **Step 5: Commit**

```bash
git add knowledge/ tests/
git commit -m "feat: expose turnover_time on operation graph nodes"
```

---

### Task 11: 偿付 graph 改造的顺序代价

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-unified-graph-context-design.md`（§7.3）
- Modify: graph 改造落地的 `builder_version` 常量
- Regenerate: graph 改造的 golden fixtures 与 benchmark 基线

**Interfaces:**
- Consumes: Task 1–10 的全部改动

> **本 Task 的存在理由**：用户已确认「graph 改造先行，turnover 在其之后」。该顺序的代价是 turnover 修改 `operations` schema 会作废 graph plan Task 1 冻结的基线。这些不是意外，是已知代价，必须在此显式偿付。

- [ ] **Step 1: 修订 fingerprint 输入清单**

`docs/superpowers/specs/2026-07-16-unified-graph-context-design.md` 的 §7.3「feature_hash 输入」，在「加工时间」旁并列补入：

```markdown
- 加工时间；
- 流转等待时间；
```

**§7.2「topology_hash 输入」不改** —— turnover 不改变任何边的存在性，只改变边的时间权重。

- [ ] **Step 2: bump `builder_version`**

按 graph design §4.1，`builder_version` 在「业务边解释、特征公式、规范化或排序规则变化时变化」。turnover 进入 `CanonicalGraph` 节点属性即构成此类变化。

```bash
grep -rn "builder_version\s*=\|BUILDER_VERSION" --include="*.py" .
```

按该常量的既有格式递增。按 graph design §7.4 的保守失效规则，这会使所有既有缓存失效并重建——**这是预期行为，不是缺陷**。

- [ ] **Step 3: 重新生成 golden fixtures**

graph plan Task 1 Step 1 建立的确定性 fixture 因 `operations` schema 变化而失效。

```bash
grep -rn "golden\|fixture" docs/superpowers/plans/2026-07-16-unified-graph-context.md | head -20
```

按该计划记载的生成方式重新生成。新 fixture 的 `operations` 应含 `turnover_time`，且**至少一个 golden 实例带非零 turnover**——否则 graph 改造的回归测试永远覆盖不到该字段。

- [ ] **Step 4: 重跑 benchmark 基线**

```bash
ls docs/benchmarks/
```

按 graph plan Task 8 记载的 benchmark 工具重跑 `docs/benchmarks/graph-context-baseline.json`。

- [ ] **Step 5: 更新 characterization 测试**

graph plan Task 1 Step 2 的 characterization 测试中，凡断言 `operations` 列结构或 `instance_hash` 具体值的，按新 schema 更新。

**不得**为了让测试变绿而放宽断言——若某条断言因 turnover 而失效，正确做法是更新其期望值，而非删除或弱化它。

- [ ] **Step 6: 全量验证**

Run: `python -m pytest tests/ -v 2>&1 | tail -10`
Expected: 全绿。

- [ ] **Step 7: Commit**

```bash
git add docs/ tests/
git commit -m "chore: reconcile turnover_time with the graph context baseline

Bumps builder_version, adds turnover to the feature_hash input list,
and regenerates the golden fixtures and benchmark baseline that the
operations schema change invalidated."
```

---

## Self-Review

**1. Spec coverage**

| Spec 章节 | 覆盖 Task |
|---|---|
| §3 口径 1（自然时间） | Task 4 Step 10 |
| §3 口径 2（任务级前驱） | Task 3 Step 1（末测试）、Task 5 Step 4、Task 6 Step 3 |
| §3 口径 3（机器 end_time 释放） | Task 4 Step 3（`test_machine_is_free_during_turnover`）；spec 已确认无需改代码 |
| §3 口径 4（缺省 0） | Task 1 Step 2、Task 2 Step 1 |
| §4.1 模板 | Task 1 Step 7–10 |
| §4.2 模型 | Task 1 Step 4 |
| §4.3 持久化（5 处） | Task 2 Step 3–9 |
| §4.4 接口与校验 | Task 9 |
| §5.1 仿真器闸门 + 时钟倒流加固 | Task 4 |
| §5.2 派生时间 | Task 5 |
| §5.2 `get_ready_ops` 不改（死代码） | 无需 Task —— spec 已论证零调用点 |
| §5.3 滚动排产 | Task 8 |
| §5.4 近似评价 | Task 6 |
| §5.5 精确求解 | Task 7 |
| §5.6 图谱 | Task 10 |
| §7 graph 改造代价 | Task 11 |
| §8 测试 1–10 | 用例 1 → Task 1 Step 1/6 与各 Task 的零回归步；2 → Task 4 Step 3；3 → Task 3 Step 1；4 → Task 4 Step 3；5 → Task 7 Step 1；6 → Task 4 Step 10；7 → **见下方缺口**；8 → Task 2 Step 1/10；9 → Task 9 Step 1；10 → Task 6 Step 1 |
| §9 完成定义 | 全部 Task |

**已修补的缺口**：spec §8 用例 7（initial_state 已完工工序，`end_time` 为负或 0 时 turnover 仍生效、闸门 ≤ 0 时后继立即就绪）在初稿中无对应步骤。它已由 Task 3 Step 1 的 `test_gate_tolerates_predecessor_without_end_time` 部分覆盖（`end_time is None` 分支），但**负 `end_time` 的分支仍未覆盖**。实施 Task 4 时须在 `TestSimulatorTurnover` 补一条：构造 `initial_state` 中 `end_time=-2`、`turnover=1` 的已完工前驱，断言后继在 t=0 即就绪（闸门 = -1 ≤ 0）。

**2. Placeholder scan**

本计划有 **9 处标注「待核」/「先勘查」的定位点**（Task 2/4/6/7/8/9/10）。它们**不是可省略的占位符**，而是诚实标注的「行号与签名会因 graph 改造而漂移、必须开工时用 grep 实测」的位置——每处都给了确切的勘查命令与判断依据。这是 Global Constraints 里「不得盲信行号」的直接体现。

**3 处标注「实施者必须补测试」**（Task 2 Step 10 的 round-trip、Task 6 Step 1 的 ApproxEvaluator 直接断言、Task 9 Step 1 的负值被拒断言）：这些是真实的计划弱点——我无法在不勘查这些模块调用签名的前提下写出能跑的断言，故明确标注为必补项而非佯装完备。初稿中 Task 9 曾用 `self.skipTest()` 占位，已删除。

**3. Type consistency**

- `Operation.turnover_time: float`（Task 1）在 Task 2–10 全程一致。
- `ShopFloor.get_operation_flow_ready_time(op, release_time=None) -> float`（Task 3）在 Task 4（经 `Simulator._flow_ready_time` 封装）与 Task 8（直接调用）签名一致。
- DB 列名 `turnover_time` 与 Excel 列名 `turnover_time_hrs` 始终区分（比照既有 `processing_time` / `processing_time_hrs` 双名约定），未混用。
- `task_meta["critical_path_with_turnover"]`（Task 5 Step 3 产出、Step 4 消费）键名一致。

**4. 已知薄弱环节**（供审阅者重点关注）

- **Task 5 风险最高**：改的是关键路径语义，波及 `critical_slack` 及依赖它的派工规则。其 `task_flow_finish` 方案在 spec 中仅为「展开点」，未经代码验证。
- **Task 6 被 spec 低估**：任务级前驱需把 `task_completion` 拆成逐工序遍历，改动形状大于 spec §5.4 描述的「取值时加上 turnover」。
- **Task 8 Step 2** 的跨窗口 turnover 丢失是真实未决问题，计划明确要求「不得凭猜测跳过，判断困难则上报」。
