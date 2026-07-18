# 定位提示词：仿真报「依赖环 / 任务级前驱互锁」，根因是否与转运时间有关？

> 你是资深排产系统工程师。请在 LLM4DRD 项目（`/Users/zhouwentao/Desktop/llm4drd`）中**定位**下面这个仿真报错的根因，并给出修复建议。**默认只定位、只解释、给方案，不要直接改代码**——除非最后明确让我动手。

---

## 1. 现象（仿真规则界面报出）

```
仿真完成，但结果不完整：仅排出 16230/16536 道工序，指标只反映部分排程；
220 道工序：依赖环：工序/任务前驱互相矛盾，环内工序永远无法就绪（需打断环）
（任务级前驱互锁(改任务令 predecessor_tasks),
  涉事任务:QWZPGT625A1;QWZPGT625A1-02,
  涉事任务:QWZPGT626A3;QWZPGT626A3-04,
  涉事任务:QXYBJT518A1;QXYBJT602A2,
  涉事任务:QXYBJT518A4;QXYBJT518A5,
  涉事任务:QXYBJT519A5;QXYBJT519A6;QXYBJT519A8,
  涉事任务:QXYBJT603A4;QXYBJT603A5,
  涉事任务:QXYBJT605A8;QXYBJT615A3,
  涉事任务:QXYBJT611A4;QXYBJT611A5,
  涉事任务:QXYBJT611A8;QXYBJT611A9;QXYBJT611AA,
  涉事任务:QXYBJT617A1;QXYBJT617A2,
  涉事任务:QXYBJT623A3;QXYBJT702A5)
示例工序：IO(op_QWZPGT625A1-1)，EDM(op_QWZPGT625A1-02-2)，
  IO(op_QWZPGT626A3-1)，EDM(op_QWZPGT626A3-04-1)，
  UC(op_QXYBJT518A1-1)，QC(op_QXYBJT518A1-2)，IO(op_QXYBJT518A1-3)，IO(op_QXYBJT518A4-1)
```

根因分类文案由 `api/server.py:2428`（`dependency_cycle` 文案）与 `api/server.py:2527-2551`（根因报告）生成；硬门禁在 `core/simulator.py:733-734`（`if self._dependency_cycles: ... feasible=False`）。

## 2. 用户假设（请重点验证 / 纠正）

用户原话：「以前正常**不加转运时间**运行没问题的话，加了转运时间运行也不该有这种问题，**除非把转运时间错误地加成了加工的时间上面**。」

**请你先做一个关键的技术纠偏，再动手查：**
- 依赖环是**拓扑（前驱边）属性**，由 Tarjan SCC 在开工前检测（`core/sim_runtime.py:detect_dependency_cycles`，约 75–130 行）。该函数**只读取 `op.predecessor_ops` 与 `op.predecessor_tasks`，全文不引用 `turnover_time` / 转运时间**。
- 因此：**把转运时间错误地并入 `processing_time` 只会改变时长/工期，绝不可能“造出”一个依赖环**。用户这个具体假设在“环”的语境下不成立——转运时间即便写错，也只是权重，动不了边。
- 真正要查的是：**环到底来自哪里**——是数据里本就有任务级前驱互锁，还是某条代码路径在“开启转运时间”时**错误地新增了前驱边/自环**。

## 3. 必做的决定性验证（A/B 对照，先把“转运时间是否背锅”钉死）

1. **纯结构性环检测（不跑仿真、不读转运时间）**：运行 `tools/analyze_unscheduled.py`（其内部 21/167/180 行附近做纯 SCC，独立于仿真与 turnover）。对**当前这个报错的同一实例**跑一次。
   - 若它**同样报出这些任务级环** → 证明环存在于原始 `predecessor_tasks` 数据，**与转运时间无关**，转运时间只是时间上的巧合。
2. **零转运对照**：把该实例**复制一份**，把所有 `operations.turnover_time` 强制置 0（或直接用 `turnover=0` 的老口径构造），再跑一次结构性环检测 / 仿真。
   - 若零转运**仍报同样的环** → 彻底排除转运时间，根因是数据或门禁版本。
   - 若零转运**不报环、开转运才报环** → 转运代码路径确实改了边，立即去查下面第 4 点的“边来源”。
3. **直接查数据**：在实例库 / xlsx 里查上述涉事任务的 `predecessor_tasks` 字段，确认是否互相把对方列为前驱（尤其注意 `QWZPGT625A1` 与其子任务 `QWZPGT625A1-02` 这类“主任务 ↔ 子任务”互相引用）。把查到的原始前驱关系原文贴出来。

## 4. 需要逐文件核查的代码路径（grep 定位，关注“是否在开启转运时新增/改写前驱边”）

- `core/sim_runtime.py:detect_dependency_cycles`（75–130）：确认它**只**用 `predecessor_ops`/`predecessor_tasks` 建边，**没有任何 `turnover_time` 参与**。若你发现任何 `if turnover > 0: adj[...].add(...)` 之类逻辑，那就是 bug 源头。
- `data/db.py` / `data/generator.py`：导入与构造 `predecessor_tasks` / `predecessor_ops` 的路径。重点看**转运时间相关迁移或导入步骤**是否顺带写了/推导了任务前驱（例如某处把“转运关系”误写成 `predecessor_tasks`）。
- `knowledge/graph.py` 或 `CanonicalGraphBuilder`：图谱构建是否把 turnover 错误地建成了一条**前驱边**（设计上 turnover 只应是节点属性，不是边；见 `docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md` §5.6）。若图构建新增了边，且该边又被环检测复用，则会造环。
- `core/models.py`：`build_indexes`、流转就绪 `get_operation_flow_ready_time` 等，确认 `turnover>0` 时**不向 `predecessor_ops`/`predecessor_tasks` 追加任何东西**（turnover 只应作为 `end_time + turnover_time` 的“时间闸门”，不增边）。
- **版本/时机核查**：`git log` 确认“依赖环硬门禁（`feasible=False`）+ `detect_dependency_cycles` 迁入 `sim_runtime.py`”来自哪次提交（已知属 `sim-optimizer-performance` 改造，commit `bfadd12`/`d391c40`），与转运时间功能（commit `2b86f22`）是否独立。若硬门禁是后来加的，则“以前能跑”可能只是旧版**未检测/容忍**了这个本就存在的环——这能解释“不加转运能跑、加了才报错”的错觉。

## 5. 交付要求（给我什么）

请输出一份**根因定位报告**，至少回答：
1. 该实例的环是**数据固有**（原始 `predecessor_tasks` 互锁）还是**代码在开启转运时新增了边**？用第 3 节的 A/B 验证结论说话。
2. 若属数据固有：列出互相矛盾的具体任务对及其原始 `predecessor_tasks` 取值（贴库/表查询原文），指明应在“实例与约束”页如何打断环（去掉哪条矛盾前驱）。
3. 若属代码新增边：精确定位到文件:行号与改动建议。
4. 给出结论：**转运时间是否真的是本次报错的原因**？如果不是，真正的触发因素是什么（数据？还是后加的硬门禁）？
5. 给出**最小修复建议**（定位阶段不强制改代码，但请说明改哪、怎么改最安全，且不破坏 `docs/superpowers/specs/2026-07-16-operation-turnover-time-design.md` 的“turnover 只是时间闸门、绝不加边/不改 processing_time”语义）。

## 6. 关键约束
- 不要把时间浪费在“找 turnover 被加到 processing_time 的位置”上——那不会造成环；除非 A/B 验证（第 3 节）证明开转运才出环，才去查边来源。
- 引用行号时以 grep 实际结果为准（代码可能已偏移）。
- 结论要基于实证（跑 `analyze_unscheduled.py`、查库），不要凭直觉。
