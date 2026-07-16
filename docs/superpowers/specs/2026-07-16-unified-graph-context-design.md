# 统一图谱与 GraphContext 渐进式改造设计

- 日期：2026-07-16
- 状态：已确认设计，待实施规划
- 适用范围：异构图构建、图谱持久化、混合 NSGA-III/ALNS 初始化与图特征访问

## 1. 背景

系统当前存在两条彼此独立的图谱路径：

1. “仿真与洞察”页面从 `ShopFloor` 构建 `HeterogeneousGraph`，将节点和边持久化到 SQLite，供图谱分页、搜索、订单子图和前端可视化使用。
2. `HybridNSGA3ALNSOptimizer` 启动时再次从同一个 `ShopFloor` 构建 NetworkX 图，并重新计算图特征和图谱画像，用于候选生成、近似评价和 ALNS 邻域选择。

两条路径来源相同，但没有共享构建语义、版本标识或持久化计算结果。因此，用户提前构建展示图谱不会缩短优化器初始化时间；两套关系解释逻辑也存在长期漂移风险。

本设计采用“逻辑统一、物理分层”的方式解决问题：建立唯一的标准图构建器，同时生成面向展示和面向计算的两种投影。展示投影继续服务现有图谱 API；计算投影编译为版本化、不可变的 `GraphContext`，由混合优化器直接使用。

## 2. 目标与验收原则

### 2.1 本期目标

- 建立唯一的 `CanonicalGraphBuilder`，统一解释订单、任务、工序和资源关系。
- 从同一个 `CanonicalGraph` 生成展示投影和计算投影。
- 使用现有 SQLite 持久化计算图，不引入外部缓存文件或新基础设施。
- 增加稳定 fingerprint、缓存校验、自动构建和事务失效机制。
- 让混合 NSGA-III/ALNS 从 `GraphContext` 读取图特征和邻接关系，不再自行构造 NetworkX 图。
- 支持进程内 L1 缓存和 SQLite L2 缓存。
- 缓存未命中或失效时自动构建，不要求用户提前操作。
- 固定实例、配置和随机种子下，缓存冷构建、SQLite 热加载和 L1 命中的优化结果完全一致。
- 中型和大型实例在缓存命中时，优化器初始化耗时相比 legacy 至少降低 50%。
- 固定评价次数下，端到端优化性能不劣化。

### 2.2 非目标

- 不修改 `Simulator` 的离散事件推进核心。
- 不让普通仿真依赖 `GraphContext`。
- 不引入 GNN、代理模型或训练流水线。
- 不更换 SQLite。
- 不改变优化目标、候选参数、NSGA-III、ALNS 或精确仿真的业务语义。
- 不在本期实现局部图更新；任一相关版本不匹配时保守重建完整双投影。
- 不在本期移除 NetworkX 依赖；它可作为兼容适配器和测试参照保留，但退出优化热路径。

## 3. 方案选择

采用方案 A：统一构建器，双投影输出。

```text
InstanceStore / ShopFloor
          ↓
GraphFingerprint
          ↓
CanonicalGraphBuilder
          ↓
CanonicalGraph
     ┌────┴────┐
     ↓         ↓
DisplayGraph   ComputeGraph
Projection     Projection
     ↓         ↓
graph_*        graph_context_*
     ↓         ↓
前端图谱       Hybrid NSGA-III/ALNS
```

不采用“从展示表反向编译计算图”，因为这会让计算层依赖展示 JSON 和查询结构；不采用“继续保留两套构建逻辑只共享版本号”，因为它无法消除关系语义漂移。

## 4. 核心领域模型

### 4.1 GraphFingerprint

```python
@dataclass(frozen=True)
class GraphFingerprint:
    instance_hash: str
    topology_hash: str
    feature_hash: str
    schema_version: int
    builder_version: str
```

职责：

- `instance_hash` 覆盖完整规范化实例，包括资源日历和停机数据。
- `topology_hash` 覆盖节点归属、前驱关系和资源兼容关系。
- `feature_hash` 覆盖静态图特征公式所读取的业务字段。
- `schema_version` 在计算图表结构、字段含义或序列化约定变化时递增。
- `builder_version` 在业务边解释、特征公式、规范化或排序规则变化时变化。

哈希输入必须经过稳定排序和规范化序列化。不得依赖 Python 对象地址、进程 hash、数据库未声明的返回顺序或字典插入顺序。哈希算法使用 SHA-256。

### 4.2 CanonicalGraphBuilder

```python
@dataclass(frozen=True)
class CanonicalNode:
    node_id: str
    node_type: str
    entity_id: str
    attrs: Mapping[str, ScalarValue]

@dataclass(frozen=True)
class CanonicalEdge:
    source: str
    target: str
    edge_type: str
    attrs: Mapping[str, ScalarValue]

@dataclass(frozen=True)
class CanonicalGraph:
    nodes: tuple[CanonicalNode, ...]
    edges: tuple[CanonicalEdge, ...]
    fingerprint: GraphFingerprint
```

`CanonicalGraphBuilder` 是唯一解释业务关系的模块，统一生成：

- 订单→任务；
- 任务前驱；
- 任务→工序；
- 工序前驱；
- 工序依赖前置任务；
- 工序→候选机器；
- 工序→候选工装；
- 工序→候选人员。

现有 `HeterogeneousGraph.build_from_shopfloor()` 中的关系定义迁入构建器。NetworkX 适配器只能消费 `CanonicalGraph`，不能再次解释 `ShopFloor`。

### 4.3 DisplayGraphProjection

展示投影负责把标准图转换为现有 `graph_nodes`、`graph_edges` 和 `graph_meta` 格式。

要求：

- 保持现有图谱 API 字段兼容。
- 保留标签、时间文本、节点类型和 JSON 展示属性。
- 保持订单子图、模糊搜索、邻居查询和 `OS_` 资源过滤行为。
- 在 `graph_meta` 中增加 fingerprint、构建器版本和构建耗时。
- 前端无需因本次计算图改造重写渲染逻辑。

### 4.4 ComputeGraphProjection 与 GraphContext

```python
@dataclass(frozen=True)
class GraphContext:
    fingerprint: GraphFingerprint
    operation_ids: tuple[str, ...]
    operation_index: Mapping[str, int]
    machine_ids: tuple[str, ...]
    predecessor_offsets: tuple[int, ...]
    predecessor_indices: tuple[int, ...]
    successor_offsets: tuple[int, ...]
    successor_indices: tuple[int, ...]
    eligible_machine_offsets: tuple[int, ...]
    eligible_machine_indices: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_matrix: tuple[tuple[float, ...], ...]
    operation_groups: Mapping[tuple[str, str], tuple[int, ...]]
```

第一期特征：

- `predecessor_depth`；
- `assembly_criticality`；
- `shared_resource_degree`；
- `bottleneck_adjacency`；
- `graph_out_degree`。

第一期分组索引：

- `process_type`；
- `tooling_type`；
- `personnel_skill`。

第一期直接邻接：

- 工序前驱和后继；
- 工序候选机器；
- 任务前驱；
- 任务→工序；
- 订单→任务。

优化循环只访问已加载的不可变 `GraphContext`，不访问 SQLite、不解析 JSON、不查询 NetworkX。

### 4.5 GraphContextService

统一入口：

```python
context, diagnostics = graph_context_service.get_or_build(shop)
```

职责：

1. 计算当前 fingerprint。
2. 查询进程内 L1 缓存。
3. L1 未命中时查询 SQLite L2 缓存。
4. 校验加载结果。
5. 未命中、失效或首次损坏时调用标准构建器。
6. 在单个事务中保存展示投影和计算投影。
7. 返回不可变 `GraphContext` 和缓存诊断信息。

优化器领域层不读取数据库。图谱 API 和优化 API 都通过该服务触发统一构建或加载。

## 5. SQLite 数据模型

项目当前不依赖 NumPy。本期使用关系表持久化，在加载时一次扫描组装紧凑元组，不使用外部 `.npz` 文件或自定义二进制 BLOB。

### 5.1 计算图元数据

```sql
CREATE TABLE graph_context_meta (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    instance_hash       TEXT NOT NULL,
    topology_hash       TEXT NOT NULL,
    feature_hash        TEXT NOT NULL,
    schema_version      INTEGER NOT NULL,
    builder_version     TEXT NOT NULL,
    status              TEXT NOT NULL,
    operation_count     INTEGER NOT NULL,
    relation_count      INTEGER NOT NULL,
    feature_count       INTEGER NOT NULL,
    build_time_ms       REAL NOT NULL,
    created_at          REAL NOT NULL
);
```

只有 `status='ready'` 且 fingerprint 完全匹配的缓存可被加载。

### 5.2 实体整数索引

```sql
CREATE TABLE graph_entity_index (
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    PRIMARY KEY (entity_type, entity_id),
    UNIQUE (entity_type, ordinal)
);
```

保存 `order`、`task`、`operation`、`machine`、`tooling` 和 `personnel`。ordinal 由实体类型和业务 ID 的稳定排序产生。

### 5.3 计算邻接关系

```sql
CREATE TABLE graph_context_relations (
    relation_type  TEXT NOT NULL,
    source_ordinal INTEGER NOT NULL,
    target_ordinal INTEGER NOT NULL,
    PRIMARY KEY (relation_type, source_ordinal, target_ordinal)
);

CREATE INDEX idx_graph_context_rel_src
ON graph_context_relations(relation_type, source_ordinal);

CREATE INDEX idx_graph_context_rel_tgt
ON graph_context_relations(relation_type, target_ordinal);
```

加载时按 `relation_type, source_ordinal, target_ordinal` 稳定排序，一次扫描生成 offsets/indices。

### 5.4 工序静态特征

```sql
CREATE TABLE graph_operation_features (
    op_ordinal             INTEGER PRIMARY KEY,
    predecessor_depth      REAL NOT NULL,
    assembly_criticality   REAL NOT NULL,
    shared_resource_degree REAL NOT NULL,
    bottleneck_adjacency   REAL NOT NULL,
    graph_out_degree       REAL NOT NULL
);
```

使用显式列保证类型明确、加载快速并支持逐特征一致性检查。字段变化由幂等迁移、`schema_version` 和 `builder_version` 共同管理。

### 5.5 工序分组索引

```sql
CREATE TABLE graph_operation_groups (
    group_type TEXT NOT NULL,
    group_key  TEXT NOT NULL,
    op_ordinal INTEGER NOT NULL,
    PRIMARY KEY (group_type, group_key, op_ordinal)
);

CREATE INDEX idx_graph_operation_groups_lookup
ON graph_operation_groups(group_type, group_key);
```

用于替换图谱画像构建中按工艺、工装和人员技能反复全量扫描工序的逻辑。

### 5.6 展示元数据扩展

现有 `graph_meta` 增加：

- `instance_hash`；
- `topology_hash`；
- `feature_hash`；
- `schema_version`；
- `builder_version`；
- `build_time_ms`。

旧数据缺少任一必需字段时按 stale 处理，不做复杂回填。

## 6. 构建、保存与并发

### 6.1 构建流程

```text
ShopFloor
   ↓
规范化并计算 fingerprint
   ↓
CanonicalGraphBuilder
   ↓
DisplayGraphProjection + ComputeGraphProjection
   ↓
内存完整性校验
   ↓
SQLite 单事务提交
```

### 6.2 事务规则

```text
BEGIN IMMEDIATE
  再次确认当前实例 fingerprint
  删除旧展示投影
  删除旧计算投影
  插入展示节点和边
  插入实体索引、计算关系、特征和分组
  最后写 graph_meta 和 graph_context_meta
COMMIT
```

- 任一失败整体回滚。
- 元数据必须最后写入。
- 失败后不能存在半成品 `ready` 缓存。
- WAL 读者在提交前继续看到旧快照。
- 构建期间实例变化时放弃持久化，但调用者可以继续使用与其 ShopFloor 快照匹配的内存上下文。

### 6.3 并发控制

- 使用进程内构建锁和 fingerprint 维度的单飞机制。
- 同一 fingerprint 的并发请求等待同一次构建。
- 不同 fingerprint 的旧请求不能覆盖新实例缓存。
- `GraphContext` 不可变，可安全共享给多个优化线程。
- 当前系统为单进程任务模型，本期不引入分布式锁。
- 多 Uvicorn worker 支持属于后续范围；届时使用 SQLite 租约锁或独立构建服务。

## 7. 缓存与失效

### 7.1 缓存层级

```text
L1：当前进程中的不可变 GraphContext
L2：SQLite graph_context_* 表
L3：CanonicalGraphBuilder 完整重建
```

- L1 默认只保留当前活动实例。
- L1 和 L2 均以完整 fingerprint 为 key。
- 服务重启后从 L2 恢复。
- 运行任务持有的旧 `GraphContext` 不因全局失效而被修改。

### 7.2 topology_hash 输入

- 订单、任务、工序归属；
- 任务前驱；
- 工序前驱和前置任务；
- 工艺类型；
- 机器 ID、机器类型和关键资源标记；
- 显式候选机器；
- 工装类型和实例；
- 人员技能和实例；
- 工序所需工装和技能。

### 7.3 feature_hash 输入

- 加工时间；
- 订单和任务交期；
- 发布时间；
- 优先级；
- 主任务标记；
- 初始工序状态；
- 图特征公式读取的所有资源属性。

### 7.4 保守失效规则

第一期任一条件成立就重建完整双投影：

- `instance_hash` 不匹配；
- `topology_hash` 不匹配；
- `feature_hash` 不匹配；
- `schema_version` 不匹配；
- `builder_version` 不匹配；
- 缓存完整性检查失败；
- 实例更新接口显式调用 `invalidate()`。

现有实例更新接口统一调用：

```python
graph_context_service.invalidate(reason="operation_updated")
```

失效只使缓存不可用，不同步阻塞重建。下一次打开图谱或启动优化时自动构建，图谱页面的手动构建继续作为预热入口。

## 8. 优化器接入

### 8.1 API 流程

```text
POST /api/optimize/hybrid
        ↓
捕获 ShopFloor 快照
        ↓
GraphContextService.get_or_build(shop)
        ↓
HybridNSGA3ALNSOptimizer(shop, config, graph_context)
        ↓
coarse → exact_promotion → elite_refine → finalize
```

新增任务阶段：

- `graph_context_loading`；
- `graph_context_building`。

任务状态增加：

```json
{
  "graph_context": {
    "cache_level": "l1|sqlite|built",
    "cache_hit": true,
    "instance_hash": "12位前缀",
    "topology_hash": "12位前缀",
    "feature_hash": "12位前缀",
    "schema_version": 1,
    "load_time_ms": 18.4,
    "build_time_ms": 0.0,
    "operation_count": 1200,
    "relation_count": 8450
  }
}
```

### 8.2 优化器接口

```python
optimizer = HybridNSGA3ALNSOptimizer(
    shop=current_shop,
    config=config,
    graph_context=context,
)
```

优化器移除自行创建 `HeterogeneousGraph` 和重复计算静态图特征的路径，改为使用：

```python
self.graph_context = graph_context
self.graph_features = graph_context.feature_view_by_operation_id()
```

### 8.3 只读访问接口

```python
context.operation_features(op_id)
context.predecessors(op_id)
context.successors(op_id)
context.eligible_machines(op_id)
context.operations_in_group("process_type", process_type)
context.operations_in_group("tooling_type", tooling_type)
context.operations_in_group("personnel_skill", skill_id)
```

### 8.4 图谱画像

保留现有画像名称和业务语义：

- `balanced`；
- `assembly_focus`；
- `bottleneck_focus`；
- `shared_resource_focus`；
- `due_focus`。

画像在优化任务初始化时从 `GraphContext` 构建，不在本期持久化。邻域扩展改用前驱/后继和工序分组索引，消除按每个种子工序扫描全部工序的热点。

### 8.5 结果一致性

计算图只替换静态构图、图特征计算和邻域检索，不改变：

- 候选参数；
- 随机数调用次数和顺序；
- 初始种群顺序；
- 图谱画像名称和目标权重；
- 特征公式和归一化；
- destroy/repair 权重；
- 候选过滤排序；
- 近似评价；
- 精确仿真；
- NSGA-III 和 ALNS 选择。

整数 ordinal 只用于查找，不能隐式替换现有算法的 tie-break 顺序。对同分排序必须保留 legacy 业务顺序。

一致性定义：

```text
相同实例 + 相同配置 + 相同随机种子 + 固定评价次数
    ↓
legacy、冷构建、SQLite 热加载、L1 命中
    ↓
候选序列、方案 ID、排程签名、目标向量、Pareto rank 和输出顺序一致
```

## 9. 完整性校验与异常处理

### 9.1 加载校验

- fingerprint 完全匹配；
- ordinal 连续且无重复；
- 每个工序都有特征行；
- 所有关系端点在合法范围；
- offsets 单调且终值等于 indices 数量；
- 所有特征值有限；
- ShopFloor 中每个工序都能被解析；
- 实际规模与元数据一致。

校验失败时：

1. 标记当前缓存损坏。
2. 清除损坏的展示和计算投影。
3. 自动完整重建一次。
4. 再次执行相同校验。
5. 第二次仍失败则终止优化，不静默切换到另一条求解语义。

### 9.2 异常类型

```python
class GraphContextError(Exception): ...
class GraphContextStaleError(GraphContextError): ...
class GraphContextCorruptError(GraphContextError): ...
class GraphContextBuildError(GraphContextError): ...
class GraphContextPersistenceError(GraphContextError): ...
```

用户信息需要区分：

- 构建期间实例变化；
- 缓存损坏并正在自动重建；
- 图谱规模超过安全限制；
- SQLite 持久化失败；
- 二次完整性校验失败。

日志包含任务 ID、fingerprint 前缀、阶段、图规模和异常堆栈，不记录完整业务数据。

## 10. 兼容与发布模式

增加：

```text
LLM4DRD_GRAPH_CONTEXT_MODE=legacy|shadow|active
```

- `legacy`：完全使用现有优化器路径。
- `shadow`：构建或加载 `GraphContext`，比较关系、特征和画像，但仍由 legacy 求解。
- `active`：优化器正式使用 `GraphContext`。

发布顺序为 `legacy → shadow → active`。shadow 不执行第二次完整优化，只比较静态结构和派生结果。

## 11. 测试设计

### 11.1 Fingerprint 测试

- 相同业务数据、不同字典顺序得到相同 hash。
- 加工时间变化导致 `instance_hash` 和 `feature_hash` 变化。
- 前驱关系或候选机器变化导致 `topology_hash` 变化。
- 班次或停机变化至少导致 `instance_hash` 变化。
- 构建器或 schema 版本变化使缓存失效。
- 非有限浮点、空 ID 和重复 ID 被拒绝。

### 11.2 标准图与展示投影测试

- 覆盖所有节点和边类型。
- 覆盖显式候选机器和按工艺推导机器。
- 覆盖工装、人员、无候选资源、悬空前驱和依赖环。
- 与 legacy `HeterogeneousGraph` 比较节点集合、边集合和关键属性。
- 验证图谱 API、订单子图、邻居查询、模糊搜索和 `OS_` 过滤兼容。

### 11.3 计算投影测试

- ordinal 连续、稳定且可逆。
- 前驱和后继互为逆关系。
- 资源索引和工序分组无重复。
- 每个工序特征完整。
- SQLite 加载结果与内存直接构建结果相等。

### 11.4 事务与服务测试

- 节点、边、特征或元数据写入故障均整体回滚。
- SQLite 锁冲突不会暴露半成品。
- 构建期间实例变化不会覆盖新缓存。
- L1、SQLite 和完整构建三条路径均可验证。
- 相同 fingerprint 的并发请求只构建一次。
- 损坏缓存只自动重建一次，二次失败明确报错。

### 11.5 优化一致性测试

准备小型、中型和大型固定实例：

| 规模 | 工序数 | 用途 |
|---|---:|---|
| 小型 | 30–80 | 快速回归和全量精确断言 |
| 中型 | 300–800 | 多目标和并行路径 |
| 大型 | 2,000–5,000 | 性能和内存测试 |

固定随机种子、目标、种群、代数、ALNS 轮数、worker 数和评价次数，比较：

- 初始及每代候选签名序列；
- 近似目标向量；
- 精确晋级候选；
- ALNS 操作器选择序列；
- 最终 `solution_id`；
- `schedule_signature`；
- 目标向量；
- Pareto rank 和输出顺序。

shadow 特征差异阈值为绝对误差 `1e-12`；最终序列化方案必须完全相同。

## 12. 性能基准

记录：

- `fingerprint_time`；
- `canonical_build_time`；
- `display_projection_time`；
- `compute_projection_time`；
- `sqlite_write_time`；
- `sqlite_load_time`；
- `graph_context_validation_time`；
- `optimizer_init_time`；
- `total_optimization_time`。

验收：

1. L1 或 SQLite 命中时不调用 `CanonicalGraphBuilder`。
2. 命中时不调用 legacy `_build_graph_features()`。
3. 中型和大型实例缓存命中初始化时间至少降低 50%。
4. 固定评价次数下端到端耗时不劣化。
5. 每个场景运行 7 次，前 2 次预热不计，使用其余结果中位数。
6. 端到端允许最多 3% 测量噪声，超过 3% 判定为劣化。
7. 峰值内存不超过 legacy 的 1.25 倍。

报告保存到：

- `docs/benchmarks/graph-context-baseline.json`；
- `docs/benchmarks/graph-context-result.json`；
- `docs/benchmarks/graph-context-report.md`。

## 13. 可观测性

结构化日志事件：

- `graph_context.l1_hit`；
- `graph_context.sqlite_hit`；
- `graph_context.miss`；
- `graph_context.build_started`；
- `graph_context.build_completed`；
- `graph_context.invalidated`；
- `graph_context.corrupt`；
- `graph_context.rebuild_failed`。

图谱 meta 和优化任务状态暴露缓存层级、fingerprint 前缀、构建器版本、规模、加载/构建/校验耗时和最近失效原因。本期不新增独立监控系统。

## 14. 数据迁移

- 使用现有 `init_db()` 幂等创建新表和索引。
- 现有展示图数据不做复杂回填。
- 缺少 fingerprint 的旧 `graph_meta` 视为 stale。
- 第一次图谱访问或优化启动时按需自动重建。
- 不修改实例业务表。
- 不在服务启动时同步构建大图。
- 回滚旧版本时新增表被忽略，不要求数据库降级。

## 15. 分阶段发布

### 阶段 0：基线

- 固化 golden 实例和 legacy 优化输出。
- 保存初始化、端到端和内存基线。
- 确立固定评价次数的可重复配置。

### 阶段 1：统一构建器和双投影

- 引入 fingerprint 和 `CanonicalGraph`。
- 建立 SQLite 计算图表。
- 图谱页面切换到展示投影。
- 优化器继续使用 legacy。

退出条件：图谱 API 兼容、标准图与 legacy 边集合一致、事务故障测试通过。

### 阶段 2：Shadow

- 每次优化获取或构建 `GraphContext`。
- 逐项比较关系、特征和画像。
- 求解仍使用 legacy。

退出条件：golden 实例无结构差异，特征误差不超过 `1e-12`，无并发覆盖或损坏缓存。

### 阶段 3：Active

- 混合优化器正式使用 `GraphContext`。
- 对比冷构建、SQLite 命中和 L1 命中结果。
- 运行完整性能基准。

退出条件：优化输出严格一致，缓存命中初始化至少降低 50%，端到端无劣化，峰值内存满足限制。

### 阶段 4：稳定期

- 默认模式设为 `active`。
- `legacy` 保留至少一个发布周期。
- 收集真实实例的命中率和构建耗时。
- 移除 NetworkX 热路径另行设计和实施。

## 16. 回滚

设置：

```text
LLM4DRD_GRAPH_CONTEXT_MODE=legacy
```

即可恢复旧优化路径。仿真器、实例业务数据和现有图谱 API 不受影响；新增表可以保留；后续重新启用时继续通过 schema 和 builder 版本判断缓存是否可用。

## 17. 完成定义

必须同时满足：

- 展示图和计算图来自同一个 `CanonicalGraphBuilder`。
- 图谱 API 保持兼容。
- 优化器不再自行构建 NetworkX 图。
- 缓存未命中自动构建。
- L1 和 SQLite 命中可观测、可测试。
- fingerprint、事务和失效机制完整。
- 固定输入下优化结果严格一致。
- 缓存命中初始化至少降低 50%。
- 固定评价次数下端到端优化无劣化。
- 峰值内存不超过 legacy 的 1.25 倍。
- 可通过环境配置立即回退 legacy。
- 迁移、性能报告和运维说明完整。
