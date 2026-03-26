# LLM4DRD 前台重设计交接文档

## 1. 文档目标

这份文档面向“重新设计整个前台页面展示逻辑”的工作，目标不是解释全部算法细节，而是帮助新的前端程序、设计师或产品经理快速理解：

- 产品当前有哪些核心能力
- 页面按什么业务流程组织
- 每个页面依赖哪些接口和数据
- 哪些指标和口径需要特别注意
- 哪些内容适合重做，哪些内容建议保留

建议把这份文档作为新的前端信息架构、页面原型和组件拆分的基础说明书。

---

## 2. 产品定位

LLM4DRD 当前不是一个单纯的甘特图展示工具，而是一个“装配型离散制造调度决策平台”。系统的主链路是：

1. 生成或导入问题实例
2. 维护订单、任务、工序、机器、工装、人员、班次和停机
3. 构建异构图，用于结构洞察与优化引导
4. 用规则仿真快速验证实例与排程表现
5. 用混合优化生成多个 Pareto 候选方案
6. 用 AI 对方案进行比较、推荐、问答
7. 导出最终方案供线下复盘

当前前台已经具备完整闭环，但仍然偏“工程工作台”风格，适合继续重构成更清晰、更业务化的产品界面。

---

## 3. 当前页面结构

当前顶部 5 个主页面：

1. `问题配置`
2. `异构图`
3. `仿真排产`
4. `混合优化`
5. `大模型配置`

页面入口文件：
- [index.html](D:/github/llm4drd-platform/frontend/index.html)
- [app.js](D:/github/llm4drd-platform/frontend/app.js)
- [app.css](D:/github/llm4drd-platform/frontend/app.css)

### 3.1 问题配置

定位：实例建立与主数据维护中心。

包含功能：

- 随机生成实例
- Excel 导入
- Excel 模板导出
- CSV 导出
- 实例概览
- 机器类型 / 工装类型 / 人员概览
- 停机维护
- 订单维护
- 任务维护
- 工序维护
- 设备班次维护

用户价值：

- 把“模型世界”调成接近业务现实
- 让计划起点、班次、停机、工装、人员这些约束在仿真和优化前就具备可维护性

### 3.2 异构图

定位：结构认知与约束关系可视化页面。

包含功能：

- 异构图构建
- 节点类型筛选
- 边类型筛选
- 关键字搜索
- 图形化展示
- 节点详情
- 边类型概览
- 节点表 / 边表

当前图中实体层次：

- 订单
- 任务
- 工序
- 机器
- 工装
- 人员

### 3.3 仿真排产

定位：规则仿真与实例正确性快速检查页面。

包含功能：

- 选择启发式规则并运行仿真
- KPI 卡片
- 甘特图
- 排程明细表

当前甘特图已经支持：

- 任务条块
- 日期时间坐标轴
- 班次外遮罩
- 计划停机遮罩
- 非计划停机遮罩

### 3.4 混合优化

定位：多目标方案生成、对比、解释和导出中心。

包含功能：

- 业务目标选择（1-5 个）
- 优化参数设置
- 优化进度与历史
- 基线方案
- Pareto 方案库
- 启发式参考规则
- 方案对比
- AI 方案协同助手
- 选定方案导出

### 3.5 大模型配置

定位：AI 能力配置页。

包含功能：

- 常用模型快捷预设
- 手工配置 `base_url / model / api_key`
- 读取当前配置
- 保存配置
- 测试连通性

---

## 4. 推荐的业务主流程

如果未来重构前台，建议以“业务流程”而不是“技术模块”来组织页面。

推荐主流程：

1. `实例准备`
2. `约束校准`
3. `结构洞察`
4. `快速仿真`
5. `方案优化`
6. `方案评审`
7. `方案导出`

其中：

- `实例准备` 和 `约束校准` 可以合并成一个强引导式页面
- `结构洞察` 适合做成辅助页或侧边抽屉
- `方案评审` 应该独立出来，避免和优化参数混在一页

---

## 5. 关键接口清单

下面只列前台重设计最核心的接口。

### 5.1 实例与主数据

- `POST /api/instance/generate`
- `GET /api/instance/details`
- `GET /api/instance/db`
- `GET /api/instance/template`
- `POST /api/instance/import-excel`
- `GET /api/instance/csv`
- `PUT /api/instance/order/{order_id}`
- `PUT /api/instance/task/{task_id}`
- `PUT /api/instance/operation/{op_id}`
- `PUT /api/instance/machine/{machine_id}`

### 5.2 停机维护

- `GET /api/downtime`
- `POST /api/downtime`
- `PUT /api/downtime/{dt_id}`
- `DELETE /api/downtime/{dt_id}`

### 5.3 异构图

- `POST /api/graph/build`
- `GET /api/graph/meta`
- `GET /api/graph/nodes`
- `GET /api/graph/edges`
- `GET /api/graph/node/{node_id}/neighbors`

### 5.4 仿真

- `POST /api/simulate`
- `POST /api/simulate/compare`
- `POST /api/simulate/reference-solutions`

### 5.5 混合优化

- `GET /api/optimize/objectives`
- `POST /api/optimize/hybrid`
- `GET /api/optimize/hybrid/status/{task_id}`
- `GET /api/optimize/hybrid/result/{task_id}`
- `POST /api/optimize/hybrid/export-solution`

### 5.6 AI 方案助手

- `POST /api/ai/pareto/compare`
- `POST /api/ai/pareto/recommend`
- `POST /api/ai/pareto/ask`

### 5.7 大模型配置

- `GET /api/config/llm`
- `PUT /api/config/llm`
- `POST /api/config/llm/test`

---

## 6. 页面与接口映射

### 6.1 问题配置页

主要读取：

- `GET /api/instance/db`
- `GET /api/instance/details`
- `GET /api/downtime`

主要写入：

- `POST /api/instance/generate`
- `POST /api/instance/import-excel`
- `PUT /api/instance/order/{order_id}`
- `PUT /api/instance/task/{task_id}`
- `PUT /api/instance/operation/{op_id}`
- `PUT /api/instance/machine/{machine_id}`
- `POST /api/downtime`
- `PUT /api/downtime/{dt_id}`
- `DELETE /api/downtime/{dt_id}`

### 6.2 异构图页

主要读取：

- `GET /api/graph/meta`
- `GET /api/graph/nodes`
- `GET /api/graph/edges`
- `GET /api/graph/node/{node_id}/neighbors`

主要写入：

- `POST /api/graph/build`

### 6.3 仿真排产页

主要写入：

- `POST /api/simulate`

### 6.4 混合优化页

主要读取：

- `GET /api/optimize/objectives`
- `GET /api/optimize/hybrid/status/{task_id}`
- `GET /api/optimize/hybrid/result/{task_id}`

主要写入：

- `POST /api/optimize/hybrid`
- `POST /api/simulate/reference-solutions`
- `POST /api/ai/pareto/compare`
- `POST /api/ai/pareto/recommend`
- `POST /api/ai/pareto/ask`
- `POST /api/optimize/hybrid/export-solution`

### 6.5 大模型配置页

主要读取：

- `GET /api/config/llm`

主要写入：

- `PUT /api/config/llm`
- `POST /api/config/llm/test`

---

## 7. 关键返回数据结构

下面是前台重设计最关键的几个数据对象。

### 7.1 `instanceDetails`

来源：

- `GET /api/instance/details`

主要字段：

- `plan_start_at`
- `summary`
- `orders`
- `machines`
- `machine_types`
- `tooling_types`
- `toolings`
- `personnel`

重点说明：

- `machines` 现在已经包含：
  - `shifts`
  - `shift_windows`
  - `downtimes`

这意味着前台无需再额外请求班次和停机结构，就能在甘特图中绘制不可用时间块。

### 7.2 仿真结果 `simResult`

来源：

- `POST /api/simulate`

主要字段：

- `metrics`
- `gantt`

`gantt` 中常用字段：

- `order_id / order_name`
- `task_id`
- `op_id / op_name`
- `machine_id / machine_name`
- `tooling_ids`
- `personnel_ids`
- `start / end`
- `start_at / end_at`
- `duration`
- `elapsed_duration`
- `due_at`
- `is_main`
- `is_tardy`

### 7.3 混合优化结果 `optimizeResult`

来源：

- `GET /api/optimize/hybrid/result/{task_id}`

主要字段：

- `objective_keys`
- `objective_catalog`
- `baseline`
- `solutions`
- `archive_size`
- `requested_solution_count`
- `found_solution_count`
- `coarse_pool_size`
- `promoted_solution_count`
- `refined_solution_count`
- `generations_completed`
- `total_evaluations`
- `approximate_evaluations`
- `exact_evaluations`
- `elapsed_s`

其中：

- `baseline` 是基线规则方案
- `solutions` 是最终精确评价后的 Pareto 解

每个 `solution` 主要包含：

- `solution_id`
- `source`
- `generation`
- `rank`
- `feasible`
- `evaluation_mode`
- `objectives`
- `delta_vs_baseline`
- `candidate`
- `summary`
- `schedule`

### 7.4 AI 助手结果

来源：

- `POST /api/ai/pareto/compare`
- `POST /api/ai/pareto/recommend`
- `POST /api/ai/pareto/ask`

共同字段：

- `mode`
- `task_id`
- `used_solution_ids`
- `analysis`
- `display_text`
- `used_model`

UI 层当前主要直接消费：

- `display_text`
- `analysis.recommended_solution_id`

---

## 8. 当前指标体系

定义文件：

- [objectives.py](D:/github/llm4drd-platform/optimization/objectives.py)

当前系统支持的主要业务目标：

- `total_tardiness`
- `makespan`
- `main_order_tardy_count`
- `main_order_tardy_total_time`
- `main_order_tardy_ratio`
- `avg_utilization`
- `critical_utilization`
- `total_wait_time`
- `avg_flowtime`
- `max_tardiness`
- `tardy_job_count`
- `avg_tardiness`
- `total_completion_time`
- `max_flowtime`
- `bottleneck_load_balance`
- `tooling_utilization`
- `personnel_utilization`
- `assembly_sync_penalty`

### 指标口径提醒

这个部分前台在展示时必须谨慎，尤其不要擅自改名字而不改解释。

#### 8.1 利用率

当前 `avg_utilization / critical_utilization / tooling_utilization / personnel_utilization` 的口径更接近：

- `纯加工忙碌时间 / 计划总历时`

不是：

- `忙碌时间 / 净可用时间`

也不是：

- `忙碌时间 / 总历时扣除计划停机`

所以 UI 上建议写成：

- `机器占用率`
- 或 `当前口径利用率`

并增加一个说明 tooltip。

#### 8.2 等待时间

当前 `total_wait_time` 更接近：

- `flowtime - productive_time`

它混合了：

- 排队等待
- 前置等待
- 跨班次等待
- 停机造成的等待

所以不建议在前台把它写成“纯排队等待时间”。

#### 8.3 Makespan

当前对不可行解的分析层 `makespan` 带有惩罚性补值，不完全等于“实际最后完工时刻”。

对最终精确可行 Pareto 解来说问题不大，但在中间态或失败态里，展示文案要写得保守。

---

## 9. 时间语义

这是前台重构必须保留的一条原则。

### 9.1 内部计算

内部优化和仿真时间统一使用：

- 相对 `plan_start_at` 的数值小时

原因：

- 仿真更快
- 算法实现更稳定
- 求解器和近似评估器更容易处理

### 9.2 前台展示

前台展示应该优先用：

- `start_at / end_at / due_at / plan_start_at`

也就是绝对日期时间。

当前甘特图已经改成：

- 横轴显示日期时间
- 条块 tooltip 显示日期时间范围
- 不可用时间块也显示日期时间范围

未来如果重构，建议继续坚持：

- 计算用数值
- 展示用日期时间

---

## 10. 当前前台状态流

### 10.1 关键全局状态

当前前端主要状态在 [app.js](D:/github/llm4drd-platform/frontend/app.js) 中。

核心状态有：

- `instanceDb`
- `instanceDetails`
- `downtimes`
- `graphMeta / graphNodes / graphEdges`
- `simResult`
- `objectiveCatalog`
- `selectedObjectiveKeys`
- `optimizeTaskId`
- `optimizeStatus`
- `optimizeResult`
- `selectedReferenceRules`
- `referenceRuleSolutions`
- `llmConfig`
- `paretoAssistant`
- `paretoConversation`

### 10.2 当前交互模式

前端目前是：

- 单页应用
- 前台 JS 全局状态驱动
- 点击事件通过 `data-action` 派发

如果你要用别的程序重做前台，建议把它拆成：

- 页面级 store
- 接口服务层
- 图表组件层
- 对话组件层

---

## 11. 推荐的新前台信息架构

建议不要继续把“优化参数配置”和“方案评审”混在同一块里。

推荐拆成以下页面或子流程：

### 11.1 实例准备

聚焦：

- 新建实例
- 导入模板
- 资源概览

### 11.2 约束维护

聚焦：

- 停机维护
- 订单维护
- 任务维护
- 工序维护
- 设备班次维护

### 11.3 结构洞察

聚焦：

- 异构图
- 关键链
- 共享资源冲突
- 装配同步风险

### 11.4 快速验证

聚焦：

- 启发式规则仿真
- KPI 看板
- 甘特图

### 11.5 方案生成

聚焦：

- 目标选择
- 搜索预算
- 快速 / 平衡 / 深度模式
- 运行进度

### 11.6 方案评审

聚焦：

- 基线 vs Pareto vs 启发式参考
- 甘特差异
- 指标差异
- 订单差异
- 瓶颈差异
- AI 比较与推荐

### 11.7 导出与复盘

聚焦：

- 导出当前方案
- 导出对比报告
- 导出规则画像

---

## 12. 最值得重构的展示点

### 12.1 混合优化页应该拆分

当前混合优化页功能很多，建议拆成三段：

- `优化配置`
- `优化进度`
- `方案评审`

### 12.2 甘特图建议升级

推荐新增：

- 顶部时间缩放
- 机器/订单筛选
- 颜色图例固定
- 点击条块右侧详情抽屉
- 不可用时间块开关

### 12.3 方案对比建议升级

当前已支持多方案对比，但还可以继续补：

- 订单差异视图
- 瓶颈差异视图
- 甘特差异视图
- 方案原因卡

### 12.4 AI 助手建议独立成侧边协同面板

现在 AI 助手已经是聊天形式，但仍嵌在优化页底部。

更好的形态是：

- 固定侧边栏
- 当前上下文始终绑定“选中方案集”
- 支持引用具体方案卡与甘特片段

---

## 13. 当前已经补齐的结果交付能力

这一部分是重设计时建议保留的亮点。

### 13.1 方案导出

接口：

- `POST /api/optimize/hybrid/export-solution`

导出 Excel 当前包含：

- `summary`
- `schedule`
- `machine_calendar`
- `rule_profile`

### 13.2 甘特图不可用时间显示

当前最终方案展示已经支持：

- 班次外
- 计划停机
- 非计划停机

这对业务解释“为什么这里空着”非常关键。

### 13.3 启发式参考规则参与方案评审

当前支持把：

- `ATC / EDD / SPT / ...`

这些规则作为“参考候选”一起参与：

- 方案对比
- AI 比较
- AI 推荐
- AI 问答

这是一条非常有价值的业务解释路径，建议保留。

---

## 14. 当前已知产品层限制

前台重设计时建议显式说明，不要在视觉上暗示“系统已经全覆盖所有工业约束”。

当前系统虽然已经支持：

- 订单 / 任务 / 工序
- 装配依赖
- 机器 / 工装 / 人员
- 班次 / 停机
- 异构图
- 仿真
- 多目标优化
- AI 比较与推荐

但仍未完整覆盖的典型工业要素包括：

- sequence-dependent setup / changeover
- 批处理设备
- 运输 / 缓冲区
- 物料齐套 / 库存 / 到料
- 不可中断工序
- lot split / overlap
- 返工 / 报废
- 稳定性惩罚 / 冻结窗口

所以产品展示建议定位为：

- `装配型离散制造调度与方案评审平台`

不要过早包装成“全约束 APS”。

---

## 15. 给新前端程序的实现建议

如果你要用另一个程序重做前台，建议：

### 15.1 组件拆分

- `InstanceSetupPanel`
- `ConstraintEditor`
- `GraphExplorer`
- `SimulationWorkbench`
- `OptimizationWorkbench`
- `ScenarioComparisonPanel`
- `ParetoChatAssistant`
- `LLMConfigPanel`

### 15.2 数据层拆分

- `instanceApi`
- `graphApi`
- `simulationApi`
- `optimizationApi`
- `aiApi`
- `configApi`

### 15.3 统一的展示策略

- 所有表格字段统一带中英文 key 对照
- 所有时间字段默认展示绝对时间
- 所有 KPI 卡统一带 tooltip 解释口径
- 所有方案 ID 与规则名都支持复制和导出

---

## 16. 建议优先级

如果只做一轮前台重构，建议优先顺序：

1. 重新设计“方案生成 + 方案评审”主页面
2. 把 AI 助手做成真正的协同侧栏
3. 把实例准备与约束维护做成更强引导式页面
4. 升级异构图页的筛选与详情交互
5. 统一 KPI 口径说明与时间展示规范

---

## 17. 关联文件

关键前端文件：

- [index.html](D:/github/llm4drd-platform/frontend/index.html)
- [app.js](D:/github/llm4drd-platform/frontend/app.js)
- [app.css](D:/github/llm4drd-platform/frontend/app.css)

关键后端文件：

- [server.py](D:/github/llm4drd-platform/api/server.py)
- [hybrid_nsga3_alns.py](D:/github/llm4drd-platform/optimization/hybrid_nsga3_alns.py)
- [objectives.py](D:/github/llm4drd-platform/optimization/objectives.py)
- [simulator.py](D:/github/llm4drd-platform/core/simulator.py)
- [models.py](D:/github/llm4drd-platform/core/models.py)

---

## 18. 一句话总结

这个产品当前最有价值的，不是“把所有算法都堆在一个页面里”，而是：

**把实例准备、约束维护、结构洞察、方案生成、方案评审、AI 协同和结果导出，组织成一条让业务能理解、能比较、能落地的完整决策链路。**

