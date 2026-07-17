# -*- coding: utf-8 -*-
"""附录A 术语表 / 附录B 参考文献与源码锚点 / 附录C 致谢与下一步。"""

BLOCKS = [
    # ================================================================ 附录A
    ("h1", "附录A 术语表（Glossary）"),
    ("p", "读完全书再回头，你会发现调度这个领域的行话其实就那么几十个。本附录把它们按四个主题"
          "汇总成表，每条只用一句大白话解释，方便你随时查阅。表中所有定义都与正文口径一致，"
          "凡是涉及公式的术语（松弛、延误、利用率等），都可以在正文相应章节找到完整推导。"),

    ("h2", "A.1 车间里的基本对象"),
    ("table", {
        "headers": ["术语", "英文", "一句话大白话定义"],
        "rows": [
            ["**调度 / 排产**", "Scheduling",
             "决定『哪道工序、在哪台机器、什么时候开始干』的工作，相当于给整个后厨排一张时刻表。"],
            ["**订单**", "Order",
             "客户下的要货单，写着要什么、什么时候要（交期），例如迷你案例里的订单A（齿轮轴）。"],
            ["**任务令 / 任务**", "Task",
             "订单下达到车间后的执行单元，一个订单对应一个任务，任务里装着该订单的全部工序。"],
            ["**工序**", "Operation",
             "一道具体的加工步骤，相当于菜谱里的一步，例如 A1 粗车 3h。"],
            ["**机器**", "Machine",
             "干活的设备，相当于厨师；同一时刻一台机器只能干一道工序。"],
            ["**工艺类型**", "process type",
             "工序需要的加工类别（如车、铣、磨），机器只有具备对应类型才能接这道工序。"],
            ["**工装**", "Tooling",
             "加工时要用的夹具、模具等辅助工具，数量有限，可能要多道工序共用一套。"],
            ["**人员**", "Personnel",
             "操作机器的工人资源；机器再空，没有有资质的工人也开不了工。"],
            ["**前驱 / 后继**", "Predecessor / Successor",
             "工序间的先后约束：前驱没干完，后继不能开工，就像不粗车就不能精车。"],
            ["**流转时间**", "turnover time",
             "前驱工序完工后，工件搬运、冷却等不占机器的挂钟等待；本工序最早开工不早于"
             "前驱完工时刻 + 前驱的 turnover（见 core/models.py:get_operation_flow_ready_time）。"],
            ["**可选机器**", "eligible machines",
             "具备本工序所需工艺类型、能合法加工它的机器集合"
             "（见 core/models.py:get_eligible_machines）。"],
        ],
        "caption": "表A-1 基本对象类术语",
    }),

    ("h2", "A.2 时间与绩效指标"),
    ("table", {
        "headers": ["术语", "英文", "一句话大白话定义"],
        "rows": [
            ["**交期**", "due date",
             "客户要求交货的时刻，例如订单B要求 8h 内交付；晚于它就是延误。"],
            ["**放行时间**", "release time",
             "工件到达车间、可以开始加工的时刻；迷你案例里三张订单都是 t=0 放行。"],
            ["**派生交期**", "derived due date",
             "把订单交期沿工艺链反推到每道工序的内部小交期（A1=5、A2=7、A3=11……），"
             "见 core/models.py:derive_internal_targets。"],
            ["**总完工时间**", "makespan",
             "所有任务完工时刻的最大值，即最后一桌客人吃完买单的时刻；越短说明机器排得越紧凑。"],
            ["**流程时间**", "flow time",
             "任务从放行到完工的挂钟时长；迷你案例 FIFO 下 A/B/C 为 13/12/9，平均 11.33。"],
            ["**延误**", "tardiness",
             "max(0, 完工时刻 − 交期)，晚交几个小时就是几；提前完工不计负，延误最小就是 0。"],
            ["**松弛**", "slack",
             "派生交期 − 当前时刻 − 任务剩余工时；负数表示照现在的节奏已经来不及，"
             "例如 t=0 时 A1 的 slack = 5 − 9 = −4。"],
            ["**在制品**", "WIP (Work In Process)",
             "车间里已开工但还没完工的工件总量；堆得越多，占用资金和场地越多。"],
            ["**设备利用率**", "utilization",
             "单台机器 = 该机器累计忙碌时间 / makespan，项目 KPI 取各机器平均；"
             "迷你案例 FIFO 下两台机器合计忙碌 25h，25 / (2 × 13) ≈ 0.96。"],
        ],
        "caption": "表A-2 时间与指标类术语",
    }),

    ("h2", "A.3 派工与仿真"),
    ("table", {
        "headers": ["术语", "英文", "一句话大白话定义"],
        "rows": [
            ["**派工规则**", "dispatching rule",
             "机器空出来时的排队叫号规则，给每个等待工序打个分、分高者先上，"
             "例如 SPT、EDD、ATC（见 core/rules.py 的 11 条内置规则）。"],
            ["**就绪队列**", "ready queue",
             "已放行且前驱全部完工、此刻可以开工的工序集合，即叫号机前排着的那一队。"],
            ["**离散事件仿真**", "discrete-event simulation",
             "不一秒一秒地磨时间，而是直接跳到下一个事件（开工、完工、故障）发生的时刻"
             "推进时钟的仿真方法（见 core/simulator.py:Simulator.run）。"],
            ["**事件堆 / 事件队列**", "event queue",
             "按发生时刻排好序的事件列表，仿真器每次弹出最近的一个事件来处理。"],
        ],
        "caption": "表A-3 派工与仿真类术语",
    }),

    ("h2", "A.4 优化算法"),
    ("table", {
        "headers": ["术语", "英文", "一句话大白话定义"],
        "rows": [
            ["**适应度**", "fitness",
             "衡量一个候选解（一组参数或一条规则）好坏的分数，进化算法靠它决定谁留下。"],
            ["**支配**", "dominance",
             "解 a 在所有目标上都不比 b 差、且至少一个目标更好，就说 a 支配 b"
             "（见 optimization/pareto.py:dominates）。"],
            ["**帕累托最优**", "Pareto optimal",
             "没有任何解能支配它：想再改进一个目标，就必然牺牲另一个，已经没法白赚了。"],
            ["**帕累托前沿**", "Pareto front",
             "所有帕累托最优解在目标空间连成的边界线，是多目标优化交给决策者的菜单。"],
            ["**非支配排序**", "non-dominated sort",
             "按支配关系把解分层：谁都不支配的是第 1 层，被第 1 层支配的是第 2 层，依此类推"
             "（见 optimization/nsga3_core.py:fast_nondominated_sort）。"],
            ["**参考点**", "reference point",
             "NSGA-III 在目标空间均匀撒下的一把锚点，让每个解挂靠到最近的锚点，"
             "保证前沿摊开而不是挤成一团。"],
            ["**小生境**", "niche",
             "挂靠到同一个参考点的解形成的小圈子；圈子里人太多就得择优淘汰，维持多样性。"],
            ["**交叉 / 变异**", "crossover / mutation",
             "交叉是把两个父代解的参数拼出孩子，变异是随机小改几个参数，"
             "两者是进化算法产生新解的方式。"],
            ["**大邻域搜索**", "LNS (Large Neighborhood Search)",
             "每次把当前解拆掉一大块再重新修好的改进方法，步子比局部微调大得多。"],
            ["**破坏 / 修复算子**", "destroy / repair operator",
             "ALNS 里负责拆和装的两类操作手，例如拆掉同一装配链的工序再按装配同步装回"
             "（见 optimization/hybrid_nsga3_alns.py）。"],
            ["**模拟退火**", "simulated annealing",
             "允许以一定概率接受变差解的搜索策略，温度越高越敢冒险，慢慢冷却后只接受改进，"
             "借此跳出局部最优。"],
            ["**约束规划**", "constraint programming",
             "把问题的硬约束写清楚、交给求解器自动搜索可行最优解的建模范式。"],
            ["**CP-SAT**", "CP-SAT",
             "Google OR-Tools 自带的约束规划求解器，本项目精确求解靠它"
             "（见 optimization/exact.py:ExactSolver.solve）。"],
            ["**AddNoOverlap**", "AddNoOverlap",
             "CP-SAT 的一条建模原语：同一台机器上的工序区间两两不得重叠，"
             "一句话写尽『一台机器一次只干一道工序』。"],
            ["**区间变量**", "interval variable",
             "CP-SAT 里带开始时刻、时长、结束时刻的决策变量，一道工序占机器的一段就是一个区间。"],
            ["**滚动时域**", "rolling horizon",
             "在线排产策略：每次只认真排眼前一段窗口，时间到了就用最新现场状态重排下一段"
             "（见 scheduling/online.py:OnlineSchedulerV3）。"],
            ["**LLM 规则进化**", "LLM rule evolution",
             "让大模型当『变异算子』生成、改写派工规则代码，再用仿真打适应度逐代筛选，"
             "见 ai/evolution.py 与 core/rules.py:compile_rule_from_code。"],
        ],
        "caption": "表A-4 优化算法类术语",
    }),

    # ================================================================ 附录B
    ("h1", "附录B 参考文献与源码锚点"),
    ("p", "本书有个别的教材没有的特点：所有概念都落在一个真实项目的源码上。B.1 把正文引用过的"
          "源码位置汇成一张索引表，方便你按图索骥去读代码；B.2 是几篇经典文献，想往深走可以从"
          "它们开始。"),

    ("h2", "B.1 本书引用的项目源码锚点"),
    ("p", "下表按『先基础后进阶』的阅读顺序排列。建议读代码时手里同时拿着正文对应章节，"
          "对照着看。"),
    ("table", {
        "headers": ["文件", "关键函数或类", "讲了什么"],
        "rows": [
            ["`core/models.py`",
             "`ShopFloor`、`derive_internal_targets`、"
             "`get_operation_flow_ready_time`、`get_eligible_machines`",
             "全部数据模型；订单交期沿工艺链反推派生交期、工序流转就绪时刻、可选机器的计算。"],
            ["`core/rules.py`",
             "`edd_rule` 等 11 条内置规则、`BUILTIN_RULES`、`compile_rule_from_code`",
             "派工规则打分函数库（分数越大越优先），以及把一段 Python 源码编译成规则的桥梁。"],
            ["`core/simulator.py`",
             "`Simulator.run`、`_features`、`_compute_kpi`",
             "离散事件仿真主循环；特征字典（slack、wait_time 等）的构建；makespan、延误、"
             "利用率等 KPI 的统计口径。"],
            ["`core/sim_runtime.py`",
             "`SimulationRuntime`、`detect_dependency_cycles`、`SimulationRuntimePool`",
             "在线运行时的仿真托管：开工前检测依赖环，按车间实例池化复用仿真器。"],
            ["`scheduling/online.py`",
             "`OnlineSchedulerV3`、`reschedule`、`on_breakdown`",
             "在线滚动排产：滚动时域重排、机器故障后的应急响应。"],
            ["`optimization/pareto.py`",
             "`dominates`、`ParetoOptimizer`",
             "支配关系的判定，多目标优化器的骨架。"],
            ["`optimization/nsga3_core.py`",
             "`fast_nondominated_sort`、`generate_reference_points`、"
             "`normalize_vectors`、`associate_to_reference`、`select_survivors`",
             "NSGA-III 核心：非支配排序、参考点生成、目标归一化、参考点挂靠与生存选择。"],
            ["`optimization/alns_core.py`",
             "`ALNSCore.refine`、`_update_operator`",
             "ALNS 精化主循环，以及按表现动态调整破坏/修复算子权重的机制。"],
            ["`optimization/hybrid_nsga3_alns.py`",
             "`assembly_chain_destroy`、`bottleneck_machine_destroy`、"
             "`shared_tooling_destroy`、`assembly_sync_repair`、`HybridConfig`",
             "混合算法的领域算子：按装配链/瓶颈机器/共用工装拆，按装配同步修；混合配置项。"],
            ["`optimization/exact.py`",
             "`ExactSolver.solve`",
             "CP-SAT 精确求解：用区间变量与 AddNoOverlap 建模，求 makespan 或总延误最优。"],
            ["`optimization/archive.py`",
             "`ParetoArchive`",
             "非支配解集（帕累托档案）的保存与增量维护。"],
            ["`optimization/approx_eval.py`",
             "`ApproximateScheduleEvaluator`",
             "候选参数的近似快速评估，少跑仿真也能粗筛解。"],
            ["`optimization/solution_model.py`",
             "`CandidateParameters`",
             "候选解的数据结构：一套待评估的排产参数。"],
            ["`ai/evolution.py`",
             "`RuleIndividual`、`LLMInterface`、`EvolutionConfig`、`EvolutionEngine.evolve`",
             "LLM 规则进化：规则个体、大模型接口、进化配置与逐代进化主流程。"],
            ["`knowledge/graph.py`",
             "`HeterogeneousGraph`",
             "车间异构知识图谱：订单、任务、工序、机器等节点与它们之间的关系。"],
        ],
        "caption": "表B-1 项目源码锚点索引",
    }),

    ("h2", "B.2 经典文献（课外阅读）"),
    ("table", {
        "headers": ["文献", "一句话说明"],
        "rows": [
            ["Pinedo, Scheduling: Theory, Algorithms, and Systems",
             "调度领域的标准教科书，从单机排到车间调度的理论体系最全，本书多处术语口径以它为参照。"],
            ["Deb & Jain (2014), IEEE TEVC",
             "NSGA-III 原始论文，『参考点 + 小生境』保持多目标多样性的思想就出自这里。"],
            ["Ropke & Pisinger (2006), Transportation Science",
             "ALNS 原始论文，破坏-修复框架与算子权重自适应机制的出处。"],
            ["Google OR-Tools CP-SAT 官方文档",
             "本项目精确求解器所用工具的官方说明，区间变量、AddNoOverlap 等建模原语都在这里。"],
            ["Vepsalainen & Morton (1987), Management Science",
             "ATC 规则原始论文，用指数项在『赶交期』与『挑短活』之间做权衡的经典设计。"],
        ],
        "caption": "表B-2 推荐课外阅读",
    }),

    # ================================================================ 附录C
    ("h1", "附录C 致谢与下一步"),
    ("p", "本书基于 LLM4DRD 项目的真实源码撰写，没有虚构的玩具示例：正文里的每一条规则、每一个"
          "指标、每一张甘特图，都能在项目里找到对应的代码。书中统一使用的迷你案例（3 张订单、"
          "7 道工序、2 台机器）由 docgen/mini_case.py 直接调用项目的 core.models、core.simulator "
          "与 optimization.exact 跑出，所有数字均可原样复现。感谢项目中每一位写代码、写测试、"
          "挑毛病的同事。"),
    ("p", "复现方式如下（在项目根目录执行）："),
    ("code", "cd /Users/zhouwentao/Desktop/llm4drd\n"
             ".venv/bin/python docgen/mini_case.py\n"
             "# 敏感性实验：用环境变量改三张订单的交期\n"
             "DUES=\"13,10,15\" .venv/bin/python docgen/mini_case.py"),

    ("h2", "下一步建议"),
    ("numbers", [
        "**动手改一条规则**：从 core/rules.py 抄一条最简单的内置规则（比如 spt_rule），"
        "改一改它的打分方式，或者用 compile_rule_from_code 写一条完全属于你自己的规则，"
        "然后跑 docgen/mini_case.py 对比 makespan 和总延误——亲手让指标变好或变差一次，"
        "胜过再读十页书。",
        "**按顺序读一遍源码**：core/models.py（数据模型与派生交期）→ core/rules.py（打分）"
        "→ core/simulator.py（事件循环与 KPI）→ optimization/pareto.py 与 "
        "optimization/nsga3_core.py（多目标）→ optimization/exact.py（精确解对照）"
        "→ ai/evolution.py（LLM 规则进化）。这条路线与本书章节顺序一致，"
        "每一站都有正文可以对照。",
        "**复现并扰动案例**：先原样运行 docgen/mini_case.py，核对书中的每一组数字；"
        "再用 DUES 环境变量把交期改紧或改松，观察各条规则名次的翻转——规则没有绝对好坏，"
        "只有适不适合当前的交期压力，这是全书最想让你带走的一课。",
    ]),
    ("p", "调度的水很深，但池子是陪你蹚过来的。祝排产顺利，机器不闲，交期不误。"),
]
