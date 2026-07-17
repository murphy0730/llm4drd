# 提示词：让 Codex 定位并修复「优化求解前后台进度不一致 / 进度条疑似造假」问题

> 用途：把下面 ```` ``` ```` 代码块里的提示词直接交给 Codex（或任何具备代码读写与执行能力的 Agent），
> 它会基于本项目**真实代码**定位"优化求解界面进度条一直缓慢前进但 20 分钟无结果"的根因，并修复，
> 使前台如实反映后台进度、后台能正常跑完算法并产出排产方案。
> 本文档为你（和同事）排查该问题而写；交付物是"改好后的代码 + 验证结果"，不是文档。

---

## 直接复制使用的提示词

```
你是一名资深全栈/运筹工程工程师。项目在 /Users/zhouwentao/Desktop/llm4drd（LLM4DRD 智能调度平台，
Python + FastAPI 后端 + 原生 JS 前端）。请定位并修复"优化求解界面进度条疑似造假"的问题：
用户在优化求解界面点击求解后，按钮一直显示"优化运行中"，进度条缓慢向前，但 20 分钟过去界面无任何
实质变化、无结果产出。目标：① 前台能正确反映后台真实进度；② 后台能正常执行算法，最终求解出优化的
排产方案（Pareto 解 / 排程）。

## 一、现象与规模（复现背景）
- 现象：进度条缓慢前进但长时间（≥20 分钟）无结果；用户怀疑进度条是假的。
- 数据规模（用户实测）：订单 516、任务令 4,363、工序 16,536、机器 1,149、工装 0、人员 0。
- 主入口界面是"混合优化（hybrid）"求解；另有两个相关入口：NSGA2、精确求解（exact），请一并检查是否
  有同类问题。

## 二、先读这些代码（用 grep/read 核实，禁止凭印象）。下面给出已定位的可疑点，请逐一验证：
后端进度机制（api/server.py）：
- POST /api/optimize/hybrid  （约 server.py:2883）：提交任务，bg.add_task(_run) 后台跑。
- _run 内的心跳线程（约 server.py:2923-2937）：每隔 OPTIMIZE_HEARTBEAT_INTERVAL_S 更新
  task["elapsed_s"] 和 task["updated_at"] —— 注意这是**墙钟时间**，与真实工作量无关。
- _progress 回调（约 server.py:3007）：把 optimizer 的 snapshot（generation / archive_size /
  total_evaluations / exact_evaluations / elapsed_s 等）写进 task 字典。
- optimizer.run(progress_callback=_progress)（约 server.py:3033）：真正跑算法。
- GET /api/optimize/hybrid/status/{task_id}（约 server.py:3081）：前端轮询的进度接口，返回上述字段。
优化引擎（optimization/hybrid_nsga3_alns.py）：
- HybridNSGA3ALNSOptimizer.run()（约 :250）：分阶段 coarse → exact_promotion → elite_refine →
  finalize；每个候选要做近似评估（仿真）和/或精确评估（OR-Tools CP-SAT）。请确认 progress_callback
  的**触发频率**：是否在单个耗时很长的评估（一次大仿真 / 一次 CP-SAT 求解）期间完全不回调？
前端进度（frontend/app_v2.js）：
- optimizeProgress(status)（约 app_v2.js:523）：计算进度百分比，请重点看：
  · generationRatio = current_generation / generations（约 :529）
  · timeRatio = elapsed_s / time_limit_s（约 :530）—— 由后端心跳按墙钟推进
  · coarse 阶段返回 8 + min(1, max(generationRatio, timeRatio)) * 57（约 :535）：**当后台卡在某步、
    generation 不前进时，timeRatio 仍随 elapsed_s 增大而把进度条缓慢往前推** → 这正是"假进度"观感。
  · exact_promotion=72、elite_refine=84、finalize=94（约 :536-538）是**写死的固定百分比**，该阶段
    无论跑多久进度条都不动，也无法反映真实子进度。
- 轮询：getOptimizeStatus（约 app_v2.js:298）轮询 /optimize/hybrid/status/{task_id}；pollTimer
  setInterval（约 app_v2.js:4697）。

## 三、根因假设（请验证，不必局限于此）
1. 前端进度条把"墙钟时间比例(timeRatio)"当进度用，后台卡住时仍缓慢前进 → 用户认为是假进度。
2. 后台在单步大评估（大仿真 / CP-SAT 精确求解）期间不回调进度，且前端无法区分"在算"与"卡死"
   （updated_at 只由心跳更新，无"真实进度停滞"标记）。
3. 后台算法本身在 1.6 万工序 / 1149 机器规模下过慢：每个候选的离散事件仿真派工存在全扫描瓶颈
   （参考 core/simulator.py 派工候选全扫描、每候选重建 features 字典，规模下乘积极大），导致单代
   耗时分钟级、长时间无 generation 推进 → 表现为"一直运行中却无结果"。

## 四、修复要求（前台 + 后台都要改，且保持前后台契约兼容）
【前台 frontend/app_v2.js】
- 进度条必须基于**后台真实工作量信号**，不要再用 timeRatio（elapsed_s/time_limit_s）去推进进度条；
  若要用时间做参考，单独标注为"预计剩余时间"，且不得与"进度%"混为一谈。
- 用后台返回的真实进度分数（见后台新增字段）驱动进度条；exact_promotion / elite_refine / finalize
  这些阶段必须给出**真实子进度**（如 本阶段候选 i/N、评估 j/M），替换掉写死的 72/84/94。
- 增加"诚实状态"提示：若超过阈值（如 60s）没有新的真实进度（generation/evaluations 未变），进度条
  停止前进，并显示"正在计算，已 N 秒无新进度，请稍候"之类文案，而不是继续默默往前走。
- 把任务"最近真实进度时间"醒目展示，让使用者能判断是真的在跑还是在卡。

【后台 api/server.py + optimization/】
- 让 progress_callback 以**较高频率**回传一个真实进度分数（0~1 或 0~100），覆盖所有阶段包括
  exact_promotion / elite_refine / finalize 的内部子进度（候选序号 / 评估序号 / CP-SAT 搜索进度等）。
- 增加"真实进度活跃度"标记：记录 last_real_progress_at（真实进度变化的时刻）。若长时间无真实进度，
  在 status 里给出 stalled/slow 提示字段，供前端诚实展示。
- 确保算法在 time_limit_s 内**能真正跑完并产出结果**：若瓶颈是单候选仿真/精确求解过慢，请优化它
  （例如复用已识别的派工反向索引、缓存 features、或对该阶段做迭代级进度回报），目标是后台"正常执行
  算法，求解出优化的排产方案"并最终 status=done、有 Pareto 解与排程。
- 保持 GET /api/optimize/hybrid/status/{task_id} 现有字段契约（前端在用），新字段以新增方式补充，
  不要重命名/删除已有字段，避免前后台断裂。
- NSGA2（server.py:3383 附近）与 exact 入口若有同样的"时间假进度 / 无子进度 / 卡死无提示"问题，
  按同样原则一并修复。

## 五、验证（修复后必须做）
1. 用中等规模实例（可临时缩小数据或用代表性子集）跑混合优化：确认进度条随真实工作前进、能到达 100%/done，
   并产出 Pareto 解与排程结果、写入可评审状态。
2. 注入"卡顿"对照：在评估循环里临时 sleep，确认前端不再把进度条当假进度往前推，而是显示诚实的
   "计算中/已 N 秒无新进度"。
3. 在用户真实规模（若能在本机承受）跑一次，记录总耗时与每阶段进度是否如实；若仍超时，至少给出明确的
   阶段进度与瓶颈说明，而不是无反馈。
4. 跑通回归：确认状态接口字段、前端渲染、结果评审链路未被破坏。

## 六、交付与约束
- 改动前用一段话说明要改的文件、逻辑、验证方式（不要一次性大改）。
- 改动后复测并按上面验证清单确认有效；核心引擎(optimization/、core/)改动需谨慎，优先用索引/缓存/
  参数/边界处理，避免改动算法语义。
- 最后汇报：① 根因结论（前/后台分别是什么）② 改了哪些文件+具体改动 ③ 验证结果（进度是否真实、
  是否产出方案、耗时）④ 遗留风险。
- 运行环境：项目自带 .venv（含 ortools/numpy）；导入包需 PYTHONPATH=/Users/zhouwentao/Desktop
  （磁盘目录名 llm4drd，包名 llm4drd_platform）。启动后端参考 run_server.py / api/server.py。
```

---

## 给你的使用说明（不进入上面的提示词，只给你看）
1. **为什么这样改原提示词**：你原来那句描述偏口语、没锚定代码，Codex 容易泛泛而谈或改错地方。
   优化版做了 4 件关键事：
   - **锚定真实代码**：列出确切的 `文件:行号` 可疑点（前端 `optimizeProgress` 的 `timeRatio` 假进度、
     `exact_promotion=72` 写死百分比；后端心跳只更新 `elapsed_s` 墙钟；`optimizer.run` 长评估不回调），
     让 Codex 直奔病灶。
   - **把"假进度"拆成可验证的假设**：前端用墙钟当进度、阶段写死百分比、后台卡住无子进度且无停滞标记
     三类，逐一验证。
   - **前后台都要修且保持契约**：明确要求前台停用地钟假进度、后台补真实子进度与停滞标记，并强调
     不能破坏 status 接口现有字段（否则前后台断裂）。
   - **验收闭环**：要求 Codex 用中等规模实例跑通、注入卡顿做对照、在真实规模验证，确保"后台真能产出
     排产方案"而不只是把进度条做得好看。
2. **怎么用**：把 ```` ``` ```` 内整段提示词交给 Codex（或支持代码执行的 Agent）即可。
3. **我的建议**：这个问题后台根因很可能就是你之前让我诊断过的**仿真派工瓶颈**（16k 工序规模下单候选
   仿真要分钟级），Codex 大概率会同时动 `core/simulator.py`。按你"先方案后实现"的偏好，建议让 Codex
   先只做**前台诚实化**（停用地钟假进度 + 停滞提示）这一轮，后台性能优化单独评估——这样即使后台暂时
   仍慢，界面也会如实告诉你是"在算"而非"假进度"，体验先止血。
