"""
LLM 双专家进化引擎 v3
=====================
核心变更:
  - 所有LLM调用的输入/输出完整记录和回调
  - 支持任意 OpenAI 兼容 API
  - 适配新数据模型
"""
import json, random, time, copy, logging, inspect, math
from dataclasses import dataclass, field
from typing import Optional, Callable
from ..core.models import ShopFloor
from ..core.simulator import Simulator
from ..core.rules import BUILTIN_RULES, compile_rule_from_code

logger = logging.getLogger(__name__)

@dataclass
class RuleIndividual:
    id: str
    code: str
    name: str = ""
    fitness: float = float('inf')
    llm_score: float = 0.0
    hybrid_score: float = float('inf')
    generation: int = 0

    @property
    def compiled(self):
        return compile_rule_from_code(self.code)

@dataclass
class EvolutionConfig:
    population_size: int = 8
    elite_size: int = 3
    max_generations: int = 15
    patience: int = 5
    temperature_min: float = 0.3
    temperature_max: float = 1.5
    objective_weight: float = 0.7
    llm_eval_weight: float = 0.3


@dataclass
class LLMCallLog:
    """LLM 调用记录 — 让用户看到双AI系统在做什么"""
    timestamp: float
    role: str          # "LLM-A" or "LLM-S"
    action: str        # "generate" / "crossover" / "mutate" / "evaluate"
    prompt: str
    response: str
    generation: int = 0
    duration_s: float = 0.0


class LLMInterface:
    """支持任意 OpenAI 兼容 API, 所有调用记录到 call_logs"""

    def __init__(self, api_key=None, base_url=None, model=None):
        from ..config import get_config
        cfg = get_config().llm
        self.api_key = api_key or cfg.api_key
        self.base_url = base_url or cfg.base_url
        self.model = model or cfg.model
        self.max_tokens = cfg.max_tokens
        self.timeout = cfg.timeout
        self.use_real = bool(self.api_key)
        self.call_logs: list[LLMCallLog] = []
        self._on_call: Optional[Callable] = None  # 回调: fn(log_entry)

    def set_callback(self, fn):
        """设置调用回调, 每次LLM调用后触发, 用于前端实时展示"""
        self._on_call = fn

    def call(self, prompt: str, role: str, action: str, gen: int = 0, temp: float = 1.0) -> str:
        t0 = time.time()
        if self.use_real:
            resp = self._call_api(prompt, temp)
        else:
            resp = self._template_fallback(prompt)
        dur = time.time() - t0

        log = LLMCallLog(
            timestamp=time.time(), role=role, action=action,
            prompt=prompt[:2000], response=resp[:2000],
            generation=gen, duration_s=round(dur, 2)
        )
        self.call_logs.append(log)
        if self._on_call:
            try: self._on_call(log)
            except: pass
        return resp

    def _call_api(self, prompt, temp):
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            r = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp, max_tokens=self.max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM API error: {e}")
            return self._template_fallback(prompt)

    def _template_fallback(self, prompt):
        feats = [
            "f.get('slack',0)", "f.get('remaining',1)", "f.get('urgency',0)",
            "f.get('priority',1)/5", "f.get('processing_time',1)",
            "f.get('prereq_ratio',0.5)", "f.get('is_main',0)",
            "f.get('wait_time',0)", "f.get('progress',0)",
        ]
        sel = random.sample(feats, min(4, len(feats)))
        ws = [round(random.uniform(-3, 3), 2) for _ in range(4)]
        formulas = [
            f"{sel[0]}*{ws[0]}+{sel[1]}*{ws[1]}",
            f"({sel[0]}+1)/({sel[1]}+0.01)*{ws[0]}+{sel[2]}*{ws[1]}",
            f"max({sel[0]}*{ws[0]},{sel[1]}*{ws[1]})+{sel[2]}*{ws[2]}",
            f"math.exp(-{sel[0]}/({sel[1]}+1))*{ws[0]}+{sel[2]}*{ws[1]}",
        ]
        formula = random.choice(formulas)
        return f'''import math
def evolved_rule(op, machine, f, shop):
    try:
        urg_bonus = abs(f.get('slack',0))*2 if f.get('slack',0)<0 else 0
        return {formula} + urg_bonus
    except:
        return -f.get('due_date',9999)
'''


SYSTEM_PROMPT = """你是算法专家(LLM-A), 负责设计调度优先级规则。
函数签名: def evolved_rule(op, machine, f, shop) -> float (越大越优先)

f 中可用特征:
  slack: 交期-当前时间-剩余加工时间 (负数=紧急)
  remaining: 任务剩余总加工时间
  processing_time: 当前工序加工时间
  due_date: 订单交期
  urgency: max(0, -slack)
  progress: 任务完成进度 0-1
  priority: 订单优先级 1-5
  is_main: 是否主任务 (1.0/0.0)
  wait_time: 已等待时间
  prereq_ratio: 前置完成比例 0-1
  machine_busy_time: 机器累计忙碌时间

请输出完整的 Python 函数代码, 函数名 evolved_rule。"""

EVAL_PROMPT = """你是调度专家(LLM-S), 评估以下规则的质量。
规则代码:
```python
{code}
```
在测试实例上的表现: 总延迟={fitness}, 主订单延误数={tardy}
请用JSON回复: {{"score": 0-10, "feedback": "改进建议", "weaknesses": ["弱点1"]}}"""


class EvolutionEngine:
    def __init__(self, config: EvolutionConfig, llm: LLMInterface = None):
        self.config = config
        self.llm = llm or LLMInterface()
        self.population: list[RuleIndividual] = []
        self.best: Optional[RuleIndividual] = None
        self.history: list[dict] = []

    def initialize(self, seeds=None):
        self.population = []
        names = seeds or ["ATC", "EDD", "KIT_AWARE"]
        for n in names[:self.config.elite_size]:
            if n in BUILTIN_RULES:
                src = inspect.getsource(BUILTIN_RULES[n])
                # 重命名为 evolved_rule
                src = src.replace(f"def {n.lower()}_rule(", "def evolved_rule(")
                src = src.replace(f"def {n.lower().replace('_','')}(", "def evolved_rule(")
                # Simpler: just wrap it
                wrapped = f"import math\ndef evolved_rule(op, machine, f, shop):\n    return BUILTIN_RULES['{n}'](op, machine, f, shop)"
                # Actually let's generate properly
                self.population.append(RuleIndividual(
                    id=f"elite_{n}", code=self._make_elite_code(n), name=f"Elite_{n}"))
        # Fill with LLM generated
        for i in range(self.config.population_size - len(self.population)):
            code = self.llm.call(SYSTEM_PROMPT + "\n设计一个新颖的调度规则。", "LLM-A", "generate", 0)
            code = self._extract_code(code)
            self.population.append(RuleIndividual(id=f"init_{i}", code=code, name=f"Init_{i}"))

    def _make_elite_code(self, name):
        """生成独立可编译的内置规则包装"""
        mapping = {
            "EDD": "return -f.get('due_date',9999)",
            "SPT": "return -f.get('processing_time',1)",
            "ATC": """import math
    p=f.get('processing_time',1)
    if p<=0: return 1000
    s=max(f.get('slack',0),0)
    return (1/p)*math.exp(-s/(2*p+0.01))""",
            "CR": """r=f.get('remaining',1)
    if r<=0: return 1000
    return -(f.get('slack',0)/r)""",
            "KIT_AWARE": "return f.get('prereq_ratio',0.5)*5+f.get('urgency',0)*2+f.get('priority',1)",
            "BOTTLENECK": "return f.get('is_main',0)*8+f.get('urgency',0)*3+f.get('priority',1)*2",
        }
        body = mapping.get(name, "return -f.get('due_date',9999)")
        return f"import math\ndef evolved_rule(op, machine, f, shop):\n    {body}"

    def evolve(self, train_instances: list[ShopFloor], callback=None) -> Optional[RuleIndividual]:
        if not self.population:
            self.initialize()

        best_fit = float('inf')
        patience = 0

        for gen in range(self.config.max_generations):
            # Evaluate
            for ind in self.population:
                try:
                    func = compile_rule_from_code(ind.code)
                    fits = []
                    for inst in train_instances:
                        r = Simulator(inst, func).run()
                        fits.append(r.total_tardiness)
                    ind.fitness = sum(fits) / len(fits)
                except Exception:
                    ind.fitness = float('inf')

            # LLM-S evaluate top-3
            sorted_pop = sorted(self.population, key=lambda x: x.fitness)
            for ind in sorted_pop[:3]:
                if ind.fitness < float('inf'):
                    prompt = EVAL_PROMPT.format(code=ind.code[:500], fitness=f"{ind.fitness:.1f}", tardy="?")
                    resp = self.llm.call(prompt, "LLM-S", "evaluate", gen)
                    try:
                        d = json.loads(resp)
                        ind.llm_score = d.get("score", 5.0)
                    except:
                        ind.llm_score = 5.0 if ind.fitness < 1000 else 3.0

            # Hybrid score
            fits = [p.fitness for p in self.population if p.fitness < float('inf')]
            if fits:
                mn, mx = min(fits), max(fits)
                rng = mx - mn if mx > mn else 1
                for p in self.population:
                    if p.fitness >= float('inf'):
                        p.hybrid_score = float('inf')
                    else:
                        p.hybrid_score = (self.config.objective_weight * (p.fitness - mn) / rng +
                                          self.config.llm_eval_weight * (10 - p.llm_score) / 10)

            self.population.sort(key=lambda x: x.hybrid_score)
            if self.population[0].fitness < best_fit:
                best_fit = self.population[0].fitness
                self.best = copy.deepcopy(self.population[0])
                patience = 0
            else:
                patience += 1

            info = {"gen": gen+1, "best": round(best_fit, 1),
                    "avg": round(sum(p.fitness for p in self.population if p.fitness<float('inf')) / max(1,len(fits)), 1)}
            self.history.append(info)
            if callback: callback(info)
            if patience >= self.config.patience: break

            # Next generation
            elites = self.population[:self.config.elite_size]
            new_pop = list(elites)
            while len(new_pop) < self.config.population_size:
                if random.random() < 0.5 and len(elites) >= 2:
                    a, b = random.sample(elites, 2)
                    prompt = f"{SYSTEM_PROMPT}\n\n组合以下两个高性能规则:\nA({a.fitness:.0f}):\n{a.code[:400]}\nB({b.fitness:.0f}):\n{b.code[:400]}\n输出新规则。"
                    code = self.llm.call(prompt, "LLM-A", "crossover", gen+1,
                                         temp=random.uniform(self.config.temperature_min, self.config.temperature_max))
                else:
                    parent = random.choice(elites)
                    prompt = f"{SYSTEM_PROMPT}\n\n改进此规则(fitness={parent.fitness:.0f}):\n{parent.code[:400]}\n输出改进后的规则。"
                    code = self.llm.call(prompt, "LLM-A", "mutate", gen+1,
                                         temp=random.uniform(self.config.temperature_min, self.config.temperature_max))
                code = self._extract_code(code)
                new_pop.append(RuleIndividual(id=f"g{gen+1}_{len(new_pop)}", code=code, generation=gen+1))
            self.population = new_pop

        return self.best

    def _extract_code(self, resp):
        if "```python" in resp:
            s = resp.index("```python") + 9
            e = resp.index("```", s)
            return resp[s:e].strip()
        if "```" in resp:
            s = resp.index("```") + 3
            e = resp.index("```", s) if "```" in resp[s+1:] else len(resp)
            return resp[s:e].strip()
        if "def evolved_rule" in resp:
            return resp[resp.index("def evolved_rule"):].strip()
        return resp.strip()

    def get_llm_logs(self) -> list[dict]:
        return [{"timestamp": l.timestamp, "role": l.role, "action": l.action,
                 "prompt": l.prompt, "response": l.response,
                 "generation": l.generation, "duration_s": l.duration_s}
                for l in self.llm.call_logs]
