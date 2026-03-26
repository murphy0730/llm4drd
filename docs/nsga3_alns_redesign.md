# NSGA-III / ALNS Hybrid Optimization Redesign

## Scope

This redesign covers the modules after:

- problem configuration
- heterogeneous graph
- LLM configuration

The goal is to replace the current rule-comparison oriented flow with a
high-quality multi-objective optimization flow that can:

- use the generated heterogeneous graph as a first-class optimization context
- let business users choose `1-5` optimization objectives
- generate a user-specified number of Pareto-optimal solutions
- support accurate solution comparison and later LLM-assisted scheme selection


## Product Modules

The current pages around compare / pareto / exact / online / scenario should be
reorganized into the following product modules.

### 1. Baseline Simulation

Purpose:

- run a deterministic baseline schedule
- validate that the instance, graph, calendar, downtime, tooling and personnel
  constraints are all feasible before optimization

Key outputs:

- baseline KPI snapshot
- baseline schedule
- infeasibility or bottleneck diagnostics

Why it stays:

- optimization accuracy depends on a trusted baseline
- this is the reference point for later Pareto comparison


### 2. Optimization Setup

Purpose:

- choose `1-5` objectives
- choose the desired Pareto solution count
- configure search budget and solution quality mode

Core inputs:

- selected objectives
- target solution count
- search time budget
- random seed / reproducibility mode
- optional hard preferences:
  - prioritize main orders
  - preserve schedule stability
  - penalize bottleneck overload

This module should replace the current separate "Pareto / NSGA-II / Exact"
setup flow.


### 3. Hybrid Solve Monitor

Purpose:

- start, monitor and stop a long-running `NSGA-III + ALNS` task
- display progress, incumbent archive size, feasible ratio and convergence

Core outputs:

- current generation
- archive size
- feasible solution ratio
- best-known hypervolume
- average evaluation time
- dominant bottleneck resources


### 4. Pareto Solution Library

Purpose:

- display the generated Pareto-optimal solutions
- store them as a reusable solution set for business review

Each solution card should contain:

- solution id
- rank
- objective vector
- delta vs baseline
- feasibility flag
- schedule summary
- resource utilization summary
- bottleneck summary


### 5. Solution Comparison

Purpose:

- compare `2-4` selected Pareto solutions side by side
- explain where each solution wins and loses

Comparison views:

- KPI matrix
- Gantt overlay / schedule difference
- order-level difference
- bottleneck machine / tooling / personnel difference
- main-order completion path difference


### 6. Business Selection and Explanation

Purpose:

- let users pick candidate solutions from the Pareto library
- use the LLM later only for explanation, tradeoff analysis and business-facing
  recommendation

Important principle:

- the optimizer generates solutions
- the LLM does not generate schedules directly
- the LLM explains and compares already computed schedules


## Objective System

Accuracy is more important than quantity. We should only expose objectives that
the current data model can calculate correctly.

### Phase 1 objectives: accurate with current model

These are safe to expose immediately for user selection:

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

### Phase 2 objectives: expose only after more constraints are modeled

These should be deferred until the underlying data and simulator are extended:

- sequence-dependent setup cost
- changeover count / time
- material shortage penalty
- WIP peak
- transport time
- frozen-window violation
- schedule stability / reschedule cost
- energy cost / peak power
- overtime cost

### Objective selection rules

- allow only `1-5` objectives
- do not allow duplicate objectives
- validate objective compatibility before solve
- show whether each objective is `min` or `max`
- normalize all objective values before NSGA-III selection


## Hybrid NSGA-III / ALNS Architecture

The optimizer should be hybrid instead of monolithic.

### Outer layer: NSGA-III

Responsibilities:

- maintain population diversity for `1-5` objectives
- manage reference points
- select survivors
- manage the Pareto archive

Why NSGA-III:

- better than NSGA-II when the objective count grows beyond `2-3`
- suited for user-selected multi-objective combinations

### Inner layer: ALNS

Responsibilities:

- improve candidate schedules locally
- use graph-aware destroy and repair neighborhoods
- intensify around bottlenecks and tardy chains

Why ALNS:

- much better fit than pure evolutionary search for schedule refinement
- easier to inject industrial heuristics and graph knowledge

### Candidate representation

Each candidate should not be just a dispatch rule weight vector.

Recommended encoding:

- operation priority policy parameters
- neighborhood weight vector
- repair policy parameters
- optional rolling-window parameters
- optional stability penalty weight

This gives ALNS enough freedom to change actual schedules instead of only
switching among existing rules.


## How the Heterogeneous Graph Should Be Used

The graph is not only for visualization. It should drive the neighborhoods and
evaluation logic.

### Graph-aware destroy operators

- tardy order subgraph destroy
- main assembly chain destroy
- bottleneck machine neighborhood destroy
- shared tooling cluster destroy
- shared personnel cluster destroy
- critical predecessor chain destroy

### Graph-aware repair operators

- earliest-feasible insertion
- due-date biased insertion
- main-order first insertion
- bottleneck smoothing insertion
- assembly synchronization insertion
- shared-resource conflict repair

### Graph-derived features

Useful candidate features:

- predecessor depth
- assembly criticality
- shared-resource degree
- bottleneck adjacency
- tardiness contribution estimate
- order priority propagation


## Accuracy Strategy

To ensure solution quality is not "fake Pareto", the compute flow should be:

1. Construct a feasible schedule
2. Repair infeasibilities immediately
3. Evaluate with the authoritative simulator
4. Store only feasible solutions in the Pareto archive
5. Keep infeasible candidates only as temporary search states

### Required evaluation contract

Every candidate solution must be evaluated by the same scheduling kernel:

- machine calendars
- planned / unplanned downtime
- order / task release times
- tooling constraints
- personnel constraints
- assembly precedence

### Recommended quality safeguards

- deterministic seed mode
- duplicate solution detection by schedule signature
- archive deduplication by objective vector plus assignment signature
- hard feasibility check before archive insertion
- reference-solution regression tests


## Backend Module Redesign

### New modules

- `optimization/objectives.py`
  - objective registry
  - metadata
  - normalization
  - validation

- `optimization/hybrid_nsga3_alns.py`
  - hybrid optimizer entry
  - population management
  - archive management

- `optimization/nsga3_core.py`
  - reference point generation
  - normalization
  - niche-preserving survivor selection

- `optimization/alns_core.py`
  - destroy / repair operator registry
  - adaptive operator scoring
  - local refinement loop

- `optimization/solution_model.py`
  - solution schema
  - objective vector
  - schedule signature
  - comparison helpers

- `optimization/archive.py`
  - Pareto archive
  - deduplication
  - export / summary

### Existing modules to demote or retire

- current `optimization/pareto.py`
  - keep only for transitional compatibility
  - do not treat built-in rule comparison as the future optimization core

- current NSGA-II page / logic
  - retire after new hybrid pipeline is stable

- current exact solver page
  - keep as small-instance validation or local repair kernel
  - not as the main optimization product


## API Redesign

### Objective catalog

`GET /api/optimize/objectives`

Returns:

- objective key
- label
- direction
- description
- available now / later

### Start optimization

`POST /api/optimize/hybrid`

Request:

- `objective_keys: list[str]` with length `1-5`
- `target_solution_count: int`
- `time_limit_s: int`
- `population_size: int`
- `generations: int`
- `alns_iterations_per_candidate: int`
- `seed: int | null`

Response:

- `task_id`
- normalized config

### Task status

`GET /api/optimize/hybrid/status/{task_id}`

Returns:

- progress
- current generation
- feasible solutions
- archive size
- hypervolume trend
- best solutions preview

### Solution library

`GET /api/optimize/hybrid/result/{task_id}`

Returns:

- selected objectives
- baseline metrics
- Pareto solutions
- archive summary


## Frontend Redesign

The current separate pages should be consolidated.

### Keep

- baseline simulation

### Replace with new pages

- `Optimization Setup`
- `Solve Monitor`
- `Pareto Library`
- `Solution Compare`
- `Business Decision`

### Remove as top-level product concepts

- old rule-only compare
- old two-objective Pareto scatter
- old standalone NSGA-II
- old exact solver as main page
- old scenario analysis as a separate first-class flow

These can survive as internal tools or advanced tabs, but not as the main
business workflow.


## Implementation Order

### Step 1

- objective registry
- objective validation
- new optimization request schema
- new solution schema

### Step 2

- NSGA-III core
- Pareto archive
- baseline + archive comparison API

### Step 3

- ALNS destroy / repair operators using graph neighborhoods
- hybrid outer-inner loop

### Step 4

- frontend optimization setup
- monitor
- Pareto library

### Step 5

- solution compare
- LLM-assisted explanation and recommendation


## Non-Negotiable Engineering Requirements

- the optimizer must be deterministic under a fixed seed
- every archived solution must be feasible
- objective calculation must come from the authoritative simulator, not from
  shortcut estimates
- the user must be able to request an exact number of solutions, but the system
  should clearly report if the search only found fewer unique feasible
  non-dominated solutions
- all objective names, directions and formulas must be centrally registered

