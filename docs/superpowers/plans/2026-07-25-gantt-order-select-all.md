# 甘特图订单全选状态修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用户主动全选甘特图订单并点击组件外部后，甘特图保持全部订单，不回退为第一个订单。

**Architecture:** 保留大实例首次进入时“空状态默认首单”的性能保护，只修改通用多选组件的提交表示。组件关闭时提交完整工作集，使主动全选与尚未交互的空状态可被调用方区分；现有甘特多选和机器分组逻辑继续消费字符串 ID 数组。

**Tech Stack:** 原生 JavaScript、Python `unittest` 前端契约测试、pytest

## Global Constraints

- 大实例首次进入甘特图时仍默认只选择第一个订单。
- 用户主动全选后必须提交所有订单 ID，并在点击组件外部后保持全部订单。
- 不调整数据量阈值、分页策略或服务端单订单选择器。
- 不改动与本问题无关的 `optimization/baseline_build.py` 工作区修改。

---

### Task 1: 保留多选组件的显式全选状态

**Files:**
- Modify: `tests/test_review_frontend_contract.py`
- Modify: `frontend/app_v2.js:576-718`

**Interfaces:**
- Consumes: `working: Set<string>`，由 `mountMultiSelectFilters()` 维护当前勾选集合。
- Produces: `source.onChange(ids: string[])`，其中全选提交全部选项 ID，只有实际空工作集提交空数组。

- [ ] **Step 1: 写入失败的前端契约测试**

在 `ReviewFrontendContractTests` 中加入：

```python
def test_multi_select_outside_click_commits_explicit_full_selection(self):
    start = JS.index("function mountMultiSelectFilters()")
    end = JS.index("\n// —— 通用「可搜索 + 单选」", start)
    source = JS[start:end]

    self.assertIn("const ids = Array.from(working);", source)
    self.assertNotIn("working.size >= total ? [] : Array.from(working)", source)
    self.assertIn('if (!container.contains(event.target)) close();', source)

    timeline_start = JS.index("function renderTimeline(")
    timeline_end = JS.index("\nfunction ", timeline_start)
    timeline_source = JS[timeline_start:timeline_end]
    self.assertIn(
        "if (!localSelectedIds.length && !allowAll && orderOptions.length) localSelectedIds = [orderOptions[0]];",
        timeline_source,
    )
```

该测试同时锁定两条行为：外部点击仍会关闭并提交；主动全选不再被压缩为空状态，同时大实例初始首单保护仍存在。

- [ ] **Step 2: 运行测试并确认它因旧提交逻辑失败**

Run:

```bash
.venv/bin/python -m pytest tests/test_review_frontend_contract.py::ReviewFrontendContractTests::test_multi_select_outside_click_commits_explicit_full_selection -q
```

Expected: FAIL，错误指出 `const ids = Array.from(working);` 不存在；当前源码仍包含 `working.size >= total ? [] : Array.from(working)`。

- [ ] **Step 3: 实施最小修复并校正组件注释**

将 `mountMultiSelectFilters()` 的 `commit` 改为：

```javascript
const commit = () => {
  // 提交实际工作集：显式全选保留全部 ID，避免与“尚未选择”的空数组状态混淆。
  const ids = Array.from(working);
  source.onChange(ids);
};
```

同时把组件头部的语义注释从“选中 0 个 = 全部”改为：

```javascript
// 语义：初始 selectedIds 为空时按全部渲染；用户显式全选时提交全部 ID；输入框按 label 模糊过滤。
```

不要修改 `renderTimeline()` 中的大实例首单兜底，也不要修改订单筛选的 `onChange`。

- [ ] **Step 4: 运行定向测试并确认通过**

Run:

```bash
.venv/bin/python -m pytest tests/test_review_frontend_contract.py::ReviewFrontendContractTests::test_multi_select_outside_click_commits_explicit_full_selection -q
```

Expected: `1 passed`。

- [ ] **Step 5: 运行前端相关回归测试**

Run:

```bash
.venv/bin/python -m pytest tests/test_review_frontend_contract.py tests/test_review_frontend_runtime.py tests/test_verify_review_ui.py -q
```

Expected: 全部通过，无失败或错误。

- [ ] **Step 6: 检查差异并提交修复**

Run:

```bash
git diff --check
git diff -- frontend/app_v2.js tests/test_review_frontend_contract.py
```

Expected: `git diff --check` 无输出；差异只包含回归测试、提交逻辑和对应注释。

Commit:

```bash
git add frontend/app_v2.js tests/test_review_frontend_contract.py
git commit -m "fix: 保留甘特图订单全选状态"
```
