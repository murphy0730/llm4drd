"""what-if patch 领域逻辑的单测：纯函数，不碰数据库。"""
import copy
import unittest

from llm4drd.api.whatif_scenario import (
    PatchError,
    ScenarioStore,
    apply_patches,
    rows_scale,
)


def base_rows() -> dict:
    return {
        "planning_context": {"id": 1, "plan_start_at": "2026-07-19T08:00:00+08:00"},
        "orders": [
            {"order_id": "O1", "order_name": "订单一", "release_time": 0.0, "due_date": 100.0, "priority": 1},
        ],
        "tasks": [
            {"task_id": "T1", "order_id": "O1", "task_name": "总装", "is_main": 1,
             "predecessor_task_ids": "", "release_time": 0.0, "due_date": 100.0},
        ],
        "operations": [
            {"op_id": "OP1", "task_id": "T1", "op_name": "车", "process_type": "turning",
             "processing_time": 5.0, "turnover_time": 0.0, "predecessor_ops": "",
             "predecessor_tasks": "", "eligible_machine_ids": "M1;M2",
             "required_tooling_types": "", "required_personnel_skills": "",
             "initial_status": "", "initial_start_time": None, "initial_end_time": None,
             "initial_remaining_processing_time": None, "initial_assigned_machine_id": "",
             "initial_assigned_tooling_ids": "", "initial_assigned_personnel_ids": ""},
        ],
        "machine_types": [
            {"type_id": "turning", "type_name": "车床", "is_critical": 1},
            {"type_id": "milling", "type_name": "铣床", "is_critical": 0},
        ],
        "machines": [
            {"machine_id": "M1", "machine_name": "车床1", "type_id": "turning", "shifts": "0/8/10"},
            {"machine_id": "M2", "machine_name": "车床2", "type_id": "turning", "shifts": "0/8/10"},
            {"machine_id": "M3", "machine_name": "铣床1", "type_id": "milling", "shifts": "0/8/10;0/20/8"},
        ],
        "tooling_types": [],
        "toolings": [],
        "personnel": [],
        "downtimes": [],
    }


class ApplyPatchesTests(unittest.TestCase):
    def test_add_machine(self):
        rows = base_rows()
        patched, effects = apply_patches(rows, [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M4", "machine_name": "车床3", "type_id": "turning", "shifts": "0/8/10"}},
        ])
        self.assertEqual(rows_scale(patched)["machines"], 4)
        self.assertEqual(effects[0]["matched"], 1)
        self.assertIn("M4", effects[0]["message"])

    def test_does_not_mutate_input(self):
        rows = base_rows()
        snapshot = copy.deepcopy(rows)
        apply_patches(rows, [{"op": "remove", "entity": "machine", "id": "M1"}])
        self.assertEqual(rows, snapshot)

    def test_remove_machine(self):
        patched, effects = apply_patches(base_rows(), [
            {"op": "remove", "entity": "machine", "id": "M1"},
        ])
        self.assertEqual([m["machine_id"] for m in patched["machines"]], ["M2", "M3"])
        self.assertEqual(effects[0]["matched"], 1)

    def test_update_by_where_is_batch(self):
        patched, effects = apply_patches(base_rows(), [
            {"op": "update", "entity": "machine", "where": {"type_id": "turning"},
             "values": {"shifts": "0/8/10"}},
        ])
        self.assertEqual(effects[0]["matched"], 2)
        self.assertTrue(all(
            m["shifts"] == "0/8/10" for m in patched["machines"] if m["type_id"] == "turning"
        ))

    def test_shifts_accepts_structured_array(self):
        patched, _ = apply_patches(base_rows(), [
            {"op": "update", "entity": "machine", "id": "M1",
             "values": {"shifts": [{"day": 0, "start_hour": 8, "hours": 10}]}},
        ])
        self.assertEqual(patched["machines"][0]["shifts"], "0/8.0/10.0")

    def test_update_operation_processing_time(self):
        patched, _ = apply_patches(base_rows(), [
            {"op": "update", "entity": "operation", "id": "OP1", "values": {"processing_time": 4.5}},
        ])
        self.assertEqual(patched["operations"][0]["processing_time"], 4.5)

    def test_add_downtime_autoincrements_id(self):
        patched, _ = apply_patches(base_rows(), [
            {"op": "add", "entity": "downtime", "values": {
                "machine_id": "M1", "downtime_type": "maintenance", "start_time": 48, "end_time": 56}},
            {"op": "add", "entity": "downtime", "values": {
                "machine_id": "M2", "downtime_type": "maintenance", "start_time": 10, "end_time": 12}},
        ])
        self.assertEqual([d["id"] for d in patched["downtimes"]], [1, 2])

    def test_planning_context_update(self):
        patched, _ = apply_patches(base_rows(), [
            {"op": "update", "entity": "planning_context",
             "values": {"plan_start_at": "2026-08-01T08:00:00+08:00"}},
        ])
        self.assertEqual(patched["planning_context"]["plan_start_at"], "2026-08-01T08:00:00+08:00")

    def test_patches_apply_in_order(self):
        patched, _ = apply_patches(base_rows(), [
            {"op": "add", "entity": "machine", "values": {
                "machine_id": "M4", "machine_name": "车床3", "type_id": "turning", "shifts": "0/8/10"}},
            {"op": "remove", "entity": "machine", "id": "M4"},
        ])
        self.assertEqual(rows_scale(patched)["machines"], 3)

    def test_id_list_normalizes_separators(self):
        patched, _ = apply_patches(base_rows(), [
            {"op": "update", "entity": "operation", "id": "OP1",
             "values": {"eligible_machine_ids": "M1，M2, M3"}},
        ])
        self.assertEqual(patched["operations"][0]["eligible_machine_ids"], "M1;M2;M3")


class PatchValidationTests(unittest.TestCase):
    def test_unknown_entity(self):
        with self.assertRaises(PatchError) as ctx:
            apply_patches(base_rows(), [{"op": "add", "entity": "robot", "values": {"id": "R1"}}])
        self.assertEqual(ctx.exception.code, "UNKNOWN_ENTITY")
        self.assertTrue(ctx.exception.suggestions)

    def test_unknown_field(self):
        with self.assertRaises(PatchError) as ctx:
            apply_patches(base_rows(), [
                {"op": "update", "entity": "machine", "id": "M1", "values": {"speed": 3}},
            ])
        self.assertEqual(ctx.exception.code, "UNKNOWN_FIELD")

    def test_duplicate_id_rejected(self):
        with self.assertRaises(PatchError) as ctx:
            apply_patches(base_rows(), [
                {"op": "add", "entity": "machine", "values": {
                    "machine_id": "M1", "machine_name": "重复", "type_id": "turning", "shifts": ""}},
            ])
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")

    def test_target_not_found(self):
        with self.assertRaises(PatchError) as ctx:
            apply_patches(base_rows(), [{"op": "remove", "entity": "machine", "id": "M99"}])
        self.assertEqual(ctx.exception.code, "TARGET_NOT_FOUND")

    def test_add_requires_primary_key(self):
        with self.assertRaises(PatchError) as ctx:
            apply_patches(base_rows(), [
                {"op": "add", "entity": "machine", "values": {"machine_name": "无号机"}},
            ])
        self.assertEqual(ctx.exception.code, "INVALID_PATCH")

    def test_update_without_selector_rejected(self):
        with self.assertRaises(PatchError):
            apply_patches(base_rows(), [
                {"op": "update", "entity": "machine", "values": {"shifts": "0/8/10"}},
            ])

    def test_multiple_selectors_rejected(self):
        with self.assertRaises(PatchError):
            apply_patches(base_rows(), [
                {"op": "update", "entity": "machine", "id": "M1", "where": {"type_id": "turning"},
                 "values": {"shifts": "0/8/10"}},
            ])

    def test_batch_update_cannot_change_primary_key(self):
        with self.assertRaises(PatchError):
            apply_patches(base_rows(), [
                {"op": "update", "entity": "machine", "where": {"type_id": "turning"},
                 "values": {"machine_id": "MX"}},
            ])

    def test_invalid_field_value(self):
        with self.assertRaises(PatchError) as ctx:
            apply_patches(base_rows(), [
                {"op": "update", "entity": "operation", "id": "OP1",
                 "values": {"processing_time": "很久"}},
            ])
        self.assertEqual(ctx.exception.code, "INVALID_FIELD_VALUE")


class ScenarioStoreTests(unittest.TestCase):
    def test_lru_evicts_oldest(self):
        store = ScenarioStore(max_size=2)
        first = store.create("A", 1)
        store.create("B", 1)
        store.create("C", 1)
        self.assertIsNone(store.get(first.scenario_id))
        self.assertEqual(len(store.list()), 2)

    def test_digest_changes_with_patches(self):
        store = ScenarioStore()
        scenario = store.create("A", 1)
        before = scenario.digest()
        store.set_patches(scenario.scenario_id, [{"op": "remove", "entity": "machine", "id": "M1"}])
        self.assertNotEqual(before, store.get(scenario.scenario_id).digest())

    def test_delete(self):
        store = ScenarioStore()
        scenario = store.create("A", 1)
        self.assertTrue(store.delete(scenario.scenario_id))
        self.assertFalse(store.delete(scenario.scenario_id))


if __name__ == "__main__":
    unittest.main()
