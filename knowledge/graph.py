"""
异构图建模引擎
=============
将车间调度问题转化为基于有向异构图的可行边排序问题。

节点类型: 订单、任务、工序、机器
边类型: 订单→任务、任务前置、任务→工序、工序顺序、前置任务约束、机器兼容
"""
import networkx as nx
from typing import Optional
from ..core.models import ShopFloor


class HeterogeneousGraph:
    """有向异构图 - 统一表示所有调度约束"""

    # 节点类型
    NODE_ORDER = "order"
    NODE_TASK = "task"
    NODE_OPERATION = "operation"
    NODE_MACHINE = "machine"
    NODE_TOOLING = "tooling"
    NODE_PERSONNEL = "personnel"

    # 边类型
    EDGE_ORDER_TASK = "order_has_task"
    EDGE_TASK_PREDECESSOR = "task_predecessor"
    EDGE_TASK_OPERATION = "task_has_operation"
    EDGE_OP_SEQUENCE = "operation_sequence"
    EDGE_OP_PRED_TASK = "op_depends_task"
    EDGE_MACHINE_ELIGIBLE = "machine_eligible"
    EDGE_TOOLING_ELIGIBLE = "tooling_eligible"
    EDGE_PERSONNEL_ELIGIBLE = "personnel_eligible"

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_shopfloor(self, shop: ShopFloor):
        """从车间配置构建异构图"""
        self.graph.clear()

        # 添加机器节点
        for mid, machine in shop.machines.items():
            mt = shop.machine_types.get(machine.type_id)
            self.graph.add_node(
                f"M:{mid}",
                node_type=self.NODE_MACHINE,
                entity_id=mid,
                label=machine.name,
                type_id=machine.type_id,
                type_name=mt.name if mt else "",
                is_critical=mt.is_critical if mt else False,
            )

        for tooling_id, tooling in shop.toolings.items():
            tooling_type = shop.tooling_types.get(tooling.type_id)
            self.graph.add_node(
                f"TL:{tooling_id}",
                node_type=self.NODE_TOOLING,
                entity_id=tooling_id,
                label=tooling.name,
                type_id=tooling.type_id,
                type_name=tooling_type.name if tooling_type else "",
            )

        for person_id, person in shop.personnel.items():
            self.graph.add_node(
                f"P:{person_id}",
                node_type=self.NODE_PERSONNEL,
                entity_id=person_id,
                label=person.name,
                skills=";".join(person.skills),
            )

        # 添加订单节点
        for oid, order in shop.orders.items():
            self.graph.add_node(
                f"O:{oid}",
                node_type=self.NODE_ORDER,
                entity_id=oid,
                label=order.name,
                due_date=order.due_date,
                due_at=shop.time_label(order.due_date),
                priority=order.priority,
                release_time=order.release_time,
                release_at=shop.time_label(order.release_time),
            )

        # 添加任务节点
        for tid, task in shop.tasks.items():
            self.graph.add_node(
                f"T:{tid}",
                node_type=self.NODE_TASK,
                entity_id=tid,
                label=task.name,
                order_id=task.order_id,
                is_main=task.is_main,
                due_date=task.due_date,
                release_time=task.release_time,
                due_at=shop.time_label(task.due_date),
                release_at=shop.time_label(task.release_time),
                derived_due_date=task.derived_due_date,
                derived_due_at=shop.time_label(task.derived_due_date),
                derived_start_time=task.derived_start_time,
                derived_start_at=shop.time_label(task.derived_start_time),
                critical_path_time=task.critical_path_time,
                critical_slack=task.critical_slack,
            )
            # 订单 → 任务
            self.graph.add_edge(
                f"O:{task.order_id}", f"T:{tid}",
                edge_type=self.EDGE_ORDER_TASK,
            )
            # 任务前置约束
            for pred_tid in task.predecessor_task_ids:
                self.graph.add_edge(
                    f"T:{pred_tid}", f"T:{tid}",
                    edge_type=self.EDGE_TASK_PREDECESSOR,
                )

        # 添加工序节点
        for opid, op in shop.operations.items():
            self.graph.add_node(
                f"OP:{opid}",
                node_type=self.NODE_OPERATION,
                entity_id=opid,
                label=op.name,
                task_id=op.task_id,
                process_type=op.process_type,
                processing_time=op.processing_time,
                required_tooling_types=";".join(op.required_tooling_types),
                required_personnel_skills=";".join(op.required_personnel_skills),
                status=op.status.value,
                derived_due_date=op.derived_due_date,
                derived_due_at=shop.time_label(op.derived_due_date),
                derived_start_time=op.derived_start_time,
                derived_start_at=shop.time_label(op.derived_start_time),
                critical_slack=op.critical_slack,
            )
            # 任务 → 工序
            self.graph.add_edge(
                f"T:{op.task_id}", f"OP:{opid}",
                edge_type=self.EDGE_TASK_OPERATION,
            )
            # 工序顺序约束
            for pred_op in op.predecessor_ops:
                self.graph.add_edge(
                    f"OP:{pred_op}", f"OP:{opid}",
                    edge_type=self.EDGE_OP_SEQUENCE,
                )
            # 工序依赖前置任务
            for pred_task in op.predecessor_tasks:
                self.graph.add_edge(
                    f"T:{pred_task}", f"OP:{opid}",
                    edge_type=self.EDGE_OP_PRED_TASK,
                )
            # 机器兼容边 — 如果指定了具体机器则用指定的, 否则按工艺类型
            eligible_mids = op.eligible_machine_ids
            if not eligible_mids:
                eligible_mids = shop._machine_by_type.get(op.process_type, [])
            for mid in eligible_mids:
                self.graph.add_edge(
                    f"OP:{opid}", f"M:{mid}",
                    edge_type=self.EDGE_MACHINE_ELIGIBLE,
                )
            for tooling_type in op.required_tooling_types:
                for tooling in shop.get_toolings_for_type(tooling_type):
                    self.graph.add_edge(
                        f"OP:{opid}", f"TL:{tooling.id}",
                        edge_type=self.EDGE_TOOLING_ELIGIBLE,
                    )
            for skill_id in op.required_personnel_skills:
                for person in shop.get_personnel_for_skill(skill_id):
                    self.graph.add_edge(
                        f"OP:{opid}", f"P:{person.id}",
                        edge_type=self.EDGE_PERSONNEL_ELIGIBLE,
                    )

    def get_graph_stats(self) -> dict:
        """获取图的统计信息"""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }
