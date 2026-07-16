"""
异构图建模引擎
=============
将车间调度问题转化为基于有向异构图的可行边排序问题。

节点类型: 订单、任务、工序、机器
边类型: 订单→任务、任务前置、任务→工序、工序顺序、前置任务约束、机器兼容
"""
import networkx as nx
from ..core.models import ShopFloor
from .canonical import CanonicalGraphBuilder


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

    def build_from_shopfloor(self, shop: ShopFloor, progress_callback=None, deadline: float | None = None):
        """从车间配置构建异构图"""
        canonical = CanonicalGraphBuilder().build(shop, progress_callback, deadline)
        self.graph.clear()
        for node in canonical.nodes:
            self.graph.add_node(
                node.node_id,
                node_type=node.node_type,
                entity_id=node.entity_id,
                **dict(node.attrs),
            )
        for edge in canonical.edges:
            self.graph.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type,
                **dict(edge.attrs),
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
