"""
Critical Path Method (CPM) for synthesis DAG.

Each reaction edge has:
  duration_h: estimated reaction time in hours (default 2.0)
  cost:       relative reagent cost (default 1.0)

Forward pass: ES, EF
Backward pass: LS, LF
Float = LF - EF
Critical path = edges with float == 0
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


DEFAULT_DURATION = 2.0   # hours
DEFAULT_COST = 1.0


@dataclass
class EdgeCPM:
    src: str
    dst: str
    reaction_name: str
    duration_h: float
    cost: float
    es: float = 0.0   # Earliest Start
    ef: float = 0.0   # Earliest Finish
    ls: float = 0.0   # Latest Start
    lf: float = 0.0   # Latest Finish
    float_h: float = 0.0
    is_critical: bool = False


def run_cpm(G: nx.DiGraph) -> tuple[list[EdgeCPM], list[str]]:
    """
    Run CPM on synthesis DAG G.
    Returns (edge_data, critical_path_node_sequence).
    """
    topo = list(nx.topological_sort(G))

    # Assign durations from edge attributes (or defaults)
    edge_data: dict[tuple, EdgeCPM] = {}
    for src, dst, attrs in G.edges(data=True):
        conds = attrs.get("conditions", {})
        duration = float(conds.get("duration_h", DEFAULT_DURATION))
        cost = float(conds.get("cost", DEFAULT_COST))
        edge_data[(src, dst)] = EdgeCPM(
            src=src, dst=dst,
            reaction_name=attrs.get("reaction_name", ""),
            duration_h=duration, cost=cost,
        )

    # Node earliest finish times
    ef_node: dict[str, float] = {n: 0.0 for n in G.nodes()}

    # Forward pass
    for node in topo:
        # ES for all outgoing edges = max EF of predecessors
        pred_ef = max((ef_node[p] for p in G.predecessors(node)), default=0.0)
        ef_node[node] = pred_ef
        for dst in G.successors(node):
            e = edge_data[(node, dst)]
            e.es = pred_ef
            e.ef = pred_ef + e.duration_h
            ef_node[dst] = max(ef_node[dst], e.ef)

    project_duration = max(ef_node.values(), default=0.0)

    # Backward pass
    lf_node: dict[str, float] = {n: project_duration for n in G.nodes()}
    for node in reversed(topo):
        succ_ls = min((lf_node[s] for s in G.successors(node)), default=project_duration)
        lf_node[node] = succ_ls
        for src in G.predecessors(node):
            e = edge_data[(src, node)]
            e.lf = succ_ls
            e.ls = succ_ls - e.duration_h
            lf_node[src] = min(lf_node[src], e.ls)

    # Float and critical flag
    for e in edge_data.values():
        e.float_h = round(e.lf - e.ef, 6)
        e.is_critical = abs(e.float_h) < 1e-6

    # Critical path node sequence
    critical_edges = [e for e in edge_data.values() if e.is_critical]
    if critical_edges:
        critical_nodes: list[str] = []
        for e in critical_edges:
            if e.src not in critical_nodes:
                critical_nodes.append(e.src)
            if e.dst not in critical_nodes:
                critical_nodes.append(e.dst)
    else:
        critical_nodes = list(topo)

    return list(edge_data.values()), critical_nodes
