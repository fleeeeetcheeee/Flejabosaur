"""
Critical Path Method (CPM) for synthesis DAG.

Each reaction edge has:
  duration_h: estimated reaction time in hours (default 2.0)
  cost:       relative reagent cost (default 1.0)

Forward pass:  ES_edge = max(EF of all incoming edges to src node)
               EF_edge = ES_edge + duration
Backward pass: LF_edge = min(LS of all outgoing edges from dst node)
               LS_edge = LF_edge - duration
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
    if not G.edges():
        return [], list(G.nodes())

    topo = list(nx.topological_sort(G))

    # Build edge data from graph attributes
    edge_data: dict[tuple, EdgeCPM] = {}
    for src, dst, attrs in G.edges(data=True):
        conds = attrs.get("conditions", {})
        if not isinstance(conds, dict):
            conds = {}
        duration = float(conds.get("duration_h", DEFAULT_DURATION))
        cost = float(conds.get("cost", DEFAULT_COST))
        edge_data[(src, dst)] = EdgeCPM(
            src=src, dst=dst,
            reaction_name=attrs.get("reaction_name", ""),
            duration_h=duration, cost=cost,
        )

    # -----------------------------------------------------------------------
    # Forward pass: compute Earliest Start / Earliest Finish for each edge
    # A node's "earliest availability" = max EF of all incoming edges
    # -----------------------------------------------------------------------
    node_earliest: dict[str, float] = {n: 0.0 for n in G.nodes()}

    for node in topo:
        # This node's earliest availability = max EF of incoming edges
        incoming_efs = [edge_data[(p, node)].ef
                        for p in G.predecessors(node)
                        if (p, node) in edge_data]
        node_earliest[node] = max(incoming_efs, default=0.0)

        # Set ES/EF for all outgoing edges
        for dst in G.successors(node):
            if (node, dst) in edge_data:
                e = edge_data[(node, dst)]
                e.es = node_earliest[node]
                e.ef = e.es + e.duration_h

    project_duration = max((e.ef for e in edge_data.values()), default=0.0)

    # -----------------------------------------------------------------------
    # Backward pass: compute Latest Start / Latest Finish for each edge
    # A node's "latest need" = min LS of all outgoing edges
    # -----------------------------------------------------------------------
    node_latest: dict[str, float] = {n: project_duration for n in G.nodes()}

    for node in reversed(topo):
        # This node's latest need = min LS of outgoing edges
        outgoing_lss = [edge_data[(node, s)].ls
                        for s in G.successors(node)
                        if (node, s) in edge_data]
        node_latest[node] = min(outgoing_lss, default=project_duration)

        # Set LF/LS for all incoming edges
        for src in G.predecessors(node):
            if (src, node) in edge_data:
                e = edge_data[(src, node)]
                e.lf = node_latest[node]
                e.ls = e.lf - e.duration_h

    # Float and critical flag
    for e in edge_data.values():
        e.float_h = round(e.lf - e.ef, 6)
        e.is_critical = abs(e.float_h) < 1e-6

    # Critical path node sequence (preserving topological order)
    critical_edges = [e for e in edge_data.values() if e.is_critical]
    if critical_edges:
        critical_node_set: set[str] = set()
        for e in critical_edges:
            critical_node_set.add(e.src)
            critical_node_set.add(e.dst)
        # Preserve topological order
        critical_nodes = [n for n in topo if n in critical_node_set]
    else:
        critical_nodes = list(topo)

    return list(edge_data.values()), critical_nodes
