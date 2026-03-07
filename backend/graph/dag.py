"""
Build a NetworkX DiGraph from retrosynthetic candidates.
Nodes = molecules (SMILES), edges = reactions (directed: precursor → product).
"""
from __future__ import annotations

import networkx as nx

from ..chem.retrosynthesis import RetroCandidate


def build_synthesis_dag(
    target_smiles: str,
    candidates_per_step: list[list[RetroCandidate]],
) -> nx.DiGraph:
    """
    Build a synthesis DAG from a list of retrosynthetic steps.
    candidates_per_step[0] = candidates for the final step (target)
    candidates_per_step[i] = candidates for intermediate at depth i

    Returns a directed graph where edges point from precursor → product.
    Node attributes: smiles, is_target, is_starting_material
    Edge attributes: reaction_name, conditions, template_smarts, source
    """
    G = nx.DiGraph()
    G.add_node(target_smiles, smiles=target_smiles, is_target=True, is_starting_material=False)

    _add_candidates(G, target_smiles, candidates_per_step, depth=0)
    return G


def _add_candidates(
    G: nx.DiGraph,
    product_smiles: str,
    candidates_per_step: list[list[RetroCandidate]],
    depth: int,
) -> None:
    if depth >= len(candidates_per_step):
        return

    for cand in candidates_per_step[depth]:
        for reactant in cand.reactant_smiles:
            if not G.has_node(reactant):
                is_leaf = (depth + 1 >= len(candidates_per_step))
                G.add_node(
                    reactant,
                    smiles=reactant,
                    is_target=False,
                    is_starting_material=is_leaf,
                )
            G.add_edge(
                reactant,
                product_smiles,
                reaction_name=cand.reaction_name,
                conditions=cand.conditions,
                template_smarts=cand.template_smarts,
                source=cand.source,
            )

        # Recurse for each precursor (next retrosynthetic layer)
        if depth + 1 < len(candidates_per_step):
            for reactant in cand.reactant_smiles:
                _add_candidates(G, reactant, candidates_per_step, depth + 1)


def dag_to_dict(G: nx.DiGraph) -> dict:
    """Serialize DAG to React Flow compatible node/edge lists."""
    nodes = []
    for idx, (node, attrs) in enumerate(G.nodes(data=True)):
        nodes.append({
            "id": node,
            "type": "molecule",
            "data": {
                "smiles": attrs.get("smiles", node),
                "isTarget": attrs.get("is_target", False),
                "isStartingMaterial": attrs.get("is_starting_material", False),
            },
            "position": {"x": 0, "y": idx * 120},  # positions set by frontend layout
        })

    edges = []
    for i, (src, dst, attrs) in enumerate(G.edges(data=True)):
        edges.append({
            "id": f"e{i}",
            "source": src,
            "target": dst,
            "label": attrs.get("reaction_name", ""),
            "data": {
                "conditions": attrs.get("conditions", {}),
                "templateSmarts": attrs.get("template_smarts", ""),
                "source": attrs.get("source", ""),
            },
        })

    return {"nodes": nodes, "edges": edges}


def topological_order(G: nx.DiGraph) -> list[str]:
    """Return molecules in topological synthesis order (starting materials first)."""
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return list(G.nodes())
