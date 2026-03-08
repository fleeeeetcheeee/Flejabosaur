"""
Build a NetworkX DiGraph from retrosynthetic candidates.
Nodes = molecules (SMILES), edges = reactions (directed: precursor → product).

FIX: Uses a per-reactant candidate tree (dict mapping molecule → its candidates)
instead of a flat list-of-lists, which previously created a Cartesian product of
candidates across unrelated reactants.
"""
from __future__ import annotations

import networkx as nx

from chem.retrosynthesis import RetroCandidate, get_retro_candidates


def build_retro_tree(
    target_smiles: str,
    max_depth: int = 3,
    max_candidates_per_molecule: int = 2,
) -> dict[str, list[RetroCandidate]]:
    """
    Build a per-molecule retrosynthetic candidate tree.

    Returns {smiles: [candidates]} where each molecule maps to its own
    retrosynthetic decomposition candidates. This avoids the Cartesian
    product bug where candidates for molecule A were also applied to
    unrelated molecule B at the same depth.
    """
    tree: dict[str, list[RetroCandidate]] = {}
    frontier = [target_smiles]

    for _ in range(max_depth):
        next_frontier: list[str] = []
        for smi in frontier:
            if smi in tree:
                continue  # already decomposed
            cands = get_retro_candidates(smi, max_candidates=max_candidates_per_molecule)
            tree[smi] = cands
            for c in cands:
                next_frontier.extend(c.reactant_smiles)
        frontier = list(dict.fromkeys(next_frontier))  # dedupe, preserve order
        if not frontier:
            break

    return tree


def build_synthesis_dag(
    target_smiles: str,
    retro_tree: dict[str, list[RetroCandidate]],
) -> nx.DiGraph:
    """
    Build a synthesis DAG from a per-molecule retrosynthetic tree.

    Nodes = molecules (SMILES)
    Edges = reactions (directed: precursor → product)
    Node attrs: smiles, is_target, is_starting_material
    Edge attrs: reaction_name, conditions, template_smarts, source
    """
    G = nx.DiGraph()

    # Determine which molecules are starting materials (not further decomposed)
    all_reactants: set[str] = set()
    for cands in retro_tree.values():
        for c in cands:
            all_reactants.update(c.reactant_smiles)
    starting_materials = all_reactants - set(retro_tree.keys())

    # Add all nodes
    for smi in set(retro_tree.keys()) | all_reactants:
        G.add_node(
            smi,
            smiles=smi,
            is_target=(smi == target_smiles),
            is_starting_material=(smi in starting_materials),
        )

    # Add edges: for each molecule with candidates, add edges from its precursors
    for product_smi, cands in retro_tree.items():
        for cand in cands:
            for reactant in cand.reactant_smiles:
                G.add_edge(
                    reactant,
                    product_smi,
                    reaction_name=cand.reaction_name,
                    conditions=cand.conditions,
                    template_smarts=cand.template_smarts,
                    source=cand.source,
                )

    return G


def build_synthesis_dag_from_flat(
    target_smiles: str,
    candidates_per_step: list[list[RetroCandidate]],
) -> nx.DiGraph:
    """
    Legacy interface: build DAG from flat candidates_per_step list.
    Used by the /synthesize endpoint for single-step analysis.
    """
    G = nx.DiGraph()
    G.add_node(target_smiles, smiles=target_smiles, is_target=True, is_starting_material=False)

    if candidates_per_step:
        for cand in candidates_per_step[0]:
            for reactant in cand.reactant_smiles:
                if not G.has_node(reactant):
                    G.add_node(reactant, smiles=reactant, is_target=False, is_starting_material=True)
                G.add_edge(
                    reactant,
                    target_smiles,
                    reaction_name=cand.reaction_name,
                    conditions=cand.conditions,
                    template_smarts=cand.template_smarts,
                    source=cand.source,
                )

    return G


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
            "position": {"x": 0, "y": idx * 120},
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
                "mu": attrs.get("mu", None),
            },
        })

    return {"nodes": nodes, "edges": edges}


def topological_order(G: nx.DiGraph) -> list[str]:
    """Return molecules in topological synthesis order (starting materials first)."""
    try:
        return list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return list(G.nodes())
