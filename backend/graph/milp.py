"""
MILP pathway selection using PuLP.

Given multiple synthesis pathways (sequences of reactions), select the optimal
pathway that maximizes log-yield while penalizing hazard and cost.

Variables:
  x_r ∈ {0,1}  — is reaction r used in the chosen pathway?

Objective:
  maximize  Σ_r x_r · log(μ_r) - λ1 · Σ_r x_r · hazard_r - λ2 · Σ_r x_r · cost_r

Constraints:
  Flow conservation: for each intermediate molecule m,
    Σ_{r produces m} x_r = Σ_{r consumes m} x_r
  Start: Σ_{r producing target} x_r = 1  (exactly one final step)
  x_r ∈ {0, 1}
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import networkx as nx
import pulp

logger = logging.getLogger(__name__)

LAMBDA_HAZARD = 0.2
LAMBDA_COST = 0.1


@dataclass
class PathwayMILP:
    selected_edges: list[tuple[str, str]]   # (src, dst) pairs
    selected_reactions: list[str]           # reaction names
    objective_value: float
    log_yield_sum: float
    hazard_penalty: float
    cost_penalty: float
    status: str


def select_optimal_pathway(G: nx.DiGraph, target_smiles: str) -> PathwayMILP:
    """
    Select the optimal pathway through the synthesis DAG using MILP.
    Returns selected edges and objective breakdown.
    """
    edges = list(G.edges(data=True))
    if not edges:
        return PathwayMILP([], [], 0.0, 0.0, 0.0, 0.0, "no_edges")

    # Build reaction data
    reactions = []
    for i, (src, dst, attrs) in enumerate(edges):
        rname = attrs.get("reaction_name", "unknown")
        mu = attrs.get("mu", 0.65)          # PERT expected yield
        hazard = float(attrs.get("hazard", 0.3))
        cost = float(attrs.get("cost", 1.0))
        log_mu = math.log(max(mu, 1e-6))
        reactions.append({
            "idx": i, "src": src, "dst": dst,
            "name": rname, "mu": mu, "log_mu": log_mu,
            "hazard": hazard, "cost": cost,
        })

    prob = pulp.LpProblem("pathway_selection", pulp.LpMaximize)
    x = {r["idx"]: pulp.LpVariable(f"x_{r['idx']}", cat="Binary") for r in reactions}

    # Objective
    prob += (
        pulp.lpSum(r["log_mu"] * x[r["idx"]] for r in reactions)
        - LAMBDA_HAZARD * pulp.lpSum(r["hazard"] * x[r["idx"]] for r in reactions)
        - LAMBDA_COST * pulp.lpSum(r["cost"] * x[r["idx"]] for r in reactions)
    )

    # Exactly one reaction must produce the target
    target_producers = [r for r in reactions if r["dst"] == target_smiles]
    if target_producers:
        prob += pulp.lpSum(x[r["idx"]] for r in target_producers) == 1

    # Flow conservation for intermediate nodes
    intermediates = {n for n in G.nodes()
                     if not G.nodes[n].get("is_target") and not G.nodes[n].get("is_starting_material")}
    for node in intermediates:
        produces = [r for r in reactions if r["dst"] == node]
        consumes = [r for r in reactions if r["src"] == node]
        if produces and consumes:
            prob += (
                pulp.lpSum(x[r["idx"]] for r in produces)
                == pulp.lpSum(x[r["idx"]] for r in consumes)
            )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    selected = [r for r in reactions if pulp.value(x[r["idx"]]) == 1.0]

    if not selected:
        # Fallback: pick all edges (trivial pathway)
        selected = reactions

    obj_val = pulp.value(prob.objective) or 0.0
    log_yield_sum = sum(r["log_mu"] for r in selected)
    hazard_sum = LAMBDA_HAZARD * sum(r["hazard"] for r in selected)
    cost_sum = LAMBDA_COST * sum(r["cost"] for r in selected)

    return PathwayMILP(
        selected_edges=[(r["src"], r["dst"]) for r in selected],
        selected_reactions=[r["name"] for r in selected],
        objective_value=round(obj_val, 4),
        log_yield_sum=round(log_yield_sum, 4),
        hazard_penalty=round(hazard_sum, 4),
        cost_penalty=round(cost_sum, 4),
        status=status,
    )
