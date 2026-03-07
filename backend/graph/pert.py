"""
PERT (Program Evaluation and Review Technique) for synthesis pathway probability.

Each reaction step r has three yield estimates:
  a_r = optimistic (90th percentile)
  m_r = most likely (median)
  b_r = pessimistic (10th percentile)

Expected yield: μ_r = (a_r + 4·m_r + b_r) / 6
Variance:       σ²_r = ((b_r - a_r) / 6)²

Pathway probability = Π μ_r
Total variance      = Σ σ²_r  (independence assumption)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx


# Default yield estimates by reaction type (tunable)
DEFAULT_YIELDS: dict[str, tuple[float, float, float]] = {
    "Fischer_Esterification":   (0.80, 0.90, 0.60),
    "Amide_Coupling":           (0.85, 0.92, 0.70),
    "Aldol_Condensation":       (0.55, 0.70, 0.35),
    "Diels_Alder":              (0.80, 0.88, 0.60),
    "Grignard_Addition":        (0.70, 0.80, 0.50),
    "Suzuki_Coupling":          (0.85, 0.92, 0.65),
    "Wittig":                   (0.65, 0.75, 0.45),
    "SN2_Substitution":         (0.70, 0.80, 0.50),
    "Reductive_Amination":      (0.75, 0.85, 0.55),
    "Friedel_Crafts_Acylation": (0.65, 0.78, 0.45),
    "Michael_Addition":         (0.75, 0.85, 0.55),
    "Acetal_Formation":         (0.80, 0.88, 0.60),
    "Epoxide_Opening":          (0.78, 0.88, 0.55),
    "Hydrogenation":            (0.90, 0.95, 0.75),
    # AiZynthFinder generic
    "unknown":                  (0.50, 0.65, 0.30),
}
DEFAULT_YIELD = (0.50, 0.65, 0.30)


@dataclass
class StepPERT:
    reaction_name: str
    a: float      # optimistic
    m: float      # most likely
    b: float      # pessimistic
    mu: float     # expected yield
    sigma2: float # variance


@dataclass
class PathwayPERT:
    steps: list[StepPERT]
    total_probability: float     # Π μ_r
    total_variance: float        # Σ σ²_r
    total_std: float             # √(total_variance)


def run_pert(G: nx.DiGraph) -> PathwayPERT:
    """Compute PERT estimates for all reaction edges in the DAG."""
    steps = []
    for _, _, attrs in G.edges(data=True):
        rname = attrs.get("reaction_name", "unknown")
        a, m, b = DEFAULT_YIELDS.get(rname, DEFAULT_YIELD)

        # Override with database yield if available
        db_yield = attrs.get("db_yield")
        if db_yield is not None:
            m = float(db_yield)
            a = min(m + 0.15, 0.99)
            b = max(m - 0.25, 0.05)

        mu = (a + 4 * m + b) / 6
        sigma2 = ((b - a) / 6) ** 2
        steps.append(StepPERT(reaction_name=rname, a=a, m=m, b=b,
                               mu=round(mu, 4), sigma2=round(sigma2, 6)))

    if not steps:
        return PathwayPERT(steps=[], total_probability=1.0, total_variance=0.0, total_std=0.0)

    total_prob = math.prod(s.mu for s in steps)
    total_var = sum(s.sigma2 for s in steps)
    total_std = math.sqrt(total_var)

    return PathwayPERT(
        steps=steps,
        total_probability=round(total_prob, 6),
        total_variance=round(total_var, 6),
        total_std=round(total_std, 6),
    )
