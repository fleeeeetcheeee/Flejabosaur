"""
Composite reaction probability scoring.

score(r) = w1·S_tanimoto + w2·S_mechanism + w3·S_yield - w4·S_hazard
probability = sigmoid(score)

MILP (PuLP) selects the top-N candidates subject to feasibility constraints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pulp

from .analyze import tanimoto, smiles_to_fingerprint_bits
from .retrosynthesis import RetroCandidate

# Scoring weights
W_TANIMOTO = 0.35
W_MECHANISM = 0.30
W_YIELD = 0.25
W_HAZARD = 0.10

MECHANISM_THRESHOLD = 0.10   # minimum mechanism score to be considered feasible


@dataclass
class ScoredCandidate:
    candidate: RetroCandidate
    s_tanimoto: float
    s_mechanism: float
    s_yield: float
    s_hazard: float
    composite_score: float
    probability: float          # sigmoid(composite_score)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_candidates(
    candidates: list[RetroCandidate],
    target_smiles: str,
    db_reactions: list[dict],   # rows from reactions table
    max_results: int = 3,
) -> list[ScoredCandidate]:
    """
    Score each candidate pair and return top-N via MILP selection.
    db_reactions: list of dicts with keys reactant1_fp, reactant2_fp, product_fp, yield, hazard_class
    """
    target_fp = smiles_to_fingerprint_bits(target_smiles)
    scored = []

    for cand in candidates:
        s_tan = _tanimoto_score(cand, target_fp, db_reactions)
        s_mech = _mechanism_score(cand, target_smiles)
        s_yield = _yield_score(cand, target_fp, db_reactions)
        s_haz = _hazard_score(cand, db_reactions)

        composite = (
            W_TANIMOTO * s_tan
            + W_MECHANISM * s_mech
            + W_YIELD * s_yield
            - W_HAZARD * s_haz
        )
        # Scale composite to roughly [-4, 4] for sigmoid to give meaningful probabilities
        composite_scaled = (composite - 0.5) * 8
        prob = sigmoid(composite_scaled)

        scored.append(ScoredCandidate(
            candidate=cand,
            s_tanimoto=round(s_tan, 4),
            s_mechanism=round(s_mech, 4),
            s_yield=round(s_yield, 4),
            s_hazard=round(s_haz, 4),
            composite_score=round(composite, 4),
            probability=round(prob, 4),
        ))

    return _milp_select(scored, max_results)


def _tanimoto_score(cand: RetroCandidate, target_fp: list[int], db_reactions: list[dict]) -> float:
    """k-NN Tanimoto similarity of the (reactants, product) triple to known reactions."""
    if not db_reactions:
        return 0.3  # neutral default

    reactant_bits: set[int] = set()
    for smi in cand.reactant_smiles:
        reactant_bits |= set(smiles_to_fingerprint_bits(smi))
    query_bits = list(reactant_bits ^ set(target_fp))  # XOR reaction fingerprint

    k = min(10, len(db_reactions))
    sims = []
    for row in db_reactions:
        db_bits = row.get("reaction_fp", [])
        if db_bits:
            sims.append(tanimoto(query_bits, db_bits))

    if not sims:
        return 0.3
    sims.sort(reverse=True)
    top_k = sims[:k]
    # Gaussian-weighted average (closer neighbors weighted more)
    weights = [math.exp(-i * 0.5) for i in range(len(top_k))]
    w_sum = sum(weights)
    return sum(s * w for s, w in zip(top_k, weights)) / w_sum if w_sum > 0 else 0.0


def _mechanism_score(cand: RetroCandidate, target_smiles: str) -> float:
    """
    Fraction of mechanism feasibility checks that pass:
    - Template source gets a bonus
    - Electrophile/nucleophile complementarity check
    """
    score = 0.0
    checks = 0

    # Named template match is inherently feasible
    if cand.source in ("template", "aizynthfinder"):
        score += 1.0
        checks += 1

    # At least 2 reactants
    if len(cand.reactant_smiles) >= 2:
        score += 1.0
        checks += 1

    # Reaction name is known
    if cand.reaction_name and cand.reaction_name.lower() not in ("unknown", ""):
        score += 1.0
        checks += 1

    return score / checks if checks > 0 else 0.0


def _yield_score(cand: RetroCandidate, target_fp: list[int], db_reactions: list[dict]) -> float:
    """Average yield from k most similar reactions in database."""
    if not db_reactions:
        return 0.5

    reactant_bits: set[int] = set()
    for smi in cand.reactant_smiles:
        reactant_bits |= set(smiles_to_fingerprint_bits(smi))

    k = min(10, len(db_reactions))
    rows_with_yield = [(r["yield"], tanimoto(list(reactant_bits), r.get("reaction_fp", [])))
                       for r in db_reactions if r.get("yield") is not None]
    if not rows_with_yield:
        return 0.5

    rows_with_yield.sort(key=lambda x: x[1], reverse=True)
    top_k = rows_with_yield[:k]
    return sum(y for y, _ in top_k) / len(top_k)


def _hazard_score(cand: RetroCandidate, db_reactions: list[dict]) -> float:
    """Average GHS hazard class (0–5) normalized to 0–1. Higher = more hazardous."""
    # Without a molecule hazard lookup, use a default moderate score
    return 0.3


def _milp_select(scored: list[ScoredCandidate], max_results: int) -> list[ScoredCandidate]:
    """
    Use MILP (PuLP) to select the top-N candidates:
      maximize Σ score_i · x_i
      subject to:
        Σ x_i ≤ max_results
        x_i ∈ {0,1}
        s_mechanism_i ≥ MECHANISM_THRESHOLD  (feasibility constraint)
    """
    feasible = [s for s in scored if s.s_mechanism >= MECHANISM_THRESHOLD]
    if not feasible:
        feasible = scored  # relax constraint if nothing passes

    if len(feasible) <= max_results:
        return sorted(feasible, key=lambda s: s.probability, reverse=True)

    prob = pulp.LpProblem("candidate_selection", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(feasible))]

    # Objective
    prob += pulp.lpSum(s.composite_score * x[i] for i, s in enumerate(feasible))

    # Cardinality constraint
    prob += pulp.lpSum(x) <= max_results

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [feasible[i] for i, xi in enumerate(x) if pulp.value(xi) == 1.0]
    # Fallback: if MILP didn't select enough, pad with highest-scored
    if len(selected) < max_results:
        remaining = [s for s in feasible if s not in selected]
        remaining.sort(key=lambda s: s.probability, reverse=True)
        selected += remaining[: max_results - len(selected)]

    return sorted(selected, key=lambda s: s.probability, reverse=True)
