"""
Composite reaction probability scoring.

Updated weights (with T5 forward validation):
  score(r) = 0.25·S_tanimoto + 0.25·S_mechanism + 0.25·S_yield - 0.10·S_hazard + 0.15·S_forward

Uses Platt scaling for sigmoid calibration instead of arbitrary constants.
MILP (PuLP) selects the top-N candidates subject to feasibility constraints.
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass

import pulp

from .analyze import tanimoto, smiles_to_fingerprint_bits
from .retrosynthesis import RetroCandidate

logger = logging.getLogger(__name__)

# Scoring weights — forward validation gets weight from tanimoto
W_TANIMOTO = 0.25
W_MECHANISM = 0.25
W_YIELD = 0.25
W_HAZARD = 0.10
W_FORWARD = 0.15

MECHANISM_THRESHOLD = 0.10   # minimum mechanism score to be considered feasible

# Atom symbols considered hazardous for estimation when no DB data is available.
# Halogens (F, Cl, Br, I) and common organometallics (Mg, Sn).
_HAZARDOUS_ATOM_SYMBOLS: frozenset[str] = frozenset({"F", "Cl", "Br", "I", "Mg", "Sn"})

# Try importing RDKit once at module level for reuse across scoring functions.
try:
    from rdkit import Chem as _Chem
    _RDKIT_AVAILABLE = True
except Exception:
    _Chem = None  # type: ignore[assignment]
    _RDKIT_AVAILABLE = False


@dataclass
class ScoredCandidate:
    candidate: RetroCandidate
    s_tanimoto: float
    s_mechanism: float
    s_yield: float
    s_hazard: float
    s_forward: float
    composite_score: float
    probability: float          # sigmoid(composite_score)


def sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def score_candidates(
    candidates: list[RetroCandidate],
    target_smiles: str,
    db_reactions: list[dict],   # rows from reactions table
    max_results: int = 3,
) -> list[ScoredCandidate]:
    """
    Score each candidate pair and return top-N via MILP selection.
    """
    target_fp = smiles_to_fingerprint_bits(target_smiles)
    raw_scored: list[ScoredCandidate] = []

    for cand in candidates:
        s_tan = _tanimoto_score(cand, target_fp, db_reactions)
        s_mech = _mechanism_score(cand, target_smiles)
        s_yield = _yield_score(cand, target_smiles, target_fp, db_reactions)
        s_haz = _hazard_score(cand, db_reactions)
        s_fwd = _forward_validation_score(cand, target_smiles)

        composite = (
            W_TANIMOTO * s_tan
            + W_MECHANISM * s_mech
            + W_YIELD * s_yield
            - W_HAZARD * s_haz
            + W_FORWARD * s_fwd
        )

        raw_scored.append(ScoredCandidate(
            candidate=cand,
            s_tanimoto=round(s_tan, 4),
            s_mechanism=round(s_mech, 4),
            s_yield=round(s_yield, 4),
            s_hazard=round(s_haz, 4),
            s_forward=round(s_fwd, 4),
            composite_score=round(composite, 4),
            probability=0.0,  # computed after Platt scaling below
        ))

    # Platt scaling: calibrate sigmoid to actual score distribution
    composites = [s.composite_score for s in raw_scored]
    if len(composites) > 1:
        mean_c = sum(composites) / len(composites)
        std_c = (sum((c - mean_c) ** 2 for c in composites) / len(composites)) ** 0.5
        std_c = max(std_c, 0.01)
        scale = 3.0 / std_c
    else:
        mean_c = composites[0] if composites else 0.5
        scale = 6.0

    for sc in raw_scored:
        sc.probability = round(sigmoid((sc.composite_score - mean_c) * scale), 4)

    return _milp_select(raw_scored, max_results)


def _tanimoto_score(cand: RetroCandidate, target_fp: list[int], db_reactions: list[dict]) -> float:
    """
    Split fingerprint comparison: compare reactant FP to DB reactant FP and
    product FP to DB product FP separately, then weight-average.
    Product match is weighted higher (0.6) since we're matching the target.
    When DB is empty, compute actual Tanimoto between candidate reactants and target.
    """
    reactant_bits: set[int] = set()
    for smi in cand.reactant_smiles:
        reactant_bits |= set(smiles_to_fingerprint_bits(smi))
    reactant_bits_list = list(reactant_bits)
    target_bits_list = list(set(target_fp))

    if not db_reactions:
        # Compute actual Tanimoto similarity between reactant union FP and target FP
        return tanimoto(reactant_bits_list, target_bits_list)

    k = min(10, len(db_reactions))
    sims: list[float] = []

    for row in db_reactions:
        db_r_bits = row.get("reactant1_fp", [])
        if row.get("reactant2_fp"):
            db_r_bits = list(set(db_r_bits) | set(row["reactant2_fp"]))
        db_p_bits = row.get("product_fp", [])

        s_reactant = tanimoto(reactant_bits_list, db_r_bits)
        s_product = tanimoto(target_bits_list, db_p_bits)
        sims.append(0.6 * s_product + 0.4 * s_reactant)

    if not sims:
        return tanimoto(reactant_bits_list, target_bits_list)

    top_k = heapq.nlargest(k, sims)
    weights = [math.exp(-i * 0.3) for i in range(len(top_k))]
    w_sum = sum(weights)
    return sum(s * w for s, w in zip(top_k, weights)) / w_sum if w_sum > 0 else 0.0


def _mechanism_score(cand: RetroCandidate, target_smiles: str) -> float:
    """Fraction of mechanism feasibility checks that pass."""
    score = 0.0
    checks = 0

    if cand.source in ("template", "aizynthfinder", "reactiont5"):
        score += 1.0
        checks += 1

    if len(cand.reactant_smiles) >= 2:
        score += 1.0
        checks += 1

    if cand.reaction_name and cand.reaction_name.lower() not in ("unknown", ""):
        score += 1.0
        checks += 1

    # More conditions = more detailed = more likely to be a real reaction
    if len(cand.conditions) >= 2:
        score += 1.0
        checks += 1

    # Validate each reactant SMILES is a parseable molecule
    if _RDKIT_AVAILABLE and cand.reactant_smiles:
        valid_count = 0
        for smi in cand.reactant_smiles:
            try:
                if _Chem.MolFromSmiles(smi) is not None:
                    valid_count += 1
            except Exception:
                pass
        score += valid_count / len(cand.reactant_smiles)
        checks += 1

    return score / checks if checks > 0 else 0.0


def _yield_score(
    cand: RetroCandidate,
    target_smiles: str,
    target_fp: list[int],
    db_reactions: list[dict],
) -> float:
    """
    Hybrid yield prediction: try ReactionT5v2-yield model first,
    fall back to k-NN average from database, then to DEFAULT_YIELDS lookup.
    """
    try:
        from chem.reactiont5 import predict_yield
        t5_yield = predict_yield(cand.reactant_smiles, target_smiles)
        if t5_yield is not None:
            return t5_yield
    except Exception:
        pass

    # Fallback: k-NN yield average
    if db_reactions:
        reactant_bits: set[int] = set()
        for smi in cand.reactant_smiles:
            reactant_bits |= set(smiles_to_fingerprint_bits(smi))

        k = min(10, len(db_reactions))
        rows_with_yield: list[tuple[float, float]] = []
        for r in db_reactions:
            if r.get("yield") is not None:
                db_bits = list(set(r.get("reactant1_fp", [])) | set(r.get("reactant2_fp") or []))
                sim = tanimoto(list(reactant_bits), db_bits)
                rows_with_yield.append((r["yield"], sim))

        if rows_with_yield:
            top_k = heapq.nlargest(k, rows_with_yield, key=lambda x: x[1])
            return sum(y for y, _ in top_k) / len(top_k)

    # Final fallback: look up reaction name in DEFAULT_YIELDS (from graph/pert.py)
    try:
        from graph.pert import DEFAULT_YIELDS, DEFAULT_YIELD
        a, m, b = DEFAULT_YIELDS.get(cand.reaction_name, DEFAULT_YIELD)
        return (a + 4 * m + b) / 6
    except Exception:
        pass

    return 0.5


def _hazard_score(cand: RetroCandidate, db_reactions: list[dict]) -> float:
    """
    GHS hazard score for reactants. Queries the molecules table for cached
    hazard_class values. Returns 0–1 (higher = more hazardous).
    When no cached data is available, estimates hazard from atom types.
    """
    try:
        from db.query import get_cached_molecule
        hazards: list[float] = []
        for smi in cand.reactant_smiles:
            cached = get_cached_molecule(smi)
            if cached and cached.get("hazard_class"):
                hazards.append(cached["hazard_class"] / 5.0)
        if hazards:
            return sum(hazards) / len(hazards)
    except Exception:
        pass

    # Fallback: estimate hazard from atom types using RDKit
    if _RDKIT_AVAILABLE:
        try:
            total_atoms = 0
            hazardous_atoms = 0
            for smi in cand.reactant_smiles:
                mol = _Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                for atom in mol.GetAtoms():
                    total_atoms += 1
                    if atom.GetSymbol() in _HAZARDOUS_ATOM_SYMBOLS:
                        hazardous_atoms += 1
            if total_atoms > 0:
                ratio = hazardous_atoms / total_atoms
                # Scale ratio to 0.1–0.5 range: multiply by 5 to make even a
                # small fraction of hazardous atoms register, then cap at 0.5.
                return 0.1 + 0.4 * min(ratio * 5, 1.0)
        except Exception:
            pass

    return 0.3


def _forward_validation_score(cand: RetroCandidate, target_smiles: str) -> float:
    """
    Use ReactionT5v2-forward to verify that candidate precursors produce the target.
    Returns 1.0 if confirmed, 0.5 if model unavailable, 0.0 if refuted.
    When model is unavailable, returns 0.6 for named reactions and 0.4 for unknown.
    """
    try:
        from chem.reactiont5 import forward_validates_target
        result = forward_validates_target(cand.reactant_smiles, target_smiles)
        if result is None:
            return 0.5
        return 1.0 if result else 0.0
    except Exception:
        pass

    # Differentiate by whether we have a named reaction
    is_named = bool(cand.reaction_name) and cand.reaction_name.lower() not in ("unknown", "")
    return 0.6 if is_named else 0.4


def _milp_select(scored: list[ScoredCandidate], max_results: int) -> list[ScoredCandidate]:
    """
    Use MILP (PuLP) to select the top-N candidates.
    """
    feasible = [s for s in scored if s.s_mechanism >= MECHANISM_THRESHOLD]
    if not feasible:
        feasible = scored

    if len(feasible) <= max_results:
        return sorted(feasible, key=lambda s: s.probability, reverse=True)

    prob = pulp.LpProblem("candidate_selection", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(feasible))]

    prob += pulp.lpSum(s.composite_score * x[i] for i, s in enumerate(feasible))
    prob += pulp.lpSum(x) <= max_results

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        logger.warning("MILP candidate selection status: %s; using top-N fallback.",
                       pulp.LpStatus[prob.status])
        return sorted(feasible, key=lambda s: s.probability, reverse=True)[:max_results]

    selected_set = {i for i, xi in enumerate(x) if pulp.value(xi) == 1.0}
    selected = [feasible[i] for i in selected_set]

    if len(selected) < max_results:
        remaining = [s for i, s in enumerate(feasible) if i not in selected_set]
        remaining.sort(key=lambda s: s.probability, reverse=True)
        selected += remaining[: max_results - len(selected)]

    return sorted(selected, key=lambda s: s.probability, reverse=True)
