"""
Retrosynthetic candidate generation.

Priority order:
  1. AiZynthFinder  — MCTS over USPTO retrosynthetic templates (most accurate)
  2. ReactionT5v2   — T5 seq2seq model trained on USPTO_50k + ORD (sagawa/ReactionT5v2-retrosynthesis)
  3. SMARTS templates — hand-coded rules for ~20 common named reactions (always-available fallback)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


@dataclass
class RetroCandidate:
    reactant_smiles: list[str]          # typically 2 precursors
    reaction_name: str
    template_smarts: str
    conditions: dict = field(default_factory=dict)  # temperature, solvent, catalyst
    source: str = "template"            # "aizynthfinder" | "template"


# ---------------------------------------------------------------------------
# Named reaction SMARTS templates  (retrosynthetic direction: product → precursors)
# Format: (name, product_smarts, [precursor_smarts_list], default_conditions)
# ---------------------------------------------------------------------------
RETRO_TEMPLATES: list[tuple] = [
    # Esterification: ester → carboxylic acid + alcohol
    (
        "Fischer_Esterification",
        "[CX3](=O)[OX2][#6]",
        ["[CX3](=O)[OX2H]", "[OX2H][#6]"],
        {"temperature_c": 60, "catalyst": "H2SO4", "solvent": "neat"},
    ),
    # Amide bond formation: amide → carboxylic acid + amine
    (
        "Amide_Coupling",
        "[CX3](=O)[NX3]",
        ["[CX3](=O)[OX2H]", "[NX3H2,NX3H1]"],
        {"temperature_c": 25, "catalyst": "EDC/HOBt", "solvent": "DMF"},
    ),
    # Aldol condensation: β-hydroxy carbonyl → two carbonyls
    (
        "Aldol_Condensation",
        "[CX3](=O)[CX4][CX3](=O)",
        ["[CX3H](=O)", "[CX3](=O)[CX4H2]"],
        {"temperature_c": 25, "catalyst": "NaOH", "solvent": "H2O/EtOH"},
    ),
    # Diels-Alder: any cyclohexene → diene + dienophile (no stereo restriction)
    (
        "Diels_Alder",
        
        "C1CC=CCC1",
        ["C=CC=C", "C=C"],
        {"temperature_c": 150, "solvent": "toluene"},
    ),
    # Grignard addition: alcohol → aldehyde/ketone + Grignard
    (
        "Grignard_Addition",
        "[CX4]([OX2H])([#6])[#6]",
        ["[CX3H](=O)", "[MgBr][#6]"],
        {"temperature_c": 0, "solvent": "Et2O", "catalyst": "Mg"},
    ),
    # Suzuki coupling: any aryl-aryl bond → aryl halide + aryl boronic acid
    (
        "Suzuki_Coupling",
        "[c:1]-[c:2]",
        ["[c:1][Br]", "[c:2]B(O)O"],
        {"temperature_c": 80, "catalyst": "Pd(PPh3)4", "solvent": "DMF/H2O", "base": "K2CO3"},
    ),
    # Wittig reaction: alkene → aldehyde + phosphonium ylide (generic ylide)
    (
        "Wittig",
        "[CX3]=[CX3]",
        ["[CX3H](=O)", "[#6]=[P]"],
        {"temperature_c": 25, "solvent": "THF"},
    ),
    # Nucleophilic substitution (SN2): secondary alkyl → alkyl halide + nucleophile
    (
        "SN2_Substitution",
        "[CX4][NX3,OX2,SX2]",
        ["[CX4][Cl,Br,I]", "[NX3H2,OX2H,SX2H]"],
        {"temperature_c": 25, "solvent": "acetone"},
    ),
    # Reductive amination: amine → ketone/aldehyde + amine
    (
        "Reductive_Amination",
        "[CX4H][NX3H]",
        ["[CX3](=O)", "[NX3H2]"],
        {"temperature_c": 25, "catalyst": "NaBH3CN", "solvent": "MeOH"},
    ),
    # Acylation (Friedel-Crafts): aryl ketone → arene + acyl chloride
    (
        "Friedel_Crafts_Acylation",
        "c1ccccc1C(=O)",
        ["c1ccccc1", "[CX3](=O)[Cl]"],
        {"temperature_c": 25, "catalyst": "AlCl3", "solvent": "DCM"},
    ),
    # Michael addition: 1,4-addition product → Michael acceptor + donor
    (
        "Michael_Addition",
        "[CX3](=O)[CX4][CX4][CX3](=O)",
        ["[CX3](=O)[CX3]=[CX3]", "[CX4H2][CX3](=O)"],
        {"temperature_c": 25, "catalyst": "Et3N", "solvent": "EtOH"},
    ),
    # Acetal formation: acetal → aldehyde + alcohol
    (
        "Acetal_Formation",
        "[CX4]([OX2][#6])([OX2][#6])[H]",
        ["[CX3H](=O)", "[OX2H][#6]"],
        {"temperature_c": 25, "catalyst": "p-TsOH", "solvent": "benzene"},
    ),
    # Epoxide opening: amino alcohol → epoxide + amine
    (
        "Epoxide_Opening",
        "[CX4]([OX2H])[CX4][NX3]",
        ["[C1OC1]", "[NX3H2]"],
        {"temperature_c": 25, "solvent": "H2O/EtOH"},
    ),
    # Hydrogenation: alkane → alkene (retro)
    (
        "Hydrogenation",
        "[CX4H2][CX4H2]",
        ["[CX3H]=[CX3H]", "[H][H]"],
        {"temperature_c": 25, "catalyst": "Pd/C", "solvent": "EtOH", "pressure": "1 atm H2"},
    ),
]


def get_retro_candidates(target_smiles: str, max_candidates: int = 5) -> "list[RetroCandidate]":
    """
    Generate retrosynthetic candidates for target_smiles.

    Priority:
      1. AiZynthFinder (MCTS, most accurate)
      2. ReactionT5v2-retrosynthesis (T5 seq2seq, literature-trained)
      3. SMARTS template matching (always-available fallback)

    Candidates from ReactionT5v2 are merged with template candidates when
    both are available, giving the scoring layer more options to rank.
    """
    candidates = _aizynthfinder_retro(target_smiles, max_candidates)
    if candidates:
        while len(candidates) > max_candidates:
            candidates.pop()
        return candidates

    # ReactionT5v2 retrosynthesis
    t5_candidates = _reactiont5_retro(target_smiles, max_candidates)

    # SMARTS templates (always run to fill gaps and provide named-reaction labels)
    template_candidates = _template_retro(target_smiles, max_candidates)

    # Merge: T5 first (higher accuracy), then templates, deduplicate by reactant set
    seen_reactant_sets: set[frozenset] = set()
    merged: list[RetroCandidate] = []
    for cand in t5_candidates + template_candidates:
        key = frozenset(cand.reactant_smiles)
        if key not in seen_reactant_sets:
            seen_reactant_sets.add(key)
            merged.append(cand)

    while len(merged) > max_candidates:
        merged.pop()
    return merged


def _aizynthfinder_retro(smiles: str, max_candidates: int) -> "list[RetroCandidate]":
    """Wrap AiZynthFinder. Returns empty list if unavailable."""
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder  # type: ignore[import-not-found]
        from aizynthfinder.context.config import Configuration  # type: ignore[import-not-found]

        config = Configuration.from_dict({
            "finder": {"time_limit": 30, "max_transforms": 3},
            "policy": {"files": {}},      # uses bundled USPTO policy by default
        })
        finder = AiZynthFinder(configdict=config.as_dict())
        finder.target_smiles = smiles
        finder.tree_search()
        finder.build_routes()

        candidates = []
        for route in finder.routes.dict_with_scores()[:max_candidates]:
            rxn = route.get("reaction", {})
            reactants = rxn.get("reactants", [])
            if not reactants:
                continue
            candidates.append(RetroCandidate(
                reactant_smiles=reactants,
                reaction_name=rxn.get("name", "unknown"),
                template_smarts=rxn.get("smarts", ""),
                conditions={},
                source="aizynthfinder",
            ))
        return candidates
    except Exception as exc:
        logger.warning("AiZynthFinder unavailable (%s), falling back to templates.", exc)
        return []


def _reactiont5_retro(smiles: str, max_candidates: int) -> "list[RetroCandidate]":
    """
    Use ReactionT5v2-retrosynthesis to predict precursors for the given product SMILES.

    The T5 model returns beam-search candidates as raw SMILES strings, each potentially
    containing multiple reactants separated by ".". We convert each beam candidate into
    a RetroCandidate with source="reactiont5".

    Because T5 retrosynthesis is trained on USPTO_50k + ORD, it provides
    literature-grounded predictions that are more accurate than SMARTS templates
    for diverse functional groups.
    """
    try:
        from chem.reactiont5 import retrosynthesis as t5_retro  # type: ignore[import-not-found]
        beam_results = t5_retro(smiles, num_beams=max(5, max_candidates), num_return=max_candidates)
        candidates = []
        for beam_smiles in beam_results:
            # Each beam result may be "reactant1.reactant2" or just "reactant1"
            reactants = [r.strip() for r in beam_smiles.split(".") if r.strip()]
            if not reactants:
                continue
            candidates.append(RetroCandidate(
                reactant_smiles=reactants,
                reaction_name="unknown",        # T5 does not return reaction names
                template_smarts="",
                conditions={},
                source="reactiont5",
            ))
        return candidates
    except Exception as exc:
        logger.warning("ReactionT5v2 retrosynthesis unavailable (%s).", exc)
        return []


def _template_retro(smiles: str, max_candidates: int) -> "list[RetroCandidate]":
    """Match target against retrosynthetic SMARTS templates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    candidates = []
    for name, product_smarts, precursor_smarts_list, conditions in RETRO_TEMPLATES:
        pattern = Chem.MolFromSmarts(product_smarts)
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            # Build placeholder precursor SMILES from SMARTS (simplified)
            precursor_smiles: list[str] = []
            for ps in precursor_smarts_list:
                pmol = Chem.MolFromSmarts(ps)
                if pmol:
                    # Convert SMARTS to a valid SMILES if possible
                    smi = Chem.MolToSmiles(Chem.MolFromSmarts(ps)) if Chem.MolFromSmarts(ps) else ps
                    precursor_smiles.append(smi)
                else:
                    precursor_smiles.append(ps)
            candidates.append(RetroCandidate(
                reactant_smiles=precursor_smiles,
                reaction_name=name,
                template_smarts=product_smarts,
                conditions=conditions,
                source="template",
            ))
        if len(candidates) >= max_candidates:
            break

    return candidates
