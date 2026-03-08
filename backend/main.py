"""
Flejabosaur — Retrosynthesis API
FastAPI backend
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the backend/ directory is on sys.path so that bare imports like
# `from chem.iupac import ...` resolve correctly when the server is started
# from any working directory (e.g. `cd backend && uvicorn main:app --reload`
# or `uvicorn backend.main:app` from the project root).
_BACKEND_DIR = Path(__file__).parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chem.iupac import iupac_to_smiles
from chem.analyze import analyze, MolecularAnalysis
from chem.retrosynthesis import get_retro_candidates
from chem.scoring import score_candidates, ScoredCandidate
from graph.dag import build_synthesis_dag, dag_to_dict
from graph.cpm import run_cpm
from graph.pert import run_pert
from graph.milp import select_optimal_pathway
from db.query import ensure_db, tanimoto_knn, cache_molecule, get_cached_smiles, get_cached_molecule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flejabosaur Retrosynthesis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    ensure_db()
    logger.info("Database initialized.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    iupac_name: str
    max_candidates: int = 3


class MultistepRequest(BaseModel):
    iupac_name: str
    max_steps: int = 3
    max_candidates_per_step: int = 2


class MoleculeProperties(BaseModel):
    smiles: str
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    num_rings: int
    num_aromatic_rings: int
    functional_groups: list[str]
    electrophilic_sites: list[int]
    nucleophilic_sites: list[int]
    svg: str


class ScoreBreakdown(BaseModel):
    tanimoto: float
    mechanism: float
    yield_score: float
    hazard: float


class PrecursorPair(BaseModel):
    reactant_a: MoleculeProperties
    reactant_b: MoleculeProperties | None
    reaction_name: str
    conditions: dict
    probability: float
    composite_score: float
    score_breakdown: ScoreBreakdown
    source: str


class SynthesizeResponse(BaseModel):
    smiles: str
    properties: MoleculeProperties
    precursor_pairs: list[PrecursorPair]


class MultistepResponse(BaseModel):
    smiles: str
    dag: dict
    critical_path: list[str]
    total_probability: float
    total_std: float
    optimal_pathway_edges: list[list[str]]
    milp_status: str
    topo_order: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def resolve_smiles(iupac_name: str) -> str:
    """Resolve IUPAC name to SMILES, using DB cache if available."""
    cached = get_cached_smiles(iupac_name.lower())
    if cached:
        return cached
    smiles = await iupac_to_smiles(iupac_name)
    return smiles


def analysis_to_props(analysis: MolecularAnalysis) -> MoleculeProperties:
    return MoleculeProperties(
        smiles=analysis.smiles,
        mw=analysis.mw,
        logp=analysis.logp,
        tpsa=analysis.tpsa,
        hbd=analysis.hbd,
        hba=analysis.hba,
        rotatable_bonds=analysis.rotatable_bonds,
        num_rings=analysis.num_rings,
        num_aromatic_rings=analysis.num_aromatic_rings,
        functional_groups=analysis.functional_groups,
        electrophilic_sites=analysis.electrophilic_sites,
        nucleophilic_sites=analysis.nucleophilic_sites,
        svg=analysis.svg,
    )


def scored_to_pair(sc: ScoredCandidate) -> PrecursorPair:
    reactants = sc.candidate.reactant_smiles
    r_a_smiles = reactants[0] if reactants else ""
    r_b_smiles = reactants[1] if len(reactants) > 1 else None

    try:
        a_analysis = analysis_to_props(analyze(r_a_smiles))
    except Exception:
        a_analysis = MoleculeProperties(smiles=r_a_smiles, mw=0, logp=0, tpsa=0,
                                         hbd=0, hba=0, rotatable_bonds=0, num_rings=0,
                                         num_aromatic_rings=0, functional_groups=[],
                                         electrophilic_sites=[], nucleophilic_sites=[], svg="")
    b_analysis = None
    if r_b_smiles:
        try:
            b_analysis = analysis_to_props(analyze(r_b_smiles))
        except Exception:
            b_analysis = MoleculeProperties(smiles=r_b_smiles, mw=0, logp=0, tpsa=0,
                                             hbd=0, hba=0, rotatable_bonds=0, num_rings=0,
                                             num_aromatic_rings=0, functional_groups=[],
                                             electrophilic_sites=[], nucleophilic_sites=[], svg="")

    return PrecursorPair(
        reactant_a=a_analysis,
        reactant_b=b_analysis,
        reaction_name=sc.candidate.reaction_name,
        conditions=sc.candidate.conditions,
        probability=sc.probability,
        composite_score=sc.composite_score,
        score_breakdown=ScoreBreakdown(
            tanimoto=sc.s_tanimoto,
            mechanism=sc.s_mechanism,
            yield_score=sc.s_yield,
            hazard=sc.s_hazard,
        ),
        source=sc.candidate.source,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    """
    Main endpoint: IUPAC name → top-N precursor pairs for the final synthesis step.
    """
    # 1. IUPAC → SMILES
    try:
        smiles = await resolve_smiles(req.iupac_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Molecular analysis
    try:
        mol_analysis = analyze(smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid molecule: {e}")

    # Cache molecule
    cache_molecule(
        smiles=mol_analysis.smiles,
        iupac_name=req.iupac_name.lower(),
        mw=mol_analysis.mw,
        logp=mol_analysis.logp,
        tpsa=mol_analysis.tpsa,
        svg=mol_analysis.svg,
    )

    # 3. Retrosynthetic candidate generation
    candidates = get_retro_candidates(smiles, max_candidates=req.max_candidates * 3)

    # 4. Database similarity search
    db_reactions = tanimoto_knn(mol_analysis.ecfp4, k=20)

    # 5. Score and select via MILP
    scored = score_candidates(candidates, smiles, db_reactions, max_results=req.max_candidates)

    return SynthesizeResponse(
        smiles=mol_analysis.smiles,
        properties=analysis_to_props(mol_analysis),
        precursor_pairs=[scored_to_pair(sc) for sc in scored],
    )


@app.post("/multistep", response_model=MultistepResponse)
async def multistep(req: MultistepRequest) -> MultistepResponse:
    """
    Multi-step synthesis: returns full synthesis DAG with CPM, PERT, and MILP analysis.
    """
    try:
        smiles = await resolve_smiles(req.iupac_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Collect retrosynthetic candidates for each step
    candidates_per_step = []
    current_smiles_list = [smiles]

    for step in range(req.max_steps):
        step_candidates = []
        for smi in current_smiles_list:
            cands = get_retro_candidates(smi, max_candidates=req.max_candidates_per_step)
            step_candidates.extend(cands)
        candidates_per_step.append(step_candidates)
        # Next layer: all precursors from this step
        next_smiles = []
        for c in step_candidates:
            next_smiles.extend(c.reactant_smiles)
        current_smiles_list = list(set(next_smiles))
        if not current_smiles_list:
            break

    # Build DAG
    G = build_synthesis_dag(smiles, candidates_per_step)

    # Topological order
    from graph.dag import topological_order
    topo = topological_order(G)

    # CPM
    cpm_edges, critical_path = run_cpm(G)

    # PERT
    pert_result = run_pert(G)

    # MILP pathway selection
    milp_result = select_optimal_pathway(G, smiles)

    # Serialize DAG for React Flow
    dag_dict = dag_to_dict(G)

    # Annotate critical edges in DAG dict
    critical_edge_set = {(e.src, e.dst) for e in cpm_edges if e.is_critical}
    for edge in dag_dict["edges"]:
        edge["data"]["isCritical"] = (edge["source"], edge["target"]) in critical_edge_set

    return MultistepResponse(
        smiles=smiles,
        dag=dag_dict,
        critical_path=critical_path,
        total_probability=pert_result.total_probability,
        total_std=pert_result.total_std,
        optimal_pathway_edges=[[s, d] for s, d in milp_result.selected_edges],
        milp_status=milp_result.status,
        topo_order=topo,
    )


@app.get("/molecule/{smiles:path}")
async def get_molecule(smiles: str) -> dict:
    """Return molecular properties and SVG for a given SMILES string."""
    cached = get_cached_molecule(smiles)
    if cached and cached.get("svg"):
        return cached

    try:
        mol_analysis = analyze(smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cache_molecule(
        smiles=mol_analysis.smiles,
        mw=mol_analysis.mw,
        logp=mol_analysis.logp,
        tpsa=mol_analysis.tpsa,
        svg=mol_analysis.svg,
    )
    return analysis_to_props(mol_analysis).model_dump()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
