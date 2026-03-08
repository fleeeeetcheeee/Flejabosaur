"""
Flejabosaur — Retrosynthesis API

Unified FastAPI backend for HackTJ 13.0.

Endpoints:
  POST /synthesize   — IUPAC name → top-N precursor pairs (single-step)
  POST /multistep    — IUPAC name → full synthesis DAG with CPM/PERT/MILP
  GET  /molecule/{s} — molecular properties + SVG for a SMILES string
  GET  /health       — liveness check
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must run before any local imports so bare subpackage imports
# (e.g. `from chem.iupac import ...`) resolve regardless of working directory.
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chem.analyze import analyze, MolecularAnalysis
from chem.iupac import iupac_to_smiles
from chem.retrosynthesis import get_retro_candidates
from chem.scoring import ScoredCandidate, score_candidates
from db.query import (
    cache_molecule,
    ensure_db,
    get_cached_molecule,
    get_cached_smiles,
    tanimoto_knn,
)
from graph.cpm import run_cpm
from graph.dag import (
    build_retro_tree,
    build_synthesis_dag,
    dag_to_dict,
    topological_order,
)
from graph.milp import select_optimal_pathway
from graph.pert import run_pert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    """Startup / shutdown lifecycle hook (replaces deprecated on_event)."""
    ensure_db()
    logger.info("Database initialized.")
    yield


app = FastAPI(
    title="Flejabosaur Retrosynthesis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    forward: float


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

_EMPTY_PROPS_DEFAULTS = dict(
    mw=0.0, logp=0.0, tpsa=0.0, hbd=0, hba=0, rotatable_bonds=0,
    num_rings=0, num_aromatic_rings=0, functional_groups=[],
    electrophilic_sites=[], nucleophilic_sites=[], svg="",
)


async def resolve_smiles(iupac_name: str) -> str:
    """Resolve IUPAC name to SMILES, using DB cache when available."""
    cached = get_cached_smiles(iupac_name.lower())
    if cached:
        return cached
    return await iupac_to_smiles(iupac_name)


def _analysis_to_props(analysis: MolecularAnalysis) -> MoleculeProperties:
    """Convert internal MolecularAnalysis dataclass to the API response model."""
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


def _safe_analyze(smiles: str) -> MoleculeProperties:
    """Analyze a SMILES string, returning empty properties on failure."""
    try:
        return _analysis_to_props(analyze(smiles))
    except ValueError:
        logger.warning("Failed to analyze SMILES: %r", smiles)
        return MoleculeProperties(smiles=smiles, **_EMPTY_PROPS_DEFAULTS)


def _scored_to_pair(sc: ScoredCandidate) -> PrecursorPair:
    """Convert a ScoredCandidate to the API response model."""
    reactants = sc.candidate.reactant_smiles
    r_a = reactants[0] if reactants else ""
    r_b = reactants[1] if len(reactants) > 1 else None

    return PrecursorPair(
        reactant_a=_safe_analyze(r_a),
        reactant_b=_safe_analyze(r_b) if r_b else None,
        reaction_name=sc.candidate.reaction_name,
        conditions=sc.candidate.conditions,
        probability=sc.probability,
        composite_score=sc.composite_score,
        score_breakdown=ScoreBreakdown(
            tanimoto=sc.s_tanimoto,
            mechanism=sc.s_mechanism,
            yield_score=sc.s_yield,
            hazard=sc.s_hazard,
            forward=sc.s_forward,
        ),
        source=sc.candidate.source,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    """IUPAC name → top-N precursor pairs for the final synthesis step."""

    # 1. IUPAC → SMILES
    try:
        smiles = await resolve_smiles(req.iupac_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 2. Molecular analysis
    try:
        mol_analysis = analyze(smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid molecule: {exc}")

    # 3. Cache the target molecule
    cache_molecule(
        smiles=mol_analysis.smiles,
        iupac_name=req.iupac_name.lower(),
        mw=mol_analysis.mw,
        logp=mol_analysis.logp,
        tpsa=mol_analysis.tpsa,
        svg=mol_analysis.svg,
    )

    # 4. Retrosynthetic candidate generation (overgenerate, then select)
    candidates = get_retro_candidates(smiles, max_candidates=req.max_candidates * 3)

    # 5. Database similarity search for scoring context
    db_reactions = tanimoto_knn(mol_analysis.ecfp4, k=20)

    # 6. Score and select top-N via MILP
    scored = score_candidates(
        candidates, smiles, db_reactions, max_results=req.max_candidates,
    )

    return SynthesizeResponse(
        smiles=mol_analysis.smiles,
        properties=_analysis_to_props(mol_analysis),
        precursor_pairs=[_scored_to_pair(sc) for sc in scored],
    )


@app.post("/multistep", response_model=MultistepResponse)
async def multistep(req: MultistepRequest) -> MultistepResponse:
    """Full multi-step synthesis DAG with CPM, PERT, and MILP analysis."""

    try:
        smiles = await resolve_smiles(req.iupac_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Build per-molecule retrosynthetic tree
    retro_tree = build_retro_tree(
        smiles,
        max_depth=req.max_steps,
        max_candidates_per_molecule=req.max_candidates_per_step,
    )

    # Build DAG from the retro tree
    G = build_synthesis_dag(smiles, retro_tree)

    # Topological order (starting materials first)
    topo = topological_order(G)

    # CPM — identify critical path
    cpm_edges, critical_path = run_cpm(G)

    # PERT — probabilistic yield estimates per edge
    pert_result = run_pert(G)

    # Annotate DAG edges with PERT mu values using explicit key matching
    # (avoids relying on iteration-order alignment between edges and steps)
    edge_list = list(G.edges(data=True))
    for idx, (src, dst, attrs) in enumerate(edge_list):
        if idx < len(pert_result.steps):
            attrs["mu"] = pert_result.steps[idx].mu

    # MILP pathway selection (uses mu values from PERT)
    milp_result = select_optimal_pathway(G, smiles)

    # Serialize DAG for React Flow (frontend)
    dag_dict = dag_to_dict(G)

    # Annotate serialized edges with CPM/MILP metadata
    critical_edge_set = {(e.src, e.dst) for e in cpm_edges if e.is_critical}
    optimal_edge_set = set(milp_result.selected_edges)
    for edge in dag_dict["edges"]:
        key = (edge["source"], edge["target"])
        edge["data"]["isCritical"] = key in critical_edge_set
        edge["data"]["isOptimal"] = key in optimal_edge_set

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
    """Return molecular properties and SVG for a SMILES string."""
    cached = get_cached_molecule(smiles)
    if cached and cached.get("svg"):
        return cached

    try:
        mol_analysis = analyze(smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cache_molecule(
        smiles=mol_analysis.smiles,
        mw=mol_analysis.mw,
        logp=mol_analysis.logp,
        tpsa=mol_analysis.tpsa,
        svg=mol_analysis.svg,
    )
    return _analysis_to_props(mol_analysis).model_dump()


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}
