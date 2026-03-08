# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Flejabosaur** — HackTJ 13.0 retrosynthesis web app. User inputs a target molecule by IUPAC name; the system finds the most probable precursor molecule pair(s) for the final synthesis step, scored by literature-verified reaction mechanisms, Tanimoto similarity, and safety. An optional multi-step pathway planner uses graph theory (CPM, PERT, MILP) to optimize full synthesis routes.

## Commands

### Backend (Python + FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Load Figshare Reactron dataset into SQLite
python db/loader.py --input ../data/reactron/reactions.csv

# Run tests
pytest
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev       # localhost:3000
npm run build
npm run lint
```

## Architecture

```
User (IUPAC name)
  → [Next.js Frontend]
  → POST /synthesize  [FastAPI Backend]
      ├── iupac.py        IUPAC → SMILES via PubChem REST (OPSIN fallback)
      ├── analyze.py      RDKit molecular analysis → feature vectors + fingerprints
      ├── retrosynthesis.py  AiZynthFinder MCTS → candidate precursor pairs
      ├── scoring.py      MILP (PuLP) + Tanimoto k-NN scoring → probability
      └── graph/          NetworkX DAG → topological sort → CPM/PERT/MILP pathway
  → [SQLite DB]  Figshare Reactron reactions (Morgan FPs pre-computed)
```

### Key modules

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI app, routes: `POST /synthesize`, `POST /multistep`, `GET /molecule/{smiles}` |
| `backend/chem/iupac.py` | IUPAC name → SMILES (PubChem REST primary, OPSIN subprocess fallback) |
| `backend/chem/analyze.py` | RDKit: atom features, Gasteiger charges, ECFP4 fingerprint, functional group SMARTS |
| `backend/chem/retrosynthesis.py` | 3-tier retro: AiZynthFinder MCTS → ReactionT5v2 beam search → SMARTS templates |
| `backend/chem/reactiont5.py` | ReactionT5v2 integration: retrosynthesis, yield prediction, forward validation (HuggingFace) |
| `backend/chem/scoring.py` | Composite score: Tanimoto (w=0.25) + mechanism (w=0.30) + yield (w=0.20) + forward (w=0.15) - hazard (w=0.10); Platt-scaled sigmoid |
| `backend/graph/dag.py` | Build per-molecule retro tree → NetworkX DiGraph (avoids Cartesian product) |
| `backend/graph/cpm.py` | Critical Path Method: forward/backward pass over reaction DAG |
| `backend/graph/pert.py` | PERT: μ = (a+4m+b)/6, σ²=((b-a)/6)², cumulative pathway probability |
| `backend/graph/milp.py` | PuLP MILP: maximize log-yield − hazard − cost over pathway selection |
| `backend/db/schema.sql` | SQLite tables: `reactions` (with serialized FP BLOBs), `molecules` |
| `backend/db/loader.py` | Parse Figshare Reactron CSV → compute RDKit FPs → insert into SQLite |
| `backend/db/query.py` | Tanimoto k-NN search (k=10) against reactions table |
| `frontend/src/app/page.tsx` | Landing: IUPAC search with PubChem autocomplete + 12-molecule quick-pick gallery |
| `frontend/src/app/results/page.tsx` | Single-step results: target card + ranked precursor pairs with radar charts |
| `frontend/src/app/multistep/page.tsx` | Multi-step DAG: React Flow visualization with yield-colored edges, CPM/MILP overlays |
| `frontend/src/components/MoleculeCard.tsx` | Molecule properties card with inline SVG, functional group badges |
| `frontend/src/components/ScoreRadar.tsx` | Recharts radar chart for 5-axis score breakdown |
| `frontend/src/components/PrecursorCard.tsx` | Precursor pair display with reaction conditions + radar |
| `frontend/src/components/LoadingSkeleton.tsx` | Animated step indicators during API calls |
| `frontend/src/lib/api.ts` | API client: synthesize, multistep, PubChem autocomplete |

### Scoring formula
```
score(r) = 0.25·S_tanimoto + 0.30·S_mechanism + 0.20·S_yield + 0.15·S_forward − 0.10·S_hazard
probability = sigmoid(Platt_scale(score))
```

### Retrosynthesis priority
1. AiZynthFinder (MCTS, most accurate)
2. ReactionT5v2-retrosynthesis (T5 seq2seq beam search)
3. SMARTS template matching (14 named reactions, always-available fallback)

### Graph theory in use
- **Topological sort** (`networkx.topological_sort`): valid reaction ordering in synthesis DAG
- **CPM**: forward/backward pass → float → critical path (highlighted in UI)
- **PERT**: 3-point yield estimates → expected yield μ ± σ per pathway
- **MILP** (PuLP/CBC): multi-objective pathway selection across yield, safety, cost

## Deployment

- Frontend → Vercel
- Backend → Railway (use `rdkit-pypi` wheel — pip-installable, no conda needed)
- Database → SQLite file committed to Railway volume or repo (pre-processed)
- PubChem API calls are cached in the `molecules` table to avoid rate limits

## Data

- `data/reactron/` — Figshare Reactron dataset (download separately; not committed to git due to size)
- Pre-processed SQLite DB (`backend/db/reactions.db`) should be committed after running `loader.py`
