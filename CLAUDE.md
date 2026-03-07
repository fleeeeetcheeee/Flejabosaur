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
| `backend/chem/retrosynthesis.py` | AiZynthFinder wrapper; fallback: SMARTS template matching for common named reactions |
| `backend/chem/scoring.py` | Composite score: Tanimoto (w=0.35) + mechanism (w=0.30) + yield (w=0.25) - hazard (w=0.10); PuLP MILP selects top-N |
| `backend/graph/dag.py` | Build NetworkX DiGraph from AiZynthFinder synthesis tree |
| `backend/graph/cpm.py` | Critical Path Method: forward/backward pass over reaction DAG |
| `backend/graph/pert.py` | PERT: μ = (a+4m+b)/6, σ²=((b-a)/6)², cumulative pathway probability |
| `backend/graph/milp.py` | PuLP MILP: maximize log-yield − hazard − cost over pathway selection |
| `backend/db/schema.sql` | SQLite tables: `reactions` (with serialized FP BLOBs), `molecules` |
| `backend/db/loader.py` | Parse Figshare Reactron CSV → compute RDKit FPs → insert into SQLite |
| `backend/db/query.py` | Tanimoto k-NN search (k=10) against reactions table |
| `frontend/app/page.tsx` | Landing: IUPAC input + 2D structure preview |
| `frontend/app/results/page.tsx` | Top-3 precursor pair cards with probability bars |
| `frontend/app/synthesis/page.tsx` | Interactive React Flow synthesis DAG |
| `frontend/components/MoleculeViewer.tsx` | smiles-drawer wrapper for 2D structure rendering |
| `frontend/components/SynthesisDAG.tsx` | React Flow DAG with critical path highlighting |

### Scoring formula
```
score(r) = 0.35·S_tanimoto + 0.30·S_mechanism + 0.25·S_yield − 0.10·S_hazard
probability = sigmoid(score)
```

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
