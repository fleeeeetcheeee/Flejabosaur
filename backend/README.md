# Flejabosaur — Backend

FastAPI retrosynthesis API for HackTJ 13.0.

## Quick start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/health` — you should see `{"status": "ok"}`.

## Trying the full ML version

The server runs with base dependencies only, falling back to SMARTS template
matching for retrosynthesis and default scoring when ML models are absent.

To enable the full ML-powered version with ReactionT5v2 (HuggingFace seq2seq
retrosynthesis, yield prediction, and forward validation):

```bash
cd backend
pip install -r requirements.txt -r requirements-full.txt
uvicorn main:app --reload --port 8000
```

This unlocks:
- **ReactionT5v2-retrosynthesis** — T5 seq2seq beam search (replaces SMARTS templates as primary fallback after AiZynthFinder)
- **ReactionT5v2-yield** — ML yield prediction (replaces k-NN database averaging)
- **ReactionT5v2-forward** — forward reaction validation (new 5th scoring axis, weight 0.15)

Models are downloaded from HuggingFace on first use (~1–2 GB). Alternatively,
set `HF_TOKEN` to use the HuggingFace Inference API instead of local models.

> **Note:** `aizynthfinder` is commented out in `requirements-full.txt` because
> it has a complex install (conda recommended). See the
> [AiZynthFinder docs](https://molecularai.github.io/aizynthfinder/) if you
> want to enable the MCTS tier.

### Retrosynthesis priority (3-tier fallback)

| Priority | Source | Requires |
|----------|--------|----------|
| 1 | AiZynthFinder MCTS | `aizynthfinder` (conda) |
| 2 | ReactionT5v2 beam search | `torch` + `transformers` (requirements-full.txt) |
| 3 | SMARTS template matching | Base deps only (always available) |

## Database

The reaction database (`db/reactions.db`) is pre-populated from the Figshare
Reactron dataset. To rebuild it from a fresh CSV:

```bash
python db/loader.py --input ../data/reactron/reactions.csv
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/synthesize` | IUPAC name → top-N precursor pairs for the final step |
| `POST` | `/multistep` | Full multi-step DAG with CPM, PERT, and MILP analysis |
| `GET`  | `/molecule/{smiles}` | Molecular properties + SVG for a SMILES string |
| `GET`  | `/health` | Liveness check |

### `/synthesize` example

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"iupac_name": "aspirin", "max_candidates": 3}'
```

## Running tests

```bash
cd backend
pytest
```

## Changes from main branch

### Dependency & import fixes (from PR #5)
- Removed `aizynthfinder` as a hard dependency (complex install, now optional)
- Updated `rdkit-pypi==2022.9.5` → `>=2023.9.1`
- Fixed `graph/dag.py` invalid relative import (`from ..chem.retrosynthesis` → `from chem.retrosynthesis`)
- Added `sys.path` setup in `main.py` so bare subpackage imports resolve from any working directory
- Created `requirements-full.txt` with optional ML deps (torch, transformers)

### Fletcher branch improvements (merged here)
- **Scoring**: Added forward validation as 5th scoring axis (weight 0.15) using ReactionT5v2-forward; updated weights to 0.25/0.25/0.25/0.10/0.15; added Platt scaling for calibrated sigmoid probabilities
- **DAG builder**: Fixed Cartesian product bug — now builds per-molecule retro tree instead of flat candidate list
- **Molecular analysis**: Fixed Gasteiger charge bug (charges computed on wrong mol object); added percentile-based electrophilic/nucleophilic site detection; precompiled SMARTS patterns
- **IUPAC**: Added RDKit canonicalization of all SMILES; improved logging
- **CPM/PERT/MILP**: Refactored to work with new per-molecule DAG structure
- **DB loader**: Added progress logging and validation improvements
- **Frontend**: Added full Next.js frontend with results page, multistep DAG visualization, radar score charts, and PubChem autocomplete

## Deployment (Railway)

Use `rdkit-pypi` (pip-installable wheel — no conda needed). Set the start
command to:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
