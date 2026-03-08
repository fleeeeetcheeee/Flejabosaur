# Flejabosaur — Backend

FastAPI retrosynthesis API for HackTJ 13.0.

## Quick start

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/health` — you should see `{"status": "ok"}`.

## Optional ML dependencies

The server runs with base dependencies only, falling back to SMARTS template
matching for retrosynthesis and default scoring when ML models are absent.

To enable ReactionT5v2 (HuggingFace seq2seq retrosynthesis + yield prediction):

```bash
pip install -r requirements.txt -r requirements-full.txt
```

> **Note:** `aizynthfinder` is commented out in `requirements-full.txt` because
> it has a complex install (conda recommended). See the
> [AiZynthFinder docs](https://molecularai.github.io/aizynthfinder/) if you
> want to enable the MCTS tier.

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

## Deployment (Railway)

Use `rdkit-pypi` (pip-installable wheel — no conda needed).  Set the start
command to:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
