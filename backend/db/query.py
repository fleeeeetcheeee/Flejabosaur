"""
Database query helpers.
Tanimoto k-NN search, molecule caching, reaction lookup.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from functools import lru_cache

DB_PATH = Path(__file__).parent / "reactions.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_db() -> None:
    """Initialize DB schema if not already present."""
    schema = (Path(__file__).parent / "schema.sql").read_text()
    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()


def tanimoto_knn(
    query_fp_bits: list[int],
    k: int = 10,
    reaction_type: str | None = None,
) -> list[dict]:
    """
    Return the k most similar reactions to query_fp_bits using Tanimoto similarity.
    Computes similarity in Python (no RDKit cartridge required for SQLite).
    """
    conn = get_connection()
    try:
        sql = "SELECT * FROM reactions WHERE reaction_fp IS NOT NULL"
        params: list = []
        if reaction_type:
            sql += " AND reaction_type = ?"
            params.append(reaction_type)
        # Limit scan to 50k rows for performance
        sql += " LIMIT 50000"

        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    query_set = set(query_fp_bits)
    scored = []
    for row in rows:
        try:
            db_bits = json.loads(row["reaction_fp"])
        except (TypeError, json.JSONDecodeError):
            continue
        db_set = set(db_bits)
        intersection = len(query_set & db_set)
        union = len(query_set | db_set)
        sim = intersection / union if union > 0 else 0.0
        scored.append((sim, dict(row)))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, row in scored[:k]:
        row["tanimoto"] = round(sim, 4)
        # Deserialize fingerprints for downstream use
        for fp_col in ("reactant1_fp", "reactant2_fp", "product_fp", "reaction_fp"):
            if row.get(fp_col):
                try:
                    row[fp_col] = json.loads(row[fp_col])
                except (TypeError, json.JSONDecodeError):
                    row[fp_col] = []
        results.append(row)
    return results


def cache_molecule(
    smiles: str,
    iupac_name: str | None = None,
    mw: float | None = None,
    logp: float | None = None,
    tpsa: float | None = None,
    hazard_class: int = 0,
    pubchem_cid: int | None = None,
    svg: str | None = None,
) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO molecules
           (smiles, iupac_name, mw, logp, tpsa, hazard_class, pubchem_cid, svg)
           VALUES (?,?,?,?,?,?,?,?)""",
        (smiles, iupac_name, mw, logp, tpsa, hazard_class, pubchem_cid, svg),
    )
    conn.commit()
    conn.close()


def get_cached_molecule(smiles: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM molecules WHERE smiles = ?", (smiles,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_cached_smiles(iupac_name: str) -> str | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT smiles FROM molecules WHERE iupac_name = ? LIMIT 1",
        (iupac_name.lower(),),
    ).fetchone()
    conn.close()
    return row["smiles"] if row else None
