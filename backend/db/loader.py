"""
Load reaction datasets into SQLite.

Supports:
  - Figshare Reactron CSV (columns: reactant1, reactant2, product, reaction_type, yield, ...)
  - Generic SMILES reaction CSV with columns: reactants_smiles, products_smiles

Usage:
  python db/loader.py --input ../data/reactron/reactions.csv --db reactions.db
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_DEFAULT = Path(__file__).parent / "reactions.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def smiles_to_fp_json(smiles: str) -> str | None:
    """Compute ECFP4 fingerprint and return as JSON list of on-bit indices."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return json.dumps(list(fp.GetOnBits()))


def reaction_fp_json(r1_bits: list[int], r2_bits: list[int], prod_bits: list[int]) -> str:
    """XOR reaction fingerprint: (r1 | r2) XOR product."""
    reactant_bits = set(r1_bits) | set(r2_bits)
    xor_bits = list(reactant_bits.symmetric_difference(set(prod_bits)))
    return json.dumps(xor_bits)


def load_csv(path: Path, conn: sqlite3.Connection, source: str = "figshare_reactron") -> int:
    """
    Load a CSV reaction file into the reactions table.
    Tries multiple column name conventions.
    """
    import csv

    loaded = 0
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Stream rows instead of loading all into memory
        for row in reader:
            # Flexible column name detection
            r1 = (row.get("reactant1") or row.get("reactant1_smiles") or
                   row.get("reactant") or row.get("REACTANT1") or "").strip()
            r2 = (row.get("reactant2") or row.get("reactant2_smiles") or
                   row.get("REACTANT2") or "").strip()
            prod = (row.get("product") or row.get("product_smiles") or
                    row.get("PRODUCT") or "").strip()
            rtype = (row.get("reaction_type") or row.get("type") or row.get("TYPE") or "").strip()
            yld_str = (row.get("yield") or row.get("YIELD") or "").strip()
            temp_str = (row.get("temperature_c") or row.get("temp") or "").strip()
            solvent = (row.get("solvent") or row.get("SOLVENT") or "").strip()
            catalyst = (row.get("catalyst") or row.get("CATALYST") or "").strip()
            mechanism = (row.get("mechanism") or row.get("mechanism_desc") or "").strip()

            if not r1 or not prod:
                skipped += 1
                continue

            # Compute fingerprints
            r1_fp = smiles_to_fp_json(r1)
            r2_fp = smiles_to_fp_json(r2) if r2 else None
            prod_fp = smiles_to_fp_json(prod)

            if r1_fp is None or prod_fp is None:
                skipped += 1
                continue

            r1_bits = json.loads(r1_fp)
            r2_bits = json.loads(r2_fp) if r2_fp else []
            prod_bits = json.loads(prod_fp)
            rxn_fp = reaction_fp_json(r1_bits, r2_bits, prod_bits)

            yld = float(yld_str) if yld_str else None
            if yld is not None and yld > 1.0:
                yld /= 100.0   # convert % to fraction

            temp = float(temp_str) if temp_str else None

            conn.execute(
                """INSERT INTO reactions
                   (reactant1_smiles, reactant2_smiles, product_smiles, reaction_type,
                    mechanism_desc, yield, temperature_c, solvent, catalyst, source,
                    reactant1_fp, reactant2_fp, product_fp, reaction_fp)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r1, r2 or None, prod, rtype or None, mechanism or None,
                 yld, temp, solvent or None, catalyst or None, source,
                 r1_fp, r2_fp, prod_fp, rxn_fp),
            )
            loaded += 1

            if loaded % 1000 == 0:
                conn.commit()
                logger.info("  %d rows inserted...", loaded)

    conn.commit()
    logger.info("Done: %d loaded, %d skipped", loaded, skipped)
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Load reaction dataset into SQLite")
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--db", default=str(DB_DEFAULT), help="SQLite DB path")
    parser.add_argument("--source", default="figshare_reactron")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    init_db(conn)
    n = load_csv(Path(args.input), conn, source=args.source)
    conn.close()
    logger.info("Loaded %d reactions into %s", n, args.db)


if __name__ == "__main__":
    main()
