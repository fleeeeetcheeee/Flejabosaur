"""
IUPAC name → SMILES conversion.
Primary: PubChem REST API
Fallback: OPSIN (Java CLI, if available on PATH)

All results are canonicalized via RDKit before returning to ensure
consistent SMILES representation downstream.
"""
import logging
import subprocess

import httpx
from rdkit import Chem

logger = logging.getLogger(__name__)

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"


def _canonicalize(smiles: str) -> str:
    """Canonicalize SMILES via RDKit; return original if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    return smiles


async def iupac_to_smiles(name: str) -> str:
    """Convert an IUPAC name to a canonical isomeric SMILES string."""
    smiles = await _pubchem_lookup(name)
    if smiles:
        return _canonicalize(smiles)
    smiles = _opsin_lookup(name)
    if smiles:
        return _canonicalize(smiles)
    raise ValueError(f"Could not resolve IUPAC name: {name!r}")


async def _pubchem_lookup(name: str) -> str | None:
    url = PUBCHEM_URL.format(name=name)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES")
        else:
            logger.warning("PubChem returned status %d for %r", resp.status_code, name)
    except Exception as exc:
        logger.warning("PubChem lookup failed for %r: %s", name, exc)
    return None


def _opsin_lookup(name: str) -> str | None:
    """Try OPSIN (Open Parser for Systematic IUPAC Nomenclature) via subprocess."""
    try:
        result = subprocess.run(
            ["opsin", "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        if output and result.returncode == 0:
            return output
        logger.info("OPSIN failed for %r (returncode=%d)", name, result.returncode)
    except FileNotFoundError:
        logger.debug("OPSIN not installed, skipping fallback.")
    except subprocess.TimeoutExpired:
        logger.warning("OPSIN timed out for %r", name)
    return None
