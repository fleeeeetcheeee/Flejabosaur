"""
IUPAC name → SMILES conversion.
Primary: PubChem REST API
Fallback: OPSIN (Java CLI, if available on PATH)
"""
import subprocess
import httpx


PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"


async def iupac_to_smiles(name: str) -> str:
    """Convert an IUPAC name to a canonical isomeric SMILES string."""
    smiles = await _pubchem_lookup(name)
    if smiles:
        return smiles
    smiles = _opsin_lookup(name)
    if smiles:
        return smiles
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
    except Exception:
        pass
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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
