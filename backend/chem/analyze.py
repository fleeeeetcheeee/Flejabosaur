"""
Molecular analysis using RDKit.
Extracts atom/bond features, molecular descriptors, ECFP4 fingerprint,
Gasteiger partial charges, and functional group matches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdPartialCharges
from rdkit.Chem.Draw import rdMolDraw2D


# Common functional groups as SMARTS patterns — precompiled at module load
_FUNCTIONAL_GROUP_DEFS: dict[str, str] = {
    "alcohol": "[OX2H]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[CX3](=O)([#6])[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[CX3](=O)[NX3]",
    "amine_primary": "[NX3H2]",
    "amine_secondary": "[NX3H1]([#6])[#6]",
    "amine_tertiary": "[NX3]([#6])([#6])[#6]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "aromatic": "c1ccccc1",
    "halide": "[F,Cl,Br,I]",
    "nitrile": "[CX2]#N",
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "epoxide": "[C1OC1]",
    "sulfide": "[#16X2]([#6])[#6]",
    "phosphate": "[PX4](=O)([OX2H])([OX2H])[OX2]",
}

# Precompile SMARTS patterns once (avoids recompilation per molecule)
FUNCTIONAL_GROUPS: dict[str, Chem.rdchem.Mol] = {}
for _fg_name, _fg_smarts in _FUNCTIONAL_GROUP_DEFS.items():
    _pat = Chem.MolFromSmarts(_fg_smarts)
    if _pat is not None:
        FUNCTIONAL_GROUPS[_fg_name] = _pat

# Pauling electronegativities for oxidation state calculation
_ELECTRONEGATIVITIES: dict[int, float] = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}


@dataclass
class AtomFeatures:
    idx: int
    atomic_num: int
    symbol: str
    formal_charge: int
    hybridization: str
    is_aromatic: bool
    num_hs: int
    degree: int
    in_ring: bool
    gasteiger_charge: float
    oxidation_state: int


@dataclass
class BondFeatures:
    begin_idx: int
    end_idx: int
    bond_type: str
    is_aromatic: bool
    is_conjugated: bool
    stereo: str


@dataclass
class MolecularAnalysis:
    smiles: str
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    num_rings: int
    num_aromatic_rings: int
    atoms: list[AtomFeatures] = field(default_factory=list)
    bonds: list[BondFeatures] = field(default_factory=list)
    functional_groups: list[str] = field(default_factory=list)
    ecfp4: list[int] = field(default_factory=list)          # non-zero bit indices
    electrophilic_sites: list[int] = field(default_factory=list)   # atom indices
    nucleophilic_sites: list[int] = field(default_factory=list)
    svg: str = ""


def analyze(smiles: str) -> MolecularAnalysis:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # FIX: Compute Gasteiger charges on the final heavy-atom molecule directly.
    # Previously charges were computed on Hs-included mol then RemoveHs() created
    # a new mol object where the _GasteigerCharge properties didn't transfer.
    mol_noH = Chem.RemoveHs(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol_noH)

    # Molecular descriptors
    mw = Descriptors.MolWt(mol_noH)
    logp = Descriptors.MolLogP(mol_noH)
    tpsa = Descriptors.TPSA(mol_noH)
    hbd = rdMolDescriptors.CalcNumHBD(mol_noH)
    hba = rdMolDescriptors.CalcNumHBA(mol_noH)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol_noH)
    rings = rdMolDescriptors.CalcNumRings(mol_noH)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol_noH)

    # Atom features — collect charges first for percentile-based site detection
    atom_features = []
    charges: list[float] = []
    for atom in mol_noH.GetAtoms():
        charge = float(atom.GetPropsAsDict().get("_GasteigerCharge", 0.0))
        if math.isnan(charge):
            charge = 0.0
        charges.append(charge)

        oxstate = _oxidation_state(atom)
        af = AtomFeatures(
            idx=atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            symbol=atom.GetSymbol(),
            formal_charge=atom.GetFormalCharge(),
            hybridization=str(atom.GetHybridization()),
            is_aromatic=atom.GetIsAromatic(),
            num_hs=atom.GetTotalNumHs(),
            degree=atom.GetDegree(),
            in_ring=atom.IsInRing(),
            gasteiger_charge=round(charge, 4),
            oxidation_state=oxstate,
        )
        atom_features.append(af)

    # FIX: Percentile-based electrophilic/nucleophilic site detection.
    # Previous fixed thresholds (±0.1) missed sites in molecules with
    # uniformly distributed charges.
    elec_sites: list[int] = []
    nucl_sites: list[int] = []
    if charges:
        sorted_charges = sorted(charges)
        n = len(sorted_charges)
        elec_threshold = sorted_charges[-max(1, n // 5)]  # top 20%
        nucl_threshold = sorted_charges[max(0, n // 5 - 1)]  # bottom 20%
        for i, charge in enumerate(charges):
            if charge >= elec_threshold and charge > 0:
                elec_sites.append(i)
            if charge <= nucl_threshold and charge < 0:
                nucl_sites.append(i)

    # Bond features
    bond_features = []
    for bond in mol_noH.GetBonds():
        bf = BondFeatures(
            begin_idx=bond.GetBeginAtomIdx(),
            end_idx=bond.GetEndAtomIdx(),
            bond_type=str(bond.GetBondType()),
            is_aromatic=bond.GetIsAromatic(),
            is_conjugated=bond.GetIsConjugated(),
            stereo=str(bond.GetStereo()),
        )
        bond_features.append(bf)

    # Functional groups (using precompiled SMARTS)
    fg_present = []
    for name, pattern in FUNCTIONAL_GROUPS.items():
        if mol_noH.HasSubstructMatch(pattern):
            fg_present.append(name)

    # ECFP4 Morgan fingerprint (non-zero bit indices)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_noH, radius=2, nBits=2048)
    ecfp4_bits = list(fp.GetOnBits())

    # 2D SVG depiction
    svg = _mol_to_svg(mol_noH)

    return MolecularAnalysis(
        smiles=Chem.MolToSmiles(mol_noH),
        mw=round(mw, 3),
        logp=round(logp, 3),
        tpsa=round(tpsa, 3),
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rot,
        num_rings=rings,
        num_aromatic_rings=arom_rings,
        atoms=atom_features,
        bonds=bond_features,
        functional_groups=fg_present,
        ecfp4=ecfp4_bits,
        electrophilic_sites=elec_sites,
        nucleophilic_sites=nucl_sites,
        svg=svg,
    )


def _oxidation_state(atom) -> int:
    """
    Electronegativity-based oxidation state estimation (Allen's algorithm).

    For each bond, the more electronegative atom is assigned all the bonding
    electrons. The oxidation state = group valence electrons − assigned electrons.
    This correctly handles cases like ketone C (→ +2) and carboxylic acid C (→ +3).
    """
    atomic_num = atom.GetAtomicNum()
    own_en = _ELECTRONEGATIVITIES.get(atomic_num, 2.0)

    # Count electrons this atom "owns" from its bonds
    owned_electrons = 0
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        other_en = _ELECTRONEGATIVITIES.get(other.GetAtomicNum(), 2.0)
        bond_order = int(bond.GetBondTypeAsDouble())
        if own_en > other_en:
            # This atom is more electronegative → it owns all bonding electrons
            owned_electrons += 2 * bond_order
        elif own_en == other_en:
            # Equal electronegativity → split the electrons
            owned_electrons += bond_order  # half of 2*bond_order

    # Lone-pair / non-bonding electrons: total valence electrons − bonding electrons
    # For this atom's contribution to bonds where it's LESS electronegative, it
    # contributed electrons but doesn't own them in the oxidation state model.
    # Hydrogen bonds: each H bonded counts as owned (H is less electronegative than C,N,O,etc.)
    owned_electrons += atom.GetTotalNumHs()  # H electrons "owned" by atom (only if atom more EN than H)
    if own_en <= _ELECTRONEGATIVITIES.get(1, 2.20):
        # Atom is not more EN than H, so it doesn't own H electrons
        owned_electrons -= atom.GetTotalNumHs()

    # Group valence electrons for the neutral atom
    pt = Chem.GetPeriodicTable()
    group_valence = pt.GetNOuterElecs(atomic_num)

    # Oxidation state = group_valence − owned_electrons + formal_charge
    return group_valence - owned_electrons + atom.GetFormalCharge()


def _mol_to_svg(mol, width: int = 300, height: int = 200) -> str:
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def smiles_to_fingerprint_bits(smiles: str) -> list[int]:
    """Return ECFP4 bit indices for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return list(fp.GetOnBits())


def tanimoto(bits_a: list[int], bits_b: list[int]) -> float:
    """Tanimoto similarity between two sets of fingerprint bit indices."""
    set_a, set_b = set(bits_a), set(bits_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
