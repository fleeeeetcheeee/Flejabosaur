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


# Common functional groups as SMARTS patterns
FUNCTIONAL_GROUPS: dict[str, str] = {
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

    # Sanitize and add Hs for charge calculation (then remove for descriptors)
    mol = Chem.AddHs(mol)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    mol_noH = Chem.RemoveHs(mol)

    # Molecular descriptors
    mw = Descriptors.MolWt(mol_noH)
    logp = Descriptors.MolLogP(mol_noH)
    tpsa = Descriptors.TPSA(mol_noH)
    hbd = rdMolDescriptors.CalcNumHBD(mol_noH)
    hba = rdMolDescriptors.CalcNumHBA(mol_noH)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol_noH)
    rings = rdMolDescriptors.CalcNumRings(mol_noH)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol_noH)

    # Atom features
    atom_features = []
    elec_sites = []
    nucl_sites = []
    for atom in mol_noH.GetAtoms():
        charge = float(atom.GetPropsAsDict().get("_GasteigerCharge", 0.0))
        if math.isnan(charge):
            charge = 0.0
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
        # Electrophilic: positive Gasteiger charge (electron-poor)
        if charge > 0.1:
            elec_sites.append(atom.GetIdx())
        # Nucleophilic: negative Gasteiger charge (electron-rich)
        if charge < -0.1:
            nucl_sites.append(atom.GetIdx())

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

    # Functional groups
    fg_present = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol_noH.HasSubstructMatch(pattern):
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
    """Estimate oxidation state from formal charge and bond types."""
    valence = atom.GetTotalValence()
    bonds_to_heteroatom = sum(
        1 for bond in atom.GetBonds()
        if bond.GetOtherAtom(atom).GetAtomicNum() in (7, 8, 9, 16, 17, 35, 53)
    )
    return atom.GetFormalCharge() + bonds_to_heteroatom - (valence - atom.GetDegree())


def _mol_to_svg(mol, width: int = 300, height: int = 200) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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
