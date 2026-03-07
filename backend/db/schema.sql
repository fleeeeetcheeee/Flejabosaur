-- Flejabosaur SQLite schema

CREATE TABLE IF NOT EXISTS reactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    reactant1_smiles    TEXT NOT NULL,
    reactant2_smiles    TEXT,
    product_smiles      TEXT NOT NULL,
    reaction_type       TEXT,           -- e.g. "Suzuki", "Aldol", "DielsAlder"
    mechanism_desc      TEXT,
    yield               REAL,           -- 0.0-1.0
    temperature_c       REAL,
    solvent             TEXT,
    catalyst            TEXT,
    source              TEXT,           -- "figshare_reactron" | "uspto" | "manual"
    reactant1_fp        BLOB,           -- serialized Morgan fingerprint bit indices (JSON)
    reactant2_fp        BLOB,
    product_fp          BLOB,
    reaction_fp         BLOB            -- XOR reaction fingerprint for Tanimoto search
);

CREATE INDEX IF NOT EXISTS idx_reactions_type ON reactions(reaction_type);
CREATE INDEX IF NOT EXISTS idx_reactions_product ON reactions(product_smiles);

CREATE TABLE IF NOT EXISTS molecules (
    smiles              TEXT PRIMARY KEY,
    iupac_name          TEXT,
    mw                  REAL,
    logp                REAL,
    tpsa                REAL,
    hazard_class        INTEGER DEFAULT 0,  -- GHS 0=unknown, 1=low, 5=extreme
    pubchem_cid         INTEGER,
    svg                 TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);
