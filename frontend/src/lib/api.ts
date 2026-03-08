const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types mirroring backend Pydantic models
// ---------------------------------------------------------------------------

export interface MoleculeProperties {
  smiles: string;
  mw: number;
  logp: number;
  tpsa: number;
  hbd: number;
  hba: number;
  rotatable_bonds: number;
  num_rings: number;
  num_aromatic_rings: number;
  functional_groups: string[];
  electrophilic_sites: number[];
  nucleophilic_sites: number[];
  svg: string;
}

export interface ScoreBreakdown {
  tanimoto: number;
  mechanism: number;
  yield_score: number;
  hazard: number;
  forward: number;
}

export interface PrecursorPair {
  reactant_a: MoleculeProperties;
  reactant_b: MoleculeProperties | null;
  reaction_name: string;
  conditions: Record<string, string | number>;
  probability: number;
  composite_score: number;
  score_breakdown: ScoreBreakdown;
  source: string;
}

export interface SynthesizeResponse {
  smiles: string;
  properties: MoleculeProperties;
  precursor_pairs: PrecursorPair[];
}

export interface DAGEdge {
  source: string;
  target: string;
  data: {
    reaction_name?: string;
    mu?: number;
    isCritical?: boolean;
    isOptimal?: boolean;
    [key: string]: unknown;
  };
}

export interface DAGNode {
  id: string;
  data: Record<string, unknown>;
}

export interface MultistepResponse {
  smiles: string;
  dag: { nodes: DAGNode[]; edges: DAGEdge[] };
  critical_path: string[];
  total_probability: number;
  total_std: number;
  optimal_pathway_edges: string[][];
  milp_status: string;
  topo_order: string[];
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

export async function synthesize(
  iupacName: string,
  maxCandidates = 3
): Promise<SynthesizeResponse> {
  const res = await fetch(`${API_BASE}/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ iupac_name: iupacName, max_candidates: maxCandidates }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Synthesis failed");
  }
  return res.json();
}

export async function multistep(
  iupacName: string,
  maxSteps = 3,
  maxCandidatesPerStep = 2
): Promise<MultistepResponse> {
  const res = await fetch(`${API_BASE}/multistep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      iupac_name: iupacName,
      max_steps: maxSteps,
      max_candidates_per_step: maxCandidatesPerStep,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Multistep synthesis failed");
  }
  return res.json();
}

export async function autocompleteIUPAC(query: string): Promise<string[]> {
  if (query.length < 2) return [];
  try {
    const res = await fetch(
      `https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/${encodeURIComponent(query)}/json?limit=8`
    );
    if (!res.ok) return [];
    const data = await res.json();
    return data?.dictionary_terms?.compound ?? [];
  } catch {
    return [];
  }
}
