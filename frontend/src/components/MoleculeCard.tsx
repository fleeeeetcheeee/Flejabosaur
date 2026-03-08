"use client";

import DOMPurify from "dompurify";
import type { MoleculeProperties } from "@/lib/api";

interface Props {
  mol: MoleculeProperties;
  label?: string;
}

export default function MoleculeCard({ mol, label }: Props) {
  return (
    <div className="bg-card border border-card-border rounded-xl p-4 flex flex-col gap-2">
      {label && (
        <span className="text-xs font-semibold text-accent-light uppercase tracking-wide">
          {label}
        </span>
      )}
      {mol.svg ? (
        <div
          className="w-full flex justify-center [&>svg]:max-w-[180px] [&>svg]:h-auto"
          dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(mol.svg, { USE_PROFILES: { svg: true } }) }}
        />
      ) : (
        <div className="h-24 flex items-center justify-center text-muted text-sm bg-card-border/20 rounded-lg">
          ⚗️ Structure unavailable
        </div>
      )}
      <p className="font-mono text-xs break-all text-muted">{mol.smiles}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <span>MW</span>
        <span className="text-right">{mol.mw.toFixed(1)}</span>
        <span>LogP</span>
        <span className="text-right">{mol.logp.toFixed(2)}</span>
        <span>TPSA</span>
        <span className="text-right">{mol.tpsa.toFixed(1)}</span>
        <span>H-donors</span>
        <span className="text-right">{mol.hbd}</span>
        <span>H-acceptors</span>
        <span className="text-right">{mol.hba}</span>
        <span>Rings</span>
        <span className="text-right">{mol.num_rings}</span>
      </div>
      {mol.functional_groups.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1">
          {mol.functional_groups.map((fg) => (
            <span
              key={fg}
              className="text-[10px] bg-accent/20 text-accent-light rounded px-1.5 py-0.5"
            >
              {fg}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
