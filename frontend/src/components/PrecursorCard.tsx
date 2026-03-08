"use client";

import type { PrecursorPair } from "@/lib/api";
import MoleculeCard from "./MoleculeCard";
import ScoreRadar from "./ScoreRadar";

interface Props {
  pair: PrecursorPair;
  rank: number;
}

function conditionBadge(key: string, value: string | number) {
  return (
    <span
      key={key}
      className="text-[10px] bg-card border border-card-border rounded px-1.5 py-0.5"
    >
      {key}: {value}
    </span>
  );
}

function sourceLabel(source: string): string {
  if (source === "template") return "SMARTS Template";
  if (source === "reactiont5") return "ML Predicted";
  return source;
}

export default function PrecursorCard({ pair, rank }: Props) {
  const pct = (pair.probability * 100).toFixed(1);

  return (
    <div className="bg-card/50 border border-card-border rounded-2xl p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-accent-light">#{rank}</span>
          <span className="font-semibold">{pair.reaction_name.replace(/_/g, " ")}</span>
          <span className="text-[10px] bg-accent/20 text-accent-light rounded-full px-2 py-0.5">
            {sourceLabel(pair.source)}
          </span>
        </div>
        <span className={`text-lg font-bold ${
          pair.probability >= 0.7 ? "text-green" : pair.probability >= 0.4 ? "text-yellow" : "text-red"
        }`}>{pct}%</span>
      </div>

      {/* Conditions */}
      {Object.keys(pair.conditions).length > 0 && (
        <div className="flex flex-wrap gap-1">
          {Object.entries(pair.conditions).map(([k, v]) => conditionBadge(k, v))}
        </div>
      )}

      {/* Reactants + Radar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MoleculeCard mol={pair.reactant_a} label="Reactant A" />
        {pair.reactant_b && (
          <MoleculeCard mol={pair.reactant_b} label="Reactant B" />
        )}
        <div className="flex flex-col items-center justify-center">
          <span className="text-xs text-muted mb-1">Score Breakdown</span>
          <ScoreRadar breakdown={pair.score_breakdown} />
          <span className="text-xs text-muted mt-1">
            Composite: {pair.composite_score.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}
