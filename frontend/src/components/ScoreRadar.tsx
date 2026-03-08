"use client";

import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { ScoreBreakdown } from "@/lib/api";

interface Props {
  breakdown: ScoreBreakdown;
}

export default function ScoreRadar({ breakdown }: Props) {
  const data = [
    { axis: "Tanimoto", value: breakdown.tanimoto },
    { axis: "Mechanism", value: breakdown.mechanism },
    { axis: "Yield", value: breakdown.yield_score },
    { axis: "Hazard", value: breakdown.hazard },
    { axis: "Forward", value: breakdown.forward },
  ];

  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="#2a2b35" />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 10, fill: "#a1a1aa" }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
        <Radar
          dataKey="value"
          stroke="#818cf8"
          fill="#6366f1"
          fillOpacity={0.3}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1a1b23",
            border: "1px solid #2a2b35",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(v) => typeof v === "number" ? v.toFixed(3) : String(v)}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
