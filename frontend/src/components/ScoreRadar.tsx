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

interface RadarDataPoint {
  axis: string;
  value: number;
  raw: number;
}

export default function ScoreRadar({ breakdown }: Props) {
  const data: RadarDataPoint[] = [
    { axis: "Tanimoto", value: breakdown.tanimoto, raw: breakdown.tanimoto },
    { axis: "Mechanism", value: breakdown.mechanism, raw: breakdown.mechanism },
    { axis: "Yield", value: breakdown.yield_score, raw: breakdown.yield_score },
    { axis: "Safety", value: 1 - breakdown.hazard, raw: breakdown.hazard },
    { axis: "Forward", value: breakdown.forward, raw: breakdown.forward },
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
          formatter={(v, _name, entry) => {
            const point = entry?.payload as RadarDataPoint | undefined;
            const display = point?.raw !== undefined ? point.raw : v;
            return typeof display === "number" ? display.toFixed(3) : String(display);
          }}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
