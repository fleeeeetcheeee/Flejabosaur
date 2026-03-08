"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { multistep, type MultistepResponse } from "@/lib/api";
import LoadingSkeleton from "@/components/LoadingSkeleton";

function yieldColor(mu: number): string {
  if (mu >= 0.8) return "#22c55e";
  if (mu >= 0.5) return "#eab308";
  return "#ef4444";
}

function buildFlowGraph(data: MultistepResponse) {
  const dagNodes = data.dag.nodes;
  const dagEdges = data.dag.edges;

  // Layout: place nodes in topological order columns
  const topoIndex = new Map<string, number>();
  data.topo_order.forEach((id, i) => topoIndex.set(id, i));

  const nodes: Node[] = dagNodes.map((n) => {
    const idx = topoIndex.get(n.id) ?? 0;
    return {
      id: n.id,
      position: { x: idx * 250, y: 0 },
      data: {
        label: n.id.length > 20 ? n.id.slice(0, 20) + "..." : n.id,
      },
      style: {
        background: "#1a1b23",
        color: "#e4e4e7",
        border: data.critical_path.includes(n.id)
          ? "2px solid #ef4444"
          : "1px solid #2a2b35",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: 11,
        maxWidth: 200,
        wordBreak: "break-all" as const,
      },
    };
  });

  // Auto-layout: assign y positions to avoid overlap
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  const colNodes = new Map<number, string[]>();
  nodes.forEach((n) => {
    const col = Math.round(n.position.x / 250);
    if (!colNodes.has(col)) colNodes.set(col, []);
    colNodes.get(col)!.push(n.id);
  });
  colNodes.forEach((ids) => {
    ids.forEach((id, i) => {
      const node = nodeById.get(id);
      if (node) node.position.y = i * 120;
    });
  });

  const edges: Edge[] = dagEdges.map((e, i) => {
    const mu = e.data?.mu ?? 0.65;
    const isCritical = e.data?.isCritical ?? false;
    const isOptimal = e.data?.isOptimal ?? false;
    return {
      id: `e-${i}`,
      source: e.source,
      target: e.target,
      label: `${e.data?.reaction_name?.replace(/_/g, " ") ?? "Unknown"} (${(mu * 100).toFixed(0)}%)`,
      labelStyle: { fontSize: 9, fill: "#a1a1aa" },
      style: {
        stroke: yieldColor(mu),
        strokeWidth: isCritical ? 3 : 1.5,
        strokeDasharray: isOptimal ? "6 3" : undefined,
      },
      animated: isOptimal,
    };
  });

  return { nodes, edges };
}

function MultistepContent() {
  const params = useSearchParams();
  const name = params.get("name") ?? "";
  const [data, setData] = useState<MultistepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const prevName = useRef(name);

  useEffect(() => {
    if (!name) return;
    if (prevName.current !== name) {
      prevName.current = name;
    }
    let cancelled = false;
    const fetchData = async () => {
      try {
        const result = await multistep(name);
        if (!cancelled) setData(result);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchData();
    return () => { cancelled = true; };
  }, [name]);

  if (!name) {
    return <p className="text-muted text-center py-20">No molecule specified.</p>;
  }

  if (loading) return <LoadingSkeleton />;

  if (error) {
    return (
      <div className="text-center py-20 space-y-2">
        <p className="text-red font-semibold">Error</p>
        <p className="text-muted text-sm">{error}</p>
        <Link href="/" className="text-accent-light text-sm underline">
          Go back
        </Link>
      </div>
    );
  }

  if (!data) return null;

  const { nodes, edges } = buildFlowGraph(data);

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold">Multi-step Synthesis DAG</h1>
        <p className="font-mono text-sm text-muted break-all">{data.smiles}</p>
        <div className="flex gap-4 text-xs text-muted">
          <span>
            Total probability:{" "}
            <strong className="text-green">
              {(data.total_probability * 100).toFixed(1)}%
            </strong>
          </span>
          <span>Std: {data.total_std.toFixed(4)}</span>
          <span>MILP: {data.milp_status}</span>
        </div>
        <Link href="/" className="text-accent-light text-xs underline">
          New search
        </Link>
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-muted">
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-green inline-block" /> High yield (&gt;80%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-yellow inline-block" /> Moderate (50-80%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-red inline-block" /> Low (&lt;50%)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-4 h-0.5 border-t-2 border-red inline-block" /> Critical path
        </span>
        <span className="flex items-center gap-1">
          <span className="w-4 h-0.5 border-t-2 border-dashed border-accent inline-block" />{" "}
          MILP optimal
        </span>
      </div>

      {/* DAG */}
      <div className="w-full h-[500px] bg-card border border-card-border rounded-xl overflow-hidden">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#2a2b35" gap={20} />
          <Controls
            style={{ background: "#1a1b23", borderColor: "#2a2b35" }}
          />
          <MiniMap
            style={{ background: "#1a1b23" }}
            nodeColor="#6366f1"
            maskColor="rgba(0,0,0,0.5)"
          />
        </ReactFlow>
      </div>

      {/* Critical path */}
      {data.critical_path.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold mb-2">Critical Path</h2>
          <div className="flex flex-wrap gap-1 items-center">
            {data.critical_path.map((node, i) => (
              <span key={i} className="flex items-center gap-1">
                <span className="font-mono text-xs bg-card border border-card-border rounded px-2 py-0.5 max-w-[200px] truncate">
                  {node}
                </span>
                {i < data.critical_path.length - 1 && (
                  <span className="text-muted text-xs">&rarr;</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Topological order */}
      <div>
        <h2 className="text-sm font-semibold mb-2">Topological Order</h2>
        <ol className="list-decimal list-inside text-xs text-muted space-y-0.5">
          {data.topo_order.map((node, i) => (
            <li key={i} className="font-mono truncate max-w-md">
              {node}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

export default function MultistepPage() {
  return (
    <Suspense fallback={<LoadingSkeleton />}>
      <MultistepContent />
    </Suspense>
  );
}
