"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { synthesize, type SynthesizeResponse } from "@/lib/api";
import MoleculeCard from "@/components/MoleculeCard";
import PrecursorCard from "@/components/PrecursorCard";
import LoadingSkeleton from "@/components/LoadingSkeleton";

function ResultsContent() {
  const params = useSearchParams();
  const name = params.get("name") ?? "";
  const [data, setData] = useState<SynthesizeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const prevName = useRef(name);

  useEffect(() => {
    if (!name) return;
    // Only reset state when name changes
    if (prevName.current !== name) {
      prevName.current = name;
    }
    let cancelled = false;
    const fetchData = async () => {
      try {
        const result = await synthesize(name);
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

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row gap-6 items-start">
        <MoleculeCard mol={data.properties} label="Target" />
        <div className="flex-1 space-y-2">
          <h1 className="text-2xl font-bold">{name}</h1>
          <p className="font-mono text-sm text-muted break-all">{data.smiles}</p>
          <Link href="/" className="text-accent-light text-xs underline">
            New search
          </Link>
        </div>
      </div>

      {/* Precursor pairs */}
      <h2 className="text-lg font-semibold">
        Top precursor pairs ({data.precursor_pairs.length})
      </h2>
      {data.precursor_pairs.length === 0 ? (
        <p className="text-muted text-sm">
          No retrosynthetic candidates found for this molecule.
        </p>
      ) : (
        <div className="space-y-4">
          {data.precursor_pairs.map((pair, i) => (
            <PrecursorCard key={i} pair={pair} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<LoadingSkeleton />}>
      <ResultsContent />
    </Suspense>
  );
}
