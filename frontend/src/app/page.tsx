"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { autocompleteIUPAC } from "@/lib/api";

const COMMON_MOLECULES = [
  { name: "Aspirin", iupac: "2-acetoxybenzoic acid" },
  { name: "Ibuprofen", iupac: "2-(4-isobutylphenyl)propanoic acid" },
  { name: "Caffeine", iupac: "1,3,7-trimethylxanthine" },
  { name: "Paracetamol", iupac: "N-(4-hydroxyphenyl)acetamide" },
  { name: "Glucose", iupac: "D-glucose" },
  { name: "Lidocaine", iupac: "2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide" },
  { name: "Vanillin", iupac: "4-hydroxy-3-methoxybenzaldehyde" },
  { name: "Citric Acid", iupac: "2-hydroxypropane-1,2,3-tricarboxylic acid" },
  { name: "Nicotinamide", iupac: "pyridine-3-carboxamide" },
  { name: "Salicylic Acid", iupac: "2-hydroxybenzoic acid" },
  { name: "Benzocaine", iupac: "ethyl 4-aminobenzoate" },
  { name: "Coumarin", iupac: "2H-chromen-2-one" },
];

export default function Home() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleQueryChange(value: string) {
    setQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (value.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      const results = await autocompleteIUPAC(value);
      setSuggestions(results);
      setShowSuggestions(results.length > 0);
      setSelectedIdx(-1);
    }, 300);
  }

  function go(name: string, mode: "single" | "multistep" = "single") {
    const encoded = encodeURIComponent(name.trim());
    if (mode === "multistep") {
      router.push(`/multistep?name=${encoded}`);
    } else {
      router.push(`/results?name=${encoded}`);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!showSuggestions) {
      if (e.key === "Enter" && query.trim()) go(query);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const name = selectedIdx >= 0 ? suggestions[selectedIdx] : query;
      if (name.trim()) go(name);
    } else if (e.key === "Escape") {
      setShowSuggestions(false);
    }
  }

  return (
    <div className="flex flex-col items-center gap-12 pt-12">
      {/* Hero */}
      <div className="text-center space-y-3 max-w-xl">
        <h1 className="text-4xl font-bold tracking-tight">
          Retrosynthesis Explorer
        </h1>
        <p className="text-muted text-sm">
          Enter a molecule name to find the most probable precursor pairs for
          synthesis, powered by graph theory and ML-driven reaction analysis.
        </p>
      </div>

      {/* Search */}
      <div className="w-full max-w-lg relative">
        <input
          type="text"
          value={query}
          onChange={(e) => handleQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          placeholder="Enter IUPAC name (e.g. aspirin, caffeine...)"
          className="w-full px-4 py-3 rounded-xl bg-card border border-card-border text-foreground placeholder:text-muted focus:outline-none focus:border-accent transition"
        />

        {/* Autocomplete dropdown */}
        {showSuggestions && (
          <ul className="absolute z-40 w-full mt-1 bg-card border border-card-border rounded-xl overflow-hidden shadow-lg">
            {suggestions.map((s, i) => (
              <li
                key={s}
                onMouseDown={() => go(s)}
                className={`px-4 py-2 text-sm cursor-pointer transition ${
                  i === selectedIdx
                    ? "bg-accent/20 text-accent-light"
                    : "hover:bg-card-border/40"
                }`}
              >
                {s}
              </li>
            ))}
          </ul>
        )}

        {/* Action buttons */}
        <div className="flex gap-2 mt-3 justify-center">
          <button
            onClick={() => query.trim() && go(query)}
            className="px-5 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-light transition disabled:opacity-40"
            disabled={!query.trim()}
          >
            Single-step synthesis
          </button>
          <button
            onClick={() => query.trim() && go(query, "multistep")}
            className="px-5 py-2 rounded-lg border border-accent text-accent-light text-sm font-medium hover:bg-accent/10 transition disabled:opacity-40"
            disabled={!query.trim()}
          >
            Multi-step DAG
          </button>
        </div>
      </div>

      {/* Quick-pick gallery */}
      <div className="w-full max-w-3xl">
        <h2 className="text-sm font-semibold text-muted mb-3 text-center">
          Or try a common molecule
        </h2>
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
          {COMMON_MOLECULES.map((m) => (
            <button
              key={m.name}
              onClick={() => go(m.iupac)}
              className="px-3 py-2 rounded-lg bg-card border border-card-border text-xs font-medium hover:border-accent hover:text-accent-light transition text-center"
            >
              {m.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
