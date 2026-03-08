"use client";

import { useEffect, useState } from "react";

const STEPS = [
  "Resolving IUPAC name...",
  "Analyzing molecular structure...",
  "Searching retrosynthetic routes...",
  "Scoring candidates...",
];

export default function LoadingSkeleton() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((s) => (s < STEPS.length - 1 ? s + 1 : s));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center py-20 gap-6">
      {/* Spinner */}
      <div className="w-12 h-12 border-4 border-card-border border-t-accent rounded-full animate-spin" />

      {/* Step indicators */}
      <div className="flex flex-col gap-2 w-64">
        {STEPS.map((label, i) => (
          <div key={label} className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                i < step
                  ? "bg-green"
                  : i === step
                    ? "bg-accent-light animate-pulse"
                    : "bg-card-border"
              }`}
            />
            <span
              className={`text-sm transition-colors duration-300 ${
                i <= step ? "text-foreground" : "text-muted"
              }`}
            >
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
