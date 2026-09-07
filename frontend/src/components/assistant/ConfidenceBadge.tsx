import React from "react";
import type { CONFIDENCE_LEVEL } from "../../types/schema";

const STYLES: Record<
  CONFIDENCE_LEVEL,
  { label: string; className: string; dot: string }
> = {
  High: {
    label: "High confidence",
    className:
      "bg-green-100 text-green-800 border-green-300 dark:bg-green-900/30 dark:text-green-300 dark:border-green-700",
    dot: "bg-green-500",
  },
  Medium: {
    label: "Medium confidence",
    className:
      "bg-orange-100 text-orange-800 border-orange-300 border-dashed dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-700",
    dot: "bg-orange-500",
  },
  Low: {
    label: "Low confidence — missing evidence",
    className:
      "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-300 dark:border-red-700",
    dot: "bg-red-500",
  },
};

export function ConfidenceBadge({
  level,
  score,
}: {
  level: CONFIDENCE_LEVEL;
  score?: number;
}) {
  const s = STYLES[level];
  return (
    <span
      title={s.label}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${s.className}`}
    >
      <span className={`h-2 w-2 rounded-full ${s.dot}`} />
      {level}
      {typeof score === "number" && (
        <span className="opacity-70">({score.toFixed(2)})</span>
      )}
    </span>
  );
}
