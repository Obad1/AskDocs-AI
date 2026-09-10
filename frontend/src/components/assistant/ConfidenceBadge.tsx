import React from "react";
import type { CONFIDENCE_LEVEL } from "../../types/schema";

const STYLES: Record<
  CONFIDENCE_LEVEL,
  { label: string; className: string; dot: string }
> = {
  High: {
    label: "High confidence",
    className:
      "bg-[var(--success-soft)] text-[var(--success-fg)] border-[var(--success)]",
    dot: "bg-[var(--success)]",
  },
  Medium: {
    label: "Medium confidence",
    className:
      "bg-[var(--warn-soft)] text-[var(--warn-fg)] border-[var(--warn)] border-dashed",
    dot: "bg-[var(--warn)]",
  },
  Low: {
    label: "Low confidence — missing evidence",
    className:
      "bg-[var(--danger-soft)] text-[var(--danger-fg)] border-[var(--danger)]",
    dot: "bg-[var(--danger)]",
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
      {s.label}
      {typeof score === "number" && (
        <span className="opacity-70">{Math.round(score * 100)}%</span>
      )}
    </span>
  );
}
