import React from "react";
import type { CONFIDENCE_LEVEL } from "../../types/schema";

/**
 * The retrieval pipeline currently uses raw RRF/logit ranking scores, not
 * calibrated confidence probabilities. Suppress misleading percentages and
 * make the uncertainty explicit until a labeled evaluation is available.
 */
export function ConfidenceBadge({
  level,
  score: _score,
}: {
  level: CONFIDENCE_LEVEL;
  score?: number;
}) {
  const hasMatches = level !== "Low";
  return (
    <span
      title="Retrieval signals are uncalibrated: check the cited document before relying on an answer."
      className="inline-flex items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--bg)] px-2.5 py-1 text-xs text-[var(--fg-muted)]"
    >
      <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
      {hasMatches ? "Check source evidence" : "Evidence not confirmed"}
    </span>
  );
}
