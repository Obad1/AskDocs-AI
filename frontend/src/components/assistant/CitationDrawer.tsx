import React, { useState } from "react";
import type { CONFIDENCE_LEVEL, RetrievedChunk } from "../../types/schema";
import { useWorkspace } from "../../context/WorkspaceContext";
import { ConfidenceBadge } from "./ConfidenceBadge";

function cosineToPct(score: number): number {
  // Cosine similarity in [-1,1]; maps to a friendly relevance percentage.
  const pct = Math.round(Math.max(0, Math.min(1, score)) * 100);
  return pct;
}

function HighlightableChunk({ chunk }: { chunk: RetrievedChunk }) {
  const [highlighted, setHighlighted] = useState(false);
  const { requestDocFocus } = useWorkspace();

  const jump = () => {
    if (chunk.page == null) return;
    requestDocFocus({ docId: chunk.docId, page: chunk.page, text: chunk.text });
  };

  return (
    <div className="rounded-md border border-[var(--border)] p-3">
      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-[var(--fg-muted)]">
        <span className="font-mono">{chunk.docId}</span>
        {chunk.page != null && <span>· Page {chunk.page}</span>}
        <span>· Match {cosineToPct(chunk.score)}%</span>
        <button
          className="btn-ghost ml-auto px-2 py-0.5"
          onClick={() => setHighlighted((v) => !v)}
        >
          {highlighted ? "Hide quote highlight" : "Highlight quote"}
        </button>
        <button
          className="btn-ghost px-2 py-0.5 disabled:opacity-40"
          onClick={jump}
          disabled={chunk.page == null}
          aria-label={`Open ${chunk.docId} on page ${chunk.page}`}
        >
          View in source
        </button>
      </div>
      <p
        className={
          highlighted
            ? "rounded bg-[var(--warn-soft)] px-1 text-[var(--warn-fg)]"
            : "text-sm"
        }
      >
        {chunk.text}
      </p>
    </div>
  );
}

export function CitationDrawer({
  chunks,
  confidence,
}: {
  chunks: RetrievedChunk[];
  /** Precomputed level from the message (keeps badge and bubble consistent). */
  confidence?: CONFIDENCE_LEVEL;
}) {
  const [open, setOpen] = useState(true);
  if (!chunks || chunks.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-[var(--border)] px-3 py-2 text-xs text-[var(--fg-muted)]">
        No source evidence retrieved — this answer isn’t grounded in your documents.
      </div>
    );
  }
  // Highest-similarity chunk determines the badge level shown in the header
  // when the message didn't already compute one.
  const top = chunks.reduce((a, b) => (b.score > a.score ? b : a), chunks[0]);
  const level: CONFIDENCE_LEVEL =
    confidence ??
    (top.score >= 0.85
      ? "High"
      : top.score >= 0.5
        ? "Medium"
        : "Low");
  return (
    <div data-tour="citations" className="rounded-lg border border-[var(--border)]">
      <button
        className="flex w-full items-center justify-between px-3 py-2 text-sm font-medium"
        onClick={() => setOpen((v) => !v)}
      >
        <span>
          Sources & Citations ({chunks.length})
        </span>
        <span className="flex items-center gap-2">
          <ConfidenceBadge level={level} score={confidence ? undefined : top.score} />
          <span className="text-xs text-[var(--fg-muted)]">{open ? "▾" : "▸"}</span>
        </span>
      </button>
      {open && (
        <div className="space-y-2 px-3 pb-3">
          {chunks.map((c) => (
            <HighlightableChunk key={c.chunkId} chunk={c} />
          ))}
        </div>
      )}
    </div>
  );
}
