import React, { useState } from "react";
import type { RetrievedChunk } from "../../types/schema";
import { ConfidenceBadge } from "./ConfidenceBadge";

function cosineToPct(score: number): number {
  // Cosine similarity in [-1,1]; clamp to [0,1] for a friendly percentage.
  return Math.round(Math.max(0, Math.min(1, score)) * 100);
}

function HighlightableChunk({ chunk }: { chunk: RetrievedChunk }) {
  const [highlighted, setHighlighted] = useState(false);
  return (
    <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
        <span className="font-mono">{chunk.docId}</span>
        {chunk.page != null && <span>· Page {chunk.page}</span>}
        <span>· cosine {cosineToPct(chunk.score)}%</span>
        <button
          className="ml-auto rounded bg-gray-100 px-2 py-0.5 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700"
          onClick={() => setHighlighted((v) => !v)}
        >
          {highlighted ? "Hide highlight" : "Highlight in source"}
        </button>
      </div>
      <p
        className={
          highlighted
            ? "bg-yellow-200/70 dark:bg-yellow-500/30 rounded px-1"
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
}: {
  chunks: RetrievedChunk[];
}) {
  const [open, setOpen] = useState(true);
  if (!chunks || chunks.length === 0) {
    return (
      <div className="text-xs text-red-600 dark:text-red-400">
        ⚠ No source evidence retrieved.
      </div>
    );
  }
  // Highest-similarity chunk determines the badge level shown in the header.
  const top = chunks.reduce((a, b) => (b.score > a.score ? b : a), chunks[0]);
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700">
      <button
        className="flex w-full items-center justify-between px-3 py-2 text-sm font-medium"
        onClick={() => setOpen((v) => !v)}
      >
        <span>
          Sources & Citations ({chunks.length})
        </span>
        <span className="flex items-center gap-2">
          <ConfidenceBadge
            level={
              top.score >= 0.85
                ? "High"
                : top.score >= 0.5
                  ? "Medium"
                  : "Low"
            }
            score={top.score}
          />
          <span className="text-xs text-gray-400">{open ? "▾" : "▸"}</span>
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
