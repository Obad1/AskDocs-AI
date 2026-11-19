import React, { useState } from "react";
import type { CONFIDENCE_LEVEL, RetrievedChunk } from "../../types/schema";
import { useWorkspace } from "../../context/WorkspaceContext";

/**
 * Retrieval scores come from Reciprocal Rank Fusion or raw cross-encoder logits.
 * They are not probabilities. Present auditable passages, never made-up
 * "match percentages" or unsupported confidence claims.
 */
function SourcePassage({ chunk }: { chunk: RetrievedChunk }) {
  const [expanded, setExpanded] = useState(false);
  const { ws, requestDocFocus } = useWorkspace();
  const name = ws.doc_names?.[chunk.docId] || chunk.docId;

  const jump = () => {
    requestDocFocus({
      docId: chunk.docId,
      page: chunk.page ?? 1,
      text: chunk.text,
    });
  };

  return (
    <article className="rounded-xl border border-[var(--border)] bg-[var(--bg)] p-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--fg-muted)]">
        <span title={name} className="max-w-full truncate font-semibold text-[var(--fg)]">{name}</span>
        {chunk.page != null && <span>Page {chunk.page}</span>}
        <button
          type="button"
          className="btn-ghost ml-auto px-3 py-1.5 text-xs"
          onClick={jump}
          aria-label={"Open source document " + name + (chunk.page != null ? " on page " + chunk.page : "")}
        >
          View source
        </button>
      </div>
      <p className={expanded
        ? "mt-3 whitespace-pre-wrap break-words rounded-lg bg-[var(--warn-soft)] p-2.5 text-sm leading-relaxed text-[var(--warn-fg)]"
        : "mt-3 whitespace-pre-wrap break-words text-sm leading-relaxed text-[var(--fg)]"}>
        {chunk.text}
      </p>
      <button type="button" className="mt-2 text-xs font-medium text-[var(--fg-muted)] underline-offset-4 hover:underline" onClick={() => setExpanded((v) => !v)}>
        {expanded ? "Remove highlight" : "Highlight excerpt"}
      </button>
    </article>
  );
}

export function CitationDrawer({
  chunks,
  confidence: _confidence,
}: {
  chunks: RetrievedChunk[];
  confidence?: CONFIDENCE_LEVEL;
}) {
  const [open, setOpen] = useState(true);
  if (!chunks?.length) {
    return (
      <div role="note" className="rounded-lg border border-dashed border-[var(--border)] px-3 py-2 text-xs leading-relaxed text-[var(--fg-muted)]">
        No supporting document passages were retrieved. Treat this answer as unverified.
      </div>
    );
  }

  return (
    <section data-tour="citations" className="overflow-hidden rounded-xl border border-[var(--border)]">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left text-sm font-semibold text-[var(--fg)]"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span>{chunks.length} {chunks.length === 1 ? "source passage" : "source passages"}</span>
        <span className="text-xs font-normal text-[var(--fg-muted)]">
          Verify in source {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-[var(--border)] p-2.5">
          <p className="px-1 text-xs text-[var(--fg-muted)]">
            Ranked passages are evidence candidates, not verified facts or confidence percentages.
          </p>
          {chunks.map((chunk) => <SourcePassage key={chunk.chunkId} chunk={chunk} />)}
        </div>
      )}
    </section>
  );
}
