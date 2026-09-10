// MultiDocMatrix.tsx — comparison table of ingested documents (spec §3.1).
// Uses useWorkspace() to read the workspace document state.
import React, { useMemo } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { FORMAT_TYPE } from "../../types/schema";

export function MultiDocMatrix() {
  const { ws, setActiveDocId } = useWorkspace();

  const rows = useMemo(() => {
    const docs = Object.keys(ws.documents);
    const hashCounts = new Map<string, number>();
    for (const id of docs) {
      const h = ws.doc_hashes[id];
      if (h) hashCounts.set(h, (hashCounts.get(h) || 0) + 1);
    }
    return docs.map((id) => {
      const hash = ws.doc_hashes[id] ?? "";
      const format = (ws.doc_formats[id] ?? "PDF") as FORMAT_TYPE;
      const chunkCount = (ws.doc_chunks[id] ?? []).length;
      const conflict = hash ? (hashCounts.get(hash) ?? 0) > 1 : false;
      // Best-effort readable title: first few words of the document text.
      const text = ws.documents[id] ?? "";
      const title =
        text
          .replace(/\s+/g, " ")
          .trim()
          .slice(0, 60) || id;
      return {
        id,
        title,
        format,
        chunkCount,
        conflict,
      };
    });
  }, [ws]);

  if (rows.length === 0) {
    return (
      <div className="p-4 text-sm text-[var(--fg-muted)]">No documents ingested yet.</div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--border)] text-left text-[var(--fg-muted)]">
            <th className="px-3 py-2">Document</th>
            <th className="px-3 py-2">Format</th>
            <th className="px-3 py-2">Chunks</th>
            <th className="px-3 py-2">Conflict</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.id}
              className="border-b border-[var(--border)] hover:bg-[var(--accent-soft)]"
            >
              <td className="px-3 py-2 text-[var(--fg)]">
                <button
                  className="block max-w-xs truncate text-left font-medium underline-offset-2 hover:underline"
                  title="Open this document"
                  onClick={() => setActiveDocId(r.id)}
                >
                  {r.title}
                </button>
              </td>
              <td className="px-3 py-2">{r.format}</td>
              <td className="px-3 py-2">{r.chunkCount}</td>
              <td className="px-3 py-2">
                {r.conflict ? (
                  <span className="rounded bg-[var(--danger-soft)] px-2 py-0.5 text-xs text-[var(--danger-fg)]">
                    duplicate
                  </span>
                ) : (
                  <span className="text-[var(--fg-muted)]">—</span>
                )}
              </td>
              <td className="px-3 py-2 text-right">
                <button
                  className="btn-ghost px-2 py-0.5 text-xs"
                  onClick={() => setActiveDocId(r.id)}
                >
                  Open →
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
