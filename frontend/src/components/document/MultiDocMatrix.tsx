// MultiDocMatrix.tsx — comparison table of ingested documents (spec §3.1).
// Uses useWorkspace() to read the workspace document state.
import React, { useMemo } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { FORMAT_TYPE } from "../../types/schema";

export function MultiDocMatrix() {
  const { ws } = useWorkspace();

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
      return {
        id,
        format,
        chunkCount,
        hashShort: hash.slice(0, 10) + (hash.length > 10 ? "…" : ""),
        conflict,
      };
    });
  }, [ws]);

  if (rows.length === 0) {
    return (
      <div className="p-4 text-sm text-gray-500">No documents ingested yet.</div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-left text-gray-500 dark:border-gray-700">
            <th className="px-3 py-2">Document</th>
            <th className="px-3 py-2">Format</th>
            <th className="px-3 py-2">Chunks</th>
            <th className="px-3 py-2">Hash</th>
            <th className="px-3 py-2">Conflict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.id}
              className="border-b border-gray-100 dark:border-gray-800"
            >
              <td className="px-3 py-2 font-mono text-xs text-gray-700 dark:text-gray-200">
                {r.id.slice(0, 16)}
              </td>
              <td className="px-3 py-2">{r.format}</td>
              <td className="px-3 py-2">{r.chunkCount}</td>
              <td className="px-3 py-2 font-mono text-xs text-gray-500">
                {r.hashShort}
              </td>
              <td className="px-3 py-2">
                {r.conflict ? (
                  <span className="rounded bg-red-100 px-2 py-0.5 text-xs text-red-700 dark:bg-red-900/40 dark:text-red-300">
                    duplicate
                  </span>
                ) : (
                  <span className="text-gray-400">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
