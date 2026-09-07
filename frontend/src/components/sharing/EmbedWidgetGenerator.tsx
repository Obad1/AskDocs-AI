import React, { useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";

// Embed widget generator (spec §3.8). Produces an iframe snippet for Notion,
// Canvas, or any external page — no third-party embed service.
export default function EmbedWidgetGenerator({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { ws } = useWorkspace();
  const [width, setWidth] = useState(800);
  const [height, setHeight] = useState(600);

  const origin =
    typeof window !== "undefined" ? window.location.origin : "";
  const src = `${origin}/share/${ws.active_workspace}`;
  const snippet = `<iframe src="${src}" width="${width}" height="${height}" style="border:0"></iframe>`;

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-[520px] rounded-lg bg-[var(--bg-elevated)] p-4 shadow-xl">
        <h2 className="mb-2 text-lg font-semibold">Embed Widget</h2>
        <label className="block text-xs">Width</label>
        <input
          type="number"
          value={width}
          onChange={(e) => setWidth(parseInt(e.target.value) || 800)}
          className="mb-2 w-full rounded bg-[var(--bg)] px-2 py-1"
        />
        <label className="block text-xs">Height</label>
        <input
          type="number"
          value={height}
          onChange={(e) => setHeight(parseInt(e.target.value) || 600)}
          className="mb-2 w-full rounded bg-[var(--bg)] px-2 py-1"
        />
        <textarea
          readOnly
          value={snippet}
          className="h-24 w-full rounded bg-[var(--bg)] p-2 font-mono text-xs"
        />
        <div className="mt-3 flex justify-end gap-2">
          <button
            onClick={() => navigator.clipboard?.writeText(snippet)}
            className="rounded bg-[var(--accent)] px-3 py-1 text-[var(--accent-fg)]"
          >
            Copy
          </button>
          <button onClick={onClose} className="rounded bg-[var(--bg)] px-3 py-1">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
