import React from "react";
import Drawer from "./Drawer";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { MODE } from "../../types/schema";

const GRANULARITY_LABELS = ["", "Short (1 line)", "Medium (3 paragraphs)", "Outline"];

/**
 * Answers & retrieval settings (RAG params). Moved out of the top bar so the
 * default view has a single line of chrome: Strict/Expanded mode, confidence
 * threshold and summary depth now live behind a single "Answer settings" entry.
 */
export default function SettingsDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { ws, setActiveMode, setConfidenceThreshold, setSummaryGranularity } =
    useWorkspace();

  return (
    <Drawer open={open} title="Answer settings" onClose={onClose}>
      <section className="mb-5">
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-[var(--fg-muted)]">
          Answer mode
        </h3>
        <div className="grid grid-cols-2 gap-1 rounded-md border border-[var(--border)] p-1">
          {(
            [
              ["StrictDocumentOnly", "Documents only"],
              ["ExpandedAI", "Docs + general knowledge"],
            ] as [MODE, string][]
          ).map(([val, label]) => (
            <button
              key={val}
              onClick={() => setActiveMode(val)}
              aria-pressed={ws.active_mode === val}
              className={`rounded-md px-2 py-2 text-left text-xs ${
                ws.active_mode === val
                  ? "bg-[var(--accent)] text-[var(--accent-fg)]"
                  : "text-[var(--fg-muted)] hover:bg-[var(--bg-sunken)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </section>

      <section className="mb-5">
        <div className="mb-1 flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--fg-muted)]">
            Confidence threshold
          </h3>
          <span className="font-mono text-sm text-[var(--fg)]">
            {Math.round(ws.confidence_threshold * 100)}%
          </span>
        </div>
        <p className="mb-2 text-xs text-[var(--fg-muted)]">
          Retrieval cut-off — raise it to keep answers grounded in only the
          most relevant passages.
        </p>
        <input
          aria-label="Confidence threshold"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={ws.confidence_threshold}
          onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
          className="w-full accent-[var(--accent)]"
        />
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-[var(--fg-muted)]">
          Summary depth
        </h3>
        <p className="mb-2 text-xs text-[var(--fg-muted)]">
          Default length used when summarizing documents.
        </p>
        <label className="block text-xs text-[var(--fg-muted)]">
          {GRANULARITY_LABELS[ws.summary_granularity]}
        </label>
        <input
          aria-label="Summary granularity"
          type="range"
          min={1}
          max={3}
          step={1}
          value={ws.summary_granularity}
          onChange={(e) => setSummaryGranularity(Number(e.target.value))}
          className="w-full accent-[var(--accent)]"
        />
      </section>
    </Drawer>
  );
}