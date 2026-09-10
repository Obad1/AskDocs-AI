// TextCleanerModal.tsx — split raw/cleaned view with toggles (spec §3.1).
import React, { useMemo, useState } from "react";
import {
  cleanText,
  DEFAULT_CLEAN_OPTIONS,
  type CleanOptions,
} from "../../lib/parsing/textCleaner";

interface Props {
  rawText: string;
  onAccept: (cleanText: string) => void;
  onClose: () => void;
}

const TOGGLES: { key: keyof CleanOptions; label: string }[] = [
  { key: "normalizeNFKC", label: "NFKC Normalize" },
  { key: "stripHeadersFooters", label: "Strip Headers/Footers" },
  { key: "stripLineNumbers", label: "Strip Line Numbers" },
  { key: "stripPageNumbers", label: "Strip Page Numbers" },
  { key: "stripBrokenUnicode", label: "Strip Broken Unicode" },
];

export function TextCleanerModal({ rawText, onAccept, onClose }: Props) {
  const [opts, setOpts] = useState<CleanOptions>(DEFAULT_CLEAN_OPTIONS);

  const result = useMemo(() => cleanText(rawText, opts), [rawText, opts]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="surface-card flex h-[80vh] w-full max-w-5xl flex-col">
        <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
          <h2 className="text-lg font-semibold text-[var(--fg)]">
            Text Cleaner
          </h2>
          <button
            className="btn-ghost rounded px-2 py-1"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="flex flex-wrap gap-3 border-b border-[var(--border)] px-4 py-2 text-sm">
          {TOGGLES.map((t) => (
            <label key={t.key} className="flex items-center gap-1 text-[var(--fg)]">
              <input
                type="checkbox"
                checked={opts[t.key]}
                onChange={(e) =>
                  setOpts((p) => ({ ...p, [t.key]: e.target.checked }))
                }
              />
              {t.label}
            </label>
          ))}
        </div>

        <div className="grid flex-1 grid-cols-2 gap-2 overflow-hidden p-2">
          <div className="flex flex-col">
            <span className="px-1 text-xs font-medium text-[var(--fg-muted)]">Raw</span>
            <textarea
              readOnly
              value={rawText}
              className="field h-full w-full resize-none p-2 font-mono text-xs"
            />
          </div>
          <div className="flex flex-col">
            <span className="px-1 text-xs font-medium text-[var(--fg-muted)]">
              Cleaned {result.junkDetected && <em>(junk detected)</em>}
            </span>
            <textarea
              readOnly
              value={result.clean}
              className="h-full w-full resize-none rounded border border-[var(--success)] bg-[var(--success-soft)] p-2 font-mono text-xs text-[var(--success-fg)]"
            />
          </div>
        </div>

        <div className="border-t border-[var(--border)] px-4 py-2 text-xs text-[var(--fg-muted)]">
          Removed:{" "}
          {result.removed.length ? result.removed.join(", ") : "none"}
        </div>

        <div className="flex justify-end gap-2 border-t border-[var(--border)] px-4 py-3">
          <button
            className="btn-ghost px-3 py-1.5 text-sm"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="btn-primary px-3 py-1.5 text-sm"
            onClick={() => onAccept(result.clean)}
          >
            Accept Cleaned
          </button>
        </div>
      </div>
    </div>
  );
}
