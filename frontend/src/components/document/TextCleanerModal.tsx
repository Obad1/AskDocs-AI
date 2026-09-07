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
      <div className="flex h-[80vh] w-full max-w-5xl flex-col rounded-lg bg-white shadow-xl dark:bg-gray-900">
        <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">
            Text Cleaner
          </h2>
          <button
            className="rounded px-2 py-1 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="flex flex-wrap gap-3 border-b border-gray-200 px-4 py-2 text-sm dark:border-gray-700">
          {TOGGLES.map((t) => (
            <label key={t.key} className="flex items-center gap-1 text-gray-700 dark:text-gray-200">
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
            <span className="px-1 text-xs font-medium text-gray-500">Raw</span>
            <textarea
              readOnly
              value={rawText}
              className="h-full w-full resize-none rounded border border-gray-300 bg-gray-50 p-2 font-mono text-xs text-gray-700 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200"
            />
          </div>
          <div className="flex flex-col">
            <span className="px-1 text-xs font-medium text-gray-500">
              Cleaned {result.junkDetected && <em>(junk detected)</em>}
            </span>
            <textarea
              readOnly
              value={result.clean}
              className="h-full w-full resize-none rounded border border-green-300 bg-green-50 p-2 font-mono text-xs text-gray-800 dark:border-green-700 dark:bg-green-950 dark:text-gray-100"
            />
          </div>
        </div>

        <div className="border-t border-gray-200 px-4 py-2 text-xs text-gray-500 dark:border-gray-700">
          Removed:{" "}
          {result.removed.length ? result.removed.join(", ") : "none"}
        </div>

        <div className="flex justify-end gap-2 border-t border-gray-200 px-4 py-3 dark:border-gray-700">
          <button
            className="rounded px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
            onClick={() => onAccept(result.clean)}
          >
            Accept Cleaned
          </button>
        </div>
      </div>
    </div>
  );
}
