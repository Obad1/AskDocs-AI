import React, { useCallback, useState } from "react";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { summarize, type RoleProfile } from "../../lib/llm/engine";

const PROFILES: RoleProfile[] = [
  "General",
  "Student",
  "Researcher",
  "Executive",
  "Legal",
];

const LEVEL_LABEL = ["", "Level 1 · 1-line", "Level 2 · 3-paragraph", "Level 3 · Outline"];

export function SummaryGranularitySlider({
  sourceText,
}: {
  /** Text to summarize. Falls back to the first ingested document. */
  sourceText?: string;
}) {
  const { state } = useModelEngine();
  const { ws, setSummaryGranularity } = useWorkspace();

  const [profile, setProfile] = useState<RoleProfile>("General");
  const [busy, setBusy] = useState(false);
  const [output, setOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    const text = sourceText ?? Object.values(ws.documents)[0] ?? "";
    if (!text) {
      setError("No document text available to summarize.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const out = await summarize(text, ws.summary_granularity, profile, {
        backend: state.active_backend,
        modelId: state.llm_model,
      });
      setOutput(out);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [sourceText, ws.documents, ws.summary_granularity, profile, state]);

  return (
    <div className="rounded-lg border border-gray-200 p-3 dark:border-gray-700">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium">Summarize</span>
        <select
          className="rounded-md border border-gray-300 px-2 py-1 text-xs dark:border-gray-600 dark:bg-gray-900"
          value={profile}
          onChange={(e) => setProfile(e.target.value as RoleProfile)}
        >
          {PROFILES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>

      <label className="block text-xs text-gray-500">
        Granularity: {LEVEL_LABEL[ws.summary_granularity]}
      </label>
      <input
        type="range"
        min={1}
        max={3}
        step={1}
        value={ws.summary_granularity}
        onChange={(e) => setSummaryGranularity(Number(e.target.value))}
        className="w-full"
      />

      <button
        className="mt-2 w-full rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-50"
        onClick={run}
        disabled={busy}
      >
        {busy ? "Generating…" : "Generate summary"}
      </button>

      {error && (
        <div className="mt-2 rounded-md border border-red-300 bg-red-50 p-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}
      {output && (
        <div className="mt-3 whitespace-pre-wrap rounded-md bg-gray-50 p-3 text-sm dark:bg-gray-800/60">
          {output}
        </div>
      )}
    </div>
  );
}
