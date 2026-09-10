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

const LEVEL_LABEL = ["", "Short (1 line)", "Medium (3 paragraphs)", "Outline"];

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

  const sourceTitle = sourceText
    ? "Provided text"
    : (() => {
        const t = Object.values(ws.documents)[0] ?? "";
        return t ? `${t.replace(/\s+/g, " ").trim().slice(0, 48)}…` : "";
      })();

  return (
    <div className="surface-card p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <label htmlFor="summary-profile" className="text-sm font-medium">
          Summarize for
        </label>
        <select
          id="summary-profile"
          aria-label="Audience for the summary"
          className="field px-2 py-1.5 text-xs"
          value={profile}
          disabled={busy}
          onChange={(e) => setProfile(e.target.value as RoleProfile)}
        >
          {PROFILES.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </div>

      <label htmlFor="summary-granularity" className="block text-xs text-[var(--fg-muted)]">
        Granularity: {LEVEL_LABEL[ws.summary_granularity]}
      </label>
      <input
        id="summary-granularity"
        type="range"
        min={1}
        max={3}
        step={1}
        value={ws.summary_granularity}
        disabled={busy}
        onChange={(e) => setSummaryGranularity(Number(e.target.value))}
        className="w-full accent-[var(--accent)]"
      />

      {sourceTitle && (
        <div className="mt-1 truncate text-xs text-[var(--fg-muted)]">
          Source: {sourceTitle}
        </div>
      )}

      <button
        className="btn-primary mt-2 w-full px-3 py-2 text-sm"
        onClick={run}
        disabled={busy}
      >
        {busy ? "Generating…" : "Generate summary"}
      </button>

      {error && (
        <div className="mt-2 rounded-md border border-[var(--danger)] bg-[var(--danger-soft)] p-2 text-xs text-[var(--danger-fg)]">
          {error}
        </div>
      )}
      {output && (
        <div className="mt-3 whitespace-pre-wrap rounded-md bg-[var(--bg-sunken)] p-3 text-sm">
          {output}
        </div>
      )}
    </div>
  );
}
