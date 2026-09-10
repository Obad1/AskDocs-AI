import React, { useRef, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useUserProfile } from "../../context/UserProfileContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useIngest } from "../../lib/parsing/ingestClient";
import type { MODE } from "../../types/schema";

const THEMES = ["auto", "slate", "obsidian", "sepia", "graphite"] as const;
const THEME_LABELS: Record<(typeof THEMES)[number], string> = {
  auto: "Auto (system)",
  slate: "Slate",
  obsidian: "Obsidian",
  sepia: "Sepia",
  graphite: "Graphite",
};

const TIER_LABELS: Record<string, string> = {
  Tier0_Minimal: "Tier 0 · Minimal",
  Tier1_Standard: "Tier 1 · Standard",
  Tier2_Performance: "Tier 2 · Performance",
  Tier3_Workstation: "Tier 3 · Workstation",
};

interface TopNavigationProps {
  onOpenBenchmark: () => void;
  onOpenTour: () => void;
  onOpenDemo: () => void;
}

/**
 * Top bar (spec §3.2 / §3.9): workspace switcher, Strict/Expanded mode toggle,
 * confidence threshold, zen toggle, hardware benchmark entry, theme switch.
 * Hidden while zen mode is active (see App).
 */
export default function TopNavigation({
  onOpenBenchmark,
  onOpenTour,
  onOpenDemo,
}: TopNavigationProps) {
  const { ws, setActiveMode, setConfidenceThreshold, setZenMode } =
    useWorkspace();
  const { profile, update } = useUserProfile();
  const { state: engine } = useModelEngine();
  const ingestFile = useIngest();
  const fileRef = useRef<HTMLInputElement>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const mode: MODE = ws.active_mode;
  const tierLabel = TIER_LABELS[engine.hardware_tier] ?? engine.hardware_tier;

  return (
    <>
      <header className="flex items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2 text-sm">
        <div className="flex items-center gap-2 font-semibold text-[var(--fg)]">
          <span className="brand-mark">◆</span>
          <span>AskDocs AI</span>
        </div>

      {/* Workspace switcher */}
      <label className="flex items-center gap-1 text-[var(--fg-muted)]">
        <span className="sr-only">Workspace</span>
        <select
          aria-label="Active workspace"
          value={ws.active_workspace}
          onChange={(e) => {
            // Single local workspace in v2.0; switch is a no-op placeholder
            // kept for future multi-workspace support.
            void e.target.value;
          }}
          className="field px-2 py-1"
        >
          <option value={ws.active_workspace}>
            {ws.active_workspace === "default" ? "My Workspace" : ws.active_workspace}
          </option>
        </select>
      </label>

      {/* Mode toggle: Strict / Expanded */}
      <div
        role="group"
        aria-label="Answer mode"
        className="flex overflow-hidden rounded border border-[var(--border)]"
      >
        {(
          [
            ["StrictDocumentOnly", "Strict"],
            ["ExpandedAI", "Expanded"],
          ] as [MODE, string][]
        ).map(([val, label]) => (
          <button
            key={val}
            aria-pressed={mode === val}
            onClick={() => setActiveMode(val)}
            className={`px-3 py-1 ${
              mode === val
                ? "bg-[var(--accent)] text-[var(--accent-fg)]"
                : "bg-[var(--bg)] text-[var(--fg-muted)]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Confidence threshold */}
      <label className="flex items-center gap-2 text-[var(--fg-muted)]">
        <span>Confidence ≥</span>
        <input
          aria-label="Confidence threshold"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={ws.confidence_threshold}
          onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
          className="w-28 accent-[var(--accent)]"
        />
        <span className="w-10 tabular-nums text-[var(--fg)]">
          {Math.round(ws.confidence_threshold * 100)}%
        </span>
      </label>

      <div className="ml-auto flex items-center gap-2">
        {/* Local document ingestion (zero login). */}
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.docx,.pptx,.epub,.txt,.md"
          multiple
          className="hidden"
          onChange={async (e) => {
            const files = Array.from(e.target.files ?? []);
            if (!files.length) return;
            setIngesting(true);
            setIngestError(null);
            try {
              for (const f of files) await ingestFile(f);
            } catch (err) {
              const msg =
                err instanceof Error
                  ? err.message
                  : "That file could not be read. Try a PDF, DOCX, PPTX, EPUB, TXT, or MD file.";
              setIngestError(msg);
            } finally {
              setIngesting(false);
              if (fileRef.current) fileRef.current.value = "";
            }
          }}
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={ingesting}
          data-tour="ingest"
          className="btn-ghost px-3 py-1"
        >
          {ingesting ? "Ingesting…" : "+ Add"}
        </button>

        {/* Theme switch */}
        <label className="flex items-center gap-1 text-[var(--fg-muted)]">
          <span className="sr-only">Theme</span>
          <select
            aria-label="Theme"
            value={profile.theme}
            onChange={(e) =>
              update({ theme: e.target.value as (typeof THEMES)[number] })
            }
            className="field px-2 py-1"
          >
            {THEMES.map((t) => (
              <option key={t} value={t}>
                {THEME_LABELS[t]}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={onOpenDemo}
          data-tour="demo"
          className="btn-ghost px-3 py-1"
        >
          Demo
        </button>
        <button onClick={onOpenTour} className="btn-ghost px-3 py-1">
          Tour
        </button>
        <button onClick={onOpenBenchmark} className="btn-ghost px-3 py-1">
          Benchmark
        </button>
        <span
          title={`Active local model quality tier — re-run Benchmark to change it. ${engine.active_backend}`}
          className="hidden cursor-help items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-1 text-xs text-[var(--fg-muted)] lg:flex"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
          {tierLabel}
        </span>
        <button
          onClick={() => setZenMode(true)}
          data-tour="zen"
          aria-label="Enter Zen focus mode (Ctrl/Cmd+Shift+Z)"
          title="Zen Focus Mode (Ctrl/Cmd+Shift+Z)"
          className="btn-primary px-3 py-1"
        >
          Zen
        </button>
      </div>
      </header>
      {ingestError && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 border-b border-[var(--danger)] bg-[var(--danger-soft)] px-4 py-2 text-sm text-[var(--danger-fg)]"
        >
          <span>Ingest failed — {ingestError}</span>
          <button
            onClick={() => setIngestError(null)}
            aria-label="Dismiss error"
            className="rounded px-2 py-0.5 font-medium hover:bg-[var(--danger)]/10"
          >
            ✕
          </button>
        </div>
      )}
    </>
  );
}
