import React, { useRef, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useUserProfile } from "../../context/UserProfileContext";
import { useIngest } from "../../lib/parsing/ingestClient";
import type { MODE } from "../../types/schema";

const THEMES = ["slate", "obsidian", "sepia", "graphite"] as const;

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
  const ingestFile = useIngest();
  const fileRef = useRef<HTMLInputElement>(null);
  const [ingesting, setIngesting] = useState(false);

  const mode: MODE = ws.active_mode;

  return (
    <header className="flex items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2 text-sm">
      <div className="flex items-center gap-2 font-semibold text-[var(--fg)]">
        <span className="text-[var(--accent)]">◆</span>
        AskDocs AI
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
          className="rounded border border-[var(--border)] bg-[var(--bg)] px-2 py-1 text-[var(--fg)]"
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
            try {
              for (const f of files) await ingestFile(f);
            } catch (err) {
              console.error("[TopNavigation] ingest failed:", err);
            } finally {
              setIngesting(false);
              if (fileRef.current) fileRef.current.value = "";
            }
          }}
        />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={ingesting}
          className="rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-1 text-[var(--fg-muted)] hover:text-[var(--fg)] disabled:opacity-50"
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
            className="rounded border border-[var(--border)] bg-[var(--bg)] px-2 py-1 text-[var(--fg)]"
          >
            {THEMES.map((t) => (
              <option key={t} value={t}>
                {t[0].toUpperCase() + t.slice(1)}
              </option>
            ))}
          </select>
        </label>

        <button
          onClick={onOpenDemo}
          className="rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-1 text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          Demo
        </button>
        <button
          onClick={onOpenTour}
          className="rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-1 text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          Tour
        </button>
        <button
          onClick={onOpenBenchmark}
          className="rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-1 text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          Benchmark
        </button>
        <button
          onClick={() => setZenMode(true)}
          aria-label="Enter Zen focus mode (Ctrl/Cmd+Shift+Z)"
          title="Zen Focus Mode (Ctrl/Cmd+Shift+Z)"
          className="rounded bg-[var(--accent)] px-3 py-1 font-medium text-[var(--accent-fg)]"
        >
          Zen
        </button>
      </div>
    </header>
  );
}
