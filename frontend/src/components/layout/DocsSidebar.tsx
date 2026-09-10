import React, { useRef, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useUserProfile } from "../../context/UserProfileContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useIngest } from "../../lib/parsing/ingestClient";
import type { FORMAT_TYPE } from "../../types/schema";

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

const FORMAT_LABELS: Record<string, string> = {
  PDF: "PDF",
  DOCX: "DOCX",
  PPTX: "PPTX",
  EPUB: "EPUB",
  TXT: "TXT",
  MD: "MD",
};

interface DocsSidebarProps {
  open: boolean;
  onClose: () => void;
  onOpenSettings: () => void;
  onOpenSummarize: () => void;
  onOpenBenchmark: () => void;
  onOpenTour: () => void;
  onOpenDemo: () => void;
  onOpenDocument: (docId: string) => void;
}

/**
 * Knowledge sidebar (workspace-split layout). Replaces the old packed top bar:
 * file ingestion + library list, appearance, RAG settings entry, and global
 * actions all live here, next to the content they act on.
 */
export default function DocsSidebar({
  open,
  onClose,
  onOpenSettings,
  onOpenSummarize,
  onOpenBenchmark,
  onOpenTour,
  onOpenDemo,
  onOpenDocument,
}: DocsSidebarProps) {
  const { ws, setZenMode, activeDocId } = useWorkspace();
  const { profile, update } = useUserProfile();
  const { state: engine } = useModelEngine();
  const ingestFile = useIngest();

  const fileRef = useRef<HTMLInputElement>(null);
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  const docIds = Object.keys(ws.documents);
  const tierLabel = TIER_LABELS[engine.hardware_tier] ?? engine.hardware_tier;

  return (
    <>
      <aside
        aria-label="Knowledge sidebar"
        className={`relative z-30 h-full overflow-hidden border-r border-[var(--border)] bg-[var(--bg-elevated)] transition-[width] duration-200 ${
          open ? "w-72" : "w-0 border-r-0"
        }`}
      >
        {open && (
          <div className="flex h-full w-72 min-w-0 flex-col">
            {/* Documents */}
            <div className="flex shrink-0 items-center justify-between border-b border-[var(--border)] px-3 py-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[var(--fg-muted)]">
                Documents
              </span>
              <span className="text-xs text-[var(--fg-muted)]">
                {docIds.length} {docIds.length === 1 ? "doc" : "docs"}
              </span>
            </div>

            <div className="flex shrink-0 items-center gap-2 border-b border-[var(--border)] px-3 py-2">
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
                    for (const f of files)
                      await ingestFile(f, (p) => {
                        // Model download is the one long, silent step on first
                        // use; surface it as a live status line.
                        const msg = p.message?.startsWith("Downloading")
                          ? p.message
                          : null;
                        setIngestStatus(msg);
                      });
                  } catch (err) {
                    setIngestError(
                      err instanceof Error
                        ? err.message
                        : "That file could not be read. Try a PDF, DOCX, PPTX, EPUB, TXT, or MD file.",
                    );
                  } finally {
                    setIngesting(false);
                    setIngestStatus(null);
                    if (fileRef.current) fileRef.current.value = "";
                  }
                }}
              />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={ingesting}
                data-tour="ingest"
                className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-left text-sm font-medium text-[var(--fg)] hover:bg-[var(--bg-sunken)] disabled:opacity-60"
              >
                {ingesting ? "Ingesting…" : "+ Add documents"}
              </button>
            </div>

            {ingestStatus && (
              <div
                role="status"
                className="shrink-0 px-3 py-1 text-xs text-[var(--fg-muted)]"
              >
                {ingestStatus}
              </div>
            )}
            {ingestError && (
              <div
                role="alert"
                className="shrink-0 border-b border-[var(--danger)] bg-[var(--danger-soft)] px-3 py-2 text-xs text-[var(--danger-fg)]"
              >
                Ingest failed — {ingestError}
              </div>
            )}

            <ul className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
              {docIds.length === 0 && (
                <li className="px-2 py-2 text-xs leading-relaxed text-[var(--fg-muted)]">
                  No documents yet. Add a PDF, DOCX, PPTX, EPUB, TXT or MD file
                  and it will be indexed on your device.
                </li>
              )}
              {docIds.map((id) => (
                <li key={id}>
                  <button
                    onClick={() => onOpenDocument(id)}
                    aria-pressed={activeDocId === id}
                    className={`mb-1 w-full rounded-md px-3 py-2 text-left ${
                      activeDocId === id
                        ? "bg-[var(--accent)] text-[var(--accent-fg)]"
                        : "text-[var(--fg)] hover:bg-[var(--bg-sunken)]"
                    }`}
                  >
                    <span className="block truncate text-sm font-medium">
                      {id}
                    </span>
                    <span
                      className={`block text-[10px] ${
                        activeDocId === id
                          ? "text-[var(--accent-fg)]/80"
                          : "text-[var(--fg-muted)]"
                      }`}
                    >
                      {FORMAT_LABELS[ws.doc_formats[id] as FORMAT_TYPE] ??
                        "DOC"}{" "}
                      · {ws.doc_chunks[id]?.length ?? 0} chunks
                    </span>
                  </button>
                </li>
              ))}
            </ul>

            {/* Appearance + settings */}
            <div className="shrink-0 border-t border-[var(--border)] px-3 py-2">
              <button
                onClick={onOpenSettings}
                className="mb-1 w-full rounded-md px-3 py-2 text-left text-sm font-medium text-[var(--fg)] hover:bg-[var(--bg-sunken)]"
              >
                Answer settings
              </button>
              <button
                onClick={onOpenSummarize}
                className="mb-2 w-full rounded-md px-3 py-2 text-left text-sm font-medium text-[var(--fg)] hover:bg-[var(--bg-sunken)]"
              >
                Summarize documents
              </button>
              <label
                className="flex items-center justify-between gap-2 text-xs text-[var(--fg-muted)]"
              >
                <span>Theme</span>
                <select
                  data-tour="theme"
                  aria-label="Theme"
                  value={profile.theme}
                  onChange={(e) =>
                    update({
                      theme: e.target.value as (typeof THEMES)[number],
                    })
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
            </div>

            {/* Global tools */}
            <div className="flex shrink-0 flex-col gap-1 border-t border-[var(--border)] px-3 py-2">
              <div className="flex gap-1">
                <button
                  onClick={onOpenDemo}
                  data-tour="demo"
                  className="btn-ghost px-2 py-1 text-xs"
                >
                  Demo
                </button>
                <button
                  onClick={onOpenTour}
                  className="btn-ghost px-2 py-1 text-xs"
                >
                  Tour
                </button>
                <button
                  onClick={onOpenBenchmark}
                  className="btn-ghost px-2 py-1 text-xs"
                >
                  Benchmark
                </button>
              </div>
              <div className="flex items-center gap-2">
                <span
                  title={`Active local model quality tier — re-run Benchmark to change it. ${engine.active_backend}`}
                  className="flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--bg)] px-2 py-1 text-xs text-[var(--fg-muted)]"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
                  {tierLabel}
                </span>
                <button
                  onClick={() => setZenMode(true)}
                  data-tour="zen"
                  aria-label="Enter Zen focus mode (Ctrl/Cmd+Shift+Z)"
                  title="Zen Focus Mode (Ctrl/Cmd+Shift+Z)"
                  className="btn-primary ml-auto px-3 py-1 text-xs"
                >
                  Zen
                </button>
              </div>
            </div>
          </div>
        )}
      </aside>

      {open && (
        <button
          aria-label="Close sidebar"
          onClick={onClose}
          className="fixed inset-0 z-20 bg-black/40 lg:hidden"
        />
      )}
    </>
  );
}