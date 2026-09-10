import React from "react";
import { useWorkspace } from "../../context/WorkspaceContext";

interface TopNavigationProps {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

/**
 * Slim top bar (workspace-split layout). Only brand, workspace identity and the
 * sidebar toggle live here; ingestion, theme, RAG settings and global actions
 * moved into the knowledge sidebar to cut the toolbar overload.
 */
export default function TopNavigation({
  sidebarOpen,
  onToggleSidebar,
}: TopNavigationProps) {
  const { ws } = useWorkspace();
  const docCount = Object.keys(ws.documents).length;

  return (
    <header className="flex shrink-0 items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-2 text-sm">
      <div className="flex items-center gap-2 font-semibold text-[var(--fg)]">
        <span className="brand-mark">◆</span>
        <span>AskDocs AI</span>
      </div>

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
            {ws.active_workspace === "default"
              ? "My Workspace"
              : ws.active_workspace}
          </option>
        </select>
      </label>

      <span className="hidden text-xs text-[var(--fg-muted)] lg:inline">
        {docCount} {docCount === 1 ? "document" : "documents"} indexed locally
      </span>

      <div className="ml-auto">
        <button
          onClick={onToggleSidebar}
          aria-pressed={sidebarOpen}
          aria-label="Toggle knowledge sidebar"
          title="Browse documents and settings"
          className="btn-ghost px-3 py-1"
        >
          {sidebarOpen ? "Hide" : "Browse"} panel
        </button>
      </div>
    </header>
  );
}