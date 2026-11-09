import React from "react";
import { useWorkspace } from "../../context/WorkspaceContext";

interface TopNavigationProps {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}

/**
 * Honest single-workspace header: do not present a fake, non-functional
 * workspace selector until switching between workspaces is implemented.
 */
export default function TopNavigation({
  sidebarOpen,
  onToggleSidebar,
}: TopNavigationProps) {
  const { ws } = useWorkspace();
  const docCount = Object.keys(ws.documents).length;
  const mode = ws.active_mode === "StrictDocumentOnly" ? "Document-only" : "Expanded AI";

  return (
    <header className="app-topbar flex shrink-0 items-center gap-3 border-b border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-2 sm:px-5">
      <div className="flex min-w-0 items-center gap-3">
        <span className="brand-mark" aria-hidden="true">◆</span>
        <div className="min-w-0 leading-tight">
          <div className="truncate text-sm font-bold tracking-tight text-[var(--fg)] sm:text-base">AskDocs AI</div>
          <div className="hidden text-[11px] text-[var(--fg-muted)] sm:block">Read · Discover · Understand</div>
        </div>
      </div>

      <div className="ml-2 hidden min-w-0 items-center gap-3 border-l border-[var(--border)] pl-4 text-xs text-[var(--fg-muted)] md:flex">
        <span className="truncate font-medium text-[var(--fg)]">
          {ws.active_workspace === "default" ? "My library" : ws.active_workspace}
        </span>
        <span aria-label="Number of uploaded documents" className="rounded-full bg-[var(--accent-soft)] px-2.5 py-1 text-[var(--fg)]">
          {docCount} {docCount === 1 ? "file" : "files"}
        </span>
        <span className="hidden lg:inline">{mode}</span>
      </div>

      <div className="ml-auto">
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-expanded={sidebarOpen}
          aria-controls="askdocs-knowledge-sidebar"
          aria-label={sidebarOpen ? "Close document library" : "Open document library"}
          className="btn-ghost inline-flex min-h-10 items-center gap-2 px-3 py-2 text-xs font-semibold sm:text-sm"
        >
          <span aria-hidden="true" className="text-base leading-none">{sidebarOpen ? "✕" : "☰"}</span>
          {sidebarOpen ? "Close library" : "Browse files"}
        </button>
      </div>
    </header>
  );
}
