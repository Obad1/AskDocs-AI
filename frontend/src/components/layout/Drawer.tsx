import React, { useEffect } from "react";

interface DrawerProps {
  open: boolean;
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

/**
 * Right slide-over panel used for contextual settings (RAG params, summary
 * generator). Replaces permanently-pinned controls: only opens when the user
 * asks for it, keeping the primary canvas focused on the active task.
 */
export default function Drawer({ open, title, onClose, children }: DrawerProps) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[60]" role="dialog" aria-modal="true" aria-label={title}>
      <button
        aria-label="Close panel"
        onClick={onClose}
        className="absolute inset-0 h-full w-full bg-black/40"
      />
      <div className="absolute right-0 top-0 flex h-full w-full max-w-sm flex-col border-l border-[var(--border)] bg-[var(--bg-elevated)] shadow-xl">
        <header className="flex shrink-0 items-center justify-between border-b border-[var(--border)] px-4 py-3">
          <h2 className="text-sm font-semibold text-[var(--fg)]">{title}</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="btn-ghost rounded px-2 py-1 text-xs"
          >
            ✕
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
      </div>
    </div>
  );
}