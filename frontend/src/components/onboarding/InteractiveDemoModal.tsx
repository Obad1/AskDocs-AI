import React, { useState } from "react";

interface InteractiveDemoModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Interactive Demo Modal (spec §3.9). Launches a zero-setup demo workspace with
 * preloaded local sample documents — no login, no API keys, no telemetry.
 * The actual sample ingestion is performed by the document subsystem; this
 * shell simply triggers the demo state and shows progress.
 */
export default function InteractiveDemoModal({
  open,
  onClose,
}: InteractiveDemoModalProps) {
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);

  if (!open) return null;

  const launch = () => {
    setLoading(true);
    // Simulate loading the bundled local sample dataset. In the full app this
    // calls the ingestion worker with pre-bundled sample docs.
    window.setTimeout(() => {
      setLoading(false);
      setReady(true);
    }, 900);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Interactive demo"
    >
      <div className="w-full max-w-md rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 text-[var(--fg)] shadow-2xl">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Zero-Setup Demo</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-[var(--fg-muted)] hover:text-[var(--fg)]"
          >
            ✕
          </button>
        </div>
        <p className="mb-4 text-sm text-[var(--fg-muted)]">
          Explore AskDocs AI with preloaded sample documents — entirely offline.
          No account required.
        </p>

        {ready ? (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-4 text-sm">
            <p className="font-medium text-[var(--fg)]">Demo workspace ready ✓</p>
            <p className="mt-1 text-[var(--fg-muted)]">
              Sample docs are loaded. Close this to start exploring.
            </p>
          </div>
        ) : (
          <button
            onClick={launch}
            disabled={loading}
            className="btn-primary w-full px-4 py-2 disabled:opacity-60"
          >
            {loading ? "Loading samples…" : "Load sample dataset"}
          </button>
        )}

        <div className="mt-5 flex justify-end">
          <button
            onClick={onClose}
            className="btn-ghost px-4 py-2 text-sm"
          >
            {ready ? "Start exploring" : "Cancel"}
          </button>
        </div>
      </div>
    </div>
  );
}
