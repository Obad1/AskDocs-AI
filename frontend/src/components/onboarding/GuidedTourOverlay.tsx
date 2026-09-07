import React, { useState } from "react";

interface GuidedTourOverlayProps {
  open: boolean;
  onClose: () => void;
  onOpenDemo?: () => void;
}

const STEPS = [
  {
    title: "1 · Ingest your documents",
    body: "Drop a PDF, DOCX, EPUB or paste text. Everything is parsed and embedded locally — nothing is uploaded.",
  },
  {
    title: "2 · Ask with citations",
    body: "Chat with your docs. Every answer links back to the source chunk via the Citation Drawer so you can verify.",
  },
  {
    title: "3 · Turn notes into flashcards",
    body: "Generate a spaced-repetition Flashcard Deck from key passages and review on a schedule.",
  },
  {
    title: "4 · Listen on the go",
    body: "Switch to podcast/audio mode: a local TTS reads summaries while the karaoke transcript highlights along.",
  },
];

/**
 * 60-Second Guided Tour (spec §3.9). Walks ingestion → citation → flashcard →
 * audio. Pure overlay, no telemetry.
 */
export default function GuidedTourOverlay({
  open,
  onClose,
  onOpenDemo,
}: GuidedTourOverlayProps) {
  const [step, setStep] = useState(0);
  if (!open) return null;

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-6"
      role="dialog"
      aria-modal="true"
      aria-label="Guided tour"
    >
      <div className="w-full max-w-xl rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 text-[var(--fg)] shadow-2xl">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Quick Tour · {current.title}</h2>
          <button
            onClick={onClose}
            aria-label="Skip tour"
            className="text-[var(--fg-muted)] hover:text-[var(--fg)]"
          >
            Skip
          </button>
        </div>
        <p className="min-h-[3rem] text-sm leading-relaxed text-[var(--fg-muted)]">
          {current.body}
        </p>
        <div className="mt-5 flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <span
                key={i}
                className={`h-1.5 w-6 rounded-full ${
                  i === step ? "bg-[var(--accent)]" : "bg-[var(--border)]"
                }`}
              />
            ))}
          </div>
          <div className="flex gap-2">
            {step > 0 && (
              <button
                onClick={() => setStep((s) => s - 1)}
                className="rounded border border-[var(--border)] px-4 py-2 text-sm text-[var(--fg-muted)]"
              >
                Back
              </button>
            )}
            {isLast ? (
              <button
                onClick={() => {
                  onClose();
                  onOpenDemo?.();
                }}
                className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[var(--accent-fg)]"
              >
                Try the demo
              </button>
            ) : (
              <button
                onClick={() => setStep((s) => s + 1)}
                className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[var(--accent-fg)]"
              >
                Next
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
