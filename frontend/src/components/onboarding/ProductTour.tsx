import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

interface TourStep {
  title: string;
  body: string;
  /** CSS selector for the UI element to spotlight. Omit for a centered card. */
  target?: string;
  /** Where the callout card sits relative to the target. */
  placement?: "top" | "bottom" | "right" | "left";
}

const STEPS: TourStep[] = [
  {
    title: "Your workspace, fully local",
    body: "Welcome to AskDocs AI — a study workspace that runs entirely on your device. No account, no API keys, nothing uploaded. Everything is parsed, embedded and searched locally.",
  },
  {
    title: "1 · Add your documents",
    body: "Hit “+ Add” to pick PDF, DOCX, EPUB, TXT or Markdown files. They are ingested and embedded on-device so you can ask questions over them.",
    target: "[data-tour='ingest']",
    placement: "bottom",
  },
  {
    title: "2 · Explore your source library",
    body: "The left tabs switch between the Document viewer, the cross-document Matrix, the concept Graph, focus-topic Analytics, the adaptive Quiz and your Flashcard deck.",
    target: "[data-tour='tabs']",
    placement: "bottom",
  },
  {
    title: "3 · Ask with citations",
    body: "Chat with your materials. Keep Strict mode for answers grounded only in your documents, or switch to Expanded when you want the model to reason more openly. Every claim cites its source.",
    target: "[data-tour='chat']",
    placement: "top",
  },
  {
    title: "4 · Verify every answer",
    body: "The Citation drawer lists the exact chunks behind each answer, with confidence so you can double-check before trusting it.",
    target: "[data-tour='citations']",
    placement: "top",
  },
  {
    title: "5 · Listen on the go",
    body: "The media dock turns summaries into audio with a live karaoke-style transcript, and the interrupter lets you cut in when you notice something wrong.",
    target: "[data-tour='audio']",
    placement: "top",
  },
  {
    title: "Ready when you are",
    body: "That’s the whole loop: ingest → ask → verify → review → listen. Load a sample dataset below to do it hands-on, or start with your own files.",
  },
];

const DISMISS_KEY = "askdocs.tour_dismissed";
const MARGIN = 12;

interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

/**
 * Product tour (spec §3.9). A short spotlight guide that anchors steps to the
 * actual UI, so the copy always points at the control the user needs next.
 *
 * Friction controls:
 *  - "Skip” closes cleanly; “Don't show again” persists to localStorage.
 *  - First-run gating lives in App (this component never auto-opens itself).
 *  - Escape dismisses; focus moves into the card for keyboard users.
 *  - Back/Next walk in release order; last step hands off to the demo.
 */
export default function ProductTour({
  open,
  onClose,
  onOpenDemo,
}: {
  open: boolean;
  onClose: () => void;
  onOpenDemo?: () => void;
}) {
  const [step, setStep] = useState(0);
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [anchor, setAnchor] = useState<Box | null>(null);
  const cardRef = useRef<HTMLDivElement | null>(null);

  if (open) {
    try {
      if (localStorage.getItem(DISMISS_KEY)) setDontShowAgain(true);
    } catch {
      /* storage unavailable */
    }
  }

  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;

  const measure = () => {
    const el = current.target
      ? document.querySelector<HTMLElement>(current.target)
      : null;
    if (!el) {
      setAnchor(null);
      return;
    }
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) {
      setAnchor(null);
      return;
    }
    setAnchor({ top: r.top, left: r.left, width: r.width, height: r.height });
  };

  useLayoutEffect(() => {
    if (!open) return;
    measure();
    const t = window.setTimeout(measure, 80);
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, step]);

  useEffect(() => {
    if (!open) return;
    setStep(0);
    setDontShowAgain(() => {
      try {
        return localStorage.getItem(DISMISS_KEY) === "1";
      } catch {
        return false;
      }
    });
    const h = (e: KeyboardEvent) => {
      if (e.key === "Escape") close(false);
      if (e.key === "ArrowRight") setStep((s) => Math.min(STEPS.length - 1, s + 1));
      if (e.key === "ArrowLeft") setStep((s) => Math.max(0, s - 1));
    };
    window.addEventListener("keydown", h);
    cardRef.current?.focus();
    return () => window.removeEventListener("keydown", h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => () => setDontShowAgain(false), []);

  if (!open) return null;

  const close = (persist: boolean) => {
    if (persist) {
      try {
        localStorage.setItem(DISMISS_KEY, "1");
      } catch {
        /* ignore */
      }
    }
    onClose();
  };

  // ---- Card geometry -------------------------------------------------------
  const CARD_W = 300;
  let cardStyle: React.CSSProperties = {
    left: "50%",
    top: "50%",
    transform: "translate(-50%, -50%)",
    width: "min(88vw, 24rem)",
  };

  if (anchor) {
    const cx = anchor.left + anchor.width / 2;
    const cy = anchor.top + anchor.height / 2;
    const place = current.placement ?? "bottom";
    if (place === "bottom") {
      cardStyle = {
        left: Math.min(Math.max(cx - CARD_W / 2, MARGIN), window.innerWidth - CARD_W - MARGIN),
        top: anchor.top + anchor.height + MARGIN,
        width: CARD_W,
      };
    } else if (place === "top") {
      cardStyle = {
        left: Math.min(Math.max(cx - CARD_W / 2, MARGIN), window.innerWidth - CARD_W - MARGIN),
        top: anchor.top - 180,
        width: CARD_W,
      };
    } else if (place === "right") {
      cardStyle = {
        left: anchor.left + anchor.width + MARGIN,
        top: Math.min(Math.max(cy - 70, MARGIN), window.innerHeight - 140),
        width: CARD_W,
      };
    } else {
      cardStyle = {
        left: anchor.left - CARD_W - MARGIN,
        top: Math.min(Math.max(cy - 70, MARGIN), window.innerHeight - 140),
        width: CARD_W,
      };
    }
  }

  return (
    <div
      className="fixed inset-0 z-[70]"
      role="dialog"
      aria-modal="true"
      aria-label="Product tour"
    >
      {/* Spotlight: cut a hole over the target with a soft full-screen shadow. */}
      {anchor && (
        <div
          aria-hidden
          className="pointer-events-none absolute z-10"
          style={{
            top: anchor.top,
            left: anchor.left,
            width: anchor.width,
            height: anchor.height,
            border: "2px solid var(--accent)",
            borderRadius: 10,
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.45)",
          }}
        />
      )}
      {/* Backdrop click blocks the page while the tour is open. */}
      <div
        className="absolute inset-0 z-20 bg-transparent"
        onClick={() => close(false)}
      />

      {/* Callout card */}
      <div
        ref={cardRef}
        tabIndex={-1}
        role="group"
        aria-label={`Step ${step + 1} of ${STEPS.length}`}
        className="surface-card absolute z-30 p-5 outline-none shadow-xl"
        style={cardStyle}
      >
        <div className="mb-2 flex items-center justify-between gap-4">
          <h2 className="text-base font-semibold">{current.title}</h2>
          <button
            onClick={() => close(true)}
            className="btn-ghost rounded px-2 py-1 text-xs"
          >
            ✕
          </button>
        </div>
        <p className="min-h-[4rem] text-sm leading-relaxed text-[var(--fg-muted)]">
          {current.body}
        </p>

        <div className="mt-4 mb-3 flex gap-1.5">
          {STEPS.map((_, i) => (
            <span
              key={i}
              className={`h-1.5 w-6 rounded-full transition-colors ${
                i === step ? "bg-[var(--accent)]" : "bg-[var(--border)]"
              }`}
            />
          ))}
        </div>

        {!isLast && (
          <label className="mb-3 flex items-center gap-2 text-xs text-[var(--fg-muted)]">
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={(e) => setDontShowAgain(e.target.checked)}
            />
            Don’t show the tour again
          </label>
        )}

        <div className="flex items-center justify-between gap-2">
          {step > 0 ? (
            <button
              onClick={() => setStep((s) => s - 1)}
              className="btn-ghost px-3 py-1.5 text-sm"
            >
              Back
            </button>
          ) : (
            <button
              onClick={() => close(true)}
              className="btn-ghost px-3 py-1.5 text-sm"
            >
              Skip
            </button>
          )}

          {isLast ? (
            <button
              onClick={() => {
                if (dontShowAgain) close(true);
                else onClose();
                onOpenDemo?.();
              }}
              className="btn-primary px-3 py-1.5 text-sm"
            >
              Try the demo
            </button>
          ) : (
            <button
              onClick={() => setStep((s) => s + 1)}
              className="btn-primary px-3 py-1.5 text-sm"
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
}