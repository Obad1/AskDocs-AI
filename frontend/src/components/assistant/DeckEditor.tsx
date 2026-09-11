import React from "react";

type Slide = [string, string[]];

interface DeckEditorProps {
  slides: Slide[];
  busy: boolean;
  onChange: (slides: Slide[]) => void;
  /** Compiles the current (possibly edited) deck to .pptx and downloads it. */
  onExport: () => Promise<void>;
}

/**
 * Live slide deck studio (spec §3.1 artifact path). Renders each generated
 * slide as an editable card — titles and bullets can be tweaked before export,
 * mirroring "inline AI editing" with a fully local compile step at the end.
 */
export default function DeckEditor({
  slides,
  busy,
  onChange,
  onExport,
}: DeckEditorProps) {
  if (slides.length === 0) {
    return (
      <div className="py-8 text-center text-sm text-[var(--fg-muted)]">
        {busy
          ? "Rendering slides as the model writes them…"
          : "No slides yet. Run a slide generation first."}
      </div>
    );
  }

  const update = (i: number, patch: Partial<Slide>) => {
    const next = slides.map((s, idx) => (idx === i ? [...patch] as Slide : s));
    onChange(next);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-[var(--fg-muted)]">
          {slides.length} {slides.length === 1 ? "slide" : "slides"} · editable
          before export
        </span>
        <button
          onClick={() => void onExport()}
          disabled={busy || slides.length === 0}
          className="btn-primary px-3 py-1.5 text-sm"
        >
          Export to .pptx
        </button>
      </div>

      {slides.map((s, i) => (
        <div key={i} className="rounded-xl border border-[var(--border)] bg-[var(--bg)] p-3">
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded bg-[var(--accent)] px-1.5 py-0.5 text-xs font-semibold text-[var(--accent-fg)]">
              {i + 1}
            </span>
            <button
              onClick={() => onChange(slides.filter((_, idx) => idx !== i))}
              aria-label={`Remove slide ${i + 1}`}
              className="btn-ghost ml-auto rounded px-2 py-0.5 text-xs"
            >
              Remove
            </button>
          </div>
          <label className="mb-1 block text-xs text-[var(--fg-muted)]">
            Title
          </label>
          <input
            value={s[0]}
            onChange={(e) => update(i, [e.target.value, s[1]])}
            aria-label={`Slide ${i + 1} title`}
            className="field mb-2 w-full px-2 py-1.5 text-sm font-semibold"
          />
          <label className="mb-1 block text-xs text-[var(--fg-muted)]">
            Bullets (one per line)
          </label>
          <textarea
            value={s[1].join("\n")}
            onChange={(e) =>
              update(i, [
                s[0],
                e.target.value.split("\n").map((l) => l.trim()).filter(Boolean),
              ])
            }
            rows={Math.min(6, Math.max(2, s[1].length + 1))}
            aria-label={`Slide ${i + 1} bullets`}
            className="field w-full resize-y px-2 py-1.5 text-sm"
          />
          <button
            onClick={() => update(i, [s[0], [...s[1], ""]])}
            className="btn-ghost mt-1 rounded px-2 py-1 text-xs"
          >
            + Add bullet
          </button>
        </div>
      ))}

      <button
        onClick={() => onChange([...slides, ["New slide", []]])}
        className="btn-ghost w-full rounded-lg px-3 py-2 text-sm"
      >
        + Add slide
      </button>
    </div>
  );
}