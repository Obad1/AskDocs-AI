import React, { useMemo, useRef } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useModelEngine } from "../../context/ModelEngineContext";

// Social Study Card exporter (spec §3.8). Renders a downloadable image summary
// via the Canvas API — no third-party service.

export default function SocialCardExporter({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { ws } = useWorkspace();
  const { state } = useModelEngine();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const summary = useMemo(() => {
    const ids = Object.keys(ws.documents);
    const title = ids.length ? `AskDocs AI — ${ids.length} doc(s)` : "AskDocs AI";
    const chunks = Object.keys(ws.chunks).length;
    const cards = ws.flashcard.card_id.length;
    const tier = state.hardware_tier;
    return { title, chunks, cards, tier };
  }, [ws, state.hardware_tier]);

  const draw = useMemo(
    () => () => {
      const c = canvasRef.current;
      if (!c) return;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "#0b0f17";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "bold 28px system-ui";
      ctx.fillText(summary.title, 24, 60);
      ctx.font = "18px system-ui";
      ctx.fillStyle = "#94a3b8";
      ctx.fillText(`${summary.chunks} chunks · ${summary.cards} flashcards`, 24, 110);
      ctx.fillText(`Local model tier: ${summary.tier}`, 24, 140);
      ctx.fillText("100% offline · zero API keys", 24, 170);
    },
    [summary],
  );

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-[420px] rounded-lg bg-[var(--bg-elevated)] p-4 shadow-xl">
        <h2 className="mb-2 text-lg font-semibold">Share Study Card</h2>
        <canvas
          ref={canvasRef}
          width={384}
          height={200}
          className="w-full rounded border border-[var(--border)]"
        />
        <div className="mt-3 flex justify-end gap-2">
          <button
            onClick={draw}
            className="rounded bg-[var(--bg)] px-3 py-1"
          >
            Render
          </button>
          <button
            onClick={() => {
              draw();
              const c = canvasRef.current;
              if (!c) return;
              const url = c.toDataURL("image/png");
              const a = document.createElement("a");
              a.href = url;
              a.download = "askdocs-card.png";
              a.click();
            }}
            className="rounded bg-[var(--accent)] px-3 py-1 text-[var(--accent-fg)]"
          >
            Download PNG
          </button>
          <button onClick={onClose} className="rounded bg-[var(--bg)] px-3 py-1">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
