import React, { useCallback, useMemo, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { CHUNKID, RATING } from "../../types/schema";

// SM-2 rating mapping per spec §3.6: Again=0, Hard=3, Good=4, Easy=5.
const BUTTONS: { label: string; rating: RATING; cls: string }[] = [
  { label: "Again", rating: 0, cls: "bg-[var(--danger-soft)] text-[var(--danger-fg)]" },
  { label: "Hard", rating: 3, cls: "bg-[var(--warn-soft)] text-[var(--warn-fg)]" },
  { label: "Good", rating: 4, cls: "bg-[var(--success-soft)] text-[var(--success-fg)]" },
  { label: "Easy", rating: 5, cls: "bg-[var(--info-soft)] text-[var(--info-fg)]" },
];

function CardFace({
  front,
  back,
  onRate,
}: {
  front: string;
  back: string;
  onRate: (r: RATING) => void;
}) {
  const [flipped, setFlipped] = useState(false);
  return (
    <div className="surface-card p-4">
      <div
        role="button"
        tabIndex={0}
        aria-label={flipped ? "Show front (press Enter)" : "Show answer (press Enter)"}
        aria-pressed={flipped}
        className="min-h-[6rem] cursor-pointer text-sm"
        onClick={() => setFlipped((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setFlipped((v) => !v);
          }
        }}
      >
        {flipped ? back : front}
      </div>
      <div className="mt-3 text-xs text-[var(--fg-muted)]">
        {flipped ? "press Enter or click to see front" : "press Enter or click to reveal answer"}
      </div>
      {flipped && (
        <div className="mt-3 grid grid-cols-4 gap-2">
          {BUTTONS.map((b) => (
            <button
              key={b.label}
              className={`rounded-md px-2 py-1.5 text-xs font-medium ${b.cls}`}
              onClick={() => onRate(b.rating)}
            >
              {b.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function FlashcardDeck() {
  const { ws, getDueCards, reviewFlashcard } = useWorkspace();

  const due = useMemo(() => getDueCards(), [getDueCards, ws.flashcard]);

  const handleRate = useCallback(
    (chunkId: CHUNKID, rating: RATING) => {
      reviewFlashcard(chunkId, rating, Date.now());
    },
    [reviewFlashcard],
  );

  if (ws.flashcard.card_id.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border)] p-6 text-center text-sm text-[var(--fg-muted)]">
        No flashcards registered yet. Ingest documents to create study cards.
      </div>
    );
  }

  if (due.length === 0) {
    return (
      <div className="surface-card p-6 text-center text-sm text-[var(--fg-muted)]">
        🎉 All caught up — no cards due right now.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="text-sm font-medium">
        Due cards: {due.length}
      </div>
      {due.map((chunkId) => {
        const chunk = ws.chunks[chunkId];
        const text = chunk?.text ?? "(card content unavailable)";
        return (
          <CardFace
            key={chunkId}
            front={text.length > 280 ? text.slice(0, 280) + "…" : text}
            back={text}
            onRate={(r) => handleRate(chunkId, r)}
          />
        );
      })}
    </div>
  );
}
