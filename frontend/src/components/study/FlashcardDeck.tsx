import React, { useCallback, useMemo, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { CHUNKID, RATING } from "../../types/schema";

// SM-2 rating mapping per spec §3.6: Again=0, Hard=3, Good=4, Easy=5.
const BUTTONS: { label: string; rating: RATING; cls: string }[] = [
  { label: "Again", rating: 0, cls: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" },
  { label: "Hard", rating: 3, cls: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300" },
  { label: "Good", rating: 4, cls: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300" },
  { label: "Easy", rating: 5, cls: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300" },
];

function CardFace({
  chunkId,
  front,
  back,
  onRate,
}: {
  chunkId: CHUNKID;
  front: string;
  back: string;
  onRate: (r: RATING) => void;
}) {
  const [flipped, setFlipped] = useState(false);
  return (
    <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-700">
      <div
        className="min-h-[6rem] cursor-pointer text-sm"
        onClick={() => setFlipped((v) => !v)}
      >
        {flipped ? back : front}
      </div>
      <div className="mt-3 text-xs text-gray-400">
        {flipped ? "tap to see front" : "tap to reveal answer"} · {chunkId}
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
      <div className="rounded-lg border border-dashed border-gray-300 p-6 text-center text-sm text-gray-400 dark:border-gray-600">
        No flashcards registered yet. Ingest documents to create study cards.
      </div>
    );
  }

  if (due.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 p-6 text-center text-sm text-gray-500 dark:border-gray-700">
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
            chunkId={chunkId}
            front={text.length > 280 ? text.slice(0, 280) + "…" : text}
            back={text}
            onRate={(r) => handleRate(chunkId, r)}
          />
        );
      })}
    </div>
  );
}
