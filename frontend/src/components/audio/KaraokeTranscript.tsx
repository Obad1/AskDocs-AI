import React, { useEffect, useRef } from "react";
import { useAudio } from "../../context/AudioContext";

// Karaoke transcript (spec §3.7). Auto-scrolling sentence view that follows
// the active TTS sentence. Tapping a sentence is intentionally NOT a seek
// control — the local TTS can't seek, so a clickable affordance would lie to
// the user. This is a passive readout only.

export default function KaraokeTranscript() {
  const { transcript, currentSentenceIndex } = useAudio();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const activeRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({
      block: "nearest",
      inline: "nearest",
    });
  }, [currentSentenceIndex]);

  if (!transcript.length) {
    return (
      <div className="truncate text-xs text-[var(--fg-muted)]">
        No active transcript. Press play in the media dock.
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      role="region"
      aria-label="Transcript"
      className="max-h-16 overflow-y-auto text-sm leading-relaxed"
    >
      {transcript.map((sentence, i) => {
        const active = i === currentSentenceIndex;
        return (
          <span
            key={i}
            ref={active ? activeRef : undefined}
            className={
              active
                ? "rounded bg-[var(--accent)] px-0.5 text-[var(--accent-fg)]"
                : i < currentSentenceIndex
                  ? "text-[var(--fg-muted)]"
                  : "text-[var(--fg)]"
            }
          >
            {sentence}{" "}
          </span>
        );
      })}
    </div>
  );
}