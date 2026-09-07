import React from "react";
import { useAudio } from "../../context/AudioContext";

// Karaoke transcript (spec §3.7). Auto-scrolling sentence view; clicking a
// sentence seeks playback to that sentence (best-effort for the active TTS).

export default function KaraokeTranscript() {
  const { transcript, currentSentenceIndex, setCurrentSentence } = useAudio();

  if (!transcript.length) {
    return (
      <div className="truncate text-xs text-[var(--fg-muted)]">
        No active transcript. Press play in the media dock.
      </div>
    );
  }

  return (
    <div className="h-10 overflow-y-auto text-sm leading-relaxed">
      {transcript.map((sentence, i) => (
        <span
          key={i}
          onClick={() => setCurrentSentence(i)}
          className={`mr-1 cursor-pointer rounded px-0.5 ${
            i === currentSentenceIndex
              ? "bg-[var(--accent)] text-[var(--accent-fg)]"
              : i < currentSentenceIndex
                ? "text-[var(--fg-muted)]"
                : "text-[var(--fg)]"
          }`}
        >
          {sentence}{" "}
        </span>
      ))}
    </div>
  );
}
