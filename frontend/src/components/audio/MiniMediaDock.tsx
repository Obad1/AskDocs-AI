import React, { useCallback, useRef, useState } from "react";
import { useAudio } from "../../context/AudioContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { getTTSEngine, type TTSControls } from "../../lib/audio/tts";

// Mini floating media dock (spec §3.7). Play/stop, speed selector, voice
// picker and a sentence-based progress readout. Skip controls were removed —
// the local TTS exposes no seek, so they were misleading buttons.

const SPEEDS = [0.8, 1.0, 1.25, 1.5, 2.0];

const VOICES: { id: string; label: string }[] = [
  { id: "en_US-lessac-medium", label: "Natural (small)" },
  { id: "en_US-lessac-low", label: "Soft (low)" },
  { id: "en_US-libritts-high", label: "Expressive (high)" },
  { id: "en_US-multi-high", label: "Multi-speaker (high)" },
];

function firstDocumentText(ws: ReturnType<typeof useWorkspace>["ws"]): string {
  const ids = Object.keys(ws.documents);
  if (!ids.length) return "";
  return ws.documents[ids[0]] ?? "";
}

export default function MiniMediaDock() {
  const audio = useAudio();
  const { ws } = useWorkspace();
  const { state } = useModelEngine();
  const controlsRef = useRef<TTSControls | null>(null);
  const [error, setError] = useState<string | null>(null);

  const docCount = Object.keys(ws.documents).length;

  const play = useCallback(async () => {
    const text = firstDocumentText(ws);
    if (!text) return;
    setError(null);
    const engine = getTTSEngine();
    const sentences = text
      .replace(/\s+/g, " ")
      .trim()
      .split(/(?<=[.!?])\s+/)
      .filter(Boolean);
    audio.setTranscript(sentences);
    audio.setCurrentSentence(0);
    audio.setIsPlaying(true);
    try {
      controlsRef.current?.stop();
      const controls = await engine.speak(text, {
        voice: audio.voice || state.tts_model,
        speed: audio.speed,
        onSentence: (_s, i) => audio.setCurrentSentence(i),
        onEnd: () => {
          audio.setIsPlaying(false);
          audio.setCurrentSentence(0);
        },
      });
      controlsRef.current = controls;
    } catch (e) {
      console.error("[MiniMediaDock] TTS failed:", e);
      audio.setIsPlaying(false);
      setError("Playback failed — the local voice may not be ready yet.");
    }
  }, [ws, audio, state.tts_model]);

  const stop = useCallback(() => {
    controlsRef.current?.stop();
    controlsRef.current = null;
    audio.setIsPlaying(false);
    audio.setCurrentSentence(0);
  }, [audio]);

  return (
    <div className="flex w-full items-center gap-3 text-sm">
      <button
        onClick={() => (audio.isPlaying ? stop() : play())}
        disabled={docCount === 0}
        title={
          docCount === 0
            ? "Add a document first, then play it back"
            : audio.isPlaying
              ? "Stop playback"
              : "Read the first document aloud"
        }
        aria-label={audio.isPlaying ? "Stop" : "Play"}
        className="rounded bg-[var(--accent)] px-3 py-1 text-[var(--accent-fg)] disabled:opacity-50"
      >
        {audio.isPlaying ? "■" : "▶"}
      </button>

      <select
        value={audio.speed}
        onChange={(e) => {
          const s = parseFloat(e.target.value);
          audio.setSpeed(s);
          controlsRef.current?.setSpeed(s);
        }}
        className="rounded bg-[var(--bg)] px-1 py-1"
        aria-label="Playback speed"
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}x
          </option>
        ))}
      </select>

      <select
        value={audio.voice}
        onChange={(e) => audio.setVoice(e.target.value)}
        className="rounded bg-[var(--bg)] px-1 py-1"
        aria-label="Voice"
      >
        {VOICES.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label}
          </option>
        ))}
      </select>

      {/* Progress readout (TTS reports sentence, not time, so we show that). */}
      {audio.isPlaying ? (
        <span
          className="min-w-0 truncate text-xs text-[var(--fg-muted)]"
          aria-live="polite"
        >
          Sentence {audio.currentSentenceIndex + 1} / {audio.transcript.length} ·{" "}
          {audio.speed}x
        </span>
      ) : error ? (
        <span className="min-w-0 truncate text-xs text-[var(--danger-fg)]">
          {error}
        </span>
      ) : (
        <span className="min-w-0 truncate text-xs text-[var(--fg-muted)]">
          {docCount === 0
            ? "Add a document to enable playback"
            : "Ready — press play to hear the first document"}
        </span>
      )}
    </div>
  );
}