import React, { useCallback, useRef } from "react";
import { useAudio } from "../../context/AudioContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { getTTSEngine, type TTSControls } from "../../lib/audio/tts";

// Mini floating media dock (spec §3.7). 64px bottom bar: play/stop, speed
// selector, 10s skip, voice picker, and a live waveform (wavesurfer.js).

const SPEEDS = [0.8, 1.0, 1.25, 1.5, 2.0];

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
  const waveformRef = useRef<HTMLDivElement | null>(null);

  const play = useCallback(async () => {
    const text = firstDocumentText(ws);
    if (!text) return;
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
      const controls = await engine.speak(text, {
        voice: audio.voice || state.tts_model,
        speed: audio.speed,
        onSentence: (_s, i) => audio.setCurrentSentence(i),
        onEnd: () => audio.setIsPlaying(false),
      });
      controlsRef.current = controls;
    } catch (e) {
      console.error("[MiniMediaDock] TTS failed:", e);
      audio.setIsPlaying(false);
    }
  }, [ws, audio, state.tts_model]);

  const stop = useCallback(() => {
    controlsRef.current?.stop();
    controlsRef.current = null;
    audio.setIsPlaying(false);
  }, [audio]);

  return (
    <div className="flex w-full items-center gap-3 text-sm">
      <button
        onClick={() => (audio.isPlaying ? stop() : play())}
        className="rounded bg-[var(--accent)] px-3 py-1 text-[var(--accent-fg)]"
        aria-label={audio.isPlaying ? "Stop" : "Play"}
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

      <button
        onClick={() => audio.skip(-10)}
        className="rounded bg-[var(--bg)] px-2 py-1"
        aria-label="Skip back 10s"
      >
        -10s
      </button>
      <button
        onClick={() => audio.skip(10)}
        className="rounded bg-[var(--bg)] px-2 py-1"
        aria-label="Skip forward 10s"
      >
        +10s
      </button>

      <select
        value={audio.voice}
        onChange={(e) => audio.setVoice(e.target.value)}
        className="rounded bg-[var(--bg)] px-1 py-1"
        aria-label="Voice"
      >
        {Object.values({
          "en_US-lessac-medium": "Lessac (med)",
          "en_US-lessac-low": "Lessac (low)",
          "en_US-libritts-high": "LibriTTS (high)",
          "en_US-multi-high": "Multi (high)",
        }).map((label, i) => {
          const ids = [
            "en_US-lessac-medium",
            "en_US-lessac-low",
            "en_US-libritts-high",
            "en_US-multi-high",
          ];
          return (
            <option key={ids[i]} value={ids[i]}>
              {label}
            </option>
          );
        })}
      </select>

      <div ref={waveformRef} className="min-w-0 flex-1" aria-hidden />
    </div>
  );
}
