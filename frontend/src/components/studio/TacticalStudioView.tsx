import React, { useCallback, useEffect, useRef, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useAudio } from "../../context/AudioContext";
import { getTTSEngine, TTSControls } from "../../lib/audio/tts";
import {
  buildSimulation,
  StudioSimulation,
  StudioNode,
  ROLE_POSITIONS,
} from "../../lib/studio/simulation";

export default function TacticalStudioView() {
  const { ws, requestDocFocus } = useWorkspace();
  const { state: engine } = useModelEngine();
  const { isPlaying, setIsPlaying } = useAudio();
  const [sim, setSim] = useState<StudioSimulation | null>(null);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [isFrozen, setIsFrozen] = useState(false);
  const [selectedChoiceId, setSelectedChoiceId] = useState<string | null>(null);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [consequenceText, setConsequenceText] = useState<string | null>(null);
  const [currentSentence, setCurrentSentence] = useState<string>("");
  const controlsRef = useRef<TTSControls | null>(null);

  // Build simulation when workspace changes
  useEffect(() => {
    setSim(buildSimulation(ws));
  }, [ws]);

  // Cleanup TTS on unmount
  useEffect(() => {
    return () => {
      controlsRef.current?.stop();
    };
  }, []);

  const stop = useCallback(() => {
    controlsRef.current?.stop();
    controlsRef.current = null;
    setIsPlaying(false);
  }, [setIsPlaying]);

  const speakLine = useCallback(
    async (idx: number) => {
      if (!sim) return;
      if (idx >= sim.transcript.length) {
        stop();
        return;
      }
      const line = sim.transcript[idx];
      setPlaybackIndex(idx);
      setActiveNodeId(line.nodeId ?? null);
      const controls = await getTTSEngine().speak(line.text, {
        voice: engine.tts_model,
        onSentence: (s) => setCurrentSentence(s),
        onEnd: () => {
          if (idx === sim.decision.atLine) {
            setIsFrozen(true);
            stop();
          } else if (idx < sim.transcript.length - 1) {
            speakLine(idx + 1);
          } else {
            stop();
          }
        },
        onError: (e) => {
          console.error("[Studio] TTS error:", e);
          stop();
        },
      });
      controlsRef.current = controls;
    },
    [sim, engine.tts_model, stop],
  );

  const togglePlayPause = useCallback(() => {
    if (isPlaying) {
      stop();
    } else {
      setIsPlaying(true);
      setIsFrozen(false);
      setSelectedChoiceId(null);
      setConsequenceText(null);
      speakLine(0).catch((e) => {
        console.error("[Studio] play error:", e);
        stop();
      });
    }
  }, [isPlaying, stop, setIsPlaying, speakLine]);

  const selectChoice = useCallback(
    async (choiceId: string) => {
      if (!sim || !isFrozen) return;
      const choice = sim.decision.choices.find((c) => c.id === choiceId);
      if (!choice) return;
      stop();
      setSelectedChoiceId(choiceId);
      setIsFrozen(false);
      setConsequenceText(choice.effect);
      setActiveNodeId(null);
      // Speak consequence after short delay
      setTimeout(async () => {
        const controls = await getTTSEngine().speak(choice.effect, {
          voice: engine.tts_model,
          onEnd: () => {
            stop();
          },
        });
        controlsRef.current = controls;
        setIsPlaying(true);
      }, 400);
    },
    [sim, isFrozen, engine.tts_model, stop, setIsPlaying],
  );

  const reset = useCallback(() => {
    stop();
    setPlaybackIndex(0);
    setIsFrozen(false);
    setSelectedChoiceId(null);
    setConsequenceText(null);
    setActiveNodeId(null);
    setSim(buildSimulation(ws));
  }, [stop, ws]);

  if (!sim) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center text-sm text-[var(--fg-muted)]">
        <p className="mb-2 text-base font-medium text-[var(--fg)]">
          Studio needs at least 4 substantial chunks to field a team.
        </p>
        <p>Upload or ingest documents in the Document tab, then return here.</p>
      </div>
    );
  }

  return (
    <div className="flex h-full gap-3 p-3">
      {/* Pitch canvas */}
      <div className="relative flex-1 overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--bg)]">
        <svg viewBox="0 0 100 100" className="h-full w-full" preserveAspectRatio="xMidYMid meet">
          {/* Pitch markings */}
          <rect x="0" y="0" width="100" height="100" fill="var(--bg)" rx="4" />
          <line x1="50" y1="0" x2="50" y2="100" stroke="var(--border)" strokeWidth="0.5" />
          <circle cx="50" cy="50" r="10" fill="none" stroke="var(--border)" strokeWidth="0.5" />
          <rect x="10" y="30" width="15" height="40" fill="none" stroke="var(--border)" strokeWidth="0.5" />
          <rect x="75" y="30" width="15" height="40" fill="none" stroke="var(--border)" strokeWidth="0.5" />
          {/* Nodes */}
          {sim.nodes.map((node) => {
            const isActive = activeNodeId === node.id;
            const roleColor =
              node.role === "Goalkeeper"
                ? "var(--accent)"
                : node.role === "Defender"
                  ? "var(--bg-sunken)"
                  : node.role === "Striker"
                    ? "var(--danger)"
                    : "var(--fg-muted)";
            return (
              <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                <circle
                  r={isActive ? 4.5 : 3.5}
                  fill={roleColor}
                  opacity={node.status === "locked" ? 0.6 : 1}
                  style={{ transition: "all 0.3s ease" }}
                />
                <text
                  dy={-5}
                  textAnchor="middle"
                  fontSize="2.5"
                  fill="var(--fg)"
                  fontWeight={isActive ? "bold" : "normal"}
                  style={{ pointerEvents: "none" }}
                >
                  {node.label.split(" ").slice(0, 3).join(" ")}
                </text>
              </g>
            );
          })}
          {/* Ball */}
          {!isFrozen && selectedChoiceId === null && (
            <circle cx={sim.nodes[0]?.x ?? 50} cy={sim.nodes[0]?.y ?? 50} r="1.5" fill="var(--fg)" opacity="0.8" />
          )}
          {selectedChoiceId && (
            <circle
              cx={selectedChoiceId === "play_A" ? 85 : 15}
              cy={selectedChoiceId === "play_A" ? 60 : 40}
              r="1.5"
              fill={selectedChoiceId === "play_A" ? "var(--accent)" : "var(--danger)"}
              style={{ transition: "cx 1s ease, cy 1s ease, fill 1s ease" }}
            />
          )}
        </svg>
      </div>

      {/* Transcript + controls */}
      <div className="flex w-72 flex-col gap-2 text-sm">
        {/* Transport */}
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlayPause}
            className="btn-primary px-3 py-1.5 text-xs"
            title={isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
          <button onClick={reset} className="btn-ghost px-3 py-1.5 text-xs" title="Reset">
            ↺ Reset
          </button>
        </div>

        {/* Transcript panel */}
        <div className="flex-1 overflow-auto rounded-lg border border-[var(--border)] bg-[var(--bg-sunken)] p-2">
          <p className="mb-1 text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
            Narration
          </p>
          {sim.transcript.map((line, i) => {
            const isActive = playbackIndex === i;
            return (
              <p
                key={i}
                className={`mb-1 rounded px-2 py-1 transition-colors ${
                  isActive ? "bg-[var(--accent)] text-[var(--accent-fg)]" : ""
                } ${i === sim.decision.atLine && !isFrozen ? "italic" : ""}`}
              >
                {line.text}
              </p>
            );
          })}
        </div>

        {/* Decision HUD */}
        {isFrozen && (
          <div className="rounded-lg border-2 border-[var(--accent)] bg-[var(--bg)] p-3">
            <p className="mb-1 text-[10px] font-bold uppercase tracking-wider text-[var(--accent)]">
              ⚸ FREEZE FRAME
            </p>
            <p className="mb-2 text-sm text-[var(--fg)]">{sim.decision.prompt}</p>
            <div className="flex flex-col gap-2">
              {sim.decision.choices.map((choice) => (
                <button
                  key={choice.id}
                  onClick={() => selectChoice(choice.id)}
                  className={`rounded border px-3 py-2 text-left text-xs transition-colors ${
                    choice.isCorrect
                      ? "border-[var(--accent)] hover:bg-[var(--accent)] hover:text-[var(--accent-fg)]"
                      : "border-[var(--danger)] hover:bg-[var(--danger)] hover:text-white"
                  }`}
                >
                  <strong>{choice.label}</strong>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Outcome + citation */}
        {selectedChoiceId && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-3">
            <p className="mb-1 text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
              Consequence
            </p>
            <p className="mb-2 text-sm text-[var(--fg)]">{consequenceText}</p>
            {(() => {
              const choice = sim.decision.choices.find((c) => c.id === selectedChoiceId);
              if (!choice) return null;
              return (
                <button
                  onClick={() =>
                    requestDocFocus({
                      docId: choice.source.docId,
                      page: choice.source.page ?? 1,
                      text: choice.source.text,
                    })
                  }
                  className="btn-ghost w-full text-xs"
                >
                  📄 Open source (page {choice.source.page ?? "–"})
                </button>
              );
            })()}
          </div>
        )}

        {/* Emphasized concepts */}
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-3">
          <p className="mb-1 text-[10px] uppercase tracking-wider text-[var(--fg-muted)]">
            Top of mind
          </p>
          <ul className="flex flex-wrap gap-1">
            {sim.emphasizes.map((e, i) => (
              <li
                key={i}
                className="rounded bg-[var(--bg-sunken)] px-2 py-0.5 text-[10px] text-[var(--fg-muted)]"
              >
                {e}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}