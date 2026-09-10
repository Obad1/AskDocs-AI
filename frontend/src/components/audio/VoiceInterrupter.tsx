import React, { useCallback, useEffect, useRef, useState } from "react";
import { useAudio } from "../../context/AudioContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { startVAD, type VADHandle } from "../../lib/audio/vad";
import { getSTTEngine } from "../../lib/audio/stt";
import { getTTSEngine } from "../../lib/audio/tts";
import { retrieve } from "../../lib/retrieval/retrieve";
import { getLLMWorker } from "../../lib/llm/workerClient";

// Voice Interrupter (spec §3.7). VAD detects user speech during playback,
// captures the query via STT, retrieves from local RAG, and speaks the answer
// with the local TTS engine — all offline. Falls back gracefully if any local
// model is unavailable.

export default function VoiceInterrupter() {
  const { setVoiceInterrupt } = useAudio();
  const { state } = useModelEngine();
  const [armed, setArmed] = useState(false);
  const vadRef = useRef<VADHandle | null>(null);
  const [status, setStatus] = useState("idle");

  const answer = useCallback(
    async (query: string) => {
      try {
        const chunks = await retrieve(query, { useReranker: state.reranker_enabled });
        const context = chunks
          .map((c) => `[${c.docId}${c.page ? ", p" + c.page : ""}]\n${c.text}`)
          .join("\n\n");
        const worker = getLLMWorker();
        const reply = await worker.generate(
          [
            {
              role: "system",
              content:
                "Answer the user's question using ONLY the provided document context. If the context lacks the answer, say so.",
            },
            {
              role: "user",
              content: `Context:\n${context}\n\nQuestion: ${query}`,
            },
          ],
          { backend: state.active_backend, modelId: state.llm_model },
        );
        const engine = getTTSEngine();
        await engine.speak(reply, {
          voice: state.tts_model,
          onEnd: () => {
            setVoiceInterrupt(false);
            // Back to listening so the user can ask again without re-arming.
            setStatus("listening");
          },
        });
      } catch (e) {
        console.error("[VoiceInterrupter] answer failed:", e);
        setVoiceInterrupt(false);
        setStatus("listening");
      }
    },
    [state, setVoiceInterrupt],
  );

  const arm = useCallback(async () => {
    setStatus("listening");
    setArmed(true);
    setVoiceInterrupt(false);
    try {
      const handle = await startVAD({
        onSpeechStart: () => setVoiceInterrupt(true),
        onSpeechEnd: async () => {
          setStatus("transcribing");
          const stt = getSTTEngine(state.stt_model);
          const stop = await stt.listen((text) => {
            if (!text.trim()) {
              setStatus("listening");
              return;
            }
            setStatus("answering");
            void answer(text.trim());
          });
          // The STT listen loop streams; for interrupter we capture one utterance.
          setTimeout(() => stop(), 4000);
        },
        onError: (err) => {
          console.warn("[VoiceInterrupter] VAD error:", err);
          setStatus("unavailable");
          setArmed(false);
        },
      });
      vadRef.current = handle;
    } catch (e) {
      console.warn("[VoiceInterrupter] could not start:", e);
      setStatus("unavailable");
      setArmed(false);
    }
  }, [state.stt_model, answer, setVoiceInterrupt]);

  const disarm = useCallback(() => {
    vadRef.current?.stop();
    vadRef.current = null;
    setArmed(false);
    setVoiceInterrupt(false);
    setStatus("idle");
  }, [setVoiceInterrupt]);

  useEffect(() => () => vadRef.current?.stop(), []);

  const unavailable = status === "unavailable";
  return (
    <button
      onClick={() => (armed ? disarm() : arm())}
      title={
        unavailable
          ? "Voice input unavailable — check that the microphone is allowed for this site."
          : "Voice interrupter (hands-free Q&A)"
      }
      aria-pressed={armed}
      className={`rounded px-3 py-1 text-[var(--accent-fg)] ${
        unavailable || armed ? "bg-[var(--danger)]" : "bg-[var(--accent)]"
      }`}
    >
      {unavailable ? "🎤 Off" : armed ? `● ${status}` : "🎤 Ask"}
    </button>
  );
}
