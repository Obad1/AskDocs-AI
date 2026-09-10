import React, { useCallback, useRef, useState } from "react";
import { useAudio } from "../../context/AudioContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { answerQuery } from "../../lib/llm/engine";
import { getTTSEngine } from "../../lib/audio/tts";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationDrawer } from "./CitationDrawer";
import type { MODE, CONFIDENCE_LEVEL, RetrievedChunk } from "../../types/schema";

type Intent = "chat" | "slides" | "audio";

const INTENTS: { id: Intent; label: string }[] = [
  { id: "chat", label: "Chat with docs" },
  { id: "slides", label: "Generate slides" },
  { id: "audio", label: "Convert to audio" },
];

const PLACEHOLDERS: Record<Intent, string> = {
  chat: "Ask your documents…",
  slides: "Topic for the slide deck…",
  audio: "Text to speak aloud, or leave empty to read the first document…",
};

interface Msg {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: CONFIDENCE_LEVEL;
  score?: number;
  chunks?: RetrievedChunk[];
  /** Blob URL of a generated .pptx for the assistant message to re-download. */
  deck?: string;
}

function downloadFile(url: string, name: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function parseSlides(raw: string, fallbackTitle: string): [string, string[]][] {
  const slides: [string, string[]][] = [];
  let cur: [string, string[]] | null = null;
  for (const line of raw.split("\n")) {
    const t = line.trim();
    if (/^SLIDE:/i.test(t)) {
      cur = [t.replace(/^SLIDE:\s*/i, "").trim() || fallbackTitle, []];
      slides.push(cur);
    } else if (cur && /^[-•*]/.test(t)) {
      cur[1].push(t.replace(/^[-•*]\s*/, "").trim());
    }
  }
  if (!slides.length) {
    const bullets = raw
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 5);
    slides.push([fallbackTitle, bullets.length ? bullets : ["(no detail extracted)"]]);
  }
  slides.forEach((s) => {
    if (!s[1].length) s[1] = ["(no detail extracted)"];
  });
  return slides;
}

export function ChatPane() {
  const { state } = useModelEngine();
  const { ws } = useWorkspace();
  const audio = useAudio();

  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [intent, setIntent] = useState<Intent>("chat");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef(false);

  const answerConfig = {
    mode: ws.active_mode,
    useReranker: state.reranker_enabled,
    backend: state.active_backend,
    modelId: state.llm_model,
    threshold: ws.confidence_threshold,
  };

  const reveal = useCallback(
    async (assistantId: string, text: string) => {
      const tokens = text.split(/(\s+)/);
      for (let i = 0; i < tokens.length; i++) {
        if (cancelRef.current) break;
        await new Promise((r) => setTimeout(r, 8));
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: msg.content + tokens[i] }
              : msg,
          ),
        );
      }
    },
    [],
  );

  const scrollEnd = useCallback(() => {
    requestAnimationFrame(() =>
      scrollRef.current?.scrollTo({ top: 1e9, behavior: "smooth" }),
    );
  }, []);

  const send = useCallback(
    async (raw: string) => {
      const query = raw.trim();
      if (!query) return;
      cancelRef.current = false;
      setError(null);
      setMessages((m) => [
        ...m,
        { id: `u-${Date.now()}`, role: "user", content: query },
      ]);
      setInput("");
      setBusy(true);

      const assistantId = `a-${Date.now()}`;
      setMessages((m) => [
        ...m,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      try {
        const { text, result } = await answerQuery(query, answerConfig);
        await reveal(assistantId, text);
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  confidence: result.confidence,
                  score: result.score,
                  chunks: result.chunks,
                }
              : msg,
          ),
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
        scrollEnd();
      }
    },
    [answerConfig, reveal, scrollEnd],
  );

  const speak = useCallback(
    async (raw: string) => {
      const typed = raw.trim();
      const source = typed || (Object.values(ws.documents)[0] ?? "");
      if (!source) {
        setError(
          "Nothing to speak — type text, or add a document and leave the box empty to read it aloud.",
        );
        return;
      }
      cancelRef.current = false;
      setError(null);
      setMessages((m) => [
        ...m,
        {
          id: `u-${Date.now()}`,
          role: "user",
          content: typed ? `Speak aloud: ${typed}` : "Read a document aloud",
        },
      ]);
      setInput("");
      setBusy(true);
      try {
        const sentences = source
          .replace(/\s+/g, " ")
          .trim()
          .split(/(?<=[.!?])\s+/)
          .filter(Boolean);
        audio.setTranscript(sentences);
        audio.setCurrentSentence(0);
        audio.setIsPlaying(true);
        const engine = getTTSEngine();
        await engine.speak(source, {
          voice: audio.voice || state.tts_model,
          speed: audio.speed,
          onSentence: (_s, i) => audio.setCurrentSentence(i),
          onEnd: () => {
            audio.setIsPlaying(false);
            audio.setCurrentSentence(0);
          },
        });
        setMessages((m) => [
          ...m,
          {
            id: `a-${Date.now()}`,
            role: "assistant",
            content: `Played ${sentences.length} ${sentences.length === 1 ? "sentence" : "sentences"} aloud — follow along in the transcript strip below.`,
          },
        ]);
      } catch (e) {
        console.error("[ChatPane] TTS failed:", e);
        audio.setIsPlaying(false);
        setError("Playback failed — the local voice may not be ready yet.");
      } finally {
        setBusy(false);
        scrollEnd();
      }
    },
    [ws.documents, audio, state.tts_model, scrollEnd],
  );

  const makeSlides = useCallback(
    async (raw: string) => {
      const query = raw.trim();
      if (!query) {
        setError("Describe the topic for the slide deck first.");
        return;
      }
      cancelRef.current = false;
      setError(null);
      setMessages((m) => [
        ...m,
        { id: `u-${Date.now()}`, role: "user", content: `Generate slides: ${query}` },
      ]);
      setInput("");
      setBusy(true);

      const assistantId = `a-${Date.now()}`;
      setMessages((m) => [
        ...m,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      try {
        const { text } = await answerQuery(query, answerConfig);
        const slides = parseSlides(text, query);
        const res = await fetch("/api/v1/export/pptx", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slides }),
        });
        if (!res.ok)
          throw new Error(
            `Slide export failed (${res.status}) — the export service did not return a deck.`,
          );
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        downloadFile(url, "askdocs-deck.pptx");
        await reveal(
          assistantId,
          `Built a ${slides.length}-slide .pptx from your sources and started a download. The deck is grounded in your documents under ${ws.active_mode === "StrictDocumentOnly" ? "Strict (document-only)" : "Expanded"} mode.`,
        );
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId ? { ...msg, deck: url } : msg,
          ),
        );
      } catch (e) {
        console.error("[ChatPane] slides failed:", e);
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: "Slide generation did not complete." }
              : msg,
          ),
        );
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
        scrollEnd();
      }
    },
    [answerConfig, reveal, ws.active_mode, scrollEnd],
  );

  const run = useCallback(() => {
    if (busy) return;
    if (intent === "slides") void makeSlides(input);
    else if (intent === "audio") void speak(input);
    else void send(input);
  }, [busy, intent, input, makeSlides, speak, send]);

  const cancel = useCallback(() => {
    cancelRef.current = true;
    setError(null);
    setBusy(false);
  }, []);

  return (
    <div data-tour="chat" className="surface-card flex h-full flex-col">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 && (
          <p className="text-sm text-[var(--fg-muted)]">
            Ask a question, generate a slide deck, or turn text into speech.
            Everything runs 100% locally.
          </p>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            className={`rounded-lg p-3 text-sm ${
              m.role === "user"
                ? "msg-user ml-auto max-w-[85%]"
                : "msg-ai mr-auto max-w-[90%]"
            }`}
          >
            {m.content || (busy ? "Thinking…" : "")}
            {m.role === "assistant" && m.confidence && (
              <div className="mt-2">
                <ConfidenceBadge level={m.confidence} score={m.score} />
              </div>
            )}
            {m.role === "assistant" && m.chunks && (
              <div className="mt-2">
                <CitationDrawer chunks={m.chunks} confidence={m.confidence} />
              </div>
            )}
            {m.role === "assistant" && m.deck && (
              <button
                type="button"
                onClick={() => downloadFile(m.deck!, "askdocs-deck.pptx")}
                className="btn-ghost mt-2 px-3 py-1 text-xs"
              >
                Download deck again (.pptx)
              </button>
            )}
          </div>
        ))}
        {error && (
          <div
            className="rounded-md border border-[var(--danger)] bg-[var(--danger-soft)] p-2 text-xs text-[var(--danger-fg)]"
            role="alert"
          >
            {error}
          </div>
        )}
      </div>

      {/* Intent-mode prompt bar */}
      <div className="shrink-0 border-t border-[var(--border)] p-3">
        <div role="group" aria-label="Task to run" className="mb-2 flex flex-wrap gap-1">
          {INTENTS.map((i) => (
            <button
              key={i.id}
              onClick={() => setIntent(i.id)}
              aria-pressed={intent === i.id}
              className={`rounded-md px-3 py-1.5 text-xs ${
                intent === i.id
                  ? "bg-[var(--accent)] text-[var(--accent-fg)]"
                  : "text-[var(--fg-muted)] hover:bg-[var(--bg-sunken)]"
              }`}
            >
              {i.label}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            className="field flex-1 px-3 py-2 text-sm"
            placeholder={PLACEHOLDERS[intent]}
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run();
            }}
          />
          <button
            className="btn-primary px-4 py-2 text-sm"
            onClick={busy ? cancel : run}
          >
            {busy ? "Cancel" : "Run"}
          </button>
        </div>
      </div>
    </div>
  );
}