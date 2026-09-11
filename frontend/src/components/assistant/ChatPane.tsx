import React, { useCallback, useRef, useState } from "react";
import { useAudio } from "../../context/AudioContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useUserProfile } from "../../context/UserProfileContext";
import { answerQuery } from "../../lib/llm/engine";
import { getTTSEngine } from "../../lib/audio/tts";
import { personaInstruction } from "../../lib/personas";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationDrawer } from "./CitationDrawer";
import PersonaPicker from "./PersonaPicker";
import DeckEditor from "./DeckEditor";
import Drawer from "../layout/Drawer";
import type { CONFIDENCE_LEVEL, RetrievedChunk } from "../../types/schema";

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
  /** Assistant message that produced a deck; offers a "Open deck" action. */
  deckReady?: boolean;
}

type Slide = [string, string[]];

function downloadFile(url: string, name: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function parseSlides(raw: string, fallbackTitle: string): Slide[] {
  const slides: Slide[] = [];
  let cur: Slide | null = null;
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
    slides.push([
      fallbackTitle,
      bullets.length ? bullets : ["(no detail extracted)"],
    ]);
  }
  slides.forEach((s) => {
    if (!s[1].length) s[1] = ["(no detail extracted)"];
  });
  return slides;
}

export function ChatPane() {
  const { state } = useModelEngine();
  const { ws } = useWorkspace();
  const { profile } = useUserProfile();
  const audio = useAudio();

  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [intent, setIntent] = useState<Intent>("chat");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deckOpen, setDeckOpen] = useState(false);
  const [deckTitle, setDeckTitle] = useState("");
  const [deckSlides, setDeckSlides] = useState<Slide[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef(false);

  const persona = personaInstruction(profile.personaId, profile.customPersonas);

  const answerConfig = {
    mode: ws.active_mode,
    useReranker: state.reranker_enabled,
    backend: state.active_backend,
    modelId: state.llm_model,
    threshold: ws.confidence_threshold,
    personaInstruction: persona,
  };

  const reveal = useCallback(
    async (
      assistantId: string,
      text: string,
      onProgress?: (content: string) => void,
    ) => {
      const tokens = text.split(/(\s+)/);
      for (let i = 0; i < tokens.length; i++) {
        if (cancelRef.current) break;
        await new Promise((r) => setTimeout(r, 8));
        setMessages((m) => {
          const next = m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: msg.content + tokens[i] }
              : msg,
          );
          if (onProgress) {
            const cur = next.find((msg) => msg.id === assistantId);
            onProgress(cur?.content ?? "");
          }
          return next;
        });
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
      setDeckTitle(query);
      setDeckSlides([]);
      setDeckOpen(true);

      const assistantId = `a-${Date.now()}`;
      setMessages((m) => [
        ...m,
        { id: assistantId, role: "assistant", content: "" },
      ]);
      const slidesInstruction = [
        persona,
        "Format the response as slide deck content. Use one line starting with 'SLIDE: <short title>' per slide, followed by indented bullets starting with '-' for each point. Keep bullets short, specific, and free of markdown.",
      ]
        .filter(Boolean)
        .join(" ");

      try {
        const { text } = await answerQuery(query, {
          ...answerConfig,
          personaInstruction: slidesInstruction,
        });
        await reveal(assistantId, text, (content) => {
          setDeckSlides(parseSlides(content, query));
        });
        const final = parseSlides(text, query);
        setDeckSlides(final);
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: `Drafted a ${final.length}-slide deck in the editor — tweak any slide, then export it as .pptx.`,
                }
              : msg,
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
    [answerConfig, reveal, scrollEnd],
  );

  const exportDeck = useCallback(async () => {
    if (!deckSlides.length) return;
    try {
      const res = await fetch("/api/v1/export/pptx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slides: deckSlides }),
      });
      if (!res.ok)
        throw new Error(`Slide export failed (${res.status}).`);
      const blob = await res.blob();
      downloadFile(URL.createObjectURL(blob), "askdocs-deck.pptx");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [deckSlides]);

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
            {m.role === "assistant" && m.deckReady && (
              <button
                type="button"
                onClick={() => setDeckOpen(true)}
                className="btn-ghost mt-2 px-3 py-1 text-xs"
              >
                Open slide deck
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
        <div className="mb-2 flex flex-wrap items-center gap-1">
          <div role="group" aria-label="Task to run" className="flex flex-wrap gap-1">
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
          <div className="ml-auto">
            <PersonaPicker />
          </div>
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

      {/* Live slide deck editor (artifact studio) */}
      <Drawer
        open={deckOpen}
        title={deckTitle ? `Slide deck — ${deckTitle}` : "Slide deck"}
        onClose={() => setDeckOpen(false)}
      >
        <DeckEditor
          slides={deckSlides}
          busy={busy}
          onChange={setDeckSlides}
          onExport={exportDeck}
        />
      </Drawer>
    </div>
  );
}