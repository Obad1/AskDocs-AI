import React, { useCallback, useRef, useState } from "react";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { answerQuery } from "../../lib/llm/engine";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationDrawer } from "./CitationDrawer";
import type { MODE, CONFIDENCE_LEVEL, RetrievedChunk } from "../../types/schema";

interface Msg {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: CONFIDENCE_LEVEL;
  score?: number;
  chunks?: RetrievedChunk[];
}

export function ChatPane() {
  const { state } = useModelEngine();
  const { ws, setActiveMode } = useWorkspace();

  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const send = useCallback(async () => {
    const query = input.trim();
    if (!query || busy) return;
    setError(null);
    const userMsg: Msg = { id: `u-${Date.now()}`, role: "user", content: query };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setBusy(true);

    const assistantId = `a-${Date.now()}`;
    setMessages((m) => [
      ...m,
      { id: assistantId, role: "assistant", content: "" },
    ]);

    try {
      const { text, result } = await answerQuery(query, {
        mode: ws.active_mode,
        useReranker: state.reranker_enabled,
        backend: state.active_backend,
        modelId: state.llm_model,
        threshold: ws.confidence_threshold,
      });
      // Reveal progressively for a streaming-like feel.
      const tokens = text.split(/(\s+)/);
      for (let i = 0; i < tokens.length; i++) {
        await new Promise((r) => setTimeout(r, 8));
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content: msg.content + tokens[i],
                  confidence: result.confidence,
                  score: result.score,
                  chunks: result.chunks,
                }
              : msg,
          ),
        );
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantId
            ? { ...msg, content: `⚠ ${msg.content || "Generation failed."}` }
            : msg,
        ),
      );
    } finally {
      setBusy(false);
      requestAnimationFrame(() =>
        scrollRef.current?.scrollTo({ top: 1e9, behavior: "smooth" }),
      );
    }
  }, [input, busy, ws.active_mode, ws.confidence_threshold, state]);

  return (
    <div className="flex h-full flex-col rounded-lg border border-gray-200 dark:border-gray-700">
      {/* Verification toggle */}
      <div className="flex items-center gap-3 border-b border-gray-200 px-3 py-2 dark:border-gray-700">
        <span className="text-xs font-medium text-gray-500">Verification:</span>
        <div className="inline-flex overflow-hidden rounded-md border border-gray-300 dark:border-gray-600">
          {(["StrictDocumentOnly", "ExpandedAI"] as MODE[]).map((m) => (
            <button
              key={m}
              onClick={() => setActiveMode(m)}
              className={`px-3 py-1 text-xs ${
                ws.active_mode === m
                  ? "bg-blue-600 text-white"
                  : "bg-white text-gray-600 dark:bg-gray-800 dark:text-gray-300"
              }`}
            >
              {m === "StrictDocumentOnly" ? "Strict (Docs only)" : "Expanded (AI + docs)"}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 && (
          <p className="text-sm text-gray-400">
            Ask a question about your documents. Answers are generated 100% locally.
          </p>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            className={`rounded-lg p-3 text-sm ${
              m.role === "user"
                ? "ml-auto max-w-[85%] bg-blue-50 dark:bg-blue-900/30"
                : "mr-auto max-w-[90%] bg-gray-50 dark:bg-gray-800/60"
            }`}
          >
            {m.content || (busy ? "Thinking…" : "")}
            {m.role === "assistant" && m.confidence && (
              <div className="mt-2">
                <ConfidenceBadge level={m.confidence} score={m.score} />
              </div>
            )}
            {m.role === "assistant" && m.chunks && m.chunks.length > 0 && (
              <div className="mt-2">
                <CitationDrawer chunks={m.chunks} />
              </div>
            )}
          </div>
        ))}
        {error && (
          <div className="rounded-md border border-red-300 bg-red-50 p-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
            {error}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-2 border-t border-gray-200 p-3 dark:border-gray-700">
        <input
          className="flex-1 rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-900"
          placeholder="Ask your documents…"
          value={input}
          disabled={busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") send();
          }}
        />
        <button
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          onClick={send}
          disabled={busy}
        >
          {busy ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
