import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useModelEngine } from "../../context/ModelEngineContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import {
  generateQuiz,
  type QuizQuestion,
  type QuizType,
  type Difficulty,
  type SourceChunk,
} from "../../lib/llm/quiz";

const TYPES: QuizType[] = ["MCQ", "TrueFalse", "FillIn", "Matching"];

interface Result {
  question: QuizQuestion;
  correct: boolean;
}

const NEXT_DIFFICULTY: Record<Difficulty, Difficulty> = {
  Easy: "Medium",
  Medium: "Hard",
  Hard: "Hard",
};
const PREV_DIFFICULTY: Record<Difficulty, Difficulty> = {
  Easy: "Easy",
  Medium: "Easy",
  Hard: "Medium",
};

export function AdaptiveQuiz() {
  const { state } = useModelEngine();
  const { ws } = useWorkspace();

  const [type, setType] = useState<QuizType>("MCQ");
  const [difficulty, setDifficulty] = useState<Difficulty>("Easy");
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const allChunks: SourceChunk[] = useMemo(
    () =>
      Object.values(ws.chunks).map((c) => ({
        text: c.text,
        docId: c.docId,
        chunkId: c.id,
        page: c.page,
      })),
    [ws.chunks],
  );

  const generate = useCallback(
    async (overrideType?: QuizType, chunks?: SourceChunk[]) => {
      if (allChunks.length === 0) {
        setError("Ingest documents first to generate a quiz.");
        return;
      }
      setBusy(true);
      setError(null);
      try {
        const qs = await generateQuiz(
          chunks ?? allChunks,
          overrideType ?? type,
          difficulty,
          { backend: state.active_backend, modelId: state.llm_model },
        );
        setQuestions(qs);
        setResults([]);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [allChunks, type, difficulty, state],
  );

  const record = useCallback((q: QuizQuestion, correct: boolean) => {
    setResults((r) => [...r, { question: q, correct }]);
  }, []);

  const weakTopics = useMemo(() => {
    const wrong = results.filter((r) => !r.correct).map((r) => r.question.topic);
    return Array.from(new Set(wrong.filter(Boolean))) as string[];
  }, [results]);

  const score = results.length
    ? Math.round((results.filter((r) => r.correct).length / results.length) * 100)
    : 0;

  // Auto-scale difficulty once the current set is fully answered (visible in
  // the Level dropdown, so the behavior is honest and observable).
  useEffect(() => {
    if (questions.length === 0 || results.length !== questions.length) return;
    if (score >= 80) setDifficulty((d) => NEXT_DIFFICULTY[d]);
    else if (score <= 40) setDifficulty((d) => PREV_DIFFICULTY[d]);
  }, [results.length, questions.length, score]);

  const remedial = useCallback(async () => {
    if (weakTopics.length === 0) return;
    const focused = allChunks.filter((c) =>
      weakTopics.some((t) => c.text.toLowerCase().includes(t!.toLowerCase())),
    );
    await generate(type, focused.length ? focused : allChunks);
  }, [weakTopics, allChunks, generate, type]);

  return (
    <div className="surface-card space-y-3 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1 text-xs">
          <span className="text-[var(--fg-muted)]">Type</span>
          <select
            className="field px-2 py-1.5 text-xs"
            value={type}
            onChange={(e) => setType(e.target.value as QuizType)}
          >
            {TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs">
          <span className="text-[var(--fg-muted)]">Level</span>
          <select
            className="field px-2 py-1.5 text-xs"
            aria-label="Difficulty level"
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value as Difficulty)}
          >
            {(["Easy", "Medium", "Hard"] as Difficulty[]).map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <button
          className="btn-primary px-3 py-1.5 text-xs"
          onClick={() => generate()}
          disabled={busy}
        >
          {busy ? "Generating…" : "Generate quiz"}
        </button>
        {weakTopics.length > 0 && (
          <button
            className="btn-ghost rounded-md border border-[var(--danger)] px-3 py-1.5 text-xs font-medium text-[var(--danger-fg)]"
            onClick={remedial}
            disabled={busy}
          >
            Remedial on weak areas
          </button>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-[var(--danger)] bg-[var(--danger-soft)] p-2 text-xs text-[var(--danger-fg)]">
          {error}
        </div>
      )}

      {questions.map((q) => (
        <QuizItem key={q.id} q={q} onAnswered={record} />
      ))}

      {results.length > 0 && (
        <div className="text-sm">
          Score: <b>{score}%</b> ({results.filter((r) => r.correct).length}/
          {results.length}) · Difficulty auto-adjusts once you finish a set
          (80%+ up, 40%- down).
        </div>
      )}
    </div>
  );
}

function QuizItem({
  q,
  onAnswered,
}: {
  q: QuizQuestion;
  onAnswered: (q: QuizQuestion, correct: boolean) => void;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  const [fill, setFill] = useState("");
  const [matched, setMatched] = useState(false);

  if (q.type === "MCQ") {
    return (
      <div className="rounded-md border border-[var(--border)] p-3">
        <div className="text-sm font-medium">{q.prompt}</div>
        <div className="mt-2 space-y-1">
          {(q.options ?? []).map((opt) => {
            const isAnswer = opt === q.answer;
            const isPicked = picked === opt;
            const tone =
              picked && isAnswer
                ? "bg-[var(--success-soft)] text-[var(--success-fg)]"
                : picked && isPicked && !isAnswer
                  ? "bg-[var(--danger-soft)] text-[var(--danger-fg)]"
                  : "bg-[var(--bg-sunken)] text-[var(--fg)]";
            return (
              <button
                key={opt}
                disabled={!!picked}
                className={`block w-full rounded px-2 py-1 text-left text-sm ${tone}`}
                onClick={() => {
                  setPicked(opt);
                  onAnswered(q, isAnswer);
                }}
              >
                {opt}
              </button>
            );
          })}
        </div>
        {picked && q.explanation && (
          <div className="mt-2 text-xs text-[var(--fg-muted)]">{q.explanation}</div>
        )}
      </div>
    );
  }

  if (q.type === "TrueFalse") {
    return (
      <div className="rounded-md border border-[var(--border)] p-3">
        <div className="text-sm font-medium">{q.prompt}</div>
        <div className="mt-2 flex gap-2">
          {["True", "False"].map((opt) => {
            const correct = opt === q.answer;
            const tone =
              picked && correct
                ? "bg-[var(--success-soft)] text-[var(--success-fg)]"
                : picked && picked === opt && !correct
                  ? "bg-[var(--danger-soft)] text-[var(--danger-fg)]"
                  : "bg-[var(--bg-sunken)] text-[var(--fg)]";
            return (
              <button
                key={opt}
                disabled={!!picked}
                className={`rounded px-3 py-1 text-sm ${tone}`}
                onClick={() => {
                  setPicked(opt);
                  onAnswered(q, correct);
                }}
              >
                {opt}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  if (q.type === "FillIn") {
    const submit = () => {
      const ok =
        fill.trim().toLowerCase() === String(q.answer).trim().toLowerCase();
      setPicked(fill);
      onAnswered(q, ok);
    };
    return (
      <div className="rounded-md border border-[var(--border)] p-3">
        <div className="text-sm font-medium">{q.prompt}</div>
        <div className="mt-2 flex gap-2">
          <input
            aria-label="Type your answer"
            className="field flex-1 px-2 py-1 text-sm"
            value={fill}
            disabled={!!picked}
            onChange={(e) => setFill(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <button
            disabled={!!picked}
            className="btn-primary px-3 py-1 text-sm"
            onClick={submit}
          >
            Check
          </button>
        </div>
        {picked && (
          <div className="mt-1 text-xs text-[var(--fg-muted)]">
            Answer: <b>{String(q.answer)}</b>
            {q.explanation ? ` — ${q.explanation}` : ""}
          </div>
        )}
      </div>
    );
  }

  // Matching
  return (
    <div className="rounded-md border border-[var(--border)] p-3">
      <div className="text-sm font-medium">{q.prompt || "Match the pairs"}</div>
      <ul className="mt-2 space-y-1 text-sm">
        {(q.pairs ?? []).map((p, i) => (
          <li key={i} className="flex justify-between gap-2">
            <span className="font-mono text-xs">{p.left}</span>
            <span>→</span>
            <span className="text-xs text-[var(--fg-muted)]">
              {matched ? p.right : "?"}
            </span>
          </li>
        ))}
      </ul>
      <button
        className="btn-ghost mt-2 px-3 py-1 text-xs"
        onClick={() => {
          setMatched(true);
          onAnswered(q, false);
        }}
      >
        {matched ? "Answer shown" : "Reveal & mark as not correct"}
      </button>
    </div>
  );
}
