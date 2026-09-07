import React, { useCallback, useMemo, useState } from "react";
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

  // Auto-scale difficulty once the current set is fully answered.
  const maybeRescale = useCallback(() => {
    if (results.length !== questions.length) return;
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
    <div className="space-y-3 rounded-lg border border-gray-200 p-3 dark:border-gray-700">
      <div className="flex flex-wrap items-center gap-2">
        <select
          className="rounded-md border border-gray-300 px-2 py-1 text-xs dark:border-gray-600 dark:bg-gray-900"
          value={type}
          onChange={(e) => setType(e.target.value as QuizType)}
        >
          {TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <span className="text-xs text-gray-500">Difficulty: {difficulty}</span>
        <button
          className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
          onClick={() => generate()}
          disabled={busy}
        >
          {busy ? "Generating…" : "Generate quiz"}
        </button>
        {weakTopics.length > 0 && (
          <button
            className="rounded-md border border-red-300 px-3 py-1.5 text-xs font-medium text-red-600 dark:border-red-700 dark:text-red-300"
            onClick={remedial}
            disabled={busy}
          >
            Remedial on weak areas
          </button>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 p-2 text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      {questions.map((q) => (
        <QuizItem key={q.id} q={q} onAnswered={record} />
      ))}

      {results.length > 0 && (
        <div className="text-sm">
          Score: <b>{score}%</b> ({results.filter((r) => r.correct).length}/
          {results.length}) · Next difficulty will adapt automatically.
          <button
            className="ml-3 rounded bg-gray-100 px-2 py-1 text-xs dark:bg-gray-800"
            onClick={maybeRescale}
          >
            Apply difficulty scaling
          </button>
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
      <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
        <div className="text-sm font-medium">{q.prompt}</div>
        <div className="mt-2 space-y-1">
          {(q.options ?? []).map((opt) => {
            const isAnswer = opt === q.answer;
            const isPicked = picked === opt;
            const tone =
              picked && isAnswer
                ? "bg-green-100 dark:bg-green-900/40"
                : picked && isPicked && !isAnswer
                  ? "bg-red-100 dark:bg-red-900/40"
                  : "bg-gray-50 dark:bg-gray-800/60";
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
          <div className="mt-2 text-xs text-gray-500">{q.explanation}</div>
        )}
      </div>
    );
  }

  if (q.type === "TrueFalse") {
    return (
      <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
        <div className="text-sm font-medium">{q.prompt}</div>
        <div className="mt-2 flex gap-2">
          {["True", "False"].map((opt) => {
            const correct = opt === q.answer;
            const tone =
              picked && correct
                ? "bg-green-100 dark:bg-green-900/40"
                : picked && picked === opt && !correct
                  ? "bg-red-100 dark:bg-red-900/40"
                  : "bg-gray-50 dark:bg-gray-800/60";
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
      <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
        <div className="text-sm font-medium">{q.prompt}</div>
        <div className="mt-2 flex gap-2">
          <input
            className="flex-1 rounded border border-gray-300 px-2 py-1 text-sm dark:border-gray-600 dark:bg-gray-900"
            value={fill}
            disabled={!!picked}
            onChange={(e) => setFill(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <button
            disabled={!!picked}
            className="rounded bg-blue-600 px-3 py-1 text-sm text-white"
            onClick={submit}
          >
            Check
          </button>
        </div>
        {picked && (
          <div className="mt-1 text-xs text-gray-500">
            Answer: <b>{String(q.answer)}</b>
            {q.explanation ? ` — ${q.explanation}` : ""}
          </div>
        )}
      </div>
    );
  }

  // Matching
  return (
    <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
      <div className="text-sm font-medium">{q.prompt || "Match the pairs"}</div>
      <ul className="mt-2 space-y-1 text-sm">
        {(q.pairs ?? []).map((p, i) => (
          <li key={i} className="flex justify-between gap-2">
            <span className="font-mono text-xs">{p.left}</span>
            <span>→</span>
            <span className="text-xs text-gray-500">
              {matched ? p.right : "?"}
            </span>
          </li>
        ))}
      </ul>
      <button
        className="mt-2 rounded bg-gray-100 px-3 py-1 text-xs dark:bg-gray-800"
        onClick={() => {
          setMatched(true);
          onAnswered(q, true);
        }}
      >
        {matched ? "Reveal shown" : "Reveal & mark attempted"}
      </button>
    </div>
  );
}
