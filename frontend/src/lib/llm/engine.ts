// Local LLM answer/summary engine (spec §6.3 generation path).
//
// answerQuery: Query -> Hybrid Retrieval (RAG agent's retrieve()) -> Context
//   Assembly -> Local LLM (WebLLM/Ollama/llama.cpp) -> Confidence Scoring ->
//   { text, result: QueryResult }.
//
// Dependencies owned by other agents:
//   - retrieve() from ../retrieval/retrieve (RAG agent). Imported, not defined here.

import type {
  MODE,
  ENGINE_BACKEND,
  MODELID,
  RetrievedChunk,
  QueryResult,
  CONFIDENCE_LEVEL,
} from "../../types/schema";
import { retrieve } from "../retrieval/retrieve";
import { computeRetrievalConfidence } from "./confidence";
import { getLLMWorker } from "./workerClient";

export type RoleProfile =
  | "General"
  | "Student"
  | "Researcher"
  | "Executive"
  | "Legal";

export interface AnswerOpts {
  mode: MODE;
  useReranker?: boolean;
  backend: ENGINE_BACKEND;
  modelId: MODELID;
  /** Confidence threshold from workspace (default 0.5). */
  threshold?: number;
  temperature?: number;
  onToken?: (delta: string) => void;
  /** Explanation persona (system-instruction suffix). Empty = standard. */
  personaInstruction?: string;
}

function assembleContext(chunks: RetrievedChunk[]): string {
  if (chunks.length === 0) return "(no retrieved context)";
  return chunks
    .map(
      (c) =>
        `[Doc_ID: ${c.docId} | Page_No: ${c.page ?? "n/a"} | Score: ${c.score.toFixed(
          3,
        )}]\n${c.text}`,
    )
    .join("\n\n---\n\n");
}

export async function answerQuery(
  query: string,
  opts: AnswerOpts,
): Promise<{ text: string; result: QueryResult }> {
  const threshold = opts.threshold ?? 0.5;
  const chunks = await retrieve(query, { useReranker: opts.useReranker ?? false });

  const topScore = chunks.length ? chunks[0].score : 0;
  const { level, score } = computeRetrievalConfidence(topScore, threshold);

  const context = assembleContext(chunks);

  const systemPrompt = [
    opts.mode === "StrictDocumentOnly"
      ? [
          "You are a strict document-grounded assistant for AskDocs AI.",
          "Answer ONLY using the provided document context.",
          "If the answer is not contained in the context, respond exactly:",
          '"I could not find evidence for that in the provided documents."',
          "Do not use outside knowledge in Strict mode.",
        ].join(" ")
      : [
          "You are AskDocs AI, a helpful assistant.",
          "Prefer the provided document context. You MAY supplement with general knowledge,",
          "but clearly mark any claim that is NOT grounded in the provided context with [general knowledge].",
        ].join(" "),
    opts.personaInstruction ? `Explanation persona: ${opts.personaInstruction}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  const userPrompt = `DOCUMENT CONTEXT:\n${context}\n\nQUESTION: ${query}`;

  const worker = getLLMWorker();
  const text = await worker.generate(
    [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
    {
      backend: opts.backend,
      modelId: opts.modelId,
      temperature: opts.temperature ?? 0.2,
    },
  );

  const result: QueryResult = {
    retrieved_chunks: chunks.map((c) => c.chunkId),
    confidence: level as CONFIDENCE_LEVEL,
    score,
    chunks,
  };

  return { text, result };
}

const GRANULARITY_INSTRUCTION: Record<number, string> = {
  1: "Produce a SINGLE one-line summary (Level 1).",
  2: "Produce a 3-paragraph summary (Level 2).",
  3: "Produce a structured section outline with headings and bullet points (Level 3).",
};

const PROFILE_INSTRUCTION: Record<RoleProfile, string> = {
  General: "Write for a general audience.",
  Student: "Write for a student studying the material; emphasize key concepts.",
  Researcher: "Write for a researcher; preserve nuance and methodology.",
  Executive: "Write for an executive; lead with conclusions and actionable takeaways.",
  Legal: "Write for a legal reviewer; be precise about obligations and caveats.",
};

export async function summarize(
  text: string,
  granularity: number,
  profile: RoleProfile,
  opts: {
    backend: ENGINE_BACKEND;
    modelId: MODELID;
    temperature?: number;
  },
): Promise<string> {
  const g = Math.min(3, Math.max(1, Math.round(granularity)));
  const system = [
    "You are AskDocs AI's summarization engine.",
    GRANULARITY_INSTRUCTION[g],
    PROFILE_INSTRUCTION[profile] ?? PROFILE_INSTRUCTION.General,
    "Also emit 2-3 'Actionable Extract' callouts as lines prefixed with 'ACTION:'.",
  ].join(" ");

  const worker = getLLMWorker();
  return worker.generate(
    [
      { role: "system", content: system },
      { role: "user", content: `TEXT TO SUMMARIZE:\n${text}` },
    ],
    {
      backend: opts.backend,
      modelId: opts.modelId,
      temperature: opts.temperature ?? 0.3,
    },
  );
}
