// Local-LLM quiz + flashcard generation with grammar-constrained JSON output.
// (spec §4.4 / §4.8: Ollama `format: "json"` or WebLLM JSON mode.)

import type { ENGINE_BACKEND, MODELID, CHUNKID, DOCID } from "../../types/schema";
import { getLLMWorker } from "./workerClient";

export type QuizType = "MCQ" | "TrueFalse" | "FillIn" | "Matching";
export type Difficulty = "Easy" | "Medium" | "Hard";

export interface QuizQuestion {
  id: string; // stable id (we assign if the model omits)
  type: QuizType;
  prompt: string;
  options?: string[]; // MCQ
  pairs?: { left: string; right: string }[]; // Matching
  answer: string | string[]; // string for MCQ/TF/Fill, array for Matching
  explanation?: string;
  difficulty: Difficulty;
  topic?: string;
}

export interface FlashcardSpec {
  chunkId?: CHUNKID;
  docId?: DOCID;
  front: string;
  back: string;
}

export interface GenOpts {
  backend: ENGINE_BACKEND;
  modelId: MODELID;
  temperature?: number;
}

export interface SourceChunk {
  text: string;
  docId?: DOCID;
  chunkId?: CHUNKID;
  page?: number;
}

function contextBlock(chunks: SourceChunk[]): string {
  return chunks
    .map(
      (c, i) =>
        `CHUNK ${i + 1}${c.docId ? ` (Doc: ${c.docId})` : ""}${
          c.page ? ` [p.${c.page}]` : ""
        }:\n${c.text}`,
    )
    .join("\n\n");
}

const TYPE_INSTRUCTION: Record<QuizType, string> = {
  MCQ: 'Generate multiple-choice questions. Each item: {"type":"MCQ","prompt":string,"options":[4 strings],"answer":string (must be one of options),"explanation":string,"topic":string}.',
  TrueFalse:
    'Generate true/false questions. Each item: {"type":"TrueFalse","prompt":string,"answer":"True"|"False","explanation":string,"topic":string}.',
  FillIn:
    'Generate fill-in-the-blank questions. Each item: {"type":"FillIn","prompt":string (with "___" for blank),"answer":string,"explanation":string,"topic":string}.',
  Matching:
    'Generate a matching set. Return ONE item of type "Matching" with "pairs":[{"left":term,"right":definition}] and "answer":[left1,right1,left2,right2,...] in matched order.',
};

export async function generateQuiz(
  contextChunks: SourceChunk[],
  type: QuizType,
  difficulty: Difficulty,
  opts: GenOpts,
): Promise<QuizQuestion[]> {
  const context = contextBlock(contextChunks);
  const system = [
    "You are AskDocs AI's quiz generator.",
    `Create ${difficulty} difficulty questions based ONLY on the provided chunks.`,
    "Return a JSON object: {\"questions\": [ ... ]}.",
    TYPE_INSTRUCTION[type],
    "Return ONLY JSON.",
  ].join(" ");

  const worker = getLLMWorker();
  // Comlink's Remote type drops method generics, so cast at the call site.
  const data = (await (worker as any).generateJSON(
    [
      { role: "system", content: system },
      { role: "user", content: `CONTEXT:\n${context}` },
    ],
    { backend: opts.backend, modelId: opts.modelId, temperature: opts.temperature ?? 0.5 },
  )) as { questions?: QuizQuestion[] };

  const list = data.questions ?? [];
  return list.map((q, i) => ({
    ...q,
    id: q.id ?? `${type}-${i}-${Date.now()}`,
    difficulty,
  }));
}

export async function generateFlashcards(
  chunks: SourceChunk[],
): Promise<FlashcardSpec[]> {
  if (chunks.length === 0) return [];
  const context = contextBlock(chunks);
  const worker = getLLMWorker();
  const data = (await (worker as any).generateJSON(
    [
      {
        role: "system",
        content:
          "You are AskDocs AI's flashcard generator. Create study flashcards from the chunks. " +
          'Return JSON: {"flashcards":[{"front":string,"back":string}]}. Front = question/clue, back = answer. Return ONLY JSON.',
      },
      { role: "user", content: `CONTEXT:\n${context}` },
    ],
    { backend: "Ollama_Local", modelId: "llama3.1:8b" },
  )) as { flashcards?: FlashcardSpec[] };

  return (data.flashcards ?? []).map((f) => ({
    ...f,
    chunkId: f.chunkId,
    docId: f.docId,
  }));
}
