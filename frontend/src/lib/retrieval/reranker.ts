// True sentence-pair cross-encoder reranking, not embedding cosine similarity.
// The ONNX model and tokenizer run locally through Transformers.js.
// This is opt-in because loading a cross-encoder on low-end phones is expensive.
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
} from "@xenova/transformers";

env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = "/ort/";
env.backends.onnx.wasm.numThreads = 1;

const DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";
let loadedModel: string | null = null;
let loading: Promise<{
  tokenizer: any;
  model: any;
}> | null = null;

async function load(modelId: string) {
  if (modelId !== loadedModel) {
    loadedModel = modelId;
    loading = Promise.all([
      AutoTokenizer.from_pretrained(modelId),
      AutoModelForSequenceClassification.from_pretrained(modelId, { quantized: true }),
    ]).then(([tokenizer, model]) => ({ tokenizer, model }))
      .catch((error) => {
        loading = null;
        loadedModel = null;
        throw error;
      });
  }
  return loading!;
}

export interface RerankCandidate {
  id: string;
  text: string;
}

export async function rerank(
  query: string,
  candidates: RerankCandidate[],
  modelId = DEFAULT_MODEL,
  topN = 10,
): Promise<{ id: string; score: number }[]> {
  if (!candidates.length) return [];
  const { tokenizer, model } = await load(modelId);
  // Cross-encoders score [query, passage] pairs in a single forward pass.
  const features = tokenizer(
    candidates.map(() => query),
    {
      text_pair: candidates.map((candidate) => candidate.text),
      truncation: true,
      padding: true,
    },
  );
  const outputs = await model(features);
  const logits = Array.from(outputs.logits.data as Float32Array);
  if (logits.length !== candidates.length || logits.some((n) => !Number.isFinite(n))) {
    throw new Error("Cross-encoder returned invalid sentence-pair scores.");
  }
  return candidates
    .map((candidate, i) => ({ id: candidate.id, score: logits[i] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}
