// Cross-encoder reranking via Transformers.js (Xenova/ms-marco-MiniLM-L-6-v2).
// The ms-marco model is used in feature-extraction mode: query and candidate
// are embedded (mean-pooled, L2-normalized) and their dot product is the
// relevance score. Falls back to the original ranking on any failure.

import { pipeline, env } from "@xenova/transformers";

env.allowLocalModels = false;

let extractor: any = null;

async function getExtractor(modelId: string): Promise<any> {
  if (!extractor) {
    extractor = await pipeline("feature-extraction", modelId);
  }
  return extractor;
}

export interface RerankCandidate {
  id: string;
  text: string;
}

export async function rerank(
  query: string,
  candidates: RerankCandidate[],
  modelId = "Xenova/ms-marco-MiniLM-L-6-v2",
  topN = 10,
): Promise<{ id: string; score: number }[]> {
  if (!candidates.length) return [];

  try {
    const ext = await getExtractor(modelId);

    const qOut = await ext(query, { pooling: "mean", normalize: true });
    const qVec = Array.from(qOut.data as Float32Array);

    const scored = [];
    for (const c of candidates) {
      const out = await ext(c.text, { pooling: "mean", normalize: true });
      const v = Array.from(out.data as Float32Array);
      let dot = 0;
      for (let i = 0; i < qVec.length; i++) dot += qVec[i] * v[i];
      scored.push({ id: c.id, score: dot });
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topN);
  } catch {
    // Robust fallback: preserve original order, neutral scores.
    return candidates.slice(0, topN).map((c) => ({ id: c.id, score: 0 }));
  }
}
