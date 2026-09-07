// Comlink worker exposing local, zero-auth embedding via Transformers.js.
// Runs ONNX-quantized sentence-embedding models from public HF repos only.
// Contracts relied on: @xenova/transformers (feature-extraction pipeline),
// comlink (Comlink.expose). Feature vectors are L2-normalized.

import { pipeline, env } from "@xenova/transformers";
import { expose } from "comlink";

// Allow downloading models directly from the public HuggingFace hub (no auth).
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;

type Extractor = any;

const pipelines = new Map<string, Extractor>();

async function getPipeline(modelId: string): Promise<Extractor> {
  let p = pipelines.get(modelId);
  if (!p) {
    p = await pipeline("feature-extraction", modelId);
    pipelines.set(modelId, p);
  }
  return p;
}

function l2Normalize(vec: number[]): number[] {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  return vec.map((v) => v / norm);
}

export interface EmbeddingApi {
  embed(texts: string[], modelId?: string): Promise<number[][]>;
  embedQuery(text: string, modelId?: string): Promise<number[]>;
}

const api: EmbeddingApi = {
  async embed(texts: string[], modelId?: string): Promise<number[][]> {
    const mid = modelId || "Xenova/all-MiniLM-L6-v2";
    const extractor = await getPipeline(mid);
    const out = await extractor(texts, { pooling: "mean", normalize: true });
    const data = out.data as Float32Array;
    const dim = Math.floor(data.length / texts.length) || data.length;
    const result: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      const slice = Array.from(data.subarray(i * dim, (i + 1) * dim));
      result.push(l2Normalize(slice));
    }
    return result;
  },

  async embedQuery(text: string, modelId?: string): Promise<number[]> {
    const [vec] = await api.embed([text], modelId);
    return vec;
  },
};

expose(api);
