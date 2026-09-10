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

/** Normalized model-download progress reported to callers. */
export interface EmbedProgress {
  status: string;
  file?: string;
  loaded?: number;
  total?: number;
  /** 0-100 when the runtime reports an overall fraction. */
  percent?: number;
}

const pipelines = new Map<string, Extractor>();

async function getPipeline(
  modelId: string,
  onProgress?: (p: EmbedProgress) => void,
): Promise<Extractor> {
  let p = pipelines.get(modelId);
  if (!p) {
    // progress_callback fires for each downloaded artifact (config.json,
    // tokenizer, the quantized ONNX weights) on the first load only.
    p = await pipeline("feature-extraction", modelId, {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      progress_callback: (info: any) => {
        onProgress?.({
          status: info?.status ?? "",
          file: info?.file,
          loaded: info?.loaded,
          total: info?.total,
          percent:
            typeof info?.progress === "number"
              ? Math.round(info.progress * 100)
              : undefined,
        });
      },
    });
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
  embed(
    texts: string[],
    modelId?: string,
    onProgress?: (p: EmbedProgress) => void,
  ): Promise<number[][]>;
  embedQuery(
    text: string,
    modelId?: string,
    onProgress?: (p: EmbedProgress) => void,
  ): Promise<number[]>;
}

const api: EmbeddingApi = {
  async embed(
    texts: string[],
    modelId?: string,
    onProgress?: (p: EmbedProgress) => void,
  ): Promise<number[][]> {
    const mid = modelId || "Xenova/all-MiniLM-L6-v2";
    const extractor = await getPipeline(mid, onProgress);
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

  async embedQuery(
    text: string,
    modelId?: string,
    onProgress?: (p: EmbedProgress) => void,
  ): Promise<number[]> {
    const [vec] = await api.embed([text], modelId, onProgress);
    return vec;
  },
};

expose(api);
