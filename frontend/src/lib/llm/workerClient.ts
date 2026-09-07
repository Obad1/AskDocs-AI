// Thin Comlink client around the local LLM worker.
// Owns the single Worker instance and exposes a typed `getLLMWorker()` accessor.
import * as Comlink from "comlink";
import type { LLMWorkerApi } from "../../workers/llm.worker";

let worker: Worker | null = null;
let api: Comlink.Remote<LLMWorkerApi> | null = null;

function ensureWorker(): Comlink.Remote<LLMWorkerApi> {
  if (api) return api;
  // Vite resolves the .ts worker via `new URL(..., import.meta.url)`.
  worker = new Worker(
    new URL("../../workers/llm.worker.ts", import.meta.url),
    { type: "module" },
  );
  api = Comlink.wrap<LLMWorkerApi>(worker);
  return api;
}

export function getLLMWorker(): Comlink.Remote<LLMWorkerApi> {
  return ensureWorker();
}

export async function preloadModel(
  modelId: string,
  backend: Parameters<LLMWorkerApi["loadModel"]>[1],
): Promise<void> {
  await getLLMWorker().loadModel(modelId, backend);
}
