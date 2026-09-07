// Local LLM inference worker (zero-API-key, zero-login).
//
// Exposes (via Comlink):
//   - loadModel(modelId, backend)            -> preload a model for the given backend
//   - generate(messages, opts)               -> single completion (string)
//   - generateJSON(messages, opts)           -> grammar-constrained JSON object/array
//   - ping()                                 -> { ok, backend }
//
// Backends:
//   - WebGPU_WebLLM : @mlc-ai/web-llm (CreateMLCEngine) running client-side on WebGPU.
//   - Ollama_Local  : POST http://localhost:11434/api/chat with `format: "json"` for JSON.
//   - LlamaCpp_Local: POST http://localhost:8080/v1/chat/completions (OpenAI-compatible).
//
// IMPORTANT: This worker NEVER imports React context. The caller (engine.ts) is
// responsible for passing `backend` and `modelId` resolved from useModelEngine().

import * as Comlink from "comlink";
import type { ENGINE_BACKEND } from "../types/schema";

export type ChatRole = "system" | "user" | "assistant";
export interface ChatMessage {
  role: ChatRole;
  content: string;
}
export interface GenerateOpts {
  json?: boolean;
  temperature?: number;
  backend?: ENGINE_BACKEND;
  modelId?: string;
  /** llama.cpp server base URL (defaults to localhost:8080). */
  llamaBaseUrl?: string;
}

const OLLAMA_URL = "http://localhost:11434/api/chat";
const LLAMACPP_URL = "http://localhost:8080/v1/chat/completions";

// Best-effort mapping from catalog model id -> WebLLM prebuilt model id.
const WEBLLM_MODEL_MAP: Record<string, string> = {
  "phi3.5:3.8b-mini-instruct-q4_K_M": "Phi-3.5-mini-instruct-q4f16_1-MLC",
  "llama3.1:8b-instruct-q4_K_M": "Llama-3.1-8B-Instruct-q4f32_1-MLC",
  "qwen2.5:14b-instruct-q4_K_M": "Qwen2.5-14B-Instruct-q4f32_1-MLC",
  "llama3.1:70b-instruct-q4_K_M": "Llama-3.1-70B-Instruct-q4f32_1-MLC",
};

function toWebLLMModelId(modelId: string | undefined): string {
  if (!modelId) return "Llama-3.1-8B-Instruct-q4f32_1-MLC";
  if (WEBLLM_MODEL_MAP[modelId]) return WEBLLM_MODEL_MAP[modelId];
  // Already a WebLLM id, or let the engine attempt as-is.
  if (modelId.endsWith("-MLC")) return modelId;
  return "Llama-3.1-8B-Instruct-q4f32_1-MLC";
}

// ---- WebLLM (lazy singleton) ----
let webllmEngine: any = null;
let webllmLoadedId: string | null = null;

async function getWebLLMEngine(modelId: string): Promise<any> {
  const target = toWebLLMModelId(modelId);
  if (webllmEngine && webllmLoadedId === target) return webllmEngine;
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const webllm = await import("@mlc-ai/web-llm");
  webllmEngine = await webllm.CreateMLCEngine(target, {
    initProgressCallback: () => {},
  });
  webllmLoadedId = target;
  return webllmEngine;
}

// ---- Ollama (streaming-free, JSON-capable) ----
async function ollamaGenerate(
  messages: ChatMessage[],
  opts: GenerateOpts,
): Promise<string> {
  const model = opts.modelId ?? "llama3.1:8b";
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: false,
    options: { temperature: opts.temperature ?? 0.2 },
  };
  if (opts.json) body.format = "json";
  const res = await fetch(OLLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Ollama request failed (${res.status}): ${txt}`);
  }
  const data = await res.json();
  const content: string = data?.message?.content ?? "";
  return content;
}

// ---- llama.cpp (OpenAI-compatible) ----
async function llamaCppGenerate(
  messages: ChatMessage[],
  opts: GenerateOpts,
): Promise<string> {
  const url = (opts.llamaBaseUrl ?? LLAMACPP_URL) + "/chat/completions";
  const body: Record<string, unknown> = {
    model: opts.modelId ?? "local",
    messages,
    temperature: opts.temperature ?? 0.2,
    stream: false,
  };
  if (opts.json) body.response_format = { type: "json_object" };
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`llama.cpp request failed (${res.status}): ${txt}`);
  }
  const data = await res.json();
  return data?.choices?.[0]?.message?.content ?? "";
}

const api = {
  async ping(): Promise<{ ok: boolean; backend?: ENGINE_BACKEND }> {
    return { ok: true };
  },

  async loadModel(
    modelId: string,
    backend: ENGINE_BACKEND,
  ): Promise<{ loaded: boolean; backend: ENGINE_BACKEND; modelId: string }> {
    try {
      if (backend === "WebGPU_WebLLM") {
        await getWebLLMEngine(modelId);
      } else if (backend === "Ollama_Local") {
        const res = await fetch("http://localhost:11434/api/tags");
        if (!res.ok) throw new Error("Ollama server unreachable");
      }
      return { loaded: true, backend, modelId };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`loadModel failed: ${msg}`);
    }
  },

  async generate(
    messages: ChatMessage[],
    opts: GenerateOpts = {},
  ): Promise<string> {
    const backend: ENGINE_BACKEND = opts.backend ?? "Ollama_Local";
    try {
      if (backend === "WebGPU_WebLLM") {
        const engine = await getWebLLMEngine(opts.modelId ?? "");
        const resp = await engine.chat.completions.create({
          messages,
          temperature: opts.temperature ?? 0.2,
          stream: false,
        });
        return resp?.choices?.[0]?.message?.content ?? "";
      }
      if (backend === "Ollama_Local") {
        return await ollamaGenerate(messages, opts);
      }
      if (backend === "LlamaCpp_Local") {
        return await llamaCppGenerate(messages, opts);
      }
      throw new Error(`Unknown backend: ${backend}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`generate failed (${backend}): ${msg}`);
    }
  },

  async generateJSON<T = unknown>(
    messages: ChatMessage[],
    opts: GenerateOpts = {},
  ): Promise<T> {
    const raw = await api.generate(messages, { ...opts, json: true });
    try {
      return JSON.parse(raw) as T;
    } catch {
      // Tolerate fenced code blocks sometimes returned by models.
      const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
      if (fenced) return JSON.parse(fenced[1]) as T;
      throw new Error("generateJSON: model did not return valid JSON");
    }
  },
};

export type LLMWorkerApi = typeof api;

Comlink.expose(api);
