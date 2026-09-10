import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
} from "react";
import type {
  ModelEngineState,
  HARDWARE_TIER,
  ENGINE_BACKEND,
  MODELID,
} from "../types/schema";
import { loadModelEngine, saveModelEngine } from "../lib/storage/indexeddb";

// Local Model Catalog (spec §5). All from public, unauthenticated sources.
export interface ModelSpec {
  id: MODELID;
  label: string;
  sizeMB: number;
  source: "huggingface" | "ollama";
  hfRepo?: string;
  ollamaName?: string;
}

export const MODEL_CATALOG: Record<HARDWARE_TIER, {
  embedding: ModelSpec;
  llm: ModelSpec;
  tts: ModelSpec;
  stt: ModelSpec;
}> = {
  Tier0_Minimal: {
    embedding: { id: "Xenova/all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2", sizeMB: 30, source: "huggingface", hfRepo: "Xenova/all-MiniLM-L6-v2" },
    llm: { id: "phi3.5:3.8b-mini-instruct-q4_K_M", label: "Phi-3.5-mini (Q4_K_M)", sizeMB: 2200, source: "ollama", ollamaName: "phi3.5:3.8b-mini-instruct-q4_K_M" },
    tts: { id: "en_US-lessac-low", label: "Piper lessac-low", sizeMB: 20, source: "huggingface", hfRepo: "rhasspy/piper-voices" },
    stt: { id: "whisper-tiny.en", label: "Whisper tiny.en", sizeMB: 75, source: "huggingface", hfRepo: "Xenova/whisper-tiny.en" },
  },
  Tier1_Standard: {
    embedding: { id: "Xenova/bge-small-en-v1.5", label: "bge-small-en-v1.5", sizeMB: 33, source: "huggingface", hfRepo: "Xenova/bge-small-en-v1.5" },
    llm: { id: "llama3.1:8b-instruct-q4_K_M", label: "Llama-3.1-8B (Q4_K_M)", sizeMB: 4900, source: "ollama", ollamaName: "llama3.1:8b-instruct-q4_K_M" },
    tts: { id: "en_US-lessac-medium", label: "Piper lessac-medium", sizeMB: 60, source: "huggingface", hfRepo: "rhasspy/piper-voices" },
    stt: { id: "whisper-base.en", label: "Whisper base.en", sizeMB: 145, source: "huggingface", hfRepo: "Xenova/whisper-base.en" },
  },
  Tier2_Performance: {
    embedding: { id: "Xenova/bge-base-en-v1.5", label: "bge-base-en-v1.5", sizeMB: 109, source: "huggingface", hfRepo: "Xenova/bge-base-en-v1.5" },
    llm: { id: "qwen2.5:14b-instruct-q4_K_M", label: "Qwen2.5-14B (Q4_K_M)", sizeMB: 9000, source: "ollama", ollamaName: "qwen2.5:14b-instruct-q4_K_M" },
    tts: { id: "en_US-libritts-high", label: "Piper libritts-high", sizeMB: 110, source: "huggingface", hfRepo: "rhasspy/piper-voices" },
    stt: { id: "whisper-small.en", label: "Whisper small.en", sizeMB: 484, source: "huggingface", hfRepo: "Xenova/whisper-small.en" },
  },
  Tier3_Workstation: {
    embedding: { id: "Xenova/bge-large-en-v1.5", label: "bge-large-en-v1.5", sizeMB: 335, source: "huggingface", hfRepo: "Xenova/bge-large-en-v1.5" },
    llm: { id: "llama3.1:70b-instruct-q4_K_M", label: "Llama-3.1-70B (Q4_K_M)", sizeMB: 40000, source: "ollama", ollamaName: "llama3.1:70b-instruct-q4_K_M" },
    tts: { id: "en_US-multi-high", label: "Piper multi-speaker high", sizeMB: 120, source: "huggingface", hfRepo: "rhasspy/piper-voices" },
    stt: { id: "whisper-medium.en", label: "Whisper medium.en", sizeMB: 1500, source: "huggingface", hfRepo: "Xenova/whisper-medium.en" },
  },
};

export const RERANKER: ModelSpec = {
  id: "Xenova/ms-marco-MiniLM-L-6-v2",
  label: "ms-marco-MiniLM-L-6-v2",
  sizeMB: 23,
  source: "huggingface",
  hfRepo: "Xenova/ms-marco-MiniLM-L-6-v2",
};

export interface HardwareBenchmark {
  tier: HARDWARE_TIER;
  webgpu: boolean;
  cores: number;
  memoryMB: number;
  notes: string[];
}

async function detectHardware(): Promise<HardwareBenchmark> {
  const cores = navigator.hardwareConcurrency || 4;
  // approximation of available memory (not exact)
  const memoryMB = (navigator as any).deviceMemory
    ? (navigator as any).deviceMemory * 1024
    : 4096;
  const webgpu =
    typeof (navigator as any).gpu !== "undefined";
  let tier: HARDWARE_TIER = "Tier0_Minimal";
  if (webgpu && memoryMB >= 16384) tier = "Tier2_Performance";
  else if (webgpu && memoryMB >= 8192) tier = "Tier1_Standard";
  else if (!webgpu && memoryMB >= 16384) tier = "Tier2_Performance";
  return {
    tier,
    webgpu,
    cores,
    memoryMB,
    notes: [
      `cores=${cores}`,
      `memory~=${memoryMB}MB`,
      `webgpu=${webgpu}`,
    ],
  };
}

// Z §7.4 SelectModelTier
function selectTier(
  prev: ModelEngineState,
  detected_tier: HARDWARE_TIER,
  webgpu_present: boolean,
): ModelEngineState {
  const cat = MODEL_CATALOG[detected_tier];
  let active_backend: ENGINE_BACKEND;
  if (webgpu_present && detected_tier !== "Tier3_Workstation") {
    active_backend = "WebGPU_WebLLM";
  } else {
    active_backend = "Ollama_Local";
  }
  return {
    hardware_tier: detected_tier,
    active_backend,
    embedding_model: cat.embedding.id,
    llm_model: cat.llm.id,
    tts_model: cat.tts.id,
    stt_model: cat.stt.id,
    reranker_enabled: true,
    model_cached: {
      [cat.embedding.id]: prev.model_cached[cat.embedding.id] ?? false,
      [cat.llm.id]: prev.model_cached[cat.llm.id] ?? false,
      [cat.tts.id]: prev.model_cached[cat.tts.id] ?? false,
      [cat.stt.id]: prev.model_cached[cat.stt.id] ?? false,
      [RERANKER.id]: prev.model_cached[RERANKER.id] ?? false,
    },
  };
}

interface ModelEngineCtx {
  state: ModelEngineState;
  benchmark: HardwareBenchmark | null;
  runBenchmark: () => Promise<HardwareBenchmark>;
  applyDetectedTier: (tier: HARDWARE_TIER, webgpu: boolean) => void;
  markCached: (modelId: MODELID, cached: boolean) => void;
  setBackend: (backend: ENGINE_BACKEND) => void;
  isCached: (modelId: MODELID) => boolean;
}

const Ctx = createContext<ModelEngineCtx | null>(null);

const DEFAULT_STATE: ModelEngineState = {
  hardware_tier: "Tier0_Minimal",
  active_backend: "Ollama_Local",
  embedding_model: MODEL_CATALOG.Tier0_Minimal.embedding.id,
  llm_model: MODEL_CATALOG.Tier0_Minimal.llm.id,
  tts_model: MODEL_CATALOG.Tier0_Minimal.tts.id,
  stt_model: MODEL_CATALOG.Tier0_Minimal.stt.id,
  reranker_enabled: true,
  model_cached: {},
};

export function ModelEngineProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<ModelEngineState>(DEFAULT_STATE);
  const [benchmark, setBenchmark] = useState<HardwareBenchmark | null>(null);
  const loaded = useRef(false);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    loadModelEngine()
      .then((s) => {
        if (s) setState(s);
      })
      .catch((e) => {
        console.error("[ModelEngine] failed to load saved model settings:", e);
      });
  }, []);

  useEffect(() => {
    saveModelEngine(state).catch((e) => {
      console.warn("[ModelEngine] could not persist model settings:", e);
    });
  }, [state]);

  const runBenchmark = useCallback(async () => {
    const b = await detectHardware();
    setBenchmark(b);
    setState((prev) => selectTier(prev, b.tier, b.webgpu));
    return b;
  }, []);

  const applyDetectedTier = useCallback(
    (tier: HARDWARE_TIER, webgpu: boolean) => {
      setState((prev) => selectTier(prev, tier, webgpu));
    },
    [],
  );

  const markCached = useCallback((modelId: MODELID, cached: boolean) => {
    setState((prev) => ({
      ...prev,
      model_cached: { ...prev.model_cached, [modelId]: cached },
    }));
  }, []);

  const setBackend = useCallback((backend: ENGINE_BACKEND) => {
    setState((prev) => ({ ...prev, active_backend: backend }));
  }, []);

  const isCached = useCallback(
    (modelId: MODELID) => state.model_cached[modelId] ?? false,
    [state.model_cached],
  );

  return (
    <Ctx.Provider
      value={{
        state,
        benchmark,
        runBenchmark,
        applyDetectedTier,
        markCached,
        setBackend,
        isCached,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useModelEngine(): ModelEngineCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useModelEngine must be used within ModelEngineProvider");
  return c;
}
