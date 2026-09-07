// Shared domain types for AskDocs AI v2.0.
// Mirrors the formal Z specification in spec §7. These are the contracts every
// subsystem (frontend workers, React components, FastAPI backend) must honor.

export type DOCID = string;
export type CHUNKID = string;
export type USERID = string;
export type WORKSPACEID = string;
export type MODELID = string;
export type TEXT = string;
export type URI = string;
export type VECTOR = Float32Array | number[];
export type TIMESTAMP = number; // epoch ms

export type RATING = 0 | 1 | 2 | 3 | 4 | 5;

export type CONFIDENCE_LEVEL = "High" | "Medium" | "Low";

export type MODE = "StrictDocumentOnly" | "ExpandedAI";

export type FORMAT_TYPE =
  | "PDF"
  | "DOCX"
  | "PPTX"
  | "EPUB"
  | "AUDIO"
  | "VIDEO"
  | "YOUTUBE";

export type HARDWARE_TIER =
  | "Tier0_Minimal"
  | "Tier1_Standard"
  | "Tier2_Performance"
  | "Tier3_Workstation";

export type ENGINE_BACKEND = "WebGPU_WebLLM" | "Ollama_Local" | "LlamaCpp_Local";

// ---- DocumentState (Z §7.2) ----
export interface ChunkRecord {
  id: CHUNKID;
  docId: DOCID;
  text: TEXT;
  vector: VECTOR;
  page?: number;
}

export interface DocumentState {
  documents: Record<DOCID, TEXT>;
  doc_formats: Record<DOCID, FORMAT_TYPE>;
  doc_hashes: Record<DOCID, TEXT>;
  chunks: Record<CHUNKID, ChunkRecord>;
  doc_chunks: Record<DOCID, CHUNKID[]>;
}

// ---- FlashcardState (Z §7.2) ----
export interface FlashcardState {
  card_id: CHUNKID[];
  repetitions: Record<CHUNKID, number>;
  interval: Record<CHUNKID, number>;
  easiness: Record<CHUNKID, number>;
  due_date: Record<CHUNKID, TIMESTAMP>;
}

// ---- AskDocsWorkspace (Z §7.2) ----
export interface AskDocsWorkspace {
  documents: Record<DOCID, TEXT>;
  doc_formats: Record<DOCID, FORMAT_TYPE>;
  doc_hashes: Record<DOCID, TEXT>;
  chunks: Record<CHUNKID, ChunkRecord>;
  doc_chunks: Record<DOCID, CHUNKID[]>;
  flashcard: FlashcardState;
  active_workspace: WORKSPACEID;
  active_mode: MODE;
  confidence_threshold: number; // 0.0 .. 1.0
  zen_mode_enabled: boolean;
  summary_granularity: number; // 1..3
  privacy_cloud_sync_enabled: Record<DOCID, boolean>;
}

// ---- ModelEngineState (Z §7.4) ----
export interface ModelEngineState {
  hardware_tier: HARDWARE_TIER;
  active_backend: ENGINE_BACKEND;
  embedding_model: MODELID;
  llm_model: MODELID;
  tts_model: MODELID;
  stt_model: MODELID;
  reranker_enabled: boolean;
  model_cached: Record<MODELID, boolean>;
}

// ---- Retrieval result ----
export interface RetrievedChunk {
  chunkId: CHUNKID;
  docId: DOCID;
  text: TEXT;
  score: number;
  page?: number;
}

export interface QueryResult {
  retrieved_chunks: CHUNKID[];
  confidence: CONFIDENCE_LEVEL;
  score: number;
  chunks: RetrievedChunk[];
}
