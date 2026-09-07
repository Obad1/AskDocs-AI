// Hybrid retrieval orchestrator (RAG retrieve step).
// 1. embed query via the Comlink embedding worker
// 2. vector search (Voy cosine) + BM25 search (FlexSearch)
// 3. fuse with Reciprocal Rank Fusion (RRF)
// 4. optional cross-encoder rerank (top candidates) before context assembly
// 5. map to RetrievedChunk (docId, text, page from storage), sorted by score
//
// Contracts relied on:
//  - embedding.worker.ts (Comlink): embed(texts, modelId?), embedQuery(text, modelId?)
//  - storage/indexeddb.ts: getAllEmbeddings(), db.chunks.toArray(), loadModelEngine()
//  - schema.ts: RetrievedChunk, CHUNKID

import * as Comlink from "comlink";
// @ts-ignore - vite ?worker import (typed by vite/client)
import EmbeddingWorker from "../../workers/embedding.worker.ts?worker";
import { VectorIndex } from "./vector_index";
import { BM25Index } from "./bm25_index";
import { reciprocalRankFusion } from "./rrf_fusion";
import { rerank } from "./reranker";
import { db, loadModelEngine } from "../storage/indexeddb";
import { rebuildIndexes } from "./index_state";
import type { RetrievedChunk } from "../../types/schema";

interface EmbeddingApi {
  embed(texts: string[], modelId?: string): Promise<number[][]>;
  embedQuery(text: string, modelId?: string): Promise<number[]>;
}

let workerProxy: Comlink.Remote<EmbeddingApi> | null = null;

function getWorker(): Comlink.Remote<EmbeddingApi> {
  if (!workerProxy) {
    const worker = new EmbeddingWorker();
    workerProxy = Comlink.wrap<EmbeddingApi>(worker);
  }
  return workerProxy;
}

let vectorIndex: VectorIndex | null = null;
let bm25Index: BM25Index | null = null;
let indexesReady = false;

async function ensureIndexes(): Promise<void> {
  if (indexesReady) return;
  const { vector, bm25 } = await rebuildIndexes();
  vectorIndex = vector;
  bm25Index = bm25;
  indexesReady = true;
}

export interface RetrieveOpts {
  topK?: number;
  useReranker?: boolean;
  embeddingModelId?: string;
}

export async function retrieve(
  query: string,
  opts: RetrieveOpts = {},
): Promise<RetrievedChunk[]> {
  const topK = opts.topK ?? 10;
  const useReranker = opts.useReranker ?? true;

  await ensureIndexes();

  let embeddingModelId = opts.embeddingModelId;
  if (!embeddingModelId) {
    const eng = await loadModelEngine();
    embeddingModelId = eng?.embedding_model ?? "Xenova/all-MiniLM-L6-v2";
  }

  const worker = getWorker();
  const queryVec = await worker.embedQuery(query, embeddingModelId);

  const vecResults = vectorIndex!.search(queryVec, topK * 2);
  const bm25Results = bm25Index!.search(query, topK * 2);

  // Resilient: fall back to vector-only when BM25 has nothing.
  const rankings =
    bm25Results.length > 0 ? [vecResults, bm25Results] : [vecResults];
  const fused = reciprocalRankFusion(rankings, 60);

  const chunkRows = await db.chunks.toArray();
  const chunkMap = new Map<
    string,
    { docId: string; text: string; page?: number }
  >();
  for (const c of chunkRows) {
    chunkMap.set(c.id, { docId: c.docId, text: c.text, page: c.page });
  }

  let ordered = fused;
  if (useReranker && fused.length > 0) {
    const candidates = fused
      .map((f) => {
        const cm = chunkMap.get(f.id);
        return cm ? { id: f.id, text: cm.text } : null;
      })
      .filter((x): x is { id: string; text: string } => x !== null);

    try {
      ordered = await rerank(
        query,
        candidates,
        "Xenova/ms-marco-MiniLM-L-6-v2",
        topK,
      );
    } catch {
      ordered = fused;
    }
  }

  const results: RetrievedChunk[] = [];
  for (const item of ordered.slice(0, topK)) {
    const cm = chunkMap.get(item.id);
    if (!cm) continue;
    results.push({
      chunkId: item.id,
      docId: cm.docId,
      text: cm.text,
      score: item.score,
      page: cm.page,
    });
  }
  return results;
}
