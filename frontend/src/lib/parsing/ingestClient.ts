import * as Comlink from "comlink";
import { useCallback } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useModelEngine } from "../../context/ModelEngineContext";
import { getDocument, getChunksForDoc, type ChunkRow } from "../storage/indexeddb";
import type { ChunkRecord, FORMAT_TYPE } from "../../types/schema";
import type { IngestApi, Progress } from "../../workers/indexing.worker";

// Client-side ingestion entry point (spec §6.1). Runs the indexing worker,
// then syncs React workspace state from IndexedDB so retrieval/flashcards work
// in-memory. The worker writes raw data; this hook mirrors it into context.
export function useIngest() {
  const { addDocument } = useWorkspace();
  const { state } = useModelEngine();

  return useCallback(
    async (file: File, onProgress?: (p: Progress) => void) => {
      const docId =
        (crypto as Crypto & { randomUUID?: () => string }).randomUUID?.() ??
        `doc-${Date.now()}-${Math.random().toString(36).slice(2)}`;

      // The worker script is a hashed /assets build of this bundle; if it fails
      // to load (deploy flip, cached HTML fallback, offline), new Worker rejects
      // asynchronously and Comlink calls hang — catch it here with a clear line.
      let worker: Worker;
      let api: Comlink.Remote<IngestApi>;
      try {
        worker = new Worker(
          new URL("../../workers/indexing.worker.ts", import.meta.url),
          { type: "module" },
        );
        api = Comlink.wrap<IngestApi>(worker);
      } catch (e) {
        throw new Error(
          "Could not start the local indexing engine (worker failed to load). Reload the page and try again.",
        );
      }

      try {
        const result = await api.ingest(
          { file, docId, format: undefined, modelId: state.embedding_model },
          Comlink.proxy((p: Progress) => onProgress?.(p)),
        );

        // Mirror into React workspace state from IndexedDB.
        const doc = await getDocument(docId);
        if (!doc?.text) {
          throw new Error("Ingest finished, but no document text was stored.");
        }
        const rows: ChunkRow[] = await getChunksForDoc(docId);
        const chunks: ChunkRecord[] = rows.map((r) => ({
          id: r.id,
          docId: r.docId,
          text: r.text,
          page: r.page,
          vector: [],
        }));
        await addDocument(
          docId,
          doc.text,
          result.format as FORMAT_TYPE,
          result.hash,
          chunks,
        );
        return result;
      } finally {
        worker.terminate();
      }
    },
    [addDocument, state.embedding_model],
  );
}
