// Vector index wrapper around Voy (Rust->WASM approximate nearest neighbor).
// Stores L2-normalized embeddings; search returns cosine-similarity scores.
// Seeds from IndexedDB embeddings (contract: getAllEmbeddings() -> EmbeddingRow[]).

import { Voy } from "voy-search";
import { getAllEmbeddings } from "../storage/indexeddb";

interface Entry {
  id: string;
  docId: string;
  vector: number[];
}

export class VectorIndex {
  private entries = new Map<string, Entry>();
  private voy: any = null;
  private dirty = true;

  /** Load all stored embeddings into the in-memory index. */
  async seed(): Promise<void> {
    const rows = await getAllEmbeddings();
    for (const r of rows) {
      this.entries.set(r.chunkId, {
        id: r.chunkId,
        docId: r.docId,
        vector: r.vector,
      });
    }
    this.dirty = true;
    this.voy = null;
  }

  /** Insert/replace a single vector. */
  add(id: string, vector: number[] | Float32Array, docId?: string): void {
    this.entries.set(id, {
      id,
      docId: docId ?? "",
      vector: Array.from(vector as number[]),
    });
    this.dirty = true;
    this.voy = null;
  }

  private build(): void {
    const embeddings = Array.from(this.entries.values()).map((e) => ({
      id: e.id,
      title: e.id,
      url: "",
      embeddings: e.vector,
      metadata: { docId: e.docId },
    }));
    this.voy = new Voy({ embeddings });
    this.dirty = false;
  }

  /** Cosine search; higher score = more similar. */
  search(vector: number[] | Float32Array, k = 10): { id: string; score: number }[] {
    if (this.entries.size === 0) return [];
    if (this.dirty || !this.voy) this.build();
    const results = this.voy.search(Array.from(vector as number[]), k);
    return results.map((r: any) => ({ id: r.id, score: r.score }));
  }
}
