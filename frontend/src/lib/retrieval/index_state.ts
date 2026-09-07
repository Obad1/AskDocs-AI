// (Re)build both retrieval indexes from local IndexedDB storage.
// Vector index seeds from getAllEmbeddings(); BM25 index seeds from chunk texts.
// Contracts relied on: getAllEmbeddings() (embeddings store), db.chunks (text store).

import { VectorIndex } from "./vector_index";
import { BM25Index } from "./bm25_index";
import { db } from "../storage/indexeddb";

export async function rebuildIndexes(): Promise<{
  vector: VectorIndex;
  bm25: BM25Index;
}> {
  const vector = new VectorIndex();
  await vector.seed();

  const bm25 = new BM25Index();
  const chunks = await db.chunks.toArray();
  for (const c of chunks) bm25.add(c.id, c.text);

  return { vector, bm25 };
}
