import Dexie, { type Table } from "dexie";
import type {
  DocumentState,
  ChunkRecord,
  FlashcardState,
  AskDocsWorkspace,
  ModelEngineState,
} from "../../types/schema";

// Zero-cloud local persistence (spec §6.4). Dexie wraps IndexedDB.
// Stores: documents, embeddings (chunks), study_artifacts (flashcards),
// workspace_state (workspace + model engine state).

export interface DocRow {
  id: string;
  docId: string;
  format: string;
  hash: string;
  text: string;
  createdAt: number;
}

export interface ChunkRow {
  id: string; // CHUNKID
  docId: string;
  text: string;
  page?: number;
  // vector stored separately in embeddings store (typed array not directly storable)
}

export interface EmbeddingRow {
  chunkId: string;
  docId: string;
  vector: number[]; // flattened for IndexedDB
}

export interface FlashcardRow {
  chunkId: string;
  repetitions: number;
  interval: number;
  easiness: number;
  dueDate: number;
}

export interface WorkspaceRow {
  id: string; // workspace id
  state: AskDocsWorkspace;
}

export interface ModelEngineRow {
  id: string; // singleton "active"
  state: ModelEngineState;
}

export class AskDocsDB extends Dexie {
  documents!: Table<DocRow, string>;
  chunks!: Table<ChunkRow, string>;
  embeddings!: Table<EmbeddingRow, string>;
  flashcards!: Table<FlashcardRow, string>;
  workspaces!: Table<WorkspaceRow, string>;
  modelEngine!: Table<ModelEngineRow, string>;

  constructor() {
    super("askdocs");
    this.version(1).stores({
      documents: "id, docId, hash",
      chunks: "id, docId",
      embeddings: "chunkId, docId",
      flashcards: "chunkId",
      workspaces: "id",
      modelEngine: "id",
    });
  }
}

export const db = new AskDocsDB();

export async function putDocument(
  doc: DocRow,
  chunkRows: ChunkRow[],
  embeddingRows: EmbeddingRow[],
  flashcardRows: FlashcardRow[],
): Promise<void> {
  await db.transaction(
    "rw",
    db.documents,
    db.chunks,
    db.embeddings,
    db.flashcards,
    async () => {
      await db.documents.put(doc);
      await db.chunks.bulkPut(chunkRows);
      await db.embeddings.bulkPut(embeddingRows);
      if (flashcardRows.length) await db.flashcards.bulkPut(flashcardRows);
    },
  );
}

export async function getDocument(docId: string): Promise<DocRow | undefined> {
  return db.documents.get(docId);
}

export async function getAllDocuments(): Promise<DocRow[]> {
  return db.documents.toArray();
}

export async function getChunksForDoc(docId: string): Promise<ChunkRow[]> {
  return db.chunks.where("docId").equals(docId).toArray();
}

export async function getEmbedding(chunkId: string): Promise<number[] | undefined> {
  const row = await db.embeddings.get(chunkId);
  return row?.vector;
}

export async function getAllEmbeddings(): Promise<EmbeddingRow[]> {
  return db.embeddings.toArray();
}

export async function getFlashcard(chunkId: string): Promise<FlashcardRow | undefined> {
  return db.flashcards.get(chunkId);
}

export async function getAllFlashcards(): Promise<FlashcardRow[]> {
  return db.flashcards.toArray();
}

export async function saveFlashcard(row: FlashcardRow): Promise<void> {
  await db.flashcards.put(row);
}

export async function saveWorkspace(state: AskDocsWorkspace): Promise<void> {
  await db.workspaces.put({ id: state.active_workspace, state });
}

export async function loadWorkspace(
  id: string,
): Promise<AskDocsWorkspace | undefined> {
  const row = await db.workspaces.get(id);
  return row?.state;
}

export async function saveModelEngine(state: ModelEngineState): Promise<void> {
  await db.modelEngine.put({ id: "active", state });
}

export async function loadModelEngine(): Promise<ModelEngineState | undefined> {
  const row = await db.modelEngine.get("active");
  return row?.state;
}
