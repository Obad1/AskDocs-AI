// indexing.worker.ts — Parallel off-thread ingestion pipeline (spec §6.1, §8.1).
// Runs Comlink-exposed `ingest`. The main thread instantiates it via:
//   new Worker(new URL('../public/workers/indexing.worker.ts', import.meta.url), { type: 'module' })
// so Vite compiles it together with its imports. The embedding worker is a
// sibling (owned by another agent) exposing `embed(texts, modelId)`.
import * as Comlink from "comlink";
import type { FORMAT_TYPE } from "../types/schema";
import type {
  DocRow,
  ChunkRow,
  EmbeddingRow,
  FlashcardRow,
} from "../lib/storage/indexeddb";
import { detectFormat } from "../lib/parsing/formatDetect";
import { parsePdf, pagesToText, renderPageToImageData } from "../lib/parsing/pdfParse";
import { parseEpub } from "../lib/parsing/epubParse";
import { parseOffice } from "../lib/parsing/officeParse";
import { ocrImage, tokenCount } from "../lib/parsing/ocr";
import { cleanText } from "../lib/parsing/textCleaner";
import { chunkText } from "../lib/parsing/chunker";
import { sha256Hex } from "../lib/storage/local_encryption";
import { putDocument } from "../lib/storage/indexeddb";

// Embedding worker (another agent owns embedding.worker.ts in public/workers).
import EmbeddingWorker from "./embedding.worker.ts?worker";

const embeddingWorker = new EmbeddingWorker();
interface EmbeddingApi {
  embed(texts: string[], modelId: string): Promise<number[][]>;
}
const embeddingApi = Comlink.wrap<EmbeddingApi>(embeddingWorker);

export type Stage =
  | "Parsing"
  | "OCR"
  | "Text Clean"
  | "Chunking"
  | "Embedding"
  | "Indexing"
  | "Done"
  | "Error";

export interface Progress {
  stage: Stage;
  progress: number;
  message?: string;
}

export interface IngestInput {
  file: File;
  docId: string;
  format?: FORMAT_TYPE;
  modelId: string;
}

export interface IngestResult {
  docId: string;
  format: FORMAT_TYPE;
  hash: string;
  numChunks: number;
  brokenPages?: number[];
  truncated?: boolean;
  junkDetected?: boolean;
}

const TEXT_DENSITY_FLOOR = 20; // §8.1 cascade threshold (tokens/page)

const api = {
  async ingest(
    input: IngestInput,
    onProgress?: (p: Progress) => void,
  ): Promise<IngestResult> {
    const report = (stage: Stage, progress: number, message?: string) => {
      try {
        onProgress?.({ stage, progress, message });
      } catch {
        /* ignore proxy errors */
      }
    };

    const { file, docId } = input;
    const modelId = input.modelId;
    const format: FORMAT_TYPE = input.format ?? detectFormat(file) ?? "PDF";

    let rawText = "";
    let brokenPages: number[] | undefined;
    let truncated = false;
    let junkDetected = false;

    report("Parsing", 0);
    try {
      if (format === "PDF") {
        const res = await parsePdf(file);
        brokenPages = res.brokenPages;
        truncated = res.truncated;
        // §8.1 multimodal cascade: OCR low-text pages.
        const pages = await Promise.all(
          res.pages.map(async (pg) => {
            if (tokenCount(pg.text) < TEXT_DENSITY_FLOOR) {
              report("OCR", 0, `OCR page ${pg.page}`);
              try {
                const img = await renderPageToImageData(file, pg.page);
                if (img) {
                  const ocr = await ocrImage(img);
                  return { ...pg, text: ocr.text || pg.text };
                }
              } catch {
                /* keep original/empty text for this page */
              }
            }
            return pg;
          }),
        );
        rawText = pagesToText(pages);
      } else if (format === "EPUB") {
        const res = await parseEpub(file);
        rawText = res.fullText;
        truncated = res.broken;
      } else if (format === "DOCX" || format === "PPTX") {
        const res = await parseOffice(format, file);
        rawText = res.text;
        truncated = res.broken;
      } else {
        // AUDIO / VIDEO / YOUTUBE are not parsed client-side; route to backend.
        report("Error", 1, "Client worker cannot ingest media/YouTube");
        throw new Error(
          "Client ingest unsupported for " +
            format +
            "; use backend /ingest endpoint.",
        );
      }
    } catch (err) {
      report("Error", 1, (err as Error).message);
      throw err;
    }

    report("Text Clean", 0);
    const cleaned = cleanText(rawText);
    junkDetected = cleaned.junkDetected;
    const clean = cleaned.clean;

    report("Chunking", 0);
    const chunks = chunkText(clean, { targetTokens: 500, overlapTokens: 50 });

    report("Embedding", 0);
    const texts = chunks.map((c) => c.text);
    const vectors: number[][] =
      texts.length > 0 ? await embeddingApi.embed(texts, modelId) : [];

    report("Indexing", 0);
    const enc = new TextEncoder();
    const hash = await sha256Hex(enc.encode(clean));

    const docRow: DocRow = {
      id: docId,
      docId,
      format,
      hash,
      text: clean,
      createdAt: Date.now(),
    };

    const chunkRows: ChunkRow[] = [];
    const embeddingRows: EmbeddingRow[] = [];
    const flashcardRows: FlashcardRow[] = [];

    chunks.forEach((c, i) => {
      const cid = `${docId}::chunk-${i}`;
      chunkRows.push({ id: cid, docId, text: c.text, page: c.page });
      embeddingRows.push({ chunkId: cid, docId, vector: vectors[i] ?? [] });
    });

    await putDocument(docRow, chunkRows, embeddingRows, flashcardRows);

    report("Done", 1, `Indexed ${chunks.length} chunks`);
    return {
      docId,
      format,
      hash,
      numChunks: chunks.length,
      brokenPages,
      truncated,
      junkDetected,
    };
  },
};

Comlink.expose(api);

export type IngestApi = typeof api;
