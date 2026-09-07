// pdfParse.ts — pdfjs-dist wrapper: per-page text extraction (spec §4.1, §8.1).
// Per §8.1 partial-success recovery: corrupt pages are skipped and reported
// rather than failing the whole document.
import * as pdfjsLib from "pdfjs-dist";
// Vite resolves this to a URL for the worker bundle.
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export interface ParsedPage {
  page: number;
  text: string;
}

export interface PdfParseResult {
  pages: ParsedPage[];
  numPages: number;
  brokenPages: number[];
  truncated: boolean;
}

export async function parsePdf(
  data: ArrayBuffer | Uint8Array | File,
): Promise<PdfParseResult> {
  const buffer =
    data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;

  const doc = await pdfjsLib.getDocument({ data: buffer as Uint8Array }).promise;
  const numPages = doc.numPages;
  const pages: ParsedPage[] = [];
  const brokenPages: number[] = [];

  for (let p = 1; p <= numPages; p++) {
    try {
      const page = await doc.getPage(p);
      const content = await page.getTextContent();
      const text = content.items
        .map((it: any) => ("str" in it ? it.str : ""))
        .join(" ")
        .replace(/\s{2,}/g, " ")
        .trim();
      pages.push({ page: p, text });
    } catch (err) {
      // §8.1: flag broken page, continue with the rest.
      brokenPages.push(p);
    }
  }

  await doc.destroy().catch(() => {});
  return {
    pages,
    numPages,
    brokenPages,
    truncated: brokenPages.length > 0,
  };
}

// Helper to assemble a single concatenated string with page markers.
export function pagesToText(pages: ParsedPage[]): string {
  return pages
    .map((pg) => `<<<PAGE ${pg.page}>>>\n${pg.text}`)
    .join("\n\n");
}

// Render a single PDF page to an ImageData for OCR fallback (§8.1 multimodal
// cascade). Uses OffscreenCanvas, which is only available inside a Worker.
export async function renderPageToImageData(
  data: ArrayBuffer | Uint8Array | File,
  pageNum: number,
): Promise<ImageData | null> {
  if (typeof OffscreenCanvas === "undefined") return null;
  const buffer =
    data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;
  const doc = await pdfjsLib.getDocument({ data: buffer as Uint8Array }).promise;
  try {
    const page = await doc.getPage(pageNum);
    const viewport = page.getViewport({ scale: 2 });
    const canvas = new OffscreenCanvas(viewport.width, viewport.height);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    await page.render({
      canvasContext: ctx as unknown as CanvasRenderingContext2D,
      viewport,
    }).promise;
    return ctx.getImageData(0, 0, viewport.width, viewport.height);
  } finally {
    await doc.destroy().catch(() => {});
  }
}
