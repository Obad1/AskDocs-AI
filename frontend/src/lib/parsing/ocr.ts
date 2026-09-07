// ocr.ts — Tesseract.js wrapper used as a fallback (spec §4.1, §8.1 multimodal cascade).
// `ocrImage` takes raw image-like input (ImageData | HTMLCanvasElement | Blob |
// ArrayBuffer) and returns recognized text. Falls back per §8.1 when text
// density from structural parsing is below threshold.
import Tesseract from "tesseract.js";

export interface OcrResult {
  text: string;
  confidence: number;
}

export async function ocrImage(
  input: ImageData | HTMLCanvasElement | Blob | ArrayBuffer | Uint8Array,
  lang = "eng",
  onProgress?: (p: number) => void,
): Promise<OcrResult> {
  // Tesseract.js accepts a canvas, image element, blob, or typed array.
  let source: any = input;
  if (input instanceof ArrayBuffer) {
    source = new Uint8Array(input);
  }

  const worker = await Tesseract.createWorker(lang, 1, {
    logger: (m: any) => {
      if (m.status === "recognizing text" && onProgress) {
        onProgress(m.progress ?? 0);
      }
    },
  });

  try {
    const { data } = await worker.recognize(source);
    return { text: (data.text || "").trim(), confidence: data.confidence ?? 0 };
  } finally {
    await worker.terminate().catch(() => {});
  }
}

// Rough token-density estimate used by the §8.1 cascade decision.
export function tokenCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}
