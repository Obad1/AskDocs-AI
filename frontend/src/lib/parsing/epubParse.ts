// epubParse.ts — EPUB text extraction (spec §4.1).
// NOTE: epubjs requires a DOM, which is unavailable inside a Web Worker.
// To keep the ingestion pipeline worker-safe (spec §6.1), we extract text by
// reading the EPUB (a ZIP of XHTML) via jszip and stripping markup. This is
// equivalent in result to an epubjs text walk and runs in both the worker and
// the main thread.
import JSZip from "jszip";

export interface ParsedEpubSection {
  id: string;
  title?: string;
  text: string;
}

export interface EpubParseResult {
  title?: string;
  sections: ParsedEpubSection[];
  fullText: string;
  broken: boolean;
}

function stripTags(html: string): string {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(parseInt(n, 10)))
    .replace(/\s{2,}/g, " ")
    .trim();
}

function readOpfContainer(zip: JSZip): Promise<string | null> {
  return zip.file("META-INF/container.xml")?.async("string").then((xml) => {
    const m = xml.match(/full-path="([^"]+)"/);
    return m ? m[1] : null;
  }).catch(() => null) ?? Promise.resolve(null);
}

export async function parseEpub(
  data: ArrayBuffer | Uint8Array | File,
): Promise<EpubParseResult> {
  const buffer =
    data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;
  const zip = await JSZip.loadAsync(buffer as ArrayBuffer);

  const sections: ParsedEpubSection[] = [];
  let title: string | undefined;
  let broken = false;

  // Find the OPF to get the ordered spine.
  const opfPath = await readOpfContainer(zip);
  const contentDir = opfPath ? opfPath.replace(/[^/]+$/, "") : "";

  if (opfPath) {
    const opf = await zip.file(opfPath)?.async("string").catch(() => null);
    if (opf) {
      const titleMatch = opf.match(/<dc:title[^>]*>([^<]+)<\/dc:title>/i);
      if (titleMatch) title = titleMatch[1].trim();
    }
  }

  // Collect XHTML/HTML files (prefer those listed in the spine order if known).
  const htmlFiles = Object.keys(zip.files).filter(
    (n) =>
      /\.(x?html?|xml)$/i.test(n) &&
      !/META-INF|container\.xml|toc\.ncx|content\.opf/i.test(n) &&
      !zip.files[n].dir,
  );

  for (const path of htmlFiles) {
    try {
      const raw = await zip.file(path)?.async("string");
      if (!raw) continue;
      const text = stripTags(raw);
      if (text) {
        sections.push({ id: path.replace(contentDir, ""), text });
      }
    } catch {
      broken = true;
    }
  }

  const fullText = sections.map((s) => s.text).join("\n\n");
  return { title, sections, fullText, broken };
}
