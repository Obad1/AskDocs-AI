// officeParse.ts — DOCX / PPTX text + structure extraction (spec §4.1).
// NOTE: `officeparser` is not present in package.json, so we extract text by
// unzipping the OOXML package with jszip (available) and reading the document
// part(s). This is worker-safe and returns the same {text, structure} shape.
import JSZip from "jszip";

export interface OfficeParagraph {
  index: number;
  text: string;
}

export interface OfficeParseResult {
  format: "DOCX" | "PPTX";
  text: string;
  paragraphs: OfficeParagraph[];
  broken: boolean;
}

function stripTags(xml: string): string {
  return xml
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(parseInt(n, 10)))
    .replace(/<[^>]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export async function parseDocx(
  data: ArrayBuffer | Uint8Array | File,
): Promise<OfficeParseResult> {
  const buffer =
    data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;
  const zip = await JSZip.loadAsync(buffer as ArrayBuffer);
  const docXml = await zip.file("word/document.xml")?.async("string");
  const paragraphs: OfficeParagraph[] = [];
  if (!docXml) {
    return { format: "DOCX", text: "", paragraphs, broken: true };
  }
  const matches = docXml.match(/<w:p[ >][\s\S]*?<\/w:p>/g) ?? [];
  let idx = 0;
  for (const p of matches) {
    const t = stripTags(p);
    if (t) paragraphs.push({ index: idx++, text: t });
  }
  return {
    format: "DOCX",
    text: paragraphs.map((p) => p.text).join("\n"),
    paragraphs,
    broken: false,
  };
}

export async function parsePptx(
  data: ArrayBuffer | Uint8Array | File,
): Promise<OfficeParseResult> {
  const buffer =
    data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;
  const zip = await JSZip.loadAsync(buffer as ArrayBuffer);
  const slideFiles = Object.keys(zip.files)
    .filter((n) => /^ppt\/slides\/slide\d+\.xml$/i.test(n))
    .sort((a, b) => {
      const na = parseInt(a.match(/\d+/)![0], 10);
      const nb = parseInt(b.match(/\d+/)![0], 10);
      return na - nb;
    });
  const paragraphs: OfficeParagraph[] = [];
  let idx = 0;
  if (slideFiles.length === 0) {
    return { format: "PPTX", text: "", paragraphs, broken: true };
  }
  for (const f of slideFiles) {
    const xml = await zip.file(f)?.async("string");
    if (!xml) continue;
    const texts = xml.match(/<a:t>([\s\S]*?)<\/a:t>/g) ?? [];
    for (const t of texts) {
      const clean = stripTags(t);
      if (clean) paragraphs.push({ index: idx++, text: clean });
    }
  }
  return {
    format: "PPTX",
    text: paragraphs.map((p) => p.text).join("\n"),
    paragraphs,
    broken: false,
  };
}

export async function parseOffice(
  format: "DOCX" | "PPTX",
  data: ArrayBuffer | Uint8Array | File,
): Promise<OfficeParseResult> {
  return format === "DOCX" ? parseDocx(data) : parsePptx(data);
}
