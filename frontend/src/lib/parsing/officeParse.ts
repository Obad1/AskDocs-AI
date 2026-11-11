// Worker-safe Office Open XML parsing with document order and table context.
// No DOMParser (not consistently available in Web Workers) or paid parser API.
import JSZip from "jszip";

export interface OfficeParagraph {
  index: number;
  text: string;
  page?: number;
}

export interface OfficeParseResult {
  format: "DOCX" | "PPTX";
  text: string;
  paragraphs: OfficeParagraph[];
  broken: boolean;
}

function decodeXml(value: string): string {
  return value
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex: string) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec: string) => String.fromCodePoint(parseInt(dec, 10)));
}

function textRuns(xml: string, namespace: "w" | "a"): string {
  const pattern = new RegExp("<" + namespace + ":t(?:\\s[^>]*)?>([\\s\\S]*?)<\\/" + namespace + ":t>", "g");
  const runs = Array.from(xml.matchAll(pattern), (match) => decodeXml(match[1]));
  return runs.join("").replace(/\s+/g, " ").trim();
}

function tableRowText(cells: string[], rowNumber: number): string {
  return "Row " + rowNumber + " - " +
    cells.map((value, i) => "C" + (i + 1) + ": " + (value.replace(/\|/g, "/") || "(empty)")).join(" | ");
}

function docxTable(xml: string): string[] {
  const result: string[] = [];
  const rows = xml.match(/<w:tr\b[^>]*>[\s\S]*?<\/w:tr>/g) ?? [];
  rows.forEach((row, index) => {
    const cells = row.match(/<w:tc\b[^>]*>[\s\S]*?<\/w:tc>/g) ?? [];
    result.push(tableRowText(cells.map((cell) => {
      const ps = cell.match(/<w:p\b[^>]*>[\s\S]*?<\/w:p>/g) ?? [];
      return ps.map((p) => textRuns(p, "w")).filter(Boolean).join(" ");
    }), index + 1));
  });
  return result;
}

export async function parseDocx(
  data: ArrayBuffer | Uint8Array | File,
): Promise<OfficeParseResult> {
  const buffer = data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;
  const zip = await JSZip.loadAsync(buffer as ArrayBuffer);
  const xml = await zip.file("word/document.xml")?.async("string");
  const paragraphs: OfficeParagraph[] = [];
  if (!xml) return { format: "DOCX", text: "", paragraphs, broken: true };

  const body = xml.match(/<w:body\b[^>]*>([\s\S]*?)<\/w:body>/)?.[1] ?? xml;
  const blocks = body.match(/<w:(p|tbl)\b[^>]*>[\s\S]*?<\/w:\1>/g) ?? [];
  for (const block of blocks) {
    if (block.startsWith("<w:tbl")) {
      for (const line of docxTable(block)) {
        paragraphs.push({ index: paragraphs.length, text: line });
      }
    } else {
      const content = textRuns(block, "w");
      if (content) paragraphs.push({ index: paragraphs.length, text: content });
    }
  }
  return {
    format: "DOCX",
    text: paragraphs.map((p) => p.text).join("\n\n"),
    paragraphs,
    broken: false,
  };
}

function slideTable(xml: string): string {
  const rows = xml.match(/<a:tr\b[^>]*>[\s\S]*?<\/a:tr>/g) ?? [];
  return rows.map((row, i) => {
    const cells = row.match(/<a:tc\b[^>]*>[\s\S]*?<\/a:tc>/g) ?? [];
    return tableRowText(cells.map((cell) => textRuns(cell, "a")), i + 1);
  }).join("\n");
}

export async function parsePptx(
  data: ArrayBuffer | Uint8Array | File,
): Promise<OfficeParseResult> {
  const buffer = data instanceof File ? new Uint8Array(await data.arrayBuffer()) : data;
  const zip = await JSZip.loadAsync(buffer as ArrayBuffer);
  const slideFiles = Object.keys(zip.files)
    .filter((name) => /^ppt\/slides\/slide\d+\.xml$/i.test(name))
    .sort((a, b) => Number(a.match(/slide(\d+)/i)?.[1]) - Number(b.match(/slide(\d+)/i)?.[1]));
  const paragraphs: OfficeParagraph[] = [];
  if (!slideFiles.length) return { format: "PPTX", text: "", paragraphs, broken: true };
  const slides: string[] = [];

  for (const filename of slideFiles) {
    const page = Number(filename.match(/slide(\d+)/i)?.[1]);
    const xml = await zip.file(filename)?.async("string");
    if (!xml) continue;
    // Match whole DrawingML table rows before nested paragraphs so cell
    // relationships survive, while preserving original slide reading order.
    const blocks = xml.match(/<a:tr\b[^>]*>[\s\S]*?<\/a:tr>|<a:p\b[^>]*>[\s\S]*?<\/a:p>/g) ?? [];
    const lines: string[] = [];
    for (const block of blocks) {
      const content = block.startsWith("<a:tr") ? slideTable(block) : textRuns(block, "a");
      if (content) {
        lines.push(content);
        paragraphs.push({ index: paragraphs.length, text: content, page });
      }
    }
    slides.push("<<<PAGE " + page + ">>>\n" + lines.join("\n"));
  }
  return { format: "PPTX", text: slides.join("\n\n"), paragraphs, broken: false };
}

export async function parseOffice(
  format: "DOCX" | "PPTX",
  data: ArrayBuffer | Uint8Array | File,
): Promise<OfficeParseResult> {
  return format === "DOCX" ? parseDocx(data) : parsePptx(data);
}
