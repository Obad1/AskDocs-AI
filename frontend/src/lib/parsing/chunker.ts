// chunker.ts — client-side recursive text splitter (spec §6.1).
// Paragraph -> sentence -> token fallback. Target ~500 tokens/chunk with overlap.
// Dependency-free. Honors `<<<PAGE n>>>` markers produced by pdfParse.pagesToText.

export interface Chunk {
  text: string;
  page?: number;
}

export interface ChunkOptions {
  targetTokens?: number;
  overlapTokens?: number;
}

function countTokens(s: string): number {
  return s.trim().split(/\s+/).filter(Boolean).length;
}

function splitSentences(block: string): string[] {
  return block
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function splitWords(block: string): string[] {
  return block.split(/\s+/).filter(Boolean);
}

export function chunkText(
  text: string,
  opts: ChunkOptions = {},
): Chunk[] {
  const target = opts.targetTokens ?? 500;
  const overlap = opts.overlapTokens ?? 50;

  // 1) Split into page-tagged blocks.
  const blocks: { text: string; page?: number }[] = [];
  let currentPage: number | undefined;
  let buf = "";
  const flush = () => {
    if (buf.trim()) blocks.push({ text: buf.trim(), page: currentPage });
    buf = "";
  };
  for (const line of text.split("\n")) {
    const m = line.match(/^<<<PAGE\s+(\d+)>>>$/);
    if (m) {
      flush();
      currentPage = parseInt(m[1], 10);
      continue;
    }
    buf += line + "\n";
  }
  flush();
  if (blocks.length === 0) return [];

  // 2) Greedily assemble chunks, splitting long units recursively.
  const chunks: Chunk[] = [];
  let prevTail: string[] = [];

  for (const block of blocks) {
    const units: string[] = [];
    const paragraphs = block.text
      .split(/\n{2,}/)
      .map((p) => p.trim())
      .filter(Boolean);

    for (let p of paragraphs) {
      let parts = [p];
      if (countTokens(p) > target) {
        const sents = splitSentences(p);
        parts = sents.length > 1 ? sents : [p];
      }
      for (const u of parts) {
        if (countTokens(u) > target) {
          const words = splitWords(u);
          for (let i = 0; i < words.length; i += target) {
            units.push(words.slice(i, i + target).join(" "));
          }
        } else {
          units.push(u);
        }
      }
    }

    let acc: string[] = [];
    let accTokens = 0;
    for (const u of units) {
      const t = countTokens(u);
      if (accTokens + t > target && acc.length > 0) {
        chunks.push({
          text: prevTail.concat(acc).join(" ").replace(/\s{2,}/g, " ").trim(),
          page: block.page,
        });
        prevTail = acc.slice(-overlap);
        acc = [];
        accTokens = 0;
      }
      acc.push(u);
      accTokens += t;
    }
    if (acc.length > 0) {
      chunks.push({
        text: prevTail.concat(acc).join(" ").replace(/\s{2,}/g, " ").trim(),
        page: block.page,
      });
      prevTail = acc.slice(-overlap);
    }
  }

  return chunks.filter((c) => c.text.length > 0);
}
