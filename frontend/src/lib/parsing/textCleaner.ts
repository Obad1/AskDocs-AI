// textCleaner.ts - text normalization & junk stripping (spec section 4.2, 8.1).
// Uses String.prototype.normalize('NFKC') and heuristic cleaners. The spec
// references compromise.js for NLP, but it is not in package.json, so we use a
// lightweight, dependency-free tokenizer/heuristic instead. sha256Hex is reused
// from the storage module so dedup hashing stays in one place.
import { sha256Hex } from "../storage/local_encryption";

export interface CleanOptions {
  normalizeNFKC: boolean;
  stripHeadersFooters: boolean;
  stripLineNumbers: boolean;
  stripPageNumbers: boolean;
  stripBrokenUnicode: boolean;
}

export const DEFAULT_CLEAN_OPTIONS: CleanOptions = {
  normalizeNFKC: true,
  stripHeadersFooters: true,
  stripLineNumbers: true,
  stripPageNumbers: true,
  stripBrokenUnicode: true,
};

export interface CleanResult {
  clean: string;
  removed: string[];
  junkDetected: boolean;
}

// ASCII control chars 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F. Built at runtime
// so the source file stays pure ASCII (no literal control bytes).
const CONTROL = new RegExp(
  "[" +
    String.fromCharCode(0) +
    "-" +
    String.fromCharCode(8) +
    String.fromCharCode(11) +
    String.fromCharCode(12) +
    String.fromCharCode(14) +
    "-" +
    String.fromCharCode(31) +
    String.fromCharCode(127) +
    "]",
  "g",
);
const REPLACEMENT = new RegExp("[" + String.fromCharCode(0xfffd) + "]", "g");
const PAGE_NUMBER = /^\s*(page\s*)?(\d{1,4})\s*(\/\s*\d{1,4})?\s*$/i;

// Very light tokenizer (word-ish) - substitute for the compromise token stream.
export function tokenize(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

export function normalizeNFKC(raw: string): string {
  return raw.normalize("NFKC");
}

export function cleanText(
  raw: string,
  opts: Partial<CleanOptions> = {},
): CleanResult {
  const o = { ...DEFAULT_CLEAN_OPTIONS, ...opts };
  const removed: string[] = [];
  let junkDetected = false;

  let text = o.normalizeNFKC ? normalizeNFKC(raw) : raw;

  // --- broken unicode / control characters (section 8.1) ---
  if (o.stripBrokenUnicode) {
    const replCount = (text.match(REPLACEMENT) || []).length;
    if (replCount > 0) {
      junkDetected = junkDetected || replCount > 5;
      removed.push(`replacement-characters:${replCount}`);
      text = text.replace(REPLACEMENT, "");
    }
    if (CONTROL.test(text)) {
      const ctlCount = (text.match(CONTROL) || []).length;
      junkDetected = junkDetected || ctlCount > 20;
      removed.push(`control-characters:${ctlCount}`);
      text = text.replace(CONTROL, " ");
    }
  }

  let lines = text.split(/\r?\n/);

  // --- page numbers (e.g. "12", "Page 3", "3 / 14") ---
  if (o.stripPageNumbers) {
    const filtered: string[] = [];
    for (const ln of lines) {
      if (PAGE_NUMBER.test(ln)) {
        removed.push(`page-number:${ln.trim()}`);
        continue;
      }
      filtered.push(ln);
    }
    lines = filtered;
  }

  // --- line numbers (leading "  12  " repeated across lines) ---
  if (o.stripLineNumbers) {
    const lnRe = /^\s*\d{1,4}[.)]?\s{1,3}/;
    let numbered = 0;
    const sample = Math.min(lines.length, 50);
    for (let i = 0; i < sample; i++) if (lnRe.test(lines[i])) numbered++;
    if (sample > 0 && numbered / sample > 0.6) {
      lines = lines.map((ln) => {
        const m = ln.match(lnRe);
        if (m) {
          removed.push(`line-number:${m[0].trim()}`);
          return ln.slice(m[0].length);
        }
        return ln;
      });
    }
  }

  // --- headers / footers: repeated short lines across the document ---
  if (o.stripHeadersFooters) {
    const freq = new Map<string, number>();
    for (const ln of lines) {
      const t = ln.trim();
      if (t.length > 0 && t.length < 80) freq.set(t, (freq.get(t) || 0) + 1);
    }
    const repeats = new Set<string>();
    for (const [k, v] of freq) if (v >= 3) repeats.add(k);
    if (repeats.size > 0) {
      const filtered: string[] = [];
      for (const ln of lines) {
        const t = ln.trim();
        if (repeats.has(t)) {
          removed.push(`header-footer:${t}`);
          continue;
        }
        filtered.push(ln);
      }
      lines = filtered;
    }
  }

  const clean = lines
    .join("\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  if (clean.length === 0 && raw.length > 0) junkDetected = true;

  return { clean, removed, junkDetected };
}

export { sha256Hex };
