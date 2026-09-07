// formatDetect.ts — FORMAT_TYPE detection from File / MIME / extension (spec §6.1, §4.1).
import type { FORMAT_TYPE } from "../../types/schema";

const EXT_MAP: Record<string, FORMAT_TYPE> = {
  pdf: "PDF",
  docx: "DOCX",
  pptx: "PPTX",
  epub: "EPUB",
  mp3: "AUDIO",
  wav: "AUDIO",
  m4a: "AUDIO",
  ogg: "AUDIO",
  flac: "AUDIO",
  aac: "AUDIO",
  mp4: "VIDEO",
  mov: "VIDEO",
  webm: "VIDEO",
  mkv: "VIDEO",
  avi: "VIDEO",
  m4v: "VIDEO",
};

const MIME_MAP: Record<string, FORMAT_TYPE> = {
  "application/pdf": "PDF",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
  "application/epub+zip": "EPUB",
  "audio/mpeg": "AUDIO",
  "audio/wav": "AUDIO",
  "audio/x-wav": "AUDIO",
  "audio/mp4": "AUDIO",
  "audio/ogg": "AUDIO",
  "audio/flac": "AUDIO",
  "audio/x-m4a": "AUDIO",
  "video/mp4": "VIDEO",
  "video/quicktime": "VIDEO",
  "video/webm": "VIDEO",
  "video/x-matroska": "VIDEO",
  "video/x-msvideo": "VIDEO",
};

export function extensionOf(name?: string): string {
  if (!name) return "";
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

export function detectFormatFromName(
  name?: string,
  mime?: string,
): FORMAT_TYPE | null {
  const ext = extensionOf(name);
  if (mime && MIME_MAP[mime]) return MIME_MAP[mime];
  if (ext && EXT_MAP[ext]) return EXT_MAP[ext];
  if (mime && mime.startsWith("audio/")) return "AUDIO";
  if (mime && mime.startsWith("video/")) return "VIDEO";
  return null;
}

export function detectFormat(file: File): FORMAT_TYPE | null {
  return detectFormatFromName(file.name, file.type);
}

// A direct YouTube URL (no Data API key needed; spec §4.1).
export function isYouTubeUrl(input: string): boolean {
  return /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/i.test(input.trim());
}

export function detectFormatFromUrl(url: string): FORMAT_TYPE | null {
  if (isYouTubeUrl(url)) return "YOUTUBE";
  return detectFormatFromName(url);
}
