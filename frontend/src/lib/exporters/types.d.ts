// Minimal ambient module declarations for untyped / loosely-typed export libs.
// Keeps `tsc -b` green without modifying the shared foundation or package.json.

declare module "anki-apkg-export" {
  export interface AnkiCardOptions {
    tags?: string[];
    media?: Array<{ name: string; data: Uint8Array | ArrayBuffer; mime: string }>;
  }
  export default class AnkiExport {
    constructor(deckName: string, options?: { deckId?: number; includeRevLog?: boolean });
    addCard(front: string, back: string, opts?: AnkiCardOptions): void;
    addMedia(name: string, data: Uint8Array | ArrayBuffer, mime: string): void;
    save(): Promise<Blob>;
  }
}

declare module "pptxgenjs" {
  interface TextProps {
    text?: string;
    options?: Record<string, unknown>;
    [key: string]: unknown;
  }
  class PptxGenJS {
    constructor(options?: Record<string, unknown>);
    defineLayout(layout: Record<string, unknown>): void;
    layout(layoutName: string): void;
    author(name: string): void;
    title(name: string): void;
    addSlide(): Slide;
    write(options: { outputType: "blob" | "base64" | "arraybuffer" | "nodebuffer" }): Promise<Blob>;
    writeFile(options?: { fileName?: string }): Promise<string>;
  }
  interface Slide {
    addText(text: string | TextProps | TextProps[], options?: Record<string, unknown>): void;
    addImage(options: Record<string, unknown>): void;
    addNotes(notes: string): void;
    addTable(rows: unknown[], options?: Record<string, unknown>): void;
    addShape(shape: unknown, options?: Record<string, unknown>): void;
  }
  export default PptxGenJS;
}
