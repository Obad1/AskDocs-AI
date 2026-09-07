// BM25 keyword index wrapper around FlexSearch (client side).
// FlexSearch returns ranked ids; we expose {id, score} with score = 1/rank
// so it can be fused via RRF alongside the vector ranking.

// @ts-ignore - flexsearch has no bundled types (see flexsearch.d.ts)
import FlexSearch from "flexsearch";

export class BM25Index {
  private index: any;
  private texts = new Map<string, string>();

  constructor() {
    this.index = new FlexSearch.Index({
      tokenize: "forward",
      cache: false,
    });
  }

  /** Index a chunk's text under its id. */
  add(id: string, text: string): void {
    this.texts.set(id, text);
    this.index.add(id, text);
  }

  /** Keyword search; results ordered best-first. */
  search(query: string, k = 10): { id: string; score: number }[] {
    if (!query || this.texts.size === 0) return [];
    const hits: string[] = this.index.search(query, { limit: k });
    return hits.map((id, i) => ({ id, score: 1 / (i + 1) }));
  }
}
