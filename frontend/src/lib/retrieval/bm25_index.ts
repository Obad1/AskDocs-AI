// Dependency-free, actual Okapi BM25 for exact-phrase/identifier recall.
// Unlike FlexSearch, these scores account for term frequency, rarity and length.
interface DocumentTerms {
  length: number;
  frequencies: Map<string, number>;
}

export class BM25Index {
  private docs = new Map<string, DocumentTerms>();
  private postings = new Map<string, Set<string>>();
  private totalTokens = 0;
  private readonly k1 = 1.2;
  private readonly b = 0.75;

  private tokenize(text: string): string[] {
    return (text.toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
  }

  remove(id: string): void {
    const old = this.docs.get(id);
    if (!old) return;
    this.totalTokens -= old.length;
    for (const term of old.frequencies.keys()) {
      const posting = this.postings.get(term);
      posting?.delete(id);
      if (posting?.size === 0) this.postings.delete(term);
    }
    this.docs.delete(id);
  }

  add(id: string, text: string): void {
    this.remove(id);
    const tokens = this.tokenize(text);
    if (!tokens.length) return;
    const frequencies = new Map<string, number>();
    for (const term of tokens) {
      frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
    }
    this.docs.set(id, { length: tokens.length, frequencies });
    this.totalTokens += tokens.length;
    for (const term of frequencies.keys()) {
      let posting = this.postings.get(term);
      if (!posting) {
        posting = new Set<string>();
        this.postings.set(term, posting);
      }
      posting.add(id);
    }
  }

  search(query: string, k = 10): { id: string; score: number }[] {
    const terms = new Set(this.tokenize(query));
    const n = this.docs.size;
    if (!terms.size || !n || k <= 0) return [];
    const avgLength = this.totalTokens / n;
    const scores = new Map<string, number>();
    for (const term of terms) {
      const posting = this.postings.get(term);
      if (!posting?.size) continue;
      // Robertson/Sparck Jones positive IDF (avoids negative scores).
      const idf = Math.log(1 + (n - posting.size + 0.5) / (posting.size + 0.5));
      for (const id of posting) {
        const doc = this.docs.get(id)!;
        const tf = doc.frequencies.get(term) ?? 0;
        const normalizer = this.k1 * (1 - this.b + this.b * doc.length / avgLength);
        const score = idf * tf * (this.k1 + 1) / (tf + normalizer);
        scores.set(id, (scores.get(id) ?? 0) + score);
      }
    }
    return [...scores.entries()]
      .map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
      .slice(0, k);
  }
}
