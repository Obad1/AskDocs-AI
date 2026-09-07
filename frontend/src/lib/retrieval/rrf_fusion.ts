// Hand-rolled Reciprocal Rank Fusion (RRF) for hybrid retrieval.
// Pure function: takes a list of rankings (each ordered best-first) and combines
// them using the canonical RRF formula: score(id) = sum 1/(k + rank).
// `score` fields on the input rankings are intentionally ignored; only the
// position (1-based rank) matters, per the RRF specification.

export interface RankedItem {
  id: string;
  score: number;
}

export function reciprocalRankFusion(
  rankings: RankedItem[][],
  k = 60,
): RankedItem[] {
  const fused = new Map<string, number>();

  for (const ranking of rankings) {
    if (!ranking || ranking.length === 0) continue;
    for (let rank = 0; rank < ranking.length; rank++) {
      const id = ranking[rank].id;
      fused.set(id, (fused.get(id) ?? 0) + 1 / (k + rank + 1));
    }
  }

  const out: RankedItem[] = [];
  for (const [id, score] of fused) out.push({ id, score });
  out.sort((a, b) => b.score - a.score);
  return out;
}
