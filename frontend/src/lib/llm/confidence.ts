import type { CONFIDENCE_LEVEL } from "../../types/schema";

// Z §7.3 QueryRetrieval confidence rules:
//   score >= threshold        -> "High"
//   0.5 <= score < threshold  -> "Medium"
//   score < 0.5               -> "Low"
export function scoreToLevel(
  score: number,
  threshold: number,
): CONFIDENCE_LEVEL {
  if (score >= threshold) return "High";
  if (score >= 0.5) return "Medium";
  return "Low";
}

export function computeRetrievalConfidence(
  topScore: number,
  threshold: number,
): { level: CONFIDENCE_LEVEL; score: number } {
  return {
    level: scoreToLevel(topScore, threshold),
    score: topScore,
  };
}
