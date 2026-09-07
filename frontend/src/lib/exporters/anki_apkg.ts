import AnkiExport from "anki-apkg-export";

// Export a set of study cards to an Anki .apkg deck (client-side, zero API keys).
// Returns a Blob that can be written to disk or fed to <a download>.
//
// Contracts relied on:
//  - anki-apkg-export: new AnkiExport(deckName) -> .addCard(front, back) -> .save(): Promise<Blob>
//  - Card shape mirrors ChunkRecord-derived flashcards in WorkspaceContext (front/back text).
export interface AnkiCard {
  front: string;
  back: string;
  context?: string;
}

export async function exportAnkiDeck(
  cards: AnkiCard[],
  deckName = "AskDocs AI Study Deck",
): Promise<Blob> {
  const apkg = new AnkiExport(deckName);
  for (const card of cards) {
    const front = card.front ?? "";
    const back =
      card.context && card.context.length > 0
        ? `${card.back ?? ""}\n\n---\nSource context:\n${card.context}`
        : (card.back ?? "");
    apkg.addCard(front, back);
  }
  return apkg.save();
}
