// Version event shared by the main window and the ingestion Web Worker.
// In-browser IndexedDB writes must invalidate cached vector and BM25 indexes.
let revision = 0;
const channel =
  typeof BroadcastChannel === "undefined"
    ? null
    : new BroadcastChannel("askdocs-retrieval-index");

channel?.addEventListener("message", (event: MessageEvent) => {
  if (event.data?.type === "index-changed") revision += 1;
});

export function currentIndexRevision(): number {
  return revision;
}

export function invalidateIndex(): void {
  revision += 1;
  channel?.postMessage({ type: "index-changed" });
}
