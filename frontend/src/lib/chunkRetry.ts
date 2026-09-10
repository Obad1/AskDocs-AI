// Vite emits content-hashed chunk filenames. After a redeploy, a tab opened on
// an older build can request a chunk that no longer exists (404). React.lazy
// surfaces that as a load failure, which would otherwise leave a permanent
// "{Component} unavailable" subtree. Detect failing dynamic imports and reload
// once so the tab picks up the current deployment without the user losing work
// to confusion about "broken" modals/panes.
let reloading = false;

function isChunkLoadError(reason: unknown): boolean {
  const msg = reason instanceof Error ? reason.message : String(reason);
  return /dynamically imported module|ChunkLoadError|Loading chunk .* failed/i.test(
    msg,
  );
}

export function installChunkReload(): void {
  if (!import.meta.env.PROD) return;
  window.addEventListener("unhandledrejection", (event) => {
    if (isChunkLoadError(event.reason)) {
      reloading = true;
      window.location.reload();
    }
  });
  window.addEventListener("error", (event) => {
    if (isChunkLoadError(event.error)) {
      if (reloading) return;
      reloading = true;
      window.location.reload();
    }
  });
}