/* AskDocs AI — offline shell. Not built by Vite; served verbatim from /sw.js.
 *
 * Strategy:
 *  - navigations: network-first, falling back to the last cached shell so a
 *    refresh with no network still shows the app (data itself lives in
 *    IndexedDB, which is already offline).
 *  - same-origin static assets (/assets/*, /fonts/*, /icons/*, manifests):
 *    stale-while-revalidate — instant repeat loads, updated in the background.
 *  - /api and /exports are never cached; everything else (cross-origin model
 *    downloads) passes through untouched.
 */
const CACHE = "askdocs-shell-v2";

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return; // don't touch cross-origin
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/exports")) return;

  // Navigations: network-first with offline fallback.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((resp) => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put("./", copy));
          }
          return resp;
        })
        .catch(() =>
          caches
            .open(CACHE)
            .then((c) => c.match("./"))
            .then((hit) => hit || Response.error()),
        ),
    );
    return;
  }

  // Static assets: stale-while-revalidate.
  event.respondWith(
    caches.match(request).then((hit) => {
      const refresh = fetch(request)
        .then((resp) => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put(request, copy));
          }
          return resp;
        })
        .catch(() => hit || Response.error());
      return hit || refresh;
    }),
  );
});