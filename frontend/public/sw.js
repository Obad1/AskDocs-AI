/* AskDocs AI — offline shell. Not built by Vite; served verbatim from /sw.js.
 *
 * Strategy:
 *  - navigations: network-first, falling back to the last cached shell so a
 *    refresh with no network still shows the app (data itself lives in
 *    IndexedDB, which is already offline).
 *  - same-origin static assets (/assets/*, /fonts/*, /icons/*, manifests):
 *    stale-while-revalidate — instant repeat loads, updated in the background.
 *    ONLY bytes with a real asset content-type are cached: during a deploy
 *    window a stale replica used to serve the HTML shell for missing hashed
 *    files, and caching that HTML under the .js key poisoned module worker and
 *    lazy-chunk fetches (F1). v3 += content-type guard, which also purges any
 *    poisoned v2 entries via the activate step.
 *  - /api and /exports are never cached; /sw.js itself is an update channel
 *    (no-cache) and is excluded too; everything else (cross-origin model
 *    downloads) passes through untouched.
 */
const CACHE = "askdocs-shell-v3";

const STATIC_PREFIXES = ["/assets/", "/fonts/", "/icons/", "/manifest"];

function isHtml(resp: Response): boolean {
  const type = resp.headers.get("content-type") ?? "";
  return type.includes("text/html");
}

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
  if (url.pathname === "/sw.js") return; // update channel, never cached

  // Navigations: network-first with offline fallback.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((resp) => {
          if (resp.ok && isHtml(resp)) {
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

  const isStatic =
    url.pathname === "/favicon.ico" ||
    STATIC_PREFIXES.some((p) => url.pathname.startsWith(p));

  // Non-asset same-origin (e.g. weird old paths) — network only, never cache.
  if (!isStatic) {
    event.respondWith(fetch(request).catch(() => Response.error()));
    return;
  }

  // Static assets: stale-while-revalidate.
  event.respondWith(
    caches.match(request).then((hit) => {
      const refresh = fetch(request)
        .then((resp) => {
          // Never cache an HTML fallback under an asset key.
          if (resp.ok && !isHtml(resp)) {
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