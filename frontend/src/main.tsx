import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/calm_tech_theme.css";
import { installChunkReload } from "./lib/chunkRetry";
import App from "./App";

installChunkReload();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

// Offline shell (PWA): precaches the app bundle at runtime so the local-first
// promise survives a refresh with no network. Only relevant in production
// builds; dev (Vite) does hot reload and needs no cache.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js");
  });

  // A new service worker (skipWaiting + clients.claim in sw.js) is already
  // active once the controller changes; reload once so the new shell is used
  // immediately. Keeps /sw.js no-cache effective and never leaves stale code
  // running behind a freshly-published bundle (audit F3).
  let reloading = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloading) return;
    reloading = true;
    window.location.reload();
  });
}