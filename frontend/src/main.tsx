import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/calm_tech_theme.css";
import App from "./App";

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
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // Local-first app, no telemetry; a failed SW is not fatal.
    });
  });
}