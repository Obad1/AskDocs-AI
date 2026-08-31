import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import wasm from "vite-plugin-wasm";

export default defineConfig({
  plugins: [react(), wasm()],
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["voy-search"],
  },
  build: {
    target: "esnext",
    outDir: "dist",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL || "http://localhost:$PORT",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
