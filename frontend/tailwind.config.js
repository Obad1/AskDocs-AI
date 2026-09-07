/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        slate: { DEFAULT: "#475569" },
        obsidian: "#0b0f17",
        sepia: "#f4ecd8",
        graphite: "#2b2f36",
      },
      fontFamily: {
        dyslexic: ["OpenDyslexic", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
