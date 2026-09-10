import React, { useEffect, useState } from "react";
import { useUserProfile } from "../../context/UserProfileContext";

interface ThemeCtxValue {
  theme: "auto" | "slate" | "obsidian" | "sepia" | "graphite";
  dyslexiaFont: boolean;
  lineHeight: number;
}

const ThemeCtx = React.createContext<ThemeCtxValue | null>(null);

export function useTheme(): ThemeCtxValue {
  const c = React.useContext(ThemeCtx);
  if (!c) throw new Error("useTheme must be used within ThemeProvider");
  return c;
}

/** Resolves "auto" (follow OS) to a concrete theme; light→slate, dark→graphite. */
function resolveTheme(
  theme: ThemeCtxValue["theme"],
  systemDark: boolean,
): "slate" | "obsidian" | "sepia" | "graphite" {
  if (theme === "auto") return systemDark ? "graphite" : "slate";
  return theme;
}

/**
 * Applies the user's theme, dyslexia font and line-height preferences to the
 * root <html> element via data attributes and CSS variables. Must be rendered
 * inside UserProfileProvider. Keeps the browser chrome (theme-color meta) in
 * sync with the active theme so the address bar never stays dark in a light
 * theme (or vice-versa).
 */
export default function ThemeProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { profile, update } = useUserProfile();
  const [systemDark, setSystemDark] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onSysChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", onSysChange);
    return () => mq.removeEventListener("change", onSysChange);
  }, []);

  const resolved = resolveTheme(profile.theme, systemDark);

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", resolved);
    root.style.setProperty(
      "--reading-line-height",
      String(profile.lineHeight),
    );
    root.classList.toggle("dyslexic", profile.dyslexiaFont);

    // Mirror the active background into the browser chrome color.
    let meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
    if (!meta) {
      meta = document.createElement("meta");
      meta.name = "theme-color";
      document.head.appendChild(meta);
    }
    const bg = getComputedStyle(root).getPropertyValue("--bg").trim();
    meta.content = bg || "#eef2f7";
  }, [resolved, profile.lineHeight, profile.dyslexiaFont]);

  const value: ThemeCtxValue = {
    theme: profile.theme,
    dyslexiaFont: profile.dyslexiaFont,
    lineHeight: profile.lineHeight,
  };

  return <ThemeCtx.Provider value={value}>{children}</ThemeCtx.Provider>;
}

export { ThemeCtx, ThemeProvider };
export function useThemeUpdate() {
  const { update } = useUserProfile();
  return update;
}
