import React, { useEffect } from "react";
import { useUserProfile } from "../../context/UserProfileContext";

interface ThemeCtxValue {
  theme: "slate" | "obsidian" | "sepia" | "graphite";
  dyslexiaFont: boolean;
  lineHeight: number;
}

const ThemeCtx = React.createContext<ThemeCtxValue | null>(null);

export function useTheme(): ThemeCtxValue {
  const c = React.useContext(ThemeCtx);
  if (!c) throw new Error("useTheme must be used within ThemeProvider");
  return c;
}

/**
 * Applies the user's theme, dyslexia font and line-height preferences to the
 * root <html> element via data attributes and CSS variables. Must be rendered
 * inside UserProfileProvider.
 */
export default function ThemeProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { profile, update } = useUserProfile();

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-theme", profile.theme);
    root.style.setProperty(
      "--reading-line-height",
      String(profile.lineHeight),
    );
    root.classList.toggle("dyslexic", profile.dyslexiaFont);
  }, [profile.theme, profile.lineHeight, profile.dyslexiaFont]);

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
