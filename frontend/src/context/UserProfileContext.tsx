import React, { createContext, useContext, useState, useCallback } from "react";

export interface UserProfile {
  displayName: string;
  preferredVoice: string;
  playbackSpeed: number;
  theme: "auto" | "slate" | "obsidian" | "sepia" | "graphite";
  dyslexiaFont: boolean;
  lineHeight: number;
}

interface ProfileCtx {
  profile: UserProfile;
  update: (patch: Partial<UserProfile>) => void;
}

const Ctx = createContext<ProfileCtx | null>(null);

const DEFAULT: UserProfile = {
  displayName: "Guest",
  preferredVoice: "en_US-lessac-medium",
  playbackSpeed: 1.0,
  theme: "auto",
  dyslexiaFont: false,
  lineHeight: 1.6,
};

const STORAGE_KEY = "askdocs.profile";

function loadInitial(): UserProfile {
  const saved: UserProfile = { ...DEFAULT };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<UserProfile>;
      if (parsed && typeof parsed === "object") {
        if (typeof parsed.displayName === "string") saved.displayName = parsed.displayName;
        if (typeof parsed.preferredVoice === "string") saved.preferredVoice = parsed.preferredVoice;
        if (typeof parsed.playbackSpeed === "number") saved.playbackSpeed = parsed.playbackSpeed;
        if (
          typeof parsed.theme === "string" &&
          ["auto", "slate", "obsidian", "sepia", "graphite"].includes(parsed.theme)
        )
          saved.theme = parsed.theme;
        if (typeof parsed.dyslexiaFont === "boolean") saved.dyslexiaFont = parsed.dyslexiaFont;
        if (typeof parsed.lineHeight === "number") saved.lineHeight = parsed.lineHeight;
      }
    }
  } catch {
    // Corrupt or unavailable storage — fall back to defaults.
  }
  return saved;
}

export function UserProfileProvider({ children }: { children: React.ReactNode }) {
  const [profile, setProfile] = useState<UserProfile>(loadInitial);
  const update = useCallback(
    (patch: Partial<UserProfile>) =>
      setProfile((p) => {
        const next = { ...p, ...patch };
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {
          // Storage unavailable (private mode, quota) — keep in-memory profile.
        }
        return next;
      }),
    [],
  );
  return <Ctx.Provider value={{ profile, update }}>{children}</Ctx.Provider>;
}

export function useUserProfile(): ProfileCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useUserProfile must be used within UserProfileProvider");
  return c;
}
