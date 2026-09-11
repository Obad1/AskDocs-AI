import React, { createContext, useContext, useState, useCallback } from "react";
import type { CustomPersona } from "../lib/personas";

export interface UserProfile {
  displayName: string;
  preferredVoice: string;
  playbackSpeed: number;
  theme: "auto" | "slate" | "obsidian" | "sepia" | "graphite";
  dyslexiaFont: boolean;
  lineHeight: number;
  /** Active explanation persona id (matches BUILT_IN_PERSONAS or a custom id). */
  personaId: string;
  /** User-defined personas; persisted locally, never uploaded. */
  customPersonas: CustomPersona[];
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
  personaId: "standard",
  customPersonas: [],
};

function isCustomPersona(v: unknown): v is CustomPersona {
  if (!v || typeof v !== "object") return false;
  const o = v as Record<string, unknown>;
  return (
    o.custom === true &&
    typeof o.id === "string" &&
    typeof o.label === "string" &&
    typeof o.description === "string" &&
    typeof o.instruction === "string"
  );
}

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
        if (typeof parsed.personaId === "string") saved.personaId = parsed.personaId;
        if (Array.isArray(parsed.customPersonas)) {
          saved.customPersonas = parsed.customPersonas.filter(isCustomPersona);
        }
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
