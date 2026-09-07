import React, { createContext, useContext, useState, useCallback } from "react";

export interface UserProfile {
  displayName: string;
  preferredVoice: string;
  playbackSpeed: number;
  theme: "slate" | "obsidian" | "sepia" | "graphite";
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
  theme: "slate",
  dyslexiaFont: false,
  lineHeight: 1.6,
};

export function UserProfileProvider({ children }: { children: React.ReactNode }) {
  const [profile, setProfile] = useState<UserProfile>(DEFAULT);
  const update = useCallback(
    (patch: Partial<UserProfile>) => setProfile((p) => ({ ...p, ...patch })),
    [],
  );
  return <Ctx.Provider value={{ profile, update }}>{children}</Ctx.Provider>;
}

export function useUserProfile(): ProfileCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useUserProfile must be used within UserProfileProvider");
  return c;
}
