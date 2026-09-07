import React from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useUserProfile } from "../../context/UserProfileContext";

interface ZenOverlayProps {
  onExit: () => void;
  onOpenDemo?: () => void;
}

/**
 * Full-screen focus mode (spec §3.2). Rendered when ws.zen_mode_enabled is
 * true. Hides the chrome (TopNavigation / sidebars) and shows only a small
 * translucent control cluster. Activated via Ctrl/Cmd+Shift+Z (handled in App).
 */
export default function ZenOverlay({ onExit, onOpenDemo }: ZenOverlayProps) {
  const { ws, setActiveMode, setZenMode } = useWorkspace();
  const { profile, update } = useUserProfile();

  return (
    <div className="zen-active fixed inset-0 z-40 flex flex-col bg-[var(--bg)]">
      {/* Translucent floating controls */}
      <div className="zen-controls pointer-events-auto absolute right-4 top-4 z-50 flex items-center gap-2 rounded-full px-3 py-1.5 text-xs text-[var(--fg)] shadow-lg">
        <button
          onClick={() =>
            setActiveMode(
              ws.active_mode === "StrictDocumentOnly"
                ? "ExpandedAI"
                : "StrictDocumentOnly",
            )
          }
          className="rounded-full px-2 py-0.5 hover:bg-[var(--bg-elevated)]"
        >
          {ws.active_mode === "StrictDocumentOnly" ? "Strict" : "Expanded"}
        </button>
        <button
          onClick={() => update({ dyslexiaFont: !profile.dyslexiaFont })}
          className="rounded-full px-2 py-0.5 hover:bg-[var(--bg-elevated)]"
          aria-pressed={profile.dyslexiaFont}
        >
          Dyslexic
        </button>
        <button
          onClick={onOpenDemo}
          className="rounded-full px-2 py-0.5 hover:bg-[var(--bg-elevated)]"
        >
          Demo
        </button>
        <button
          onClick={onExit}
          className="rounded-full bg-[var(--accent)] px-2 py-0.5 font-medium text-[var(--accent-fg)]"
        >
          Exit Zen
        </button>
      </div>

      {/* Hotkey hint, fades after a moment via CSS */}
      <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-[var(--fg-muted)] opacity-60">
        Press Ctrl/Cmd+Shift+Z to exit focus mode
      </div>
    </div>
  );
}

export { ZenOverlay };
// re-export setter for callers that want to toggle directly
export function useZenToggle() {
  const { setZenMode } = useWorkspace();
  return setZenMode;
}
