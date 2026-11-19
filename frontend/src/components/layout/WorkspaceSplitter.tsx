import React, { useCallback, useEffect, useRef, useState } from "react";

export type MobilePane = "document" | "assistant";

interface WorkspaceSplitterProps {
  left: React.ReactNode;
  right: React.ReactNode;
  initial?: number;
  min?: number;
  max?: number;
  mobilePane?: MobilePane;
  onMobilePaneChange?: (pane: MobilePane) => void;
}

const STORAGE_KEY = "askdocs.workspace.split.v1";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function readSplit(initial: number, min: number, max: number): number {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored !== null && Number.isFinite(Number(stored))) {
      return clamp(Number(stored), min, max);
    }
  } catch {
    // Private browsing can disable storage. Layout must still work.
  }
  return clamp(initial, min, max);
}

/**
 * Desktop: accessible, pointer/touch + keyboard resizable side-by-side panes.
 * Small screens: a real Document / Assistant switcher instead of two narrow
 * unusable columns. Switching views does not unmount the underlying content.
 */
export default function WorkspaceSplitter({
  left,
  right,
  initial = 60,
  min = 35,
  max = 75,
  mobilePane = "document",
  onMobilePaneChange,
}: WorkspaceSplitterProps) {
  const [leftPct, setLeftPct] = useState(() => readSplit(initial, min, max));
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, String(leftPct));
    } catch {
      // Non-persistent mode is safe.
    }
  }, [leftPct]);

  useEffect(() => {
    return () => { document.body.style.userSelect = ""; };
  }, []);

  const updateFromPointer = useCallback(
    (clientX: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect?.width) return;
      setLeftPct(clamp(((clientX - rect.left) / rect.width) * 100, min, max));
    },
    [min, max],
  );

  const startPointerDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    document.body.style.userSelect = "none";
    updateFromPointer(event.clientX);
  };

  const stopPointerDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    dragging.current = false;
    document.body.style.userSelect = "";
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div
      ref={containerRef}
      className="workspace-splitter"
      data-mobile-pane={mobilePane}
      style={{ "--askdocs-split-left": leftPct + "%" } as React.CSSProperties}
    >
      <nav className="workspace-mobile-switch" aria-label="Workspace views">
        <button
          type="button"
          aria-pressed={mobilePane === "document"}
          className={mobilePane === "document" ? "is-active" : ""}
          onClick={() => onMobilePaneChange?.("document")}
        >
          Document
        </button>
        <button
          type="button"
          aria-pressed={mobilePane === "assistant"}
          className={mobilePane === "assistant" ? "is-active" : ""}
          onClick={() => onMobilePaneChange?.("assistant")}
        >
          Assistant
        </button>
      </nav>
      <section
        className="workspace-split-pane workspace-pane-primary"
        aria-label="Document and study workspace"
      >
        {left}
      </section>
      <div
        role="separator"
        aria-label="Resize document and assistant panes"
        aria-orientation="vertical"
        aria-valuenow={Math.round(leftPct)}
        aria-valuemin={min}
        aria-valuemax={max}
        tabIndex={0}
        title="Drag, use arrow keys, or double-click to reset"
        onPointerDown={startPointerDrag}
        onPointerMove={(event) => {
          if (dragging.current) updateFromPointer(event.clientX);
        }}
        onPointerUp={stopPointerDrag}
        onPointerCancel={stopPointerDrag}
        onDoubleClick={() => setLeftPct(clamp(initial, min, max))}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            event.preventDefault();
            setLeftPct((pct) => clamp(pct + (event.key === "ArrowLeft" ? -3 : 3), min, max));
          } else if (event.key === "Home") {
            event.preventDefault();
            setLeftPct(min);
          } else if (event.key === "End") {
            event.preventDefault();
            setLeftPct(max);
          }
        }}
        className="workspace-split-handle"
      >
        <span aria-hidden="true" />
      </div>
      <section className="workspace-split-pane workspace-pane-assistant" aria-label="AI assistant">
        {right}
      </section>
    </div>
  );
}
