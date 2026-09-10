import React, { useCallback, useRef, useState } from "react";

interface WorkspaceSplitterProps {
  left: React.ReactNode;
  right: React.ReactNode;
  /** Initial split for the left pane, percent (default 65). */
  initial?: number;
  min?: number;
  max?: number;
}

/**
 * Collapsible dual workspace: 65/35 dynamic splitter (spec §3.2).
 * The main (left) pane defaults to 65% and is resizable between 30% and 70%.
 */
export default function WorkspaceSplitter({
  left,
  right,
  initial = 65,
  min = 30,
  max = 70,
}: WorkspaceSplitterProps) {
  const [leftPct, setLeftPct] = useState(Math.min(max, Math.max(min, initial)));
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const onMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setLeftPct(Math.min(max, Math.max(min, pct)));
    },
    [min, max],
  );

  const stopDrag = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", stopDrag);
    window.removeEventListener("blur", stopDrag);
  }, [onMouseMove]);

  const startDrag = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", stopDrag);
      // If the pointer leaves the window mid-drag, release the lock anyway so
      // text selection isn't disabled for the whole page next time.
      window.addEventListener("blur", stopDrag);
    },
    [onMouseMove, stopDrag],
  );

  return (
    <div ref={containerRef} className="flex h-full w-full overflow-hidden">
      <div
        className="h-full min-w-0"
        style={{ width: `${leftPct}%` }}
        aria-label="Workspace primary pane"
      >
        {left}
      </div>
      <div
        role="separator"
        aria-orientation="vertical"
        aria-valuenow={Math.round(leftPct)}
        aria-valuemin={min}
        aria-valuemax={max}
        tabIndex={0}
        onMouseDown={startDrag}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft")
            setLeftPct((p) => Math.min(max, Math.max(min, p - 2)));
          if (e.key === "ArrowRight")
            setLeftPct((p) => Math.min(max, Math.max(min, p + 2)));
        }}
        className="group relative w-1.5 cursor-col-resize bg-[var(--border)] transition-colors hover:bg-[var(--accent)]"
      >
        <span className="absolute left-1/2 top-1/2 h-8 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded bg-[var(--fg-muted)] opacity-40 group-hover:opacity-80" />
      </div>
      <div
        className="h-full min-w-0 flex-1"
        style={{ width: `${100 - leftPct}%` }}
        aria-label="Assistant pane"
      >
        {right}
      </div>
    </div>
  );
}
