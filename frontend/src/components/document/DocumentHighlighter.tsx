// DocumentHighlighter.tsx — overlays/marks source passages (used by CitationDrawer
// jump-to-source). Renders the given source text and highlights occurrences of
// `highlight` (case-insensitive).
import React, { useMemo } from "react";

interface Props {
  text: string;
  highlight?: string;
  className?: string;
}

export function DocumentHighlighter({ text, highlight, className }: Props) {
  const segments = useMemo(() => {
    if (!highlight || !highlight.trim()) return [{ mark: false, value: text }];
    const escaped = highlight.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(${escaped})`, "gi");
    return text
      .split(re)
      .map((part) => ({ mark: part.toLowerCase() === highlight.trim().toLowerCase(), value: part }));
  }, [text, highlight]);

  return (
    <span className={className}>
      {segments.map((s, i) =>
        s.mark ? (
          <mark
            key={i}
            className="rounded bg-yellow-200 px-0.5 text-gray-900 dark:bg-yellow-500/60"
          >
            {s.value}
          </mark>
        ) : (
          <span key={i}>{s.value}</span>
        ),
      )}
    </span>
  );
}
