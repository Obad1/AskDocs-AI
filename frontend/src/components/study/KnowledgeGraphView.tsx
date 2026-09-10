import React, { useEffect, useMemo, useRef } from "react";
import cytoscape from "cytoscape";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { DOCID } from "../../types/schema";

const STOPWORDS = new Set(
  "the a an and or but if then of to in on for with as at by from is are was were be been being this that these those it its their our your his her he she they we you i not no can will would should may might must do does did has have had".split(
    " ",
  ),
);

function keywords(text: string, max = 6): string[] {
  const freq = new Map<string, number>();
  for (const raw of text.toLowerCase().match(/[a-z][a-z'-]{2,}/g) ?? []) {
    if (STOPWORDS.has(raw)) continue;
    freq.set(raw, (freq.get(raw) ?? 0) + 1);
  }
  return [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, max)
    .map(([w]) => w);
}

function docTitle(docId: DOCID, text: string): string {
  const t = text.replace(/\s+/g, " ").trim().slice(0, 32);
  return t || docId.slice(0, 12);
}

export function KnowledgeGraphView() {
  const { ws, setActiveDocId } = useWorkspace();
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const { elements } = useMemo(() => {
    const nodes: cytoscape.ElementDefinition[] = [];
    const edges: cytoscape.ElementDefinition[] = [];
    const seen = new Set<string>();
    const docKeyword = new Map<DOCID, Set<string>>();

    for (const [docId, text] of Object.entries(ws.documents)) {
      const kws = new Set(keywords(text, 8));
      docKeyword.set(docId, kws);
      nodes.push({
        data: { id: `doc:${docId}`, label: docTitle(docId, text), kind: "doc" },
      });
      for (const k of kws) {
        const id = `kw:${k}`;
        if (!seen.has(id)) {
          seen.add(id);
          nodes.push({ data: { id, label: k, kind: "kw" } });
        }
        edges.push({ data: { id: `${id}->doc:${docId}`, source: id, target: `doc:${docId}` } });
      }
    }

    // Cross-document concept edges (share a keyword).
    const docs = [...docKeyword.entries()];
    for (let i = 0; i < docs.length; i++) {
      for (let j = i + 1; j < docs.length; j++) {
        const shared = [...docs[i][1]].filter((k) => docs[j][1].has(k));
        for (const k of shared) {
          edges.push({
            data: {
              id: `cross:${docs[i][0]}:${docs[j][0]}:${k}`,
              source: `doc:${docs[i][0]}`,
              target: `doc:${docs[j][0]}`,
              label: k,
            },
          });
        }
      }
    }
    return { elements: [...nodes, ...edges] };
  }, [ws.documents]);

  useEffect(() => {
    if (!containerRef.current) return;
    if (cyRef.current) cyRef.current.destroy();

    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim() || "#2563eb";
    const fgMuted = css.getPropertyValue("--fg-muted").trim() || "#94a3b8";
    const danger = css.getPropertyValue("--danger").trim() || "#ef4444";
    const border = css.getPropertyValue("--border").trim() || "#cbd5e1";
    const paperFg = css.getPropertyValue("--paper-fg").trim() || "#111";

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node[kind='doc']",
          style: {
            "background-color": accent,
            label: "data(label)",
            color: "#fff",
            "font-size": 10,
            "text-valign": "center",
            width: 40,
            height: 40,
          },
        },
        {
          selector: "node[kind='kw']",
          style: {
            "background-color": fgMuted,
            label: "data(label)",
            color: paperFg,
            "font-size": 9,
            width: 22,
            height: 22,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1,
            "line-color": border,
            "curve-style": "haystack",
          },
        },
        {
          selector: "edge[label]",
          style: { label: "data(label)", "font-size": 8, "line-color": danger },
        },
      ],
      layout: { name: "cose", animate: false, padding: 20 } as cytoscape.LayoutOptions,
    });
    cyRef.current = cy;

    cy.on("tap", "node[kind='doc']", (evt) => {
      const id = String(evt.target.id());
      const docId = id.startsWith("doc:") ? id.slice(4) : id;
      setActiveDocId(docId);
    });

    return () => cy.destroy();
  }, [elements, setActiveDocId]);

  if (Object.keys(ws.documents).length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border)] p-6 text-center text-sm text-[var(--fg-muted)]">
        Knowledge graph appears once documents are ingested.
      </div>
    );
  }

  return (
    <div className="surface-card p-3">
      <div className="mb-2 text-sm font-medium">Cross-document concept graph</div>
      <div ref={containerRef} style={{ height: 360, width: "100%" }} />
      <div className="mt-1 text-xs text-[var(--fg-muted)]">
        Click a document node to open it in the Document tab.
      </div>
    </div>
  );
}
