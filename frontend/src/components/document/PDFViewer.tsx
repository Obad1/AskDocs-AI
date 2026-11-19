// PDFViewer.tsx — canvas PDF viewer with text layer, highlight hooks and
// jump-to-source (spec §3.1). Renders real PDF bytes when `data` is supplied;
// otherwise falls back to the stored extracted text (paginated by the
// `<<<PAGE n>>>` markers) so the component still works from a docId alone.
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import * as pdfjsLib from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { getDocument } from "../../lib/storage/indexeddb";
import { useWorkspace } from "../../context/WorkspaceContext";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

function normalizeSourceText(value: string): string {
  return value.replace(/\s+/g, " ").trim().toLowerCase();
}

export interface PdfViewerHandle {
  scrollToPage: (page: number) => void;
  highlight: (text: string) => void;
}

export interface HighlightInfo {
  page: number;
  text: string;
}

interface Props {
  docId?: string | null;
  data?: ArrayBuffer | Uint8Array;
  onHighlight?: (info: HighlightInfo) => void;
}

interface PageRef {
  page: number;
  el: HTMLDivElement | null;
}

export const PDFViewer = forwardRef<PdfViewerHandle, Props>(
  function PDFViewer({ docId, data, onHighlight }, ref) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const pageRefs = useRef<PageRef[]>([]);
    const [status, setStatus] = useState<string>("Loading document…");

    const setPageEl = useCallback((page: number, el: HTMLDivElement | null) => {
      const existing = pageRefs.current.find((p) => p.page === page);
      if (existing) existing.el = el;
      else pageRefs.current.push({ page, el });
    }, []);

    const flash = useCallback((page: number) => {
      const r = pageRefs.current.find((p) => p.page === page);
      const el = r?.el;
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("ring-4", "ring-yellow-400");
      window.setTimeout(() => el.classList.remove("ring-4", "ring-yellow-400"), 1200);
    }, []);

    // Citation → source jump: flash the source page (and, when the quote text
    // survives the extraction round-trip, scroll straight to the matching line).
    const { ws, focusRequest, requestDocFocus } = useWorkspace();
    const pendingFocusRef = useRef<{ page: number; text?: string } | null>(null);

    const applyFocus = useCallback(() => {
      const f = pendingFocusRef.current;
      if (!f) return;
      if (!pageRefs.current.length) return; // wait for async page rendering
      let jumped = false;
      if (f.text) {
        const target = normalizeSourceText(f.text);
        const match = pageRefs.current.find((p) => {
          const t = p.el?.textContent;
          return t ? normalizeSourceText(t).includes(target) : false;
        });
        if (match) {
          flash(match.page);
          jumped = true;
        }
      }
      if (!jumped) {
        const page = pageRefs.current.find((p) => p.page === f.page);
        if (!page) return; // requested page has not been rendered yet
        flash(f.page);
      }
      pendingFocusRef.current = null;
      requestDocFocus(null);
    }, [flash, requestDocFocus]);

    useEffect(() => {
      if (!focusRequest || focusRequest.docId !== docId) return;
      pendingFocusRef.current = { page: focusRequest.page, text: focusRequest.text };
      applyFocus();
    }, [focusRequest, docId, applyFocus]);

    useImperativeHandle(
      ref,
      () => ({
        scrollToPage(page: number) {
          flash(page);
        },
        highlight(text: string) {
          const match = pageRefs.current.find((p) =>
            p.el?.textContent?.toLowerCase().includes(text.toLowerCase()),
          );
          if (match) flash(match.page);
        },
      }),
      [flash],
    );

    useEffect(() => {
      let cancelled = false;
      const container = containerRef.current;
      if (!container) return;

      async function renderPdfBytes(bytes: Uint8Array) {
        setStatus("Rendering PDF…");
        const doc = await pdfjsLib.getDocument({ data: bytes }).promise;
        if (cancelled) return;
        for (let p = 1; p <= doc.numPages; p++) {
          setStatus(`Rendering page ${p} / ${doc.numPages}`);
          const page = await doc.getPage(p);
          if (cancelled) break;
          const viewport = page.getViewport({ scale: 1.4 });

          const pageDiv = document.createElement("div");
          pageDiv.setAttribute("data-page", String(p));
          pageDiv.className = "document-paper relative mx-auto mb-5 overflow-hidden bg-white shadow";
          pageDiv.style.width = `${viewport.width}px`;
          pageDiv.style.height = `${viewport.height}px`;
          container!.appendChild(pageDiv);
          setPageEl(p, pageDiv);

          const canvas = document.createElement("canvas");
          canvas.width = viewport.width;
          canvas.height = viewport.height;
          pageDiv.appendChild(canvas);
          const ctx = canvas.getContext("2d");
          if (ctx) {
            await page.render({ canvasContext: ctx, viewport }).promise;
          }

          // Minimal text layer for selection / highlight hooks.
          const content = await page.getTextContent();
          const textLayer = document.createElement("div");
          textLayer.className = "absolute left-0 top-0 select-text";
          textLayer.style.width = `${viewport.width}px`;
          textLayer.style.height = `${viewport.height}px`;
          for (const item of content.items as any[]) {
            if (!("str" in item)) continue;
            const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
            const fontHeight = Math.hypot(tx[2], tx[3]);
            const span = document.createElement("span");
            span.textContent = item.str;
            span.style.position = "absolute";
            span.style.left = `${tx[4]}px`;
            span.style.top = `${tx[5] - fontHeight}px`;
            span.style.fontSize = `${fontHeight}px`;
            span.style.transform = `scaleX(${item.width / (item.str.length * fontHeight) || 1})`;
            span.style.transformOrigin = "left top";
            span.style.color = "transparent";
            span.style.whiteSpace = "pre";
            textLayer.appendChild(span);
          }
          pageDiv.appendChild(textLayer);
        }
        if (!cancelled) setStatus(`Rendered ${doc.numPages} pages`);
      }

      async function renderStoredText() {
        if (!docId) {
          setStatus("No document selected — add one with + Add");
          return;
        }
        const row = await getDocument(docId);
        if (cancelled) return;
        if (!row) {
          setStatus("This document is no longer in local storage.");
          return;
        }
        const text = row.text;
        const parts = text.split(/<<<PAGE\s+(\d+)>>>/g);
        let pageNum = 1;
        for (let i = 0; i < parts.length; i++) {
          const seg = parts[i];
          if (/^\d+$/.test(seg)) {
            pageNum = parseInt(seg, 10);
            continue;
          }
          if (!seg.trim()) continue;
          const pageDiv = document.createElement("div");
          pageDiv.setAttribute("data-page", String(pageNum));
          pageDiv.className = "document-paper document-text-page mx-auto mb-5 w-full max-w-3xl";
          const pageLabel = document.createElement("div");
          pageLabel.className = "document-page-label";
          pageLabel.textContent = "PAGE " + pageNum;
          const body = document.createElement("div");
          body.className = "document-text-content";
          body.textContent = seg.trim();
          pageDiv.append(pageLabel, body);
          container!.appendChild(pageDiv);
          setPageEl(pageNum, pageDiv);
          pageNum++;
        }
        setStatus("Text preview · original file rendering is not stored locally");
      }

      (async () => {
        container.innerHTML = "";
        pageRefs.current = [];
        try {
          if (data) {
            await renderPdfBytes(
              data instanceof Uint8Array ? data : new Uint8Array(data),
            );
          } else {
            await renderStoredText();
          }
        } catch (e) {
          setStatus("Failed to render: " + (e as Error).message);
        }
        if (!cancelled) applyFocus();
      })();

      return () => {
        cancelled = true;
      };
    }, [docId, data, setPageEl, applyFocus]);

    const handleMouseUp = () => {
      const sel = window.getSelection();
      const text = sel?.toString().trim();
      if (!text || !onHighlight) return;
      const anchor = sel?.anchorNode;
      const pageEl =
        anchor instanceof Element
          ? anchor.closest("[data-page]")
          : anchor?.parentElement?.closest("[data-page]");
      const page = pageEl ? parseInt(pageEl.getAttribute("data-page") || "0", 10) : 0;
      onHighlight({ page, text });
    };

    return (
      <div className="flex h-full min-h-0 flex-col">
        <div className="document-reader-header flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-[var(--border)] px-4 py-3">
          <div className="min-w-0 flex-1 truncate text-sm font-semibold text-[var(--fg)]">
            {docId ? (ws.doc_names?.[docId] || "Document preview") : "Document reader"}
          </div>
          <span aria-live="polite" role="status" className="text-xs text-[var(--fg-muted)]">
            {status}
          </span>
        </div>
        {!docId && !data ? (
          <div className="document-empty-state flex min-h-0 flex-1 flex-col items-center justify-center px-6 text-center">
            <span aria-hidden="true" className="text-4xl">▤</span>
            <h2 className="mt-3 text-base font-semibold text-[var(--fg)]">Your reading space</h2>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-[var(--fg-muted)]">
              Choose a document from your library or use Browse files to upload one.
            </p>
          </div>
        ) : (
          <div
            ref={containerRef}
            aria-label="Document pages"
            onMouseUp={handleMouseUp}
            className="document-reader-scroll min-h-0 flex-1 overflow-auto bg-[var(--bg-sunken)] px-3 py-6 sm:px-6"
          />
        )}
      </div>
    );
  },
);
