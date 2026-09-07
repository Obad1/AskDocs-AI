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

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

export interface PdfViewerHandle {
  scrollToPage: (page: number) => void;
  highlight: (text: string) => void;
}

export interface HighlightInfo {
  page: number;
  text: string;
}

interface Props {
  docId: string;
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
    const [status, setStatus] = useState<string>("Loading…");

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
          const page = await doc.getPage(p);
          if (cancelled) break;
          const viewport = page.getViewport({ scale: 1.4 });

          const pageDiv = document.createElement("div");
          pageDiv.setAttribute("data-page", String(p));
          pageDiv.className = "relative mb-4 mx-auto bg-white shadow";
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
        const row = await getDocument(docId);
        if (cancelled) return;
        const text = row?.text ?? "";
        const parts = text.split(/<<<PAGE\s+(\d+)>>>/g);
        let pageNum = 1;
        for (let i = 0; i < parts.length; i++) {
          const seg = parts[i];
          if (/^\d+$/.test(seg)) {
            pageNum = parseInt(seg, 10);
            continue;
          }
          const pageDiv = document.createElement("div");
          pageDiv.setAttribute("data-page", String(pageNum));
          pageDiv.className =
            "relative mb-4 mx-auto max-w-3xl whitespace-pre-wrap p-4 font-mono text-sm text-gray-800 bg-white shadow";
          pageDiv.textContent = seg.trim() || "(empty page)";
          container!.appendChild(pageDiv);
          setPageEl(pageNum, pageDiv);
          pageNum++;
        }
        setStatus(row ? "Rendered extracted text" : "No document found");
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
      })();

      return () => {
        cancelled = true;
      };
    }, [docId, data, setPageEl]);

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
      <div className="flex h-full flex-col">
        <div className="border-b border-gray-200 px-3 py-1 text-xs text-gray-500 dark:border-gray-700">
          {status}
        </div>
        <div
          ref={containerRef}
          onMouseUp={handleMouseUp}
          className="flex-1 overflow-auto bg-gray-100 p-4 dark:bg-gray-800"
        />
      </div>
    );
  },
);
