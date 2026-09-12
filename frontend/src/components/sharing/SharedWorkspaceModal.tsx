import React, { useCallback, useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { exportObsidian } from "../../lib/exporters/obsidian_md";
import { exportNotionBundle } from "../../lib/exporters/notion_md_export";

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Shared Workspace modal (spec §3.8). 1-click packaging to Anki / Obsidian /
// Notion bundle, plus a credential-free public link + iframe embed snippet.
//
// The .apkg is compiled by the backend export unit (/api/v1/export/anki) —
// anki-apkg-export's sql.js runtime doesn't survive the Vite chunk graph, so
// shipping that client-side was broken the Modal open.
export default function SharedWorkspaceModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { ws } = useWorkspace();
  const [error, setError] = useState<string | null>(null);

  const collectCards = useCallback(() => {
    return ws.flashcard.card_id.map((cid) => {
      const chunk = ws.chunks[cid];
      return {
        front: chunk?.text.slice(0, 120) ?? cid,
        back: chunk?.text ?? "",
        context: chunk ? `[${chunk.docId}]` : undefined,
      };
    });
  }, [ws]);

  const collectPages = useCallback(() => {
    return Object.entries(ws.documents).map(([id, text]) => ({
      title: id,
      markdown: text.slice(0, 4000),
    }));
  }, [ws]);

  const downloadAnki = useCallback(async () => {
    setError(null);
    const cards = collectCards().map((c) => [
      c.front,
      c.context ? `${c.back}\n\n---\nSource context:\n${c.context}` : c.back,
    ]);
    try {
      const res = await fetch("/api/v1/export/anki", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cards, deck_name: "AskDocs AI Study Deck" }),
      });
      if (!res.ok) throw new Error(`Anki export failed (${res.status}).`);
      download(await res.blob(), "askdocs.apkg");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [collectCards]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Export and share"
    >
      <div className="w-full max-w-xl rounded-lg bg-[var(--bg-elevated)] p-4 shadow-xl text-[var(--fg)]">
        <h2 className="mb-1 text-lg font-semibold">Export &amp; Share</h2>
        <p className="mb-3 text-xs text-[var(--fg-muted)]">
          Everything is local-first: sharing means downloading files or
          self-hosting a public link — nothing is uploaded or synced to a cloud.
        </p>
        <div className="flex flex-col gap-2">
          <button
            onClick={() => void downloadAnki()}
            className="btn-primary w-full px-3 py-2 text-left"
          >
            Download Anki deck (.apkg)
          </button>
          <button
            onClick={async () =>
              download(
                await exportObsidian(
                  collectPages().map((p) => ({ title: p.title, body: p.markdown })),
                ),
                "askdocs-vault.zip",
              )
            }
            className="btn-primary w-full px-3 py-2 text-left"
          >
            Download Obsidian vault (.zip)
          </button>
          <button
            onClick={async () =>
              download(
                await exportNotionBundle(
                  collectPages(),
                  collectCards().map((c, i) => ({
                    Name: `Card ${i + 1}`,
                    Front: c.front,
                    Back: c.back,
                  })),
                ),
                "notion-bundle.zip",
              )
            }
            className="btn-primary w-full px-3 py-2 text-left"
          >
            Download Notion-compatible bundle (.zip)
          </button>

          {error && (
            <div
              className="rounded-md border border-[var(--danger)] bg-[var(--danger-soft)] p-2 text-xs text-[var(--danger-fg)]"
              role="alert"
            >
              {error}
            </div>
          )}

          <div className="mt-2 rounded border border-[var(--border)] p-2 text-xs">
            <div className="mb-1 font-semibold">
              Read-only link (only if this app is publicly hosted)
            </div>
            <code className="block break-all">
              {typeof window !== "undefined"
                ? `${window.location.origin}/share/${ws.active_workspace}`
                : `/share/${ws.active_workspace}`}
            </code>
            <p className="mb-1 mt-2 text-[var(--fg-muted)]">
              This points back at your own instance; it is not a cloud-synced
              workspace. On a local-only machine it only works for you.
            </p>
            <div className="mb-1 mt-2 font-semibold">Embed snippet</div>
            <code className="block break-all">
              {`<iframe src="${typeof window !== "undefined" ? window.location.origin : ""}/share/${ws.active_workspace}" width="100%" height="600"></iframe>`}
            </code>
          </div>
        </div>

        <div className="mt-3 flex justify-end">
          <button onClick={onClose} className="btn-ghost px-3 py-1">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}