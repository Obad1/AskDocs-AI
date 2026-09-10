import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";
import type {
  AskDocsWorkspace,
  DOCID,
  CHUNKID,
  TEXT,
  FORMAT_TYPE,
  MODE,
  ChunkRecord,
  RATING,
  CONFIDENCE_LEVEL,
} from "../types/schema";
import { loadWorkspace, saveWorkspace, db } from "../lib/storage/indexeddb";

function emptyWorkspace(id: string): AskDocsWorkspace {
  return {
    documents: {},
    doc_formats: {},
    doc_hashes: {},
    chunks: {},
    doc_chunks: {},
    flashcard: {
      card_id: [],
      repetitions: {},
      interval: {},
      easiness: {},
      due_date: {},
    },
    active_workspace: id,
    active_mode: "StrictDocumentOnly",
    confidence_threshold: 0.5,
    zen_mode_enabled: false,
    summary_granularity: 2,
    privacy_cloud_sync_enabled: {},
  };
}

interface WorkspaceCtx {
  ws: AskDocsWorkspace;
  /** UI-only: the document currently open in the Document viewer. */
  activeDocId: DOCID | null;
  setActiveDocId: (id: DOCID | null) => void;
  addDocument: (
    docId: DOCID,
    text: TEXT,
    format: FORMAT_TYPE,
    hash: TEXT,
    chunks: ChunkRecord[],
  ) => Promise<void>;
  setActiveMode: (mode: MODE) => void;
  setConfidenceThreshold: (t: number) => void;
  setZenMode: (on: boolean) => void;
  setSummaryGranularity: (g: number) => void;
  setPrivacySync: (docId: DOCID, on: boolean) => void;
  reviewFlashcard: (chunkId: CHUNKID, rating: RATING, now: number) => void;
  getDueCards: () => CHUNKID[];
}

const Ctx = createContext<WorkspaceCtx | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [ws, setWs] = useState<AskDocsWorkspace>(() => emptyWorkspace("default"));
  const [activeDocId, setActiveDocIdState] = useState<DOCID | null>(null);

  useEffect(() => {
    loadWorkspace("default")
      .then((w) => {
        if (w) setWs(w);
      })
      .catch((e) => {
        // A failed load silently emptied the workspace before; surface it.
        console.error("[Workspace] failed to load saved workspace:", e);
      });
  }, []);

  const setActiveDocId = useCallback((id: DOCID | null) => {
    setActiveDocIdState(id);
  }, []);

  useEffect(() => {
    saveWorkspace(ws).catch((e) => {
      console.warn("[Workspace] could not persist changes:", e);
    });
  }, [ws]);

  const addDocument = useCallback(
    async (
      docId: DOCID,
      text: TEXT,
      format: FORMAT_TYPE,
      hash: TEXT,
      chunks: ChunkRecord[],
    ) => {
      // Z §7.3 IngestDocument: dedup by hash, append, default privacy=false
      setWs((prev) => {
        if (prev.documents[docId]) return prev;
        for (const ex of Object.values(prev.doc_hashes)) {
          if (ex === hash) return prev;
        }
        const next: AskDocsWorkspace = {
          ...prev,
          documents: { ...prev.documents, [docId]: text },
          doc_formats: { ...prev.doc_formats, [docId]: format },
          doc_hashes: { ...prev.doc_hashes, [docId]: hash },
          privacy_cloud_sync_enabled: {
            ...prev.privacy_cloud_sync_enabled,
            [docId]: false,
          },
        };
        const docChunks = prev.doc_chunks[docId] ?? [];
        for (const c of chunks) {
          next.chunks = { ...next.chunks, [c.id]: c };
          docChunks.push(c.id);
        }
        next.doc_chunks = { ...prev.doc_chunks, [docId]: docChunks };
        return next;
      });
    },
    [],
  );

  const setActiveMode = useCallback(
    (mode: MODE) => setWs((p) => ({ ...p, active_mode: mode })),
    [],
  );
  const setConfidenceThreshold = useCallback(
    (t: number) => setWs((p) => ({ ...p, confidence_threshold: Math.min(1, Math.max(0, t)) })),
    [],
  );
  const setZenMode = useCallback(
    (on: boolean) => setWs((p) => ({ ...p, zen_mode_enabled: on })),
    [],
  );
  const setSummaryGranularity = useCallback(
    (g: number) =>
      setWs((p) => ({
        ...p,
        summary_granularity: Math.min(3, Math.max(1, Math.round(g))),
      })),
    [],
  );
  const setPrivacySync = useCallback(
    (docId: DOCID, on: boolean) =>
      setWs((p) => ({
        ...p,
        privacy_cloud_sync_enabled: {
          ...p.privacy_cloud_sync_enabled,
          [docId]: on,
        },
      })),
    [],
  );

  // Z §7.3 ReviewFlashcard — full SM-2 step
  const reviewFlashcard = useCallback(
    (chunkId: CHUNKID, q: RATING, now: number) => {
      setWs((prev) => {
        const f = prev.flashcard;
        if (!f.card_id.includes(chunkId)) return prev;
        const reps = f.repetitions[chunkId] ?? 0;
        const ease = f.easiness[chunkId] ?? 2.5;
        const interval = f.interval[chunkId] ?? 0;
        const newEase = Math.max(
          1.3,
          ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)),
        );
        let newReps = reps;
        let newInterval = interval;
        if (q < 3) {
          newReps = 0;
          newInterval = 1;
        } else if (reps === 0) {
          newReps = 1;
          newInterval = 1;
        } else if (reps === 1) {
          newReps = 2;
          newInterval = 6;
        } else {
          newReps = reps + 1;
          newInterval = Math.round(interval * newEase);
        }
        const due = now + newInterval * 86400000;
        return {
          ...prev,
          flashcard: {
            ...f,
            repetitions: { ...f.repetitions, [chunkId]: newReps },
            interval: { ...f.interval, [chunkId]: newInterval },
            easiness: { ...f.easiness, [chunkId]: newEase },
            due_date: { ...f.due_date, [chunkId]: due },
          },
        };
      });
    },
    [],
  );

  const getDueCards = useCallback((): CHUNKID[] => {
    const now = Date.now();
    return ws.flashcard.card_id.filter(
      (c) => (ws.flashcard.due_date[c] ?? 0) <= now,
    );
  }, [ws.flashcard]);

  return (
    <Ctx.Provider
      value={{
        ws,
        activeDocId,
        setActiveDocId,
        addDocument,
        setActiveMode,
        setConfidenceThreshold,
        setZenMode,
        setSummaryGranularity,
        setPrivacySync,
        reviewFlashcard,
        getDueCards,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useWorkspace(): WorkspaceCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useWorkspace must be used within WorkspaceProvider");
  return c;
}
