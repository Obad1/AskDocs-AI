import React, { Suspense, lazy, useEffect, useState } from "react";

import { UserProfileProvider } from "./context/UserProfileContext";
import { ModelEngineProvider } from "./context/ModelEngineContext";
import { WorkspaceProvider, useWorkspace } from "./context/WorkspaceContext";
import { AudioProvider } from "./context/AudioContext";

import ThemeProvider from "./components/layout/ThemeProvider";
import TopNavigation from "./components/layout/TopNavigation";
import WorkspaceSplitter from "./components/layout/WorkspaceSplitter";
import ZenOverlay from "./components/layout/ZenOverlay";
import DocsSidebar from "./components/layout/DocsSidebar";
import SettingsDrawer from "./components/layout/SettingsDrawer";
import SummaryDrawer from "./components/layout/SummaryDrawer";

import HardwareBenchmarkModal from "./components/onboarding/HardwareBenchmarkModal";
import ProductTour from "./components/onboarding/ProductTour";
import InteractiveDemoModal from "./components/onboarding/InteractiveDemoModal";

// ---- Heavy sibling components (lazy-loaded to keep initial bundle small) ----
// Owned by sibling subsystems; all expose named exports, so map them for React.lazy.
const lazyNamed = <M extends Record<string, any>>(
  loader: () => Promise<M>,
  name: keyof M,
) =>
  lazy(() =>
    loader().then((m) => ({ default: m[name] as React.ComponentType<any> })),
  );

const PDFViewer = lazyNamed(() => import("./components/document/PDFViewer"), "PDFViewer");
const TextCleanerModal = lazyNamed(
  () => import("./components/document/TextCleanerModal"),
  "TextCleanerModal",
);
const MultiDocMatrix = lazyNamed(
  () => import("./components/document/MultiDocMatrix"),
  "MultiDocMatrix",
);

const ChatPane = lazyNamed(() => import("./components/assistant/ChatPane"), "ChatPane");
const CitationDrawer = lazyNamed(
  () => import("./components/assistant/CitationDrawer"),
  "CitationDrawer",
);
const ConfidenceBadge = lazyNamed(
  () => import("./components/assistant/ConfidenceBadge"),
  "ConfidenceBadge",
);

const FlashcardDeck = lazyNamed(
  () => import("./components/study/FlashcardDeck"),
  "FlashcardDeck",
);
const AdaptiveQuiz = lazyNamed(
  () => import("./components/study/AdaptiveQuiz"),
  "AdaptiveQuiz",
);
const FocusTopicAnalytics = lazyNamed(
  () => import("./components/study/FocusTopicAnalytics"),
  "FocusTopicAnalytics",
);
const KnowledgeGraphView = lazyNamed(
  () => import("./components/study/KnowledgeGraphView"),
  "KnowledgeGraphView",
);

const MiniMediaDock = lazy(() => import("./components/audio/MiniMediaDock"));
const KaraokeTranscript = lazy(
  () => import("./components/audio/KaraokeTranscript"),
);
const VoiceInterrupter = lazy(
  () => import("./components/audio/VoiceInterrupter"),
);

const SocialCardExporter = lazy(
  () => import("./components/sharing/SocialCardExporter"),
);
const SharedWorkspaceModal = lazy(
  () => import("./components/sharing/SharedWorkspaceModal"),
);
const EmbedWidgetGenerator = lazy(
  () => import("./components/sharing/EmbedWidgetGenerator"),
);

// ---------------------------------------------------------------------------
// Error boundary so a single sibling component failing doesn't blank the app.
// ---------------------------------------------------------------------------
class SafeBoundary extends React.Component<
  { name: string; overlay?: boolean; children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { name: string; overlay?: boolean; children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch() {
    // Intentionally silent: local-first app, no telemetry.
  }
  render() {
    if (this.state.hasError) {
      if (this.props.overlay) {
        return (
          <OverlayMessage
            icon="⚠"
            title={`${this.props.name} hit an error`}
            body="The app is still running — reload to restore this panel."
            action="Reload app"
          />
        );
      }
      return (
        <div className="flex h-full w-full items-center justify-center p-4 text-center text-sm text-[var(--fg-muted)]">
          {this.props.name} is temporarily unavailable. Reload the page to bring it back.
        </div>
      );
    }
    return this.props.children;
  }
}

function OverlayMessage({
  icon,
  title,
  body,
  action,
}: {
  icon: string;
  title: string;
  body: string;
  action?: string;
}) {
  return (
    <div
      role="alertdialog"
      aria-label={title}
      aria-describedby="safe-error-body"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <div className="w-full max-w-md rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 text-center shadow-xl">
        <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-[var(--danger)]/10 text-lg text-[var(--danger)]">
          {icon}
        </div>
        <h2 className="text-base font-semibold text-[var(--fg)]">{title}</h2>
        <p id="safe-error-body" className="mt-1 text-sm text-[var(--fg-muted)]">
          {body}
        </p>
        {action && (
          <button
            type="button"
            className="btn-primary mt-4 w-full px-4 py-2"
            onClick={() => window.location.reload()}
          >
            {action}
          </button>
        )}
      </div>
    </div>
  );
}

function Safe({
  name,
  children,
  overlay = false,
}: {
  name: string;
  children: React.ReactNode;
  overlay?: boolean;
}) {
  return (
    <SafeBoundary name={name} overlay={overlay}>
      <Suspense
        fallback={
          overlay ? (
            <OverlayMessage
              icon="⋯"
              title={`Loading ${name}…`}
              body="Starting up a heavy module; this only takes a moment."
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-sm text-[var(--fg-muted)]">
              Loading {name}…
            </div>
          )
        }
      >
        {children}
      </Suspense>
    </SafeBoundary>
  );
}

type LeftTab =
  | "document"
  | "matrix"
  | "graph"
  | "analytics"
  | "quiz"
  | "flashcards";

const LEFT_TABS: { id: LeftTab; label: string }[] = [
  { id: "document", label: "Document" },
  { id: "matrix", label: "Matrix" },
  { id: "graph", label: "Graph" },
  { id: "analytics", label: "Topics" },
  { id: "quiz", label: "Quiz" },
  { id: "flashcards", label: "Cards" },
];

// ---------------------------------------------------------------------------
// Main shell
// ---------------------------------------------------------------------------
function Shell() {
  const { ws, setZenMode, activeDocId, setActiveDocId } = useWorkspace();

  const [leftTab, setLeftTab] = useState<LeftTab>("document");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [summaryOpen, setSummaryOpen] = useState(false);
  const [modals, setModals] = useState({
    benchmark: false,
    tour: false,
    demo: false,
    share: false,
    social: false,
    embed: false,
    textCleaner: false,
  });

  const open = (k: keyof typeof modals) =>
    setModals((m) => ({ ...m, [k]: true }));
  const close = (k: keyof typeof modals) =>
    setModals((m) => ({ ...m, [k]: false }));

  // Opening a document from the sidebar/Matrix/Graph jumps to Document tab.
  const openDocument = (id: string) => {
    setActiveDocId(id);
    setLeftTab("document");
    if (window.innerWidth < 1024) setSidebarOpen(false);
  };

  // Opening a document from the Matrix/Graph jumps to the Document tab.
  useEffect(() => {
    if (activeDocId) setLeftTab("document");
  }, [activeDocId]);

  // Hash deep-linking (#matrix, #document:<docId>, …): bookmarkable views that
  // survive reload, plus meaningful Back/Forward navigation between views.
  useEffect(() => {
    const views = new Set<LeftTab>([
      "document",
      "matrix",
      "graph",
      "analytics",
      "quiz",
      "flashcards",
    ]);
    const applyHash = () => {
      const hash = window.location.hash.replace(/^#/, "");
      const [view, doc] = hash.split(":");
      if (view && views.has(view as LeftTab)) {
        setLeftTab(view as LeftTab);
        if (doc) setActiveDocId(doc);
      }
    };
    applyHash();
    window.addEventListener("hashchange", applyHash);
    return () => window.removeEventListener("hashchange", applyHash);
  }, [setActiveDocId]);

  // Mirror the current view into the URL (replaceState keeps history tidy).
  useEffect(() => {
    const frag = activeDocId ? `${leftTab}:${activeDocId}` : leftTab;
    const expected = `#${frag}`;
    if (window.location.hash !== expected) {
      history.replaceState(null, "", expected);
    }
  }, [leftTab, activeDocId]);

  // Auto-select the first ingested document so the Document tab greets the
  // workspace instead of an empty "No document selected" state.
  useEffect(() => {
    if (!activeDocId) {
      const first = Object.keys(ws.documents)[0];
      if (first) setActiveDocId(first);
    }
  }, [ws.documents, activeDocId, setActiveDocId]);

  // Ctrl/Cmd+Shift+Z toggles Zen focus mode.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "z") {
        e.preventDefault();
        setZenMode(!ws.zen_mode_enabled);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [ws.zen_mode_enabled, setZenMode]);

  // Zero-setup onboarding: show a short tour on the first visit only (no login
  // required), never the demo repeatedly. Persisted so it can't nag again.
  useEffect(() => {
    let onboarded = false;
    try {
      onboarded = localStorage.getItem("askdocs.onboarded") === "1";
    } catch {
      /* ignore */
    }
    if (!onboarded) {
      open("tour");
      try {
        localStorage.setItem("askdocs.onboarded", "1");
      } catch {
        /* ignore */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Tour steps spotlight the sidebar (+ Add) and the audio strip — open the
  // sidebar first so the spotlight can find its anchors.
  useEffect(() => {
    if (modals.tour) setSidebarOpen(true);
  }, [modals.tour]);

  const leftPane = (
    <div className="flex h-full flex-col bg-[var(--bg)]">
      <nav
        data-tour="tabs"
        className="flex shrink-0 gap-1 border-b border-[var(--border)] bg-[var(--bg-elevated)] px-2 py-1 text-xs"
      >
        {LEFT_TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setLeftTab(t.id)}
            aria-pressed={leftTab === t.id}
            className={`rounded px-3 py-1.5 ${
              leftTab === t.id
                ? "bg-[var(--accent)] text-[var(--accent-fg)]"
                : "text-[var(--fg-muted)] hover:text-[var(--fg)]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="min-h-0 flex-1 overflow-hidden reading-surface">
        {leftTab === "document" && (
          <Safe name="Document">
            <PDFViewer docId={activeDocId} />
          </Safe>
        )}
        {leftTab === "matrix" && (
          <Safe name="MultiDocMatrix">
            <MultiDocMatrix />
          </Safe>
        )}
        {leftTab === "graph" && (
          <Safe name="KnowledgeGraphView">
            <KnowledgeGraphView />
          </Safe>
        )}
        {leftTab === "analytics" && (
          <Safe name="FocusTopicAnalytics">
            <FocusTopicAnalytics />
          </Safe>
        )}
        {leftTab === "quiz" && (
          <Safe name="AdaptiveQuiz">
            <AdaptiveQuiz />
          </Safe>
        )}
        {leftTab === "flashcards" && (
          <Safe name="FlashcardDeck">
            <FlashcardDeck />
          </Safe>
        )}
      </div>
    </div>
  );

  const rightPane = (
    <div className="flex h-full flex-col bg-[var(--bg-elevated)]">
      <div className="min-h-0 flex-[1.2] overflow-hidden">
        <Safe name="ChatPane">
          <ChatPane />
        </Safe>
      </div>

      {/* Audio studio: contextual to the chat/artifact column, not a global bar. */}
      <div
        data-tour="audio"
        className="flex shrink-0 items-center gap-3 border-t border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-2"
      >
        <Safe name="MiniMediaDock">
          <MiniMediaDock />
        </Safe>
        <div className="min-w-0 flex-1">
          <Safe name="KaraokeTranscript">
            <KaraokeTranscript />
          </Safe>
        </div>
        <Safe name="VoiceInterrupter">
          <VoiceInterrupter />
        </Safe>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden border-t border-[var(--border)]">
        <Safe name="CitationDrawer">
          <CitationDrawer />
        </Safe>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden">
      {!ws.zen_mode_enabled && (
        <TopNavigation
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
        />
      )}

      <div className="flex min-h-0 flex-1">
        {!ws.zen_mode_enabled && (
          <DocsSidebar
            open={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            onOpenSettings={() => setSettingsOpen(true)}
            onOpenSummarize={() => setSummaryOpen(true)}
            onOpenBenchmark={() => open("benchmark")}
            onOpenTour={() => open("tour")}
            onOpenDemo={() => open("demo")}
            onOpenDocument={openDocument}
          />
        )}

        <main className="min-h-0 min-w-0 flex-1">
          {ws.zen_mode_enabled ? (
            <div className="h-full">
              {leftTab === "document" ? (
                <Safe name="Document">
                  <PDFViewer docId={activeDocId} />
                </Safe>
              ) : (
                leftPane
              )}
            </div>
          ) : (
            <WorkspaceSplitter left={leftPane} right={rightPane} initial={65} />
          )}
        </main>
      </div>

      {ws.zen_mode_enabled && (
        <ZenOverlay
          onExit={() => setZenMode(false)}
          onOpenDemo={() => open("demo")}
        />
      )}

      {/* ---- Modals ---- */}
      <HardwareBenchmarkModal
        open={modals.benchmark}
        onClose={() => close("benchmark")}
      />
      <ProductTour
        open={modals.tour}
        onClose={() => close("tour")}
        onOpenDemo={() => open("demo")}
      />
      <InteractiveDemoModal open={modals.demo} onClose={() => close("demo")} />

      <Safe name="TextCleanerModal" overlay>
        <TextCleanerModal
          open={modals.textCleaner}
          onClose={() => close("textCleaner")}
        />
      </Safe>
      <Safe name="SharedWorkspaceModal" overlay>
        <SharedWorkspaceModal
          open={modals.share}
          onClose={() => close("share")}
        />
      </Safe>
      <Safe name="SocialCardExporter" overlay>
        <SocialCardExporter
          open={modals.social}
          onClose={() => close("social")}
        />
      </Safe>
      <Safe name="EmbedWidgetGenerator" overlay>
        <EmbedWidgetGenerator
          open={modals.embed}
          onClose={() => close("embed")}
        />
      </Safe>

      <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <SummaryDrawer open={summaryOpen} onClose={() => setSummaryOpen(false)} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Provider composition.
// Ordering note (assumption): AudioProvider depends on useModelEngine(), so it
// must sit inside ModelEngineProvider. ThemeProvider depends on
// useUserProfile(), so it sits inside UserProfileProvider.
// ---------------------------------------------------------------------------
export default function App() {
  return (
    <UserProfileProvider>
      <ThemeProvider>
        <ModelEngineProvider>
          <WorkspaceProvider>
            <AudioProvider>
              <Shell />
            </AudioProvider>
          </WorkspaceProvider>
        </ModelEngineProvider>
      </ThemeProvider>
    </UserProfileProvider>
  );
}
