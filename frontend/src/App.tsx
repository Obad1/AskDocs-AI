import React, { Suspense, lazy, useEffect, useState } from "react";

import { UserProfileProvider } from "./context/UserProfileContext";
import { ModelEngineProvider } from "./context/ModelEngineContext";
import { WorkspaceProvider, useWorkspace } from "./context/WorkspaceContext";
import { AudioProvider } from "./context/AudioContext";

import ThemeProvider from "./components/layout/ThemeProvider";
import TopNavigation from "./components/layout/TopNavigation";
import WorkspaceSplitter from "./components/layout/WorkspaceSplitter";
import ZenOverlay from "./components/layout/ZenOverlay";

import HardwareBenchmarkModal from "./components/onboarding/HardwareBenchmarkModal";
import GuidedTourOverlay from "./components/onboarding/GuidedTourOverlay";
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
const SummaryGranularitySlider = lazyNamed(
  () => import("./components/assistant/SummaryGranularitySlider"),
  "SummaryGranularitySlider",
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
  { name: string; children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { name: string; children: React.ReactNode }) {
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
      return (
        <div className="flex h-full w-full items-center justify-center p-4 text-center text-sm text-[var(--fg-muted)]">
          {this.props.name} unavailable.
        </div>
      );
    }
    return this.props.children;
  }
}

function Safe({
  name,
  children,
}: {
  name: string;
  children: React.ReactNode;
}) {
  return (
    <SafeBoundary name={name}>
      <Suspense
        fallback={
          <div className="flex h-full w-full items-center justify-center text-sm text-[var(--fg-muted)]">
            Loading {name}…
          </div>
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
  const { ws, setZenMode } = useWorkspace();

  const [leftTab, setLeftTab] = useState<LeftTab>("document");
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

  // Zero-setup: open the demo workspace on first load (no login required).
  useEffect(() => {
    open("demo");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const leftPane = (
    <div className="flex h-full flex-col bg-[var(--bg)]">
      <nav className="flex shrink-0 gap-1 border-b border-[var(--border)] bg-[var(--bg-elevated)] px-2 py-1 text-xs">
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
            <PDFViewer />
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
      <div className="shrink-0 border-b border-[var(--border)] p-2">
        <Safe name="SummaryGranularitySlider">
          <SummaryGranularitySlider />
        </Safe>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden">
        <Safe name="ChatPane">
          <ChatPane />
        </Safe>
      </div>
      <div className="min-h-0 flex-1 overflow-hidden border-t border-[var(--border)]">
        <Safe name="CitationDrawer">
          <CitationDrawer />
        </Safe>
      </div>
    </div>
  );

  const bottomDock = (
    <footer className="flex shrink-0 items-center gap-3 border-t border-[var(--border)] bg-[var(--bg-elevated)] px-3 py-2">
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
    </footer>
  );

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden">
      {!ws.zen_mode_enabled && (
        <TopNavigation
          onOpenBenchmark={() => open("benchmark")}
          onOpenTour={() => open("tour")}
          onOpenDemo={() => open("demo")}
        />
      )}

      <main className="min-h-0 flex-1">
        {ws.zen_mode_enabled ? (
          <div className="h-full">
            {leftTab === "document" ? (
              <Safe name="Document">
                <PDFViewer />
              </Safe>
            ) : (
              leftPane
            )}
          </div>
        ) : (
          <WorkspaceSplitter left={leftPane} right={rightPane} initial={65} />
        )}
      </main>

      {!ws.zen_mode_enabled && bottomDock}

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
      <GuidedTourOverlay
        open={modals.tour}
        onClose={() => close("tour")}
        onOpenDemo={() => open("demo")}
      />
      <InteractiveDemoModal open={modals.demo} onClose={() => close("demo")} />

      <Safe name="TextCleanerModal">
        <TextCleanerModal
          open={modals.textCleaner}
          onClose={() => close("textCleaner")}
        />
      </Safe>
      <Safe name="SharedWorkspaceModal">
        <SharedWorkspaceModal
          open={modals.share}
          onClose={() => close("share")}
        />
      </Safe>
      <Safe name="SocialCardExporter">
        <SocialCardExporter
          open={modals.social}
          onClose={() => close("social")}
        />
      </Safe>
      <Safe name="EmbedWidgetGenerator">
        <EmbedWidgetGenerator
          open={modals.embed}
          onClose={() => close("embed")}
        />
      </Safe>
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
