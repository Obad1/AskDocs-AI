import React, { useEffect, useState } from "react";
import { useModelEngine, MODEL_CATALOG } from "../../context/ModelEngineContext";

interface HardwareBenchmarkModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Hardware Benchmark Panel (spec §3.9). Feature-detects WebGPU / cores / RAM
 * and recommends a hardware tier (§5). No telemetry. On open it runs the
 * benchmark; the Apply button commits the detected tier to the engine.
 */
export default function HardwareBenchmarkModal({
  open,
  onClose,
}: HardwareBenchmarkModalProps) {
  const { benchmark, runBenchmark, applyDetectedTier } = useModelEngine();
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (open && !benchmark && !running) {
      setRunning(true);
      runBenchmark()
        .catch(() => {})
        .finally(() => setRunning(false));
    }
  }, [open, benchmark, running, runBenchmark]);

  if (!open) return null;

  const tier = benchmark?.tier ?? "Tier0_Minimal";
  const recommended = MODEL_CATALOG[tier];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Hardware benchmark"
    >
      <div className="w-full max-w-lg rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 text-[var(--fg)] shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Hardware Benchmark</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-[var(--fg-muted)] hover:text-[var(--fg)]"
          >
            ✕
          </button>
        </div>

        {running || !benchmark ? (
          <p className="py-8 text-center text-[var(--fg-muted)]">
            Checking this device for available local models…
          </p>
        ) : (
          <>
            <div className="mb-4 grid grid-cols-3 gap-3 text-center">
              <Stat label="Tier" value={tier.replace("Tier", "T")} />
              <Stat label="WebGPU" value={benchmark.webgpu ? "Yes" : "No"} />
              <Stat label="Cores" value={String(benchmark.cores)} />
              <Stat
                label="RAM"
                value={`${Math.round(benchmark.memoryMB / 1024)} GB`}
              />
            </div>

            <p className="mb-3 text-sm text-[var(--fg-muted)]">
              Recommended local models (no API keys, fully offline):
            </p>
            <ul className="mb-5 space-y-1 text-sm">
              <li>
                <span className="text-[var(--fg-muted)]">Embedding:</span>{" "}
                {recommended.embedding.label}
              </li>
              <li>
                <span className="text-[var(--fg-muted)]">LLM:</span>{" "}
                {recommended.llm.label}
              </li>
              <li>
                <span className="text-[var(--fg-muted)]">TTS:</span>{" "}
                {recommended.tts.label}
              </li>
              <li>
                <span className="text-[var(--fg-muted)]">STT:</span>{" "}
                {recommended.stt.label}
              </li>
            </ul>

            <div className="flex justify-end gap-2">
              <button
                onClick={onClose}
                className="btn-ghost px-4 py-2"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  applyDetectedTier(benchmark.tier, benchmark.webgpu);
                  onClose();
                }}
                className="btn-primary px-4 py-2"
              >
                Apply recommended tier
              </button>
            </div>
          </>
        )}
        <p className="mt-4 text-xs text-[var(--fg-muted)]">
          No data leaves your device. Benchmark runs locally only.
        </p>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-3">
      <div className="text-xs text-[var(--fg-muted)]">{label}</div>
      <div className="mt-1 text-base font-semibold">{value}</div>
    </div>
  );
}
