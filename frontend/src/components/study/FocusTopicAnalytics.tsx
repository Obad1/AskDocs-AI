import React, { useEffect, useMemo, useRef } from "react";
import Chart from "chart.js/auto";

export interface QuizOutcome {
  topic?: string;
  correct: boolean;
}

// Chart.js breakdown of cumulative score + weak-topic misses.
export function FocusTopicAnalytics({
  results = [],
}: {
  results?: QuizOutcome[];
}) {
  const lineRef = useRef<HTMLCanvasElement>(null);
  const barRef = useRef<HTMLCanvasElement>(null);
  const lineChart = useRef<Chart | null>(null);
  const barChart = useRef<Chart | null>(null);

  const { cumulative, weak } = useMemo(() => {
    let running = 0;
    const cum: number[] = [];
    const missCount: Record<string, number> = {};
    results.forEach((r, i) => {
      if (r.correct) running += 1;
      cum.push(results.length ? Math.round((running / (i + 1)) * 100) : 0);
      if (!r.correct && r.topic) {
        missCount[r.topic] = (missCount[r.topic] ?? 0) + 1;
      }
    });
    const weak = Object.entries(missCount)
      .map(([topic, misses]) => ({ topic, misses }))
      .sort((a, b) => b.misses - a.misses);
    return { cumulative: cum, weak };
  }, [results]);

  useEffect(() => {
    if (!lineRef.current) return;
    lineChart.current?.destroy();
    lineChart.current = new Chart(lineRef.current, {
      type: "line",
      data: {
        labels: cumulative.map((_, i) => `Q${i + 1}`),
        datasets: [
          {
            label: "Cumulative score %",
            data: cumulative,
            borderColor: "#2563eb",
            fill: false,
            tension: 0.3,
          },
        ],
      },
      options: { responsive: true, scales: { y: { min: 0, max: 100 } } },
    });
    return () => lineChart.current?.destroy();
  }, [cumulative]);

  useEffect(() => {
    if (!barRef.current) return;
    barChart.current?.destroy();
    barChart.current = new Chart(barRef.current, {
      type: "bar",
      data: {
        labels: weak.map((w) => w.topic),
        datasets: [
          {
            label: "Misses (weak topics)",
            data: weak.map((w) => w.misses),
            backgroundColor: "#dc2626",
          },
        ],
      },
      options: { responsive: true },
    });
    return () => barChart.current?.destroy();
  }, [weak]);

  if (results.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-gray-300 p-6 text-center text-sm text-gray-400 dark:border-gray-600">
        Complete quizzes to see focus-topic analytics.
      </div>
    );
  }

  return (
    <div className="grid gap-4 rounded-lg border border-gray-200 p-3 dark:border-gray-700 md:grid-cols-2">
      <div>
        <div className="mb-1 text-sm font-medium">Score trend</div>
        <canvas ref={lineRef} />
      </div>
      <div>
        <div className="mb-1 text-sm font-medium">Weak topics</div>
        {weak.length ? (
          <canvas ref={barRef} />
        ) : (
          <p className="text-sm text-gray-400">No weak topics — nice work!</p>
        )}
      </div>
    </div>
  );
}
