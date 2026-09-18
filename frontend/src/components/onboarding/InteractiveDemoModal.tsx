import React, { useState } from "react";
import { useIngest } from "../../lib/parsing/ingestClient";
import { useWorkspace } from "../../context/WorkspaceContext";
import type { Progress } from "../../workers/indexing.worker";

interface InteractiveDemoModalProps {
  open: boolean;
  onClose: () => void;
}

// Bundled zero-setup sample documents (spec §3.9). They go through the same
// indexing worker as user uploads, so the "demo workspace" is genuinely
// queryable — not a simulated success screen.
const SAMPLE_DOCS: { name: string; markdown: string }[] = [
  {
    name: "sample-photosynthesis.txt",
    markdown: `Photosynthesis — Study Guide

Overview

Photosynthesis is the process by which green plants, algae, and certain
bacteria convert light energy into chemical energy stored in glucose. The
overall chemical equation is 6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2.
It is the foundation of almost every food chain on Earth and produces the
oxygen we breathe.

The chloroplast

Photosynthesis takes place inside chloroplasts, organelles that contain stacks
of thylakoid membranes called grana. The fluid surrounding the grana is the
stroma. Chlorophyll, the pigment that gives plants their green colour, sits
embedded in the thylakoid membranes and absorbs light most strongly in the
blue-violet and red parts of the spectrum while reflecting green.

The light-dependent reactions

These reactions occur in the thylakoid membrane and require light. Photons
excite electrons in chlorophyll, which pass along an electron transport chain.
This pump of electrons powers chemiosmosis: hydrogen ions accumulate inside the
thylakoid lumen and flow back out through ATP synthase, generating ATP. At the
end of the chain the electrons are passed to NADP+ together with a proton,
forming NADPH. Water is split at the start of the chain, releasing oxygen gas
as a by-product — this is the oxygen we breathe.

The Calvin cycle

The Calvin cycle takes place in the stroma and does not require light directly,
which is why it is sometimes called the light-independent reactions. Carbon
dioxide is fixed onto a five-carbon acceptor, ribulose bisphosphate (RuBP), by
the enzyme rubisco. The resulting six-carbon intermediate immediately splits
into two molecules of 3-phosphoglycerate. Using the ATP and NADPH produced in
the light-dependent reactions, these molecules are reduced to
glyceraldehyde-3-phosphate (G3P). Most G3P molecules are recycled to
regenerate RuBP; one in every six is exported to build glucose and other
sugars.

Factors affecting the rate

The rate of photosynthesis is limited by light intensity, carbon dioxide
concentration and temperature, and whichever factor is closest to its minimum
at any moment is the limiting factor. Increasing any other factor has no
effect until the limiting factor is raised. At very high temperatures the
rubisco enzyme denatures, and the rate collapses despite abundant light and
carbon dioxide.

C4 and CAM plants

Some plants evolved alternate pathways to concentrate carbon dioxide around
rubisco. C4 plants such as maize spatially separate carbon fixation (in mesophyll
cells) from the Calvin cycle (in bundle-sheath cells), using a four-carbon
intermediate. CAM plants such as cacti and pineapple open their stomata at
night, fix CO2 into malate overnight, and release it during the day inside the
leaf, letting them avoid water loss while photosynthesis is active.

Key terms to remember

Chloroplast, granum, stroma, thylakoid, chlorophyll, photolysis, electron
transport chain, chemiosmosis, ATP synthase, NADPH, Calvin cycle, rubisco,
RuBP, 3-phosphoglycerate, G3P, limiting factor, C4 pathway, CAM pathway.`,
  },
  {
    name: "sample-ww2-outline.txt",
    markdown: `World War II — Outline

Causes and the road to war

The Treaty of Versailles (1919) imposed heavy reparations and territorial
losses on Germany, fuelling resentment exploited by the Nazi party. The
global economic depression of the 1930s destroyed German and Japanese export
markets. Japan's militarist government sought resources and living space in
China and Southeast Asia, invading Manchuria in 1931 and launching full-scale
war against China in 1937. In Europe, Adolf Hitler remilitarised the
Rhineland in 1936, annexed Austria in 1938 and, after the Munich Agreement
allowed Germany to take the Sudetenland, occupied the rest of Czechoslovakia
in March 1939.

The outbreak of war

Germany and the Soviet Union signed a non-aggression pact in August 1939. On
1 September 1939 Germany invaded Poland, and Britain and France declared war
two days later. The Soviet Union invaded Poland from the east on 17
September. The first phase of the war was a series of quick, mobile
campaigns using the concept of blitzkrieg — coordinated tanks, aircraft and
infantry designed to break through enemy lines and encircle armies.

The major fronts

The European theatre: after the defeat of Poland came intensive air warfare
the Battle of Britain in 1940, followed by the invasion of the Soviet Union
in June 1941. The Eastern Front became the largest and bloodiest theatre of
the whole conflict, culminating at Stalingrad in the winter of 1942-43 where
the German Sixth Army was surrounded and destroyed — widely seen as the
turning point in Europe.

The Pacific theatre: the war began for the United States with the surprise
attack on Pearl Harbor on 7 December 1941. The US then fought a series of
island-hopping campaigns in the Pacific, while naval battles such as Midway
in June 1942 destroyed four Japanese aircraft carriers and checked Japanese
expansion.

The war in Africa and the Mediterranean saw the Western Desert campaign
between British and Axis forces centring on El Alamein, followed by an
Anglo-American invasion of North Africa and then Italy.

Turning points

Stalingrad (1942-43) marked the failure of the German offensive in the east.
Midway (1942) was the decisive reversal in the Pacific. The Allied landings
in Normandy on 6 June 1944, known as D-Day, opened the second front in
western Europe, leading to the liberation of France and the final drive into
Germany. The Soviet Red Army advanced from the east, capturing Berlin in
April-May 1945. Germany surrendered unconditionally on 8 May 1945.

The end of the war and the atomic bomb

Japan continued to fight after Germany's surrender. President Truman chose to
use atomic bombs on Hiroshima (6 August 1945) and Nagasaki (9 August 1945).
Japan surrendered on 15 August 1945, and the formal surrender was signed on 2
September 1945 aboard the USS Missouri.

Consequences

Over sixty million people died, making it the deadliest conflict in history.
The war reshaped the world order: the United Nations was founded in 1945, the
United States and the Soviet Union emerged as superpowers, and the division
of Europe and of Korea set the stage for the Cold War that would dominate the
second half of the twentieth century.`,
  },
  {
    name: "sample-python-primer.txt",
    markdown: `Python — Quick Primer

Why Python

Python is a high-level, interpreted programming language known for its
readable syntax and large standard library. It is used for web development
with frameworks such as Flask and Django, for data analysis with pandas and
NumPy, for machine learning with scikit-learn and PyTorch, and for scripting
and automation.

Variables and basic types

Python is dynamically typed: a variable takes the type of whatever is assigned
to it. Common built-in types are int, float, str, bool, list, tuple, dict and
set. Strings are immutable sequences of characters; lists are mutable ordered
collections; tuples are immutable sequences; dictionaries map keys to values;
sets hold unique unordered elements.

Control flow

The conditional statement if, elif and else controls branching. Indentation,
not braces, defines blocks. The for loop iterates over any iterable, while the
while loop repeats while a condition is true. The keywords break, continue
and pass control single-loop behaviour; break exits the nearest loop,
continue skips to the next iteration, and pass is a no-op placeholder.

Functions

Functions are defined with the def keyword, can accept positional and
keyword arguments, may declare default values, and return values with the
return statement. Lambda expressions create small anonymous functions.
Arguments can be captured with *args for a tuple of positional arguments and
**kwargs for a dictionary of keyword arguments.

Comprehensions

Comprehensions are a concise way to build collections. A list comprehension
[expr for item in iterable if condition] replaces an explicit loop. Similar
syntax exists for sets and dictionaries. Generators use the same syntax with
round parentheses and yield values lazily, which saves memory for large
sequences.

Errors and exceptions

Errors are raised explicitly with the raise statement; the try, except,
else and finally clauses handle them. A try block runs, and if an exception
matching the except clause is raised, that handler runs. The else clause runs
only if no exception occurred; the finally clause always runs, which makes it
useful for cleanup.

Working with files

The open function returns a file object; the with statement ensures the file
is closed automatically even when an exception occurs: with open('data.txt')
as f: text = f.read(). There are read, readline and readlines methods for
text, and csv and json modules for structured data.

Built-in helpers worth memorising

len, type, range, enumerate, zip, sorted, min, max, sum, any, all, map,
filter, and the str formatting methods. The dir and help functions give
interactive documentation in the interpreter.`,
  },
];

/**
 * Interactive Demo Modal (spec §3.9). Launches a zero-setup demo workspace with
 * preloaded local sample documents — no login, no API keys, no telemetry.
 * The samples are ingested through the real local indexing worker, so the demo
 * workspace is genuinely queryable and any failure is surfaced honestly.
 */
export default function InteractiveDemoModal({
  open,
  onClose,
}: InteractiveDemoModalProps) {
  const ingest = useIngest();
  const { ws } = useWorkspace();
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [counts, setCounts] = useState({ docs: 0, chunks: 0 });
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const alreadyLoaded = () => {
    const total = Object.keys(ws.documents).length;
    return total > 0 ? `${total}` : "0";
  };

  const launch = async () => {
    if (loading) return;
    setLoading(true);
    setError(null);
    setReady(false);
    setCounts({ docs: 0, chunks: 0 });

    for (let i = 0; i < SAMPLE_DOCS.length; i++) {
      const sample = SAMPLE_DOCS[i];
      try {
        const file = new File([sample.markdown], sample.name, {
          type: "text/plain",
        });
        const result = await ingest(file, (p: Progress) => setStage(p.stage));
        setCounts((c) => ({
          docs: c.docs + 1,
          chunks: c.chunks + result.numChunks,
        }));
      } catch (e) {
        setLoading(false);
        setStage(null);
        setError(e instanceof Error ? e.message : String(e));
        return;
      }
    }

    setLoading(false);
    setStage(null);
    setReady(true);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Interactive demo"
    >
      <div className="w-full max-w-md rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 text-[var(--fg)] shadow-2xl">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Zero-Setup Demo</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-[var(--fg-muted)] hover:text-[var(--fg)]"
          >
            ✕
          </button>
        </div>
        <p className="mb-4 text-sm text-[var(--fg-muted)]">
          Explore AskDocs AI with preloaded sample documents — entirely offline.
          No account required.
        </p>

        {error ? (
          <div className="rounded-lg border border-[var(--danger)] bg-[var(--bg)] p-4 text-sm">
            <p className="font-medium text-[var(--danger)]">Demo load failed</p>
            <p className="mt-1 break-words text-[var(--fg-muted)]">{error}</p>
            <p className="mt-2 text-xs text-[var(--fg-muted)]">
              The local model weights may still be downloading — try again in a
              moment, or upload your own documents.
            </p>
          </div>
        ) : ready ? (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-4 text-sm">
            <p className="font-medium text-[var(--fg)]">Demo workspace ready ✓</p>
            <p className="mt-1 text-[var(--fg-muted)]">
              {counts.docs} sample {counts.docs === 1 ? "document" : "documents"},{" "}
              {counts.chunks} chunks indexed locally. Close this to start
              exploring.
            </p>
          </div>
        ) : (
          <button
            onClick={launch}
            disabled={loading}
            className="btn-primary w-full px-4 py-2 disabled:opacity-60"
          >
            {loading
              ? `Loading samples… ${counts.docs + 1}/${SAMPLE_DOCS.length}${
                  stage ? ` · ${stage}` : ""
                }`
              : "Load sample dataset"}
          </button>
        )}

        <div className="mt-5 flex items-center justify-end gap-3">
          <span className="mr-auto text-xs text-[var(--fg-muted)]">
            {!loading && !ready && !error && ws.documents
              ? `${alreadyLoaded()} doc${
                  alreadyLoaded() === "1" ? "" : "s"
                } already in workspace`
              : "\u00A0"}
          </span>
          <button
            onClick={onClose}
            className="btn-ghost px-4 py-2 text-sm"
          >
            {ready || error ? "Close" : "Cancel"}
          </button>
        </div>
      </div>
    </div>
  );
}