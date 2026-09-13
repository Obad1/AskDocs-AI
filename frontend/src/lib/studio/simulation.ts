// simulation.ts — Avam Search Studio: local, zero-LLM tactical co-simulation
// builder (PRD err_log.txt §2). Maps chunks of the current workspace onto an
// 11v11-style pitch metaphor, picks a source-grounded "freeze-frame" decision
// point, and writes the synced-Audio transcript. Purely heuristic + grounded
// in the real chunk text; no API keys, no model required.

import type { ChunkRecord, AskDocsWorkspace } from "../../types/schema";

export type StudioRole =
  | "Goalkeeper"
  | "Defender"
  | "Midfielder"
  | "Playmaker"
  | "Striker";

export type StudioStatus = "locked" | "pressed" | "open";

export interface StudioSource {
  docId: string;
  page?: number;
  text: string;
}

export interface StudioNode {
  id: string;
  label: string;
  role: StudioRole;
  x: number;
  y: number;
  status: StudioStatus;
  source: StudioSource;
}

export interface StudioChoice {
  id: string;
  label: string;
  effect: string;
  isCorrect: boolean;
  source: StudioSource;
}

export interface StudioDecision {
  prompt: string;
  atLine: number;
  choices: [StudioChoice, StudioChoice];
}

export interface TranscriptLine {
  text: string;
  nodeId?: string;
}

export interface StudioSimulation {
  title: string;
  sourceCount: number;
  nodes: StudioNode[];
  transcript: TranscriptLine[];
  decision: StudioDecision;
  emphasizes: string[]; // top-of-mind concepts surfaced in the briefing
}

// ---- Heading-to-role heuristics (tactical mapping of abstract concepts) ----
const ROLE_RULES: { role: StudioRole; match: RegExp }[] = [
  {
    role: "Goalkeeper",
    match:
      /debt|illiquid|liabilit|volatil|risk|constraint|bottleneck|deadlock|starv|burn|deplet|exhaust|crash|fail|downside|loss/i,
  },
  {
    role: "Defender",
    match:
      /securit|defen[cs]e|threat|firewall|complian|buffer|reserve|margin|safety|guard|protect|prevent|hedge/i,
  },
  {
    role: "Midfielder",
    match:
      /working capital|operat|cash|throughput|supply|process|schedule|queue|bandwidth|capacity|liquidity|flow|trade|inventory/i,
  },
  {
    role: "Striker",
    match:
      /revenue|profit|growth|return|value|forward|advance|advantage|alpha|yield|income|upside|score|win/i,
  },
];

export const ROLE_POSITIONS: Record<StudioRole, { x: number; y: number }> = {
  Goalkeeper: { x: 6, y: 50 },
  Defender: { x: 26, y: 22 },
  Midfielder: { x: 46, y: 50 },
  Playmaker: { x: 62, y: 30 },
  Striker: { x: 84, y: 62 },
};

const MAX_NODES = 6;
const MIN_CHUNK_WORDS = 30;

function words(s: string): number {
  return s.trim().split(/\s+/).filter(Boolean).length;
}

function firstLine(s: string, max = 72): string {
  const clean = s.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1).trimEnd()}…` : clean;
}

function pickRole(text: string): StudioRole {
  for (const rule of ROLE_RULES) if (rule.match.test(text)) return rule.role;
  return "Playmaker";
}

function shortQuote(text: string, max = 48): string {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > max ? `${t.slice(0, max - 1).trimEnd()}…` : t;
}

function pick(c: ChunkRecord, idx: number): StudioSource {
  return { docId: c.docId, page: c.page, text: c.text };
}

/**
 * Build the frozen-frame simulation for the current workspace. Returns null
 * when there isn't enough material to field a team yet.
 */
export function buildSimulation(ws: AskDocsWorkspace): StudioSimulation | null {
  const chunks = Object.values(ws.chunks).filter((c) => words(c.text) >= MIN_CHUNK_WORDS);

  if (chunks.length === 0) return null;

  const ranked = [...chunks].sort(
    (a, b) => words(b.text) - words(a.text) || a.id.localeCompare(b.id),
  );

  const emphasized = ranked.slice(0, 5).map((c) =>
    firstLine(c.text, 64),
  );

  const picked = ranked.slice(0, MAX_NODES);
  if (picked.length < 4) return null;

  const nodes: StudioNode[] = picked.map((c, i) => {
    const role = pickRole(c.text);
    const pos = ROLE_POSITIONS[role];
    const bumped = i < 4 && pos.x === 0 ? { x: pos.x + 4, y: pos.y + (i % 2) * 6 } : pos;
    return {
      id: c.id,
      label: shortQuote(c.text),
      role,
      x: bumped.x + (i % 3) * 2,
      y: bumped.y + (i % 2) * 4,
      status: (["locked", "pressed", "open", "open", "pressed", "locked"] as StudioStatus[])[i] ?? "open",
      source: pick(c, i),
    };
  });

  nodes[0].status = "pressed";
  nodes[1].status = "locked";

  // Freeze-frame decision: ground option A in the top chunk, option B in a
  // different chunk's phrasing so both choices carry a real citation.
  const anchor = ranked[0];
  const distractor =
    ranked.find((c) => c.id !== anchor.id && /(rarely|never|must|always|failure|risk)/i.test(c.text)) ??
    ranked[1] ??
    anchor;

  const anchorQuote = shortQuote(anchor.text, 110);
  const distractorQuote = shortQuote(distractor.text, 110);
  const promptRoot = firstLine(anchor.text, 140);

  const transcript: TranscriptLine[] = [];
  transcript.push({
    text: `Welcome to the studio. A ${nodes.length}-player picture of your workspace, frozen at the critical decision point.`,
  });
  for (const n of nodes) {
    transcript.push({ text: `${n.role} ${n.label}`, nodeId: n.id });
  }
  const decisionLine: TranscriptLine = {
    text: `The match is frozen: ${promptRoot} Select your play.`,
  };

  const wrongLine = firstLine(distractor.text, 76);

  const decision: StudioDecision = {
    prompt: promptRoot,
    atLine: transcript.length,
    choices: [
      {
        id: "play_A",
        label: shortQuote(anchor.text, 52),
        effect: `Runs with the source: ${anchorQuote}`,
        isCorrect: true,
        source: pick(anchor, 0),
      },
      {
        id: "play_B",
        label: wrongLine,
        effect: `Straying from the material: ${distractorQuote}`,
        isCorrect: false,
        source: pick(distractor, 1),
      },
    ],
  };
  transcript.push(decisionLine);

  return {
    title: `${nodes.length}v${nodes.length} Concept Match`,
    sourceCount: Object.keys(ws.documents).length,
    nodes,
    transcript,
    decision,
    emphasizes: emphasized,
  };
}