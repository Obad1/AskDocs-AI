// Personas (explanation styles) applied as an extra system instruction to the
// local model across all generation intents (chat, slides). Built-ins ship with
// the app; custom ones are stored in the user profile and kept entirely local.

export interface Persona {
  id: string;
  label: string;
  description: string;
  /** Appended to the LLM system prompt verbatim. Empty for the default. */
  instruction: string;
}

export interface CustomPersona extends Persona {
  custom: true;
}

export const BUILT_IN_PERSONAS: Persona[] = [
  {
    id: "standard",
    label: "Standard / Objective",
    description: "Neutral, direct factual synthesis.",
    instruction: "",
  },
  {
    id: "eli5",
    label: "Explain like I'm five",
    description: "Simple words, everyday analogies, no jargon.",
    instruction:
      "Explain the answer simply, as if to a bright child. Use everyday analogies and avoid jargon; define every technical term you must use.",
  },
  {
    id: "coach",
    label: "Performance Coach",
    description: "Action-oriented, motivational, punchy highlights.",
    instruction:
      "Be action-oriented and motivational. Lead with punchy highlights and end with concrete next steps the reader can take.",
  },
  {
    id: "executive",
    label: "Executive (C-Suite)",
    description: "Conclusions first, concise and decision-focused.",
    instruction:
      "Write for an executive: lead with conclusions and actionable takeaways, keep it concise, and surface risks or caveats explicitly.",
  },
];

export function personaInstruction(id: string, custom: CustomPersona[]): string {
  if (id === "standard") return "";
  for (const p of BUILT_IN_PERSONAS) {
    if (p.id === id) return p.instruction;
  }
  return custom.find((c) => c.id === id)?.instruction ?? "";
}

export function newCustomPersonaId(): string {
  return `custom-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}