import React, { useState } from "react";
import { useUserProfile } from "../../context/UserProfileContext";
import {
  BUILT_IN_PERSONAS,
  newCustomPersonaId,
  type CustomPersona,
} from "../../lib/personas";

/**
 * Explanation-style picker for the prompt bar. Selecting a persona rephrases
 * how the local model presents answers (chat and slide generation), and custom
 * personas persist in the user profile — all on-device.
 */
export default function PersonaPicker() {
  const { profile, update } = useUserProfile();
  const [adding, setAdding] = useState(false);
  const [label, setLabel] = useState("");
  const [instruction, setInstruction] = useState("");

  const customs = profile.customPersonas;

  const addCustom = () => {
    const trimmed = { label: label.trim(), instruction: instruction.trim() };
    if (!trimmed.label || !trimmed.instruction) return;
    const persona: CustomPersona = {
      id: newCustomPersonaId(),
      label: trimmed.label,
      description: "Custom explanation style.",
      instruction: trimmed.instruction,
      custom: true,
    };
    update({
      customPersonas: [...customs, persona],
      personaId: persona.id,
    });
    setLabel("");
    setInstruction("");
    setAdding(false);
  };

  return (
    <div className="relative">
      <div className="flex items-center gap-1">
        <label className="text-xs text-[var(--fg-muted)]" htmlFor="persona-picker">
          Style
        </label>
        <select
          id="persona-picker"
          aria-label="Explanation style"
          value={profile.personaId}
          onChange={(e) => update({ personaId: e.target.value })}
          className="field px-2 py-1 text-xs"
        >
          {BUILT_IN_PERSONAS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
          {customs.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
        <button
          onClick={() => setAdding((v) => !v)}
          aria-pressed={adding}
          aria-label="Add a custom explanation style"
          className="btn-ghost rounded px-2 py-1 text-xs"
        >
          + Custom
        </button>
      </div>

      {adding && (
        <div className="absolute bottom-full right-0 z-40 mb-2 w-80 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3 shadow-xl">
          <div className="mb-2 text-xs font-semibold text-[var(--fg)]">
            Custom style
          </div>
          <label className="mb-1 block text-xs text-[var(--fg-muted)]">
            Name
          </label>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g. Legal reviewer"
            className="field mb-2 w-full px-2 py-1.5 text-sm"
          />
          <label className="mb-1 block text-xs text-[var(--fg-muted)]">
            How should answers be worded?
          </label>
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="e.g. Be precise about obligations and caveats; cite clauses literally."
            rows={3}
            className="field mb-2 w-full resize-none px-2 py-1.5 text-sm"
          />
          <div className="flex justify-end gap-1">
            <button
              onClick={() => setAdding(false)}
              className="btn-ghost px-2 py-1 text-xs"
            >
              Cancel
            </button>
            <button
              onClick={addCustom}
              className="btn-primary px-3 py-1 text-xs"
            >
              Add style
            </button>
          </div>
        </div>
      )}
    </div>
  );
}