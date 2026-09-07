import JSZip from "jszip";

// Export notes as an Obsidian-compatible vault: one Markdown file per note with
// YAML frontmatter (title, tags, source). Zipped client-side via JSZip. No API.
//
// Contracts relied on:
//  - jszip: new JSZip() -> .file(path, content) -> .generateAsync({type:"blob"})
//  - Note shape mirrors ChunkRecord-derived notes (title/body/tags).
export interface ObsidianNote {
  title: string;
  body: string;
  tags?: string[];
}

function slugify(s: string): string {
  return (
    s
      .toLowerCase()
      .replace(/[^\w\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .slice(0, 80) || "untitled"
  );
}

function yamlFrontmatter(note: ObsidianNote): string {
  const tags = note.tags && note.tags.length ? note.tags : [];
  const tagYaml = tags.map((t) => `  - ${t.replace(/[^\w-]/g, "_")}`).join("\n");
  return `---\ntitle: ${JSON.stringify(note.title)}\ntags:\n${tagYaml || "  []"}\nsource: AskDocs AI\n---\n\n`;
}

export async function exportObsidian(
  notes: ObsidianNote[],
  vaultName = "askdocs-vault",
): Promise<Blob> {
  const zip = new JSZip();
  const seen = new Set<string>();
  notes.forEach((note, i) => {
    let base = slugify(note.title);
    if (seen.has(base)) base = `${base}-${i}`;
    seen.add(base);
    const content = `${yamlFrontmatter(note)}# ${note.title}\n\n${note.body}\n`;
    zip.file(`${vaultName}/${base}.md`, content);
  });
  zip.file(
    `${vaultName}/README.md`,
    `# ${vaultName}\n\nExported from AskDocs AI. Open this folder as an Obsidian vault.\n`,
  );
  return zip.generateAsync({ type: "blob" });
}
