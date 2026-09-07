import JSZip from "jszip";

// Credential-free Notion-compatible export (spec §3.8, §4.6).
// Produces a .zip containing:
//   - one Markdown file per page (Notion imports .md directly)
//   - a CSV "database.csv" whose rows match Notion's native CSV import schema
//     (Name, plus the provided dbRows columns). No OAuth token / API key required.
//
// Contracts relied on:
//  - jszip: generateAsync({type:"blob"})
//  - Notion CSV import expects a header row with a "Name" column.
export interface NotionPage {
  title: string;
  markdown: string;
}

// dbRows: array of flat records; all rows should share the same keys.
// The union of keys becomes the CSV columns (prefixed by "Name").
export async function exportNotionBundle(
  pages: NotionPage[],
  dbRows: Record<string, string>[],
  bundleName = "askdocs-notion-bundle",
): Promise<Blob> {
  const zip = new JSZip();

  pages.forEach((page, i) => {
    const name = (page.title || `page-${i + 1}`)
      .replace(/[^\w\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .slice(0, 80) || `page-${i + 1}`;
    zip.file(`${bundleName}/pages/${name}.md`, `# ${page.title}\n\n${page.markdown}\n`);
  });

  const headers = ["Name", ...collectColumns(dbRows)];
  const lines: string[] = [csvCell(headers.join(","))];
  for (const row of dbRows) {
    const cells = [row["Name"] ?? "", ...headers.slice(1).map((h) => row[h] ?? "")];
    lines.push(cells.map(csvCell).join(","));
  }
  zip.file(`${bundleName}/database.csv`, lines.join("\n"));
  zip.file(
    `${bundleName}/README.md`,
    `# Notion import bundle\n\nImport pages/ as Markdown and database.csv via Notion's native importer. No credentials required.\n`,
  );

  return zip.generateAsync({ type: "blob" });
}

function collectColumns(rows: Record<string, string>[]): string[] {
  const set = new Set<string>();
  for (const r of rows) for (const k of Object.keys(r)) if (k !== "Name") set.add(k);
  return [...set];
}

// Minimal CSV field escaper (wrap in quotes if it contains comma/quote/newline).
function csvCell(value: string): string {
  const v = String(value ?? "");
  if (/[",\n]/.test(v)) return `"${v.replace(/"/g, '""')}"`;
  return v;
}
