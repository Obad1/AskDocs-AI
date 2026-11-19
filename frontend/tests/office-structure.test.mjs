import assert from "node:assert/strict";
import { readFile, writeFile, unlink } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { transform } from "esbuild";
import JSZip from "jszip";

const compiled = new URL("./.office-parse.test-tmp.mjs", import.meta.url);
const source = await readFile(new URL("../src/lib/parsing/officeParse.ts", import.meta.url), "utf8");
const transpiled = await transform(source, { loader: "ts", format: "esm", target: "es2022" });

try {
  await writeFile(compiled, transpiled.code, "utf8");
  const { parseDocx, parsePptx } = await import(compiled.href);

  const docx = new JSZip();
  docx.file("word/document.xml", [
    "<w:document><w:body>",
    "<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>there</w:t></w:r></w:p>",
    "<w:tbl><w:tr>",
    "<w:tc><w:p><w:r><w:t>Invoice</w:t></w:r></w:p></w:tc>",
    "<w:tc><w:p><w:r><w:t>Amount</w:t></w:r></w:p></w:tc>",
    "</w:tr><w:tr>",
    "<w:tc><w:p><w:r><w:t>INV-17</w:t></w:r></w:p></w:tc>",
    "<w:tc><w:p><w:r><w:t>2400</w:t></w:r></w:p></w:tc>",
    "</w:tr></w:tbl>",
    "<w:p><w:r><w:t>After table</w:t></w:r></w:p>",
    "</w:body></w:document>",
  ].join(""));
  const d = await parseDocx(await docx.generateAsync({ type: "uint8array" }));
  assert.equal(d.broken, false);
  assert.ok(d.text.includes("Hello there"), "Adjacent runs must not introduce spaces");
  assert.ok(d.text.includes("Row 2 - C1: INV-17 | C2: 2400"), "Table cells must remain associated");
  assert.ok(d.text.indexOf("Hello") < d.text.indexOf("INV-17"));
  assert.ok(d.text.indexOf("INV-17") < d.text.indexOf("After table"));

  const pptx = new JSZip();
  pptx.file("ppt/slides/slide2.xml", "<p:sld><p:sp><a:p><a:r><a:t>Second slide</a:t></a:r></a:p></p:sp></p:sld>");
  pptx.file("ppt/slides/slide1.xml", [
    "<p:sld><p:sp><a:p><a:r><a:t>First slide</a:t></a:r></a:p></p:sp>",
    "<p:graphicFrame><a:tbl><a:tr>",
    "<a:tc><a:p><a:r><a:t>Claim</a:t></a:r></a:p></a:tc>",
    "<a:tc><a:p><a:r><a:t>Verified</a:t></a:r></a:p></a:tc>",
    "</a:tr></a:tbl></p:graphicFrame></p:sld>",
  ].join(""));
  const p = await parsePptx(await pptx.generateAsync({ type: "uint8array" }));
  assert.equal(p.broken, false);
  assert.ok(p.text.includes("<<<PAGE 1>>>"));
  assert.ok(p.text.includes("<<<PAGE 2>>>"));
  assert.ok(p.text.includes("Row 1 - C1: Claim | C2: Verified"));
  assert.ok(p.text.indexOf("First slide") < p.text.indexOf("Second slide"));
  assert.deepEqual([...new Set(p.paragraphs.map((x) => x.page))], [1, 2]);
  console.log("Office browser parser: document order, tables, slide citations verified.");
} finally {
  await unlink(compiled).catch(() => {});
}
