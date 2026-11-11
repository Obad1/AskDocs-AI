// Regression test of actual BM25 ranking and safe re-indexing.
const assert = require("node:assert/strict");
const { BM25Index } = require("/tmp/askdocs-bm25/bm25_index.js");
const index = new BM25Index();
index.add("invoice", "The invoice INV-4088 has the customer balance due");
index.add("shipping", "The shipping bill of lading contains the cargo description");
index.add("invoice2", "This invoice also lists the payment due date");
assert.equal(index.search("INV 4088", 4)[0].id, "invoice");
assert.equal(index.search("completely-absent-token", 10).length, 0);
assert.ok(index.search("invoice due", 5).some((row) => row.id === "invoice2"));
index.add("invoice", "replacement text without original invoice number");
assert.equal(index.search("4088", 5).length, 0);
index.remove("shipping");
assert.ok(index.search("cargo", 5).every((row) => row.id !== "shipping"));
assert.deepEqual(index.search("", 5), []);
console.log("BM25 search, reranking, overwrite and delete tests passed.");
