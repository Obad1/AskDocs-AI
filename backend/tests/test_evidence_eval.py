"""Offline, deterministic negative/positive tests for AskDocs evidence quality."""
import unittest
from backend.app.services.rag.evidence_eval import (
    retrieval_metrics, citation_checks, evaluate_fixture
)


class EvidenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunkId": "docA:1", "text": "The contract starts on 1 November 2026."},
            {"chunkId": "docA:2", "text": "The renewal notice period is 30 days."},
            {"chunkId": "docB:3", "text": "This is unrelated information."},
        ]

    def test_precision_recall_are_computed_from_labeled_evidence(self):
        result = retrieval_metrics(self.chunks, ["docA:1", "docA:2"], 3)
        self.assertEqual(result["relevant_found"], 2)
        self.assertEqual(result["precision_at_k"], 0.6667)
        self.assertEqual(result["recall_at_k"], 1.0)

    def test_missing_label_or_duplicate_source_fails_closed(self):
        with self.assertRaises(ValueError):
            retrieval_metrics(self.chunks, [])
        with self.assertRaises(ValueError):
            retrieval_metrics([self.chunks[0], self.chunks[0]], ["docA:1"])

    def test_invalid_k_fails_closed(self):
        for k in (0, -1, 101, 0.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                retrieval_metrics(self.chunks, ["docA:1"], k)

    def test_supported_quotation_requires_exact_source_excerpt(self):
        result = citation_checks(self.chunks, [
            {"chunkId": "docA:2", "quote": "renewal notice   period is 30 days"}
        ])
        self.assertTrue(result["valid"])

    def test_nonexistent_source_and_hallucinated_quote_fail(self):
        a = citation_checks(self.chunks, [{"chunkId": "made-up", "quote": "x"}])
        b = citation_checks(self.chunks, [{"chunkId": "docA:1", "quote": "unlimited guarantee"}])
        self.assertEqual(a["results"][0]["reason"], "missing_source")
        self.assertEqual(b["results"][0]["reason"], "quote_not_in_source")
        self.assertFalse(a["valid"])
        self.assertFalse(b["valid"])

    def test_missing_quote_and_empty_citation_list_do_not_claim_grounding(self):
        self.assertFalse(citation_checks(self.chunks, [])["valid"])
        self.assertFalse(citation_checks(self.chunks, [{"chunkId": "docA:1"}])["valid"])

    def test_synthetic_fixture_passes_and_mutation_fails(self):
        fixture = {
            "name": "synthetic notice period", "retrieved": self.chunks,
            "relevant_chunk_ids": ["docA:2"], "k": 2,
            "citations": [{"chunkId": "docA:2", "quote": "notice period is 30 days"}],
            "min_recall": 1,
        }
        # Relevant chunk appears in the first two, so recall@2=1.
        self.assertTrue(evaluate_fixture(fixture)["pass"])
        fixture["citations"][0]["quote"] = "notice period is five minutes"
        self.assertFalse(evaluate_fixture(fixture)["pass"])


if __name__ == "__main__":
    unittest.main()
