"""Synthetic-only checks for page-aware chunking and Office structure."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.parsing.structured_chunks import chunk_marked_text
from app.services.parsing.docx_pptx_engine import (
    extract_docx_text,
    extract_pptx_text,
)
from docx import Document
from pptx import Presentation
from pptx.util import Inches


class StructuredChunkTests(unittest.TestCase):
    def test_exact_overlap_and_page_boundaries(self):
        doc = (
            "<<<PAGE 1>>>\n" +
            "one two three four five six seven eight nine\n" +
            "<<<PAGE 2>>>\nalpha beta gamma"
        )
        chunks = chunk_marked_text(doc, target=5, overlap=2)
        self.assertEqual([c["page"] for c in chunks], [1, 1, 1, 2])
        self.assertEqual(chunks[0]["text"], "one two three four five")
        self.assertEqual(chunks[1]["text"], "four five six seven eight")
        self.assertEqual(chunks[2]["text"], "seven eight nine")
        self.assertEqual(chunks[3]["text"], "alpha beta gamma")
        self.assertTrue(all(len(c["text"].split()) <= 5 for c in chunks))

    def test_long_paragraph_never_overflows(self):
        doc = " ".join("w" + str(n) for n in range(1201))
        chunks = chunk_marked_text(doc, target=100, overlap=10)
        self.assertEqual(chunks[0]["page"], None)
        self.assertTrue(all(len(c["text"].split()) <= 100 for c in chunks))
        self.assertEqual(chunks[0]["text"].split()[-10:], chunks[1]["text"].split()[:10])

    def test_empty_and_invalid_options(self):
        self.assertEqual(chunk_marked_text(" \n "), [])
        with self.assertRaises(ValueError):
            chunk_marked_text("text", target=0)
        with self.assertRaises(ValueError):
            chunk_marked_text("text", target=5, overlap=5)


class OfficeStructureTests(unittest.TestCase):
    def test_word_table_in_body_order(self):
        doc = Document()
        doc.add_paragraph("Before the schedule")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Invoice"
        table.cell(0, 1).text = "Amount"
        table.cell(1, 0).text = "INV-17"
        table.cell(1, 1).text = "2400"
        doc.add_paragraph("After the schedule")
        memory = io.BytesIO()
        doc.save(memory)
        memory.seek(0)
        text = extract_docx_text(memory)
        self.assertLess(text.index("Before"), text.index("INV-17"))
        self.assertLess(text.index("INV-17"), text.index("After"))
        self.assertIn("Row 2", text)
        self.assertIn("C2: 2400", text)

    def test_powerpoint_slide_table_and_page_identity(self):
        prs = Presentation()
        slide1 = prs.slides.add_slide(prs.slide_layouts[6])
        box = slide1.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
        box.text = "Alpha summary"
        table = slide1.shapes.add_table(1, 2, Inches(1), Inches(2), Inches(5), Inches(1)).table
        table.cell(0, 0).text = "Status"
        table.cell(0, 1).text = "Approved"
        slide2 = prs.slides.add_slide(prs.slide_layouts[6])
        box2 = slide2.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
        box2.text = "Beta conclusion"
        memory = io.BytesIO()
        prs.save(memory)
        memory.seek(0)
        text = extract_pptx_text(memory)
        self.assertIn("<<<PAGE 1>>>", text)
        self.assertIn("<<<PAGE 2>>>", text)
        self.assertIn("C2: Approved", text)
        self.assertLess(text.index("Alpha summary"), text.index("Beta conclusion"))


if __name__ == "__main__":
    unittest.main()
