"""Structure-aware local DOCX and PPTX extraction.

Preserves Word paragraphs and tables in document order, and slide numbers,
text frames, tables and grouped shapes in presentations. Uses only the
existing python-docx and python-pptx dependencies, no hosted document API.
"""
from __future__ import annotations

from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE


def _cell_text(value: str) -> str:
    """Keep cells and multiline content readable in plain text chunks."""
    return " ".join((value or "").split()).replace("|", "/")


def _table_text(rows) -> str:
    lines = ["[Table]"]
    for number, row in enumerate(rows, start=1):
        # Column labels avoid inventing a header in tables without one.
        values = [
            "C" + str(index) + ": " + (_cell_text(cell.text) or "(empty)")
            for index, cell in enumerate(row.cells, start=1)
        ]
        lines.append("Row " + str(number) + " - " + " | ".join(values))
    return "\n".join(lines)


def extract_docx_text(path) -> str:
    """Extract text from a .docx in body order, including table rows."""
    doc = DocxDocument(path)
    output: list[str] = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = DocxParagraph(child, doc)
            content = paragraph.text.strip()
            if content:
                output.append(content)
        elif child.tag.endswith("}tbl"):
            table = DocxTable(child, doc)
            if table.rows:
                output.append(_table_text(table.rows))
    return "\n\n".join(output)


def _slide_shapes(shapes):
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _slide_shapes(shape.shapes)
        else:
            yield shape


def extract_pptx_text(path) -> str:
    """Extract slide content with 1-based PAGE markers for citations."""
    presentation = Presentation(path)
    slides: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        content: list[str] = []
        for shape in _slide_shapes(slide.shapes):
            if shape.has_text_frame:
                content.extend(
                    paragraph.text.strip()
                    for paragraph in shape.text_frame.paragraphs
                    if paragraph.text.strip()
                )
            if shape.has_table:
                content.append(_table_text(shape.table.rows))
        # Do not silently shift page numbers when an empty slide is skipped.
        slides.append("<<<PAGE " + str(index) + ">>>\n" + "\n".join(content))
    return "\n\n".join(slides)
