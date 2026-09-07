"""Excel spreadsheet parser."""
from pathlib import Path
from typing import Dict
import pandas as pd
from .document_parser import DocumentParser


class ExcelParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        text = ""
        metadata = {}

        try:
            xl = pd.ExcelFile(file_path)
            metadata["sheet_names"] = xl.sheet_names
            metadata["sheet_count"] = len(xl.sheet_names)
            parts = []
            for sheet in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                parts.append(f"--- Sheet: {sheet} ---\n{df.to_string(index=False)}")
            text = "\n\n".join(parts)
            if not text.strip():
                text = "No readable data found in Excel file."
        except Exception as e:
            text = f"Error extracting text from Excel: {e}"

        return {"text": text.strip(), "metadata": metadata}
