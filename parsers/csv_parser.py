"""CSV file parser."""
from pathlib import Path
from typing import Dict
import pandas as pd
from .document_parser import DocumentParser

class CSVParser(DocumentParser):
    """Parser for CSV files."""
    
    def parse(self, file_path: Path) -> Dict[str, str]:
        """Extract text from CSV file."""
        text = ""
        metadata = {}
        
        try:
            df = pd.read_csv(file_path)
            metadata["rows"] = len(df)
            metadata["columns"] = len(df.columns)
            metadata["column_names"] = list(df.columns)
            
            # Convert to readable text format
            text = df.to_string(index=False)
            
            if not text.strip():
                text = "❌ No readable data found in CSV."
        except Exception as e:
            text = f"❌ Error extracting text from CSV: {str(e)}"
        
        return {"text": text.strip(), "metadata": metadata}

