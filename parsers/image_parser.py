"""Image parser using OCR."""
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser
from loguru import logger

try:
    import pytesseract
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class ImageParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        if not OCR_AVAILABLE:
            return {"text": "OCR not available. Install: pip install pytesseract opencv-python", "metadata": {"method": "no_ocr"}}
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return {"text": "Could not read image file.", "metadata": {"method": "error"}}
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            return {"text": text.strip(), "metadata": {"method": "image_ocr", "chars": len(text.strip())}}
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return {"text": f"Error OCRing image: {e}", "metadata": {"method": "error"}}
