"""Document summarization functionality."""
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from config import LLM_MODEL, DEVICE
from text_processor import split_into_sections

logger = logging.getLogger(__name__)

class Summarizer:
    """Generate summaries from documents."""
    
    def __init__(self):
        """Initialize the summarizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
            self.model.to(DEVICE)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Could not load {LLM_MODEL}: {e}. Using simple extractive summarization.")
            self.tokenizer = None
            self.model = None
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize a single text chunk."""
        if not text.strip():
            return "No content to summarize."
        
        # If model is not available, use extractive summarization
        if self.model is None or self.tokenizer is None:
            return self._extractive_summary(text, max_sentences=3)
        
        try:
            # Use the model for abstractive summarization
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(DEVICE)
            
            with __import__('torch').no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=50,
                    num_beams=4,
                    do_sample=False,
                    repetition_penalty=1.8
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary.strip()
        except Exception as e:
            logger.warning(f"Model summarization failed: {e}. Using extractive summary.")
            return self._extractive_summary(text, max_sentences=3)
    
    def _extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization by selecting key sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not sentences:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Select first, middle, and last sentences
        indices = [0]
        if len(sentences) > 2:
            indices.append(len(sentences) // 2)
        if len(sentences) > 1:
            indices.append(-1)
        
        selected = [sentences[i] for i in indices[:max_sentences]]
        return " ".join(selected)
    
    def summarize_document(self, text: str) -> Dict[str, str]:
        """Summarize a document section by section."""
        sections = split_into_sections(text)
        
        summary_parts = []
        section_summaries = []
        
        for section in sections:
            section_summary = self.summarize_text(section["content"])
            section_summaries.append({
                "title": section["title"],
                "summary": section_summary,
                "content_length": len(section["content"])
            })
            summary_parts.append(f"**{section['title']}**\n{section_summary}\n")
        
        # Generate overall summary
        overall_summary = self.summarize_text(text, max_length=200)
        
        return {
            "overall": overall_summary,
            "sections": section_summaries,
            "full_text": "\n\n".join(summary_parts)
        }

