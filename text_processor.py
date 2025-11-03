"""Text processing utilities for chunking and splitting."""
import re
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # First, try to split by paragraphs (double newlines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        # If paragraph alone is larger than chunk_size, split it by sentences
        if para_words > chunk_size:
            # Flush current chunk if any
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sent_words = sentence.split()
                if current_length + len(sent_words) > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Keep last overlap words for continuity
                        current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
                        current_length = sum(len(w.split()) for w in current_chunk)
                
                current_chunk.append(sentence)
                current_length += len(sent_words)
        else:
            # Check if adding this paragraph would exceed chunk size
            if current_length + para_words > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep last overlap words
                    current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else []
                    current_length = sum(len(w.split()) for w in current_chunk)
            
            current_chunk.append(para)
            current_length += para_words
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]

def split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Split text into sections based on headers or paragraphs.
    
    Returns:
        List of dictionaries with 'title' and 'content' keys
    """
    sections = []
    
    # Try to detect markdown headers
    lines = text.split('\n')
    current_section = {"title": "Introduction", "content": ""}
    
    for line in lines:
        # Check for markdown headers
        if line.strip().startswith('#'):
            # Save previous section
            if current_section["content"].strip():
                sections.append(current_section)
            
            # Start new section
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            current_section = {"title": title, "content": ""}
        else:
            current_section["content"] += line + "\n"
    
    # Add last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    # If no headers found, split by paragraphs
    if not sections:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 for performance
            sections.append({
                "title": f"Section {i+1}",
                "content": para
            })
    
    return sections

