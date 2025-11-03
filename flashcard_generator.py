"""Generate flashcards from document content."""
import logging
import re
from typing import List, Dict, Tuple
import json

logger = logging.getLogger(__name__)

class FlashcardGenerator:
    """Generate flashcards (Q&A pairs) from documents."""
    
    def __init__(self):
        """Initialize the flashcard generator."""
        pass
    
    def generate_from_text(self, text: str, num_flashcards: int = 10) -> List[Dict[str, str]]:
        """
        Generate flashcards from text.
        
        Args:
            text: Input text
            num_flashcards: Target number of flashcards
            
        Returns:
            List of flashcards with 'question' and 'answer' keys
        """
        flashcards = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Strategy 1: Fact-based flashcards (definitions, key terms)
        facts = self._extract_facts(sentences)
        flashcards.extend(facts[:num_flashcards // 2])
        
        # Strategy 2: Question-based flashcards (convert statements to Q&A)
        qa_pairs = self._create_qa_pairs(sentences)
        flashcards.extend(qa_pairs[:num_flashcards - len(flashcards)])
        
        # Ensure we have enough flashcards
        if len(flashcards) < num_flashcards:
            flashcards.extend(self._generate_simple_flashcards(sentences, num_flashcards - len(flashcards)))
        
        return flashcards[:num_flashcards]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split by sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _extract_facts(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Extract factual statements as flashcards."""
        flashcards = []
        
        # Pattern matching for definitions and facts
        definition_patterns = [
            r'([A-Z][^.!?]*(?:is|are|was|were|refers to|means|defines?)\s+[^.!?]+)',
            r'([A-Z][^.!?]*(?:known as|called|termed)\s+[^.!?]+)',
        ]
        
        for sentence in sentences:
            for pattern in definition_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    fact = match.group(1).strip()
                    if len(fact) > 20 and len(fact) < 200:
                        # Extract key term and definition
                        parts = re.split(r'\s+(?:is|are|was|were|refers to|means|defines?|known as|called|termed)\s+', fact, maxsplit=1, flags=re.IGNORECASE)
                        if len(parts) == 2:
                            question = f"What is {parts[0].strip()}?"
                            answer = parts[1].strip()
                            flashcards.append({"question": question, "answer": answer})
        
        return flashcards
    
    def _create_qa_pairs(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Create Q&A pairs from sentences."""
        flashcards = []
        
        for sentence in sentences:
            # Skip very short or very long sentences
            if len(sentence) < 30 or len(sentence) > 300:
                continue
            
            # Convert statements to questions
            # Pattern: "X does Y" -> "What does X do?" or "How does X work?"
            if re.search(r'\b(?:enables|allows|provides|offers|supports)\b', sentence, re.IGNORECASE):
                subject = re.search(r'^([^.!?]+?)\s+(?:enables|allows|provides|offers|supports)', sentence, re.IGNORECASE)
                if subject:
                    question = f"What does {subject.group(1).strip()} do?"
                    answer = sentence
                    flashcards.append({"question": question, "answer": answer})
            
            # Pattern: "X is important because Y" -> "Why is X important?"
            if re.search(r'\bis important because\b', sentence, re.IGNORECASE):
                subject = re.search(r'^([^.!?]+?)\s+is important because', sentence, re.IGNORECASE)
                if subject:
                    question = f"Why is {subject.group(1).strip()} important?"
                    answer = sentence
                    flashcards.append({"question": question, "answer": answer})
        
        return flashcards
    
    def _generate_simple_flashcards(self, sentences: List[str], num: int) -> List[Dict[str, str]]:
        """Generate simple flashcards by converting statements to questions."""
        flashcards = []
        
        for sentence in sentences[:num * 2]:  # Process more to filter
            if len(sentence) < 40:
                continue
            
            # Extract key information
            # Simple pattern: take first part as question, whole as answer
            words = sentence.split()
            if len(words) > 8:
                # Use first few words as question hint
                key_phrase = " ".join(words[:4])
                question = f"What information does this provide about {key_phrase}?"
                answer = sentence
                flashcards.append({"question": question, "answer": answer})
                
                if len(flashcards) >= num:
                    break
        
        return flashcards
    
    def export_to_json(self, flashcards: List[Dict[str, str]], filepath: str):
        """Export flashcards to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(flashcards, f, indent=2, ensure_ascii=False)
    
    def export_to_text(self, flashcards: List[Dict[str, str]], filepath: str):
        """Export flashcards to plain text file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, card in enumerate(flashcards, 1):
                f.write(f"Flashcard {i}:\n")
                f.write(f"Q: {card['question']}\n")
                f.write(f"A: {card['answer']}\n\n")

