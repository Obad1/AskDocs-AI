"""Question answering engine."""
import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import LLM_MODEL, DEVICE, TOP_K_RESULTS
import json
from pathlib import Path
from vectorstores.base import VectorStoreBase
from config import SESSIONS_DIR, OFFLINE_MODE

logger = logging.getLogger(__name__)

class QAEngine:
    """Question answering engine with conversational context."""
    
    def __init__(self, vector_store: VectorStoreBase):
        """Initialize the QA engine."""
        self.vector_store = vector_store
        self.conversation_history: Dict[str, List[Dict]] = {}  # doc_id -> history
        self.mode = "Answer"  # Answer, Debate, Socratic
        self.tone = "Formal"  # Formal, Friendly, Analytical
        
        # Load model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
            self.model.to(DEVICE)
            self.model.eval()
        except Exception as e:
            logger.error(f"Could not load model {LLM_MODEL}: {e}")
            self.tokenizer = None
            self.model = None
    
    def ask(self, doc_id: str, question: str, use_history: bool = True, show_snippets: bool = True) -> Dict[str, any]:
        """
        Ask a question about a document.
        
        Args:
            doc_id: Document ID
            question: User question
            use_history: Whether to use conversation history
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        # Retrieve relevant passages
        results = self.vector_store.search(doc_id, question, top_k=TOP_K_RESULTS)
        
        if not results:
            return {
                "answer": "I couldn't find relevant information in the document to answer this question.",
                "sources": []
            }
        
        # Build context from retrieved passages
        context = "\n".join([r["text"] for r in results[:3]])  # Use top 3
        
        # Build prompt with optional conversation history
        history_text = ""
        if use_history and doc_id in self.conversation_history:
            recent_history = self.conversation_history[doc_id][-3:]  # Last 3 exchanges
            history_text = "\n".join([
                f"Q: {h['question']}\nA: {h['answer']}"
                for h in recent_history
            ])
        
        prompt = self._build_prompt(question, context, history_text)
        
        # Generate answer
        if self.model and self.tokenizer:
            answer = self._generate_answer(prompt, question, context)
        else:
            # Fallback: return top passage
            answer = results[0]["text"][:500] if results else "No answer found."
        
        # Save to history
        if doc_id not in self.conversation_history:
            self.conversation_history[doc_id] = []
        
        self.conversation_history[doc_id].append({
            "question": question,
            "answer": answer
        })
        
        payload = {
            "answer": answer,
            "sources": [
                {
                    "snippet": r["text"][:300],
                    "chunk_index": r["metadata"].get("chunk_index"),
                    "distance": r.get("distance")
                } for r in results[:3]
            ] if show_snippets else []
        }

        # Persist session
        self._persist_history(doc_id)
        return payload
    
    def _build_prompt(self, question: str, context: str, history: str = "") -> str:
        """Build the prompt for answer generation."""
        base_prompt = """You are an AI assistant that answers questions strictly based on the provided document context.
        
If an exact answer exists in the context, return it verbatim.
If the answer is implied or partially available, explain it using logical reasoning.
If no relevant information exists, say so clearly.

Guidelines:
- Answer concisely but completely
- Use information only from the provided context
- If the context doesn't contain the answer, say "The document does not contain information about this."
"""
        mode_instructions = {
            "Answer": "Provide a direct, grounded answer.",
            "Debate": "Present pros and cons and a reasoned conclusion.",
            "Socratic": "Guide with questions and brief hints without revealing the full answer outright."
        }[self.mode]

        tone_instructions = {
            "Formal": "Use precise, neutral language.",
            "Friendly": "Be warm and encouraging, but concise.",
            "Analytical": "Be structured, with bullet points when appropriate."
        }[self.tone]

        prompt = base_prompt + f"\n\nMode: {self.mode} — {mode_instructions}\nTone: {self.tone} — {tone_instructions}\n"
        if history:
            prompt += f"\n\nPrevious conversation:\n{history}\n"
        
        prompt += f"""
Context from document:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str, question: str, context: str) -> str:
        """Generate answer using the language model."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(DEVICE)
            
            with __import__('torch').no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    min_length=20,
                    num_beams=4,
                    do_sample=False,
                    repetition_penalty=1.8
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            if len(answer) < 5:
                # Fallback to context snippet
                return context[:300] + "..." if len(context) > 300 else context
            
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return context[:300] + "..." if len(context) > 300 else context
    
    def clear_history(self, doc_id: Optional[str] = None):
        """Clear conversation history for a document or all documents."""
        if doc_id:
            if doc_id in self.conversation_history:
                del self.conversation_history[doc_id]
        else:
            self.conversation_history.clear()

    def _persist_history(self, doc_id: str):
        """Persist conversation history to disk."""
        try:
            path = Path(SESSIONS_DIR) / f"{doc_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history.get(doc_id, []), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist history for {doc_id}: {e}")

    def export_history(self, doc_id: str) -> str:
        """Return session history JSON as string."""
        return json.dumps(self.conversation_history.get(doc_id, []), indent=2, ensure_ascii=False)

    def import_history(self, doc_id: str, content: str):
        """Import session history from JSON string and persist it."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                self.conversation_history[doc_id] = data
                self._persist_history(doc_id)
        except Exception as e:
            logger.warning(f"Failed to import history for {doc_id}: {e}")

