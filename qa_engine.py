"""Question answering engine."""
import logging
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import LLM_MODEL, DEVICE, TOP_K_RESULTS
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class QAEngine:
    """Question answering engine with conversational context."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize the QA engine."""
        self.vector_store = vector_store
        self.conversation_history: Dict[str, List[Dict]] = {}  # doc_id -> history
        
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
    
    def ask(self, doc_id: str, question: str, use_history: bool = True) -> Dict[str, any]:
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
        
        return {
            "answer": answer,
            "sources": [r["text"][:200] for r in results[:3]]
        }
    
    def _build_prompt(self, question: str, context: str, history: str = "") -> str:
        """Build the prompt for answer generation."""
        prompt = """You are an AI assistant that answers questions strictly based on the provided document context.
        
If an exact answer exists in the context, return it verbatim.
If the answer is implied or partially available, explain it using logical reasoning.
If no relevant information exists, say so clearly.

Guidelines:
- Answer concisely but completely
- Use information only from the provided context
- If the context doesn't contain the answer, say "The document does not contain information about this."
"""
        
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

