"""Generate quizzes from document content."""
import logging
import re
from typing import List, Dict
import json
import random

from flashcard_generator import FlashcardGenerator

logger = logging.getLogger(__name__)

class QuizGenerator:
    """Generate quizzes from documents."""
    
    def __init__(self):
        """Initialize the quiz generator."""
        self.flashcard_gen = FlashcardGenerator()
    
    def generate_quiz(self, text: str, num_questions: int = 10) -> Dict[str, List[Dict]]:
        """
        Generate a quiz with multiple question types.
        
        Returns:
            Dictionary with 'multiple_choice', 'true_false', and 'short_answer' keys
        """
        flashcards = self.flashcard_gen.generate_from_text(text, num_questions * 2)
        
        quiz = {
            "multiple_choice": [],
            "true_false": [],
            "short_answer": []
        }
        
        # Convert flashcards to quiz questions
        for card in flashcards[:num_questions]:
            # Short answer questions
            quiz["short_answer"].append({
                "question": card["question"],
                "answer": card["answer"],
                "points": 2
            })
            
            # Multiple choice (if we have enough content)
            if len(card["answer"]) > 30:
                mc = self._create_multiple_choice(card, flashcards)
                if mc:
                    quiz["multiple_choice"].append(mc)
            
            # True/False (simple statements)
            if len(card["answer"]) < 100:
                tf = self._create_true_false(card)
                if tf:
                    quiz["true_false"].append(tf)
        
        # Limit each type
        quiz["multiple_choice"] = quiz["multiple_choice"][:num_questions]
        quiz["true_false"] = quiz["true_false"][:num_questions]
        quiz["short_answer"] = quiz["short_answer"][:num_questions]
        
        return quiz
    
    def _create_multiple_choice(self, card: Dict[str, str], all_cards: List[Dict]) -> Dict:
        """Create a multiple choice question from a flashcard."""
        # Generate distractors from other cards
        distractors = []
        for other_card in all_cards:
            if other_card != card and len(other_card["answer"]) > 20:
                # Extract key phrase from answer as distractor
                words = other_card["answer"].split()[:8]
                distractor = " ".join(words)
                if distractor not in distractors and distractor != card["answer"][:50]:
                    distractors.append(distractor)
                    if len(distractors) >= 3:
                        break
        
        # Ensure we have at least 3 options
        while len(distractors) < 3:
            distractors.append("None of the above")
        
        options = [card["answer"]] + distractors[:3]
        random.shuffle(options)
        
        correct_index = options.index(card["answer"])
        
        return {
            "question": card["question"],
            "options": options,
            "correct": correct_index,
            "points": 2
        }
    
    def _create_true_false(self, card: Dict[str, str]) -> Dict:
        """Create a true/false question from a flashcard."""
        # Convert answer to a statement
        statement = card["answer"]
        
        # For True/False, we'll make a statement that's either true or false
        # Extract a fact and create statement
        words = statement.split()
        if len(words) < 5:
            return None
        
        # Create a statement based on the answer
        if statement.lower().startswith(('the', 'a', 'an', 'this', 'that')):
            tf_statement = statement
        else:
            tf_statement = statement
        
        # Randomly decide if it should be true or false (but mostly true for now)
        is_true = random.random() > 0.3
        
        return {
            "question": f"True or False: {tf_statement}",
            "correct": is_true,
            "points": 1,
            "explanation": card["answer"] if is_true else "This statement is false."
        }
    
    def export_to_json(self, quiz: Dict, filepath: str):
        """Export quiz to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(quiz, f, indent=2, ensure_ascii=False)
    
    def calculate_score(self, quiz: Dict, answers: Dict) -> Dict:
        """Calculate quiz score based on answers."""
        total_score = 0
        max_score = 0
        results = {}
        
        # Multiple choice
        for i, q in enumerate(quiz["multiple_choice"]):
            max_score += q.get("points", 2)
            user_answer = answers.get(f"mc_{i}", -1)
            if user_answer == q["correct"]:
                total_score += q.get("points", 2)
                results[f"mc_{i}"] = {"correct": True, "points": q.get("points", 2)}
            else:
                results[f"mc_{i}"] = {"correct": False, "points": 0}
        
        # True/False
        for i, q in enumerate(quiz["true_false"]):
            max_score += q.get("points", 1)
            user_answer = answers.get(f"tf_{i}", None)
            if user_answer == q["correct"]:
                total_score += q.get("points", 1)
                results[f"tf_{i}"] = {"correct": True, "points": q.get("points", 1)}
            else:
                results[f"tf_{i}"] = {"correct": False, "points": 0}
        
        # Short answer (basic checking)
        for i, q in enumerate(quiz["short_answer"]):
            max_score += q.get("points", 2)
            user_answer = answers.get(f"sa_{i}", "").lower().strip()
            correct_answer = q["answer"].lower().strip()
            
            # Simple keyword matching
            correct_keywords = set(correct_answer.split()[:5])
            user_keywords = set(user_answer.split())
            
            if len(correct_keywords.intersection(user_keywords)) >= 2:
                total_score += q.get("points", 2)
                results[f"sa_{i}"] = {"correct": True, "points": q.get("points", 2)}
            else:
                results[f"sa_{i}"] = {"correct": False, "points": 0, "expected": q["answer"]}
        
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        return {
            "score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1),
            "results": results
        }

