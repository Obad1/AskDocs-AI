"""Main application entry point."""
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from vector_store import VectorStore
from document_manager import DocumentManager
from summarizer import Summarizer
from flashcard_generator import FlashcardGenerator
from quiz_generator import QuizGenerator
from qa_engine import QAEngine
from ui import create_ui

def main():
    """Initialize and run the application."""
    logger.info("Initializing AskDocs AI...")
    
    # Initialize components
    vector_store = VectorStore()
    doc_manager = DocumentManager(vector_store)
    summarizer = Summarizer()
    flashcard_gen = FlashcardGenerator()
    quiz_gen = QuizGenerator()
    qa_engine = QAEngine(vector_store)
    
    logger.info("Creating UI...")
    
    # Create and launch UI
    ui = create_ui(
        doc_manager=doc_manager,
        summarizer=summarizer,
        flashcard_gen=flashcard_gen,
        quiz_gen=quiz_gen,
        qa_engine=qa_engine
    )
    
    logger.info("Launching application...")
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()

