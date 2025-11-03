"""Gradio UI for AskDocs AI."""
import gradio as gr
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

from config import UI_TITLE, UI_DEBUG
from document_manager import DocumentManager
from summarizer import Summarizer
from flashcard_generator import FlashcardGenerator
from quiz_generator import QuizGenerator
from qa_engine import QAEngine

logger = logging.getLogger(__name__)

def create_ui(
    doc_manager: DocumentManager,
    summarizer: Summarizer,
    flashcard_gen: FlashcardGenerator,
    quiz_gen: QuizGenerator,
    qa_engine: QAEngine
) -> gr.Blocks:
    """Create the Gradio UI."""
    
    # State to track current document
    current_doc_id = gr.State(value=None)
    current_summary = gr.State(value=None)
    current_flashcards = gr.State(value=None)
    current_quiz = gr.State(value=None)
    
    def upload_file(file):
        """Handle file upload."""
        if file is None:
            return "Please select a file to upload.", None, None
        
        try:
            file_path = Path(file.name)
            result = doc_manager.add_document(file_path)
            
            if "error" in result:
                return result["error"], None, None
            
            doc_list = doc_manager.list_documents()
            doc_options = [f"{d['name']} (ID: {d['id']})" for d in doc_list]
            
            return f"✅ Document '{result['name']}' uploaded successfully!", result["id"], gr.update(choices=doc_options, value=doc_options[-1] if doc_options else None)
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return f"❌ Error uploading file: {str(e)}", None, None
    
    def select_document(selection, state):
        """Handle document selection."""
        if not selection:
            return None, "", "", "", "", [], {}, None, [], [], []
        
        # Extract doc_id from selection
        doc_id = selection.split("(ID: ")[1].split(")")[0] if "(ID: " in selection else None
        if not doc_id:
            return state, "", "", "", "", [], {}, None, [], [], []
        
        doc = doc_manager.get_document(doc_id)
        if not doc:
            return state, "", "", "", "", [], {}, None, [], [], []
        
        # Get text
        text = doc.get("text", "")
        
        # Generate summary
        summary_result = summarizer.summarize_document(text)
        summary_text = summary_result.get("full_text", "")
        
        # Generate flashcards
        flashcards = flashcard_gen.generate_from_text(text, num_flashcards=15)
        
        # Generate quiz
        quiz = quiz_gen.generate_quiz(text, num_questions=10)
        
        # Prepare quiz display updates
        mc_update = quiz.get("multiple_choice", [])
        tf_update = quiz.get("true_false", [])
        sa_update = quiz.get("short_answer", [])
        
        return (
            doc_id, 
            text[:1000] + "..." if len(text) > 1000 else text, 
            summary_text, 
            summary_result.get("overall", ""), 
            json.dumps(summary_result.get("sections", []), indent=2), 
            flashcards, 
            quiz, 
            doc,
            mc_update,
            tf_update,
            sa_update
        )
    
    def ask_question(question, doc_id, history):
        """Handle question asking."""
        if not question or not question.strip():
            return history, ""
        
        if not doc_id:
            history.append((question, "Please upload and select a document first."))
            return history, ""
        
        try:
            result = qa_engine.ask(doc_id, question, use_history=True)
            answer = result["answer"]
            
            # Add sources if available
            if result.get("sources"):
                sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {s[:100]}..." for s in result["sources"][:3]])
                answer += sources_text
            
            history.append((question, answer))
            return history, ""
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            history.append((question, f"Error: {str(e)}"))
            return history, ""
    
    def export_flashcards(flashcards, format_type):
        """Export flashcards."""
        if not flashcards:
            return "No flashcards to export."
        
        try:
            if format_type == "JSON":
                output = json.dumps(flashcards, indent=2, ensure_ascii=False)
            else:
                # Text format
                lines = []
                for i, card in enumerate(flashcards, 1):
                    lines.append(f"Flashcard {i}:")
                    lines.append(f"Q: {card['question']}")
                    lines.append(f"A: {card['answer']}")
                    lines.append("")
                output = "\n".join(lines)
            
            return output
        except Exception as e:
            return f"Error exporting: {str(e)}"
    
    def submit_quiz_answers(quiz, answers_json):
        """Submit quiz answers and calculate score."""
        if not quiz:
            return "Please select a document first to generate a quiz."
        
        try:
            answers = json.loads(answers_json) if answers_json else {}
            score_result = quiz_gen.calculate_score(quiz, answers)
            
            result_text = f"""
# Quiz Results

**Score:** {score_result['score']} / {score_result['max_score']} ({score_result['percentage']}%)

## Breakdown:
"""
            # Add breakdown by question type
            mc_correct = sum(1 for k, v in score_result['results'].items() if k.startswith('mc_') and v.get('correct'))
            tf_correct = sum(1 for k, v in score_result['results'].items() if k.startswith('tf_') and v.get('correct'))
            sa_correct = sum(1 for k, v in score_result['results'].items() if k.startswith('sa_') and v.get('correct'))
            
            result_text += f"- Multiple Choice: {mc_correct} correct\n"
            result_text += f"- True/False: {tf_correct} correct\n"
            result_text += f"- Short Answer: {sa_correct} correct\n"
            
            return result_text
        except Exception as e:
            return f"Error calculating score: {str(e)}"
    
    # Build UI
    with gr.Blocks(title=UI_TITLE, theme=gr.themes.Soft()) as ui:
        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown("Upload documents and interact with them through Q&A, summaries, flashcards, and quizzes.")
        
        with gr.Tabs():
            # Documents Tab
            with gr.Tab("📄 Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Document")
                        file_input = gr.File(label="Upload File", file_types=[".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".html"])
                        upload_btn = gr.Button("Upload", variant="primary")
                        upload_status = gr.Textbox(label="Status", interactive=False)
                        
                        gr.Markdown("### Select Document")
                        doc_dropdown = gr.Dropdown(
                            label="Documents",
                            choices=[],
                            interactive=True
                        )
                        select_btn = gr.Button("Select Document", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Document Preview")
                        doc_preview = gr.Textbox(label="Preview", lines=10, interactive=False)
                        doc_info = gr.JSON(label="Document Info", visible=False)
            
            # Summary Tab
            with gr.Tab("📝 Summary"):
                with gr.Column():
                    gr.Markdown("### Document Summary")
                    summary_text = gr.Markdown(label="Summary")
                    overall_summary = gr.Textbox(label="Overall Summary", lines=3, interactive=False)
                    sections_json = gr.JSON(label="Section Details", visible=False)
            
            # Q&A Tab
            with gr.Tab("💬 Q&A"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Ask Questions")
                        chatbot = gr.Chatbot(label="Conversation", height=400)
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about the document...",
                            lines=2
                        )
                        ask_btn = gr.Button("Ask", variant="primary")
                        clear_btn = gr.Button("Clear History")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Tips")
                        gr.Markdown("""
- Ask specific questions about the document
- The system uses conversation history for context
- You can ask follow-up questions
                        """)
            
            # Flashcards Tab
            with gr.Tab("🎴 Flashcards"):
                with gr.Column():
                    gr.Markdown("### Generated Flashcards")
                    flashcards_json = gr.JSON(label="Flashcards", visible=False)
                    
                    with gr.Row():
                        with gr.Column():
                            export_format = gr.Radio(
                                choices=["JSON", "Text"],
                                value="JSON",
                                label="Export Format"
                            )
                            export_btn = gr.Button("Export Flashcards")
                            export_output = gr.Textbox(label="Exported Content", lines=10)
            
            # Quiz Tab
            with gr.Tab("📝 Quiz"):
                with gr.Column():
                    gr.Markdown("### Generated Quiz")
                    
                    with gr.Tabs():
                        with gr.Tab("Multiple Choice"):
                            mc_questions = gr.JSON(label="Multiple Choice Questions", visible=True)
                        
                        with gr.Tab("True/False"):
                            tf_questions = gr.JSON(label="True/False Questions", visible=True)
                        
                        with gr.Tab("Short Answer"):
                            sa_questions = gr.JSON(label="Short Answer Questions", visible=True)
                    
                    gr.Markdown("### Submit Answers")
                    gr.Markdown("For now, you can view the quiz questions. Full answer submission will be available in future versions.")
                    quiz_results = gr.Markdown(label="Results")
        
        # Event handlers
        upload_btn.click(
            fn=upload_file,
            inputs=[file_input],
            outputs=[upload_status, current_doc_id, doc_dropdown]
        )
        
        select_btn.click(
            fn=select_document,
            inputs=[doc_dropdown, current_doc_id],
            outputs=[
                current_doc_id, doc_preview, summary_text, overall_summary,
                sections_json, flashcards_json, current_quiz, doc_info,
                mc_questions, tf_questions, sa_questions
            ]
        )
        
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input, current_doc_id, chatbot],
            outputs=[chatbot, question_input]
        )
        
        question_input.submit(
            fn=ask_question,
            inputs=[question_input, current_doc_id, chatbot],
            outputs=[chatbot, question_input]
        )
        
        clear_btn.click(
            fn=lambda: ([], current_doc_id.value) if current_doc_id.value else ([], None),
            inputs=[],
            outputs=[chatbot, current_doc_id]
        )
        
        def update_quiz_display(quiz):
            """Update quiz display when quiz is selected."""
            if not quiz:
                return gr.update(), gr.update(), gr.update()
            
            return (
                gr.update(value=quiz.get("multiple_choice", [])),
                gr.update(value=quiz.get("true_false", [])),
                gr.update(value=quiz.get("short_answer", []))
            )
        
        current_quiz.change(
            fn=update_quiz_display,
            inputs=[current_quiz],
            outputs=[mc_questions, tf_questions, sa_questions]
        )
        
        export_btn.click(
            fn=export_flashcards,
            inputs=[flashcards_json, export_format],
            outputs=[export_output]
        )
    
    return ui

