# AskDocs AI

A local-first, lightweight document interaction application similar to NotebookLM. Upload documents and interact with them through Q&A, summaries, flashcards, and quizzes—all running locally on your machine.

## ✨ Features

- **Multi-format Document Support**: PDF, DOCX, PPTX, TXT, Markdown, CSV, HTML
- **Intelligent Q&A**: Ask questions about your documents with conversational context
- **Automatic Summarization**: Section-by-section summaries with overall document summary
- **Flashcard Generation**: Auto-generate flashcards (Q&A pairs) from document content
- **Quiz Generation**: Create quizzes with multiple choice, true/false, and short answer questions
- **Local-first Architecture**: All data stored locally, no cloud dependencies required
- **Vector Search**: Fast semantic search using ChromaDB and sentence transformers

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** to `http://localhost:7860`

That's it! The application will run locally with default settings.

## 📋 Supported File Formats

- **PDF** (`.pdf`) - Text extraction from PDF files
- **Word Documents** (`.docx`) - Full text extraction
- **PowerPoint** (`.pptx`, `.ppt`) - Slide content extraction
- **Text Files** (`.txt`) - Plain text files
- **Markdown** (`.md`, `.markdown`) - Markdown documents
- **CSV** (`.csv`) - Tabular data
- **HTML** (`.html`, `.htm`) - Web pages

## 🏗️ Architecture

The application is organized into modular components:

```
AskDocs-AI/
├── app.py                 # Main entry point
├── config.py              # Configuration management
├── ui.py                  # Gradio UI interface
├── document_manager.py    # Document processing and storage
├── vector_store.py        # ChromaDB vector database
├── qa_engine.py           # Question answering engine
├── summarizer.py          # Document summarization
├── flashcard_generator.py  # Flashcard generation
├── quiz_generator.py      # Quiz generation
├── text_processor.py      # Text chunking and processing
└── parsers/               # Document parsers
    ├── document_parser.py
    ├── pdf_parser.py
    ├── docx_parser.py
    ├── pptx_parser.py
    ├── text_parser.py
    ├── markdown_parser.py
    ├── csv_parser.py
    └── html_parser.py
```

## 🔧 Configuration

Create a `.env` file (optional) to customize settings:

```env
# Embedding model (local, no API key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Language model (local)
LLM_MODEL=google/flan-t5-base

# Storage paths
DATA_DIR=./data
DOCUMENTS_DIR=./data/documents
VECTOR_DB_PATH=./data/vector_db

# UI settings
UI_TITLE=AskDocs AI
UI_PORT=7860
UI_DEBUG=false
```

**Note**: No API keys are required! The application runs entirely locally using open-source models.

## 📖 Usage Guide

### 1. Upload Documents

- Go to the **Documents** tab
- Click "Upload" and select a supported file
- Wait for processing to complete
- Select the document from the dropdown

### 2. View Summaries

- After selecting a document, go to the **Summary** tab
- View section-by-section summaries and overall summary
- Summaries are generated automatically

### 3. Ask Questions

- Navigate to the **Q&A** tab
- Type your question in the input box
- Ask follow-up questions for conversational context
- Clear history to start a new conversation

### 4. Generate Flashcards

- Select a document first
- Go to the **Flashcards** tab
- View auto-generated flashcards
- Export to JSON or text format

### 5. Take Quizzes

- Open the **Quiz** tab after selecting a document
- View generated quiz questions (multiple choice, true/false, short answer)
- Questions are auto-generated from document content

## 🔒 Privacy & Data

- **All data stays local**: Documents and embeddings are stored on your machine
- **No external API calls**: Everything runs locally by default
- **Optional cloud integration**: Can be configured to use OpenAI or Ollama if desired
- **Data storage**: Documents are stored in `./data/` directory

## 🛠️ Development

### Project Structure

- **Backend**: Python modules for document processing, embeddings, and Q&A
- **Frontend**: Gradio-based web interface
- **Storage**: ChromaDB for vector storage, local filesystem for documents

### Extending the Application

The modular architecture makes it easy to:

- **Add new file formats**: Create a new parser in `parsers/`
- **Customize summarization**: Modify `summarizer.py`
- **Enhance Q&A**: Update `qa_engine.py` with different models
- **Add features**: Extend `ui.py` with new tabs or components

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
2. **Model download**: Models download automatically on first run (may take time)
3. **Port already in use**: Change `UI_PORT` in `.env` or kill the process using port 7860
4. **Memory issues**: Reduce chunk size in `config.py` or use a smaller embedding model

### Logs

Check `app.log` for detailed error messages and debugging information.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Built with [Gradio](https://gradio.app/) for the UI
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by [sentence-transformers](https://www.sbert.net/) for embeddings
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/) for language models
