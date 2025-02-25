# AskDocs-AI
AskDocs AI is a smart chatbot that answers questions based on your documents. Using retrieval-augmented generation (RAG) techniques, it extracts relevant information from PDFs, Word files, and more, delivering quick, accurate answers to help you save time and access key insights effortlessly.

This project enables users to upload various document types and interact with an AI-driven chatbot that provides intelligent responses based on the document’s content. The system supports multiple document formats, including PDFs, Word documents, Excel files, PowerPoint presentations, and images (using OCR). It leverages advanced NLP models for question answering, reasoning, and generating helpful answers from the document's content.

---

## Features

- **File Upload Support**: Upload and process documents in multiple formats: PDF, DOCX, CSV, Excel, PPT, PPTX, and images.
- **Text Extraction**: Efficient extraction of text from different file types using libraries like `pdfplumber`, `python-docx`, `pytesseract` (OCR), and more.
- **Semantic Search with FAISS**: FAISS indexing and semantic search to retrieve relevant sections from large documents quickly.
- **Generative AI Integration**: Powered by the `google/flan-t5-base` model for accurate, context-aware question answering, reasoning, and text generation.
- **User Interaction**: AI-powered chatbot that interacts with users and provides document-based answers or casual conversation with custom vibes.
- **Document Processing**: Automatically processes uploaded documents to build a search index for easy and quick retrieval of information.
- **Fine-tuning Option**: Fine-tune the model with custom datasets like SQuAD to improve performance for domain-specific use cases.

---

## Installation

To set up the project, follow the steps below:

### Prerequisites:
1. **Python**: Version 3.7 or later
2. **Install Dependencies**: Install required packages using pip:

```bash
pip install gradio spacy pdfplumber torch numpy faiss-cpu pytesseract pdf2image transformers sentence-transformers pandas python-docx python-pptx opencv-python
```

3. **Install SpaCy Model**:
```bash
python -m spacy download en_core_web_md
```

---

## How It Works

1. **Upload File**: The user uploads a supported document (PDF, Word, PowerPoint, etc.).
2. **Text Extraction**: The system extracts text from the uploaded file using appropriate libraries.
3. **FAISS Indexing**: Extracted text is split into paragraphs and indexed using FAISS for fast retrieval of relevant content.
4. **Search and Answer Generation**: The user asks a question, and the AI performs a semantic search on the indexed text to retrieve the most relevant content, generating a detailed answer based on the retrieved information.
5. **Chatbot Interaction**: In addition to answering document-based queries, the system can also engage in casual conversation using predefined responses.

---

## Code Overview

### Key Functions

1. **`check_valid_document(file)`**: Checks if the uploaded file is a valid and supported document type (PDF, DOCX, PPTX, etc.).
2. **`extract_text_from_file(file)`**: Extracts text from different document types (PDF, Word, PowerPoint, etc.) and processes OCR for images.
3. **`build_faiss_index(text)`**: Processes the extracted text, splits it into paragraphs, and builds a FAISS index for efficient retrieval.
4. **`retrieve_relevant_passages(query)`**: Retrieves the most relevant passages from the indexed document based on the user query.
5. **`generate_answer(question)`**: Uses the `google/flan-t5-base` model to generate a detailed answer, leveraging retrieved passages and the context.
6. **`upload_and_process_file(file)`**: Handles file upload, text extraction, and FAISS index building.
7. **`chat_with_document(message, history)`**: Handles interaction with the chatbot. The chatbot can respond to queries and engage in casual conversation with custom vibes.

### Example Workflow

1. The user uploads a document (e.g., a PDF or PowerPoint presentation).
2. The system extracts the text from the file and indexes it using FAISS.
3. The user asks a question related to the document.
4. The system retrieves relevant passages and generates an answer using a pre-trained language model (Flan-T5).
5. The system engages in friendly conversation, providing responses in the “slime” style, a fun mixture of Pidgin and friendly vibes.

---

## Chatbot Interaction

The chatbot supports a variety of interactions, such as:

- **General Greetings**: “Yo! My slime! How your side? 😎”
- **Football Banter**: “You dey watch ball? Ah! My G, football na life! Which club you dey rep? ⚽”
- **Motivation**: “No wahala, my G. E go soon soft. 💯”
- **Casual Talk**: “Talk to me, my G. Wetin sup? 👀”
- **Banter**: “Omo, una don try o! But make I no lie, bottle FC still dey worry una. 🍼😂” (For Arsenal fans)

---

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. Ensure that your code is properly documented and includes tests for any new features.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to reach out with any questions or suggestions. Let's make document interaction easier and more fun!
