"""Flask web app for Avam Search — Server-side AI with optional passphrase auth."""
import os
import re
import uuid
import socket
import logging
import threading
import time
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session, g, Response
from werkzeug.utils import secure_filename

from config import DOCUMENTS_DIR, PERSISTENT_DOCS_DIR, UI_HOST, UI_PORT
from text_processor import split_sentences

class JSONFormatter(logging.Formatter):
    def format(self, record):
        try:
            req_id = g.request_id
        except Exception:
            req_id = None
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "req_id": req_id,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)

_handler = logging.StreamHandler()
_handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.jinja_env.auto_reload = True

@app.before_request
def _set_request_id():
    g.request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex[:12])
SECRET_KEY_PATH = Path("data/secret_key")
if SECRET_KEY_PATH.exists():
    app.secret_key = SECRET_KEY_PATH.read_text().strip()
else:
    import secrets
    app.secret_key = secrets.token_hex(32)
    SECRET_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRET_KEY_PATH.write_text(app.secret_key)
app.config['UPLOAD_FOLDER'] = str(DOCUMENTS_DIR)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024
app.config['GENERATED_DIR'] = 'data/generated'
app.config['SESSIONS_DIR'] = 'data/sessions'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_DIR'], exist_ok=True)
os.makedirs(app.config['SESSIONS_DIR'], exist_ok=True)
os.makedirs('data/audio', exist_ok=True)

DOCUMENTS_STORE = Path(app.config['SESSIONS_DIR']) / 'documents.json'

_documents = {}

def _save_documents():
    """Persist _documents to disk."""
    serializable = {}
    for doc_id, doc in _documents.items():
        serializable[doc_id] = {
            "id": doc.get("id"),
            "name": doc.get("name"),
            "path": doc.get("path"),
            "text": doc.get("text"),
            "metadata": doc.get("metadata"),
            "uploaded_at": doc.get("uploaded_at"),
        }
    try:
        DOCUMENTS_STORE.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to persist documents: {e}")

def _load_documents():
    """Load _documents from disk on startup."""
    global _documents
    if DOCUMENTS_STORE.exists():
        try:
            data = json.loads(DOCUMENTS_STORE.read_text(encoding="utf-8"))
            _documents = data
            logger.info(f"Loaded {len(_documents)} documents from disk")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")

_load_documents()

def _find_document(id_or_name):
    if not id_or_name:
        return None
    doc = _documents.get(id_or_name)
    if doc:
        return doc
    for d in _documents.values():
        if d.get("name") == id_or_name:
            return d
    return None

_flashcard_gen = None
_quiz_gen = None
_summarizer = None

# Model pre-loading
from model_loader import ModelRegistry

with app.app_context():
    threading.Thread(target=ModelRegistry.preload_all, daemon=True).start()


def _get_flashcard_gen():
    global _flashcard_gen
    if _flashcard_gen is None:
        from flashcard_generator import FlashcardGenerator
        _flashcard_gen = FlashcardGenerator()
    return _flashcard_gen


def _get_quiz_gen():
    global _quiz_gen
    if _quiz_gen is None:
        from quiz_generator import QuizGenerator
        _quiz_gen = QuizGenerator()
    return _quiz_gen


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        from vectorstores.chroma_store import ChromaVectorStore
        from summarizer import Summarizer
        _summarizer = Summarizer(vector_store=ChromaVectorStore())
    return _summarizer


_qa_engine = None


def _get_qa_engine():
    global _qa_engine
    if _qa_engine is None:
        from vectorstores.chroma_store import ChromaVectorStore
        from qa_engine import QAEngine
        _qa_engine = QAEngine(vector_store=ChromaVectorStore())
    return _qa_engine


def _get_powerpoint_func():
    from ancient_info.launcher import create_powerpoint, detect_tone
    return create_powerpoint, detect_tone


def _chunk_text(text, chunk_size=512, overlap=100):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    word_count = 0
    for para in paragraphs:
        para_words = len(para.split())
        if word_count + para_words > chunk_size and current:
            chunks.append(" ".join(current))
            tail = current[-3:] if len(current) > 3 else list(current)
            current = tail
            word_count = len(" ".join(tail).split())
        current.append(para)
        word_count += para_words
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


def _score_sentences(sentences, query):
    query_words = set(query.lower().split())
    scored = []
    for i, sent in enumerate(sentences):
        words = set(sent.lower().split())
        overlap = len(words & query_words)
        score = overlap / max(len(query_words), 1)
        if i < 3:
            score += 0.5
        scored.append((score, sent))
    scored.sort(reverse=True)
    return scored


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@app.route('/')
def home():
    return render_template('landing.html', doc_count=len(_documents))


@app.route('/app')
def app_page():
    initial_doc = request.args.get('doc', '')
    return render_template('index.html', initial_doc=initial_doc)

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "documents": len(_documents),
        "ai": "server-side (QAEngine+Summarizer)",
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(ModelRegistry.status())

@app.route('/api/documents/history', methods=['GET'])
def get_document_history():
    try:
        files = [f for f in os.listdir(str(PERSISTENT_DOCS_DIR)) if f.lower().endswith(('.pdf', '.docx', '.txt', '.md'))]
        docs = [{"name": f, "url": f"/uploads/{f}"} for f in files]
        return jsonify({"documents": docs}), 200
    except Exception as e:
        return jsonify({"documents": [], "error": str(e)}), 200

@app.route('/uploads/<filename>')
def serve_pdf(filename):
    return send_from_directory(str(PERSISTENT_DOCS_DIR), filename)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory('data/audio', filename)

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory('data/videos', filename)


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename or file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        original_name = secure_filename(file.filename)
        ext = original_name.rsplit('.', 1)[-1].lower() if '.' in original_name else ''

        allowed = {'pdf', 'docx', 'txt', 'md', 'csv', 'pptx', 'ppt', 'html', 'htm', 'xlsx', 'xls', 'rtf', 'odt', 'odp', 'ods', 'epub', 'xml', 'json', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp', 'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'mp4', 'webm', 'mov', 'avi', 'mkv'}
        if ext not in allowed:
            return jsonify({'success': False, 'error': f'Unsupported file type: .{ext}'}), 400

        unique_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / unique_name
        file.save(filepath)

        from parsers.document_parser import DocumentParser
        parser = DocumentParser.get_parser_for_file(filepath)
        if not parser:
            return jsonify({'success': False, 'error': f'No parser for .{ext}'}), 400

        result = parser.parse(filepath)
        text = result.get("text", "")
        if not text or text.startswith("\u274c"):
            return jsonify({'success': False, 'error': result.get("text", "Could not extract text")}), 400

        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\s+', ' ', text).strip()

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        _documents[doc_id] = {
            "id": doc_id, "name": original_name, "path": str(filepath),
            "text": text, "metadata": result.get("metadata", {}),
            "uploaded_at": datetime.now().isoformat()
        }

        _save_documents()

        # Also copy to persistent docs dir for document history
        try:
            persistent_path = PERSISTENT_DOCS_DIR / unique_name
            shutil.copy2(str(filepath), str(persistent_path))
        except Exception:
            pass

        preview = text[:2000] if text else ""
        return jsonify({
            'success': True,
            'document': {
                'id': doc_id, 'name': original_name, 'preview': preview,
                'word_count': len(text.split()),
                'uploaded_at': datetime.now().isoformat(),
            }
        })

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload/text', methods=['POST'])
def upload_text():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        name = data.get('name', 'Untitled text')
        content = data.get('content', '')
        if not content or len(content) < 10:
            return jsonify({'success': False, 'error': 'Text is too short (min 10 characters)'}), 400

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        _documents[doc_id] = {
            "id": doc_id, "name": name, "path": "",
            "text": content, "metadata": {},
            "uploaded_at": datetime.now().isoformat()
        }
        _save_documents()

        return jsonify({
            'success': True,
            'document': {
                'id': doc_id, 'name': name,
                'word_count': len(content.split()),
                'uploaded_at': datetime.now().isoformat(),
            }
        })
    except Exception as e:
        logger.error(f"Text upload error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload endpoint for the new frontend — returns pdf_url and filename."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "No filename"}), 400

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        dest = PERSISTENT_DOCS_DIR / unique_name
        file.save(dest)

        pdf_url = f"/uploads/{unique_name}"
        return jsonify({"status": "success", "filename": unique_name, "pdf_url": pdf_url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe_media():
    """Transcribe audio/video files or YouTube URLs into document text."""
    try:
        import tempfile
        url = request.form.get('url', '')
        text_content = ''

        if url:
            # YouTube/Podcast URL
            try:
                import yt_dlp
                with tempfile.TemporaryDirectory() as tmpdir:
                    ydl_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(tmpdir, '%(title)s.%(ext)s'),
                        'quiet': True,
                        'no_warnings': True,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        title = info.get('title', 'Media')
                        files = list(Path(tmpdir).glob('*'))
                        if files:
                            audio_path = str(files[0])
                            try:
                                import whisper
                                model = whisper.load_model("base")
                                result = model.transcribe(audio_path)
                                text_content = result.get("text", "")
                            except ImportError:
                                return jsonify({'success': False, 'error': 'Whisper not installed. Run: pip install openai-whisper'}), 400
            except ImportError:
                return jsonify({'success': False, 'error': 'yt-dlp not installed. Run: pip install yt-dlp'}), 400
        elif 'file' in request.files:
            file = request.files['file']
            if not file.filename:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
            audio_exts = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac', 'wma'}
            video_exts = {'mp4', 'webm', 'mov', 'avi', 'mkv', 'm4v'}
            if ext not in audio_exts | video_exts:
                return jsonify({'success': False, 'error': f'Unsupported media format: .{ext}'}), 400

            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir) / secure_filename(file.filename)
                file.save(str(filepath))
                try:
                    import whisper
                    model = whisper.load_model("base")
                    result = model.transcribe(str(filepath))
                    text_content = result.get("text", "")
                except ImportError:
                    return jsonify({'success': False, 'error': 'Whisper not installed. Run: pip install openai-whisper'}), 400
        else:
            return jsonify({'success': False, 'error': 'No file or URL provided'}), 400

        if not text_content.strip():
            return jsonify({'success': False, 'error': 'No speech detected in the media'}), 400

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        name = file.filename if 'file' in request.files and file.filename else (url.split('/')[-1][:50] if url else 'Media transcription')
        _documents[doc_id] = {
            "id": doc_id, "name": name, "path": "",
            "text": text_content, "metadata": {"source": "transcription"},
            "uploaded_at": datetime.now().isoformat()
        }
        _save_documents()

        return jsonify({
            'success': True,
            'document': {
                'id': doc_id, 'name': name,
                'word_count': len(text_content.split()),
                'uploaded_at': datetime.now().isoformat(),
            }
        })
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/documents', methods=['GET'])
def list_documents():
    docs = list(_documents.values())
    return jsonify({'documents': [
        {'id': d.get('id', ''), 'name': d.get('name', ''),
         'word_count': len((d.get('text', '') or '').split()),
         'uploaded_at': d.get('uploaded_at', '')}
        for d in docs
    ]})


@app.route('/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'success': False, 'error': 'Document not found'}), 404
    try:
        os.remove(doc.get('path', ''))
    except OSError:
        pass
    try:
        _get_qa_engine().vector_store.delete_document(doc_id)
    except Exception:
        pass
    del _documents[doc_id]
    _save_documents()
    return jsonify({'success': True, 'message': f'Deleted: {doc.get("name", doc_id)}'})


@app.route('/document/<doc_id>/text', methods=['GET'])
def get_document_text(doc_id):
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    text = doc.get('text', '')
    limit = min(request.args.get('limit', 5000, type=int), 500000)
    return jsonify({
        'id': doc_id, 'name': doc.get('name', ''),
        'text': text[:limit], 'total_length': len(text), 'truncated': len(text) > limit,
    })


@app.route('/document/<doc_id>/chunks', methods=['GET'])
def get_document_chunks(doc_id):
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    text = doc.get('text', '')
    chunk_size = request.args.get('chunk_size', 512, type=int)
    overlap = request.args.get('overlap', 100, type=int)
    chunks = _chunk_text(text, chunk_size, overlap)
    return jsonify({'chunks': chunks, 'count': len(chunks)})


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    doc_id = data.get('doc_id', '')
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'chunks': []})
    try:
        vs = _get_qa_engine().vector_store
        results = vs.search(doc_id, query, top_k=top_k)
        chunks = [{'text': r['text'], 'distance': r.get('distance'), 'metadata': r.get('metadata', {})} for r in results]
        return jsonify({'chunks': chunks, 'count': len(chunks), 'method': 'semantic'})
    except Exception as e:
        logger.error(f"Semantic search failed, falling back: {e}")
        text = doc.get('text', '')
        sentences = split_sentences(text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        scored = _score_sentences(sentences, query)
        results = [s[1] for s in scored[:top_k]]
        return jsonify({'chunks': results, 'count': len(results), 'method': 'keyword'})


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    doc_ids = data.get('doc_ids', None) or ([data.get('doc_id')] if data.get('doc_id') else [])
    question = data.get('question', '')
    if not doc_ids or not doc_ids[0]:
        return jsonify({'error': 'Document ID(s) required'}), 400
    if not question:
        return jsonify({'error': 'Question required'}), 400
    for did in doc_ids:
        if did not in _documents:
            return jsonify({'error': f'Document {did} not found'}), 404
    try:
        qa = _get_qa_engine()
        if len(doc_ids) == 1:
            result = qa.ask(doc_ids[0], question)
        else:
            result = qa.ask_multi(doc_ids, question)
        return jsonify(result)
    except Exception as e:
        logger.error(f"QA engine failed: {e}")
        return jsonify({'error': str(e), 'answer': 'Could not process question due to an error.'}), 500


@app.route('/ask/stream', methods=['POST'])
def ask_stream():
    data = request.json
    doc_ids = data.get('doc_ids', None) or ([data.get('doc_id')] if data.get('doc_id') else [])
    question = data.get('question', '')
    if not doc_ids or not doc_ids[0]:
        return jsonify({'error': 'Document ID(s) required'}), 400
    if not question:
        return jsonify({'error': 'Question required'}), 400
    for did in doc_ids:
        if did not in _documents:
            return jsonify({'error': f'Document {did} not found'}), 404

    def generate():
        try:
            qa = _get_qa_engine()
            if len(doc_ids) == 1:
                gen = qa.ask_stream(doc_ids[0], question)
            else:
                gen = qa.ask_stream_multi(doc_ids, question)
            for event_type, payload in gen:
                if event_type == 'token':
                    yield f"data: {json.dumps({'type':'token','text':payload})}\n\n"
                elif event_type == 'evidence':
                    yield f"data: {json.dumps({'type':'evidence','evidence':payload})}\n\n"
                elif event_type == 'grounding':
                    yield f"data: {json.dumps({'type':'grounding','grounding':payload})}\n\n"
                elif event_type == 'sources':
                    yield f"data: {json.dumps({'type':'sources','sources':payload})}\n\n"
                elif event_type == 'done':
                    yield f"data: {json.dumps({'type':'done','answer':payload})}\n\n"
                elif event_type == 'answer':
                    yield f"data: {json.dumps({'type':'answer','text':payload})}\n\n"
                elif event_type == 'confidence':
                    yield f"data: {json.dumps({'type':'confidence','confidence':payload})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type':'error','text':str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
        'Connection': 'keep-alive',
    })


@app.route('/summary', methods=['POST'])
def summary():
    data = request.json
    doc_id = data.get('doc_id', '')
    detail = data.get('detail', 'standard')
    focus = data.get('focus', '')
    doc = _find_document(doc_id) if doc_id else (list(_documents.values())[0] if _documents else None)
    if not doc:
        return jsonify({'error': 'No document found'}), 404
    text = doc.get('text', '')
    try:
        summarizer = _get_summarizer()
        result = summarizer.summarize_document(text, detail_level=detail, focus=focus)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return jsonify({'error': 'Could not generate summary. Please try again later.'}), 500


@app.route('/flashcards', methods=['POST'])
def flashcards():
    data = request.json
    doc_id = data.get('doc_id', '')
    count = data.get('count', 10)
    if not doc_id:
        return jsonify({'error': 'Document ID required'}), 400
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    text = doc.get('text', '')
    try:
        gen = _get_flashcard_gen()
        cards = gen.generate_from_text(text, num_flashcards=int(count))
        return jsonify({'cards': cards, 'count': len(cards)})
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        return jsonify({'error': 'Could not generate flashcards. Please try again later.'}), 500


@app.route('/quiz', methods=['POST'])
def quiz():
    data = request.json
    doc_id = data.get('doc_id', '')
    count = data.get('count', 10)
    if not doc_id:
        return jsonify({'error': 'Document ID required'}), 400
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    text = doc.get('text', '')
    try:
        gen = _get_quiz_gen()
        quiz_data = gen.generate_quiz(text, num_questions=int(count))
        breakdown = {
            'multiple_choice': len(quiz_data.get('multiple_choice', [])),
            'true_false': len(quiz_data.get('true_false', [])),
            'fill_blank': len(quiz_data.get('fill_blank', [])),
            'short_answer': len(quiz_data.get('short_answer', [])),
        }
        return jsonify({'quiz': quiz_data, 'breakdown': breakdown})
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        return jsonify({'error': 'Could not generate quiz. Please try again later.'}), 500


@app.route('/quiz/submit', methods=['POST'])
def quiz_submit():
    data = request.json
    quiz_data = data.get('quiz', {})
    answers = data.get('answers', {})
    if not quiz_data or not answers:
        return jsonify({'error': 'Quiz data and answers required'}), 400
    try:
        gen = _get_quiz_gen()
        result = gen.calculate_score(quiz_data, answers)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Quiz scoring failed: {e}")
        return jsonify({'error': 'Could not score quiz. Please try again.'}), 500

@app.route('/pptx', methods=['POST'])
def generate_pptx():
    data = request.json
    doc_id = data.get('doc_id', '')
    text = data.get('text', '')
    title = data.get('title', 'Presentation')
    detail = data.get('detail', 'standard')
    tone = data.get('tone', 'auto')
    custom_count = data.get('slide_count', None)
    focus = data.get('focus', '')

    if doc_id:
        doc = _find_document(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        text = doc.get('text', '')
    elif not text:
        return jsonify({'error': 'Document or text required'}), 400

    try:
        base_count = {'brief': 4, 'standard': 6, 'detailed': 12}.get(detail, 6)
        slide_count = custom_count if custom_count else base_count
        slide_count = max(3, min(50, int(slide_count)))

        create_powerpoint, detect_tone = _get_powerpoint_func()

        detected_tone = detect_tone(text) if tone == 'auto' else tone

        if focus:
            text = f"[Focus: {focus}]\n{text}"

        sentences = split_sentences(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        slides = [{'title': title, 'content': f'This presentation covers the key topics from {title}.'}]

        if slide_count > 1 and sentences:
            chunk_size = max(2, len(sentences) // (slide_count - 1))
            chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
            topic_keywords = ['Introduction', 'Overview', 'Background', 'Key Points', 'Analysis', 'Details', 'Implementation', 'Results', 'Discussion', 'Conclusion', 'Summary', 'Recommendations', 'Next Steps']

            for i, chunk in enumerate(chunks[:slide_count - 1]):
                if not chunk:
                    continue
                topic = topic_keywords[i % len(topic_keywords)] if i < len(topic_keywords) else f'Section {i+1}'
                content = ' '.join(chunk)[:500]

                slides.append({'title': topic, 'content': content})

        slides = slides[:slide_count]

        output_path = os.path.join(app.config['GENERATED_DIR'], f"{uuid.uuid4().hex[:8]}_presentation.pptx")
        result_path = create_powerpoint(slides, output_path)
        if result_path:
            return send_file(result_path, as_attachment=True, download_name='presentation.pptx', mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation')
        return jsonify({'success': False, 'error': 'Failed to create PowerPoint'}), 500
    except Exception as e:
        logger.error(f"PowerPoint generation failed: {e}")
        return jsonify({'error': 'Could not generate presentation. Please try again later.'}), 500


@app.route('/pptx/preview', methods=['POST'])
def generate_pptx_preview():
    data = request.json
    doc_id = data.get('doc_id', '')
    text = data.get('text', '')
    title = data.get('title', 'Presentation')
    detail = data.get('detail', 'standard')
    custom_count = data.get('slide_count', None)
    focus = data.get('focus', '')

    if doc_id:
        doc = _find_document(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        text = doc.get('text', '')
    elif not text:
        return jsonify({'error': 'Document or text required'}), 400

    try:
        base_count = {'brief': 4, 'standard': 6, 'detailed': 10}.get(detail, 6)
        slide_count = custom_count if custom_count else base_count
        slide_count = max(3, min(50, int(slide_count)))

        if focus:
            text = f"[Focus: {focus}]\n{text}"

        sentences = split_sentences(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        slides = [{'title': title}]
        topic_keywords = ['Introduction', 'Overview', 'Background', 'Key Points', 'Analysis', 'Details', 'Implementation', 'Results', 'Discussion', 'Conclusion', 'Summary', 'Recommendations', 'Next Steps']

        if slide_count > 1 and sentences:
            chunk_size = max(2, len(sentences) // (slide_count - 1))
            chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
            for i, chunk in enumerate(chunks[:slide_count - 1]):
                topic = topic_keywords[i % len(topic_keywords)] if i < len(topic_keywords) else f'Section {i+1}'
                slides.append({'title': topic})

        slides = slides[:slide_count]
        return jsonify({'slides': slides, 'count': len(slides), 'tone': 'auto'})
    except Exception as e:
        logger.error(f"Presentation preview failed: {e}")
        return jsonify({'error': 'Could not generate preview.'}), 500


@app.route('/export/<doc_id>/<content_type>', methods=['GET'])
def export_content(doc_id, content_type):
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    try:
        export_type = request.args.get('type', content_type)
        export_format = request.args.get('format', 'json')
        if export_type == 'flashcards':
            gen = _get_flashcard_gen()
            text = doc.get('text', '')
            cards = gen.generate_from_text(text, num_flashcards=15)
            if export_format == 'json':
                return jsonify({'cards': cards, 'count': len(cards)})
            elif export_format == 'text':
                lines = []
                for i, card in enumerate(cards, 1):
                    lines.append(f"Flashcard {i}:\nQ: {card['question']}\nA: {card['answer']}\n")
                return jsonify({'text': '\n'.join(lines)})
        return jsonify({
            'text': doc.get('text', ''),
            'name': doc.get('name', ''),
            'type': content_type,
        })
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({'error': 'Could not export content. Please try again later.'}), 500


@app.route('/cleanup', methods=['POST'])
def cleanup_old_files():
    try:
        gen_dir = Path(app.config['GENERATED_DIR'])
        now = time.time()
        removed = []
        for f in gen_dir.glob('*.pptx'):
            if now - f.stat().st_mtime > 86400:
                f.unlink()
                removed.append(f.name)
        return jsonify({'message': f'Cleaned {len(removed)} old files', 'removed': removed})
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return jsonify({'error': 'Could not clean up files.'}), 500


# Feedback endpoints
from feedback import submit as submit_feedback, get_by_ref, get_summary as feedback_summary

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json or {}
    if not data.get("feature") or not data.get("rating"):
        return jsonify({"error": "feature and rating required"}), 400
    ref = submit_feedback(
        feature=data.get("feature", "other"),
        rating=data.get("rating", "bad"),
        description=data.get("description", ""),
        output_excerpt=data.get("output_excerpt", "") if data.get("include_output") else ""
    )
    return jsonify({"success": True, "reference_code": ref})

@app.route("/feedback/<ref>", methods=["GET"])
def feedback_status(ref):
    entry = get_by_ref(ref)
    if not entry:
        return jsonify({"found": False})
    return jsonify({
        "found": True,
        "feature": entry["feature"],
        "reviewed": entry.get("is_reviewed", False),
        "submitted": entry["timestamp"]
    })

@app.route("/admin/feedback", methods=["GET"])
def admin_feedback():
    token = request.headers.get("X-Admin-Token", "")
    if token != os.getenv("ADMIN_TOKEN", ""):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(feedback_summary())


# Auth routes
@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/auth/status')
def auth_status():
    from auth import user_exists, _load_users
    return jsonify({"has_passphrase": user_exists(), "authenticated": session.get("authenticated", False)})

@app.route('/auth/register', methods=['POST'])
def auth_register():
    from auth import register_passphrase
    data = request.json or {}
    passphrase = data.get('passphrase', '')
    if len(passphrase) < 4:
        return jsonify({'error': 'Passphrase must be at least 4 characters'}), 400
    success, msg = register_passphrase(passphrase)
    if success:
        from auth import verify_passphrase
        verify_passphrase(passphrase)
        return jsonify({'message': msg}), 200
    return jsonify({'error': msg}), 400

@app.route('/auth/login', methods=['POST'])
def auth_login():
    from auth import verify_passphrase
    data = request.json or {}
    passphrase = data.get('passphrase', '')
    if verify_passphrase(passphrase):
        return jsonify({'message': 'Signed in successfully'}), 200
    return jsonify({'error': 'Invalid passphrase'}), 401

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    from auth import logout
    logout()
    return jsonify({'message': 'Signed out'}), 200

# Dashboard + Notebook routes
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/notebooks', methods=['GET'])
def list_notebooks_route():
    from notebook_manager import list_notebooks
    nbs = list_notebooks()
    return jsonify({'notebooks': nbs})

@app.route('/notebooks', methods=['POST'])
def create_notebook_route():
    from notebook_manager import create_notebook
    data = request.json or {}
    name = data.get('name', 'Untitled')
    nb = create_notebook(name)
    return jsonify({'notebook': nb}), 201

@app.route('/notebooks/<nb_id>', methods=['DELETE'])
def delete_notebook_route(nb_id):
    from notebook_manager import delete_notebook
    delete_notebook(nb_id)
    return jsonify({'message': 'Deleted'})

# --- Spaced Repetition Flashcards (SM-2) ---
import math

SPACED_REP_DATA_PATH = Path(app.config['SESSIONS_DIR']) / 'spaced_repetition.json'

def _load_spaced_data():
    if SPACED_REP_DATA_PATH.exists():
        try:
            return json.loads(SPACED_REP_DATA_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_spaced_data(data):
    SPACED_REP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPACED_REP_DATA_PATH.write_text(json.dumps(data, indent=2))

def _sm2(card, quality):
    """SM-2 algorithm. Updates card dict in-place. quality: 0-5."""
    quality = max(0, min(5, quality))
    if quality >= 3:
        if card['repetitions'] == 0:
            card['interval'] = 1
        elif card['repetitions'] == 1:
            card['interval'] = 6
        else:
            card['interval'] = round(card['interval'] * card['easiness'])
        card['repetitions'] += 1
    else:
        card['repetitions'] = 0
        card['interval'] = 1
    card['easiness'] = max(1.3, card['easiness'] + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return card

@app.route('/spaced/flashcards', methods=['POST'])
def spaced_flashcards_create():
    data = request.json or {}
    doc_id = data.get('doc_id', '')
    count = data.get('count', 10)
    if not doc_id:
        return jsonify({'error': 'Document ID required'}), 400
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    try:
        gen = _get_flashcard_gen()
        cards = gen.generate_from_text(doc.get('text', ''), num_flashcards=int(count))
        spaced = _load_spaced_data()
        if doc_id not in spaced:
            spaced[doc_id] = []
        existing_ids = {c.get('card_id') for c in spaced[doc_id]}
        for i, card in enumerate(cards):
            cid = f"{doc_id}_{i}"
            if cid not in existing_ids:
                spaced[doc_id].append({
                    'card_id': cid,
                    'question': card.get('question', ''),
                    'answer': card.get('answer', ''),
                    'easiness': 2.5,
                    'interval': 0,
                    'repetitions': 0,
                    'next_review': datetime.now(timezone.utc).isoformat()
                })
        _save_spaced_data(spaced)
        return jsonify({'cards': spaced[doc_id], 'count': len(spaced[doc_id])})
    except Exception as e:
        logger.error(f"Spaced flashcards failed: {e}")
        return jsonify({'error': 'Could not generate flashcards.'}), 500

@app.route('/spaced/review', methods=['POST'])
def spaced_flashcards_review():
    data = request.json or {}
    doc_id = data.get('doc_id', '')
    card_id = data.get('card_id', '')
    quality = data.get('quality', 3)
    if not doc_id or not card_id:
        return jsonify({'error': 'Document ID and card_id required'}), 400
    spaced = _load_spaced_data()
    cards = spaced.get(doc_id, [])
    for card in cards:
        if card['card_id'] == card_id:
            from datetime import timedelta
            card = _sm2(card, int(quality))
            card['next_review'] = (datetime.now(timezone.utc) + timedelta(days=card['interval'])).isoformat()
            _save_spaced_data(spaced)
            return jsonify({'card': card, 'next_interval': card['interval']})
    return jsonify({'error': 'Card not found'}), 404

@app.route('/spaced/due', methods=['GET'])
def spaced_flashcards_due():
    doc_id = request.args.get('doc_id', '')
    spaced = _load_spaced_data()
    cards = spaced.get(doc_id, [])
    now = datetime.now(timezone.utc)
    due = [c for c in cards if c.get('next_review', '2000-01-01') <= now.isoformat()]
    return jsonify({'cards': due, 'count': len(due)})


# --- Knowledge Graph ---
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

@app.route('/knowledge-graph', methods=['POST'])
def knowledge_graph():
    data = request.json or {}
    doc_id = data.get('doc_id', '')
    if not doc_id:
        return jsonify({'error': 'Document ID required'}), 400
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    try:
        text = doc.get('text', '')
        entities = []
        if SPACY_AVAILABLE:
            doc_nlp = _nlp(text[:50000])
            seen = set()
            for ent in doc_nlp.ents:
                if ent.text.lower() not in seen:
                    seen.add(ent.text.lower())
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
        else:
            # Fallback: simple regex-based extraction
            import re
            patterns = [
                (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', 'PERSON'),
                (r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company)\b', 'ORG'),
                (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'DATE'),
            ]
            seen = set()
            for pat, label in patterns:
                for m in re.finditer(pat, text):
                    if m.group() not in seen:
                        seen.add(m.group())
                        entities.append({'text': m.group(), 'label': label, 'start': m.start(), 'end': m.end()})

        # Build relationships for visualization (co-occurrence within 500 chars)
        relationships = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue
                if abs(e1['start'] - e2['start']) < 500:
                    relationships.append({'source': e1['text'], 'target': e2['text'], 'label': 'co-occurs'})

        return jsonify({'entities': entities[:100], 'relationships': relationships[:200], 'count': len(entities)})
    except Exception as e:
        logger.error(f"Knowledge graph failed: {e}")
        return jsonify({'error': 'Could not extract entities.'}), 500


# --- Progress Analytics ---
PROGRESS_PATH = Path(app.config['SESSIONS_DIR']) / 'progress.json'

def _load_progress():
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except Exception:
            return {}
    return {"quiz_attempts": [], "documents_studied": {}, "total_questions": 0, "correct_answers": 0}

def _save_progress(data):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(data, indent=2))

@app.route('/progress', methods=['GET'])
def get_progress():
    data = _load_progress()
    return jsonify(data)

@app.route('/progress/quiz', methods=['POST'])
def record_quiz_progress():
    data = request.json or {}
    doc_id = data.get('doc_id', '')
    score = data.get('score', 0)
    total = data.get('total', 0)
    details = data.get('details', [])
    if not doc_id:
        return jsonify({'error': 'Document ID required'}), 400
    try:
        progress = _load_progress()
        progress['quiz_attempts'].append({
            'doc_id': doc_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'score': score,
            'total': total,
            'percentage': round(score / max(total, 1) * 100, 1)
        })
        progress['total_questions'] += total
        progress['correct_answers'] += score
        if doc_id not in progress['documents_studied']:
            progress['documents_studied'][doc_id] = 0
        progress['documents_studied'][doc_id] += 1
        progress['quiz_attempts'] = progress['quiz_attempts'][-100:]  # keep last 100
        _save_progress(progress)
        return jsonify({'success': True, 'progress': progress})
    except Exception as e:
        logger.error(f"Progress save failed: {e}")
        return jsonify({'error': 'Could not save progress.'}), 500


@app.route('/progress/reset', methods=['POST'])
def reset_progress():
    try:
        if PROGRESS_PATH.exists():
            PROGRESS_PATH.unlink()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Cross-document search ---
from difflib import SequenceMatcher

@app.route('/cross-search', methods=['POST'])
def cross_search():
    data = request.json or {}
    query = data.get('query', '')
    doc_ids = data.get('doc_ids', [])
    if not query or not doc_ids:
        return jsonify({'error': 'query and doc_ids required'}), 400
    results = []
    qa = _get_qa_engine()
    for doc_id in doc_ids:
        doc = _documents.get(doc_id)
        if not doc:
            continue
        try:
            chunks = qa.vector_store.search(doc_id, query, top_k=3)
            for c in chunks:
                results.append({
                    'doc_id': doc_id,
                    'doc_name': doc.get('name', doc_id),
                    'text': c['text'],
                    'distance': c.get('distance'),
                })
        except Exception:
            pass
    results.sort(key=lambda r: r.get('distance', 1))
    return jsonify({'results': results[:10], 'count': len(results[:10])})


@app.route('/compare', methods=['POST'])
def compare_documents():
    data = request.json or {}
    doc_ids = data.get('doc_ids', [])
    if len(doc_ids) < 2:
        return jsonify({'error': 'At least 2 doc_ids required'}), 400
    docs = []
    for did in doc_ids:
        d = _documents.get(did)
        if d:
            docs.append(d)
    if len(docs) < 2:
        return jsonify({'error': 'Not enough valid documents found'}), 404
    texts = [d.get('text', '') for d in docs]
    ratio = SequenceMatcher(None, texts[0], texts[1]).ratio()
    longer = texts[0] if len(texts[0]) >= len(texts[1]) else texts[1]
    shorter = texts[1] if longer is texts[0] else texts[0]
    blocks = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, longer, shorter).get_opcodes():
        if op != 'equal':
            blocks.append({
                'op': op,
                'longer_slice': longer[i1:i2][:200],
                'shorter_slice': shorter[j1:j2][:200] if j1 < len(shorter) else '',
            })
    return jsonify({
        'similarity': round(ratio * 100, 1),
        'differences': blocks[:20],
        'doc_a': {'id': doc_ids[0], 'name': docs[0].get('name', doc_ids[0])},
        'doc_b': {'id': doc_ids[1], 'name': docs[1].get('name', doc_ids[1])},
    })


@app.route('/export/<doc_id>/report', methods=['GET'])
def export_report(doc_id):
    doc = _documents.get(doc_id)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    name = doc.get('name', doc_id)
    text = doc.get('text', '')
    preview = text[:3000]
    summary = ""
    try:
        s = _get_summarizer()
        summary = s.summarize(text, detail_level='brief')
    except Exception:
        summary = preview[:500]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{name} - Report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; color: #1a1a2e; line-height: 1.7; }}
  h1 {{ font-size: 1.8em; border-bottom: 2px solid #4F46E5; padding-bottom: .3em; }}
  h2 {{ font-size: 1.3em; margin-top: 1.5em; color: #4F46E5; }}
  .summary {{ background: #f5f3ff; padding: 1em 1.5em; border-radius: 8px; margin: 1em 0; }}
  .meta {{ color: #666; font-size: .85em; }}
  pre {{ white-space: pre-wrap; background: #f8f9fa; padding: 1em; border-radius: 6px; font-size: .9em; }}
  @media print {{ body {{ margin: 0; padding: .5in; }} }}
</style></head><body>
<h1>{name}</h1>
<p class="meta">Generated by Avam Search</p>
<div class="summary"><strong>Summary:</strong> {summary}</div>
<h2>Full Document Preview</h2>
<pre>{preview}</pre>
<script>window.print()</script>
</body></html>"""
    return html, 200, {'Content-Type': 'text/html'}


# ==== NEW API ROUTES (ask, generate, feedback, export) ====

@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.json or {}
    question = data.get("question", "")
    if not question or not _documents:
        return jsonify({"error": "No question or no documents loaded"}), 400

    doc_id = list(_documents.keys())[0]
    if not ModelRegistry.is_ready():
        return jsonify({
            "status": "loading",
            "message": "Models are still loading. Please wait...",
            "progress": ModelRegistry.status()
        }), 202

    engine = _get_qa_engine()
    result = engine.ask(doc_id, question, use_history=True, show_snippets=True)

    embedder = ModelRegistry.get("embeddings")
    grounding = {"score": None, "warning": None}
    if embedder and result.get("answer"):
        try:
            from sentence_transformers import util
            ans_emb = embedder.encode(result["answer"], convert_to_tensor=True)
            ctx_parts = [s.get("sentence", "") for s in result.get("evidence", [])]
            if ctx_parts:
                ctx_emb = embedder.encode(" ".join(ctx_parts), convert_to_tensor=True)
                score = float(util.cos_sim(ans_emb, ctx_emb)[0][0])
                grounding = {
                    "score": round(score, 3),
                    "warning": None if score >= 0.4 else (
                        "This answer may not be fully grounded in your document — verify against the cited sections."
                    )
                }
        except Exception:
            pass

    return jsonify({**result, "grounding": grounding})


@app.route('/api/generate/quiz', methods=['POST'])
def api_generate_quiz():
    if not _documents:
        return jsonify({"error": "No document loaded"}), 400
    from schemas import QuizModel

    doc_id = list(_documents.keys())[0]
    text = _documents[doc_id].get("text", "")

    schema = QuizModel.model_json_schema()
    try:
        import httpx
        resp = httpx.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:7b",
            "prompt": f"Generate a comprehensive quiz based on this text:\n{text[:3000]}",
            "format": schema,
            "stream": False
        }, timeout=60)
        raw = resp.json().get("response", "{}")
    except Exception:
        from quiz_generator import QuizGenerator
        qg = QuizGenerator()
        raw = json.dumps(qg.generate_quiz(text, 5))

    try:
        validated = QuizModel.model_validate_json(raw)
        return jsonify(validated.model_dump())
    except Exception:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict) and ("multiple_choice" in parsed or "questions" in parsed):
                return jsonify(parsed)
        except Exception:
            pass
        return jsonify({"error": "Failed to parse quiz output", "raw": raw}), 500


@app.route('/api/generate/flashcards', methods=['POST'])
def api_generate_flashcards():
    if not _documents:
        return jsonify({"error": "No document loaded"}), 400
    from schemas import FlashcardDeck

    doc_id = list(_documents.keys())[0]
    text = _documents[doc_id].get("text", "")

    schema = FlashcardDeck.model_json_schema()
    try:
        import httpx
        resp = httpx.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:7b",
            "prompt": f"Generate flashcards from this text:\n{text[:3000]}",
            "format": schema,
            "stream": False
        }, timeout=60)
        raw = resp.json().get("response", "{}")
    except Exception:
        raw = json.dumps({
            "deck_name": "Core Concepts",
            "cards": [{"front": "Sample concept?", "back": "Sample answer."}]
        })

    try:
        validated = FlashcardDeck.model_validate_json(raw)
        return jsonify(validated.model_dump())
    except Exception as e:
        return jsonify({"error": f"Flashcard generation failed: {str(e)}"}), 500


@app.route('/api/generate/graph', methods=['POST'])
def api_generate_graph():
    if not _documents:
        return jsonify({"error": "No document loaded"}), 400

    doc_id = list(_documents.keys())[0]
    text = _documents[doc_id].get("text", "")

    from model_loader import ModelRegistry
    ner = ModelRegistry.get("ner")
    nodes = []
    edges = []
    seen_labels = set()

    if ner:
        entities = ner(text[:3000])
        for ent in entities:
            label = ent.get("word", "").strip()
            etype = ent.get("entity_group", "CONCEPT")
            if label and label.lower() not in seen_labels and len(label) > 2:
                seen_labels.add(label.lower())
                type_colors = {
                    "PER": "#ef4444", "PERSON": "#ef4444",
                    "ORG": "#3b82f6", "ORGANIZATION": "#3b82f6",
                    "LOC": "#22c55e", "LOCATION": "#22c55e",
                    "CONCEPT": "#a855f7", "DEFINITION": "#f59e0b",
                    "ALGORITHM": "#06b6d4",
                }
                color = type_colors.get(etype, "#6b7280")
                nodes.append({"id": label, "label": label, "type": etype, "color": color})

    if len(nodes) >= 2:
        for i in range(min(len(nodes) - 1, 30)):
            edges.append({"source": nodes[i]["id"], "target": nodes[i + 1]["id"], "label": "related"})

    return jsonify({"nodes": nodes[:50], "edges": edges})


@app.route('/api/generate/audio-overview', methods=['POST'])
def api_generate_audio_overview():
    """Generate a NotebookLM-style podcast audio overview from document text."""
    if not _documents:
        return jsonify({"error": "No document loaded"}), 400
    data = request.json or {}
    doc_id = data.get('doc_id', '')
    doc = _find_document(doc_id) if doc_id else list(_documents.values())[0]
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    text = doc.get("text", "")[:4000]

    try:
        summarizer = _get_summarizer()
        summary = summarizer.summarize_document(text, detail_level="standard", focus="")
    except Exception as e:
        logger.warning(f"Summarizer failed for audio overview: {e}")
        summary = {"overall": text[:1500], "sections": [{"title": "Overview", "summary": text[:1000]}]}

    overall = summary.get("overall", "") or ""
    sections = summary.get("sections", []) or []
    intro = f"Welcome to this audio overview of {doc.get('name', 'the document')}."
    body_parts = [s.get("summary", "") for s in sections[:5] if s.get("summary")]
    closing = "That concludes this overview. Thanks for listening."
    plain_text = " ".join([intro] + body_parts + [closing])

    lines = [{"speaker": "Host1", "text": intro}]
    for i, sp in enumerate(body_parts):
        speaker = "Host1" if i % 2 == 0 else "Host2"
        lines.append({"speaker": speaker, "text": sp[:500]})
    lines.append({"speaker": "Host2", "text": closing})

    script = "\n".join(f"{l['speaker']}: {l['text']}" for l in lines)
    audio_path = None
    try:
        from gtts import gTTS
        fname = f"overview_{uuid.uuid4().hex[:8]}.mp3"
        out = f"data/audio/{fname}"
        tts = gTTS(text=plain_text, lang="en", slow=False)
        tts.save(out)
        audio_path = f"/audio/{fname}"
    except Exception as e:
        logger.warning(f"TTS failed: {e}")

    return jsonify({
        "script": script,
        "lines": lines,
        "audio_url": audio_path,
        "document_name": doc.get("name", ""),
    })


@app.route('/api/generate/video', methods=['POST'])
def api_generate_video():
    """Generate a narrated video with text slides from document content."""
    if not _documents:
        return jsonify({"error": "No document loaded"}), 400
    data = request.json or {}
    doc_id = data.get('doc_id', '')
    doc = _find_document(doc_id) if doc_id else list(_documents.values())[0]
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    text = doc.get("text", "")[:6000]

    try:
        summarizer = _get_summarizer()
        summary = summarizer.summarize_document(text, detail_level="detailed", focus="")
    except Exception as e:
        logger.warning(f"Summarizer failed for video: {e}")
        summary = {"overall": text[:2000], "sections": []}

    overall = summary.get("overall", "") or ""
    sections = (summary.get("sections", []) or [])[:6]

    slides_data = [{"title": doc.get("name", "Document Overview"), "content": overall[:300]}]
    for s in sections:
        slides_data.append({"title": s.get("title", "Section"), "content": s.get("summary", "")[:300]})

    vid_dir = Path("data/videos")
    vid_dir.mkdir(parents=True, exist_ok=True)
    video_id = uuid.uuid4().hex[:8]

    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        W, H = 1280, 720
        slide_images = []
        audio_files = []
        durations = []

        for i, slide in enumerate(slides_data):
            img = Image.new("RGB", (W, H), (18, 18, 36))
            draw = ImageDraw.Draw(img)

            try:
                title_font = ImageFont.truetype("arial.ttf", 48)
                body_font = ImageFont.truetype("arial.ttf", 28)
            except Exception:
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()

            title = slide.get("title", f"Slide {i+1}")
            content = slide.get("content", "")

            bbox = draw.textbbox((0, 0), title, font=title_font)
            tw = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, 60), title, fill=(99, 102, 241), font=title_font)

            y = 180
            words = content.split()
            line = ""
            for w in words:
                test = line + " " + w if line else w
                bbox = draw.textbbox((0, 0), test, font=body_font)
                if (bbox[2] - bbox[0]) > W - 120:
                    draw.text((60, y), line, fill=(200, 200, 210), font=body_font)
                    y += 40
                    line = w
                else:
                    line = test
            if line:
                draw.text((60, y), line, fill=(200, 200, 210), font=body_font)

            slide_path = vid_dir / f"{video_id}_slide_{i}.png"
            img.save(str(slide_path))
            slide_images.append(str(slide_path))

            narration = f"Slide {i+1}: {title}. {content}"
            try:
                from gtts import gTTS
                audio_path = vid_dir / f"{video_id}_audio_{i}.mp3"
                gTTS(text=narration, lang="en", slow=False).save(str(audio_path))
                audio_files.append(str(audio_path))
                try:
                    from moviepy import AudioFileClip
                    dur = AudioFileClip(str(audio_path)).duration
                    durations.append(max(dur, 4.0))
                except Exception:
                    durations.append(6.0)
            except Exception:
                audio_files.append(None)
                durations.append(6.0)

        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, TextClip

        clips = []
        for i, (img_path, dur) in enumerate(zip(slide_images, durations)):
            clip = ImageClip(img_path).with_duration(dur)
            if audio_files[i]:
                try:
                    audio = AudioFileClip(audio_files[i])
                    clip = clip.with_audio(audio)
                except Exception:
                    pass
            clips.append(clip)

        final = concatenate_videoclips(clips, method="compose")
        output_path = str(vid_dir / f"{video_id}_video.mp4")
        final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        for f in slide_images:
            try:
                os.unlink(f)
            except Exception:
                pass

        video_url = f"/videos/{video_id}_video.mp4"
        return jsonify({
            "video_url": video_url,
            "slide_count": len(slides_data),
            "document_name": doc.get("name", ""),
        })
    except ImportError as e:
        return jsonify({"error": f"Missing dependency: {e}. Install with: pip install moviepy gtts pillow"}), 500
    except Exception as e:
        logger.error(f"Video generation failed: {e}", exc_info=True)
        return jsonify({"error": f"Video generation failed: {str(e)}"}), 500


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    from feedback import submit as submit_feedback
    data = request.json or {}
    ref = submit_feedback(
        feature=data.get("feature", "other"),
        rating=data.get("rating", "bad"),
        description=data.get("description", ""),
        output_excerpt=data.get("output_excerpt", "") if data.get("include_output") else ""
    )
    return jsonify({"success": True, "reference_code": ref})


@app.route('/api/feedback/<ref>', methods=['GET'])
def api_feedback_status(ref):
    from feedback import get_by_ref
    entry = get_by_ref(ref)
    if not entry:
        return jsonify({"found": False})
    return jsonify({
        "found": True,
        "feature": entry["feature"],
        "reviewed": entry.get("is_reviewed", False),
        "submitted": entry["timestamp"]
    })


@app.route('/api/export/anki', methods=['POST'])
def api_export_anki():
    data = request.json or {}
    cards = data.get("cards", [])
    title = data.get("title", "Avam Study Deck")

    try:
        import genanki
        import random
        unique_model_id = random.randrange(1 << 30, 1 << 31)
        unique_deck_id = random.randrange(1 << 30, 1 << 31)

        model = genanki.Model(
            unique_model_id, "Avam Model",
            fields=[{"name": "Question"}, {"name": "Answer"}],
            templates=[{
                "name": "Card 1",
                "qfmt": '<div style="font-size:20px;color:#1e293b;">{{Question}}</div>',
                "afmt": '{{FrontSide}}<hr id="answer"><div style="font-size:18px;color:#3b82f6;">{{Answer}}</div>',
            }]
        )
        deck = genanki.Deck(unique_deck_id, title)
        for card in cards:
            deck.add_note(genanki.Note(model=model, fields=[card.get("front", ""), card.get("back", "")]))

        out_path = Path(f"data/{title.replace(' ', '_').lower()}.apkg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        genanki.Package(deck).write_to_file(str(out_path))
        return send_file(str(out_path), as_attachment=True, download_name=out_path.name)
    except ImportError:
        return jsonify({"error": "genanki not installed. pip install genanki"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    from config import UI_HOST, UI_PORT

    try:
        from model_loader import ModelRegistry
        threading.Thread(target=ModelRegistry.preload_all, daemon=True).start()
    except Exception as e:
        logger.warning(f"Model pre-loading not available: {e}")

    local_ip = get_local_ip()
    print("\n" + "=" * 60)
    print("AVAM SEARCH — Ready (Server-side AI)")
    print("=" * 60)
    print(f"  Local:    http://{UI_HOST}:{UI_PORT}")
    print(f"  Network:  http://{local_ip}:{UI_PORT}")
    print(f"  AI runs server-side (QAEngine, LED/BART summarizer)")
    print("=" * 60 + "\n")
    app.run(host=UI_HOST, port=UI_PORT, debug=False)
