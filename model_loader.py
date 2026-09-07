import threading
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ModelRegistry:
    _models: dict = {}
    _status: dict = {}
    _all_ready: bool = False

    @classmethod
    def preload_all(cls):
        tasks = {
            "embeddings":    cls._load_embeddings,
            "summarizer":    cls._load_summarizer,
            "qa":            cls._load_qa,
            "qg":            cls._load_qg,
            "ner":           cls._load_ner,
        }
        threads = []
        for name, loader in tasks.items():
            cls._status[name] = "loading"
            t = threading.Thread(target=cls._run, args=(name, loader), daemon=True)
            t.start()
            threads.append(t)

        def await_all():
            for t in threads:
                t.join()
            cls._all_ready = True
            logger.info("[Models] All models loaded — AskDocs AI ready")

        threading.Thread(target=await_all, daemon=True).start()

    @classmethod
    def _run(cls, name: str, loader):
        try:
            logger.info(f"[Models] Loading {name}...")
            cls._models[name] = loader()
            cls._status[name] = "ready"
            logger.info(f"[Models] {name} ready")
        except Exception as e:
            logger.error(f"[Models] {name} failed: {e}")
            cls._models[name] = None
            cls._status[name] = "failed"

    @classmethod
    def _load_embeddings(cls):
        return SentenceTransformer("BAAI/bge-small-en-v1.5")

    @classmethod
    def _load_summarizer(cls):
        tok = AutoTokenizer.from_pretrained("pszemraj/led-base-book-summary")
        model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-base-book-summary")
        model.eval()
        def summarize(text, **kw):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=16384)
            ids = model.generate(inputs.input_ids, max_length=kw.get("max_len", 512), num_beams=4, early_stopping=True)
            return tok.decode(ids[0], skip_special_tokens=True)
        return summarize

    @classmethod
    def _load_qa(cls):
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
            available = [m["name"] for m in resp.json().get("models", [])]
            for preferred in ["mistral:7b-instruct", "phi3:mini", "phi3", "phi", "llama2"]:
                if any(preferred in n for n in available):
                    logger.info(f"[Models] Using Ollama model: {preferred}")
                    return {"backend": "ollama", "model": preferred}
        except Exception:
            pass
        logger.info("[Models] Ollama not found — loading flan-t5-large")
        tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        model.eval()
        def answer(question, context, **kw):
            text = f"question: {question} context: {context}"
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
            ids = model.generate(inputs.input_ids, max_length=kw.get("max_len", 256), num_beams=4, early_stopping=True)
            return tok.decode(ids[0], skip_special_tokens=True)
        return {"backend": "huggingface", "model": answer}

    @classmethod
    def _load_qg(cls):
        tok = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
        model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
        return {"tokenizer": tok, "model": model}

    @classmethod
    def _load_ner(cls):
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=-1)

    @classmethod
    def get(cls, name: str):
        return cls._models.get(name)

    @classmethod
    def is_ready(cls) -> bool:
        return cls._all_ready

    @classmethod
    def status(cls) -> dict:
        return {"ready": cls._all_ready, "models": cls._status}
