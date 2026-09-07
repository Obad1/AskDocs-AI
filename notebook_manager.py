"""Simple notebook manager for organizing documents."""
import json
import uuid
from datetime import datetime
from pathlib import Path

NOTEBOOKS_FILE = Path("data/sessions/notebooks.json")

def _load():
    if NOTEBOOKS_FILE.exists():
        try:
            return json.loads(NOTEBOOKS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save(notebooks):
    NOTEBOOKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS_FILE.write_text(json.dumps(notebooks, indent=2, ensure_ascii=False), encoding="utf-8")

def list_notebooks():
    return _load()

def create_notebook(name):
    notebooks = _load()
    nb = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "doc_ids": [],
        "created": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        "doc_count": 0,
    }
    notebooks.append(nb)
    _save(notebooks)
    return nb

def get_notebook(nb_id):
    notebooks = _load()
    for nb in notebooks:
        if nb["id"] == nb_id:
            return nb
    return None

def delete_notebook(nb_id):
    notebooks = _load()
    notebooks = [nb for nb in notebooks if nb["id"] != nb_id]
    _save(notebooks)

def add_doc_to_notebook(nb_id, doc_id):
    notebooks = _load()
    for nb in notebooks:
        if nb["id"] == nb_id:
            if doc_id not in nb["doc_ids"]:
                nb["doc_ids"].append(doc_id)
                nb["doc_count"] = len(nb["doc_ids"])
                nb["last_modified"] = datetime.now().isoformat()
            break
    _save(notebooks)
