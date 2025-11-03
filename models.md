# Models Setup (Offline)

AskDocs-AI runs fully offline by default. Models are stored under `./models/`.

## Required Models

- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Generation model: `google/flan-t5-base` (Hugging Face Transformers)

## One-command Download

```bash
python manage.py download-models
```

This downloads models into `./models/` and configures the app to use them offline.

## Manual Download (Optional)

You can manually download models using `huggingface_hub` and place them under `./models/<org>__<name>/`.

## Switching Models

Change the following in `.env` (optional):

```
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
MODELS_DIR=./models
```

## Using Ollama (Optional)

Set `ALLOW_CLOUD=true` only if you want to enable any online integrations. By default, the app stays offline.
