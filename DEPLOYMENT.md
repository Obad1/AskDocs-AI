# DEPLOYMENT GUIDE: AskDocs AI v2.0 on Render (Single Service)

## Overview
This guide deploys AskDocs AI v2.0 to [Render](https://render.com) as a **single
service** exposed at `askdocs-ai.onrender.com`.

The app runs as one process:
- **Backend**: FastAPI (`app.main`) serves the v1 API on Render's `$PORT`.
- **Frontend**: the React + Vite PWA is built to `frontend/dist/` and served by
  the same FastAPI process (`/api/v1` routes and `/exports` take precedence,
  everything else falls through to the SPA).

---

## Prerequisites
- A [Render](https://render.com) account
- The GitHub repository containing this v2.0 code

---

## Step 1: Create a Single Python Web Service

1. In Render, click **New Web Service** → **Python**
2. Repository: `your-username/AskDocs-AI` (or your fork)
3. Name: `askdocs-ai` (or any name)
4. **Runtime**: Python 3 (3.11+)
5. **Root Directory**: (leave blank — repo root; build/start reference both
   `backend/` and `frontend/`)
6. **Build Command**:
   ```
   pip install -r requirements.txt && pip install -r backend/requirements.txt && cd frontend && npm install && npm run build
   ```
   > npm MUST run inside `frontend/` — there is no `package.json` at the repo
   > root. Running `npm install && npm run build` at the root fails with
   > `ENOENT .../package.json`.
7. **Start Command**:
   ```
   bash start.sh
   ```
   > `start.sh` runs `uvicorn app.main:app --app-dir backend`. This is
   > important: running `backend.app.main:app` from the repo root resolves
   > `app` to the top-level `app.py` module instead of the `backend.app`
   > package and crashes with `ModuleNotFoundError`.
8. Click **Create Web Service**.

## Step 2: Environment / Settings

- `JOB_BROKER=sqlite`
  - The Huey background queue falls back to a file-based broker; no Redis is
    required on Render. (`backend/app/core/pipeline.py`)
- Optional **Disk** (attached at `/data`) for persistence of uploads/vector
  store. Set these so the app writes to it:
  - `DATA_ROOT=/data`
  - `VECTOR_STORE_PATH=/data/chroma`
  - `EXPORTS_DIR=/data/exports`
- Optional: `CORS_ALLOW_ORIGINS` (JSON list). Not required in production since
  the UI and API share the same origin, but useful when calling the API from
  a separate dev origin:
  ```
  CORS_ALLOW_ORIGINS=["https://askdocs-ai.onrender.com"]
  ```
- **Health Check Path**: `/api/v1/health`

## Step 3: Custom Domain

1. Go to **Settings** → **Custom Domains**.
2. Add `askdocs-ai.onrender.com` and follow Render's DNS verification
   (CNAME -> `your-service.onrender.com`).

---

## Local Development (no Render)

Build once, then run everything through the same `start.sh`:

```bash
cd frontend && npm install && npm run build && cd ..
bash start.sh            # backend on PORT (default 8000) serving API + UI
```

For hot-reload dev, run the two servers separately:

```bash
# Backend
python -m uvicorn app.main:app --app-dir backend --reload --port 8000

# Frontend (Vite dev server, proxies /api to the backend)
cd frontend
npm run dev
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Build fails `ENOENT .../package.json` | Build command must `cd frontend` before `npm install` / `npm run build` |
| Start fails `No module named 'app.api'` / `app` shadows package | Use `start.sh` (runs with `--app-dir backend`); don't start `backend.app.main:app` from the repo root |
| Build is slow at `pip install -r backend/requirements.txt` | `llama-cpp-python` compiles from source; if it exceeds Render's build time, use a slim `backend/requirements-render.txt` without it (see `backend/Dockerfile` for the native deps it needs) |
| Runtime `ImportError` for `tesseract`/`ffmpeg`/`.so` files | Those code paths need OS binaries; on the plain Python runtime they degrade gracefully. Keep `backend/Dockerfile` for a Docker-based deploy if full OCR/audio is required |
| Background jobs don't run | Ensure `JOB_BROKER=sqlite` (no Redis on Render) |
| Blank page after deploy | Confirm `frontend/dist/index.html` was produced in the build log and is being served (`curl https://askdocs-ai.onrender.com/` should return HTML) |
| Data lost on redeploy | Attach a Render Disk and set `DATA_ROOT=/data` |

---

## Success Criteria

- [x] `askdocs-ai.onrender.com` returns the PWA (`index.html`)
- [x] `askdocs-ai.onrender.com/api/v1/health` returns `{"status":"ok", ...}`
- [x] SPA deep links (e.g. `/settings`) fall back to `index.html`, not 404
- [x] One web service, one `$PORT`, no separate Node/static host required

---
*Single-service Render deployment — see `start.sh` and `backend/app/main.py`.*