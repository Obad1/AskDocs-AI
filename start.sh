#!/bin/sh
# start.sh — starts the full AskDocs AI app with one command.
#
# Production (Render): the FastAPI backend binds $PORT and serves BOTH the
# /api routes and the built frontend from frontend/dist, so a single process
# boots the whole app. Build step (Run via the Render dashboard):
#
#   pip install -r requirements.txt && pip install -r backend/requirements.txt \
#     && cd frontend && npm install && npm run build
#
# Local dev: build the frontend first (npm run build), then run this script;
# or use `npm run dev` + uvicorn separately (see DEPLOYMENT.md).

# Resolve backend as the app root so `from app...` imports resolve to the
# real package (running `backend.app.main:app` from the repo root shadows it
# with the top-level app.py module).
exec python -m uvicorn app.main:app \
  --app-dir backend \
  --host 0.0.0.0 \
  --port "${PORT:-8000}"