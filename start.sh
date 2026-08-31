#!/bin/sh
# start.sh — runs v2.0 AskDocs AI Frontend + Backend with one command on Render
#
# Frontend (React + Vite) proxies /api to Backend (FastAPI)
# Frontend UI:        http://localhost:5173
# Backend API:        http://localhost:$PORT/api/v1/...

# 1. Start Backend (FastAPI) on Render's $PORT
python -m uvicorn backend.app.main:app \
  --host 0.0.0.0 --port "$PORT" \
  > /dev/null 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID $BACKEND_PID) on port $PORT"

# 2. Set VITE_BACKEND_URL so Vite can proxy /api correctly
export VITE_BACKEND_URL="http://localhost:$PORT"

# 3. Start Frontend (Vite dev server) on port 5173
#    Vite will proxy /api to the backend via VITE_BACKEND_URL
cd frontend
npm run dev -- --port 5173 > /dev/null 2>&1 &
FRONTEND_PID=$!
echo "Frontend started (PID $FRONTEND_PID) on port 5173"

# 4. Wait for both processes
echo "========================================="
echo " v2.0 AskDocs AI is running"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:$PORT/api"
echo "========================================="
wait $BACKEND_PID $FRONTEND_PID