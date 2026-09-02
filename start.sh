#!/bin/sh
# start.sh — starts v2.0 AskDocs AI locally (not for Render production)
#
# This is for LOCAL DEVELOPMENT only. For Render deployment, see DEPLOYMENT.md.
#
# It starts both the Backend (FastAPI) and Frontend (Vite dev server)
# so you can test locally at http://localhost:5173 with API proxy to localhost:$PORT.

# 1. Start Backend (FastAPI) on port 8000 (default local port)
python -m uvicorn backend.app.main:app \
  --host 0.0.0.0 --port 8000 \
  > /dev/null 2>&1 &
BACKEND_PID=$!
echo "Backend started on port 8000 (PID $BACKEND_PID)"

# 2. Start Frontend (Vite dev server) on port 5173
#    Vite proxies /api to the backend via the VITE_BACKEND_URL env var
export VITE_BACKEND_URL="http://localhost:8000"
cd frontend
npm run dev -- --port 5173 > /dev/null 2>&1 &
FRONTEND_PID=$!
echo "Frontend started on port 5173 (PID $FRONTEND_PID)"

# 3. Wait for both processes
echo "========================================="
echo " v2.0 AskDocs AI is running locally"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000/api"
echo "========================================="
wait $BACKEND_PID $FRONTEND_PID