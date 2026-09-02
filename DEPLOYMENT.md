# DEPLOYMENT GUIDE: AskDocs AI v2.0 on Render with Custom Domain

## Overview
This guide deployes AskDocs AI v2.0 to [Render](https://render.com) with the custom domain `askdocs-ai.onrender.com`.

The app consists of two services:
- **Backend**: FastAPI service (`app.main`) — handles AI inference, document processing, etc.
- **Frontend**: React + Vite PWA (`frontend/dist/`) — user interface, document viewer, study tools

Both services share the custom domain `askdocs-ai.onrender.com`.

---

## Prerequisites
- A [Render](https://render.com) account
- The GitHub repository containing this v2.0 code

---

## Step 1: Create Python Backend Service

1. In Render, click **New Web Service** → **Python**
2. Repository: `your-username/AskDocs-AI` (or your fork)
3. Name: `askdocs-ai-backend` (or any name)
4. **Environment**: Python 3
5. **Build Command**: 
   ```
   pip install -r backend/requirements.txt
   ```
6. **Start Command**:
   ```
   python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
   ```
7. Click **Create Web Service**

8. After creation, go to **Settings** → **Environment** and add:
   - Key: `VITE_BACKEND_URL`
   - Value: `https://askdocs-ai-backend.onrender.com` (or the actual backend URL once deployed)

9. Go to **Settings** → **Custom Domains** and add:
   - Domain: `askdocs-ai-backend.onrender.com`
   - Follow Render's DNS verification steps

## Step 2: Create Static Site Frontend Service

1. In Render, click **New Static Site** → **Python**
2. Name: `askdocs-ai-frontend` (or any name)
3. **Environment**: Python (selected automatically for `pip install`)
4. **Build Command**:
   ```
   npm install && npm run build
   ```
5. **Publish Directory**:
   ```
   frontend/dist
   ```
6. **Root Directory**: Leave blank (or set to `./frontend`)
7. Click **Create Static Site**

8. After creation, go to **Settings** → **Environment** and add:
   - Key: `VITE_BACKEND_URL`
   - Value: `https://askdocs-ai-backend.onrender.com`
     - This tells the frontend where to send API requests

9. Go to **Settings** → **Custom Domains** and add:
   - Domain: `askdocs-ai.onrender.com`
   - Follow Render's DNS verification (add a CNAME record pointing to `your-site.static.render.com`)

## Step 3: Configure API Proxy (Frontend → Backend)

The frontend `vite.config.ts` already includes a proxy configuration that routes `/api` requests to `VITE_BACKEND_URL`.

When a user visits `askdocs-ai.onrender.com`:
- `http://askdocs-ai.onrender.com/documents` → served from static files
- `http://askdocs-ai.onrender.com/api/v1/documents` → proxied to `https://askdocs-ai-backend.onrender.com/api/v1/documents`

The `vite.config.ts` already has:
```ts
proxy: {
  "/api": {
    target: process.env.VITE_BACKEND_URL || "https://askdocs-ai-backend.onrender.com",
    changeOrigin: true,
    secure: false,
  },
}
```

## Step 4: Verify Deployment

1. Visit `askdocs-ai.onrender.com` — you should see the AskDocs AI PWA interface
2. Try uploading a document, asking a question, or using other features
3. Check the backend logs at `https://askdocs-ai-backend.onrender.com/logs` if needed

## Alternative: Single Service Approach

If you prefer a single Render service (instead of two separate services):

1. Create **one Python service** on Render
2. Use the `start.sh` script (modified for Render)
3. The service would need to serve both the API AND the static frontend

However, the **two-service approach** (Steps 1-4) is more reliable and has better separation of concerns.

---

## Local Development (without Render)

If you want to run locally instead of deploying:

```bash
# Start backend
cd askdocs-ai-backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start frontend (with API proxy)
cd frontend
export VITE_BACKEND_URL="http://localhost:8000"
npm run dev -- --port 5173
```

Then visit `http://localhost:5173` for the full app.

---

## File Changes Made

These files were modified for Render compatibility:

| File | Purpose |
|---|---|
| `frontend/vite.config.ts` | Proxy `/api` uses `VITE_BACKEND_URL` env var; production fallback to `https://askdocs-ai-backend.onrender.com` |
| `start.sh` | Local development script; documents the dev workflow |
| `README.md` *(recommended)* | Document the deployment setup |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Frontend shows blank page or API errors | Check that `VITE_BACKEND_URL` is set correctly in the Static Site settings |
| 404 on API routes | Ensure the backend service is running and the URL matches |
| Custom domain not working | Complete Render's DNS verification steps (add CNAME record) |
| Build fails | Ensure `npm install && npm run build` works locally first |
| Backend crashes | Check Render logs for import errors or missing dependencies |

---

## Render Dashboard Quick Links

- **Backend Service**: `https://dashboard.render.com/web/svc-askdocs-ai-backend`
- **Frontend Service**: `https://dashboard.render.com/web/svc-askdocs-ai-frontend`
- **Custom Domains**: `https://dashboard.render.com/web/svc-askdocs-ai-frontend/settings/custom-domains`

---

## Success Criteria

✅ `askdocs-ai.onrender.com` loads the PWA interface  
✅ Document upload, chat, and study features work  
✅ API calls are proxied to the backend without CORS errors  
✅ Custom domain renewal/verification remains valid  

---
*Generated for AskDocs AI v2.0 — See `vite.config.ts` and `start.sh` for configuration details.*