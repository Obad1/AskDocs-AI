"""FastAPI server with optional webhook endpoint and health checks."""
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    payload = await request.json()
    # Process webhook payload; currently echo
    return JSONResponse({"received": True, "token": token, "payload_keys": list(payload.keys())})


def run_webhook():
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run_webhook()


