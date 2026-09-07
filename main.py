#!/usr/bin/env python
"""Avam Search — consolidated entry point.

Usage:
    python main.py              # Launch web app (default)
    python main.py --mode web   # Launch web app
    python main.py --mode server  # Launch API server
"""
import sys
import argparse
import threading
from pathlib import Path


def check_python():
    if sys.version_info < (3, 10):
        print(f"Error: Python 3.10+ required. You have {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)


def create_directories():
    for d in ["data/documents", "data/sessions", "data/vector_db", "data/generated", "models"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def start_preloading():
    try:
        from model_loader import ModelRegistry
        threading.Thread(target=ModelRegistry.preload_all, daemon=True).start()
    except Exception:
        pass


def launch_web(host, port):
    from webapp import app as flask_app
    from webapp import get_local_ip, logger

    start_preloading()
    local_ip = get_local_ip()
    print("\n" + "=" * 60)
    print("  AVAM SEARCH — Web App")
    print("=" * 60)
    print(f"  Local:    http://{host}:{port}")
    print(f"  Network:  http://{local_ip}:{port}")
    print("=" * 60 + "\n")
    flask_app.run(host=host, port=port, debug=False)


def launch_server(host, port):
    import uvicorn
    print("\n" + "=" * 60)
    print("  AVAM SEARCH — API Server")
    print("=" * 60)
    print(f"  Local:    http://{host}:{port}")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Avam Search")
    parser.add_argument("--mode", choices=["web", "server"], default="web",
                        help="Launch mode: web (Flask) or server (API) (default: web)")
    parser.add_argument("--host", default=None, help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--offline", action="store_true", help="Skip HuggingFace downloads, cached models only")

    args = parser.parse_args()

    if args.offline:
        import os
        os.environ["HF_LOCAL_ONLY"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"

    check_python()
    create_directories()

    from config import UI_HOST, UI_PORT

    host = args.host or UI_HOST
    port = args.port or UI_PORT

    if args.mode == "web":
        launch_web(host, port)
    elif args.mode == "server":
        launch_server(host, port)


if __name__ == "__main__":
    main()
