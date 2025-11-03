"""Configuration management with environment variable support."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DOCUMENTS_DIR = DATA_DIR / "documents"
SESSIONS_DIR = DATA_DIR / "sessions"
VECTOR_DB_PATH = DATA_DIR / os.getenv("VECTOR_DB_PATH", "vector_db")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True, parents=True)
SESSIONS_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_DB_PATH.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

# Optional API keys (not required for local operation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# UI configuration
UI_TITLE = os.getenv("UI_TITLE", "AskDocs AI")
UI_PORT = int(os.getenv("UI_PORT", 7860))
UI_DEBUG = os.getenv("UI_DEBUG", "false").lower() == "true"

# Offline/Cloud modes
ALLOW_CLOUD = os.getenv("ALLOW_CLOUD", "false").lower() == "true"
OFFLINE_MODE = not ALLOW_CLOUD

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

# Retrieval configuration
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

