"""Local model manager to ensure offline models are available."""
import logging
from pathlib import Path
from typing import Dict
import subprocess

from config import MODELS_DIR, EMBEDDING_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage local models for embeddings and generation."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

    def ensure_models(self) -> Dict[str, str]:
        """Ensure required models are available locally.

        Returns a dict with model names and resolved paths.
        """
        models_info = {}
        models_info["embedding"] = self._ensure_hf_model(EMBEDDING_MODEL)
        models_info["llm"] = self._ensure_hf_model(LLM_MODEL)
        return models_info

    def _ensure_hf_model(self, model_name: str) -> str:
        """Download a Hugging Face model to MODELS_DIR if not present."""
        target_path = self.models_dir / model_name.replace("/", "__")
        if target_path.exists():
            return str(target_path)

        try:
            target_path.mkdir(parents=True, exist_ok=True)
            # Use huggingface-cli to download model files for offline use
            subprocess.check_call([
                "python", "-m", "huggingface_hub", "download",
                model_name, "--local-dir", str(target_path)
            ])
            logger.info(f"Downloaded model '{model_name}' to {target_path}")
            return str(target_path)
        except Exception as e:
            logger.warning(f"Could not download model '{model_name}': {e}. Will fetch on first use if online.")
            return model_name


