"""Model registry & hardware-tier selection (Z §7.4).

Maps a :class:`HARDWARE_TIER` to concrete local model identifiers for each
engine (embedding / llm / tts / stt) and tracks cached availability by
inspecting the Ollama model store and the local HuggingFace cache.

Key invariant: a ``TIER3_HIGH`` workspace may NEVER select the
``WEBGPU_WEBLLM`` backend (it has local GPU/CPU resources and must stay
fully on-device on the backend).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.models.domain import ENGINE_BACKEND, HARDWARE_TIER


# §5 model tables ----------------------------------------------------------- #
@dataclass
class ModelSet:
    embedding: str
    llm: str
    tts: Optional[str] = None
    stt: Optional[str] = None
    backend: ENGINE_BACKEND = ENGINE_BACKEND.OLLAMA


TIER_MODELS: dict[HARDWARE_TIER, ModelSet] = {
    HARDWARE_TIER.TIER1_LOW: ModelSet(
        embedding="nomic-embed-text",
        llm="llama3.2:3b",
        tts="tts-1-local",
        stt="tiny",
        backend=ENGINE_BACKEND.OLLAMA,
    ),
    HARDWARE_TIER.TIER2_MID: ModelSet(
        embedding="nomic-embed-text",
        llm="llama3.1:8b",
        tts="tts-1-local",
        stt="base",
        backend=ENGINE_BACKEND.OLLAMA,
    ),
    HARDWARE_TIER.TIER3_HIGH: ModelSet(
        embedding="nomic-embed-text",
        llm="llama3.1:70b",
        tts="tts-1-local",
        stt="small",
        backend=ENGINE_BACKEND.OLLAMA,
    ),
}

# WebGPU/WebLLM is a browser-side option only; never forced for high tier.
WEBGPU_FORBIDDEN_TIERS = {HARDWARE_TIER.TIER3_HIGH}


@dataclass
class ModelStatus:
    model_id: str
    available: bool = False
    backend: ENGINE_BACKEND = ENGINE_BACKEND.OLLAMA
    source: str = ""  # "ollama" | "hf_cache" | "missing"


@dataclass
class RegistryState:
    tier: HARDWARE_TIER = HARDWARE_TIER.TIER2_MID
    backend: ENGINE_BACKEND = ENGINE_BACKEND.OLLAMA
    models: dict[str, ModelStatus] = field(default_factory=dict)


class ModelRegistry:
    """Central, process-wide model registry and availability tracker."""

    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self._state = RegistryState()

    # -- tier selection --------------------------------------------------- #
    def select_model_tier(
        self, tier: HARDWARE_TIER, prefer_webgpu: bool = False
    ) -> ModelSet:
        """Return the :class:`ModelSet` for ``tier``.

        Enforces Z §7.4: a ``TIER3_HIGH`` tier never uses the
        ``WEBGPU_WEBLLM`` backend even if requested.
        """
        if tier in WEBGPU_FORBIDDEN_TIERS and prefer_webgpu:
            # Invariant violation -> refuse and fall back to Ollama.
            prefer_webgpu = False
        models = TIER_MODELS[tier]
        if prefer_webgpu and tier not in WEBGPU_FORBIDDEN_TIERS:
            models = ModelSet(
                embedding=models.embedding,
                llm=models.llm,
                tts=models.tts,
                stt=models.stt,
                backend=ENGINE_BACKEND.WEBGPU_WEBLLM,
            )
        return models

    def apply_tier(self, tier: HARDWARE_TIER, prefer_webgpu: bool = False) -> ModelSet:
        models = self.select_model_tier(tier, prefer_webgpu)
        self._state.tier = tier
        self._state.backend = models.backend
        for role, mid in (
            ("embedding", models.embedding),
            ("llm", models.llm),
            ("tts", models.tts),
            ("stt", models.stt),
        ):
            if mid:
                self._state.models[role] = ModelStatus(
                    model_id=mid, backend=models.backend
                )
        return models

    # -- availability probing (no network) ------------------------------- #
    def _ollama_store_path(self) -> Optional[Path]:
        env = os.environ.get("OLLAMA_MODELS")
        if env:
            return Path(env)
        # Common default locations
        candidates = [
            Path.home() / ".ollama" / "models",
            Path("/root/.ollama/models"),
            Path("/usr/share/ollama/.ollama/models"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _hf_cache_path(self) -> Optional[Path]:
        env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if env:
            return Path(env)
        return Path.home() / ".cache" / "huggingface"

    def check_availability(self) -> dict[str, ModelStatus]:
        """Probe local stores for every registered model id."""
        ollama = self._ollama_store_path()
        hf = self._hf_cache_path()
        for role, status in self._state.models.items():
            found_ollama = bool(ollama and self._name_in_ollama(ollama, status.model_id))
            found_hf = bool(hf and self._name_in_hf(hf, status.model_id))
            if found_ollama:
                status.available, status.source = True, "ollama"
            elif found_hf:
                status.available, status.source = True, "hf_cache"
            else:
                status.available, status.source = False, "missing"
        return dict(self._state.models)

    @staticmethod
    def _name_in_ollama(store: Path, model_id: str) -> bool:
        # Ollama stores models under blobs + manifests/<namespace>/<name>
        safe = model_id.replace(":", "-")
        for pattern in (f"*{safe}*", f"*{model_id}*"):
            if any(store.glob(pattern)):
                return True
        return False

    @staticmethod
    def _name_in_hf(cache: Path, model_id: str) -> bool:
        safe = model_id.replace("/", "--")
        return any(cache.glob(f"**/*{safe}*"))

    def status_dict(self) -> dict:
        return {
            "tier": self._state.tier.value,
            "backend": self._state.backend.value,
            "models": {
                role: {
                    "model_id": s.model_id,
                    "available": s.available,
                    "backend": s.backend.value,
                    "source": s.source,
                }
                for role, s in self._state.models.items()
            },
        }


# Module-level singleton
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
