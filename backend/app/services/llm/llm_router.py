"""Unified LLM service with provider failover (blueprint §4, Phase 2).

Call chain: Cerebras -> Groq -> OpenRouter. A provider without a configured API
key is skipped entirely; HTTP 429 (rate limit) and 5xx responses trigger the
next provider. Talks to each provider's OpenAI-compatible ``/chat/completions``
endpoint over stdlib ``urllib``.

Configure via env: ``CEREBRAS_API_KEY``, ``GROQ_API_KEY``, ``OPENROUTER_API_KEY``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMProvider:
    name: str
    env_var: str
    base_url: str
    default_model: str
    api_key: str = ""


def _providers_from_settings() -> List[LLMProvider]:
    s = get_settings()
    return [
        LLMProvider(
            "cerebras",
            "CEREBRAS_API_KEY",
            "https://api.cerebras.ai/v1/chat/completions",
            "llama-3.3-70b",
            s.cerebras_api_key,
        ),
        LLMProvider(
            "groq",
            "GROQ_API_KEY",
            "https://api.groq.com/openai/v1/chat/completions",
            "llama-3.1-70b-versatile",
            s.groq_api_key,
        ),
        LLMProvider(
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1/chat/completions",
            "meta-llama/llama-3.1-8b-instruct:free",
            s.openrouter_api_key,
        ),
    ]


def _chat_completion(
    provider: LLMProvider,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    model: Optional[str],
    timeout: int = 120,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model or provider.default_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = request.Request(
        provider.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:  # noqa: BLE001
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:200]
        except Exception:  # noqa: BLE001
            pass
        msg = f"{provider.name} HTTP {exc.code}: {detail}"
        if 400 <= exc.code < 500 and exc.code != 429:
            # 4xx (other than rate limit) is a config bug — do not cascade.
            raise RuntimeError(msg)
        raise CrashError(provider.name, msg)  # 429 / 5xx -> try next provider
    except error.URLError as exc:  # noqa: BLE001
        raise CrashError(provider.name, f"{provider.name} unreachable: {exc.reason}")

    content = ""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise CrashError(provider.name, f"{provider.name} empty completion")
    return {
        "text": content.strip(),
        "provider": provider.name,
        "model": str(data.get("model") or model or provider.default_model),
    }


class CrashError(RuntimeError):
    """Provider failed transiently (429/5xx/network) — safe to fail over."""


def generate(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the LLM with automatic Cerebras -> Groq -> OpenRouter failover."""
    providers = [p for p in _providers_from_settings() if p.api_key]
    if not providers:
        raise RuntimeError(
            "No LLM provider API keys configured. Set CEREBRAS_API_KEY, "
            "GROQ_API_KEY, and/or OPENROUTER_API_KEY in the environment."
        )
    last: Optional[Error] = None
    for provider in providers:
        try:
            return _chat_completion(
                provider, messages, temperature, max_tokens, model=model
            )
        except CrashError as exc:
            last = exc
            logger.warning("LLM failover: %s", exc)
            continue
        except Exception as exc:  # noqa: BLE001 - config-bug 4xx
            last = exc
            break
    raise RuntimeError(
        f"All LLM providers failed. Last error: {last}. "
        "Check API keys / rate limits."
    )


def configured_providers() -> List[str]:
    return [p.name for p in _providers_from_settings() if p.api_key]