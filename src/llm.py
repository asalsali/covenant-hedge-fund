"""Shared LLM call utility for Covenant Hedge Fund analysts.

Supports Ollama (local) as the sole LLM backend.
Chain: Ollama available -> use it. Ollama unavailable -> quant-only mode.
No paid API dependencies.

Model selection priority:
  1. --model CLI flag (passed via set_model())
  2. OLLAMA_MODEL env var
  3. Auto-detect: pick the best model already pulled in Ollama
  4. Fallback: qwen2.5:7b-instruct (default)
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.request


# Fallback string -- triggers neutral/0 in analyst parsing
_FALLBACK_RESPONSE = json.dumps({
    "signal": "neutral",
    "confidence": 0,
    "reasoning": "LLM unavailable, Ollama not running",
})

# Instruction suffix appended to every analyst system prompt
LLM_INSTRUCTION_SUFFIX = (
    '\n\nYou must respond with EXACTLY this JSON format:\n'
    '{"signal": "bullish" or "bearish" or "neutral", '
    '"confidence": 0-100, '
    '"reasoning": "your reasoning in under 180 characters"}\n'
    'Respond with ONLY the JSON, no other text.'
)

# ---------------------------------------------------------------------------
# Tiered model recommendations (best first)
# ---------------------------------------------------------------------------
# Each tuple: (model_name, approx_vram_gb, tier_label)
MODEL_TIERS: list[tuple[str, float, str]] = [
    ("llama3.3:70b-instruct", 42.0, "Tier 4 (48GB+ VRAM)"),
    ("qwen2.5:32b-instruct",  20.0, "Tier 3 (24GB VRAM)"),
    ("phi4:14b",               9.0, "Tier 2 (16GB VRAM)"),
    ("qwen2.5:7b-instruct",   4.7, "Tier 1 (8GB VRAM)"),
]

# Ordered list of recommended model names (best first) for auto-selection
_RECOMMENDED_MODELS = [name for name, _, _ in MODEL_TIERS]

_DEFAULT_MODEL = "qwen2.5:7b-instruct"

# Ollama configuration
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL: str | None = None  # resolved lazily
_OLLAMA_AVAILABLE: bool | None = None  # cached after first check
_OLLAMA_PULLED_MODELS: list[str] | None = None  # cached after first query

# ---------------------------------------------------------------------------
# Content-hash LLM response caching
# ---------------------------------------------------------------------------
_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cache" / "llm"
_CACHE_ENABLED: bool = True  # toggled off by --no-cache
_CACHE_HITS: int = 0
_CACHE_MISSES: int = 0


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable the disk-based LLM response cache."""
    global _CACHE_ENABLED
    _CACHE_ENABLED = enabled


def get_cache_stats() -> dict[str, int]:
    """Return cache hit/miss counts for the current process."""
    return {"hits": _CACHE_HITS, "misses": _CACHE_MISSES}


def _cache_hash(system_prompt: str, user_prompt: str) -> str:
    """Compute a SHA-256 content hash for a prompt pair."""
    payload = (system_prompt + "\n---\n" + user_prompt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_lookup(hash_key: str) -> str | None:
    """Return the cached LLM response for *hash_key*, or None."""
    global _CACHE_HITS, _CACHE_MISSES
    if not _CACHE_ENABLED:
        return None
    path = _CACHE_DIR / f"{hash_key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _CACHE_HITS += 1
            return data.get("response")
        except (json.JSONDecodeError, KeyError):
            pass
    _CACHE_MISSES += 1
    return None


def _cache_store(hash_key: str, response: str, model: str) -> None:
    """Persist an LLM response to the disk cache."""
    if not _CACHE_ENABLED:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    entry = {
        "hash": hash_key,
        "model": model,
        "response": response,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _CACHE_DIR / f"{hash_key}.json"
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def set_model(model: str) -> None:
    """Override the model from CLI (highest priority)."""
    global _OLLAMA_MODEL
    _OLLAMA_MODEL = model


def get_active_model() -> str:
    """Return the model that will be used for LLM calls."""
    return _resolve_model()


def _resolve_model() -> str:
    """Resolve which model to use, following the priority chain."""
    global _OLLAMA_MODEL

    # Already resolved
    if _OLLAMA_MODEL is not None:
        return _OLLAMA_MODEL

    # Priority 2: env var
    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        _OLLAMA_MODEL = env_model
        return _OLLAMA_MODEL

    # Priority 3: auto-detect from pulled models
    pulled = _get_pulled_models()
    if pulled:
        # Check recommended models in priority order (best first)
        for recommended in _RECOMMENDED_MODELS:
            if recommended in pulled:
                _OLLAMA_MODEL = recommended
                print(f"  [LLM] Auto-selected model: {recommended}", file=sys.stderr)
                return _OLLAMA_MODEL

        # No recommended model found -- use first available model
        fallback = pulled[0]
        _OLLAMA_MODEL = fallback
        print(f"  [LLM] No recommended model found, using: {fallback}",
              file=sys.stderr)
        return _OLLAMA_MODEL

    # Priority 4: default
    _OLLAMA_MODEL = _DEFAULT_MODEL
    return _OLLAMA_MODEL


def _get_pulled_models() -> list[str]:
    """Query Ollama /api/tags for locally available models. Cached."""
    global _OLLAMA_PULLED_MODELS
    if _OLLAMA_PULLED_MODELS is not None:
        return _OLLAMA_PULLED_MODELS

    _OLLAMA_PULLED_MODELS = []
    try:
        req = urllib.request.Request(
            f"{_OLLAMA_BASE_URL}/api/tags", method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                _OLLAMA_PULLED_MODELS = [
                    m["name"] for m in data.get("models", [])
                ]
    except Exception:
        pass
    return _OLLAMA_PULLED_MODELS


def _check_ollama() -> bool:
    """Check if Ollama is running by hitting its tags endpoint."""
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is not None:
        return _OLLAMA_AVAILABLE
    # _get_pulled_models hits /api/tags -- if it found models, Ollama is up
    pulled = _get_pulled_models()
    if pulled:
        _OLLAMA_AVAILABLE = True
        return True
    # Empty list could mean Ollama is up but no models pulled, or it's down.
    # _get_pulled_models sets the global to [] on both cases. Check directly.
    try:
        req = urllib.request.Request(
            f"{_OLLAMA_BASE_URL}/api/tags", method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            _OLLAMA_AVAILABLE = resp.status == 200
    except Exception:
        _OLLAMA_AVAILABLE = False
    return _OLLAMA_AVAILABLE


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an LLM and return the raw text response.

    Chain: Ollama available -> use it. Ollama unavailable -> fallback (quant-only).
    No paid API keys. No middle step.

    Args:
        system_prompt: System-level instruction (analyst philosophy).
        user_prompt: User-level content (financial facts).

    Returns:
        Raw text response from the LLM.
    """
    # 0. Cache check -- return cached response if identical prompt seen before
    hash_key = _cache_hash(system_prompt, user_prompt)
    cached = _cache_lookup(hash_key)
    if cached is not None:
        return cached

    # 1. Ollama (free, local) -- sole LLM backend
    if _check_ollama():
        result = _call_ollama(system_prompt, user_prompt)
        if result != _FALLBACK_RESPONSE:
            _cache_store(hash_key, result, _resolve_model())
            return result

    # 2. Fallback -- quant-only mode (no LLM analysts)
    if not _check_ollama():
        print("  [LLM] Ollama not available -- quant-only mode", file=sys.stderr)
    return _FALLBACK_RESPONSE


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call Ollama via its OpenAI-compatible API with retry.

    Uses the openai library pointed at Ollama's local endpoint.
    No API key required -- Ollama runs locally for free.

    If Ollama returns a CUDA or OOM error, marks Ollama as unavailable
    for the remainder of the process and falls back to quant-only mode.
    """
    global _OLLAMA_AVAILABLE

    try:
        import openai
    except ImportError:
        return _FALLBACK_RESPONSE

    model = _resolve_model()

    client = openai.OpenAI(
        base_url=f"{_OLLAMA_BASE_URL}/v1",
        api_key="ollama",  # Ollama ignores this but openai lib requires it
        timeout=120.0,  # local models can be slow on first load
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=256,
                temperature=0.3,  # lower temp for more consistent JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or _FALLBACK_RESPONSE
        except Exception as e:
            err_msg = str(e).lower()
            # CUDA crash or OOM -- disable Ollama for this process
            if "cuda" in err_msg or "out of memory" in err_msg or "terminated" in err_msg:
                _OLLAMA_AVAILABLE = False
                return _FALLBACK_RESPONSE
            if attempt == 0:
                time.sleep(2)
            else:
                return _FALLBACK_RESPONSE

    return _FALLBACK_RESPONSE
