"""Shared LLM call utility for Covenant Hedge Fund analysts.

Supports two backends:
  1. OpenRouter (free-tier cloud models) -- primary when OPENROUTER_API_KEY is set
  2. Ollama (local) -- fallback when OpenRouter is unavailable

Chain: OpenRouter available -> use it. Ollama available -> use it.
Both unavailable -> quant-only mode.

Model selection priority (OpenRouter):
  1. --openrouter-model CLI flag (passed via set_openrouter_model())
  2. OPENROUTER_MODEL env var
  3. Default: meta-llama/llama-3.3-70b-instruct:free

Model selection priority (Ollama):
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
import urllib.error


# Fallback string -- triggers neutral/0 in analyst parsing
_FALLBACK_RESPONSE = json.dumps({
    "signal": "neutral",
    "confidence": 0,
    "reasoning": "LLM unavailable, no backend running",
})

# Instruction suffix appended to every analyst system prompt
LLM_INSTRUCTION_SUFFIX = (
    '\n\nIMPORTANT: Output ONLY a single JSON object. No reasoning, no explanation, no thinking.\n'
    'Format: {"signal": "bullish", "confidence": 65, "reasoning": "brief reason under 180 chars"}\n'
    'signal must be "bullish", "bearish", or "neutral". confidence must be 0-100.\n'
    'Output the JSON object and NOTHING else.'
)

# ---------------------------------------------------------------------------
# Tiered model recommendations for Ollama (best first)
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
# OpenRouter configuration
# ---------------------------------------------------------------------------
_OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
_OPENROUTER_MODEL: str | None = None  # resolved lazily
_OPENROUTER_AVAILABLE: bool | None = None  # cached after first check

# Free models available on OpenRouter (no credit card needed)
OPENROUTER_FREE_MODELS: list[tuple[str, str]] = [
    ("meta-llama/llama-3.3-70b-instruct:free", "Best for financial reasoning"),
    ("qwen/qwen3-235b-a22b:free", "Largest free model"),
    ("deepseek/deepseek-r1-0528:free", "Strong reasoning"),
    ("google/gemma-3-27b-it:free", "Good general purpose"),
    ("mistralai/mistral-small-3.1-24b-instruct:free", "Fast, compact"),
]

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


def _cache_hash(system_prompt: str, user_prompt: str, model: str) -> str:
    """Compute a SHA-256 content hash for a prompt pair + model."""
    payload = (model + "\n===\n" + system_prompt + "\n---\n" + user_prompt).encode("utf-8")
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


# ---------------------------------------------------------------------------
# Model selection: Ollama
# ---------------------------------------------------------------------------

def set_model(model: str) -> None:
    """Override the Ollama model from CLI (highest priority)."""
    global _OLLAMA_MODEL
    _OLLAMA_MODEL = model


def get_active_model() -> str:
    """Return the model that will be used for LLM calls.

    Returns the OpenRouter model if available, otherwise the Ollama model.
    """
    if _check_openrouter():
        return _resolve_openrouter_model()
    return _resolve_model()


def get_active_backend() -> str:
    """Return the name of the backend that will be used ('openrouter', 'ollama', or 'none')."""
    if _check_openrouter():
        return "openrouter"
    if _check_ollama():
        return "ollama"
    return "none"


def _resolve_model() -> str:
    """Resolve which Ollama model to use, following the priority chain."""
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
                print(f"  [LLM] Auto-selected Ollama model: {recommended}", file=sys.stderr)
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


# ---------------------------------------------------------------------------
# Model selection: OpenRouter
# ---------------------------------------------------------------------------

def set_openrouter_model(model: str) -> None:
    """Override the OpenRouter model from CLI (highest priority)."""
    global _OPENROUTER_MODEL
    _OPENROUTER_MODEL = model


def _resolve_openrouter_model() -> str:
    """Resolve which OpenRouter model to use."""
    global _OPENROUTER_MODEL

    if _OPENROUTER_MODEL is not None:
        return _OPENROUTER_MODEL

    env_model = os.environ.get("OPENROUTER_MODEL")
    if env_model:
        _OPENROUTER_MODEL = env_model
        return _OPENROUTER_MODEL

    _OPENROUTER_MODEL = _OPENROUTER_DEFAULT_MODEL
    return _OPENROUTER_MODEL


# ---------------------------------------------------------------------------
# Backend availability checks
# ---------------------------------------------------------------------------

def _check_openrouter() -> bool:
    """Check if OpenRouter is configured (API key present)."""
    global _OPENROUTER_AVAILABLE
    if _OPENROUTER_AVAILABLE is not None:
        return _OPENROUTER_AVAILABLE
    _OPENROUTER_AVAILABLE = bool(_OPENROUTER_API_KEY and _OPENROUTER_API_KEY.strip())
    return _OPENROUTER_AVAILABLE


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


# ---------------------------------------------------------------------------
# LLM call: main entry point
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an LLM and return the raw text response.

    Chain:
      1. OpenRouter (free cloud models) -- if OPENROUTER_API_KEY is set
      2. Ollama (local) -- if running locally
      3. Fallback (quant-only mode)

    Args:
        system_prompt: System-level instruction (analyst philosophy).
        user_prompt: User-level content (financial facts).

    Returns:
        Raw text response from the LLM.
    """
    # Determine which model will be used (for cache key)
    if _check_openrouter():
        model = _resolve_openrouter_model()
    elif _check_ollama():
        model = _resolve_model()
    else:
        return _FALLBACK_RESPONSE

    # 0. Cache check -- return cached response if identical prompt+model seen before
    hash_key = _cache_hash(system_prompt, user_prompt, model)
    cached = _cache_lookup(hash_key)
    if cached is not None:
        return cached

    # 1. OpenRouter (free cloud models) -- primary backend
    if _check_openrouter():
        result = _call_openrouter(system_prompt, user_prompt, model)
        if result != _FALLBACK_RESPONSE:
            _cache_store(hash_key, result, model)
            return result

    # 2. Ollama (free, local) -- fallback backend
    if _check_ollama():
        ollama_model = _resolve_model()
        ollama_hash = _cache_hash(system_prompt, user_prompt, ollama_model)
        cached_ollama = _cache_lookup(ollama_hash)
        if cached_ollama is not None:
            return cached_ollama
        result = _call_ollama(system_prompt, user_prompt)
        if result != _FALLBACK_RESPONSE:
            _cache_store(ollama_hash, result, ollama_model)
            return result

    # 3. Fallback -- quant-only mode (no LLM analysts)
    print("  [LLM] No backend available -- quant-only mode", file=sys.stderr)
    return _FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------

def _call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> str:
    """Call OpenRouter's OpenAI-compatible chat completions API.

    Uses urllib.request (stdlib) -- no external dependencies required.
    Includes retry with exponential backoff on 429 (rate limit) responses.

    Args:
        system_prompt: System-level instruction.
        user_prompt: User-level content.
        model: OpenRouter model identifier. Defaults to _resolve_openrouter_model().
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum response tokens.

    Returns:
        Raw text response, or _FALLBACK_RESPONSE on failure.
    """
    global _OPENROUTER_AVAILABLE

    api_key = _OPENROUTER_API_KEY
    if not api_key:
        return _FALLBACK_RESPONSE

    if model is None:
        model = _resolve_openrouter_model()

    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/covenant-hedge-fund",
        "X-Title": "Covenant Hedge Fund",
    }

    max_retries = 6
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                _OPENROUTER_BASE_URL,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return content or _FALLBACK_RESPONSE

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited -- back off with longer delays for free tier
                wait = min(5 * (2 ** attempt), 60)  # 5s, 10s, 20s, 40s, 60s, 60s
                print(f"  [LLM] OpenRouter rate limited (429), "
                      f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            elif e.code == 401:
                # Bad API key -- disable OpenRouter for this process
                print("  [LLM] OpenRouter auth failed (401) -- "
                      "check OPENROUTER_API_KEY", file=sys.stderr)
                _OPENROUTER_AVAILABLE = False
                return _FALLBACK_RESPONSE
            else:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                print(f"  [LLM] OpenRouter HTTP {e.code}: {err_body}",
                      file=sys.stderr)
                if attempt == max_retries - 1:
                    return _FALLBACK_RESPONSE
                time.sleep(1)

        except Exception as e:
            print(f"  [LLM] OpenRouter error: {e}", file=sys.stderr)
            if attempt == max_retries - 1:
                return _FALLBACK_RESPONSE
            time.sleep(1)

    return _FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

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
                max_tokens=512,
                temperature=0.1,  # very low temp for reliable JSON output
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
