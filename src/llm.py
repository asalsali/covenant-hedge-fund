"""Shared LLM call utility for Covenant Hedge Fund analysts.

Supports Ollama (local), Anthropic (Claude), and OpenAI APIs.
Priority chain: Ollama (free, local) -> Anthropic -> OpenAI -> fallback.
If no LLM is available, returns a graceful fallback that produces
neutral/0 signals.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request


# Fallback string -- triggers neutral/0 in analyst parsing
_FALLBACK_RESPONSE = json.dumps({
    "signal": "neutral",
    "confidence": 0,
    "reasoning": "LLM unavailable, no API key configured",
})

# Instruction suffix appended to every analyst system prompt
LLM_INSTRUCTION_SUFFIX = (
    '\n\nYou must respond with EXACTLY this JSON format:\n'
    '{"signal": "bullish" or "bearish" or "neutral", '
    '"confidence": 0-100, '
    '"reasoning": "your reasoning in under 180 characters"}\n'
    'Respond with ONLY the JSON, no other text.'
)

# Ollama configuration
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
_OLLAMA_AVAILABLE: bool | None = None  # cached after first check


def _check_ollama() -> bool:
    """Check if Ollama is running by hitting its tags endpoint."""
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is not None:
        return _OLLAMA_AVAILABLE
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

    Priority chain: Ollama (local) -> Anthropic -> OpenAI -> fallback.
    Ollama is checked automatically (no API key needed).

    Args:
        system_prompt: System-level instruction (analyst philosophy).
        user_prompt: User-level content (financial facts).

    Returns:
        Raw text response from the LLM.
    """
    # 1. Ollama (free, local) -- highest priority
    if _check_ollama():
        result = _call_ollama(system_prompt, user_prompt)
        if result != _FALLBACK_RESPONSE:
            return result

    # 2. Anthropic (Claude)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        result = _call_anthropic(system_prompt, user_prompt, anthropic_key)
        if result != _FALLBACK_RESPONSE:
            return result
        # Anthropic failed -- fall through to OpenAI

    # 3. OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        result = _call_openai(system_prompt, user_prompt, openai_key)
        if result != _FALLBACK_RESPONSE:
            return result

    # 4. Fallback -- neutral/0
    if not anthropic_key and not openai_key:
        print("  [LLM] No API key configured", file=sys.stderr)
    return _FALLBACK_RESPONSE


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """Call Ollama via its OpenAI-compatible API with retry.

    Uses the openai library pointed at Ollama's local endpoint.
    No API key required -- Ollama runs locally for free.

    If Ollama returns a CUDA or OOM error, marks Ollama as unavailable
    for the remainder of the process so the call chain falls through
    to Anthropic/OpenAI/fallback gracefully.
    """
    global _OLLAMA_AVAILABLE

    try:
        import openai
    except ImportError:
        return _FALLBACK_RESPONSE

    client = openai.OpenAI(
        base_url=f"{_OLLAMA_BASE_URL}/v1",
        api_key="ollama",  # Ollama ignores this but openai lib requires it
        timeout=120.0,  # local models can be slow on first load
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=_OLLAMA_MODEL,
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


def _call_anthropic(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call Anthropic Claude API with retry."""
    try:
        import anthropic
    except ImportError:
        return _FALLBACK_RESPONSE

    client = anthropic.Anthropic(api_key=api_key, timeout=30.0)

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"  [LLM] Anthropic error (attempt {attempt + 1}/2): {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)
            else:
                return _FALLBACK_RESPONSE

    return _FALLBACK_RESPONSE


def _call_openai(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """Call OpenAI API with retry."""
    try:
        import openai
    except ImportError:
        return _FALLBACK_RESPONSE

    client = openai.OpenAI(api_key=api_key, timeout=30.0)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=256,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or _FALLBACK_RESPONSE
        except Exception as e:
            print(f"  [LLM] OpenAI error (attempt {attempt + 1}/2): {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)
            else:
                return _FALLBACK_RESPONSE

    return _FALLBACK_RESPONSE
