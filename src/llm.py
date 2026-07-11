"""Shared LLM call utility for Covenant Hedge Fund analysts.

Supports Anthropic (Claude) and OpenAI APIs. Checks ANTHROPIC_API_KEY
first, falls back to OPENAI_API_KEY. If neither is set, returns a
graceful fallback that produces neutral/0 signals.
"""

from __future__ import annotations

import json
import os
import time


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


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an LLM and return the raw text response.

    Tries Anthropic first, then OpenAI. Returns a fallback JSON string
    if neither API key is set or if all retries fail.

    Args:
        system_prompt: System-level instruction (analyst philosophy).
        user_prompt: User-level content (financial facts).

    Returns:
        Raw text response from the LLM.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if anthropic_key:
        return _call_anthropic(system_prompt, user_prompt, anthropic_key)
    elif openai_key:
        return _call_openai(system_prompt, user_prompt, openai_key)
    else:
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
        except Exception:
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
        except Exception:
            if attempt == 0:
                time.sleep(3)
            else:
                return _FALLBACK_RESPONSE

    return _FALLBACK_RESPONSE
