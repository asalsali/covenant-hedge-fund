"""Evidence brief formatter for LLM-first pipeline.

Converts quant analyst signals into a structured evidence brief that
gets injected into LLM analyst prompts. The quant signals become
inputs to LLM reasoning rather than independent votes in the quorum.

The evidence brief is human-readable and LLM-readable: plain text
with clear structure, not JSON. This lets LLMs process it naturally
while keeping it inspectable for debugging.
"""

from __future__ import annotations

from src.models import AnalystSignal


def format_evidence_brief(
    ticker: str,
    quant_signals: dict[str, AnalystSignal],
) -> str:
    """Format quant signals into an evidence brief for LLM consumption.

    Args:
        ticker: The ticker symbol being analyzed.
        quant_signals: Dict mapping quant analyst name to their signal.

    Returns:
        A formatted evidence brief string ready for prompt injection.
        Returns empty string if no quant signals are available.
    """
    if not quant_signals:
        return ""

    lines = [f"QUANTITATIVE EVIDENCE FOR {ticker}:"]

    for name, signal in quant_signals.items():
        if signal.abstained:
            lines.append(f"- {_display_name(name)}: ABSTAINED -- {signal.reasoning.strip()}")
            continue

        direction = signal.signal.upper()
        conf = signal.confidence
        reasoning = signal.reasoning.strip()

        lines.append(f"- {_display_name(name)}: {direction} (conf {conf}%) -- {reasoning}")

    lines.append("")
    lines.append(
        "Consider this quantitative evidence alongside your own analysis. "
        "You may agree or disagree with the quant signals -- they are "
        "inputs to your reasoning, not constraints on your judgment."
    )

    return "\n".join(lines)


def _display_name(analyst_name: str) -> str:
    """Convert internal analyst name to a readable display name."""
    name_map = {
        "technicals": "Technicals",
        "fundamentals": "Fundamentals",
        "valuation": "Valuation",
        "growth": "Growth",
        "sentiment": "Sentiment",
        "onchain": "On-Chain",
        "crypto_momentum": "Crypto Momentum",
        "defi_flow": "DeFi Flow",
        "fear_greed": "Fear & Greed",
    }
    return name_map.get(analyst_name, analyst_name.replace("_", " ").title())
