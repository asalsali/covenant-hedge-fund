"""Base analyst interface for all Covenant hedge fund analysts.

Every analyst -- whether LLM-augmented or pure computation -- inherits
from BaseAnalyst and implements the analyze() method to produce uniform
AnalystSignal outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models import AnalystSignal


class BaseAnalyst(ABC):
    """Base class for all Covenant hedge fund analysts.

    Attributes:
        name: Unique analyst identifier (e.g., "buffett", "technicals").
        domain: Domain assignment -- "value", "quant", or "macro".
        philosophy: System prompt encoding the analyst's investment
            philosophy. Used as the LLM system prompt for LLM-augmented
            analysts. Serves as documentation for computation-only analysts.
        uses_llm: Whether this analyst makes LLM calls. Quant domain
            analysts MUST set this to False.
    """

    name: str
    domain: str
    philosophy: str
    uses_llm: bool = True

    def __init__(self, name: str, domain: str, philosophy: str, *, uses_llm: bool = True) -> None:
        self.name = name
        self.domain = domain
        self.philosophy = philosophy
        self.uses_llm = uses_llm

    @abstractmethod
    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
        quant_evidence: dict[str, str] | None = None,
    ) -> dict[str, AnalystSignal]:
        """Analyze tickers and return signals.

        Args:
            tickers: List of ticker symbols to analyze.
            market_data: Pre-fetched market data relevant to this
                analyst's needs. Structure varies by analyst type.
            quant_evidence: Optional dict mapping ticker -> formatted
                evidence brief string from quant analysts. LLM analysts
                inject this into their prompts. Quant analysts ignore it.

        Returns:
            Dict mapping ticker symbol to AnalystSignal.
        """
        raise NotImplementedError

    def to_memo_payload(
        self,
        signals: dict[str, AnalystSignal],
    ) -> dict[str, Any]:
        """Format signals as a structured memo payload.

        Converts analyst signals into the Covenant memo format with
        signal ontology types for the Interpreter to consume.
        """
        entries = {}
        for ticker, signal in signals.items():
            ontology_type = {
                "bullish": "convergence",
                "bearish": "tension",
                "neutral": None,
            }.get(signal.signal)

            entries[ticker] = {
                "signal": signal.signal,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "ontology": {
                    "type": ontology_type,
                    "confidence": signal.confidence / 100.0,
                } if ontology_type else None,
            }

        return {
            "sender": f"analyst-{self.name}",
            "domain": self.domain,
            "signals": entries,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} domain={self.domain!r}>"
