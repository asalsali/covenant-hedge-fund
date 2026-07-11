"""Value investing analyst agents for the Covenant Hedge Fund.

Six analysts encoding the methodologies of legendary value investors.
All are LLM-augmented -- their philosophy strings serve as system
prompts for LLM-based analysis of fundamental data.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAnalyst
from src.llm import LLM_INSTRUCTION_SUFFIX, call_llm
from src.models import AnalystSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pad(text: str) -> str:
    if len(text) < 20:
        text = text + " " * (20 - len(text))
    return text[:200]


def _extract_value_facts(ticker: str, data: dict) -> dict:
    """Extract common financial facts for value analysts."""
    prices = data.get("prices", [])
    metrics = data.get("financial_metrics", [])
    line_items = data.get("line_items", [])

    cur = metrics[0] if metrics else {}

    # Current price
    closes = [p["close"] for p in prices if p.get("close") is not None]
    current_price = closes[-1] if closes else None
    price_start = closes[0] if closes else None
    price_change_pct = None
    if current_price and price_start and price_start > 0:
        price_change_pct = round((current_price / price_start - 1) * 100, 2)

    # Line items (last 3 periods)
    li_data = []
    for li in line_items[:3]:
        li_data.append({
            "period": li.get("period_end_date") or li.get("report_period"),
            "revenue": _to_float(li.get("revenue")),
            "net_income": _to_float(li.get("net_income")),
            "free_cash_flow": _to_float(li.get("free_cash_flow")),
            "operating_cash_flow": _to_float(li.get("operating_cash_flow")),
        })

    # Format ratios as percentages so LLMs read "14.1%" not "0.141"
    def _pct(val):
        v = _to_float(val)
        return f"{v * 100:.1f}%" if v is not None else None

    return {
        "ticker": ticker,
        "current_price": current_price,
        "price_change_pct": price_change_pct,
        "return_on_equity": _pct(cur.get("return_on_equity")),
        "debt_to_equity": _to_float(cur.get("debt_to_equity")),  # ratio, not %
        "current_ratio": _to_float(cur.get("current_ratio")),    # ratio, not %
        "net_margin": _pct(cur.get("net_margin")),
        "gross_margin": _pct(cur.get("gross_margin")),
        "price_to_earnings": _to_float(cur.get("price_to_earnings")),  # multiple
        "ev_to_ebitda": _to_float(cur.get("ev_to_ebitda")),            # multiple
        "market_cap": _to_float(cur.get("market_cap")),
        "earnings_growth": _pct(cur.get("earnings_growth")),
        "line_items": li_data,
    }


def _parse_llm_signal(response: str) -> AnalystSignal:
    """Parse LLM JSON response into AnalystSignal."""
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        parsed = json.loads(text)
        signal = parsed.get("signal", "neutral")
        if signal not in ("bullish", "bearish", "neutral"):
            signal = "neutral"
        confidence = max(0, min(100, int(parsed.get("confidence", 0))))
        reasoning = str(parsed.get("reasoning", "LLM analysis"))
        return AnalystSignal(
            signal=signal,
            confidence=confidence,
            reasoning=_pad(reasoning),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return AnalystSignal(
            signal="neutral",
            confidence=0,
            reasoning=_pad("Failed to parse LLM response"),
        )


def _run_llm_analyst(
    analyst: BaseAnalyst,
    tickers: list[str],
    market_data: dict[str, Any],
    fact_extractor: Any = None,
) -> dict[str, AnalystSignal]:
    """Shared LLM analyst execution pattern."""
    extractor = fact_extractor or _extract_value_facts
    system_prompt = analyst.philosophy + LLM_INSTRUCTION_SUFFIX
    results: dict[str, AnalystSignal] = {}

    for ticker in tickers:
        data = market_data.get(ticker, {})
        facts = extractor(ticker, data)

        # Check if we have any meaningful data
        meaningful_keys = [k for k, v in facts.items()
                          if k not in ("ticker", "line_items") and v is not None]
        if not meaningful_keys:
            results[ticker] = AnalystSignal(
                signal="neutral", confidence=0,
                reasoning=_pad("No financial data available"),
            )
            continue

        user_prompt = f"Analyze {ticker}. Here are the financial facts:\n{json.dumps(facts, indent=2)}"
        response = call_llm(system_prompt, user_prompt)
        results[ticker] = _parse_llm_signal(response)

    return results


class BuffettAnalyst(BaseAnalyst):
    """Warren Buffett-style value analyst.

    Focuses on circle of competence, durable competitive advantages,
    management quality, and margin of safety.
    """

    def __init__(self) -> None:
        super().__init__(
            name="buffett",
            domain="value",
            philosophy=(
                "You are a value investor following Warren Buffett's methodology. "
                "Evaluate businesses within your circle of competence. Look for durable "
                "competitive moats: brand power, network effects, cost advantages, "
                "switching costs, and regulatory barriers. Assess management quality -- "
                "integrity, candor, and capital allocation skill. Analyze financial "
                "strength: consistent ROE above 15%, low debt-to-equity, strong free "
                "cash flow generation. Demand a margin of safety -- only buy when the "
                "intrinsic value significantly exceeds the market price. Evaluate "
                "long-term prospects over a 10+ year horizon. Prefer businesses that "
                "are simple, predictable, and generate owner earnings."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_llm_analyst(self, tickers, market_data)


class GrahamAnalyst(BaseAnalyst):
    """Benjamin Graham-style defensive value analyst.

    Applies strict quantitative screens from The Intelligent Investor
    and Security Analysis -- the most mechanical of the value analysts.
    """

    def __init__(self) -> None:
        super().__init__(
            name="graham",
            domain="value",
            philosophy=(
                "You are a defensive value investor following Benjamin Graham's "
                "quantitative screens. Apply strict criteria: earnings stability over "
                "10 consecutive years with no deficit. Dividend record of uninterrupted "
                "payments for at least 20 years. Earnings growth of at least 33% over "
                "the past 10 years using three-year averages. Balance sheet strength: "
                "current ratio >= 2.0, long-term debt no more than net current assets. "
                "Valuation discipline: P/E ratio no more than 15x average earnings over "
                "3 years AND P/B ratio no more than 1.5x. Combined Graham Number: "
                "P/E * P/B should not exceed 22.5. Reject businesses that fail any "
                "single criterion -- there is no partial credit in defensive investing."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_llm_analyst(self, tickers, market_data)


class MungerAnalyst(BaseAnalyst):
    """Charlie Munger-style multi-disciplinary value analyst.

    Applies mental models from multiple disciplines to evaluate
    businesses. Emphasizes inversion and simplicity.
    """

    def __init__(self) -> None:
        super().__init__(
            name="munger",
            domain="value",
            philosophy=(
                "You are a value investor following Charlie Munger's multi-disciplinary "
                "approach. Apply mental models from psychology, economics, physics, and "
                "biology to evaluate businesses. Identify competitive advantages through "
                "multiple lenses -- not just financial metrics. Assess management "
                "integrity above all: dishonest management disqualifies any business. "
                "Prefer simplicity -- if you cannot explain the business model in one "
                "sentence, it is too complex. Practice inversion: instead of asking "
                "'why will this succeed?', ask 'what would cause this to fail?' and "
                "check for those conditions. Be willing to pay a fair price for a "
                "wonderful business rather than a wonderful price for a fair business. "
                "Patience is a virtue -- wait for the fat pitch."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_llm_analyst(self, tickers, market_data)


class PabraiAnalyst(BaseAnalyst):
    """Mohnish Pabrai-style concentrated value analyst.

    Applies the Dhandho framework -- low risk, high uncertainty,
    asymmetric payoffs with few concentrated bets.
    """

    def __init__(self) -> None:
        super().__init__(
            name="pabrai",
            domain="value",
            philosophy=(
                "You are a value investor following Mohnish Pabrai's Dhandho framework. "
                "Seek situations where the downside is limited but the upside is "
                "substantial -- 'heads I win, tails I don't lose much.' Distinguish "
                "between risk (permanent capital loss) and uncertainty (temporary price "
                "volatility). Favor low-risk, high-uncertainty situations where the "
                "market confuses uncertainty with risk, creating mispricing. Make few "
                "big bets infrequently -- concentrate in your highest-conviction ideas. "
                "Look for existing businesses with proven models, not speculative "
                "ventures. Clone successful investors shamelessly. Demand a margin of "
                "safety that makes the downside case still acceptable."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_llm_analyst(self, tickers, market_data)


class FisherAnalyst(BaseAnalyst):
    """Philip Fisher-style growth-at-reasonable-price analyst.

    Applies the scuttlebutt method and evaluates long-term growth
    potential through qualitative research.
    """

    def __init__(self) -> None:
        super().__init__(
            name="fisher",
            domain="value",
            philosophy=(
                "You are a growth investor following Philip Fisher's scuttlebutt "
                "method. Evaluate growth potential through qualitative research: talk "
                "to customers, competitors, suppliers, and former employees. Assess "
                "R&D effectiveness -- does the company consistently convert research "
                "spending into profitable products? Evaluate the sales organization's "
                "strength and market reach. Analyze profit margins and the company's "
                "plan to maintain or improve them. Examine management depth -- is there "
                "a strong bench beyond the CEO? Assess management integrity -- do they "
                "communicate honestly with shareholders about both successes and "
                "failures? Hold outstanding companies for the long term; sell only when "
                "the original reasons for purchase are no longer valid."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_llm_analyst(self, tickers, market_data)


class DamodaranAnalyst(BaseAnalyst):
    """Aswath Damodaran-style academic valuation analyst.

    Rigorous DCF-based valuation with explicit risk decomposition
    and life-cycle awareness.
    """

    def __init__(self) -> None:
        super().__init__(
            name="damodaran",
            domain="value",
            philosophy=(
                "You are a valuation analyst following Aswath Damodaran's academic "
                "framework. Build valuation from DCF foundations: estimate free cash "
                "flows, determine appropriate discount rates using cost of capital, "
                "and model terminal value carefully. Decompose growth into its "
                "components: reinvestment rate and return on invested capital. Apply "
                "risk-adjusted returns -- do not use a single discount rate when the "
                "risk profile varies across scenarios. Use relative valuation as a "
                "cross-check, not a substitute: compare multiples only within peer "
                "groups with similar growth, risk, and cash flow profiles. Distinguish "
                "between pricing (what the market will pay) and valuation (what the "
                "asset is worth). Be aware of business life-cycle stage: young growth "
                "companies, mature businesses, and declining firms need different models."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_llm_analyst(self, tickers, market_data)


VALUE_ANALYSTS: list[type[BaseAnalyst]] = [
    BuffettAnalyst,
    GrahamAnalyst,
    MungerAnalyst,
    PabraiAnalyst,
    FisherAnalyst,
    DamodaranAnalyst,
]
