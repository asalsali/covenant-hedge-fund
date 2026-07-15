"""Macro and thematic analyst agents for the Covenant Hedge Fund.

Seven analysts encoding diverse macro and special-situation
methodologies. All are LLM-augmented -- their philosophy strings
serve as system prompts for LLM-based analysis.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.agents.base import BaseAnalyst
from src.agents.value import (
    _ALL_SKILLS,
    _extract_value_facts,
    _pad,
    _parse_llm_signal,
    _to_float,
)
from src.llm import LLM_INSTRUCTION_SUFFIX, _FALLBACK_RESPONSE, call_llm
from src.models import AnalystSignal
from src.skills import format_skills_prompt, get_skills_for_analyst


# ---------------------------------------------------------------------------
# Macro-specific fact extractors
# ---------------------------------------------------------------------------

def _extract_druckenmiller_facts(ticker: str, data: dict) -> dict:
    """Base value facts plus volatility and moving average metrics."""
    facts = _extract_value_facts(ticker, data)
    prices = data.get("prices", [])
    closes = [p["close"] for p in prices if p.get("close") is not None]

    if len(closes) >= 2:
        returns = np.diff(closes) / np.array(closes[:-1])
        facts["recent_volatility"] = round(float(np.std(returns)) * 100, 4)
    else:
        facts["recent_volatility"] = None

    if len(closes) >= 50:
        facts["sma_50"] = round(float(np.mean(closes[-50:])), 2)
    else:
        facts["sma_50"] = None

    if len(closes) >= 200:
        facts["sma_200"] = round(float(np.mean(closes[-200:])), 2)
    else:
        facts["sma_200"] = None

    return facts


def _extract_burry_facts(ticker: str, data: dict) -> dict:
    """Base value facts plus debt trend and margin compression."""
    facts = _extract_value_facts(ticker, data)
    metrics = data.get("financial_metrics", [])

    # D/E trend
    de_values = []
    for m in metrics[:3]:
        de = _to_float(m.get("debt_to_equity"))
        if de is not None:
            de_values.append(de)
    facts["debt_to_equity_trend"] = de_values if de_values else None

    # Margin compression detection
    margin_values = []
    for m in metrics[:3]:
        nm = _to_float(m.get("net_margin"))
        if nm is not None:
            margin_values.append(nm)
    facts["net_margin_trend"] = margin_values if margin_values else None
    if len(margin_values) >= 2:
        facts["margin_compressing"] = margin_values[0] < margin_values[-1]
    else:
        facts["margin_compressing"] = None

    return facts


def _extract_wood_facts(ticker: str, data: dict) -> dict:
    """Base value facts plus revenue growth rate and R&D proxy."""
    facts = _extract_value_facts(ticker, data)
    line_items = data.get("line_items", [])

    revs = [_to_float(li.get("revenue")) for li in line_items[:3]]
    revs = [r for r in revs if r is not None and r > 0]
    if len(revs) >= 2:
        facts["revenue_growth_rate"] = round((revs[0] / revs[-1]) ** (1.0 / (len(revs) - 1)) - 1, 4)
    else:
        facts["revenue_growth_rate"] = None

    # Use earnings_growth as R&D proxy (actual R&D not in available data)
    facts["innovation_proxy_earnings_growth"] = facts.get("earnings_growth")

    return facts


def _extract_lynch_facts(ticker: str, data: dict) -> dict:
    """Base value facts plus PEG calculation and category hints."""
    facts = _extract_value_facts(ticker, data)
    pe = facts.get("price_to_earnings")
    eg = facts.get("earnings_growth")

    if pe and pe > 0 and eg and eg > 0:
        eg_pct = eg * 100 if eg < 1 else eg
        if eg_pct > 0:
            facts["peg_ratio"] = round(pe / eg_pct, 2)
        else:
            facts["peg_ratio"] = None
    else:
        facts["peg_ratio"] = None

    # Category hint based on earnings growth
    if eg is not None:
        if eg > 0.20:
            facts["category_hint"] = "fast_grower"
        elif eg > 0.05:
            facts["category_hint"] = "stalwart"
        elif eg >= 0:
            facts["category_hint"] = "slow_grower"
        else:
            facts["category_hint"] = "possible_turnaround"
    else:
        facts["category_hint"] = "unknown"

    return facts


def _extract_ackman_facts(ticker: str, data: dict) -> dict:
    """Base value facts plus FCF yield and insider buying patterns."""
    facts = _extract_value_facts(ticker, data)
    line_items = data.get("line_items", [])
    metrics = data.get("financial_metrics", [])
    insider_trades = data.get("insider_trades", [])

    # FCF yield
    fcf = _to_float(line_items[0].get("free_cash_flow")) if line_items else None
    mc = _to_float(metrics[0].get("market_cap")) if metrics else None
    if fcf and mc and mc > 0:
        facts["fcf_yield"] = round(fcf / mc, 4)
    else:
        facts["fcf_yield"] = None

    # Insider buying pattern
    buys, sells = 0, 0
    for t in insider_trades:
        tx = (t.get("transaction_type") or "").upper()
        if "P" in tx or "BUY" in tx or "PURCHASE" in tx:
            buys += 1
        elif "S" in tx or "SELL" in tx or "SALE" in tx:
            sells += 1
    facts["insider_buys"] = buys
    facts["insider_sells"] = sells
    total = buys + sells
    facts["insider_buy_ratio"] = round(buys / total, 2) if total > 0 else None

    return facts


def _extract_taleb_facts(ticker: str, data: dict) -> dict:
    """Base value facts plus volatility, leverage, and longevity."""
    facts = _extract_value_facts(ticker, data)
    prices = data.get("prices", [])
    line_items = data.get("line_items", [])
    closes = [p["close"] for p in prices if p.get("close") is not None]

    # Volatility metrics
    if len(closes) >= 2:
        returns = np.diff(closes) / np.array(closes[:-1])
        facts["daily_vol"] = round(float(np.std(returns)) * 100, 4)
        facts["annualized_vol"] = round(float(np.std(returns) * np.sqrt(252)) * 100, 2)
        # Kurtosis (fat tails indicator)
        if len(returns) >= 10:
            facts["return_kurtosis"] = round(float(
                np.mean((returns - np.mean(returns)) ** 4) /
                (np.std(returns) ** 4) - 3
            ), 2)
        else:
            facts["return_kurtosis"] = None
    else:
        facts["daily_vol"] = None
        facts["annualized_vol"] = None
        facts["return_kurtosis"] = None

    # Leverage ratio
    facts["leverage_de"] = facts.get("debt_to_equity")

    # Business longevity estimate (proxy: number of annual periods available)
    facts["data_periods_available"] = len(line_items)
    facts["lindy_proxy_years"] = len(line_items)  # each period ~1 year

    return facts


def _extract_news_facts(ticker: str, data: dict) -> dict:
    """Minimal facts -- news data not available via free API."""
    facts = _extract_value_facts(ticker, data)
    facts["news_data_available"] = False
    facts["note"] = (
        "Real-time news data not available via current data sources. "
        "Sentiment analysis limited to financial metrics inference."
    )
    return facts


# ---------------------------------------------------------------------------
# Shared runner for macro analysts
# ---------------------------------------------------------------------------

def _run_macro_analyst(
    analyst: BaseAnalyst,
    tickers: list[str],
    market_data: dict[str, Any],
    fact_extractor: Any,
) -> dict[str, AnalystSignal]:
    """Shared LLM analyst execution pattern for macro analysts.

    Three-tier failure handling mirrors _run_llm_analyst in value.py.
    """
    matched_skills = get_skills_for_analyst(analyst.name, _ALL_SKILLS)
    skills_appendix = format_skills_prompt(matched_skills)
    system_prompt = analyst.philosophy + skills_appendix + LLM_INSTRUCTION_SUFFIX
    results: dict[str, AnalystSignal] = {}

    for ticker in tickers:
        data = market_data.get(ticker, {})
        facts = fact_extractor(ticker, data)

        meaningful_keys = [k for k, v in facts.items()
                          if k not in ("ticker", "line_items", "note",
                                       "news_data_available")
                          and v is not None]
        if not meaningful_keys:
            results[ticker] = AnalystSignal(
                signal="neutral", confidence=0,
                reasoning=_pad("No financial data available"),
            )
            continue

        user_prompt = f"Analyze {ticker}. Here are the financial facts:\n{json.dumps(facts, indent=2)}"
        response = call_llm(system_prompt, user_prompt)

        # Tier 2: LLM call failure
        if response == _FALLBACK_RESPONSE:
            results[ticker] = AnalystSignal(
                signal="neutral",
                confidence=0,
                reasoning=_pad("LLM unavailable (abstained)"),
                abstained=True,
            )
            continue

        # Tier 3: LLM responded but parse may fail
        results[ticker] = _parse_llm_signal(
            response, ticker=ticker, analyst=analyst.name,
        )

    return results


# ---------------------------------------------------------------------------
# Analyst classes
# ---------------------------------------------------------------------------

class DruckenmillerAnalyst(BaseAnalyst):
    """Stanley Druckenmiller-style macro analyst.

    Top-down macro with aggressive position sizing when conviction
    is high. Focuses on liquidity flows and regime changes.
    """

    def __init__(self) -> None:
        super().__init__(
            name="druckenmiller",
            domain="macro",
            philosophy=(
                "You are a macro analyst following Stanley Druckenmiller's approach. "
                "Focus on liquidity flows: central bank policy, credit conditions, and "
                "money supply trends drive asset prices more than fundamentals in the "
                "medium term. Identify regime changes early -- shifts in monetary policy, "
                "fiscal stance, or geopolitical alignment that alter the macro landscape. "
                "When conviction is high, size positions aggressively -- the biggest "
                "mistake is being right on direction but too small on size. Monitor "
                "currency markets as the purest expression of macro forces. Be willing "
                "to change your mind quickly when evidence shifts. Never fight the Fed, "
                "but anticipate when the Fed will pivot."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_macro_analyst(self, tickers, market_data, _extract_druckenmiller_facts)


class BurryAnalyst(BaseAnalyst):
    """Michael Burry-style contrarian deep-value analyst.

    Identifies systemic mispricing, structural disconnects, and
    asymmetric shorts. Willing to be early and alone.
    """

    def __init__(self) -> None:
        super().__init__(
            name="burry",
            domain="macro",
            philosophy=(
                "You are a contrarian analyst following Michael Burry's methodology. "
                "Look for systemic mispricing where the consensus is not just wrong but "
                "structurally wrong -- where incentives, complexity, or willful blindness "
                "have created a disconnect between price and reality. Analyze the "
                "underlying assets, not the derivatives or indices. Read the footnotes, "
                "the 10-K risk factors, the covenant terms that nobody reads. Be willing "
                "to take asymmetric short positions when the downside is capped but the "
                "upside of being right is enormous. Accept that being early is "
                "indistinguishable from being wrong -- size positions to survive the "
                "carry cost. Focus on water scarcity, commodity supply constraints, and "
                "structural economic imbalances."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_macro_analyst(self, tickers, market_data, _extract_burry_facts)


class WoodAnalyst(BaseAnalyst):
    """Cathie Wood-style disruptive innovation analyst.

    Focuses on exponential growth in disruptive technology platforms
    with 5-year time horizons.
    """

    def __init__(self) -> None:
        super().__init__(
            name="wood",
            domain="macro",
            philosophy=(
                "You are a disruptive innovation analyst following Cathie Wood's "
                "methodology. Identify companies at the intersection of technology "
                "platforms: AI, robotics, energy storage, genomics, and blockchain. "
                "Focus on Wright's Law cost curves -- as cumulative production doubles, "
                "costs decline by a predictable percentage. Model 5-year revenue "
                "trajectories based on total addressable market expansion, not linear "
                "extrapolation of current growth. Accept high near-term volatility for "
                "exponential long-term returns. Favor companies that are platform "
                "leaders, not followers. Weight innovation speed and R&D pipeline over "
                "current profitability. Be willing to hold through 50%+ drawdowns if "
                "the disruption thesis remains intact."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_macro_analyst(self, tickers, market_data, _extract_wood_facts)


class LynchAnalyst(BaseAnalyst):
    """Peter Lynch-style growth-at-reasonable-price analyst.

    Categorizes stocks by growth type and applies appropriate
    valuation frameworks for each category.
    """

    def __init__(self) -> None:
        super().__init__(
            name="lynch",
            domain="macro",
            philosophy=(
                "You are a growth investor following Peter Lynch's methodology. "
                "Categorize every stock into one of six types: slow growers, stalwarts, "
                "fast growers, cyclicals, turnarounds, and asset plays. Apply the "
                "appropriate framework for each type -- do not value a cyclical like a "
                "fast grower. Use the PEG ratio as a primary screen: PEG below 1.0 "
                "signals undervaluation relative to growth. Invest in what you know -- "
                "find companies through everyday observation before Wall Street discovers "
                "them. Look for companies with a 'story' that is simple and compelling. "
                "Check the balance sheet: long-term debt to equity below 35% for "
                "non-financials. Monitor insider buying as a confirmation signal. "
                "Sell when the story changes, not when the price drops."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_macro_analyst(self, tickers, market_data, _extract_lynch_facts)


class AckmanAnalyst(BaseAnalyst):
    """Bill Ackman-style activist value analyst.

    Concentrated positions in high-quality businesses with
    identifiable catalysts for value realization.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ackman",
            domain="macro",
            philosophy=(
                "You are an activist value analyst following Bill Ackman's methodology. "
                "Identify high-quality businesses trading below intrinsic value with a "
                "specific, identifiable catalyst for value realization. Focus on simple, "
                "predictable, free-cash-flow-generative businesses with strong barriers "
                "to entry. Look for operational improvements, capital allocation changes, "
                "or strategic actions that could close the valuation gap. Concentrate "
                "heavily -- a portfolio of 8-12 positions allows deep understanding. "
                "Analyze management's capital allocation track record: buybacks, "
                "dividends, and M&A history. Be willing to take large positions that "
                "influence corporate governance. The catalyst is everything -- without "
                "it, cheap stays cheap."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_macro_analyst(self, tickers, market_data, _extract_ackman_facts)


class TalebAnalyst(BaseAnalyst):
    """Nassim Taleb-style tail risk and antifragility analyst.

    Identifies fragilities, tail risks, and positions that benefit
    from volatility and disorder.
    """

    def __init__(self) -> None:
        super().__init__(
            name="taleb",
            domain="macro",
            philosophy=(
                "You are a risk analyst following Nassim Taleb's antifragility "
                "framework. Identify fragilities: companies with high leverage, "
                "concentrated revenue, hidden optionality sellers, or dependence on "
                "low-volatility regimes. Seek antifragile positions: companies that "
                "benefit from disorder, volatility, and stress. Evaluate tail risk "
                "exposure -- what happens in the 1% scenario, not the expected case. "
                "Favor convex payoff profiles: limited downside, unlimited upside. "
                "Be skeptical of predictions, forecasts, and models that assume "
                "Gaussian distributions. Respect Lindy: things that have survived "
                "a long time are more likely to survive longer. Size positions to "
                "survive Black Swan events. The barbell strategy: very safe plus "
                "very speculative, nothing in the middle."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return _run_macro_analyst(self, tickers, market_data, _extract_taleb_facts)


class NewsSentimentAnalyst(BaseAnalyst):
    """News and market sentiment analyst.

    Uses LLM to interpret news flow, earnings call transcripts,
    and market commentary for sentiment signals.
    """

    def __init__(self) -> None:
        super().__init__(
            name="news_sentiment",
            domain="macro",
            philosophy=(
                "You are a sentiment analyst interpreting news flow and market "
                "commentary. Analyze recent news articles, earnings call transcripts, "
                "analyst upgrades/downgrades, and social media sentiment for each "
                "ticker. Distinguish between noise (daily price commentary) and signal "
                "(material business developments, regulatory changes, competitive "
                "shifts). Weight institutional commentary over retail sentiment. "
                "Detect sentiment extremes that may signal contrarian opportunities. "
                "Track narrative changes -- when the story about a company shifts, "
                "that shift often precedes price movement. Be skeptical of consensus "
                "narratives; the most profitable signals come from narrative inflections "
                "that the majority has not yet recognized."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        """Return low-confidence neutral -- news data not available."""
        results: dict[str, AnalystSignal] = {}
        for ticker in tickers:
            data = market_data.get(ticker, {})
            facts = _extract_news_facts(ticker, data)

            # Still call LLM with what we have -- it can infer sentiment
            # from financial trajectory, but will note data limitations
            my_skills = get_skills_for_analyst(self.name, _ALL_SKILLS)
            system_prompt = self.philosophy + format_skills_prompt(my_skills) + LLM_INSTRUCTION_SUFFIX
            user_prompt = (
                f"Analyze {ticker}. Note: real-time news data is NOT available. "
                f"You can only infer sentiment from financial metrics.\n"
                f"{json.dumps(facts, indent=2)}"
            )
            response = call_llm(system_prompt, user_prompt)

            if response == _FALLBACK_RESPONSE:
                results[ticker] = AnalystSignal(
                    signal="neutral", confidence=0,
                    reasoning=_pad("LLM unavailable (abstained)"),
                    abstained=True,
                )
                continue

            signal = _parse_llm_signal(
                response, ticker=ticker, analyst=self.name,
            )
            if signal.abstained:
                results[ticker] = signal
                continue

            # Cap confidence since we lack actual news data
            capped_confidence = min(signal.confidence, 30)
            results[ticker] = AnalystSignal(
                signal=signal.signal,
                confidence=capped_confidence,
                reasoning=_pad(signal.reasoning.strip()),
            )

        return results


MACRO_ANALYSTS: list[type[BaseAnalyst]] = [
    DruckenmillerAnalyst,
    BurryAnalyst,
    WoodAnalyst,
    LynchAnalyst,
    AckmanAnalyst,
    TalebAnalyst,
    NewsSentimentAnalyst,
]
