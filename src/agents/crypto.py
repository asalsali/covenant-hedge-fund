"""Crypto-native analyst agents for the Covenant Hedge Fund.

Four analysts specialized in digital asset analysis using CoinGecko
data. Two are computation-only (quant domain), two are LLM-augmented
(macro domain).
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAnalyst
from src.agents.value import _ALL_SKILLS, _pad, _parse_llm_signal
from src.llm import LLM_INSTRUCTION_SUFFIX, _FALLBACK_RESPONSE, call_llm
from src.models import AnalystSignal
from src.skills import format_skills_prompt, get_skills_for_analyst


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


def _clamp(conf: float) -> int:
    return max(0, min(100, int(round(conf))))


def _signal(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# 1. OnChainAnalyst (quant, uses_llm=False)
# ---------------------------------------------------------------------------

class OnChainAnalyst(BaseAnalyst):
    """On-chain supply dynamics analyst using CoinGecko metrics.

    Analyzes supply scarcity, dilution risk, and market cap momentum
    from crypto-specific data fields. Pure computation -- no LLM calls.
    """

    def __init__(self) -> None:
        super().__init__(
            name="onchain",
            domain="quant",
            philosophy=(
                "Analyze crypto supply dynamics from on-chain data. "
                "Evaluate circulating vs max supply ratios for scarcity, "
                "market cap vs fully diluted valuation for dilution risk, "
                "and 30-day price momentum for trend direction. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        cm = data.get("crypto_metrics", {})
        prices = data.get("prices", [])

        # Derive 30d momentum from prices when CoinGecko is unavailable
        if not cm and len(prices) >= 30:
            closes = [p["close"] for p in prices if p.get("close") is not None]
            if len(closes) >= 30 and closes[-30] > 0:
                pct_30d = (closes[-1] - closes[-30]) / closes[-30] * 100
                cm = {"price_change_percentage_30d": pct_30d}

        if not cm:
            return AnalystSignal(
                signal="neutral", confidence=0,
                reasoning=_pad("No crypto metrics or price data available"),
            )

        sc, avail, reasons = [], 0, []
        TP = 3

        # Supply scarcity: circulating_supply / max_supply
        circ = _to_float(cm.get("circulating_supply"))
        max_s = _to_float(cm.get("max_supply"))
        if circ is not None and max_s is not None and max_s > 0:
            avail += 1
            ratio = circ / max_s
            if ratio > 0.8:
                sc.append(1.0)
                reasons.append(f"Supply {ratio:.0%} released, scarce")
            elif ratio < 0.5:
                sc.append(-0.5)
                reasons.append(f"Supply {ratio:.0%} released, dilution ahead")
            else:
                sc.append((ratio - 0.65) / 0.15)
                reasons.append(f"Supply {ratio:.0%} released")

        # Dilution risk: market_cap / fully_diluted_valuation
        mc = _to_float(cm.get("market_cap"))
        fdv = _to_float(cm.get("fully_diluted_valuation"))
        if mc is not None and fdv is not None and fdv > 0:
            avail += 1
            mc_fdv = mc / fdv
            if mc_fdv < 0.5:
                sc.append(-1.0)
                reasons.append(f"MC/FDV {mc_fdv:.0%}, massive unlock risk")
            elif mc_fdv > 0.8:
                sc.append(0.5)
                reasons.append(f"MC/FDV {mc_fdv:.0%}, limited dilution")
            else:
                sc.append((mc_fdv - 0.65) / 0.15)

        # Market cap momentum: 30d price change
        pct_30d = _to_float(cm.get("price_change_percentage_30d"))
        if pct_30d is not None:
            avail += 1
            norm = max(-1.0, min(1.0, pct_30d / 30.0))
            sc.append(norm)
            if abs(pct_30d) > 10:
                reasons.append(f"30d {pct_30d:+.1f}%")

        if not sc:
            return AnalystSignal(
                signal="neutral", confidence=5,
                reasoning=_pad("Insufficient on-chain data"),
            )

        comp = sum(sc) / len(sc)
        conf = _clamp(abs(comp) * 80 * avail / TP)
        r = reasons[0] if reasons else "Mixed on-chain signals"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({avail}/{TP} factors)"),
        )


# ---------------------------------------------------------------------------
# 2. MomentumCryptoAnalyst (quant, uses_llm=False)
# ---------------------------------------------------------------------------

class MomentumCryptoAnalyst(BaseAnalyst):
    """Multi-timeframe crypto momentum analyst with ATH/ATL distance.

    Evaluates momentum alignment across 7d/30d/200d windows,
    proximity to all-time high/low, and momentum acceleration.
    Pure computation -- no LLM calls.
    """

    def __init__(self) -> None:
        super().__init__(
            name="crypto_momentum",
            domain="quant",
            philosophy=(
                "Compute multi-timeframe crypto momentum signals. "
                "Assess 7d/30d/200d price change alignment, distance "
                "from all-time high and low, and short vs long term "
                "momentum acceleration. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        # Try CoinGecko metrics first, fall back to price-derived momentum
        cm = data.get("crypto_metrics", {})
        prices = data.get("prices", [])

        # Derive momentum from price history when CoinGecko is unavailable
        if not cm and len(prices) >= 30:
            closes = [p["close"] for p in prices if p.get("close") is not None]
            if len(closes) >= 30:
                cm = self._derive_momentum_from_prices(closes)

        if not cm:
            return AnalystSignal(
                signal="neutral", confidence=0,
                reasoning=_pad("No crypto metrics or price data available"),
            )

        sc, avail, reasons = [], 0, []
        TP = 3

        # Momentum alignment: 7d, 30d, 200d
        pct_7d = _to_float(cm.get("price_change_percentage_7d"))
        pct_30d = _to_float(cm.get("price_change_percentage_30d"))
        pct_200d = _to_float(cm.get("price_change_percentage_200d"))

        timeframes = [v for v in [pct_7d, pct_30d, pct_200d] if v is not None]
        if len(timeframes) >= 2:
            avail += 1
            pos_count = sum(1 for v in timeframes if v > 0)
            neg_count = sum(1 for v in timeframes if v < 0)
            total = len(timeframes)

            if pos_count == total:
                sc.append(1.0)
                reasons.append("All timeframes bullish")
            elif neg_count == total:
                sc.append(-1.0)
                reasons.append("All timeframes bearish")
            else:
                alignment = (pos_count - neg_count) / total
                sc.append(alignment)
                reasons.append("Mixed momentum alignment")

        # ATH distance
        ath_pct = _to_float(cm.get("ath_change_percentage"))
        if ath_pct is not None:
            avail += 1
            if ath_pct > -20:
                sc.append(-0.5)
                reasons.append(f"ATH dist {ath_pct:.0f}%, distribution risk")
            elif ath_pct < -80:
                if pct_7d is not None and pct_7d > 0:
                    sc.append(0.8)
                    reasons.append(f"ATH dist {ath_pct:.0f}%, recovery signal")
                else:
                    sc.append(-0.3)
                    reasons.append(f"ATH dist {ath_pct:.0f}%, no recovery yet")
            else:
                norm = (ath_pct + 50) / 30
                sc.append(max(-1.0, min(1.0, norm)))

        # Momentum acceleration: short-term (7d) vs long-term (200d)
        if pct_7d is not None and pct_200d is not None and abs(pct_200d) > 0.01:
            avail += 1
            rate_7d = pct_7d / 7.0
            rate_200d = pct_200d / 200.0
            if rate_200d != 0:
                accel = (rate_7d - rate_200d) / max(abs(rate_200d), 0.01)
                accel_clamped = max(-1.0, min(1.0, accel / 5.0))
                sc.append(accel_clamped)
                if accel > 2:
                    reasons.append("Momentum accelerating")
                elif accel < -2:
                    reasons.append("Momentum decelerating")

        if not sc:
            return AnalystSignal(
                signal="neutral", confidence=5,
                reasoning=_pad("Insufficient momentum data"),
            )

        comp = sum(sc) / len(sc)
        conf = _clamp(abs(comp) * 80 * avail / TP)
        r = reasons[0] if reasons else "Mixed momentum signals"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({avail}/{TP} factors)"),
        )

    @staticmethod
    def _derive_momentum_from_prices(closes: list[float]) -> dict:
        """Derive CoinGecko-compatible momentum metrics from price history."""
        current = closes[-1]
        metrics: dict[str, float] = {}

        if len(closes) >= 7 and closes[-7] > 0:
            metrics["price_change_percentage_7d"] = (
                (current - closes[-7]) / closes[-7] * 100
            )
        if len(closes) >= 30 and closes[-30] > 0:
            metrics["price_change_percentage_30d"] = (
                (current - closes[-30]) / closes[-30] * 100
            )
        if len(closes) >= 200 and closes[-200] > 0:
            metrics["price_change_percentage_200d"] = (
                (current - closes[-200]) / closes[-200] * 100
            )

        # Approximate ATH from available data
        ath = max(closes)
        if ath > 0:
            metrics["ath_change_percentage"] = (current - ath) / ath * 100

        return metrics


# ---------------------------------------------------------------------------
# 3. CryptoMacroAnalyst (macro, uses_llm=True)
# ---------------------------------------------------------------------------

def _extract_crypto_macro_facts(ticker: str, data: dict) -> dict:
    """Extract crypto macro facts for LLM analysis."""
    cm = data.get("crypto_metrics", {})
    prices = data.get("prices", [])
    closes = [p["close"] for p in prices if p.get("close") is not None]
    current_price = closes[-1] if closes else None

    mc = _to_float(cm.get("market_cap"))
    vol = _to_float(cm.get("total_volume"))
    vol_mc_ratio = None
    if mc and vol and mc > 0:
        vol_mc_ratio = round(vol / mc, 4)

    return {
        "ticker": ticker,
        "current_price": current_price,
        "market_cap_rank": cm.get("market_cap_rank"),
        "market_cap": mc,
        "total_volume": vol,
        "volume_to_market_cap": vol_mc_ratio,
        "circulating_supply": _to_float(cm.get("circulating_supply")),
        "total_supply": _to_float(cm.get("total_supply")),
        "max_supply": _to_float(cm.get("max_supply")),
        "price_change_7d": _to_float(cm.get("price_change_percentage_7d")),
        "price_change_14d": _to_float(cm.get("price_change_percentage_14d")),
        "price_change_30d": _to_float(cm.get("price_change_percentage_30d")),
        "price_change_60d": _to_float(cm.get("price_change_percentage_60d")),
        "price_change_200d": _to_float(cm.get("price_change_percentage_200d")),
        "price_change_1y": _to_float(cm.get("price_change_percentage_1y")),
        "ath": _to_float(cm.get("ath")),
        "ath_change_percentage": _to_float(cm.get("ath_change_percentage")),
        "atl": _to_float(cm.get("atl")),
        "atl_change_percentage": _to_float(cm.get("atl_change_percentage")),
    }


class CryptoMacroAnalyst(BaseAnalyst):
    """Macro strategist specializing in digital assets.

    Considers crypto market cycles, institutional adoption, protocol
    utility, and macro liquidity conditions. LLM-augmented analysis.
    """

    def __init__(self) -> None:
        super().__init__(
            name="crypto_macro",
            domain="macro",
            philosophy=(
                "You are a macro strategist specializing in digital assets. "
                "Analyze the asset considering crypto market cycles (accumulation, "
                "markup, distribution, markdown), institutional adoption trends, "
                "protocol utility and network effects, and macro liquidity "
                "conditions (Fed policy, DXY strength, risk appetite). Evaluate "
                "whether the asset is positioned for the current market regime. "
                "Consider Bitcoin dominance cycles and alt-season rotation. "
                "Assess regulatory risk and geopolitical headwinds or tailwinds "
                "for the crypto sector. Weight on-chain activity and developer "
                "ecosystem health as leading indicators of fundamental value."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        _my_skills = get_skills_for_analyst(self.name, _ALL_SKILLS)
        system_prompt = self.philosophy + format_skills_prompt(_my_skills) + LLM_INSTRUCTION_SUFFIX
        results: dict[str, AnalystSignal] = {}

        for ticker in tickers:
            data = market_data.get(ticker, {})
            facts = _extract_crypto_macro_facts(ticker, data)

            meaningful = [k for k, v in facts.items()
                         if k != "ticker" and v is not None]
            if not meaningful:
                results[ticker] = AnalystSignal(
                    signal="neutral", confidence=0,
                    reasoning=_pad("No crypto data available"),
                )
                continue

            user_prompt = (
                f"Analyze {ticker} as a digital asset. "
                f"Here are the crypto market facts:\n"
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

            results[ticker] = _parse_llm_signal(
                response, ticker=ticker, analyst=self.name,
            )

        return results


# ---------------------------------------------------------------------------
# 4. TokenomicsAnalyst (macro, uses_llm=True)
# ---------------------------------------------------------------------------

def _extract_tokenomics_facts(ticker: str, data: dict) -> dict:
    """Extract tokenomics-specific facts for LLM analysis."""
    cm = data.get("crypto_metrics", {})

    circ = _to_float(cm.get("circulating_supply"))
    total = _to_float(cm.get("total_supply"))
    max_s = _to_float(cm.get("max_supply"))
    mc = _to_float(cm.get("market_cap"))
    fdv = _to_float(cm.get("fully_diluted_valuation"))
    vol = _to_float(cm.get("total_volume"))

    # Compute supply ratios
    circ_total = round(circ / total, 4) if circ and total and total > 0 else None
    circ_max = round(circ / max_s, 4) if circ and max_s and max_s > 0 else None
    total_max = round(total / max_s, 4) if total and max_s and max_s > 0 else None
    fdv_mc_ratio = round(fdv / mc, 4) if fdv and mc and mc > 0 else None
    vol_mc_ratio = round(vol / mc, 4) if vol and mc and mc > 0 else None

    return {
        "ticker": ticker,
        "circulating_supply": circ,
        "total_supply": total,
        "max_supply": max_s,
        "circulating_to_total_ratio": circ_total,
        "circulating_to_max_ratio": circ_max,
        "total_to_max_ratio": total_max,
        "market_cap": mc,
        "fully_diluted_valuation": fdv,
        "fdv_to_market_cap_ratio": fdv_mc_ratio,
        "total_volume": vol,
        "volume_to_market_cap": vol_mc_ratio,
        "price_change_7d": _to_float(cm.get("price_change_percentage_7d")),
        "price_change_30d": _to_float(cm.get("price_change_percentage_30d")),
        "price_change_200d": _to_float(cm.get("price_change_percentage_200d")),
        "price_change_1y": _to_float(cm.get("price_change_percentage_1y")),
    }


class TokenomicsAnalyst(BaseAnalyst):
    """Token economics evaluator.

    Assesses supply scarcity, inflation risk, velocity metrics, and
    store-of-value properties. LLM-augmented analysis.
    """

    def __init__(self) -> None:
        super().__init__(
            name="tokenomics",
            domain="macro",
            philosophy=(
                "You are a tokenomics specialist. Evaluate the token's economic "
                "design: supply scarcity (circulating vs max supply, emission "
                "schedule implications), inflation risk (FDV vs market cap gap "
                "indicating future dilution from vesting/unlocks), velocity "
                "metrics (volume/market cap as a proxy for transactional demand "
                "vs speculative holding), and store-of-value properties "
                "(price stability, supply cap credibility, Lindy effect of the "
                "protocol). Compare the token's economic structure to sound "
                "money principles: scarcity, durability, divisibility, "
                "verifiability. Flag tokens with aggressive unlock schedules, "
                "high inflation rates, or velocity patterns suggesting "
                "speculative churn over genuine utility demand."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        _my_skills = get_skills_for_analyst(self.name, _ALL_SKILLS)
        system_prompt = self.philosophy + format_skills_prompt(_my_skills) + LLM_INSTRUCTION_SUFFIX
        results: dict[str, AnalystSignal] = {}

        for ticker in tickers:
            data = market_data.get(ticker, {})
            facts = _extract_tokenomics_facts(ticker, data)

            meaningful = [k for k, v in facts.items()
                         if k != "ticker" and v is not None]
            if not meaningful:
                results[ticker] = AnalystSignal(
                    signal="neutral", confidence=0,
                    reasoning=_pad("No tokenomics data available"),
                )
                continue

            user_prompt = (
                f"Evaluate {ticker}'s tokenomics. "
                f"Here are the token economic facts:\n"
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

            results[ticker] = _parse_llm_signal(
                response, ticker=ticker, analyst=self.name,
            )

        return results


# ---------------------------------------------------------------------------
# 5. KwokAnalyst (macro, uses_llm=True) — Dom Kwok
# ---------------------------------------------------------------------------

class KwokAnalyst(BaseAnalyst):
    """Institutional adoption & network effects analyst (Dom Kwok style).

    Ex-Goldman/Blackstone perspective. Evaluates crypto through the lens
    of institutional capital flows, mass adoption curves, and network
    effects. The crypto growth investor.
    """

    def __init__(self) -> None:
        super().__init__(
            name="kwok",
            domain="macro",
            philosophy=(
                "You are an institutional crypto analyst with a Goldman Sachs "
                "and Blackstone background, modeled after Dom Kwok. Evaluate "
                "digital assets through the lens of institutional capital flows "
                "and mass adoption. Key questions: Is institutional money "
                "flowing into this asset (ETF approvals, custody solutions, "
                "treasury adoption)? What does the adoption curve look like — "
                "what percentage of potential users are onboard? Assess network "
                "effects: does rising price attract developers who build "
                "applications that drive more usage? Consider regulatory clarity "
                "as a catalyst — clearer regulation unlocks institutional capital. "
                "Evaluate cross-border payment potential and real-world utility "
                "beyond speculation. Be bullish when institutional infrastructure "
                "is building around an asset, bearish when adoption stalls or "
                "regulatory headwinds threaten institutional participation. "
                "Think in 3-5 year adoption arcs, not daily price action."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        _my_skills = get_skills_for_analyst(self.name, _ALL_SKILLS)
        system_prompt = self.philosophy + format_skills_prompt(_my_skills) + LLM_INSTRUCTION_SUFFIX
        results: dict[str, AnalystSignal] = {}

        for ticker in tickers:
            data = market_data.get(ticker, {})
            facts = _extract_crypto_macro_facts(ticker, data)

            meaningful = [k for k, v in facts.items()
                         if k != "ticker" and v is not None]
            if not meaningful:
                results[ticker] = AnalystSignal(
                    signal="neutral", confidence=0,
                    reasoning=_pad("No crypto data available"),
                )
                continue

            user_prompt = (
                f"Evaluate {ticker} from an institutional adoption perspective. "
                f"Here are the crypto market facts:\n"
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

            results[ticker] = _parse_llm_signal(
                response, ticker=ticker, analyst=self.name,
            )

        return results


# ---------------------------------------------------------------------------
# 6. WooAnalyst (macro, uses_llm=True) — Willy Woo
# ---------------------------------------------------------------------------

class WooAnalyst(BaseAnalyst):
    """On-chain data analyst (Willy Woo style).

    Translates blockchain data into views on network health, user
    behavior, and market cycle positioning. Pioneer of on-chain
    analysis as a discipline.
    """

    def __init__(self) -> None:
        super().__init__(
            name="woo",
            domain="macro",
            philosophy=(
                "You are an on-chain data analyst modeled after Willy Woo, "
                "a pioneer of on-chain analysis. Evaluate digital assets by "
                "interpreting blockchain-derived metrics as indicators of "
                "network health and market cycle positioning. Key frameworks: "
                "NVT ratio (network value to transaction volume — high NVT "
                "suggests overvaluation, low suggests undervaluation). Use "
                "volume/market cap as a proxy for on-chain velocity when "
                "direct on-chain data is unavailable. Assess supply dynamics: "
                "circulating vs max supply indicates holder conviction and "
                "scarcity. Track market cap rank momentum — rising rank with "
                "rising price confirms genuine demand, rising price with "
                "falling rank suggests broader market lift. Evaluate ATH "
                "distance as a cycle indicator: assets >50% below ATH in "
                "a rising market are potential recovery plays; assets near "
                "ATH with declining volume signal distribution. Think in "
                "market cycles: accumulation, markup, distribution, markdown. "
                "Where is this asset in its cycle based on the data?"
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        _my_skills = get_skills_for_analyst(self.name, _ALL_SKILLS)
        system_prompt = self.philosophy + format_skills_prompt(_my_skills) + LLM_INSTRUCTION_SUFFIX
        results: dict[str, AnalystSignal] = {}

        for ticker in tickers:
            data = market_data.get(ticker, {})
            facts = _extract_crypto_macro_facts(ticker, data)

            meaningful = [k for k, v in facts.items()
                         if k != "ticker" and v is not None]
            if not meaningful:
                results[ticker] = AnalystSignal(
                    signal="neutral", confidence=0,
                    reasoning=_pad("No crypto data available"),
                )
                continue

            user_prompt = (
                f"Analyze {ticker}'s on-chain health and cycle positioning. "
                f"Here are the network and market facts:\n"
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

            results[ticker] = _parse_llm_signal(
                response, ticker=ticker, analyst=self.name,
            )

        return results


# ---------------------------------------------------------------------------
# 7. PlanBAnalyst (macro, uses_llm=True) — PlanB
# ---------------------------------------------------------------------------

class PlanBAnalyst(BaseAnalyst):
    """Stock-to-flow & cycle-based scarcity analyst (PlanB style).

    Evaluates crypto through supply scarcity models, halving cycles,
    and stock-to-flow ratios. Quantitative scarcity as the primary
    value driver.
    """

    def __init__(self) -> None:
        super().__init__(
            name="planb",
            domain="macro",
            philosophy=(
                "You are a quantitative scarcity analyst modeled after PlanB, "
                "creator of the Bitcoin Stock-to-Flow model. Evaluate digital "
                "assets primarily through supply scarcity and monetary hardness. "
                "Key frameworks: Stock-to-flow ratio — existing supply divided "
                "by new annual production. Higher S2F = harder money = higher "
                "value (gold S2F ~62, Bitcoin post-halving ~120). Assess "
                "circulating/max supply ratio as a scarcity proxy. Assets with "
                "hard caps (like Bitcoin's 21M) are fundamentally different "
                "from inflationary tokens. Evaluate cycle positioning relative "
                "to supply events (halvings, unlock schedules). Post-halving "
                "periods historically precede price appreciation due to supply "
                "shock. Compare the asset's scarcity profile to gold and other "
                "monetary assets. Be bullish when: hard supply cap exists, "
                "high percentage of supply already circulating, approaching or "
                "recently past a supply reduction event. Be bearish when: no "
                "supply cap, high inflation rate, large unlocks ahead "
                "(FDV >> market cap). Price should converge toward scarcity-"
                "implied value over full market cycles."
            ),
            uses_llm=True,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        _my_skills = get_skills_for_analyst(self.name, _ALL_SKILLS)
        system_prompt = self.philosophy + format_skills_prompt(_my_skills) + LLM_INSTRUCTION_SUFFIX
        results: dict[str, AnalystSignal] = {}

        for ticker in tickers:
            data = market_data.get(ticker, {})
            facts = _extract_tokenomics_facts(ticker, data)

            meaningful = [k for k, v in facts.items()
                         if k != "ticker" and v is not None]
            if not meaningful:
                results[ticker] = AnalystSignal(
                    signal="neutral", confidence=0,
                    reasoning=_pad("No scarcity data available"),
                )
                continue

            user_prompt = (
                f"Evaluate {ticker}'s scarcity profile and cycle positioning. "
                f"Here are the supply and market facts:\n"
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

            results[ticker] = _parse_llm_signal(
                response, ticker=ticker, analyst=self.name,
            )

        return results


# ---------------------------------------------------------------------------
# 8. DeFiFlowAnalyst (quant, uses_llm=False)
# ---------------------------------------------------------------------------

class DeFiFlowAnalyst(BaseAnalyst):
    """DeFi capital flow analyst using DeFi Llama TVL data.

    Tracks total value locked (TVL) trends per chain as a proxy for
    capital inflows/outflows. Rising TVL = capital entering the
    ecosystem = bullish for the native token. Pure computation.
    """

    def __init__(self) -> None:
        super().__init__(
            name="defi_flow",
            domain="quant",
            philosophy=(
                "Analyze DeFi capital flows via TVL trends. "
                "Rising TVL signals capital entering an ecosystem, "
                "bullish for the chain's native token. Falling TVL "
                "signals capital flight, bearish. Evaluate 30-day "
                "trend magnitude and 7-day acceleration. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        tvl = data.get("defi_tvl", {})

        if not tvl or "tvl_change_pct_30d" not in tvl:
            return AnalystSignal(
                signal="neutral", confidence=0,
                reasoning=_pad("No DeFi TVL data available"),
            )

        sc, avail, reasons = [], 0, []
        TP = 2  # Two factors: 30d trend + 7d acceleration

        # Factor 1: 30-day TVL change
        pct_30d = tvl.get("tvl_change_pct_30d", 0)
        avail += 1
        if pct_30d > 10:
            norm = min(1.0, pct_30d / 50.0)
            sc.append(norm)
            reasons.append(f"TVL 30d +{pct_30d:.1f}%, capital inflow")
        elif pct_30d < -10:
            norm = max(-1.0, pct_30d / 50.0)
            sc.append(norm)
            reasons.append(f"TVL 30d {pct_30d:.1f}%, capital flight")
        else:
            sc.append(0.0)
            reasons.append(f"TVL 30d {pct_30d:+.1f}%, stable")

        # Factor 2: 7-day acceleration vs 30-day rate
        pct_7d = tvl.get("tvl_change_pct_7d")
        if pct_7d is not None and abs(pct_30d) > 0.01:
            avail += 1
            rate_7d = pct_7d / 7.0
            rate_30d = pct_30d / 30.0
            if abs(rate_30d) > 0.001:
                accel = (rate_7d - rate_30d) / max(abs(rate_30d), 0.01)
                accel_clamped = max(-1.0, min(1.0, accel / 3.0))
                sc.append(accel_clamped)
                if accel > 1.5:
                    reasons.append("TVL accelerating")
                elif accel < -1.5:
                    reasons.append("TVL decelerating")

        if not sc:
            return AnalystSignal(
                signal="neutral", confidence=5,
                reasoning=_pad("Insufficient TVL data"),
            )

        comp = sum(sc) / len(sc)
        conf = _clamp(abs(comp) * 80 * avail / TP)
        r = reasons[0] if reasons else "Mixed TVL signals"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({avail}/{TP} factors)"),
        )


# ---------------------------------------------------------------------------
# 9. FearGreedAnalyst (quant, uses_llm=False)
# ---------------------------------------------------------------------------

class FearGreedAnalyst(BaseAnalyst):
    """Contrarian sentiment analyst using the Crypto Fear & Greed Index.

    Applies Warren Buffett's principle: be fearful when others are greedy,
    be greedy when others are fearful. The F&G Index is BTC-dominated but
    reflects overall crypto market sentiment -- applied equally to all
    crypto tickers.

    Pure computation -- no LLM calls.
    """

    def __init__(self) -> None:
        super().__init__(
            name="fear_greed",
            domain="quant",
            philosophy=(
                "Contrarian sentiment analysis via the Crypto Fear & Greed "
                "Index. Buy fear, sell greed. Extreme readings carry higher "
                "confidence. Trend direction (rising/falling fear) modulates "
                "the signal strength."
            ),
            uses_llm=False,
        )

    def analyze(
        self,
        tickers: list[str],
        market_data: dict[str, Any],
    ) -> dict[str, AnalystSignal]:
        # F&G is market-wide -- compute once, apply to all tickers
        # Pull from any ticker's market_data (all share the same F&G data)
        fg_data: dict[str, Any] = {}
        for t in tickers:
            fg_data = market_data.get(t, {}).get("fear_greed", {})
            if fg_data:
                break

        signal = self._compute_signal(fg_data)
        return {t: signal for t in tickers}

    @staticmethod
    def _compute_signal(fg: dict[str, Any]) -> AnalystSignal:
        value = fg.get("current_value")
        if value is None:
            return AnalystSignal(
                signal="neutral", confidence=0,
                reasoning=_pad("Fear & Greed data unavailable"),
            )

        trend = fg.get("trend", "stable")
        avg_7d = fg.get("avg_7d")
        avg_30d = fg.get("avg_30d")
        classification = fg.get("current_classification", "")

        # Contrarian mapping: fear -> bullish, greed -> bearish
        if value <= 20:
            # Extreme Fear -> strong buy signal
            base_signal = "bullish"
            base_conf = 60 + (20 - value)  # 60-80
        elif value <= 40:
            # Fear -> mild buy signal
            base_signal = "bullish"
            base_conf = 40 + (40 - value)  # 40-60
        elif value <= 60:
            # Neutral zone
            base_signal = "neutral"
            base_conf = 20
        elif value <= 80:
            # Greed -> mild sell signal
            base_signal = "bearish"
            base_conf = 40 + (value - 60)  # 40-60
        else:
            # Extreme Greed -> strong sell signal
            base_signal = "bearish"
            base_conf = 60 + (value - 80)  # 60-80

        # Trend modulation: rising fear is more bullish, rising greed more bearish
        trend_adj = 0
        if trend == "falling" and base_signal == "bullish":
            # Fear is rising (values falling) -> stronger buy
            trend_adj = 5
        elif trend == "falling" and base_signal == "bearish":
            # Greed is falling -> weaker sell
            trend_adj = -5
        elif trend == "rising" and base_signal == "bearish":
            # Greed is rising (values rising) -> stronger sell
            trend_adj = 5
        elif trend == "rising" and base_signal == "bullish":
            # Fear is falling -> weaker buy
            trend_adj = -5

        final_conf = _clamp(base_conf + trend_adj)

        # Build reasoning
        trend_note = ""
        if avg_7d is not None and avg_30d is not None:
            trend_note = f", 7d avg {avg_7d:.0f} vs 30d avg {avg_30d:.0f}"

        reasoning = (
            f"F&G={value} ({classification}){trend_note}, "
            f"trend {trend} -> contrarian {base_signal}"
        )

        return AnalystSignal(
            signal=base_signal,
            confidence=final_conf,
            reasoning=_pad(reasoning),
        )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

CRYPTO_ANALYSTS: list[type[BaseAnalyst]] = [
    OnChainAnalyst,
    MomentumCryptoAnalyst,
    DeFiFlowAnalyst,
    FearGreedAnalyst,
]

CRYPTO_LLM_ANALYSTS: list[type[BaseAnalyst]] = [
    CryptoMacroAnalyst,
    TokenomicsAnalyst,
    KwokAnalyst,
    WooAnalyst,
    PlanBAnalyst,
]
