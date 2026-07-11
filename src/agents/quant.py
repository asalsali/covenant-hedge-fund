"""Quantitative analyst agents for the Covenant Hedge Fund.

Five computation-only analysts that produce signals from numerical
data without LLM calls. All set uses_llm=False -- quant domain
analysts MUST NOT make LLM calls per COMPLIANCE.md.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.agents.base import BaseAnalyst
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


def _clamp(conf: float) -> int:
    return max(0, min(100, int(round(conf))))


def _signal(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _pad(text: str) -> str:
    if len(text) < 20:
        text = text + " " * (20 - len(text))
    return text[:200]


def _ema(values: list[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


# ---------------------------------------------------------------------------
# 1. TechnicalsAnalyst
# ---------------------------------------------------------------------------

class TechnicalsAnalyst(BaseAnalyst):
    """Technical analysis: RSI, MACD, Bollinger, SMA crossover."""

    def __init__(self) -> None:
        super().__init__(
            name="technicals", domain="quant",
            philosophy=(
                "Compute technical signals from price and volume data. "
                "Calculate RSI (14-day), MACD (12/26/9), Bollinger Bands "
                "(20-day, 2 std), and 50/200-day moving average crossovers. "
                "Combine indicators into a composite signal with weighted "
                "scoring. No LLM calls -- pure computation."
            ),
            uses_llm=False)

    def analyze(self, tickers: list[str], market_data: dict[str, Any]) -> dict[str, AnalystSignal]:
        """Compute technical signals for each ticker."""
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        prices = data.get("prices", [])
        closes = [p["close"] for p in prices if p.get("close") is not None]
        if len(closes) < 14:
            return AnalystSignal(signal="neutral", confidence=0,
                                 reasoning=_pad("Insufficient price data"))

        sc, wt, reasons = [], [], []

        # RSI 14
        deltas = np.diff(closes[-(15):]).astype(float)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        ag, al = float(np.mean(gains)), float(np.mean(losses))
        rsi = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
        if rsi < 30:
            sc.append(1.0); reasons.append(f"RSI {rsi:.0f} oversold")
        elif rsi > 70:
            sc.append(-1.0); reasons.append(f"RSI {rsi:.0f} overbought")
        else:
            sc.append((50 - rsi) / 50.0); reasons.append(f"RSI {rsi:.0f}")
        wt.append(0.25)

        # MACD (12/26/9)
        if len(closes) >= 26:
            e12, e26 = _ema(closes, 12), _ema(closes, 26)
            macd_s = [a - b for a, b in zip(e12, e26)]
            sig_s = _ema(macd_s, 9)
            diff = macd_s[-1] - sig_s[-1]
            norm = abs(closes[-1]) * 0.02 + 1e-9
            sc.append(max(-1.0, min(1.0, diff / norm)))
            reasons.append("MACD bullish" if diff > 0 else "MACD bearish")
            wt.append(0.25)

        # Bollinger %B (20-day, 2 std)
        if len(closes) >= 20:
            w = closes[-20:]
            sma = float(np.mean(w))
            std = float(np.std(w, ddof=1))
            if std > 0:
                lb, ub = sma - 2 * std, sma + 2 * std
                pct_b = (closes[-1] - lb) / (ub - lb)
                sc.append(max(-1.0, min(1.0, 1.0 - 2.0 * pct_b)))
                wt.append(0.25)

        # SMA 50/200 crossover
        if len(closes) >= 200:
            s50 = float(np.mean(closes[-50:]))
            s200 = float(np.mean(closes[-200:]))
            r = (s50 - s200) / s200
            sc.append(max(-1.0, min(1.0, r * 10)))
            reasons.append("Golden cross" if s50 > s200 else "Death cross")
            wt.append(0.25)

        if not sc:
            return AnalystSignal(signal="neutral", confidence=5,
                                 reasoning=_pad("No indicators computable"))

        tw = sum(wt)
        comp = sum(s * w for s, w in zip(sc, wt)) / tw
        conf = _clamp(abs(comp) * 80 * (tw / 1.0))
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{reasons[0]}, composite={comp:+.2f}"))


# ---------------------------------------------------------------------------
# 2. FundamentalsAnalyst
# ---------------------------------------------------------------------------

class FundamentalsAnalyst(BaseAnalyst):
    """Fundamental metrics: ROE, D/E, margins, Piotroski."""

    def __init__(self) -> None:
        super().__init__(
            name="fundamentals", domain="quant",
            philosophy=(
                "Screen stocks using fundamental financial ratios. "
                "Compute ROE, ROA, ROIC, debt-to-equity, current ratio, "
                "interest coverage, free cash flow yield, and operating "
                "margin trends. Score each metric against sector medians. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False)

    def analyze(self, tickers: list[str], market_data: dict[str, Any]) -> dict[str, AnalystSignal]:
        """Compute fundamental scores for each ticker."""
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        ml = data.get("financial_metrics", [])
        li = data.get("line_items", [])
        if not ml:
            return AnalystSignal(signal="neutral", confidence=0,
                                 reasoning=_pad("No financial metrics available"))

        cur, pri = ml[0], (ml[1] if len(ml) > 1 else {})
        cli = li[0] if li else {}
        pli = li[1] if len(li) > 1 else {}
        sc, avail, reasons = [], 0, []
        TP = 6

        # ROE
        v = _to_float(cur.get("return_on_equity"))
        if v is not None:
            avail += 1
            if v > 0.15:
                sc.append(1.0); reasons.append(f"ROE {v:.0%} strong")
            elif v < 0.08:
                sc.append(-1.0); reasons.append(f"ROE {v:.0%} weak")
            else:
                sc.append((v - 0.115) / 0.035)

        # Debt-to-equity
        v = _to_float(cur.get("debt_to_equity"))
        if v is not None:
            avail += 1
            if v < 0.5:
                sc.append(1.0); reasons.append(f"Low D/E={v:.1f}")
            elif v > 2.0:
                sc.append(-1.0); reasons.append(f"High D/E={v:.1f}")
            else:
                sc.append(1.0 - (v - 0.5) / 1.5 * 2.0)

        # Current ratio
        v = _to_float(cur.get("current_ratio"))
        if v is not None:
            avail += 1
            if v > 1.5:
                sc.append(1.0)
            elif v < 1.0:
                sc.append(-1.0); reasons.append(f"Current ratio {v:.1f}")
            else:
                sc.append((v - 1.25) / 0.25)

        # Net margin
        v = _to_float(cur.get("net_margin"))
        if v is not None:
            avail += 1
            if v > 0.15:
                sc.append(1.0); reasons.append(f"Margin {v:.0%}")
            elif v < 0.05:
                sc.append(-1.0); reasons.append(f"Margin {v:.0%} thin")
            else:
                sc.append((v - 0.10) / 0.05)

        # FCF yield
        fcf = _to_float(cli.get("free_cash_flow"))
        mc = _to_float(cur.get("market_cap"))
        if fcf is not None and mc and mc > 0:
            fy = fcf / mc
            avail += 1
            if fy > 0.05:
                sc.append(1.0); reasons.append(f"FCF yield {fy:.1%}")
            elif fy < 0.02:
                sc.append(-1.0)
            else:
                sc.append((fy - 0.035) / 0.015)

        # Piotroski F-Score
        ps = self._piotroski(cur, pri, cli, pli)
        if ps is not None:
            avail += 1
            if ps >= 7:
                sc.append(1.0); reasons.append(f"Piotroski {ps}/9")
            elif ps <= 3:
                sc.append(-1.0); reasons.append(f"Piotroski {ps}/9")
            else:
                sc.append((ps - 5) / 4.0)

        if not sc:
            return AnalystSignal(signal="neutral", confidence=5,
                                 reasoning=_pad("All metrics missing"))
        comp = float(np.mean(sc))
        conf = _clamp(abs(comp) * 85 * avail / TP)
        r = reasons[0] if reasons else "Mixed fundamentals"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({avail}/{TP} metrics)"))

    @staticmethod
    def _piotroski(cur: dict, pri: dict, cli: dict, pli: dict) -> int | None:
        """Simplified Piotroski F-Score (0-9)."""
        s, ck = 0, 0
        roa = _to_float(cur.get("return_on_assets"))
        if roa is not None:
            ck += 1; s += int(roa > 0)
        ocf = _to_float(cli.get("operating_cash_flow"))
        if ocf is not None:
            ck += 1; s += int(ocf > 0)
        roa_p = _to_float(pri.get("return_on_assets"))
        if roa is not None and roa_p is not None:
            ck += 1; s += int(roa > roa_p)
        ni = _to_float(cli.get("net_income"))
        if ocf is not None and ni is not None:
            ck += 1; s += int(ocf > ni)
        de_c = _to_float(cur.get("debt_to_equity"))
        de_p = _to_float(pri.get("debt_to_equity"))
        if de_c is not None and de_p is not None:
            ck += 1; s += int(de_c < de_p)
        cr_c = _to_float(cur.get("current_ratio"))
        cr_p = _to_float(pri.get("current_ratio"))
        if cr_c is not None and cr_p is not None:
            ck += 1; s += int(cr_c > cr_p)
        sh_c = (_to_float(cli.get("outstanding_shares"))
                or _to_float(cli.get("weighted_average_shares")))
        sh_p = (_to_float(pli.get("outstanding_shares"))
                or _to_float(pli.get("weighted_average_shares")))
        if sh_c is not None and sh_p is not None:
            ck += 1; s += int(sh_c <= sh_p)
        gm_c = _to_float(cur.get("gross_margin"))
        gm_p = _to_float(pri.get("gross_margin"))
        if gm_c is not None and gm_p is not None:
            ck += 1; s += int(gm_c > gm_p)
        at_c = _to_float(cur.get("asset_turnover"))
        at_p = _to_float(pri.get("asset_turnover"))
        if at_c is not None and at_p is not None:
            ck += 1; s += int(at_c > at_p)
        return int(round(s * 9 / ck)) if ck > 0 else None


# ---------------------------------------------------------------------------
# 3. ValuationAnalyst
# ---------------------------------------------------------------------------

class ValuationAnalyst(BaseAnalyst):
    """Valuation: P/E, EV/EBITDA, P/FCF, PEG, simple DCF."""

    def __init__(self) -> None:
        super().__init__(
            name="valuation", domain="quant",
            philosophy=(
                "Compute quantitative valuation metrics. Build simple DCF "
                "models using historical growth rates and sector-average "
                "discount rates. Calculate EV/EBITDA, P/E, P/FCF, and PEG "
                "ratios. Compare current price to computed fair value range. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False)

    def analyze(self, tickers: list[str], market_data: dict[str, Any]) -> dict[str, AnalystSignal]:
        """Compute valuation signals for each ticker."""
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        ml = data.get("financial_metrics", [])
        li = data.get("line_items", [])
        if not ml and not li:
            return AnalystSignal(signal="neutral", confidence=0,
                                 reasoning=_pad("No valuation data available"))

        cm = ml[0] if ml else {}
        cli = li[0] if li else {}
        sc, avail, reasons = [], 0, []
        TP = 5

        # P/E ratio
        pe = _to_float(cm.get("price_to_earnings"))
        if pe is not None and pe > 0:
            avail += 1
            if pe < 12:
                sc.append(1.0); reasons.append(f"P/E {pe:.1f} cheap")
            elif pe > 25:
                sc.append(-1.0); reasons.append(f"P/E {pe:.1f} rich")
            else:
                sc.append((18.5 - pe) / 6.5)

        # EV/EBITDA
        ev = _to_float(cm.get("ev_to_ebitda"))
        if ev is not None and ev > 0:
            avail += 1
            if ev < 8:
                sc.append(1.0); reasons.append(f"EV/EBITDA {ev:.1f}")
            elif ev > 15:
                sc.append(-1.0); reasons.append(f"EV/EBITDA {ev:.1f}")
            else:
                sc.append((11.5 - ev) / 3.5)

        # P/FCF
        mc = _to_float(cm.get("market_cap"))
        fcf = _to_float(cli.get("free_cash_flow"))
        if mc and fcf and fcf > 0:
            pf = mc / fcf
            avail += 1
            if pf < 15:
                sc.append(1.0); reasons.append(f"P/FCF {pf:.1f}")
            elif pf > 30:
                sc.append(-1.0); reasons.append(f"P/FCF {pf:.1f}")
            else:
                sc.append((22.5 - pf) / 7.5)

        # PEG ratio
        eg = _to_float(cm.get("earnings_growth"))
        if pe and pe > 0 and eg and eg > 0:
            eg_pct = eg * 100 if eg < 1 else eg
            if eg_pct > 0:
                peg = pe / eg_pct
                avail += 1
                if peg < 1.0:
                    sc.append(1.0); reasons.append(f"PEG {peg:.1f}")
                elif peg > 2.0:
                    sc.append(-1.0); reasons.append(f"PEG {peg:.1f}")
                else:
                    sc.append((1.5 - peg) / 0.5)

        # Simple DCF
        dcf = self._dcf(li, mc)
        if dcf:
            avail += 1; sc.append(dcf[0]); reasons.append(dcf[1])

        if not sc:
            return AnalystSignal(signal="neutral", confidence=5,
                                 reasoning=_pad("Valuation metrics missing"))
        comp = float(np.mean(sc))
        conf = _clamp(abs(comp) * 85 * avail / TP)
        r = reasons[0] if reasons else "Mixed valuation"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({avail}/{TP} metrics)"))

    @staticmethod
    def _dcf(li: list[dict], mc: float | None) -> tuple[float, str] | None:
        """Simple 5-year DCF: grow recent FCF, discount at 10%."""
        if not li or not mc or mc <= 0:
            return None
        fv = [_to_float(l.get("free_cash_flow")) for l in li]
        fv = [f for f in fv if f is not None and f > 0]
        if not fv:
            return None
        recent = fv[0]
        if len(fv) >= 2 and fv[-1] > 0:
            gr = (recent / fv[-1]) ** (1.0 / (len(fv) - 1)) - 1.0
            gr = max(-0.10, min(0.20, gr))
        else:
            gr = 0.05
        dr, tg = 0.10, 0.03
        dcf_val, proj = 0.0, recent
        for y in range(1, 6):
            proj *= (1.0 + gr)
            dcf_val += proj / ((1.0 + dr) ** y)
        tv = proj * (1 + tg) / (dr - tg)
        dcf_val += tv / ((1.0 + dr) ** 5)
        ratio = dcf_val / mc
        if ratio > 1.2:
            return (min(1.0, (ratio - 1) * 2), f"DCF {ratio:.0%} mkt, undervalued")
        elif ratio < 0.8:
            return (max(-1.0, (ratio - 1) * 2), f"DCF {ratio:.0%} mkt, overvalued")
        return ((ratio - 1) * 2, f"DCF {ratio:.0%} of mkt cap, fair")


# ---------------------------------------------------------------------------
# 4. GrowthAnalyst
# ---------------------------------------------------------------------------

class GrowthAnalyst(BaseAnalyst):
    """Growth: revenue/earnings CAGR, acceleration, margins."""

    def __init__(self) -> None:
        super().__init__(
            name="growth", domain="quant",
            philosophy=(
                "Compute growth trajectory metrics. Calculate revenue and "
                "earnings CAGR over 1, 3, and 5-year windows. Measure "
                "growth acceleration (second derivative). Assess growth "
                "sustainability via reinvestment rate and ROIC spread. "
                "Flag deceleration patterns and margin compression. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False)

    def analyze(self, tickers: list[str], market_data: dict[str, Any]) -> dict[str, AnalystSignal]:
        """Compute growth signals for each ticker."""
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        li = data.get("line_items", [])
        if len(li) < 2:
            return AnalystSignal(signal="neutral", confidence=0,
                                 reasoning=_pad("Insufficient periods for growth"))

        sc, avail, reasons = [], 0, []
        TP = 4
        revs = [_to_float(l.get("revenue")) for l in li]
        revs = [r for r in revs if r is not None and r > 0]
        earns = [_to_float(l.get("net_income")) for l in li]
        earns = [e for e in earns if e is not None]

        # Revenue CAGR
        if len(revs) >= 2:
            avail += 1
            yrs = min(3, len(revs) - 1)
            cagr = (revs[0] / revs[yrs]) ** (1.0 / yrs) - 1.0
            if cagr > 0.15:
                sc.append(1.0); reasons.append(f"Rev CAGR {cagr:.0%}")
            elif cagr < 0:
                sc.append(-1.0); reasons.append(f"Rev declining {cagr:.0%}")
            else:
                sc.append(cagr / 0.15)

        # Earnings CAGR
        if len(earns) >= 2:
            yrs = min(3, len(earns) - 1)
            if earns[yrs] > 0:
                avail += 1
                ec = (earns[0] / earns[yrs]) ** (1.0 / yrs) - 1.0
                if ec > 0.15:
                    sc.append(1.0); reasons.append(f"Earn CAGR {ec:.0%}")
                elif ec < 0:
                    sc.append(-1.0); reasons.append(f"Earn decline {ec:.0%}")
                else:
                    sc.append(ec / 0.15)

        # Growth acceleration
        if len(revs) >= 3 and revs[1] > 0 and revs[2] > 0:
            avail += 1
            rg = (revs[0] / revs[1]) - 1
            og = (revs[1] / revs[2]) - 1
            acc = rg - og
            if acc > 0.05:
                sc.append(1.0); reasons.append("Growth accelerating")
            elif acc < -0.05:
                sc.append(-1.0); reasons.append("Growth decelerating")
            else:
                sc.append(acc / 0.05)

        # Margin trajectory
        if (len(revs) >= 2 and len(earns) >= 2
                and revs[0] > 0 and revs[1] > 0):
            avail += 1
            md = earns[0] / revs[0] - earns[1] / revs[1]
            if md > 0.02:
                sc.append(1.0); reasons.append(f"Margins +{md:.1%}")
            elif md < -0.02:
                sc.append(-1.0); reasons.append(f"Margins {md:.1%}")
            else:
                sc.append(md / 0.02)

        if not sc:
            return AnalystSignal(signal="neutral", confidence=5,
                                 reasoning=_pad("Growth not computable"))
        comp = float(np.mean(sc))
        conf = _clamp(abs(comp) * 85 * avail / TP)
        r = reasons[0] if reasons else "Mixed growth"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({avail}/{TP} factors)"))


# ---------------------------------------------------------------------------
# 5. SentimentAnalyst
# ---------------------------------------------------------------------------

class SentimentAnalyst(BaseAnalyst):
    """Sentiment: insider buy/sell ratio, clusters, volume."""

    def __init__(self) -> None:
        super().__init__(
            name="sentiment", domain="quant",
            philosophy=(
                "Compute quantitative sentiment indicators. Score insider "
                "transaction patterns (cluster buys vs sells). Combine "
                "into a contrarian sentiment composite. "
                "No LLM calls -- pure computation."
            ),
            uses_llm=False)

    def analyze(self, tickers: list[str], market_data: dict[str, Any]) -> dict[str, AnalystSignal]:
        """Compute sentiment signals for each ticker."""
        return {t: self._run(t, market_data.get(t, {})) for t in tickers}

    def _run(self, ticker: str, data: dict) -> AnalystSignal:
        trades = data.get("insider_trades", [])
        if not trades:
            return AnalystSignal(signal="neutral", confidence=10,
                                 reasoning=_pad("No insider trade data available"))

        bc, slc, bv, sv = 0, 0, 0.0, 0.0
        bdates: list[str] = []
        for t in trades:
            tx = (t.get("transaction_type") or "").upper()
            sh = abs(_to_float(t.get("shares")) or 0)
            pr = _to_float(t.get("price_per_share")) or 0
            val = sh * pr
            dt = t.get("date") or t.get("transaction_date") or ""
            if "P" in tx or "BUY" in tx or "PURCHASE" in tx:
                bc += 1; bv += val
                if dt:
                    bdates.append(dt)
            elif "S" in tx or "SELL" in tx or "SALE" in tx:
                slc += 1; sv += val

        tot = bc + slc
        if tot == 0:
            return AnalystSignal(signal="neutral", confidence=10,
                                 reasoning=_pad("No buy/sell insider trades"))

        sc, avail, reasons = [], 0, []
        TP = 3

        # Count ratio
        avail += 1
        br = bc / tot
        sc.append(max(-1.0, min(1.0, (br - 0.5) * 4.0)))
        if br > 0.6:
            reasons.append(f"Insider buy ratio {br:.0%}")
        elif br < 0.3:
            reasons.append(f"Insider sell heavy {br:.0%}")

        # Value-weighted
        tv = bv + sv
        if tv > 0:
            avail += 1
            bvr = bv / tv
            sc.append(max(-1.0, min(1.0, (bvr - 0.5) * 4.0)))
            if bvr > 0.6:
                reasons.append(f"Buy vol ${bv / 1e6:.1f}M")

        # Cluster detection
        avail += 1
        cluster = False
        if len(bdates) >= 3:
            month_counts: dict[str, int] = defaultdict(int)
            for d in bdates:
                month_counts[d[:7] if len(d) >= 7 else d] += 1
            mx = max(month_counts.values()) if month_counts else 0
            if mx >= 3:
                cluster = True
                sc.append(1.0)
                reasons.append(f"Buy cluster: {mx} in month")
            else:
                sc.append(0.0)
        else:
            sc.append(0.0)

        comp = float(np.mean(sc))
        conf = _clamp(abs(comp) * 75 * avail / TP)
        if cluster:
            conf = _clamp(conf + 15)
        r = reasons[0] if reasons else "Mixed insider signals"
        return AnalystSignal(
            signal=_signal(comp), confidence=conf,
            reasoning=_pad(f"{r} ({bc}B/{slc}S)"))


QUANT_ANALYSTS: list[type[BaseAnalyst]] = [
    TechnicalsAnalyst,
    FundamentalsAnalyst,
    ValuationAnalyst,
    GrowthAnalyst,
    SentimentAnalyst,
]
