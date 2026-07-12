# LLM-Lite Backtest Comparison Report

**Date:** 2026-07-11
**Period:** 2025-01-01 to 2026-07-11 (320 trading days)
**Tickers:** AAPL, MSFT, NVDA
**LLM Personas:** Buffett, Graham, Druckenmiller, Taleb
**LLM Calls:** 60 (4 personas x 3 tickers x 5 dates)
**Cost:** ~$1-2 (Anthropic Claude)

## Performance Comparison

| Metric | Quant-Only (8 tickers) | LLM-Lite (3 tickers) |
|--------|----------------------|---------------------|
| Total Return | +73.29% | +20.49% |
| Sharpe Ratio | 2.20 | 0.92 |
| Alpha vs SPY | +36.82% | N/A (different ticker set) |

**Note:** Direct comparison is imperfect — quant-only ran on 8 tickers vs 3 for LLM-lite. The value of this test is signal diversity persistence, not absolute performance.

## LLM Signal Diversity — Key Findings

### 1. Diversity persists over time
- **87% of observations** (13/15 ticker-date combos) had composite scores changed by LLM signals
- **53% of observations** (8/15) had decisions flipped (e.g., BUY→HOLD or HOLD→BUY)
- This is NOT date-specific noise — it's consistent across the full 15-month period

### 2. The "bearish moderator" finding was partially date-specific
- Single-shot validation (2026-07-11) showed LLMs as net bearish moderators
- Over 15 months: 7 bearish shifts, 6 bullish shifts
- Accurate characterization: **"LLMs add genuine diversity"** not "LLMs are bearish moderators"

### 3. Persona differentiation
| Persona | Behavior | Signal Quality |
|---------|----------|---------------|
| **Druckenmiller** | Highest-signal. Changes conviction based on market conditions (bearish during dips, bullish during recoveries). | Best |
| **Taleb** | Consistent risk-aware perspective. Adds antifragility dimension. | Good |
| **Graham** | Always bearish on high-P/E stocks. Predictable but useful as a counterweight. | Predictable |
| **Buffett** | Always bullish on quality companies. Predictable but useful as a confirmer. | Predictable |

### 4. Ticker sensitivity
- **MSFT** is the most LLM-sensitive: decision flipped on 4/5 LLM dates
- **AAPL** moderately sensitive: decision changed on 2-3 LLM dates
- **NVDA** least sensitive: strong quant signals dominate

## Implications for Prospects

1. **LLM analysts are not decoration.** They materially change portfolio decisions across time.
2. **The quant-only pipeline is a valid baseline.** It works without API keys and produces strong results.
3. **LLM augmentation adds qualitative reasoning.** The "why" behind decisions changes, even when the "what" doesn't.
4. **Cost is manageable.** 60 calls for 15 months of weekly rebalancing = ~$1-2. Full 18-analyst backtest would need selective persona sampling.

## Run Command

```bash
python -m src.main --tickers AAPL MSFT NVDA --backtest --llm-lite --start-date 2025-01-01 --show-reasoning
```
