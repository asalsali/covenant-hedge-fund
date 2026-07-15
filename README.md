# Covenant Hedge Fund

> AI-governed portfolio analysis with deterministic risk management

An 18-analyst portfolio system that produces real trade decisions with zero LLM dependencies. Built on the [Covenant Framework](https://covenant.foundation), every signal is deterministic, every risk limit is enforced in code, and every decision is auditable through a structured decision graph. Optional LLM augmentation adds macro/contrarian analysis without becoming a dependency.

## Architecture

```
                        Market Data (Yahoo Finance)
                                    |
                    +---------------+---------------+
                    |               |               |
              Quant Domain    Value Domain    Macro Domain
              (5 analysts)   (6 analysts)    (7 analysts)
              -----------    -----------     -----------
              Technicals     Buffett         Druckenmiller
              Fundamentals   Graham          Burry
              Valuation      Munger          Wood
              Growth         Pabrai          Lynch
              Sentiment      Fisher          Ackman
                             Damodaran       Taleb
                                             News Sentiment
                    |               |               |
                    +-------+-------+-------+-------+
                            |               |
                      Risk Engine      Signal Synthesis
                   (volatility,        (confidence-weighted
                    correlation,        scoring, uniform
                    position limits)    signal schema)
                            |               |
                            +-------+-------+
                                    |
                              Trade Execution
                           (deterministic sizing,
                            compliance-checked)
```

All 18 analysts produce a uniform signal: `{signal, confidence, reasoning}`. The Risk Engine enforces position limits, volatility scaling, and correlation constraints as code -- not as suggestions to an LLM. The synthesis layer uses confidence-weighted scoring: same inputs produce the same portfolio decisions every time.

## Key Features

- **18 analysts across 3 domains** (quant, value, macro/contrarian)
- **Works with 0 API keys** -- quant-only mode needs no LLM at all
- **Optional LLM augmentation** -- Ollama (free, local)
- **Deterministic risk** -- volatility-adjusted position sizing, correlation limits, max exposure caps
- **25 compliance rules** enforced in code via `COMPLIANCE.md`
- **Backtesting** with SPY benchmark, Sharpe/Sortino/drawdown metrics
- **Decision traceability** -- every trade links back to the analyst signals that informed it

## Quick Start

```bash
git clone https://github.com/asalsali/covenant-hedge-fund.git
cd covenant-hedge-fund
pip install -r requirements.txt
python demo.py
```

This runs a 6-month backtest on AAPL, MSFT, NVDA using 5 quant analysts (no API keys needed). Results appear in your terminal and as an HTML report in `reports/`.

For LLM-augmented analysis, start Ollama:

```bash
ollama serve &
ollama pull qwen2.5:7b-instruct
python -m src.main --tickers AAPL MSFT NVDA

# Use a specific model (overrides auto-detection):
python -m src.main --tickers AAPL MSFT NVDA --model phi4:14b
```

### More Examples

Run a single analysis with reasoning (quant-only, no API keys required):

```bash
python -m src.main --tickers AAPL MSFT GOOGL --show-reasoning
```

Run a backtest over a custom date range:

```bash
python -m src.main --tickers AAPL MSFT GOOGL JPM XOM \
  --backtest --start-date 2025-01-01
```

## How It Works

**1. Data Collection.** Market prices, financial metrics, and insider trades are fetched from Yahoo Finance (free) for the requested tickers and date range.

**2. Analyst Signals.** Each of the 18 analysts runs independently and produces a scored signal (-100 to +100) with a confidence weight. Quant analysts (technicals, fundamentals, valuation, growth, sentiment) run without any LLM. Value analysts (Buffett, Graham, Munger, Pabrai, Fisher, Damodaran) and macro analysts (Druckenmiller, Burry, Wood, Lynch, Ackman, Taleb, News Sentiment) use LLM reasoning when available, gracefully degrading to neutral signals when no LLM is configured.

**3. Risk Engine.** Before any trade, the risk engine computes per-ticker volatility, cross-portfolio correlation, and position limits. These are hard constraints -- no trade can exceed them regardless of analyst consensus.

**4. Signal Synthesis.** Analyst signals are combined using confidence-weighted scoring. The composite score determines BUY (+0.30 threshold), SHORT (-0.30 threshold), or HOLD. This is deterministic arithmetic, not LLM judgment.

**5. Trade Execution.** Position sizes are calculated from risk limits and available capital. Trades are executed against the portfolio state, with margin tracking for short positions.

## vs Traditional AI Hedge Funds

| | Covenant Hedge Fund | [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) |
|---|---|---|
| LLM required to trade | No (quant-only mode) | Yes (all decisions via LLM) |
| Decision determinism | Same inputs = same outputs | LLM variance on every run |
| Risk enforcement | 25 code-enforced rules | LLM-suggested risk agent |
| Dependencies | ~7 packages | LangChain + LangGraph + LLM SDKs |
| Cost to run | Free (Ollama) or $0 (quant-only) | Paid API keys required |
| Conflict resolution | Structured mediation protocol | Naive signal averaging |

## Performance

Backtest: 8 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, JNJ, XOM), 320 trading days.

| Metric | Value |
|---|---|
| Total return | +73.29% |
| Annualized return | +54.39% |
| Sharpe ratio | 2.20 |
| Sortino ratio | 4.01 |
| Max drawdown | -15.06% |
| SPY return (same period) | +36.47% |
| Alpha vs SPY | +36.82% |

See `samples/backtest-output.txt` for the full backtest report, or `samples/single-shot-output.txt` for a single analysis run.

## Configuration

### Environment Variables

```bash
# Optional: Ollama config (not needed for quant-only mode)
# Ollama is auto-detected at localhost:11434 -- no API keys required
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct    # or use --model CLI flag
```

### LLM Backend

Ollama is the sole LLM backend. It is auto-detected at localhost:11434. If Ollama is not available, value and macro analysts return neutral signals and the system operates on quant signals alone (quant-only mode). No paid APIs are used.

Model selection follows this priority chain:
1. `--model` CLI flag (per-run override)
2. `OLLAMA_MODEL` env var
3. Auto-detect: picks the best model already pulled in Ollama
4. Fallback: `qwen2.5:7b-instruct`

### Recommended Models by VRAM

| Tier | VRAM Required | Model | Pull Command |
|---|---|---|---|
| 1 (default) | 8 GB | `qwen2.5:7b-instruct` | `ollama pull qwen2.5:7b-instruct` |
| 2 | 16 GB | `phi4:14b` | `ollama pull phi4:14b` |
| 3 | 24 GB | `qwen2.5:32b-instruct` | `ollama pull qwen2.5:32b-instruct` |
| 4 | 48 GB+ | `llama3.3:70b-instruct` | `ollama pull llama3.3:70b-instruct` |

Pull any model and the system auto-selects it. Pull multiple and it picks the best one available.

### Ollama Setup (Free Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (pick one that fits your GPU)
ollama pull qwen2.5:7b-instruct      # 8GB VRAM (default)
# ollama pull phi4:14b                # 16GB VRAM
# ollama pull qwen2.5:32b-instruct   # 24GB VRAM

# No configuration needed -- auto-detected at localhost:11434
```

## Project Structure

```
covenant-hedge-fund/
  src/
    main.py              # Entry point and CLI
    models.py            # Pydantic signal and portfolio models
    risk.py              # Deterministic risk engine
    portfolio.py         # Portfolio state management
    backtest.py          # Backtesting engine with SPY benchmark
    llm.py               # LLM client with graceful degradation
    data/
      api.py             # Market data API client
    agents/
      base.py            # Base analyst interface
      quant.py           # Quant domain (5 analysts)
      value.py           # Value domain (6 analysts)
      macro.py           # Macro/contrarian domain (7 analysts)
  CLAUDE.md              # Project constitution
  COMPLIANCE.md          # 25 trading compliance rules
  demo.py                # Zero-config one-command demo
  demo.sh                # Shell-based demo script
  samples/
    single-shot-output.txt   # Example analysis output
    backtest-output.txt      # Example backtest output
```

## Built on Covenant Framework

This hedge fund is a reference implementation of the [Covenant Framework](https://covenant.foundation) -- a governance system for multi-agent AI. The framework provides constitutional constraints that agents cannot override, structured decision graphs for full auditability, and compliance enforcement via code hooks. The hedge fund demonstrates that governance is not overhead -- it is the mechanism that makes autonomous portfolio decisions trustworthy and reproducible.

## License

MIT -- see [LICENSE](LICENSE).
