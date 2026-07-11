# Covenant Hedge Fund

A constitutionally governed AI hedge fund built on the [Covenant Framework](https://github.com/Covenant-Foundry/covenant-framework). This is a reimplementation of [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) with structural governance: every analyst agent operates under constitutional constraints, risk is enforced by compliance rules (not trust), and every trade decision is auditable through the Covenant decision graph.

## Architecture

The system uses a fan-out/fan-in topology organized into three analyst domains:

**Value Domain** (6 analysts): Buffett, Graham, Munger, Pabrai, Fisher, Damodaran
**Quant Domain** (5 analysts): Technicals, Fundamentals, Valuation, Growth, Sentiment
**Macro/Contrarian Domain** (7 analysts): Druckenmiller, Burry, Wood, Lynch, Ackman, Taleb, News Sentiment

Each domain runs inside a Covenant epoch container (respecting the 8-sibling limit). All analysts produce a uniform signal: `{signal, confidence, reasoning}`. The Interpreter reads all analyst memos and makes portfolio decisions directly, with risk enforced by `COMPLIANCE.md` rules and hooks -- not by an LLM agent.

## What Governance Adds

The original ai-hedge-fund is stateless: no memory across runs, no conflict resolution, no decision audit trail. Covenant governance adds:

- **Agent memory**: Exit reports preserve what each analyst learned. Future runs inherit past findings.
- **Decision traceability**: Every trade links back through the decision graph to the analyst signals that informed it.
- **Risk as law**: Position limits and correlation constraints are constitutional rules, not suggestions to an LLM.
- **Conflict mediation**: When analysts disagree, the system has a structured mediation protocol instead of naive signal averaging.
- **Regression tracking**: Analyst performance is baselined and monitored for drift.

## Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- API keys for at least one LLM provider and the market data API

### Installation

```bash
git clone https://github.com/your-org/covenant-hedge-fund.git
cd covenant-hedge-fund
poetry install
cp .env.example .env
# Edit .env with your API keys
```

### Usage

```bash
# Analyze specific tickers
poetry run python src/main.py --tickers AAPL MSFT GOOGL

# Analyze with date range
poetry run python src/main.py --tickers AAPL --start-date 2025-01-01 --end-date 2025-06-01

# Run backtest
poetry run python src/main.py --tickers AAPL MSFT --backtest --start-date 2025-01-01 --end-date 2025-06-01
```

## Project Structure

```
covenant-hedge-fund/
  src/
    main.py              # Entry point
    models.py            # Pydantic signal and portfolio models
    risk.py              # Deterministic risk calculations
    portfolio.py         # Portfolio state management
    data/
      api.py             # Market data API client
    agents/
      base.py            # Base analyst interface
      value.py           # Value domain analysts (Buffett, Graham, etc.)
      quant.py           # Quant domain analysts (technicals, fundamentals, etc.)
      macro.py           # Macro/contrarian domain analysts
  registry/
    agent-registry.json  # Covenant agent registry
    orientation.json     # Current system orientation
    tribes.json          # Domain definitions
  CLAUDE.md              # Project constitution
  COMPLIANCE.md          # Trading risk rules
```

## Data Source

All market data comes from [financialdatasets.ai](https://financialdatasets.ai). You need an API key to fetch prices, financial metrics, insider trades, and company news.

## License

MIT -- see [LICENSE](LICENSE).
