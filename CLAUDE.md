# COVENANT HEDGE FUND -- PROJECT CONSTITUTION

> This file supplements the Covenant Framework's full Constitution.
> It cannot weaken the Constitution. It adds project-specific rules.

---

## Project Identity

This is an AI hedge fund governed by the Covenant Framework. Every analyst
is a registered agent. Every trade decision is traceable through the
decision graph. Risk is enforced by COMPLIANCE.md, not by LLM judgment.

---

## Agent Definitions

### Epoch Containers (Generation 1)

Three epoch containers manage analyst groups, respecting the 8-sibling limit:

| Container | Domain | Children |
|---|---|---|
| `epoch-value` | value | buffett, graham, munger, pabrai, fisher, damodaran |
| `epoch-quant` | quant | technicals, fundamentals, valuation, growth, sentiment |
| `epoch-macro` | macro | druckenmiller, burry, wood, lynch, ackman, taleb, news-sentiment |

### Analyst Agents (Generation 2)

All analysts produce a uniform signal per ticker:

```json
{
  "signal": "bullish | bearish | neutral",
  "confidence": 0-100,
  "reasoning": "concise explanation (max 120 chars)"
}
```

**Value Domain** -- LLM-augmented analysts. Each encodes an investor's
philosophy as a system prompt. They receive pre-computed financial data
and render judgment.

- `analyst-buffett` -- Circle of competence, moat, margin of safety
- `analyst-graham` -- Net-net valuation, margin of safety, balance sheet strength
- `analyst-munger` -- Mental models, competitive advantages, management quality
- `analyst-pabrai` -- Dhandho framework, low risk/high uncertainty, cloning
- `analyst-fisher` -- Scuttlebutt method, growth potential, management integrity
- `analyst-damodaran` -- DCF valuation, risk-adjusted returns, narrative + numbers

**Quant Domain** -- Pure computation analysts. No LLM calls.

- `analyst-technicals` -- RSI, Bollinger Bands, EMA crossovers, ADX, Hurst exponent
- `analyst-fundamentals` -- Financial ratio scoring from API metrics
- `analyst-valuation` -- DCF and comparable company analysis
- `analyst-growth` -- Revenue/earnings growth rate scoring
- `analyst-sentiment` -- Insider trade direction + news sentiment aggregation

**Macro/Contrarian Domain** -- LLM-augmented analysts with macro focus.

- `analyst-druckenmiller` -- Top-down macro, asymmetric risk/reward, position sizing
- `analyst-burry` -- Deep value, contrarian bets, balance sheet forensics
- `analyst-wood` -- Disruptive innovation, 5-year time horizon, exponential growth
- `analyst-lynch` -- GARP, PEG ratio, invest in what you know
- `analyst-ackman` -- Activist lens, catalysts, business quality at fair price
- `analyst-taleb` -- Antifragility, tail risk, optionality, black swan exposure
- `analyst-news-sentiment` -- LLM-analyzed news sentiment with source weighting

---

## Portfolio Synthesis

The Interpreter acts as portfolio manager:

1. Reads all analyst memos from `memory/memos/`
2. Applies COMPLIANCE.md risk rules (position limits, correlation constraints)
3. Computes allowed actions per ticker deterministically
4. Makes trade decisions with full signal context
5. Writes trade decisions to exit report with decision graph links

The Interpreter does NOT delegate portfolio decisions to a child agent.
Risk enforcement happens before decision-making, not after.

---

## Signal Schema

The standard interface between analysts and the Interpreter:

```python
class AnalystSignal:
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int  # 0-100
    reasoning: str   # max 120 characters
```

Analysts write their signals as structured memos. Each memo includes:
- Signal ontology type: `convergence` (bullish), `tension` (bearish), or neither (neutral)
- Confidence score mapped to signal ontology confidence (0.0-1.0 = confidence/100)
- Ticker as the subject

---

## Risk Enforcement

Risk is NOT an agent. Risk is law.

All risk rules live in COMPLIANCE.md with CF-COMP identifiers.
The risk calculation engine (`src/risk.py`) is pure Python -- no LLM.
Hooks enforce compliance before any trade execution.

See COMPLIANCE.md for the full rule set.

---

## Data Source

All market data comes from financialdatasets.ai.
The API client is trusted at Sojourner level (Progressive Trust, Constitution XXXII-A).
Cache all API responses within a session to avoid redundant calls.

---

## Rules

1. Analysts MUST NOT communicate laterally. They write memos; they do not read sibling memos.
2. The Interpreter reads ALL analyst memos before making portfolio decisions.
3. Risk limits are hard constraints. The Interpreter cannot override COMPLIANCE.md rules.
4. Quant analysts MUST NOT make LLM calls. They are pure computation.
5. LLM-augmented analysts MUST receive pre-computed data, not raw API responses.
6. Reasoning is capped at 120 characters. Concision is a design constraint.
7. Every trade decision MUST include a decision graph node linking to the analyst signals that informed it.
