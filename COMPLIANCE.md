# COMPLIANCE.md -- Covenant Hedge Fund

> Project compliance layer for the Covenant-governed hedge fund.
> Supplements the Constitution (CLAUDE.md). Cannot weaken it.
> All rules use CF-COMP-### identifiers for telemetry and audit.

---

## I. POSITION SIZING RULES

These rules encode the deterministic risk model. They are enforced by hooks,
not by agents. No agent -- including the Interpreter -- may override them.

### CF-COMP-001: Base Position Limit

The base maximum position size for any single ticker is **20% of total
portfolio value**. All subsequent adjustments multiply against this base.

```
base_position_pct = 0.20
```

### CF-COMP-002: Volatility-Adjusted Position Sizing

Volatility is calculated as the **60-day rolling standard deviation of daily
returns, annualized by multiplying by sqrt(252)**.

```
annualized_vol = std(daily_returns[-60:]) * sqrt(252)
```

The volatility tier table determines the adjustment multiplier:

| Annualized Volatility | Multiplier | Effective Max Position |
|---|---|---|
| < 15% | 1.25x | 25.0% |
| 15% -- 30% | Linear from 1.00x to 0.625x | 20.0% -- 12.5% |
| 30% -- 50% | Linear from 0.75x to 0.25x | 15.0% -- 5.0% |
| > 50% | 0.50x | 10.0% |

**Linear interpolation formula** for ranges:

- 15%--30% range: `multiplier = 1.0 - ((vol - 0.15) / 0.15) * 0.375`
- 30%--50% range: `multiplier = 0.75 - ((vol - 0.30) / 0.20) * 0.50`

The volatility-adjusted position percentage is:

```
vol_adjusted_pct = base_position_pct * vol_multiplier
```

Violation of this rule is a **BLOCK** -- the trade must not execute.

### CF-COMP-003: Correlation-Adjusted Position Limit

After volatility adjustment, the position limit is further adjusted by the
**average pairwise correlation** of the ticker against all other tickers
currently held in the portfolio. Correlation is computed from a cross-
correlation matrix of daily returns across all portfolio tickers.

| Avg Pairwise Correlation | Multiplier | Effect |
|---|---|---|
| >= 0.80 | 0.70x | Reduce limit 30% (high concentration risk) |
| 0.60 -- 0.80 | 0.85x | Reduce limit 15% |
| 0.40 -- 0.60 | 1.00x | Neutral |
| 0.20 -- 0.40 | 1.05x | Increase limit 5% (diversification benefit) |
| < 0.20 | 1.10x | Increase limit 10% |

If the portfolio holds no other positions, the correlation multiplier
defaults to **1.00x**.

```
corr_multiplier = lookup(avg_pairwise_correlation)
```

### CF-COMP-004: Final Position Limit Calculation

The final position limit for a ticker is:

```
position_limit = portfolio_value * vol_adjusted_pct * corr_multiplier
remaining_limit = min(position_limit - current_exposure, available_cash)
```

Where:
- `portfolio_value` = total portfolio value (cash + market value of holdings)
- `current_exposure` = absolute market value of current position in this ticker
- `available_cash` = uninvested cash balance

A trade that would cause `current_exposure` to exceed `position_limit` is
a **BLOCK**. The trade must be sized down to `remaining_limit` or rejected.

### CF-COMP-005: Total Portfolio Exposure Cap

The sum of all position exposures must not exceed **100% of portfolio value**.
No leverage is permitted.

```
sum(abs(position_value[i]) for all i) <= portfolio_value
```

A trade that would breach this cap is a **BLOCK**.

### CF-COMP-006: Minimum Position Threshold

Positions smaller than **0.5% of portfolio value** must not be opened. This
prevents noise trades that consume execution overhead without meaningful
portfolio impact.

---

## II. SIGNAL INTEGRITY RULES

These rules govern the output format of analyst agents. An analyst signal
that fails validation is discarded -- it does not reach the portfolio
synthesis step.

### CF-COMP-010: Analyst Signal Schema

Every analyst agent must produce output conforming to this schema:

```json
{
  "ticker": "<string, uppercase, e.g. AAPL>",
  "signal": "<string, one of: bullish | bearish | neutral>",
  "confidence": "<number, integer, 0-100 inclusive>",
  "reasoning": "<string, non-empty, minimum 20 characters>"
}
```

Validation rules:
- `signal` must be exactly one of: `"bullish"`, `"bearish"`, `"neutral"`.
  Any other value (including capitalization variants) is a **BLOCK**.
- `confidence` must be an integer in the range [0, 100]. Values outside
  this range or non-integer values are a **BLOCK**.
- `reasoning` must be a non-empty string of at least 20 characters.
  Empty or stub reasoning (e.g., "see above") is a **BLOCK**.

### CF-COMP-011: Analyst Independence

Analyst agents must not read the output of other analyst agents for the
same ticker within the same analysis run. Each analyst operates on raw
market data and its own analytical framework. Cross-reading between
analysts contaminates signal independence.

Violation is a **WARN** -- the signal is accepted but flagged as
potentially contaminated in the audit trail.

### CF-COMP-012: Confidence Calibration Disclosure

If an analyst's confidence exceeds **90**, the reasoning field must include
an explicit acknowledgment of what would invalidate the thesis. High
confidence without a falsification condition is a **WARN**.

### CF-COMP-013: Signal Staleness

An analyst signal is valid for **one analysis run only**. Signals from
prior runs must not be carried forward or reused. Each run produces fresh
signals from fresh data.

---

## III. TRADE EXECUTION RULES

These rules govern how analyst signals are synthesized into trade decisions
and how those decisions are executed.

### CF-COMP-020: Interpreter-Only Execution

Only the Interpreter may execute trades. No analyst, writer, or other
agent may place orders. Trade execution is a sovereign Interpreter
function.

Violation is a **BLOCK** -- any agent other than the Interpreter
attempting trade execution must be halted.

### CF-COMP-021: Minimum Signal Quorum

A trade decision requires signals from at least **3 distinct analyst
agents** for the same ticker in the same analysis run. Trades based on
fewer than 3 signals are a **BLOCK**.

The Interpreter may lower the quorum to 2 only if fewer than 3 analyst
types are registered in the system. This exception must be logged with
justification.

### CF-COMP-022: Trade Size Enforcement

The quantity of any trade must not exceed the `remaining_limit` calculated
per CF-COMP-004. The Interpreter must compute the remaining limit at
execution time (not at signal synthesis time) to account for any
intervening price movement.

Violation is a **BLOCK**.

### CF-COMP-023: Trade Decision Logging

Every executed trade must be logged with:

```json
{
  "timestamp": "<ISO 8601>",
  "ticker": "<string>",
  "action": "<buy | sell>",
  "quantity": "<number>",
  "price": "<number>",
  "rationale": "<string summarizing why>",
  "analyst_signals": [
    {
      "analyst_id": "<agent ID>",
      "signal": "<bullish | bearish | neutral>",
      "confidence": "<0-100>"
    }
  ],
  "position_limit_at_execution": "<number>",
  "remaining_limit_at_execution": "<number>",
  "portfolio_value_at_execution": "<number>"
}
```

Trades without complete logging are a **WARN** (trade proceeds, but the
gap is flagged for audit).

### CF-COMP-024: No Trading Without Analysis

The Interpreter must not execute trades based on its own judgment alone.
Every trade must trace back to analyst signals that passed CF-COMP-010
validation. Interpreter-originated trade ideas must be routed through the
analyst pipeline before execution.

### CF-COMP-025: Conflicting Signal Resolution

When analyst signals for a ticker are mixed (e.g., 2 bullish, 1 bearish),
the Interpreter must document the resolution logic in the trade rationale.
Acceptable resolution methods:

- **Confidence-weighted majority**: weight each signal by its confidence score
- **Unanimous-only**: trade only when all signals agree
- **Threshold**: trade only when weighted average confidence exceeds a threshold

The chosen method must be declared in `registry/orientation.json` under
`spiritOfTheWork` at project start and applied consistently within a run.

---

## IV. DATA INTEGRITY RULES

These rules govern the freshness, provenance, and reliability of market
data consumed by analyst agents.

### CF-COMP-030: Fresh Data Requirement

Market data (prices, volumes, fundamentals) must be fetched fresh at the
start of each analysis run. Analysts must not operate on cached data from
prior runs.

"Fresh" means: fetched within the current analysis run's execution window.

### CF-COMP-031: Stale Data Flagging

If any data point is older than **24 hours** for a daily-frequency analysis
run, it must be flagged as stale. Stale data does not block analysis but
triggers a **WARN** that is surfaced in the trade decision log.

For intraday analysis (if implemented), the staleness threshold is **1 hour**.

### CF-COMP-032: Data Source Logging

All external API calls for market data must be logged with:

- Timestamp of the call
- Data source (API name / endpoint)
- Tickers requested
- Response status (success / failure / partial)
- Data date range returned

Logs are written to `registry/data-source-log.json`. Failure to log is
a **WARN**.

### CF-COMP-033: Data Source Failure Handling

If a data source API call fails:

1. Retry once after a 5-second delay
2. If retry fails, log the failure and proceed without that data source
3. If the failure reduces available data below the minimum required for
   volatility calculation (60 trading days of daily returns), the analysis
   for that ticker is a **BLOCK** -- no signal may be produced

### CF-COMP-034: No Fabricated Data

Analyst agents must never fabricate, interpolate, or hallucinate market
data. If data is unavailable, the analyst must report inability to analyze
rather than substitute synthetic values.

Violation is a **BLOCK** and triggers a Constitutional violation report
(Guardian-level).

---

## V. AUDIT TRAIL RULES

These rules ensure that every decision is traceable, every agent's
performance is measurable, and the system learns from its own history.

### CF-COMP-040: Trade-to-Signal Traceability

Every trade decision must reference the specific analyst signal IDs
(agent ID + run ID) that informed it. The chain must be:

```
Market Data -> Analyst Signal -> Portfolio Synthesis -> Trade Execution
```

No link in this chain may be missing. A trade without full traceability is
a **WARN** (trade proceeds, gap flagged).

### CF-COMP-041: Performance Metrics in Exit Reports

When an analyst agent shuts down, its exit report (per Constitution
Section VI) must include these additional fields:

```json
{
  "tradingMetrics": {
    "tickersAnalyzed": ["<list of tickers>"],
    "signalsProduced": "<count>",
    "signalBreakdown": {
      "bullish": "<count>",
      "bearish": "<count>",
      "neutral": "<count>"
    },
    "averageConfidence": "<number>",
    "dataSourcesUsed": ["<list>"]
  }
}
```

When the Interpreter completes a trading run, its exit report must include:

```json
{
  "portfolioMetrics": {
    "tradesExecuted": "<count>",
    "totalPnL": "<number>",
    "sharpeContribution": "<number or null if not calculable>",
    "winRate": "<percentage>",
    "avgPositionSize": "<percentage of portfolio>",
    "maxDrawdown": "<percentage>"
  }
}
```

Missing trading metrics in exit reports are a **WARN**.

### CF-COMP-042: Analyst Accuracy Tracking

Across runs, the system must track each analyst agent type's directional
accuracy:

- For each signal produced, record the ticker's actual return over the
  subsequent holding period
- A bullish signal is "correct" if the return is positive; bearish if
  negative; neutral if abs(return) < 2%
- Accuracy is stored in `registry/analyst-accuracy.json` per agent type

This tracking is updated during Consolidation (Constitution Section V).
Analyst types with accuracy below 40% over 10+ signals should be flagged
for review during the next spawn plan.

### CF-COMP-043: Run Isolation

Each analysis run must have a unique run ID (ISO timestamp or UUID). All
signals, trades, and logs within a run are tagged with this ID. Post-run
audit queries by run ID to reconstruct the full decision chain.

### CF-COMP-044: Compliance Violation Log

All WARN and BLOCK events are logged to `registry/compliance-violations.json`
with:

```json
{
  "timestamp": "<ISO 8601>",
  "ruleId": "<CF-COMP-###>",
  "severity": "<WARN | BLOCK>",
  "agentId": "<agent that triggered the violation>",
  "runId": "<analysis run ID>",
  "details": "<human-readable description>"
}
```

This log is reviewed during Consolidation. Recurring violations of the
same rule indicate a systemic issue requiring Interpreter attention.

---

## VI. PARAMETERS AND CONSTANTS

Collected constants referenced throughout this document. These values are
the starting configuration. The Interpreter may adjust them only with
logged justification and user approval.

| Parameter | Value | Rule Reference |
|---|---|---|
| Base position limit | 20% of portfolio | CF-COMP-001 |
| Volatility lookback | 60 trading days | CF-COMP-002 |
| Annualization factor | sqrt(252) | CF-COMP-002 |
| Total exposure cap | 100% (no leverage) | CF-COMP-005 |
| Minimum position size | 0.5% of portfolio | CF-COMP-006 |
| Min analyst quorum | 3 signals per ticker | CF-COMP-021 |
| Data staleness (daily) | 24 hours | CF-COMP-031 |
| Data staleness (intraday) | 1 hour | CF-COMP-031 |
| Min data for vol calc | 60 trading days | CF-COMP-033 |
| Accuracy review threshold | 40% over 10+ signals | CF-COMP-042 |

---

*"Risk is not an agent's opinion. Risk is math, enforced by law."*
