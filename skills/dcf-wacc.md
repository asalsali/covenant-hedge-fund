---
name: dcf-wacc
description: Sector WACC reference tables for DCF valuation discount rates
analysts: [damodaran, buffett, graham, munger, pabrai, fisher]
---

# DCF Sector WACC Reference Tables

Weighted Average Cost of Capital (WACC) ranges by sector. Used by
valuation analysts to apply sector-appropriate discount rates in DCF
models instead of a blanket 10% for all companies.

## WACC Ranges by Sector

| Sector                  | Low   | Mid    | High  |
|-------------------------|-------|--------|-------|
| Technology              | 9.0%  | 10.5%  | 12.0% |
| Healthcare              | 8.0%  | 9.5%   | 11.0% |
| Financial Services      | 7.0%  | 8.5%   | 10.0% |
| Consumer Discretionary  | 8.0%  | 9.5%   | 11.0% |
| Consumer Staples        | 6.0%  | 7.5%   | 9.0%  |
| Energy                  | 9.0%  | 11.0%  | 13.0% |
| Utilities               | 5.0%  | 6.0%   | 7.0%  |
| Industrials             | 8.0%  | 9.5%   | 11.0% |
| Materials               | 8.0%  | 9.5%   | 11.0% |
| Real Estate             | 6.0%  | 7.5%   | 9.0%  |
| Communication Services  | 8.0%  | 9.5%   | 11.0% |
| Information Technology  | 9.0%  | 10.5%  | 12.0% |

**Default (unknown sector):** 8.0% - 10.0% - 12.0%

## Usage

The `get_sector_wacc(sector)` function in `src/snapshot.py` returns
the WACC range for a sector as a dict with `low`, `mid`, `high` keys.

## Known Gap

The sector field is not currently populated by the data layer.
To activate: add `sector: info.get("sector")` to `_yf_get_financial_metrics()`
in `src/data/api.py` and pass it through `market_data` in `main.py`.
The `build_snapshot()` function already accepts a `sector` parameter.
