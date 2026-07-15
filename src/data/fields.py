"""API field stripping utility for reducing LLM prompt token usage.

Strips unused fields from API responses before they are injected into
LLM prompts. This is opt-in -- analysts call strip_fields() on data
they want to slim down. No global interceptor.

Usage:
    from src.data.fields import strip_fields, PRICE_FIELDS, METRIC_FIELDS

    stripped = strip_fields(raw_prices, PRICE_FIELDS)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Field allowlists by data type
# ---------------------------------------------------------------------------

#: Price bar fields actually used by analysts.
PRICE_FIELDS: set[str] = {
    "date",
    "close",
    "volume",
    "open",
    "high",
    "low",
}

#: Financial metric fields used by value/macro fact extractors.
METRIC_FIELDS: set[str] = {
    "report_period",
    "period",
    "return_on_equity",
    "return_on_assets",
    "debt_to_equity",
    "current_ratio",
    "net_margin",
    "gross_margin",
    "operating_margin",
    "price_to_earnings",
    "ev_to_ebitda",
    "market_cap",
    "earnings_growth",
    "revenue_growth",
    "asset_turnover",
}

#: Line item fields used by analysts.
LINE_ITEM_FIELDS: set[str] = {
    "period_end_date",
    "report_period",
    "revenue",
    "net_income",
    "free_cash_flow",
    "operating_cash_flow",
    "outstanding_shares",
    "weighted_average_shares",
}

#: Insider trade fields used by sentiment/activist analysts.
INSIDER_TRADE_FIELDS: set[str] = {
    "transaction_type",
    "shares",
    "price_per_share",
    "date",
    "transaction_date",
    "owner",
}

#: Crypto metric fields used by crypto analysts.
CRYPTO_METRIC_FIELDS: set[str] = {
    "market_cap",
    "market_cap_rank",
    "fully_diluted_valuation",
    "total_volume",
    "circulating_supply",
    "total_supply",
    "max_supply",
    "ath",
    "ath_change_percentage",
    "atl",
    "atl_change_percentage",
    "price_change_percentage_7d",
    "price_change_percentage_14d",
    "price_change_percentage_30d",
    "price_change_percentage_60d",
    "price_change_percentage_200d",
    "price_change_percentage_1y",
}


# ---------------------------------------------------------------------------
# Core utility
# ---------------------------------------------------------------------------

def strip_fields(
    data: Any,
    keep_fields: set[str],
) -> Any:
    """Recursively strip fields not in keep_fields from API response data.

    Handles three shapes:
      - dict: retains only keys present in keep_fields
      - list of dicts: applies stripping to each element
      - anything else: returned unchanged

    Args:
        data: Raw API response data (dict, list of dicts, or scalar).
        keep_fields: Set of field names to retain.

    Returns:
        Stripped copy of the data. Original is not mutated.

    Examples:
        >>> strip_fields({"close": 150.0, "adj_close": 149.8}, {"close"})
        {'close': 150.0}

        >>> strip_fields([{"a": 1, "b": 2}], {"a"})
        [{'a': 1}]
    """
    if isinstance(data, dict):
        return {
            k: v for k, v in data.items()
            if k in keep_fields
        }
    if isinstance(data, list):
        return [
            strip_fields(item, keep_fields)
            if isinstance(item, dict)
            else item
            for item in data
        ]
    return data
