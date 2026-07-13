"""Covenant Hedge Fund -- Streamlit Web Interface.

Interactive signal exploration and backtest visualization.
Run with: streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Ensure src is importable
sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]

AVAILABLE_TICKERS = [
    "AAPL", "AMZN", "GOOGL", "META", "MSFT",
    "NVDA", "TSLA", "JPM", "JNJ", "V",
    "UNH", "HD", "PG", "MA", "DIS",
    "NFLX", "ADBE", "CRM", "PYPL", "INTC",
    "AMD", "QCOM", "AVGO", "TXN", "COST",
    "NKE", "MRK", "PFE", "ABT", "KO",
    "PEP", "WMT", "MCD", "BA", "CAT",
    "GS", "MS", "BLK", "SCHW", "SPY",
]

SIGNAL_COLORS = {
    "bullish": "#22c55e",
    "bearish": "#ef4444",
    "neutral": "#64748b",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subtract_months(d: date, months: int) -> date:
    """Subtract months from a date."""
    import calendar
    month = d.month - months
    year = d.year
    while month <= 0:
        month += 12
        year -= 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


def _signal_bg(signal: str, confidence: int) -> str:
    """CSS background color for a signal cell."""
    alpha = min(confidence / 100.0, 1.0) * 0.7 + 0.1
    if signal == "bullish":
        return f"rgba(34, 197, 94, {alpha:.2f})"
    elif signal == "bearish":
        return f"rgba(239, 68, 68, {alpha:.2f})"
    return "rgba(148, 163, 184, 0.15)"


def _signal_label(signal: str) -> str:
    """Short label for heatmap cells."""
    return {"bullish": "BL", "bearish": "BR", "neutral": "N"}.get(signal, "?")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Covenant Hedge Fund",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Covenant Hedge Fund")
    st.caption("AI-Governed Portfolio Analysis")

    st.divider()

    tickers = st.multiselect(
        "Tickers",
        options=AVAILABLE_TICKERS,
        default=DEFAULT_TICKERS,
        help="Select tickers to include in the backtest.",
    )

    st.subheader("Date Range")
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start",
            value=_subtract_months(date.today(), 15),
            max_value=date.today() - timedelta(days=30),
        )
    with col_end:
        end_date = st.date_input(
            "End",
            value=date.today(),
            min_value=start_date + timedelta(days=30) if start_date else date.today(),
        )

    initial_cash = st.number_input(
        "Initial Capital ($)",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        format="%d",
    )

    llm_lite = st.toggle(
        "LLM-Lite Mode",
        value=False,
        help="Add 4 LLM personas (Buffett, Graham, Druckenmiller, Taleb) on 5 evenly-spaced rebalance dates. Requires API keys.",
    )

    st.divider()

    run_button = st.button(
        "Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=len(tickers) == 0,
    )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state.results = None

if run_button and tickers:
    with st.spinner("Running backtest... this may take a minute."):
        try:
            from src.backtest import BacktestEngine

            engine = BacktestEngine(
                tickers=list(tickers),
                start_date=start_date,
                end_date=end_date,
                initial_cash=float(initial_cash),
                show_reasoning=False,
                llm_lite=llm_lite,
            )

            t0 = time.time()
            metrics = engine.run()
            elapsed = time.time() - t0

            n_days = len(engine.portfolio.daily_values)
            report_data = engine.to_report_json(metrics, n_days)
            report_data["_elapsed"] = elapsed

            st.session_state.results = report_data

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.session_state.results = None


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

results = st.session_state.results

if results is None:
    st.markdown("## Welcome")
    st.markdown(
        "Configure your backtest parameters in the sidebar and click "
        "**Run Backtest** to begin."
    )
    st.markdown("---")
    st.markdown(
        "This application wraps the Covenant Hedge Fund backtest engine. "
        "It runs the full quant analysis pipeline (5 analysts, risk management, "
        "quorum-based decision making) over historical data and visualizes the results."
    )
    st.stop()


# --- Header ---
perf = results["performance"]
meta = results["metadata"]
elapsed = results.get("_elapsed", 0)

st.markdown("## Backtest Results")
st.caption(
    f"{meta['mode'].upper()} | {', '.join(meta['tickers'])} | "
    f"{meta['start_date']} to {meta['end_date']} | "
    f"{meta['trading_days']} trading days | {elapsed:.1f}s"
)


# --- Performance cards ---
cards = st.columns(6)

with cards[0]:
    val = perf.get("total_return")
    st.metric("Total Return", f"{val:+.2%}" if val is not None else "N/A")

with cards[1]:
    val = perf.get("sharpe_ratio")
    st.metric("Sharpe Ratio", f"{val:.2f}" if val is not None else "N/A")

with cards[2]:
    val = perf.get("sortino_ratio")
    st.metric("Sortino Ratio", f"{val:.2f}" if val is not None else "N/A")

with cards[3]:
    val = perf.get("max_drawdown")
    st.metric("Max Drawdown", f"{val:.2%}" if val is not None else "N/A")

with cards[4]:
    val = perf.get("alpha_vs_spy")
    st.metric("Alpha vs SPY", f"{val:+.2%}" if val is not None else "N/A")

with cards[5]:
    val = perf.get("final_value")
    st.metric("Final Value", f"${val:,.0f}" if val is not None else "N/A")


st.divider()


# --- Equity Curve ---
st.subheader("Equity Curve")

equity_data = results.get("equity_curve", [])
if equity_data:
    eq_df = pd.DataFrame(equity_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_df["date"],
        y=eq_df["value"],
        mode="lines",
        name="Portfolio",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    ))

    if "spy_value" in eq_df.columns and eq_df["spy_value"].notna().any():
        fig.add_trace(go.Scatter(
            x=eq_df["date"],
            y=eq_df["spy_value"],
            mode="lines",
            name="SPY (normalized)",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="Date: %{x}<br>SPY: $%{y:,.0f}<extra></extra>",
        ))

    initial = meta.get("initial_cash", 100_000)
    fig.add_hline(
        y=initial, line_dash="dash", line_color="#64748b",
        annotation_text=f"Initial ${initial:,.0f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d29",
        height=450,
        margin=dict(l=60, r=30, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="#2d3348", title=""),
        yaxis=dict(gridcolor="#2d3348", title="Portfolio Value ($)", tickformat="$,.0f"),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No equity curve data available.")


# --- Signal Heatmap ---
st.subheader("Signal Heatmap")

signals = results.get("signals", {})
if signals:
    # Use the last rebalance date for the heatmap
    sorted_dates = sorted(signals.keys())
    selected_date = st.select_slider(
        "Rebalance Date",
        options=sorted_dates,
        value=sorted_dates[-1],
    )

    date_signals = signals.get(selected_date, {})

    if date_signals:
        # Collect all analyst names across all tickers for this date
        all_analysts = set()
        for ticker_sigs in date_signals.values():
            all_analysts.update(ticker_sigs.keys())
        analyst_list = sorted(all_analysts)
        ticker_list = sorted(date_signals.keys())

        # Build the HTML table
        html = '<table style="width:100%; border-collapse:collapse; font-size:14px;">'
        html += '<tr style="border-bottom:1px solid #2d3348;">'
        html += '<th style="padding:8px; text-align:left; color:#94a3b8;">Analyst</th>'
        for ticker in ticker_list:
            html += f'<th style="padding:8px; text-align:center; color:#e2e8f0;">{ticker}</th>'
        html += '</tr>'

        for analyst in analyst_list:
            html += '<tr style="border-bottom:1px solid #1a1d29;">'
            # Truncate long analyst names
            display_name = analyst[:20] + "..." if len(analyst) > 20 else analyst
            html += f'<td style="padding:6px 8px; color:#94a3b8; font-size:12px;">{display_name}</td>'
            for ticker in ticker_list:
                sig_data = date_signals.get(ticker, {}).get(analyst, {})
                sig = sig_data.get("signal", "neutral")
                conf = sig_data.get("confidence", 0)
                bg = _signal_bg(sig, conf)
                label = _signal_label(sig)
                tooltip = f"{sig} ({conf}%)"
                reasoning = sig_data.get("reasoning", "")
                if reasoning:
                    # Escape quotes for HTML attribute
                    reasoning_clean = reasoning[:80].replace('"', '&quot;')
                    tooltip += f": {reasoning_clean}"
                html += (
                    f'<td style="padding:6px 8px; text-align:center; '
                    f'background:{bg}; border-radius:4px;" '
                    f'title="{tooltip}">'
                    f'<span style="font-weight:600; font-size:12px;">{label}</span>'
                    f'<span style="font-size:10px; color:#94a3b8; display:block;">{conf}%</span>'
                    f'</td>'
                )
            html += '</tr>'

        html += '</table>'
        html += '<p style="font-size:11px; color:#64748b; margin-top:8px;">BL = Bullish, BR = Bearish, N = Neutral. Hover for details.</p>'

        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No signal data for this date.")
else:
    st.info("No signal history available.")


st.divider()


# --- Trade Log ---
st.subheader("Trade Log")

trades = results.get("trades", [])
if trades:
    trades_df = pd.DataFrame(trades)

    # Format columns
    if "notional" in trades_df.columns:
        trades_df["notional"] = trades_df["notional"].apply(lambda x: f"${x:,.2f}")
    if "price" in trades_df.columns:
        trades_df["price"] = trades_df["price"].apply(lambda x: f"${x:.2f}")

    column_order = ["date", "ticker", "action", "quantity", "price", "notional", "reasoning"]
    display_cols = [c for c in column_order if c in trades_df.columns]

    st.dataframe(
        trades_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.TextColumn("Date", width="small"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "action": st.column_config.TextColumn("Action", width="small"),
            "quantity": st.column_config.NumberColumn("Qty", width="small"),
            "price": st.column_config.TextColumn("Price", width="small"),
            "notional": st.column_config.TextColumn("Notional", width="small"),
            "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        },
    )

    st.caption(f"{len(trades)} total trades executed.")
else:
    st.info("No trades were executed during this backtest.")


st.divider()


# --- Per-Ticker Detail ---
st.subheader("Per-Ticker Detail")

for ticker in meta.get("tickers", []):
    with st.expander(f"{ticker}"):
        # Ticker equity contribution: filter trades for this ticker
        ticker_trades = [t for t in trades if t.get("ticker") == ticker]

        if ticker_trades:
            t_df = pd.DataFrame(ticker_trades)
            st.markdown(f"**Trades ({len(ticker_trades)})**")
            st.dataframe(t_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No trades for this ticker.")

        # Signal history for this ticker across all dates
        st.markdown("**Signal History**")
        sig_rows = []
        for dt in sorted(signals.keys()):
            dt_sigs = signals[dt].get(ticker, {})
            for analyst_name, sig_data in dt_sigs.items():
                sig_rows.append({
                    "date": dt,
                    "analyst": analyst_name,
                    "signal": sig_data.get("signal", ""),
                    "confidence": sig_data.get("confidence", 0),
                })

        if sig_rows:
            sig_df = pd.DataFrame(sig_rows)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No signal data for this ticker.")

        # Decision history
        decisions = results.get("decisions", {})
        dec_rows = []
        for dt in sorted(decisions.keys()):
            dt_dec = decisions[dt].get(ticker, {})
            if dt_dec:
                dec_rows.append({
                    "date": dt,
                    "action": dt_dec.get("action", ""),
                    "quantity": dt_dec.get("quantity", 0),
                    "score": f"{dt_dec.get('weighted_score', 0):+.3f}",
                    "reasoning": dt_dec.get("reasoning", ""),
                })

        if dec_rows:
            st.markdown("**Decision History**")
            dec_df = pd.DataFrame(dec_rows)
            st.dataframe(dec_df, use_container_width=True, hide_index=True)
