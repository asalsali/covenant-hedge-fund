"""HTML report generator for the Covenant Hedge Fund backtest.

Generates a single self-contained HTML file with:
- Performance summary cards
- Equity curve chart (Chart.js via CDN)
- Signal heatmap (last rebalance date)
- Trade log table
- Per-ticker detail sections

Usage:
    from src.report import generate_report
    report_path = generate_report(report_data)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def _fmt_pct(val: float | None) -> str:
    """Format a decimal as a percentage string."""
    if val is None:
        return "N/A"
    return f"{val:+.2%}"


def _fmt_number(val: float | None, decimals: int = 2) -> str:
    """Format a number with commas."""
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f}"


def _color_class(val: float | None) -> str:
    """Return CSS class for positive/negative coloring."""
    if val is None:
        return ""
    return "positive" if val >= 0 else "negative"


def _signal_color(signal: str, confidence: int) -> str:
    """Return background color for a signal cell in the heatmap.

    Green for bullish, red for bearish, neutral gray.
    Intensity scales with confidence (0-100).
    """
    alpha = min(confidence / 100.0, 1.0) * 0.7 + 0.1
    if signal == "bullish":
        return f"rgba(34, 197, 94, {alpha:.2f})"
    elif signal == "bearish":
        return f"rgba(239, 68, 68, {alpha:.2f})"
    return "rgba(148, 163, 184, 0.15)"


def _build_head() -> str:
    """Build the HTML <head> section with inline CSS."""
    return """<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
<style>
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-card: #21253a;
    --bg-hover: #2a2f45;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #2d3348;
    --accent: #3b82f6;
    --positive: #22c55e;
    --negative: #ef4444;
    --warning: #f59e0b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}
h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; color: var(--text-primary); }
h3 { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-secondary); }
.header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }
.header .subtitle { color: var(--text-secondary); font-size: 0.9rem; }
.header .meta { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem; }

/* Cards grid */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
}
.card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.25rem; }
.card .value { font-size: 1.75rem; font-weight: 700; }
.card .value.positive { color: var(--positive); }
.card .value.negative { color: var(--negative); }

/* Chart */
.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    position: relative;
    height: 400px;
}

/* Section */
.section { margin-bottom: 2rem; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th {
    text-align: left;
    padding: 0.625rem 0.75rem;
    border-bottom: 2px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    cursor: pointer;
    user-select: none;
}
th:hover { color: var(--text-secondary); }
td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
}
tr:hover td { background: var(--bg-hover); }
.action-buy { color: var(--positive); font-weight: 600; }
.action-sell, .action-short { color: var(--negative); font-weight: 600; }
.action-cover { color: var(--warning); font-weight: 600; }

/* Heatmap */
.heatmap-container { overflow-x: auto; }
.heatmap { border-collapse: collapse; font-size: 0.8rem; }
.heatmap th { padding: 0.5rem 0.625rem; white-space: nowrap; }
.heatmap td {
    padding: 0.375rem 0.5rem;
    text-align: center;
    border: 1px solid var(--border);
    min-width: 80px;
    font-size: 0.75rem;
    font-weight: 500;
}
.heatmap .domain-header {
    background: var(--bg-secondary);
    color: var(--text-muted);
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
}

/* Ticker details */
.ticker-detail { margin-bottom: 1rem; }
.ticker-toggle {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    color: var(--text-primary);
    cursor: pointer;
    width: 100%;
    text-align: left;
    font-size: 0.9rem;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ticker-toggle:hover { background: var(--bg-hover); }
.ticker-content {
    display: none;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 1rem;
}
.ticker-content.open { display: block; }
.signal-item {
    margin-bottom: 0.75rem;
    padding: 0.625rem;
    background: var(--bg-card);
    border-radius: 4px;
    border-left: 3px solid var(--border);
}
.signal-item.bullish { border-left-color: var(--positive); }
.signal-item.bearish { border-left-color: var(--negative); }
.signal-item .analyst-name { font-weight: 600; font-size: 0.85rem; }
.signal-item .signal-meta { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.125rem; }
.signal-item .reasoning { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; }

/* Footer */
.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.75rem;
    text-align: center;
}
</style>
</head>"""


def _build_header(metadata: dict[str, Any]) -> str:
    """Build the report header section."""
    tickers = ", ".join(metadata["tickers"])
    mode_label = metadata["mode"].replace("-", " ").title()
    return f"""<div class="header">
    <h1>Backtest Report</h1>
    <div class="subtitle">{mode_label} | {len(metadata['tickers'])} tickers | {metadata['analyst_count']} analysts</div>
    <div class="meta">{metadata['start_date']} to {metadata['end_date']} | {metadata['trading_days']} trading days | Tickers: {tickers}</div>
    <div class="meta">Generated: {metadata['run_date']}</div>
</div>"""


def _build_performance_cards(perf: dict[str, Any]) -> str:
    """Build the performance summary cards."""
    cards = [
        ("Total Return", _fmt_pct(perf.get("total_return")), _color_class(perf.get("total_return"))),
        ("Sharpe Ratio", _fmt_number(perf.get("sharpe_ratio"), 4), _color_class(perf.get("sharpe_ratio"))),
        ("Sortino Ratio", _fmt_number(perf.get("sortino_ratio"), 4), _color_class(perf.get("sortino_ratio"))),
        ("Max Drawdown", _fmt_pct(perf.get("max_drawdown")), "negative" if perf.get("max_drawdown") else ""),
        ("Alpha vs SPY", _fmt_pct(perf.get("alpha_vs_spy")), _color_class(perf.get("alpha_vs_spy"))),
        ("Final Value", f"${_fmt_number(perf.get('final_value'))}", _color_class((perf.get("total_return") or 0))),
    ]

    html = '<div class="cards">\n'
    for label, value, cls in cards:
        html += f"""    <div class="card">
        <div class="label">{label}</div>
        <div class="value {cls}">{value}</div>
    </div>\n"""
    html += "</div>"
    return html


def _build_equity_chart(equity_curve: list[dict[str, Any]], trades: list[dict[str, Any]]) -> str:
    """Build the equity curve Chart.js section."""
    # Prepare data arrays
    dates = [p["date"] for p in equity_curve]
    values = [p["value"] for p in equity_curve]
    spy_values = [p.get("spy_value") for p in equity_curve]
    has_spy = any(v is not None and v > 0 for v in spy_values)

    # Serialize to JSON for JavaScript embedding
    dates_json = json.dumps(dates)
    values_json = json.dumps(values)

    # Trade markers
    buy_points = []
    sell_points = []
    for t in trades:
        # Find the closest equity curve date
        trade_date = t["date"]
        for ec in equity_curve:
            if ec["date"] == trade_date:
                point = f'{{x: "{trade_date}", y: {ec["value"]}}}'
                if t["action"] == "buy":
                    buy_points.append(point)
                else:
                    sell_points.append(point)
                break

    datasets = f"""{{
                label: 'Portfolio',
                data: {values_json},
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.1,
            }}"""

    if has_spy:
        spy_data_json = json.dumps([v if v else None for v in spy_values])
        datasets += f""",
            {{
                label: 'SPY (normalized)',
                data: {spy_data_json},
                borderColor: '#64748b',
                borderWidth: 1.5,
                pointRadius: 0,
                borderDash: [5, 3],
                fill: false,
                tension: 0.1,
            }}"""

    if buy_points:
        datasets += f""",
            {{
                label: 'Buy',
                data: [{', '.join(buy_points)}],
                backgroundColor: '#22c55e',
                borderColor: '#22c55e',
                pointRadius: 5,
                pointStyle: 'triangle',
                showLine: false,
            }}"""

    if sell_points:
        datasets += f""",
            {{
                label: 'Sell/Short',
                data: [{', '.join(sell_points)}],
                backgroundColor: '#ef4444',
                borderColor: '#ef4444',
                pointRadius: 5,
                pointStyle: 'triangle',
                rotation: 180,
                showLine: false,
            }}"""

    return f"""<div class="section">
    <h2>Equity Curve</h2>
    <div class="chart-container">
        <canvas id="equityChart"></canvas>
    </div>
</div>
<script>
(function() {{
    const ctx = document.getElementById('equityChart').getContext('2d');
    const labels = {dates_json};
    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: labels,
            datasets: [{datasets}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            scales: {{
                x: {{
                    type: 'time',
                    time: {{ unit: 'month', tooltipFormat: 'yyyy-MM-dd' }},
                    grid: {{ color: 'rgba(45, 51, 72, 0.5)' }},
                    ticks: {{ color: '#64748b' }},
                }},
                y: {{
                    grid: {{ color: 'rgba(45, 51, 72, 0.5)' }},
                    ticks: {{
                        color: '#64748b',
                        callback: function(v) {{ return '$' + v.toLocaleString(); }}
                    }},
                }}
            }},
            plugins: {{
                legend: {{
                    labels: {{ color: '#94a3b8', usePointStyle: true, padding: 16 }}
                }},
                tooltip: {{
                    backgroundColor: '#1a1d29',
                    titleColor: '#e2e8f0',
                    bodyColor: '#94a3b8',
                    borderColor: '#2d3348',
                    borderWidth: 1,
                    callbacks: {{
                        label: function(ctx) {{
                            if (ctx.parsed.y !== null) {{
                                return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }});
}})();
</script>"""


def _build_signal_heatmap(signals: dict[str, Any], analysts: list[dict[str, Any]]) -> str:
    """Build the signal heatmap for the last rebalance date."""
    if not signals:
        return '<div class="section"><h2>Signal Heatmap</h2><p style="color: var(--text-muted);">No signal data available.</p></div>'

    # Get last rebalance date
    sorted_dates = sorted(signals.keys())
    last_date = sorted_dates[-1]
    last_signals = signals[last_date]

    tickers = sorted(last_signals.keys())
    if not tickers:
        return ""

    # Group analysts by domain
    domain_groups: dict[str, list[str]] = {}
    analyst_domain_map: dict[str, str] = {}
    for a in analysts:
        domain_groups.setdefault(a["domain"], []).append(a["name"])
        analyst_domain_map[a["name"]] = a["domain"]

    # Collect all analyst names that appear in the signals
    all_analyst_names: list[str] = []
    for ticker_sigs in last_signals.values():
        for name in ticker_sigs:
            if name not in all_analyst_names:
                all_analyst_names.append(name)

    # Order by domain
    domain_order = ["quant", "value", "macro"]
    ordered_analysts: list[tuple[str, str]] = []  # (name, domain)
    for domain in domain_order:
        for name in all_analyst_names:
            d = analyst_domain_map.get(name, "other")
            if d == domain:
                ordered_analysts.append((name, domain))
    # Add any remaining
    seen = {n for n, _ in ordered_analysts}
    for name in all_analyst_names:
        if name not in seen:
            d = analyst_domain_map.get(name, "other")
            ordered_analysts.append((name, d))

    # Build table
    html = f"""<div class="section">
    <h2>Signal Heatmap <span style="color: var(--text-muted); font-size: 0.8rem; font-weight: 400;">({last_date})</span></h2>
    <div class="heatmap-container">
    <table class="heatmap">
    <thead><tr><th>Analyst</th>"""

    for ticker in tickers:
        html += f"<th>{ticker}</th>"
    html += "</tr></thead>\n<tbody>\n"

    current_domain = ""
    for name, domain in ordered_analysts:
        if domain != current_domain:
            current_domain = domain
            html += f'<tr><td colspan="{len(tickers) + 1}" class="domain-header">{domain}</td></tr>\n'

        html += f"<tr><td style='white-space:nowrap; color: var(--text-primary); font-weight: 500;'>{name}</td>"
        for ticker in tickers:
            sig_data = last_signals.get(ticker, {}).get(name)
            if sig_data:
                sig = sig_data["signal"]
                conf = sig_data["confidence"]
                bg = _signal_color(sig, conf)
                abbrev = {"bullish": "BL", "bearish": "BR", "neutral": "N"}.get(sig, sig[0].upper())
                label = f"{abbrev} {conf}"
                html += f'<td style="background: {bg};" title="{sig} {conf}%">{label}</td>'
            else:
                html += '<td style="background: rgba(148, 163, 184, 0.05);">--</td>'
        html += "</tr>\n"

    html += "</tbody></table></div></div>"
    return html


def _build_trade_table(trades: list[dict[str, Any]]) -> str:
    """Build the sortable trade log table."""
    if not trades:
        return '<div class="section"><h2>Trade Log</h2><p style="color: var(--text-muted);">No trades executed.</p></div>'

    html = """<div class="section">
    <h2>Trade Log</h2>
    <table id="tradeTable">
    <thead><tr>
        <th onclick="sortTable(0)">Date</th>
        <th onclick="sortTable(1)">Ticker</th>
        <th onclick="sortTable(2)">Action</th>
        <th onclick="sortTable(3)">Shares</th>
        <th onclick="sortTable(4)">Price</th>
        <th onclick="sortTable(5)">Notional</th>
        <th>Reasoning</th>
    </tr></thead>
    <tbody>
"""
    for t in trades:
        action_cls = f"action-{t['action']}"
        html += f"""    <tr>
        <td>{t['date']}</td>
        <td style="font-weight: 600;">{t['ticker']}</td>
        <td class="{action_cls}">{t['action'].upper()}</td>
        <td>{t['quantity']:,}</td>
        <td>${t['price']:,.2f}</td>
        <td>${t['notional']:,.2f}</td>
        <td style="max-width: 300px; font-size: 0.8rem;">{t['reasoning']}</td>
    </tr>\n"""

    html += """    </tbody></table>
</div>
<script>
function sortTable(colIdx) {
    const table = document.getElementById('tradeTable');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
    table.dataset.sortDir = dir;
    rows.sort((a, b) => {
        let aVal = a.cells[colIdx].textContent.replace(/[$,]/g, '');
        let bVal = b.cells[colIdx].textContent.replace(/[$,]/g, '');
        const aNum = parseFloat(aVal);
        const bNum = parseFloat(bVal);
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return dir === 'asc' ? aNum - bNum : bNum - aNum;
        }
        return dir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    rows.forEach(r => tbody.appendChild(r));
}
</script>"""
    return html


def _build_ticker_details(signals: dict[str, Any]) -> str:
    """Build expandable per-ticker detail sections."""
    if not signals:
        return ""

    sorted_dates = sorted(signals.keys())
    last_date = sorted_dates[-1]
    last_signals = signals[last_date]
    tickers = sorted(last_signals.keys())

    if not tickers:
        return ""

    html = '<div class="section"><h2>Per-Ticker Analysis</h2>\n'

    for ticker in tickers:
        ticker_sigs = last_signals.get(ticker, {})
        if not ticker_sigs:
            continue

        # Count bullish/bearish/neutral
        counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for s in ticker_sigs.values():
            counts[s["signal"]] = counts.get(s["signal"], 0) + 1

        summary = f"B:{counts['bullish']} / N:{counts['neutral']} / S:{counts['bearish']}"

        html += f"""<div class="ticker-detail">
    <button class="ticker-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
        <span>{ticker} <span style="color: var(--text-muted); font-weight: 400; font-size: 0.8rem;">({summary})</span></span>
        <span style="color: var(--text-muted);">&#9660;</span>
    </button>
    <div class="ticker-content">\n"""

        for analyst_name, sig in sorted(ticker_sigs.items()):
            signal = sig["signal"]
            conf = sig["confidence"]
            reasoning = sig.get("reasoning", "")
            html += f"""        <div class="signal-item {signal}">
            <div class="analyst-name">{analyst_name}</div>
            <div class="signal-meta">{signal.upper()} | Confidence: {conf}%</div>
            <div class="reasoning">{reasoning}</div>
        </div>\n"""

        html += "    </div>\n</div>\n"

    html += "</div>"
    return html


def generate_report(data: dict[str, Any]) -> str:
    """Generate a self-contained HTML backtest report.

    Args:
        data: Report data from BacktestEngine.to_report_json().

    Returns:
        Path to the generated HTML report file.
    """
    metadata = data["metadata"]
    performance = data["performance"]
    equity_curve = data.get("equity_curve", [])
    trades = data.get("trades", [])
    signals = data.get("signals", {})
    analysts = metadata.get("analysts", [])

    # Build HTML sections
    head = _build_head()
    header = _build_header(metadata)
    cards = _build_performance_cards(performance)
    chart = _build_equity_chart(equity_curve, trades)
    heatmap = _build_signal_heatmap(signals, analysts)
    trade_table = _build_trade_table(trades)
    ticker_details = _build_ticker_details(signals)

    footer_text = (
        f"Generated {metadata['run_date']} | "
        f"{metadata['mode'].replace('-', ' ').title()} mode | "
        f"{metadata['analyst_count']} analysts | "
        f"{metadata['trading_days']} trading days"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
{head}
<body>
{header}
{cards}
{chart}
{heatmap}
{trade_table}
{ticker_details}
<div class="footer">{footer_text}</div>
</body>
</html>"""

    # Write to reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-report.html"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.relpath(filepath, os.path.dirname(os.path.dirname(__file__)))
