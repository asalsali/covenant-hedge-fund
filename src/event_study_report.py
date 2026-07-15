"""Formatting and HTML rendering for event study (CAR) results.

Separated from event_study.py to keep the core engine lean.

Usage:
    from src.event_study_report import format_car_results, build_car_html_section
    print(format_car_results(summary))
    html = build_car_html_section(summary)
"""

from __future__ import annotations

from typing import Any

from src.event_study import CARSummary


def format_car_results(summary: CARSummary) -> str:
    """Format CAR results as a readable terminal summary."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("STATISTICAL VALIDATION -- Cumulative Abnormal Returns (CAR)")
    lines.append("=" * 70)
    lines.append(f"  Events analyzed:     {summary.n_events}")
    lines.append(f"  Significant (p<.05): {summary.n_significant}")
    lines.append("")

    # Per-event compact table
    if summary.events:
        lines.append("  PER-EVENT RESULTS")
        lines.append("  " + "-" * 66)

        windows_present: list[tuple[int, int]] = []
        for event in summary.events:
            if not event.error:
                windows_present = [wr.window for wr in event.windows]
                break

        if windows_present:
            header = f"  {'Date':<12} {'Ticker':<8} {'Dir':<5}"
            for w in windows_present:
                header += f" {'CAR[' + str(w[0]) + ',' + str(w[1]) + ']':>14}"
            lines.append(header)
            lines.append("  " + "-" * 66)

            for event in summary.events:
                if event.error:
                    lines.append(
                        f"  {event.event_date:<12} {event.ticker:<8} "
                        f"{'--':<5} ERROR: {event.error}"
                    )
                    continue
                row = f"  {event.event_date:<12} {event.ticker:<8} {event.action:<5}"
                for wr in event.windows:
                    car_str = f"{wr.car:+.4f}{wr.significance}"
                    row += f" {car_str:>14}"
                lines.append(row)

    lines.append("")

    # Aggregate
    if summary.aggregate:
        lines.append("  AGGREGATE STATISTICS")
        lines.append("  " + "-" * 66)
        for label, agg in summary.aggregate.items():
            sig = ""
            if agg["agg_p_value"] < 0.001:
                sig = "***"
            elif agg["agg_p_value"] < 0.01:
                sig = "**"
            elif agg["agg_p_value"] < 0.05:
                sig = "*"
            lines.append(f"  Window {label}:")
            lines.append(f"    Mean CAR:       {agg['mean_car']:+.4f}{sig}")
            lines.append(f"    Median CAR:     {agg['median_car']:+.4f}")
            lines.append(f"    Std Dev:        {agg['std_car']:.4f}")
            lines.append(f"    t-stat:         {agg['agg_t_stat']:.4f}")
            lines.append(f"    p-value:        {agg['agg_p_value']:.4f}")
            lines.append(
                f"    Events:         {int(agg['n_events'])} "
                f"({int(agg['n_significant'])} significant)"
            )
            lines.append(f"    % Positive CAR: {agg['pct_positive']:.0f}%")
            lines.append("")

    lines.append("  VERDICT")
    lines.append("  " + "-" * 66)
    lines.append(f"  {summary.verdict}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def car_summary_to_dict(summary: CARSummary) -> dict[str, Any]:
    """Convert CARSummary to a JSON-serializable dict."""
    events = []
    for event in summary.events:
        if event.error:
            events.append({
                "ticker": event.ticker, "date": event.event_date,
                "action": event.action, "error": event.error,
            })
            continue
        windows = [{
            "window": f"[{wr.window[0]},{wr.window[1]}]",
            "car": round(wr.car, 6), "t_stat": round(wr.t_stat, 4),
            "p_value": round(wr.p_value, 4),
            "significance": wr.significance, "n_days": wr.n_days,
        } for wr in event.windows]

        model_info = None
        if event.model:
            model_info = {
                "alpha": round(event.model.alpha, 6),
                "beta": round(event.model.beta, 4),
                "r_squared": round(event.model.r_squared, 4),
                "residual_std": round(event.model.residual_std, 6),
                "n_obs": event.model.n_obs,
            }
        events.append({
            "ticker": event.ticker, "date": event.event_date,
            "action": event.action, "windows": windows,
            "model": model_info,
            "bootstrap_ci": {
                k: [round(v[0], 6), round(v[1], 6)]
                for k, v in event.bootstrap_ci.items()
            },
        })

    return {
        "n_events": summary.n_events,
        "n_significant": summary.n_significant,
        "verdict": summary.verdict,
        "events": events,
        "aggregate": {
            k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in summary.aggregate.items()
        },
    }


def build_car_html_section(summary: CARSummary) -> str:
    """Build HTML section for the backtest report."""
    if summary.n_events == 0:
        return (
            '<div class="section"><h2>Statistical Validation (CAR)</h2>'
            '<p style="color: var(--text-muted);">No events available.</p>'
            '</div>'
        )

    # Verdict styling
    if "CONFIRMED" in summary.verdict and "NOT" not in summary.verdict:
        v_color, v_icon = "var(--positive)", "PASS"
    elif "INCONCLUSIVE" in summary.verdict:
        v_color, v_icon = "var(--warning)", "MIXED"
    else:
        v_color, v_icon = "var(--negative)", "FAIL"

    html = f"""<div class="section">
    <h2>Statistical Validation (CAR)
        <span style="color: {v_color}; font-size: 0.8rem;
              font-weight: 600; margin-left: 12px; padding: 2px 10px;
              border: 1px solid {v_color}; border-radius: 4px;">
            {v_icon}</span>
    </h2>
    <p style="color: var(--text-secondary); margin-bottom: 16px;">
        Cumulative Abnormal Returns measure whether signals produce
        returns beyond what market exposure (beta) explains.</p>

    <div style="background: var(--bg-secondary); border: 1px solid var(--border);
                border-radius: 8px; padding: 16px; margin-bottom: 20px;">
        <p style="color: {v_color}; font-weight: 600; margin: 0;">
            {summary.verdict}</p>
        <p style="color: var(--text-muted); margin: 8px 0 0 0; font-size: 0.85rem;">
            {summary.n_events} events analyzed, {summary.n_significant} significant (p&lt;0.05)</p>
    </div>
"""

    # Aggregate table
    if summary.aggregate:
        html += _build_agg_table(summary)

    # Per-event details
    valid = [e for e in summary.events if not e.error]
    if valid:
        html += _build_event_details(valid)

    html += "</div>"
    return html


def _build_agg_table(summary: CARSummary) -> str:
    """Aggregate results table."""
    html = """
    <h3 style="color: var(--text-primary); margin-bottom: 12px;">Aggregate Results</h3>
    <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
    <thead><tr style="border-bottom: 1px solid var(--border);">
        <th style="text-align:left;padding:8px;color:var(--text-secondary);">Window</th>
        <th style="text-align:right;padding:8px;color:var(--text-secondary);">Mean CAR</th>
        <th style="text-align:right;padding:8px;color:var(--text-secondary);">t-stat</th>
        <th style="text-align:right;padding:8px;color:var(--text-secondary);">p-value</th>
        <th style="text-align:right;padding:8px;color:var(--text-secondary);">% Positive</th>
        <th style="text-align:right;padding:8px;color:var(--text-secondary);">Sig.</th>
    </tr></thead><tbody>
"""
    for label, agg in summary.aggregate.items():
        cc = "var(--positive)" if agg["mean_car"] > 0 else "var(--negative)"
        sig = ""
        if agg["agg_p_value"] < 0.001:
            sig = " ***"
        elif agg["agg_p_value"] < 0.01:
            sig = " **"
        elif agg["agg_p_value"] < 0.05:
            sig = " *"
        html += (
            f'    <tr style="border-bottom:1px solid var(--border);">'
            f'<td style="padding:8px;color:var(--text-primary);">{label}</td>'
            f'<td style="padding:8px;text-align:right;color:{cc};">'
            f'{agg["mean_car"]:+.4f}{sig}</td>'
            f'<td style="padding:8px;text-align:right;color:var(--text-primary);">'
            f'{agg["agg_t_stat"]:.3f}</td>'
            f'<td style="padding:8px;text-align:right;color:var(--text-primary);">'
            f'{agg["agg_p_value"]:.4f}</td>'
            f'<td style="padding:8px;text-align:right;color:var(--text-primary);">'
            f'{agg["pct_positive"]:.0f}%</td>'
            f'<td style="padding:8px;text-align:right;color:var(--text-primary);">'
            f'{int(agg["n_significant"])}/{int(agg["n_events"])}</td>'
            f'</tr>\n'
        )
    html += "    </tbody></table></div>\n"
    return html


def _build_event_details(events: list) -> str:
    """Collapsible per-event details table."""
    n = len(events)
    windows = events[0].windows if events else []

    html = f"""
    <details style="margin-top: 20px;">
        <summary style="cursor:pointer;color:var(--accent);font-weight:500;
                        padding:8px 0;">Per-Event Details ({n} events)</summary>
        <div style="overflow-x:auto;margin-top:12px;">
        <table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
        <thead><tr style="border-bottom:1px solid var(--border);">
            <th style="text-align:left;padding:6px;color:var(--text-secondary);">Date</th>
            <th style="text-align:left;padding:6px;color:var(--text-secondary);">Ticker</th>
            <th style="text-align:left;padding:6px;color:var(--text-secondary);">Dir</th>
            <th style="text-align:right;padding:6px;color:var(--text-secondary);">Beta</th>
"""
    for wr in windows:
        html += (
            f'            <th style="text-align:right;padding:6px;'
            f'color:var(--text-secondary);">CAR [{wr.window[0]},{wr.window[1]}]</th>\n'
        )
    html += "        </tr></thead><tbody>\n"

    for event in events:
        beta = f"{event.model.beta:.2f}" if event.model else "N/A"
        html += (
            f'        <tr style="border-bottom:1px solid var(--border);">'
            f'<td style="padding:6px;color:var(--text-primary);">{event.event_date}</td>'
            f'<td style="padding:6px;color:var(--text-primary);">{event.ticker}</td>'
            f'<td style="padding:6px;color:var(--text-primary);">{event.action}</td>'
            f'<td style="padding:6px;text-align:right;color:var(--text-primary);">{beta}</td>'
        )
        for wr in event.windows:
            cc = "var(--positive)" if wr.car > 0 else "var(--negative)"
            html += (
                f'<td style="padding:6px;text-align:right;color:{cc};">'
                f'{wr.car:+.4f}{wr.significance}</td>'
            )
        html += "</tr>\n"

    html += "        </tbody></table></div>\n    </details>\n"
    return html
