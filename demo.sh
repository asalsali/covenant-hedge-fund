#!/usr/bin/env bash
# Covenant Hedge Fund -- Demo Run
# Runs the full 18-analyst pipeline on 3 diversified tickers.
#
# Tickers chosen for sector diversity:
#   AAPL (Technology), JPM (Financials), XOM (Energy)
#
# No LLM API keys required -- runs in quant-only mode by default.
# Start Ollama to enable value and macro analysts.
# No paid API keys needed -- Ollama is free and local.

set -e

echo "======================================================================"
echo "COVENANT HEDGE FUND -- Demo Run"
echo "======================================================================"
echo ""
echo "  Pipeline:  18 analysts across 3 domains (quant, value, macro)"
echo "  Tickers:   AAPL (Tech), JPM (Finance), XOM (Energy)"
echo "  Mode:      Single analysis with reasoning"
echo ""
echo "  Risk enforcement: volatility-adjusted sizing, correlation limits"
echo "  Decision method:  confidence-weighted scoring (deterministic)"
echo ""
echo "----------------------------------------------------------------------"
echo ""

python -m src.main --tickers AAPL JPM XOM --show-reasoning

echo ""
echo "======================================================================"
echo "Demo complete. Run 'python -m src.main --help' for options."
echo "======================================================================"
