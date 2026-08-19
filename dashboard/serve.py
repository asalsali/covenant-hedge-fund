"""Simple dashboard server — serves the HTML dashboard + data files.

Usage:
    python dashboard/serve.py
    # Opens at http://localhost:8080/dashboard/
"""

import http.server
import os
import sys
from pathlib import Path

PORT = int(os.environ.get("DASHBOARD_PORT", 8080))

# Serve from the repo root so /data/ and /dashboard/ both work
REPO_ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self):
        # Allow CORS for local dev
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, format, *args):
        # Quieter logging
        if "GET /data/" in (format % args):
            return  # Don't log data polling
        super().log_message(format, *args)


def main():
    print(f"Covenant Trading Dashboard")
    print(f"  URL: http://localhost:{PORT}/dashboard/")
    print(f"  Root: {REPO_ROOT}")
    print(f"  Press Ctrl+C to stop\n")

    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
        server.shutdown()


if __name__ == "__main__":
    main()
