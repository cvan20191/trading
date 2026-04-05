#!/usr/bin/env python3
"""
Append or update today's Mag 7 basket forward P/E in the frontend JSON cache.

Run manually from the macro-dashboard repo root (requires network + FMP_API_KEY
in backend/.env). Use the backend virtualenv so dependencies resolve:

  cd macro-dashboard
  ./backend/.venv/bin/python scripts/refresh_mag7_valuation_history.py

Uses the same basket logic as the live API: fmp_client.compute_mag7_basket.
Does not start the web server and does not change valuation doctrine/rules.

If FMP coverage is insufficient (same as live fallback path), the script exits
without writing a new point — fix data or use live dashboard to confirm basket.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
OUT = ROOT / "frontend" / "src" / "data" / "mag7ValuationHistory.json"

sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env")

from app.services.providers.fmp_client import compute_mag7_basket  # noqa: E402


def main() -> int:
    key = os.getenv("FMP_API_KEY", "").strip()
    if not key:
        print("error: FMP_API_KEY not set (expected in backend/.env)", file=sys.stderr)
        return 1

    try:
        result = compute_mag7_basket(api_key=key, timeout=30)
    except Exception as exc:
        print(f"error: compute_mag7_basket failed: {exc}", file=sys.stderr)
        return 1

    if result.status != "fresh" or result.value is None:
        print(
            f"error: basket not available (status={result.status!r}). {result.note or ''}",
            file=sys.stderr,
        )
        return 1

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    value = round(float(result.value), 4)

    if OUT.exists():
        data = json.loads(OUT.read_text(encoding="utf-8"))
    else:
        data = {
            "label": "Mag 7 cap-weighted forward P/E",
            "metric": "mag7_forward_pe_cap_weighted",
            "basis": "forward",
            "source": "offline_script",
            "script_version": "1",
            "methodology_note": "",
            "last_refreshed": None,
            "history": [],
        }

    hist = [h for h in data.get("history", []) if h.get("date") != today]
    hist.append({"date": today, "value": value})
    hist.sort(key=lambda x: x["date"])

    data["history"] = hist
    data["last_refreshed"] = today
    data["methodology_note"] = (
        "Each point is produced by scripts/refresh_mag7_valuation_history.py using "
        "backend fmp_client.compute_mag7_basket (same MAG7 tickers, cap weights, "
        "coverage rules as the live dashboard). Not recomputed on page load."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} — {today} → {value}x (n={len(hist)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
