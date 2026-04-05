"""
Central mapping of all series IDs, tickers, and freshness expectations.

Keeping everything here prevents scattered magic strings across provider files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FRED series identifiers
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    # Monetary policy
    "fed_funds_rate":     "DFEDTARU",        # Fed Funds target upper bound, daily
    "balance_sheet":      "WALCL",           # Total assets (Fed balance sheet), weekly
    # Labour
    "unemployment_rate":  "UNRATE",          # Unemployment rate, monthly
    "initial_claims":     "ICSA",            # Initial jobless claims, weekly
    "nonfarm_payrolls":   "PAYEMS",          # Nonfarm payrolls, monthly
    # Inflation (verified FRED series IDs)
    "core_cpi":           "CPILFESL",        # Core CPI index, SA, monthly
    "shelter_cpi":        "CUSR0000SAH1",    # CPI Shelter, monthly
    "services_ex_energy": "CUSR0000SASLE",   # CPI Services less energy (verified working)
    # Stress
    "yield_curve":        "T10Y2Y",          # 10Y-2Y Treasury spread, daily
    "npl_ratio":          "DRCRELEXFACBS",   # Delinquency rate on real estate loans (NPL proxy), quarterly
    "m2":                 "WM2NS",           # M2 money stock, weekly
}

# ---------------------------------------------------------------------------
# Yahoo Finance tickers
# ---------------------------------------------------------------------------
YAHOO_TICKERS: dict[str, str] = {
    "wti_oil":            "CL=F",       # WTI Crude front-month futures
    "dxy":                "DX-Y.NYB",   # US Dollar Index
    "nasdaq_etf":         "QQQ",        # Nasdaq proxy for trailing P/E context
    "sp500_etf":          "SPY",        # S&P 500 for market-cap proxy
}

# ---------------------------------------------------------------------------
# Freshness windows — how old a series can be before being flagged stale
# (in calendar days)
# ---------------------------------------------------------------------------
FRESHNESS_RULES: dict[str, int] = {
    "fed_funds_rate":     5,    # daily series, allow weekend gap
    "balance_sheet":      14,   # weekly (H.4.1 releases Thursdays, allow 2 weeks)
    "unemployment_rate":  65,   # monthly (BLS releases ~4 weeks after month-end)
    "initial_claims":     14,   # weekly (Thursday release)
    "nonfarm_payrolls":   65,   # monthly (BLS first Friday of following month)
    "core_cpi":           65,   # monthly (BLS releases ~mid-month for prior month)
    "shelter_cpi":        65,
    "services_ex_energy": 65,
    "yield_curve":        5,    # daily
    "npl_ratio":          120,  # quarterly (Fed releases ~45 days after quarter-end)
    "m2":                 20,   # weekly (Fed H.6 release, allow extra days)
    "wti_oil":            5,    # daily market
    "dxy":                5,
    "nasdaq_etf":         5,
    "sp500_etf":          5,
    "forward_pe":         5,   # Mag 7 basket (FMP) or QQQ proxy (Yahoo) — treat as daily quote
    "pmi_manufacturing":  65,   # monthly (ISM releases first business day of month)
    "pmi_services":       65,
}

# ---------------------------------------------------------------------------
# Provider name labels (for source metadata)
# ---------------------------------------------------------------------------
PROVIDER_FRED = "FRED"
PROVIDER_YAHOO = "Yahoo Finance"
PROVIDER_FMP = "FMP"
PROVIDER_STUB = "stub"
