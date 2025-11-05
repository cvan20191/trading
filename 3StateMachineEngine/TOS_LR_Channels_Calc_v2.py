
# tos_lr_sd_1y_stddevall.py
# Calc-only Linear Regression SD Channels on a 1-year window (last 252 trading bars).
# Matches your mapping and TOS-style "StDevAll(price, length)" (constant SD over the window):
#   - Midline = InertiaAll(price, length=252) on the last 252 bars (yhat at the end bar)
#   - SD = StDevAll(price, length=252) on the same 252-bar window (ddof selectable)
#   - Band widths mapping (as you specified):
#       1.00 -> UB4 / LB4
#       0.75 -> UB3 / LB3
#       0.50 -> UB2 / LB2
#       0.25 -> UB1 / LB1
#   UB/LB are symmetric around the midline
#
# Usage:
#   - Set symbol, end_dt (bar to evaluate), source ("hl2" or "close"), ddof (0 or 1)
#   - Run. It prints Midline, UB1..UB4 (with your width mapping), LB1..LB4, SD.

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Tuple, Dict, Literal

# -------------------------------
# Data loading
# -------------------------------
def fetch_ohlc(symbol: str, end_date: str | pd.Timestamp, lookback_bdays: int = 400, auto_adjust: bool = False) -> pd.DataFrame:
    """
    Download raw OHLC up to end_date with cushion so we can slice exactly last 252 bars.
    auto_adjust=False to match TOS raw OHLC.
    """
    end = pd.Timestamp(end_date)
    start = (end - pd.tseries.offsets.BDay(lookback_bdays)).date().isoformat()
    df = yf.download(symbol, start=start, end=end.date().isoformat(), auto_adjust=auto_adjust, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} from {start} to {end.date()}.")

    # Handle yfinance single-symbol MultiIndex (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df = df.xs(df.columns.levels[-1][0], axis=1, level=-1)

    # Normalize columns
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    for c in ("open","high","low","close","adj_close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_price(df: pd.DataFrame, source: Literal["hl2","close"] = "hl2") -> pd.Series:
    """
    Build price series used for regression and SD.
    """
    if source == "hl2":
        if {"high","low"}.issubset(df.columns):
            s = (df["high"] + df["low"]) / 2.0
        else:
            raise RuntimeError("HL2 requested but High/Low missing; use source='close'.")
    else:
        if "close" in df.columns:
            s = df["close"]
        else:
            raise RuntimeError("Close missing.")
    return pd.Series(s.to_numpy(dtype=float), index=df.index, name=source)

# -------------------------------
# Math on last N bars ending at end_date
# -------------------------------
def _last_n_window(idx: pd.DatetimeIndex, end_date: pd.Timestamp, n: int = 252) -> Tuple[int,int]:
    """
    Return start/end indices for exactly last n bars ending at end_date (inclusive).
    """
    if end_date not in idx:
        pos = idx.searchsorted(end_date, side="right") - 1
        if pos < 0:
            raise ValueError(f"No bars on/before {end_date.date()}.")
    else:
        pos = int(idx.get_loc(end_date))
    start = pos - (n - 1)
    if start < 0:
        raise ValueError(f"Need {n} bars; only {pos+1} available before {end_date.date()}.")
    return start, pos

def _fit_lr_yhat_last(y: np.ndarray) -> float:
    """
    Fit y ~ a*x + b over x=0..n-1 and return yhat at the last x (n-1).
    This is InertiaAll(price, length=n) midline at the end bar.
    """
    n = y.size
    x = np.arange(n, dtype=float)
    A = np.vstack([x, np.ones(n)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(b + a * (n - 1))

def _stdev_all_price(y: np.ndarray, ddof: int = 1) -> float:
    """
    StDevAll(price, length=n) over the window -> single constant for that window.
    """
    if y.size <= 1:
        return float("nan")
    return float(np.std(y, ddof=ddof))

# -------------------------------
# One-day LR + SD bands on 1-year window (StDevAll of price)
# -------------------------------
def lr_sd_1y_stddevall_on_date(
    px: pd.Series,                              # price used for regression and SD (hl2 or close)
    end_date: str | pd.Timestamp,
    n: int = 252,                               # 1-year trading bars
    ddof: int = 1,                              # 1=sample, 0=population
    widths: Tuple[float,float,float,float] = (1.0, 0.75, 0.50, 0.25),  # your mapping
) -> Dict[str,float]:
    """
    Compute midline and bands at end_date using exactly the last n bars:
      - midline = InertiaAll(price, length=n) yhat at end of window
      - sd = StDevAll(price, length=n) over the same window (constant)
      - Mapping you specified:
          1.00 -> UB4 / LB4
          0.75 -> UB3 / LB3
          0.50 -> UB2 / LB2
          0.25 -> UB1 / LB1
    """
    d = pd.Timestamp(end_date)
    idx = px.index
    s_i, e_i = _last_n_window(idx, d, n=n)

    # Window data
    y = px.iloc[s_i:e_i+1].to_numpy(dtype=float)

    # Midline (LR yhat at end)
    mid = _fit_lr_yhat_last(y)

    # SD (StDevAll of price)
    sd = _stdev_all_price(y, ddof=ddof)

    # Build bands with your mapping (1.0 => UB4/LB4, ..., 0.25 => UB1/LB1)
    # widths tuple is ordered (1.0, 0.75, 0.50, 0.25)
    w4, w3, w2, w1 = widths  # map to UB4, UB3, UB2, UB1
    out = {
        "midline": mid,
        "sd_value": sd,
        # Upper bands
        "UB4": mid + w4 * sd,
        "UB3": mid + w3 * sd,
        "UB2": mid + w2 * sd,
        "UB1": mid + w1 * sd,
        # Lower bands
        "LB1": mid - w1 * sd,
        "LB2": mid - w2 * sd,
        "LB3": mid - w3 * sd,
        "LB4": mid - w4 * sd,
    }
    return out

def print_channels(ch: Dict[str,float], symbol: str, end_date: str | pd.Timestamp,
                   n: int, ddof: int, source: str):
    d = pd.Timestamp(end_date)
    print(f"\n{symbol} LR SD Channels (1Y={n} bars, StDevAll(price), source={source}, ddof={ddof}) on {d}:")
    print("============================================================")
    print(f"Midline (InertiaAll): {ch['midline']:10.4f}")
    print("\nUpper bands (your mapping 1.00→UB4, 0.75→UB3, 0.50→UB2, 0.25→UB1):")
    print(f"  UB4 (+1.00σ):       {ch['UB4']:10.4f}")
    print(f"  UB3 (+0.75σ):       {ch['UB3']:10.4f}")
    print(f"  UB2 (+0.50σ):       {ch['UB2']:10.4f}")
    print(f"  UB1 (+0.25σ):       {ch['UB1']:10.4f}")
    print("\nLower bands:")
    print(f"  LB1 (-0.25σ):       {ch['LB1']:10.4f}")
    print(f"  LB2 (-0.50σ):       {ch['LB2']:10.4f}")
    print(f"  LB3 (-0.75σ):       {ch['LB3']:10.4f}")
    print(f"  LB4 (-1.00σ):       {ch['LB4']:10.4f}")
    print(f"\nSD (StDevAll over last {n} bars): {ch['sd_value']:10.4f}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    symbol   = "SPY"
    end_dt   = "2024-07-17"      # bar to evaluate
    n        = 252               # 1-year window
    source   = "close"           # "hl2" or "close" — set to what your TOS input uses
    ddof     = 1                 # try 1 first; if off, try 0

    # Load raw OHLC up to end_dt
    df = fetch_ohlc(symbol, end_dt, lookback_bdays=400, auto_adjust=False)
    px = make_price(df, source)

    ch = lr_sd_1y_stddevall_on_date(px, end_dt, n=n, ddof=ddof, widths=(1.0, 0.75, 0.50, 0.25))
    print_channels(ch, symbol, end_dt, n, ddof, source)
