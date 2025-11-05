# mobius_rolling_backtest_true252.py
# True rolling Mobius LR+SD channels with no lookahead.
# - Each day recomputes channels from exactly the last n bars (strict window).
# - SD_base(t) = max of rolling StDev(price, StDevPeriod) over the last n bars only
#                implemented as a rolling max over rsd with window hw = n - StDevPeriod + 1.
# - Bands_t = Midline_t ± widthK * (width1 * SD_base_t)
# - Strategy: enter on touch of LBk (yesterday), exit on touch of UBk (yesterday), fill next open.

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Tuple, Literal

# =========================
# CONFIG
# =========================
N: int = 252                             # regression window (midline) - standard 252 trading days
STDEV_PERIOD: int = 12                   # StDev window
WIDTHS: Tuple[float,float,float,float] = (1.0, 2.0, 3.0, 4.0)  # widthOfChannelK
DDOF: int = 0                         # stdev ddof
PRICE_SRC: Literal["hl2","close"] = "hl2"
BAND_K: int = 2                          # legacy: used when entry/exit use same band
ENTRY_K: int = 2                         # buy when touch LB2 (default)
EXIT_K: int = 1                          # sell when touch UB1 (default)
TRADE_ON: Literal["next_open","close"] = "next_open"
SLIPPAGE_BPS: float = 1.0                # slippage in bps
INITIAL_EQUITY: float = 100000.0
SLOPE_LOOKBACK_D: int = 5                # slope lookback in days for diagnostics

# Data source: "yfinance" or "csv"
USE_CSV: bool = False                    # Set to True to use CSV file instead of yfinance
CSV_PATH: str = "SPY_OHLC_Data.csv"      # Path to CSV file when USE_CSV=True

# HighestAll mode: "to_date" for TOS HighestAll behavior (expanding max), "window" for strict rolling
HIGHEST_MODE: Literal["to_date","window"] = "window"   # Strict 252-bar sliding window by default

# Optional: emulate TOS "loaded bars" anchor. When set, data prior to this date is ignored
# This affects HighestAll (expanding max) the same way changing the TOS chart's loaded range does.
CHART_START: Optional[str] = None  # e.g., "2023-01-01"; None means use full history
RUN_MODE: Literal["normal","compare"] = "compare"
COMPARE_DATE: str = "2023-09-21"

# =========================
# Data
# =========================
def fetch_ohlc(symbol: str, start: str, end: Optional[str] = None, auto_adjust: bool = False) -> pd.DataFrame:
    end = end or pd.Timestamp.today().date().isoformat()
    df = yf.download(symbol, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} between {start} and {end}.")
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, axis=1, level=-1)
        else:
            df = df.xs(df.columns.levels[-1][0], axis=1, level=-1)
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
    for c in ("open","high","low","close","adj_close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def load_tos_csv(path: str) -> pd.DataFrame:
    """Load OHLC data from TOS-exported CSV (or any CSV with Date/DateTime, Open, High, Low, Close)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("datetime")
    open_col = cols.get("open"); high_col = cols.get("high")
    low_col  = cols.get("low");  close_col= cols.get("close")
    if not all([date_col, open_col, high_col, low_col, close_col]):
        raise ValueError("CSV must have Date/DateTime/Open/High/Low/Close columns.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    out = pd.DataFrame(index=df.index)
    out["open"]  = pd.to_numeric(df[open_col], errors="coerce")
    out["high"]  = pd.to_numeric(df[high_col], errors="coerce")
    out["low"]   = pd.to_numeric(df[low_col],  errors="coerce")
    out["close"] = pd.to_numeric(df[close_col],errors="coerce")
    out = out.dropna(subset=["open","high","low","close"])
    return out

def make_price(df: pd.DataFrame, source: Literal["hl2","close"] = PRICE_SRC) -> pd.Series:
    if source == "hl2":
        if {"high","low"}.issubset(df.columns):
            s = (df["high"] + df["low"]) / 2.0
        else:
            raise RuntimeError("HL2 requested but High/Low missing.")
    else:
        s = df["close"]
    return pd.Series(s.to_numpy(float), index=df.index, name=source)

# =========================
# Rolling LR midline (Type=line): yhat at each bar using last n bars
# =========================
def rolling_inertia_mid(price: pd.Series, n: int = N) -> pd.Series:
    y = price.to_numpy(float)
    T = y.size
    out = np.full(T, np.nan, dtype=float)
    if n <= 1 or T < n:
        return pd.Series(out, index=price.index, name="Midline")

    k = np.arange(n, dtype=float)
    Sx = float(k.sum()); Sxx = float((k*k).sum())
    den = (n * Sxx - Sx * Sx)
    if den == 0.0:
        return pd.Series(out, index=price.index, name="Midline")

    for i in range(n - 1, T):
        win = y[i - n + 1 : i + 1]
        Sy = float(win.sum()); Sxy = float((k * win).sum())
        a = (n * Sxy - Sx * Sy) / den
        b = (Sy - a * Sx) / n
        out[i] = b + a * (n - 1)
    return pd.Series(out, index=price.index, name="Midline")

# =========================
# Backtest-safe HighestAll on a strict window
# =========================
def highestall_stdev_series(
    price: pd.Series,
    stdev_len: int,
    ddof: int,
    highest_mode: Literal["to_date","window"] = "window",
    highest_window: Optional[int] = None  # length in σ-observation units when mode="window"
) -> pd.Series:
    rsd = price.rolling(stdev_len, min_periods=stdev_len).std(ddof=DDOF)
    if highest_mode == "to_date":
        return rsd.expanding().max().rename("SD_base")
    if highest_mode == "window":
        if highest_window is None or int(highest_window) <= 0:
            raise ValueError("highest_window must be positive for mode='window'.")
        hw = int(highest_window)  # window length in σ-observation units
        return rsd.rolling(hw, min_periods=hw).max().rename("SD_base")
    raise ValueError("highest_mode must be 'to_date' or 'window'.")

# =========================
# Build channels (for a given price series)
#   SD_base = HighestAll(StDev(STDEV_PERIOD)) by chosen mode
#   SD_with = w1 * SD_base
#   UB/LB = Midline ± wk * SD_with
# =========================
def build_mobius_channels_backtest(
    price: pd.Series,
    n: int,
    widthOfChannel1: float,
    widthOfChannel2: float,
    widthOfChannel3: float,
    widthOfChannel4: float,
    stdev_period: int,
    ddof: int,
    highest_mode: Literal["to_date","window"] = "window",
    highest_window: Optional[int] = None,  # in σ units when mode="window"
) -> pd.DataFrame:
    mid = rolling_inertia_mid(price, n=n)
    SD_base = highestall_stdev_series(price, stdev_len=stdev_period, ddof=DDOF,
                                      highest_mode=highest_mode, highest_window=highest_window)
    w1, w2, w3, w4 = widthOfChannel1, widthOfChannel2, widthOfChannel3, widthOfChannel4
    SD_with = (w1 * SD_base).rename("SD_with")

    ch = pd.DataFrame(index=price.index)
    ch["Midline"] = mid
    ch["SD_base"] = SD_base
    ch["SD_with"] = SD_with
    ch["UpperBand1"] = mid + w1 * SD_with
    ch["UpperBand2"] = mid + w2 * SD_with
    ch["UpperBand3"] = mid + w3 * SD_with
    ch["UpperBand4"] = mid + w4 * SD_with
    ch["LowerBand1"] = mid - w1 * SD_with
    ch["LowerBand2"] = mid - w2 * SD_with
    ch["LowerBand3"] = mid - w3 * SD_with
    ch["LowerBand4"] = mid - w4 * SD_with
    return ch

# =========================
# True rolling backtest (strict windows + next-open fills)
# =========================
def backtest_rolling_touch_strategy(
    df: pd.DataFrame,
    n: int = N,
    stdev_period: int = STDEV_PERIOD,
    widths: Tuple[float,float,float,float] = WIDTHS,
    k: int = BAND_K,
    k_entry: int = ENTRY_K,
    k_exit: int = EXIT_K,
    price_src: Literal["hl2","close"] = PRICE_SRC,
    trade_on: Literal["next_open","close"] = TRADE_ON,
    slippage_bps: float = SLIPPAGE_BPS,
    initial_equity: float = INITIAL_EQUITY,
    force_exit_on_last: bool = True,
    highest_mode: Literal["to_date","window"] = "to_date",
) -> dict:
    open_px, close_px, high_px, low_px = df["open"], df["close"], df["high"], df["low"]
    idx = df.index

    # for signal price (HL2 or close)
    px_sig = ((df["high"] + df["low"]) / 2.0) if price_src == "hl2" else df["close"]

    on = False
    shares = 0.0
    equity = initial_equity
    curve = []
    trades = []

    # σ-window length in σ-observation units for strict HighestAll over last n price bars
    # Only used when highest_mode="window"
    rsd_window_len = n - stdev_period + 1 if highest_mode == "window" else None
    if highest_mode == "window" and rsd_window_len <= 0:
        raise ValueError("n must be larger than StDevPeriod to have valid σ window.")

    # Precompute global-to-date channels once when using HighestAll (to_date)
    # This ensures SD_base uses an expanding max over the full series up to each date
    ch_full_todate = None
    if highest_mode == "to_date":
        ch_full_todate = build_mobius_channels_backtest(
            px_sig,
            n=n,
            widthOfChannel1=widths[0], widthOfChannel2=widths[1],
            widthOfChannel3=widths[2], widthOfChannel4=widths[3],
            stdev_period=stdev_period, ddof=DDOF,
            highest_mode="to_date", highest_window=None
        )

    for i, current_date in enumerate(idx):
        # Need at least n bars to compute today's bands (window i-n+1..i inclusive)
        if i < n - 1:
            curve.append(equity)
            continue

        if highest_mode == "to_date":
            # Use global-to-date channels precomputed on full series
            cur_row = ch_full_todate.iloc[i]
        else:
            # Exact window of length n: i-n+1..i
            start_idx = i - n + 1
            window_data = df.iloc[start_idx:i+1]
            window_price = ((window_data["high"] + window_data["low"]) / 2.0) if price_src == "hl2" else window_data["close"]

            # Build channels only with the last n bars (strict)
            ch_today = build_mobius_channels_backtest(
                window_price,
                n=n,
                widthOfChannel1=widths[0], widthOfChannel2=widths[1],
                widthOfChannel3=widths[2], widthOfChannel4=widths[3],
                stdev_period=stdev_period, ddof=DDOF,
                highest_mode=highest_mode, highest_window=rsd_window_len
            )

        # Today's last-row bands
        cur_row = ch_today.iloc[-1]
        cur_lb = cur_row[f"LowerBand{k_entry}"]
        cur_ub = cur_row[f"UpperBand{k_exit}"]
        cur_low = float(low_px.iloc[i])
        cur_high = float(high_px.iloc[i])
        cur_open = float(open_px.iloc[i])
        cur_close = float(close_px.iloc[i])

        # Mark-to-market
        if on:
            equity = shares * cur_close

        if trade_on == "next_open":
            # We can only trade on today's open if yesterday had enough bars to compute its own bands
            # Yesterday index: i-1; yesterday window: (i-1)-n+1..i-1 = i-n..i-1
            if i >= n:
                if highest_mode == "to_date":
                    prev_row = ch_full_todate.iloc[i-1]
                else:
                    prev_start_idx = i - n
                    prev_window = df.iloc[prev_start_idx:i]
                    prev_price = ((prev_window["high"] + prev_window["low"]) / 2.0) if price_src == "hl2" else prev_window["close"]

                    ch_prev = build_mobius_channels_backtest(
                        prev_price,
                        n=n,
                        widthOfChannel1=widths[0], widthOfChannel2=widths[1],
                        widthOfChannel3=widths[2], widthOfChannel4=widths[3],
                        stdev_period=stdev_period, ddof=DDOF,
                        highest_mode=highest_mode, highest_window=rsd_window_len
                    )

                    prev_row = ch_prev.iloc[-1]
                prev_lb = prev_row[f"LowerBand{k_entry}"]
                prev_ub = prev_row[f"UpperBand{k_exit}"]
                prev_low = float(low_px.iloc[i-1])
                prev_high = float(high_px.iloc[i-1])

                prev_touch_lb = prev_low <= prev_lb
                prev_touch_ub = prev_high >= prev_ub

                # Entry: if yesterday touched LBk_entry, buy at today's open
                if (not on) and prev_touch_lb:
                    fill = cur_open * (1.0 + slippage_bps / 10000.0)
                    if fill > 0:
                        shares = equity / fill
                        trades.append({
                            "entry_date": current_date,
                            "entry_price": fill,
                            "entry_shares": shares,
                            "entry_reason": f"touch_lb{k_entry}",
                            "band_k_entry": k_entry,
                            "band_k_exit": k_exit,
                            "prev_lb": prev_lb,
                            "prev_ub": prev_ub
                        })
                        on = True

                # Exit: if yesterday touched UBk_exit, sell at today's open
                elif on and prev_touch_ub:
                    fill = cur_open * (1.0 - slippage_bps / 10000.0)
                    pnl = shares * (fill - trades[-1]["entry_price"])
                    trades[-1]["exit_date"] = current_date
                    trades[-1]["exit_price"] = fill
                    trades[-1]["pnl"] = pnl
                    trades[-1]["ret"] = (fill / trades[-1]["entry_price"] - 1.0)
                    trades[-1]["exit_reason"] = f"touch_ub{k_exit}"
                    equity = shares * fill
                    shares = 0.0
                    on = False

        else:
            # trade_on == "close": entry/exit on same day's close if today touched bands
            touch_lb_today = cur_low <= cur_lb
            touch_ub_today = cur_high >= cur_ub

            if (not on) and touch_lb_today:
                fill = cur_close * (1.0 + slippage_bps / 10000.0)
                if fill > 0:
                    shares = equity / fill
                    trades.append({
                        "entry_date": current_date,
                        "entry_price": fill,
                        "entry_shares": shares,
                        "entry_reason": f"touch_lb{k_entry}",
                        "band_k_entry": k_entry,
                        "band_k_exit": k_exit
                    })
                    on = True
            elif on and touch_ub_today:
                fill = cur_close * (1.0 - slippage_bps / 10000.0)
                pnl = shares * (fill - trades[-1]["entry_price"])
                trades[-1]["exit_date"] = current_date
                trades[-1]["exit_price"] = fill
                trades[-1]["pnl"] = pnl
                trades[-1]["ret"] = (fill / trades[-1]["entry_price"] - 1.0)
                trades[-1]["exit_reason"] = f"touch_ub{k}"
                equity = shares * fill
                shares = 0.0
                on = False

        curve.append(equity)

    # Force exit at last bar (for reporting)
    if force_exit_on_last and on and len(idx) > 0 and len(trades) > 0:
        last_date = idx[-1]
        last_close = float(close_px.iloc[-1])
        fill_price = last_close * (1.0 - slippage_bps / 10000.0)
        pnl = shares * (fill_price - trades[-1]["entry_price"])
        trades[-1]["exit_date"] = last_date
        trades[-1]["exit_price"] = fill_price
        trades[-1]["pnl"] = pnl
        trades[-1]["ret"] = (fill_price / trades[-1]["entry_price"] - 1.0)
        trades[-1]["exit_reason"] = "force_exit"
        equity = shares * fill_price
        curve[-1] = equity

    eq_curve = pd.Series(curve, index=idx, name="equity")
    trades_df = pd.DataFrame(trades)
    return {"equity_curve": eq_curve, "trades": trades_df}

# =========================
# Diagnostics: per-day STD lines (strict 252-bar window)
# =========================
def compute_std_lines_strict(df: pd.DataFrame, n: int = N, stdev_len: int = STDEV_PERIOD,
                             price_src: Literal["hl2","close"] = PRICE_SRC,
                             ddof: int = DDOF) -> pd.DataFrame:
    px = ((df["high"] + df["low"]) / 2.0) if price_src == "hl2" else df["close"]
    mid = rolling_inertia_mid(px, n=n)
    rsd = px.rolling(stdev_len, min_periods=stdev_len).std(ddof=ddof)
    # max over the last (n - stdev_len + 1) rsd observations → strict-in-window Highest
    win = max(1, n - stdev_len + 1)
    sd_base = rsd.rolling(win, min_periods=win).max()
    sd_with = sd_base  # width1=1 by definition
    out = pd.DataFrame(index=df.index)
    out["hl2"] = px
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["mid"] = mid
    out["stdev20"] = rsd
    out["sd_with_width1"] = sd_with
    # Bands 1..4
    for k in (1,2,3,4):
        out[f"UB{k}"] = mid + k * sd_with
        out[f"LB{k}"] = mid - k * sd_with
    # slope
    out["mid_slope"] = out["mid"] - out["mid"].shift(SLOPE_LOOKBACK_D)
    return out

# =========================
# Touch/revert analysis (strict 252)
# =========================
def analyze_band_touches(std_df: pd.DataFrame) -> dict:
    d = std_df.dropna(subset=["high","low","close","UB1","UB2","UB3","LB1","LB2","LB3"]).copy()
    # Touch masks
    touch_ub1 = (d["high"] >= d["UB1"]) | (d["close"] >= d["UB1"])  # include close crosses
    touch_ub2 = (d["high"] >= d["UB2"]) | (d["close"] >= d["UB2"]) 
    touch_ub3 = (d["high"] >= d["UB3"]) | (d["close"] >= d["UB3"]) 
    touch_lb1 = (d["low"]  <= d["LB1"]) | (d["close"] <= d["LB1"]) 
    touch_lb2 = (d["low"]  <= d["LB2"]) | (d["close"] <= d["LB2"]) 
    touch_lb3 = (d["low"]  <= d["LB3"]) | (d["close"] <= d["LB3"]) 

    idx = d.index.to_list()
    # Helper: count reverts to UB1 after a UBk touch (next day or later)
    def count_reverts(touch_mask: pd.Series) -> int:
        count = 0
        touch_idx = [i for i, m in enumerate(touch_mask.to_list()) if m]
        ub1_list = d["UB1"].to_list()
        close_list = d["close"].to_list()
        for i in touch_idx:
            # search forward for first day where close <= UB1
            reverted = False
            for j in range(i+1, len(d)):
                if close_list[j] <= ub1_list[j]:
                    count += 1
                    reverted = True
                    break
            # if never reverted, skip
        return count

    results = {
        "touch_ub3": int(touch_ub3.sum()),
        "ub3_revert_to_ub1": count_reverts(touch_ub3),
        "touch_ub2": int(touch_ub2.sum()),
        "ub2_revert_to_ub1": count_reverts(touch_ub2),
        "touch_lb1": int(touch_lb1.sum()),
        "touch_lb2": int(touch_lb2.sum()),
        "touch_lb3": int(touch_lb3.sum()),
    }
    return results

# =========================
# Touch→Revert trade logging (strict 252 window, next-open fills)
# =========================
def build_touch_revert_trades(std_df: pd.DataFrame,
                              side: Literal["long","short"],
                              entry_band_k: int,
                              exit_to: Literal["UB1","UB2"] = "UB1",
                              slippage_bps: float = SLIPPAGE_BPS) -> pd.DataFrame:
    d = std_df.dropna(subset=["open","high","low","close","UB1","LB1"]).copy()
    idx = d.index.to_list()
    trades = []
    bps = slippage_bps / 10000.0

    for i in range(len(d)):
        if i >= len(d) - 1:
            break  # need next-day open to fill

        if side == "short":
            entry_band = d[f"UB{entry_band_k}"] if entry_band_k >= 1 else d["UB1"]
            entry_signal = (d["high"].iloc[i] >= float(entry_band.iloc[i]))
        else:
            entry_band = d[f"LB{entry_band_k}"] if entry_band_k >= 1 else d["LB1"]
            entry_signal = (d["low"].iloc[i] <= float(entry_band.iloc[i]))

        if not entry_signal:
            continue

        # Entry at next day's open
        j = i + 1
        entry_date = idx[j]
        op = float(d["open"].iloc[j])
        if side == "short":
            entry_price = op * (1.0 - bps)  # sell to open
        else:
            entry_price = op * (1.0 + bps)  # buy to open

        # Search for exit condition from day i+1 onward
        exit_date = None
        exit_price = None
        exit_reason = None
        for k in range(j, len(d)):
            if side == "short":
                # cover when Close <= exit_to (UB1 or UB2)
                ub_col = "UB1" if exit_to == "UB1" else "UB2"
                if float(d["close"].iloc[k]) <= float(d[ub_col].iloc[k]):
                    # exit at next-day open if available, else current close
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        op2 = float(d["open"].iloc[k+1])
                        exit_price = op2 * (1.0 + bps)  # buy to close
                    else:
                        exit_date = idx[k]
                        cl2 = float(d["close"].iloc[k])
                        exit_price = cl2 * (1.0 + bps)
                    exit_reason = f"revert_to_{ub_col}"
                    break
            else:
                # long: sell when Close >= exit_to (UB1 or UB2)
                ub_col = "UB1" if exit_to == "UB1" else "UB2"
                if float(d["close"].iloc[k]) >= float(d[ub_col].iloc[k]):
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        op2 = float(d["open"].iloc[k+1])
                        exit_price = op2 * (1.0 - bps)  # sell to close
                    else:
                        exit_date = idx[k]
                        cl2 = float(d["close"].iloc[k])
                        exit_price = cl2 * (1.0 - bps)
                    exit_reason = f"reach_{ub_col}"
                    break

        # If never found, force exit on last bar open/close accordingly
        if exit_date is None:
            last_i = len(d) - 1
            exit_date = idx[last_i]
            if side == "short":
                exit_price = float(d["close"].iloc[last_i]) * (1.0 + bps)
                exit_reason = "force_exit"
            else:
                exit_price = float(d["close"].iloc[last_i]) * (1.0 - bps)
                exit_reason = "force_exit"

        # capture mid and slope at entry (j)
        mid_entry = float(d["mid"].iloc[j]) if pd.notna(d["mid"].iloc[j]) else float("nan")
        slope_entry = float(d["mid_slope"].iloc[j]) if pd.notna(d["mid_slope"].iloc[j]) else float("nan")

        ret = (entry_price / exit_price - 1.0) if side == "short" else (exit_price / entry_price - 1.0)
        pnl = ret  # 1 unit
        trades.append({
            "side": side,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "ret": ret,
            "pnl": pnl,
            "entry_band_k": entry_band_k,
            "exit_to": exit_to,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
            "mid_entry": mid_entry,
            "mid_slope_entry": slope_entry,
        })

    return pd.DataFrame(trades)

def compare_modes_on_date(df: pd.DataFrame,
                          date: str,
                          symbol: str = "SPY",
                          n: int = 252,
                          stdev_period: int = 20,
                          widths: Tuple[float,float,float,float] = (1.0, 2.0, 3.0, 4.0),
                          price_src: Literal["hl2","close"] = "hl2"):
    """
    Print channels on date D and D-1 for:
      - strict rolling (window n)
      - OnDemand-style to_date (global HighestAll up to D)
    """
    d = pd.Timestamp(date)
    if d not in df.index:
        pos = df.index.searchsorted(d, "right") - 1
        if pos < 0:
            print(f"No data on/before {date}")
            return
        d = df.index[pos]
    d_prev_i = df.index.get_loc(d) - 1
    if d_prev_i < 0:
        print("No previous day available.")
        return
    d_prev = df.index[d_prev_i]

    # Build price series
    price_full = ((df["high"] + df["low"]) / 2.0) if price_src == "hl2" else df["close"]

    # 1) Strict-rolling on D (window i-n+1..i)
    i = df.index.get_loc(d)
    if i < n - 1:
        print("Not enough bars for strict-rolling window.")
        return
    start_i = i - n + 1
    roll_price_D = price_full.iloc[start_i:i+1]
    rsd_window_len = n - stdev_period + 1
    ch_roll_D = build_mobius_channels_backtest(
        roll_price_D, n=n,
        widthOfChannel1=widths[0], widthOfChannel2=widths[1], widthOfChannel3=widths[2], widthOfChannel4=widths[3],
        stdev_period=stdev_period, ddof=DDOF,
        highest_mode="window", highest_window=rsd_window_len
    ).iloc[-1]

    # 1b) Strict-rolling on D-1
    i_prev = i - 1
    start_prev = i_prev - n + 1
    roll_price_prev = price_full.iloc[start_prev:i_prev+1]
    ch_roll_prev = build_mobius_channels_backtest(
        roll_price_prev, n=n,
        widthOfChannel1=widths[0], widthOfChannel2=widths[1], widthOfChannel3=widths[2], widthOfChannel4=widths[3],
        stdev_period=stdev_period, ddof=DDOF,
        highest_mode="window", highest_window=rsd_window_len
    ).iloc[-1]

    # 2) OnDemand-style (global to_date) on D and D-1
    # Build once across full series (no lookahead; expanding max)
    ch_todate_full = build_mobius_channels_backtest(
        price_full, n=n,
        widthOfChannel1=widths[0], widthOfChannel2=widths[1], widthOfChannel3=widths[2], widthOfChannel4=widths[3],
        stdev_period=stdev_period, ddof=DDOF,
        highest_mode="to_date", highest_window=None
    )
    ch_todate_D = ch_todate_full.loc[d]
    ch_todate_prev = ch_todate_full.loc[d_prev]

    px_D = float(price_full.loc[d])
    px_prev = float(price_full.loc[d_prev])

    def _pfx(tag, row, px):
        return (f"{tag}: px={px:.2f} | Mid={row['Midline']:.2f} | "
                f"SDw={row.get('SD_with', row.get('SD_with_width', np.nan)):.2f} | "
                f"UB2={row['UpperBand2']:.2f} | LB2={row['LowerBand2']:.2f}")

    print(f"\n=== {symbol} {price_src.upper()} @ {d.date()} (compare modes) ===")
    print(_pfx("Strict-rolling D    ", ch_roll_D, px_D))
    print(_pfx("OnDemand to_date D  ", ch_todate_D, px_D))
    print(_pfx("Strict-rolling D-1  ", ch_roll_prev, px_prev))
    print(_pfx("OnDemand to_date D-1", ch_todate_prev, px_prev))

    # Show whether D-1 touched LB2 in each mode (this is your entry trigger for D)
    low_prev = float(df["low"].loc[d_prev])
    print(f"\nPrev day low={low_prev:.2f}")
    roll_touch = low_prev <= float(ch_roll_prev['LowerBand2'])
    od_touch   = low_prev <= float(ch_todate_prev['LowerBand2'])
    print(f"Touch LB2 on D-1? strict-rolling={roll_touch} | to_date={od_touch}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    SYMBOL = "SPY"
    START  = CHART_START or "2015-09-01"
    END    = "2020-10-01"

    # Load data from CSV or yfinance
    if USE_CSV:
        df = load_tos_csv(CSV_PATH)
    else:
        df = fetch_ohlc(SYMBOL, START, END, auto_adjust=False)
    
    # Quick compare-only mode: compute channels on COMPARE_DATE with to_date HighestAll using 2023-only range
    if RUN_MODE == "compare":
        df23 = fetch_ohlc(SYMBOL, "2022-09-01", "2023-10-01", auto_adjust=False)
        px23 = ((df23["high"] + df23["low"]) / 2.0)
        ch_full = build_mobius_channels_backtest(
            px23,
            n=N,
            widthOfChannel1=WIDTHS[0], widthOfChannel2=WIDTHS[1], widthOfChannel3=WIDTHS[2], widthOfChannel4=WIDTHS[3],
            stdev_period=STDEV_PERIOD, ddof=DDOF,
            highest_mode="to_date", highest_window=None
        )
        d = pd.Timestamp(COMPARE_DATE)
        if d not in ch_full.index:
            d = ch_full.index[ch_full.index.searchsorted(d, "right")-1]
        row = ch_full.loc[d]
        mid = float(row["Midline"])
        ub1 = float(row["UpperBand1"]); ub2 = float(row["UpperBand2"]); ub3 = float(row["UpperBand3"]); ub4 = float(row["UpperBand4"])
        lb1 = float(row["LowerBand1"]); lb2 = float(row["LowerBand2"]); lb3 = float(row["LowerBand3"]); lb4 = float(row["LowerBand4"])
        sd_with = ub1 - mid
        cl = float(px23.loc[d])
        print(f"Date: {d.date()}")
        print(f"HL2: {cl:.2f}")
        print(f"Midline: {mid:.2f}")
        print(f"SD (with width): {sd_with:.2f}")
        print(f"UB1: {ub1:.2f} | UB2: {ub2:.2f} | UB3: {ub3:.2f} | UB4: {ub4:.2f}")
        print(f"LB1: {lb1:.2f} | LB2: {lb2:.2f} | LB3: {lb3:.2f} | LB4: {lb4:.2f}")
        raise SystemExit(0)

    # Touch/revert analysis
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna()
    
    touch_stats = analyze_band_touches(std_df)
    print("\nTOUCH/REVERT STATS:")
    for k, v in touch_stats.items():
        print(f"  {k}: {v}")

    # Touch→revert trades
    short_ub3 = build_touch_revert_trades(std_df, side="short", entry_band_k=3, exit_to="UB1")
    short_ub2 = build_touch_revert_trades(std_df, side="short", entry_band_k=2)
    long_lb1  = build_touch_revert_trades(std_df, side="long",  entry_band_k=1)
    long_lb2  = build_touch_revert_trades(std_df, side="long",  entry_band_k=2)
    long_lb3  = build_touch_revert_trades(std_df, side="long",  entry_band_k=3)

    def _summ(name, df):
        if df is None or df.empty:
            print(f"{name}: 0 trades")
            return
        print(f"{name}: trades={len(df)} | hit={int((df['ret']>0).sum())}/{len(df)} | avg_ret={df['ret'].mean():.4f} | avg_days={df['duration_days'].mean():.1f}")

    _summ("SHORT UB3→UB1", short_ub3)
    _summ("SHORT UB2→UB1", short_ub2)
    _summ("LONG  LB1→UB1", long_lb1)
    _summ("LONG  LB2→UB1", long_lb2)
    _summ("LONG  LB3→UB1", long_lb3)
