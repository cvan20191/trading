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
STDEV_PERIOD: int = 20                   # StDev window
WIDTHS: Tuple[float,float,float,float] = (1.0, 2.0, 3.0, 4.0)  # widthOfChannelK
DDOF: int = 0                         # stdev ddof
PRICE_SRC: Literal["hl2","close"] = "hl2"
BAND_K: int = 2                          # legacy: used when entry/exit use same band
ENTRY_K: int = 2                         # buy when touch LB2 (default)
EXIT_K: int = 1                          # sell when touch UB1 (default)
TRADE_ON: Literal["next_open","close"] = "next_open"
SLIPPAGE_BPS: float = 1.0                # slippage in bps
INITIAL_EQUITY: float = 100000.0

# Data source: "yfinance" or "csv"
USE_CSV: bool = False                    # Set to True to use CSV file instead of yfinance
CSV_PATH: str = "SPY_OHLC_Data.csv"      # Path to CSV file when USE_CSV=True

# HighestAll mode: "to_date" for TOS HighestAll behavior (expanding max), "window" for strict rolling
HIGHEST_MODE: Literal["to_date","window"] = "window"   # Strict 252-bar sliding window by default

# Optional: emulate TOS "loaded bars" anchor. When set, data prior to this date is ignored
# This affects HighestAll (expanding max) the same way changing the TOS chart's loaded range does.
CHART_START: Optional[str] = None  # e.g., "2023-01-01"; None means use full history

# Output controls (keep defaults minimal: only one combined file if needed)
WRITE_STD_LINES: bool = False
WRITE_YEARLY_EXPORTS: bool = False
WRITE_PER_STRATEGY_TRADES: bool = False
WRITE_COMBINED_TRADES: bool = False
WRITE_BACKTEST_TRADES: bool = False

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
# Summary + pretty print
# =========================
def summarize(eq_curve: pd.Series, trades: pd.DataFrame) -> dict:
    eq = eq_curve.dropna()
    if eq.empty:
        return {"error":"empty equity curve"}
    rets = eq.pct_change().dropna()
    n = len(rets)
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252.0 / max(n, 1)) - 1.0 if n > 0 else np.nan
    vol = rets.std(ddof=DDOF)
    sharpe = (rets.mean() / vol) * np.sqrt(252.0) if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1.0).min()
    closed = trades.loc[trades["exit_price"].notna()] if ("exit_price" in trades.columns) else trades.iloc[0:0]
    win_rate = (closed["ret"] > 0).mean() if not closed.empty else np.nan
    return {
        "start_equity": float(eq.iloc[0]),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(total),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "MaxDD": float(mdd),
        "trades": int(len(closed)),
        "win_rate": float(win_rate) if pd.notna(win_rate) else np.nan
    }

def print_all_trades(trades: pd.DataFrame):
    if trades is None or trades.empty:
        print("\nNo trades to display.")
        return
    t = trades.copy()
    # Compute duration days where exit_date exists
    if "entry_date" in t and "exit_date" in t:
        try:
            t["duration_days"] = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"]))\
                .dt.days
        except Exception:
            pass
    # Order columns
    preferred_cols = [
        "entry_date","exit_date","entry_price","exit_price","pnl","ret",
        "duration_days","entry_reason","exit_reason",
        "band_k_entry","band_k_exit","prev_lb","prev_ub"
    ]
    cols = [c for c in preferred_cols if c in t.columns] + [c for c in t.columns if c not in preferred_cols]
    t = t[cols]
    print("\n=== ALL TRADES ===")
    try:
        print(t.to_string(index=False))
    except Exception:
        print(t.head(50).to_string(index=False))

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
                # long: sell when Close >= UB1
                if float(d["close"].iloc[k]) >= float(d["UB1"].iloc[k]):
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        op2 = float(d["open"].iloc[k+1])
                        exit_price = op2 * (1.0 - bps)  # sell to close
                    else:
                        exit_date = idx[k]
                        cl2 = float(d["close"].iloc[k])
                        exit_price = cl2 * (1.0 - bps)
                    exit_reason = "reach_ub1"
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
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
        })

    return pd.DataFrame(trades)

def _fmt_money(x, nd=2):
    if pd.isna(x): return "—"
    return f"${x:,.{nd}f}"

def _fmt_pct(x, nd=2):
    if pd.isna(x): return "—"
    return f"{100*x:.{nd}f}%"

def _fmt_num(x, nd=2):
    if pd.isna(x): return "—"
    return f"{x:,.{nd}f}"
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

# Add this debug function to your code
def debug_sd_calculation(df, target_date, n=252, stdev_period=20):
    """Debug the exact SD calculation for a specific date"""
    target_idx = df.index.get_loc(pd.Timestamp(target_date))
    
    # Get the rolling window for that date
    start_idx = target_idx - n + 1
    window_data = df.iloc[start_idx:target_idx+1]
    window_price = (window_data["high"] + window_data["low"]) / 2.0
    
    print(f"\n=== DEBUG SD CALCULATION for {target_date} ===")
    print(f"Window: {window_data.index[0]} to {window_data.index[-1]}")
    print(f"Window size: {len(window_price)} bars")
    
    # Calculate rolling StDev
    rolling_stdev = window_price.rolling(stdev_period).std(ddof=DDOF)  # DDOF=0 for ThinkScript parity
    
    print(f"StDev period: {stdev_period}")
    print(f"Rolling StDev range: {rolling_stdev.min():.4f} to {rolling_stdev.max():.4f}")
    print(f"Current StDev: {rolling_stdev.iloc[-1]:.4f}")
    print(f"HighestAll StDev: {rolling_stdev.max():.4f}")
    print(f"SD with width: {rolling_stdev.max() * 1.0:.4f}")
    
    # Show the actual values around the highest StDev period
    max_stdev_idx = rolling_stdev.idxmax()
    max_stdev_value = rolling_stdev.max()
    max_stdev_pos = rolling_stdev.index.get_loc(max_stdev_idx)
    
    print(f"Highest StDev occurred on: {max_stdev_idx}")
    print(f"Highest StDev value: {max_stdev_value:.4f}")
    
    # Show price data around that period
    if max_stdev_pos >= 10 and max_stdev_pos < len(window_price) - 10:
        around_data = window_price.iloc[max_stdev_pos-5:max_stdev_pos+6]
        print(f"Prices around highest StDev period:")
        for date, price in around_data.items():
            print(f"  {date}: {price:.2f}")
    
    return rolling_stdev.max()

def print_summary(stats: dict):
    print("\n===== Performance Summary =====")
    print(f"Start equity : {_fmt_money(stats.get('start_equity', np.nan))}")
    print(f"Final equity : {_fmt_money(stats.get('final_equity', np.nan))}")
    print(f"Total return : {_fmt_pct(stats.get('total_return', np.nan))}")
    print(f"CAGR         : {_fmt_pct(stats.get('CAGR', np.nan))}")
    print(f"Sharpe       : {_fmt_num(stats.get('Sharpe', np.nan), 2)}")
    print(f"Max drawdown : {_fmt_pct(stats.get('MaxDD', np.nan))}")
    print(f"Closed trades: {stats.get('trades', 0)}")
    print(f"Hit ratio    : {_fmt_pct(stats.get('win_rate', np.nan))}")


import pandas as pd
import numpy as np

def stdev20_series(price: pd.Series, ddof: int = 0) -> pd.Series:
    return price.rolling(20, min_periods=20).std(ddof=ddof).dropna()

def audit_window(price: pd.Series, end_date: str, n: int = 252, ddof: int = 0, label: str = ""):
    d = pd.Timestamp(end_date)
    if d not in price.index:
        d = price.index[price.index.searchsorted(d, "right") - 1]
    i = price.index.get_loc(d)
    if i < n - 1:
        raise ValueError("Not enough bars to get a 252-bar window ending at " + str(d.date()))
    win = price.iloc[i - n + 1 : i + 1]  # exactly 252 bars
    rsd = stdev20_series(win, ddof=ddof)  # length = 252-20+1 = 233
    mx = float(rsd.max())
    mx_dt = rsd.idxmax()
    cur = float(rsd.iloc[-1])

    # Pull the exact 20 HL2 values that created the max σ
    j = win.index.get_loc(mx_dt)
    win20 = win.iloc[j - 19 : j + 1]

    print(f"\n=== AUDIT ({label}) window end={d.date()} ===")
    print(f"Bars: {win.index[0].date()} → {win.index[-1].date()} (n={len(win)})")
    print(f"StDev(20, ddof={ddof}) current={cur:.4f} max={mx:.4f} at {mx_dt.date()}")
    print("HL2 values for the max-σ 20-bar window:")
    print(win20.to_string(float_format="{:.2f}".format))


    # quick range sanity: σ cannot exceed range; if it’s big, the HL2 spread is big
    r = float(win20.max() - win20.min())
    print(f"Range in that 20-bar window: {r:.4f}")
    
# Add this to your main function after the data is loaded:
if __name__ == "__main__":
    SYMBOL = "SPY"
    START  = CHART_START or "2015-09-01"
    END    = "2020-10-01"  # limit backtest to Sep 2015 - Sep 2020
    print(f"=== CONFIGURATION ===")
    print(f"DDOF: {DDOF}")
    print(f"N (window): {N}")
    print(f"STDEV_PERIOD: {STDEV_PERIOD}")
    print(f"HighestAll Mode: {HIGHEST_MODE} {'(TOS HighestAll - expanding max)' if HIGHEST_MODE == 'to_date' else '(Strict rolling window)'}")
    print(f"PRICE_SRC: {PRICE_SRC}")
    print(f"BAND_K: {BAND_K}")
    print()

    # Load data from CSV or yfinance
    if USE_CSV:
        print(f"Loading data from CSV: {CSV_PATH}")
        df = load_tos_csv(CSV_PATH)
    else:
        print(f"Downloading data from yfinance: {SYMBOL}")
        df = fetch_ohlc(SYMBOL, START, END, auto_adjust=False)
    
    # Create price series for audit (same as used in backtest)
    price_series = make_price(df, source=PRICE_SRC)
    
    print(f"Data range: {df.index[0].date()} → {df.index[-1].date()} | bars={len(df):,}")

    # AUDIT SPECIFIC DATES
    print("\n" + "="*60)
    print("AUDIT WINDOW ANALYSIS")
    print("="*60)
    
    # Audit specific dates you're interested in (with full yfinance data)
    audit_dates = ["2020-03-23", "2022-10-13", "2023-09-21"]  # Key volatile periods
    
    for audit_date in audit_dates:
        try:
            audit_window(
                price=price_series,
                end_date=audit_date,
                n=N,  # 252
                ddof=DDOF,  # 0
                label=f"TOS CSV - {audit_date}"
            )
        except Exception as e:
            print(f"Could not audit {audit_date}: {e}")
    
    # Then continue with your other analyses...
    print("\n" + "="*60)
    print("COMPARISON MODE ANALYSIS") 
    print("="*60)
    
    # Test dates with full historical data
    test_dates = ["2023-09-21", "2023-12-29"]
    for test_date in test_dates:
        try:
            compare_modes_on_date(
                df=df,
                date=test_date,
                symbol=SYMBOL if not USE_CSV else "CSV",
                n=N,
                stdev_period=STDEV_PERIOD,
                widths=WIDTHS,
                price_src=PRICE_SRC
            )
        except Exception as e:
            print(f"Could not compare modes for {test_date}: {e}")

    # Then continue with your backtest (only if we have enough bars)
    if len(df) >= N:
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        try:
            res = backtest_rolling_touch_strategy(
                df,
                n=N,
                stdev_period=STDEV_PERIOD,
                widths=WIDTHS,
                k=BAND_K,
                k_entry=ENTRY_K,
                k_exit=EXIT_K,
                price_src=PRICE_SRC,
                trade_on=TRADE_ON,
                slippage_bps=SLIPPAGE_BPS,
                initial_equity=INITIAL_EQUITY,
                force_exit_on_last=True,
                highest_mode=HIGHEST_MODE
            )
            
            stats = summarize(res["equity_curve"], res["trades"])
            print_summary(stats)
            
            # Show first few trades
            if not res["trades"].empty:
                print("\n=== First 5 Trades ===")
                print(res["trades"].head(5).to_string())
                # Print and (optionally) export all trades
                print_all_trades(res["trades"])
                if WRITE_BACKTEST_TRADES:
                    trades_csv = f"trades_{df.index[0].date()}_{df.index[-1].date()}.csv"
                    res["trades"].to_csv(trades_csv, index=False)
                    print(f"\nWrote trades CSV: {trades_csv}")
        except Exception as e:
            print(f"Backtest failed: {e}")
    else:
        print(f"\nSkipping backtest - need at least {N} bars, only have {len(df)}")

    # Export per-day STD lines for TOS comparison (strict 252-bar window)
    print("\n" + "="*60)
    print("EXPORT: STRICT 252-BAR STD LINES")
    print("="*60)
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna()  # keep only fully-defined rows
    if WRITE_STD_LINES:
        csv_path = f"std_lines_{df.index[0].date()}_{df.index[-1].date()}.csv"
        std_df.to_csv(csv_path, float_format="%.6f")
        print(f"Wrote per-day STD lines to: {csv_path}")
    # Touch/revert analysis
    try:
        touch_stats = analyze_band_touches(std_df)
        print("\nTOUCH/REVERT STATS (strict 252):")
        for k, v in touch_stats.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Touch/revert analysis failed: {e}")

    # Build short and long touch→revert trade logs
    print("\n" + "="*60)
    print("TOUCH→REVERT TRADE LOGS (strict 252, next-open fills)")
    print("="*60)
    short_ub3 = build_touch_revert_trades(std_df, side="short", entry_band_k=3, exit_to="UB1")
    short_ub2 = build_touch_revert_trades(std_df, side="short", entry_band_k=2)
    long_lb1  = build_touch_revert_trades(std_df, side="long",  entry_band_k=1)
    long_lb2  = build_touch_revert_trades(std_df, side="long",  entry_band_k=2)
    long_lb3  = build_touch_revert_trades(std_df, side="long",  entry_band_k=3)

    # Also compute SHORT UB3→UB2 as requested
    short_ub3_to_ub2 = build_touch_revert_trades(std_df, side="short", entry_band_k=3, exit_to="UB2")

    def _summ(name, df):
        if df is None or df.empty:
            print(f"{name}: 0 trades")
            return
        print(f"{name}: trades={len(df)} | hit={int((df['ret']>0).sum())}/{len(df)} | avg_ret={df['ret'].mean():.4f} | avg_days={df['duration_days'].mean():.1f}")

    _summ("SHORT UB3→UB1", short_ub3)
    _summ("SHORT UB3→UB2", short_ub3_to_ub2)
    _summ("SHORT UB2→UB1", short_ub2)
    _summ("LONG  LB1→UB1", long_lb1)
    _summ("LONG  LB2→UB1", long_lb2)
    _summ("LONG  LB3→UB1", long_lb3)

    # Export ONLY the requested strategy file
    short_ub3_to_ub2.to_csv("trades_short_UB3_to_UB2.csv", index=False)
    print("Wrote: trades_short_UB3_to_UB2.csv")

    # Combined single table
    combined_parts = []
    for df, name in [
        (short_ub3, "short_UB3_to_UB1"),
        (short_ub2, "short_UB2_to_UB1"),
        (long_lb1,  "long_LB1_to_UB1"),
        (long_lb2,  "long_LB2_to_UB1"),
        (long_lb3,  "long_LB3_to_UB1"),
    ]:
        if df is not None and not df.empty:
            x = df.copy()
            x["strategy"] = name
            combined_parts.append(x)
    combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
    if not combined.empty and WRITE_COMBINED_TRADES:
        print("\n=== ALL TOUCH→REVERT TRADES (combined) ===")
        try:
            print(combined.to_string(index=False))
        except Exception:
            print(combined.head(1000).to_string(index=False))
        combined.to_csv("trades_touch_revert_all.csv", index=False)
        print("\nWrote combined CSV: trades_touch_revert_all.csv")
    try:
        print("\nHead:")
        print(std_df.head(5).to_string())
        print("\nTail:")
        print(std_df.tail(5).to_string())
    except Exception:
        pass

    # Yearly exports (strict 252-bar window): 2019, 2020, 2021
    if WRITE_YEARLY_EXPORTS:
        print("\n" + "="*60)
        print("YEARLY EXPORTS (strict 252-bar window)")
        print("="*60)
        for ys in ["2019-01-01", "2020-01-01", "2021-01-01"]:
            ye = f"{int(ys[:4]) + 1}-01-01"
            print(f"Downloading {SYMBOL} for {ys} → {ye}")
            df_year = fetch_ohlc(SYMBOL, ys, ye, auto_adjust=False)
            std_year = compute_std_lines_strict(df_year, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF).dropna()
            out_path = f"std_lines_{ys}_{ye}.csv"
            std_year.to_csv(out_path, float_format="%.6f")
            print(f"  wrote: {out_path} rows={len(std_year)} first={std_year.index[0].date()} last={std_year.index[-1].date()}")

  