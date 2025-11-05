# hedge_overlay_diag.py
# Purpose: Reproduce Thinkorswim (TOS) Linear Regression SD Channels in Python,
# compare against your current "robust" channel, and show why a hedge did or did not
# trigger around a specific date (default: 2024-06-12).
#
# What this script does:
# - Downloads price data (yfinance) for the hedge symbol (QQQ by default).
# - Builds TWO versions of the channel:
#     1) "robust" (your current Python approach): rolling LR midline with robust (MAD) residual scale.
#     2) "tos_like": rolling LR midline on HL2 with a HighestAll 20-bar price stdev (constant band scale).
# - Builds the hysteresis-based hedge weight series (near=1/3, high=1/2) using your gates:
#     SMA200 gate, vol-aware eps (optional), slope gate for 1/2 (optional), dwell mins (optional).
# - Prints the channel values, price, SMA, eps thresholds, state and gates around the inspection date.
#
# Notes:
# - To match TOS exactly, set CHANNEL_MODE="tos_like", HEDGE_PRICE_SERIES="QQQ",
#   REQUIRE_SMA=True or False as you use in TOS, and use HL2 on the same symbol/timeframe.
# - TOS "InertiaAll" (Type=line) is anchored; here we match "Inertia" (Type=curve) with rolling LR.
#   If you want exact anchored behavior, switch TOS study to Type=curve. This will match closely.
#
# Requirements:
#   pip install yfinance pandas numpy
#
# Run:
#   python hedge_overlay_diag.py
#
# You can change DATE_TO_CHECK and CHANNEL_MODE below.

import pandas as pd
import numpy as np
import yfinance as yf
import math

# =========================
# Config
# =========================
DATA_START = "2010-01-01"
DATA_END = None  # None => today

# Which channel to compute
#   "robust"  -> your current Python approach (rolling LR with MAD residual SD, scaled)
#   "tos_like" -> TOS-like: rolling LR midline on HL2, HighestAll of 20-bar stdev(price)
CHANNEL_MODE = "tos_like"  # "robust" or "tos_like"

# Hedge price series and window
HEDGE_PRICE_SERIES = "QQQ"  # If TOS is on QQQ, set this to "QQQ"
HEDGE_WINDOW_N = 252
HEDGE_WIDTHS = (1.0, 2.0, 3.0, 4.0)
HEDGE_SD_SCALE = 0.95  # robust-only: scale SD of residuals

# TOS-like parameters
TOS_STDEV_LEN = 20
TOS_USE_HL2 = True       # match TOS code's input price = hl2
TOS_USE_HIGHEST_ALL = True  # emulate HighestAll(StDev(price))

# Hysteresis thresholds and gating
HEDGE_EPS_UB2 = 0.0015
HEDGE_EPS_UB3 = 0.0015
HEDGE_EPS_UB4 = 0.0300
HEDGE_W_NEAR = 1.0 / 3.0
HEDGE_W_HIGH = 0.50

REQUIRE_SMA = True         # require price > SMA200
HEDGE_SMA_LENGTH = 200

HEDGE_VOL_AWARE_EPS = True
HEDGE_VOL_LOOKBACK_D = 20
HEDGE_VOL_EPS_MIN = 0.5
HEDGE_VOL_EPS_MAX = 2.0

HEDGE_ESCALATE_NEAR_ON_UB4_LOWER = True  # escalate to 1/2 at UB4 lower band when already at 1/3
HEDGE_SLOPE_ESCALATION = True            # allow 1/2 only if midline slope <= thresh
HEDGE_SLOPE_LOOKBACK_D = 5
HEDGE_SLOPE_THRESH = 0.0

HEDGE_MIN_HOLD_ON_DAYS = 2  # require ≥2 consecutive days hedged before allowing turn-off
HEDGE_MIN_HOLD_OFF_DAYS = 2 # require ≥2 consecutive days unhedged before allowing turn-on

# Date to check
DATE_TO_CHECK = "2024-06-12"
WINDOW_BEFORE = 10
WINDOW_AFTER = 10

# =========================
# Download data
# =========================
if DATA_END is None:
    DATA_END = pd.Timestamp.today().date().isoformat()

tickers = sorted(set([HEDGE_PRICE_SERIES, "^GSPC"]))
px = yf.download(tickers, start=DATA_START, end=DATA_END, auto_adjust=False, progress=True)

# Reference index (business days where we have ^GSPC Adj Close)
if ("Adj Close" not in px) or ("^GSPC" not in px["Adj Close"].columns):
    raise RuntimeError("Could not load ^GSPC Adj Close from yfinance.")
ref_ac = px["Adj Close"]["^GSPC"].dropna()
idx = ref_ac.index

def adj_open(open_s, close_s, adj_close_s):
    return open_s * (adj_close_s / close_s)

# =========================
# Channel builders
# =========================
def build_lr_channels(price: pd.Series, n=252, widths=(1.0, 2.0, 3.0, 4.0), sd_scale=1.0) -> pd.DataFrame:
    # Robust (MAD) residual scale around rolling LR midline
    def mid_func(arr):
        x = np.arange(arr.size)
        m, b = np.polyfit(x, arr, 1)
        return b + m * (arr.size - 1)
    def sd_func(arr):
        x = np.arange(arr.size)
        m, b = np.polyfit(x, arr, 1)
        resid = arr - (b + m * x)
        mad = np.median(np.abs(resid - np.median(resid)))
        robust_sd = 1.4826 * mad
        return float(robust_sd)
    mid = price.rolling(n, min_periods=n).apply(mid_func, raw=True)
    sd  = price.rolling(n, min_periods=n).apply(sd_func, raw=True) * sd_scale
    out = {"mid": mid}
    for i, w in enumerate(widths, start=1):
        out[f"ub{i}"] = mid + w * sd
        out[f"lb{i}"] = mid - w * sd
    return pd.DataFrame(out)

def tos_style_channels(px_all, sym="QQQ", n=252, stdev_len=20, use_highest_all=True, use_hl2=True):
    if "High" not in px_all or "Low" not in px_all or "Close" not in px_all:
        raise RuntimeError("OHLC required for TOS-like channels (need High, Low, Close).")
    hi = px_all["High"][sym].reindex(idx)
    lo = px_all["Low"][sym].reindex(idx)
    cl = px_all["Close"][sym].reindex(idx)
    if use_hl2:
        price = ((hi + lo) / 2.0).reindex(idx)
    else:
        price = cl

    # Rolling linear regression midline (estimate at last bar)
    def lr_last(arr):
        x = np.arange(arr.size)
        m, b = np.polyfit(x, arr, 1)
        return b + m * (arr.size - 1)
    mid = price.rolling(n, min_periods=n).apply(lr_last, raw=True)

    # Price-based stdev (not residuals); length ~20 in thinkScript
    stdev_price = price.rolling(stdev_len, min_periods=stdev_len).std(ddof=1)

    if use_highest_all:
        # ThinkScript HighestAll -> max over all loaded bars; constant series
        sd_const_val = stdev_price.max(skipna=True)
        sd = pd.Series(sd_const_val, index=price.index)
    else:
        sd = stdev_price

    out = pd.DataFrame({"mid": mid})
    for i, w in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        out[f"ub{i}"] = mid + w * sd
        out[f"lb{i}"] = mid - w * sd
    return out, price

# =========================
# Hedge weight builder (band hysteresis)
# =========================
def build_band_hysteresis_hedge_weights(
    price: pd.Series,
    ub2: pd.Series,
    ub3: pd.Series,
    ub4: pd.Series,
    sma200: pd.Series,
    eps2=0.001,
    eps3=0.001,
    eps4=0.001,
    w_near=1.0/3.0,
    w_high=0.50,
    require_sma=True,
    allow_high=None,
    min_on_days: int = 0,
    min_off_days: int = 0,
) -> pd.Series:
    def _val(x, d, fallback):
        if isinstance(x, pd.Series):
            v = x.get(d, np.nan)
            return fallback if pd.isna(v) else float(v)
        return float(x)

    w = pd.Series(0.0, index=price.index)
    state = 0  # 0=off, 1=near, 2=high
    on_cnt = 0
    off_cnt = 1  # start as off

    for d in price.index:
        p = price.loc[d]
        m = sma200.loc[d] if d in sma200.index else np.nan
        u2 = ub2.loc[d] if d in ub2.index else np.nan
        u3 = ub3.loc[d] if d in ub3.index else np.nan
        u4 = ub4.loc[d] if d in ub4.index else np.nan
        if (np.isnan(p) or np.isnan(u2) or np.isnan(u3) or np.isnan(u4) or np.isnan(m)):
            w.iloc[w.index.get_loc(d)] = 0.0
            off_cnt += 1; on_cnt = 0; state = 0
            continue

        e2 = _val(eps2, d, 0.001)
        e3 = _val(eps3, d, 0.001)
        e4 = _val(eps4, d, 0.001)
        can_escalate = True if allow_high is None else bool(pd.Series(allow_high).get(d, False))

        # Propose next state
        ns = state
        if require_sma and not (p > m):
            ns = 0
        else:
            if state == 0:
                if can_escalate and p >= u4 * (1 - e4):
                    ns = 2
                elif p >= u3 * (1 - e3):
                    ns = 1
                else:
                    ns = 0
            elif state == 1:
                if p <= u2 * (1 - e2):
                    ns = 0
                else:
                    # Use UB4 lower band to escalate from NEAR if toggle is on; otherwise require UB4 upper band
                    thresh = u4 * ((1 - e4) if HEDGE_ESCALATE_NEAR_ON_UB4_LOWER else (1 + e4))
                    if can_escalate and p >= thresh:
                        ns = 2
                    else:
                        ns = 1
            else:  # state == 2
                if p <= u2 * (1 - e2):
                    ns = 0
                elif p <= u4 * (1 - e4):
                    if p >= u3 * (1 - e3):
                        ns = 1
                    else:
                        ns = 0
                else:
                    ns = 2

        # Dwell enforcement (off<->on only)
        curr_on, next_on = (state != 0), (ns != 0)
        if (not curr_on) and next_on and (off_cnt < int(min_off_days)):
            ns = 0
        if curr_on and (not next_on) and (on_cnt < int(min_on_days)):
            ns = state

        state = ns
        if state != 0:
            on_cnt += 1; off_cnt = 0
        else:
            off_cnt += 1; on_cnt = 0

        w.iloc[w.index.get_loc(d)] = w_near if state == 1 else (w_high if state == 2 else 0.0)

    return w

# =========================
# Build channels and hedge weights
# =========================
# Build hedge_price, channels, eps series, slope gate, hedge weight
if CHANNEL_MODE.lower() == "tos_like":
    # TOS-like: HL2 and HighestAll 20-bar stdev
    ch_df, hedge_price = tos_style_channels(
        px, sym=HEDGE_PRICE_SERIES, n=HEDGE_WINDOW_N,
        stdev_len=TOS_STDEV_LEN, use_highest_all=TOS_USE_HIGHEST_ALL,
        use_hl2=TOS_USE_HL2
    )
    ch_df = ch_df.reindex(idx)
else:
    # Robust: Adj Close, rolling LR + MAD residual SD
    try:
        hedge_price = px["Adj Close"][HEDGE_PRICE_SERIES].reindex(idx)
    except Exception:
        raise RuntimeError(f"Adj Close not found for {HEDGE_PRICE_SERIES}.")
    ch_df = build_lr_channels(hedge_price, n=HEDGE_WINDOW_N, widths=HEDGE_WIDTHS, sd_scale=HEDGE_SD_SCALE).reindex(idx)

sma200_hedge = hedge_price.rolling(HEDGE_SMA_LENGTH, min_periods=HEDGE_SMA_LENGTH).mean()

# Vol-aware hysteresis (scale eps by recent realized vol)
if HEDGE_VOL_AWARE_EPS:
    rv = hedge_price.pct_change().rolling(HEDGE_VOL_LOOKBACK_D).std()
    med = float(rv.median(skipna=True)) if not np.isnan(rv.median(skipna=True)) else 1.0
    scale = (rv / med).clip(lower=HEDGE_VOL_EPS_MIN, upper=HEDGE_VOL_EPS_MAX)
    eps2_s = (HEDGE_EPS_UB2 * scale).reindex(idx).fillna(HEDGE_EPS_UB2)
    eps3_s = (HEDGE_EPS_UB3 * scale).reindex(idx).fillna(HEDGE_EPS_UB3)
    eps4_s = (HEDGE_EPS_UB4 * scale).reindex(idx).fillna(HEDGE_EPS_UB4)
else:
    eps2_s, eps3_s, eps4_s = (pd.Series(HEDGE_EPS_UB2, index=idx),
                              pd.Series(HEDGE_EPS_UB3, index=idx),
                              pd.Series(HEDGE_EPS_UB4, index=idx))

# Slope gate for 1/2 escalation
if HEDGE_SLOPE_ESCALATION:
    mid = ch_df["mid"]
    slope = (mid - mid.shift(HEDGE_SLOPE_LOOKBACK_D)).reindex(idx)
    allow_high = (slope <= HEDGE_SLOPE_THRESH).fillna(False)
else:
    allow_high = None

hedge_w_series = build_band_hysteresis_hedge_weights(
    hedge_price,
    ch_df["ub2"], ch_df["ub3"], ch_df["ub4"],
    sma200_hedge,
    eps2=eps2_s, eps3=eps3_s, eps4=eps4_s,
    w_near=HEDGE_W_NEAR, w_high=HEDGE_W_HIGH,
    require_sma=REQUIRE_SMA,
    allow_high=allow_high,
    min_on_days=HEDGE_MIN_HOLD_ON_DAYS,
    min_off_days=HEDGE_MIN_HOLD_OFF_DAYS,
).reindex(idx).fillna(0.0)

# =========================
# Diagnostics around the date of interest
# =========================
d0 = pd.Timestamp(DATE_TO_CHECK)
if d0 not in idx:
    # Find nearest previous trading day
    pos = idx.searchsorted(d0, side="left")
    if pos == len(idx) or idx[pos] != d0:
        pos = max(0, pos - 1)
    d0 = idx[pos]

start = idx[max(0, idx.get_loc(d0) - WINDOW_BEFORE)]
end = idx[min(len(idx) - 1, idx.get_loc(d0) + WINDOW_AFTER)]
rng = slice(start, end)

def _get_eps(eps_series, d, fallback):
    return float(eps_series.get(d, fallback)) if isinstance(eps_series, pd.Series) else float(eps_series)

print("\n===== Hedge Overlay Diagnostics =====")
print(f"Channel mode: {CHANNEL_MODE}  |  Hedge Price Series: {HEDGE_PRICE_SERIES}  |  HL2 used: {TOS_USE_HL2 if CHANNEL_MODE=='tos_like' else 'n/a'}")
print(f"Inspecting date: {d0.date()}  (window {start.date()} to {end.date()})\n")

# Show series around the date
df_view = pd.DataFrame({
    "price": hedge_price.loc[rng],
    "sma200": sma200_hedge.loc[rng],
    "mid": ch_df["mid"].loc[rng],
    "ub2": ch_df["ub2"].loc[rng],
    "ub3": ch_df["ub3"].loc[rng],
    "ub4": ch_df["ub4"].loc[rng],
    "w": hedge_w_series.loc[rng],
})
print(df_view.round(4).to_string())

# Detailed gates for the chosen date
p  = float(hedge_price.loc[d0])
m  = float(sma200_hedge.loc[d0]) if not pd.isna(sma200_hedge.loc[d0]) else float("nan")
u2 = float(ch_df.loc[d0, "ub2"])
u3 = float(ch_df.loc[d0, "ub3"])
u4 = float(ch_df.loc[d0, "ub4"])
e2 = _get_eps(eps2_s, d0, HEDGE_EPS_UB2)
e3 = _get_eps(eps3_s, d0, HEDGE_EPS_UB3)
e4 = _get_eps(eps4_s, d0, HEDGE_EPS_UB4)
allow_hi = bool(pd.Series(allow_high).get(d0, False)) if (allow_high is not None) else True
state_w = float(hedge_w_series.loc[d0])

print("\n--- Gate check on", d0.date(), "---")
print(f"price: {p:.4f}  | sma200: {m:.4f}  | p>200SMA: {p > m if not math.isnan(m) else 'n/a'}")
print(f"ub2: {u2:.4f}  | ub3: {u3:.4f}  | ub4: {u4:.4f}")
print(f"eps2: {e2:.5f} | eps3: {e3:.5f} | eps4: {e4:.5f}")
print(f"p >= ub3*(1-eps3): {p >= u3*(1-e3)}")
if HEDGE_ESCALATE_NEAR_ON_UB4_LOWER:
    print(f"Escalate to 1/2 if p >= ub4*(1-eps4) AND allow_high: {p >= u4*(1-e4)} AND {allow_hi}")
else:
    print(f"Escalate to 1/2 if p >= ub4*(1+eps4) AND allow_high: {p >= u4*(1+e4)} AND {allow_hi}")
print(f"allow_high (slope gate ok): {allow_hi}")
print(f"hedge weight today (near=0.333, high=0.5): {state_w:.3f}")

print("\nTips:")
print("- If CHANNEL_MODE='tos_like', compare the mid/ubX/price values above to TOS labels on the same date.")
print("- To match TOS study closely, set your TOS to Type=curve (Inertia), price=HL2, and same symbol/timeframe.")
print("- If numbers match but w=0.0, gates are blocking (SMA, dwell, slope, or eps). Toggle REQUIRE_SMA, dwell, slope as needed.")
