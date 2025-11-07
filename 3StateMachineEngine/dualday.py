# dual_day_strategy.py
# Weekly 3-state engine with dual execution days:
# - Decide Thursday close -> execute next trading day open (usually Friday)
# - Decide Friday close   -> execute next trading day open (usually Monday)
# Keeps original gates, taxes, scheduler, and Nasdaq lockout.
# Adds: dashboard-style frontend (Decided/Entry, Next Thu/Fri decision windows, recent legs).

import pandas as pd, numpy as np, yfinance as yf
from pandas.tseries.offsets import BDay, Day
from collections import defaultdict
import math
from TOS_LR_Channels_Calc_v2 import compute_std_lines_strict, N as LRC_N, STDEV_PERIOD as LRC_STDEV, PRICE_SRC as LRC_PRICE_SRC

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# =========================
# Config
# =========================
DATA_START      = "1950-01-01"
DATA_END        = None  # None => today

DO_WALK_FORWARD = False

WF_START_DATE     = "1962-01-01"
WF_END_DATE       = "1979-12-31"
WF_TRAIN_YEARS    = 2
WF_TEST_YEARS     = 1
WF_USE_PAPER      = True
WF_CHAIN_CAPITAL  = True

IN_SAMPLE         = ("2000-01-01", "2000-01-01")
OOS_MAIN          = ("1999-01-01", None)
HOLDOUT           = ("2000-01-01", None)

START_CAPITAL   = 21000.0

# SMA/Signal params (3-state engine uses ^GSPC by default)
SIGNAL_SYMBOL = "^GSPC"
SMA_SLOW = 200
SMA_MID  = 100
BAND_PCT = 0.005

# 3-state behavior
ONE_SIDED_2X = True
ADAPTIVE_UP_WINDOW = True
MID_UP_WINDOW_WEEKS_BENIGN  = 4
MID_UP_WINDOW_WEEKS_STRESSED= 6

USE_SLOPE_FILTER = True
SLOPE_LOOKBACK_W = 4

# Additional gates
USE_DIST_GATE_200 = True
DELTA_200_PCT     = 0.010
USE_ADAPTIVE_DIST = True

USE_VOL_CAP       = True
VOL_LOOKBACK_D    = 20
VOL_TH_3X         = 0.35
VOL_TH_ROFF       = 0.45

USE_DD_THROTTLE   = True
DD_LOOKBACK_D     = 252
DD_TH_2X          = 0.20

# Risk-off (GLD vs IEF momentum)
USE_DUAL_RISK_OFF   = False   # GLD only by default
RISK_OFF_LOOKBACK_D = 63

# Trading frictions
SLIPPAGE_BPS = 5
SLIPPAGE_BPS_STRESS = 15
STRESS_THRESHOLD = 0.02

# Taxes (only if paper=False)
ST_RATE       = 0.37
LT_RATE       = 0.15
DIV_TAX_RATE  = 0.34
ORD_RATE      = 0.34
LOSS_DED_CAP  = 3000.0

# Proxy drags
LEV_ANNUAL_FEE_3X   = 0.0094
LEV_ANNUAL_FEE_2X   = 0.0095
GLD_ER_ANNUAL       = 0.0040
LEV_EXCESS          = 2.0
LEV_EXCESS_2X       = 1.0
FIN_FALLBACK_ANNUAL = 0.03
APPLY_PROXY_DRAGS   = True

# Proxy calibration
CALIBRATE_TQQQ_PROXY = True
CALIBRATE_GLD_PROXY  = False
CALIBRATE_SPXL_PROXY = True
CALIBRATE_QLD_PROXY  = True
CALIBRATE_SSO_PROXY  = True

# Wash-sale avoidance for Nasdaq legs
LOCKOUT_DAYS = 30

# Preferred gold proxy source (for pre-inception)
USE_LBMA_SPOT = True

# =========================
# LR Overlay (Mobius) Config
# =========================
ENABLE_LRC_OVERLAY       = True   # Tactical TQQQ hedging with SPY filter + TQQQ channels
LRC_APPLY_IN_STATES      = {2}             # Only 3x (UB3→UB2 hedging), disable 0x LB3 re-risk
LRC_REQUIRE_S200_POS     = True            # Slope gate must be ON
LRC_BACKBONE_FOR_RISK_OFF= "nasdaq"        # which equity backbone drives LB3 in 0x: "nasdaq" or "spx"
LRC_USE_INVERSE_IN_3X    = True            # in 3x, fade UB3 via inverse; else go GLD
LRC_N                    = 252             # regression window
LRC_STDEV                = 12              # stdev period
LRC_PRICE_SRC            = "close"         # "hl2" or "close" - using close to match TOS

# ============ Dashboard lineups (availability fallback only) ============
# Only used if the original choice has no data that day. Original order/lockout preserved.
RISK_ON_3X_LINEUP    = ["TQQQ", "SPXL", "UPRO", "UDOW", "TECL", "SOXL", "TNA", "FNGU", "SQQQ", "SPXU"]
RISK_ON_2X_LINEUP    = ["QLD",  "SSO",  "SPUU", "DDM",  "UWM",  "ROM",  "QID",  "SDS"]
RISK_OFF_GOLD_LINEUP = ["GLD", "IAU", "GLDM", "SGOL", "BAR", "AAAU"]
RISK_OFF_BOND_LINEUP = ["IEF", "VGIT", "GOVT", "SCHR", "IEI", "SHY", "TLH"]
lineup_syms = set(RISK_ON_3X_LINEUP + RISK_ON_2X_LINEUP + RISK_OFF_GOLD_LINEUP + RISK_OFF_BOND_LINEUP)

# =========================
# Helpers
# =========================
def cagr_from_curve(curve):
    if len(curve) < 2: return np.nan
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    if yrs <= 0: return np.nan
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1/yrs) - 1)

def max_drawdown_from_curve(curve):
    if curve.empty: return np.nan
    running_max = curve.cummax()
    dd = (curve / running_max - 1.0)
    return float(dd.min())

def annualized_vol_from_curve(curve):
    if len(curve) < 2: return np.nan
    rets = curve.pct_change().dropna()
    return float(rets.std() * math.sqrt(252))

def sharpe_from_curve(curve, rf=0.0):
    rets = curve.pct_change().dropna()
    if rets.empty: return np.nan
    ann_ret = (1 + rets.mean())**252 - 1
    ann_vol = rets.std() * math.sqrt(252)
    return float((ann_ret - rf) / ann_vol) if ann_vol > 0 else np.nan

def sortino_from_curve(curve, rf=0.0):
    rets = curve.pct_change().dropna()
    if rets.empty: return np.nan
    downside = rets[rets < 0]
    dd = downside.std() * math.sqrt(252)
    ann_ret = (1 + rets.mean())**252 - 1
    return float((ann_ret - rf) / dd) if dd > 0 else np.nan

def calmar_from_curve(curve):
    cagr = cagr_from_curve(curve)
    mdd  = max_drawdown_from_curve(curve)
    return float(cagr / abs(mdd)) if (mdd is not None and mdd < 0) else np.nan

def adj_open(open_s, close_s, adj_close_s):
    return open_s * (adj_close_s / close_s)

def seg_returns(ao, ac):
    gap   = (ao / ac.shift(1) - 1.0)
    intra = (ac / ao - 1.0)
    return gap, intra

def year_returns_from_curve(curve):
    if curve.empty: return pd.Series(dtype=float)
    y_end = curve.resample("YE").last()
    y_start = y_end.shift(1)
    yr = (y_end / y_start - 1.0).dropna()
    yr.index = yr.index.year
    return yr

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "nan"
    return f"{100.0 * x:.2f}%"

def next_trading_day(date_like):
    # Dashboard helper: use actual market index to find next trading day (holiday aware)
    d = pd.Timestamp(date_like).normalize()
    pos = idx.searchsorted(d, side="right")
    return idx[pos] if pos < len(idx) else None

# =========================
# Download data
# =========================
if DATA_END is None:
    DATA_END = (pd.Timestamp.today() + BDay(1)).date().isoformat()

# Ensure QQQ is included so overlay can use it instead of ^NDX
tickers = sorted(set(
    [SIGNAL_SYMBOL, "^GSPC", "^NDX", "SPY", "QQQ", "GC=F"] + list(lineup_syms)
))
dl_lookback = max(SMA_SLOW, SMA_MID, DD_LOOKBACK_D, RISK_OFF_LOOKBACK_D) + 10
dl_start = (pd.to_datetime(DATA_START) - BDay(dl_lookback)).date().isoformat()
px = yf.download(tickers, start=dl_start, end=DATA_END, auto_adjust=False, progress=False)

# Reference series for SMA/logic
if SIGNAL_SYMBOL not in px["Adj Close"].columns or px["Adj Close"][SIGNAL_SYMBOL].isnull().all():
    print(f"Warning: SIGNAL_SYMBOL {SIGNAL_SYMBOL} not found or all NaN. Defaulting to ^GSPC.")
    SIGNAL_SYMBOL = "^GSPC"

ref_ac = px["Adj Close"][SIGNAL_SYMBOL].dropna()
idx = ref_ac.index

# Daily SMA200 slope gate (shifted for next-open execution)
def build_slope200_ok_daily():
    s200 = ref_ac.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    look = SLOPE_LOOKBACK_W * 5
    s200_ok = (s200 - s200.shift(look)) > 0
    return s200_ok.shift(1).reindex(idx).fillna(False)

slope200_ok = build_slope200_ok_daily()

# Overlay event log for diagnostics
overlay_events = []

# Build adjusted open and adjusted close for symbols
def get_adj_ohlc(px, sym, idx):
    if sym not in px["Adj Close"].columns or px["Adj Close"][sym].isnull().all():
        ao = pd.Series(index=idx, dtype=float)
        ac = pd.Series(index=idx, dtype=float)
        cl = pd.Series(index=idx, dtype=float)
        return ao, ac, cl
    c  = px["Close"][sym].dropna()
    ao = adj_open(px["Open"][sym].reindex(c.index), c, px["Adj Close"][sym].reindex(c.index))
    ac = px["Adj Close"][sym].reindex(c.index)
    cl = px["Close"][sym].reindex(c.index)
    ao = ao.reindex(idx); ac = ac.reindex(idx); cl = cl.reindex(idx)
    return ao, ac, cl

ndx_ao, ndx_ac, ndx_cl     = get_adj_ohlc(px, "^NDX", idx)
tqqq_ao, tqqq_ac, tqqq_cl  = get_adj_ohlc(px, "TQQQ", idx)
gld_ao,  gld_ac,  gld_cl   = get_adj_ohlc(px, "GLD",  idx)
gcf_ao,  gcf_ac,  gcf_cl   = get_adj_ohlc(px, "GC=F", idx)
spy_ao,  spy_ac,  spy_cl   = get_adj_ohlc(px, "SPY",  idx)
spxl_ao, spxl_ac, spxl_cl  = get_adj_ohlc(px, "SPXL", idx)
qld_ao,  qld_ac,  qld_cl   = get_adj_ohlc(px, "QLD",  idx)
sso_ao,  sso_ac,  sso_cl   = get_adj_ohlc(px, "SSO",  idx)
ief_ao,  ief_ac,  ief_cl   = get_adj_ohlc(px, "IEF",  idx)
gspc_ac = px["Adj Close"]["^GSPC"].reindex(idx)

# =========================
# Backbone families and inverse helpers
# =========================
def family_of(sym: str) -> str:
    if sym in {"TQQQ","QLD","QQQ","SQQQ","QID"}: return "nasdaq"
    if sym in {"SPXL","SSO","SPY","SPXU","SDS"}: return "spx"
    return "riskoff"

def inverse_for(sym: str):
    if sym in {"TQQQ","QLD","QQQ"}:
        return "SQQQ" if sym in {"TQQQ","QQQ"} else "QID"
    if sym in {"SPXL","SSO","SPY"}:
        return "SPXU" if sym in {"SPXL","SPY"} else "SDS"
    return None

# =========================
# Build LR channels for backbones (strict rolling, no lookahead)
# =========================
def build_std_lines_backbone(sym: str) -> pd.DataFrame:
    try:
        df = pd.DataFrame({
            "open": px["Open"][sym].reindex(idx),
            "high": px["High"][sym].reindex(idx),
            "low":  px["Low"][sym].reindex(idx),
            "close":px["Close"][sym].reindex(idx),
        }).dropna()
        std = compute_std_lines_strict(df, n=LRC_N, stdev_len=LRC_STDEV, price_src=LRC_PRICE_SRC)
        return std.reindex(idx)
    except Exception:
        return pd.DataFrame(index=idx)

# Use QQQ for overlay (with ^NDX fallback if QQQ unavailable)
try:
    if "QQQ" in px["Close"].columns and px["Close"]["QQQ"].notna().sum() > 252:
        std_ndx = build_std_lines_backbone("QQQ")
    else:
        std_ndx = build_std_lines_backbone("^NDX")
except:
    std_ndx = build_std_lines_backbone("^NDX")

std_spy = build_std_lines_backbone("SPY")

def make_touch_masks(std_df: pd.DataFrame):
    if std_df is None or std_df.empty: 
        return (pd.Series(False, index=idx), pd.Series(False, index=idx), pd.Series(False, index=idx))
    d = std_df.dropna(subset=["high","low","close","UB1","UB2","UB3","LB1","LB2","LB3"]).copy()
    out_idx = d.index
    ub3_prev = ((d["high"] >= d["UB3"]) | (d["close"] >= d["UB3"])).shift(1).fillna(False)
    ub2_rev  = (d["close"] <= d["UB2"]).fillna(False)
    lb3_prev = ((d["low"]  <= d["LB3"]) | (d["close"] <= d["LB3"])).shift(1).fillna(False)
    # reindex to full idx
    return (ub3_prev.reindex(idx).fillna(False),
            ub2_rev.reindex(idx).fillna(False),
            lb3_prev.reindex(idx).fillna(False))

ub3_prev_ndx, ub2_rev_ndx, lb3_prev_ndx = make_touch_masks(std_ndx)
ub3_prev_spy, ub2_rev_spy, lb3_prev_spy = make_touch_masks(std_spy)

# Close-only signals for overlay (match standalone strategy logic)
def make_close_signals(std_df: pd.DataFrame):
    """Close above UB3 (prev day) for entry, Close <= UB2 for exit"""
    if std_df is None or std_df.empty:
        return (pd.Series(False, index=idx), pd.Series(False, index=idx))
    d = std_df.dropna(subset=["close","UB2","UB3"]).copy()
    ub3_close_prev = (d["close"] >= d["UB3"]).shift(1).fillna(False)
    ub2_close_exit = (d["close"] <= d["UB2"]).fillna(False)
    return (ub3_close_prev.reindex(idx).fillna(False),
            ub2_close_exit.reindex(idx).fillna(False))

ub3_close_ndx, ub2_close_exit_ndx = make_close_signals(std_ndx)
ub3_close_spy, ub2_close_exit_spy = make_close_signals(std_spy)

# LB2 reversion (close >= LB2) for LB3→LB2 exits
def make_lb2_revert(std_df: pd.DataFrame):
    if std_df is None or std_df.empty:
        return pd.Series(False, index=idx)
    d = std_df.dropna(subset=["close","LB2"]).copy()
    lb2_rev = (d["close"] >= d["LB2"]).fillna(False)
    return lb2_rev.reindex(idx).fillna(False)

lb2_rev_ndx = make_lb2_revert(std_ndx)
lb2_rev_spy = make_lb2_revert(std_spy)

# =========================
# Overlay gating
# =========================
def overlay_allowed(d: pd.Timestamp, regime: int) -> bool:
    if not ENABLE_LRC_OVERLAY:
        return False
    if regime not in LRC_APPLY_IN_STATES:
        return False
    if LRC_REQUIRE_S200_POS and not bool(slope200_ok.get(d, False)):
        return False
    return True

def spy_filter_active(d: pd.Timestamp) -> bool:
    """Check if SPY is between Midline and UB2 (filter for TQQQ hedge entry)"""
    if std_spy.empty:
        return False
    try:
        row = std_spy.loc[d]
        close = float(row.get("close", np.nan))
        mid = float(row.get("mid", np.nan))
        ub2 = float(row.get("UB2", np.nan))
        if np.isnan(close) or np.isnan(mid) or np.isnan(ub2):
            return False
        return mid <= close <= ub2
    except (KeyError, ValueError):
        return False

# Dividends
def get_div_series(sym, idx):
    try:
        if sym not in px["Close"].columns:
            return pd.Series(0.0, index=idx)
        d = yf.Ticker(sym).dividends
        if d is None or d.empty:
            return pd.Series(0.0, index=idx)
        d = d.sort_index()
        d.index = d.index.tz_localize(None)
        s = pd.Series(0.0, index=idx)
        ex_dates = idx.intersection(d.index)
        s.loc[ex_dates] = d.reindex(ex_dates).values
        return s
    except Exception:
        return pd.Series(0.0, index=idx)

div_tqqq = get_div_series("TQQQ", idx)
div_gld  = get_div_series("GLD",  idx)
div_spxl = get_div_series("SPXL", idx)
div_qld  = get_div_series("QLD",  idx)
div_sso  = get_div_series("SSO",  idx)
div_ief  = get_div_series("IEF",  idx)

# =========================
# Risk-free (daily) for financing; LBMA spot loader
# =========================
def load_rf_daily(idx):
    if pdr is None:
        return pd.Series(FIN_FALLBACK_ANNUAL/252.0, index=idx)
    try:
        dff = pdr.DataReader('DFF', 'fred', idx.min(), idx.max()).squeeze()
    except Exception:
        dff = None
    try:
        tb3 = pdr.DataReader('TB3MS', 'fred', idx.min(), idx.max()).squeeze()
        tb3 = tb3.resample('B').ffill()
    except Exception:
        tb3 = None

    if dff is None and tb3 is None:
        return pd.Series(FIN_FALLBACK_ANNUAL/252.0, index=idx)

    rf = None
    if dff is not None:
        rf = dff.reindex(idx)
    if rf is None or rf.isna().all():
        rf = pd.Series(index=idx, dtype=float)
    if tb3 is not None:
        rf = rf.fillna(tb3.reindex(idx))
    rf = rf.ffill().fillna(0.0)
    return rf / 100.0 / 252.0

def load_lbma_spot(idx):
    if (not USE_LBMA_SPOT) or (pdr is None):
        return None
    try:
        s = pdr.DataReader('GOLDPMGBD228NLBM', 'fred', idx.min(), idx.max()).squeeze()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='B')).ffill()
        return s.reindex(idx).ffill()
    except Exception:
        return None

rf_daily        = load_rf_daily(idx)
fee_daily_3x    = LEV_ANNUAL_FEE_3X / 252.0
fee_daily_2x    = LEV_ANNUAL_FEE_2X / 252.0
gld_fee_daily   = GLD_ER_ANNUAL     / 252.0

# =========================
# Build asset returns with pre-inception proxies
# =========================
def build_asset_returns_GLD():
    real_gap, real_intra = seg_returns(gld_ao, gld_ac)
    has = (~gld_ao.isna()) & (~gld_ac.isna())

    spot = load_lbma_spot(idx)
    if spot is not None and not spot.empty:
        ao_spot = spot.shift(1); ac_spot = spot
        proxy_gap, proxy_intra = seg_returns(ao_spot, ac_spot)
    else:
        proxy_gap, proxy_intra = seg_returns(gcf_ao, gcf_ac)

    if CALIBRATE_GLD_PROXY and has.any():
        real_T = ((1 + real_gap) * (1 + real_intra) - 1.0)
        prox_T = ((1 + proxy_gap) * (1 + proxy_intra) - 1.0)
        overlap = real_T.dropna().index.intersection(prox_T.dropna().index)
        if len(overlap) >= 200:
            b, a = np.polyfit(prox_T.loc[overlap].values, real_T.loc[overlap].values, 1)
            pre = ~has
            T_pre = ((1 + proxy_gap.loc[pre]) * (1 + proxy_intra.loc[pre]) - 1.0)
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = ((1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0)

    gap   = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied

def build_asset_returns_TQQQ():
    base_gap, base_intra = seg_returns(ndx_ao, ndx_ac)
    lev = 3.0
    proxy_gap   = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den         = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = ((1 + base_gap).fillna(1.0) * (1 + base_intra).fillna(1.0) - 1.0)
    proxy_intra = ((1.0 + lev * proxy_daily) / den - 1.0).clip(lower=-0.95).fillna(0.0)

    real_gap, real_intra = seg_returns(tqqq_ao, tqqq_ac)
    has = (~tqqq_ao.isna()) & (~tqqq_ac.isna())

    if CALIBRATE_TQQQ_PROXY:
        real_T = ((1 + real_gap) * (1 + real_intra) - 1.0)
        prox_T = ((1 + proxy_gap) * (1 + proxy_intra) - 1.0)
        overlap = real_T.dropna().index.intersection(prox_T.dropna().index)
        if len(overlap) >= 200:
            b, a = np.polyfit(prox_T.loc[overlap].values, real_T.loc[overlap].values, 1)
            pre = ~has
            T_pre = ((1 + proxy_gap.loc[pre]) * (1 + proxy_intra.loc[pre]) - 1.0)
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = ((1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0).clip(lower=-0.95)

    gap   = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied

def build_asset_returns_SPXL():
    base_gap, base_intra = seg_returns(spy_ao, spy_ac)
    lev = 3.0
    proxy_gap   = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den         = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = ((1 + base_gap).fillna(1.0) * (1 + base_intra).fillna(1.0) - 1.0)
    proxy_intra = ((1.0 + lev * proxy_daily) / den - 1.0).clip(lower=-0.95).fillna(0.0)

    real_gap, real_intra = seg_returns(spxl_ao, spxl_ac)
    has = (~spxl_ao.isna()) & (~spxl_ac.isna())

    if CALIBRATE_SPXL_PROXY:
        real_T = ((1 + real_gap) * (1 + real_intra) - 1.0)
        prox_T = ((1 + proxy_gap) * (1 + proxy_intra) - 1.0)
        overlap = real_T.dropna().index.intersection(prox_T.dropna().index)
        if len(overlap) >= 200:
            b, a = np.polyfit(prox_T.loc[overlap].values, real_T.loc[overlap].values, 1)
            pre = ~has
            T_pre = ((1 + proxy_gap.loc[pre]) * (1 + proxy_intra.loc[pre]) - 1.0)
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = ((1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0).clip(lower=-0.95)

    gap   = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied

def build_asset_returns_QLD():
    base_gap, base_intra = seg_returns(ndx_ao, ndx_ac)
    lev = 2.0
    proxy_gap   = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den         = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = ((1 + base_gap).fillna(1.0) * (1 + base_intra).fillna(1.0) - 1.0)
    proxy_intra = ((1.0 + lev * proxy_daily) / den - 1.0).clip(lower=-0.95).fillna(0.0)

    real_gap, real_intra = seg_returns(qld_ao, qld_ac)
    has = (~qld_ao.isna()) & (~qld_ac.isna())

    if CALIBRATE_QLD_PROXY:
        real_T = ((1 + real_gap) * (1 + real_intra) - 1.0)
        prox_T = ((1 + proxy_gap) * (1 + proxy_intra) - 1.0)
        overlap = real_T.dropna().index.intersection(prox_T.dropna().index)
        if len(overlap) >= 200:
            b, a = np.polyfit(prox_T.loc[overlap].values, real_T.loc[overlap].values, 1)
            pre = ~has
            T_pre = ((1 + proxy_gap.loc[pre]) * (1 + proxy_intra.loc[pre]) - 1.0)
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = ((1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0).clip(lower=-0.95)

    gap   = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied

def build_asset_returns_SSO():
    base_gap, base_intra = seg_returns(spy_ao, spy_ac)
    lev = 2.0
    proxy_gap   = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den         = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = ((1 + base_gap).fillna(1.0) * (1 + base_intra).fillna(1.0) - 1.0)
    proxy_intra = ((1.0 + lev * proxy_daily) / den - 1.0).clip(lower=-0.95).fillna(0.0)

    real_gap, real_intra = seg_returns(sso_ao, sso_ac)
    has = (~sso_ao.isna()) & (~sso_ac.isna())

    if CALIBRATE_SSO_PROXY:
        real_T = ((1 + real_gap) * (1 + real_intra) - 1.0)
        prox_T = ((1 + proxy_gap) * (1 + proxy_intra) - 1.0)
        overlap = real_T.dropna().index.intersection(prox_T.dropna().index)
        if len(overlap) >= 200:
            b, a = np.polyfit(prox_T.loc[overlap].values, real_T.loc[overlap].values, 1)
            pre = ~has
            T_pre = ((1 + proxy_gap.loc[pre]) * (1 + proxy_intra.loc[pre]) - 1.0)
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = ((1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0).clip(lower=-0.95)

    gap   = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied

gap_GLD,  intra_GLD,  prox_mask_GLD  = build_asset_returns_GLD()
gap_TQQQ, intra_TQQQ, prox_mask_TQQQ = build_asset_returns_TQQQ()
gap_SPXL, intra_SPXL, prox_mask_SPXL = build_asset_returns_SPXL()
gap_QLD,  intra_QLD,  prox_mask_QLD  = build_asset_returns_QLD()
gap_SSO,  intra_SSO,  prox_mask_SSO  = build_asset_returns_SSO()

prox_mask_GLD  = prox_mask_GLD.reindex(idx).fillna(False).astype(bool)
prox_mask_TQQQ = prox_mask_TQQQ.reindex(idx).fillna(False).astype(bool)
prox_mask_SPXL = prox_mask_SPXL.reindex(idx).fillna(False).astype(bool)
prox_mask_QLD  = prox_mask_QLD.reindex(idx).fillna(False).astype(bool)
prox_mask_SSO  = prox_mask_SSO.reindex(idx).fillna(False).astype(bool)

raw_close  = {"TQQQ": tqqq_cl, "SPXL": spxl_cl, "GLD": gld_cl, "QLD": qld_cl, "SSO": sso_cl, "IEF": ief_cl}
div_series = {"TQQQ": div_tqqq, "SPXL": div_spxl, "GLD": div_gld, "QLD": div_qld, "SSO": div_sso, "IEF": div_ief}

# =========================
# Regime builders (dual-anchor)
# =========================
def hysteresis_series(price_w, sma_w, band_pct):
    sig = pd.Series(index=price_w.index, dtype=int)
    prev = 0
    for dt in price_w.index:
        m = sma_w.loc[dt]
        if pd.isna(m):
            sig.loc[dt] = 0
            continue
        p  = price_w.loc[dt]
        up = (1 + band_pct) * m
        dn = (1 - band_pct) * m
        curr = prev
        if prev == 0 and p > up:
            curr = 1
        elif prev == 1 and p < dn:
            curr = 0
        sig.loc[dt] = curr
        prev = curr
    return sig

def _weekly_state_from_anchor(anchor="W-FRI"):
    # Weekly snapshots anchored to 'anchor' (e.g., "W-THU" or "W-FRI")
    sma200 = ref_ac.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    smamid = ref_ac.rolling(SMA_MID,  min_periods=SMA_MID ).mean()

    ref_w  = ref_ac.resample(anchor).last()
    s200_w = sma200.resample(anchor).last()
    smid_w = smamid.resample(anchor).last()

    sig200_w = hysteresis_series(ref_w, s200_w, BAND_PCT)
    sigmid_w = hysteresis_series(ref_w, smid_w, BAND_PCT)

    # Mid fresh upcross (weekly)
    up_mid = (sigmid_w.diff() == 1).astype(int)

    # Stress on ^NDX, weekly anchored to same 'anchor'
    ndx_ret = ndx_ac.pct_change()
    rv_d = ndx_ret.rolling(VOL_LOOKBACK_D).std() * math.sqrt(252)
    rv_w = rv_d.resample(anchor).last().reindex(ref_w.index).ffill()

    running_max_ndx = ndx_ac.rolling(DD_LOOKBACK_D, min_periods=max(1, int(DD_LOOKBACK_D*0.8))).max()
    dd_d = ndx_ac / running_max_ndx - 1.0
    dd_w = dd_d.resample(anchor).last().reindex(ref_w.index).ffill()

    stressed_w = (rv_w > VOL_TH_3X) | (dd_w <= -DD_TH_2X)

    # Base 3-state
    if ONE_SIDED_2X:
        if ADAPTIVE_UP_WINDOW:
            win_b = max(1, int(MID_UP_WINDOW_WEEKS_BENIGN))
            win_s = max(1, int(MID_UP_WINDOW_WEEKS_STRESSED))
            up_mid_win_b = up_mid.rolling(win_b, min_periods=1).max().fillna(0).astype(int)
            up_mid_win_s = up_mid.rolling(win_s, min_periods=1).max().fillna(0).astype(int)
            up_mid_window = up_mid_win_s.where(stressed_w, up_mid_win_b).astype(int)
        else:
            up_mid_window = up_mid.rolling(4, min_periods=1).max().fillna(0).astype(int)

        state_w = pd.Series(0, index=ref_w.index, dtype=int)
        state_w[(sig200_w == 0) & (up_mid_window == 1)] = 1
        state_w[sig200_w == 1] = 2
    else:
        state_w = pd.Series(0, index=ref_w.index, dtype=int)
        state_w[sigmid_w == 1] = 1
        state_w[sig200_w == 1] = 2

    # Slope gates (4 weeks apart within same anchoring)
    if USE_SLOPE_FILTER:
        slope200_pos = (s200_w - s200_w.shift(SLOPE_LOOKBACK_W)) > 0
        slope_mid_pos = (smid_w - smid_w.shift(SLOPE_LOOKBACK_W)) > 0
        state_w[(state_w == 2) & (~slope200_pos)] = 1
        state_w[(state_w == 1) & (~slope_mid_pos)] = 0

    # Distance gate (adaptive if stressed)
    if USE_DIST_GATE_200:
        dist200 = (ref_w / s200_w - 1.0)
        if USE_ADAPTIVE_DIST:
            state_w[(state_w == 2) & stressed_w & (dist200 <= DELTA_200_PCT)] = 1
        else:
            state_w[(state_w == 2) & (dist200 <= DELTA_200_PCT)] = 1

    # Vol cap
    if USE_VOL_CAP:
        state_w[(state_w == 2) & (rv_w > VOL_TH_3X)] = 1
        state_w[(state_w >= 1) & (rv_w > VOL_TH_ROFF)] = 0

    # Drawdown throttle
    if USE_DD_THROTTLE:
        state_w[(state_w == 2) & (dd_w <= -DD_TH_2X)] = 1
        state_w[(state_w == 1) & (dd_w <= -DD_TH_2X)] = 0

    return state_w

def _exec_next_bd(state_w):
    # Decide at weekly close, EXECUTE next trading-day open
    return state_w.reindex(idx).ffill().shift(1).fillna(0).astype(int)

def _combine_thu_fri_exec(exec_thu, exec_fri):
    # Combine two daily execution streams (Thu->Fri and Fri->Mon)
    ch_thu = exec_thu.ne(exec_thu.shift(1))
    ch_fri = exec_fri.ne(exec_fri.shift(1))
    out = pd.Series(index=idx, dtype=int)
    curr = int(exec_fri.iloc[0] if not pd.isna(exec_fri.iloc[0]) else 0)
    for d in idx:
        if bool(ch_thu.get(d, False)):
            curr = int(exec_thu.get(d, curr))
        if bool(ch_fri.get(d, False)):
            curr = int(exec_fri.get(d, curr))
        out.loc[d] = curr
    return out.astype(int)

def rebuild_regime():
    global swap_dates_all, regime_intraday_all, ref_ac, sma_slow, state_thu, state_fri
    # Ensure ref_ac uses current SIGNAL_SYMBOL
    if SIGNAL_SYMBOL in px["Adj Close"].columns and not px["Adj Close"][SIGNAL_SYMBOL].isnull().all():
        ref_ac = px["Adj Close"][SIGNAL_SYMBOL].dropna().reindex(idx)
    else:
        print(f"Warning: SIGNAL_SYMBOL {SIGNAL_SYMBOL} not found or NaN, using ^GSPC fallback.")
        ref_ac = px["Adj Close"]["^GSPC"].dropna().reindex(idx)

    sma_slow = ref_ac.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

    # Weekly states anchored to Thursday and Friday
    state_thu = _weekly_state_from_anchor("W-THU")
    state_fri = _weekly_state_from_anchor("W-FRI")

    # Execution streams (Thu->Fri, Fri->Mon)
    exec_thu = _exec_next_bd(state_thu)
    exec_fri = _exec_next_bd(state_fri)

    # Combined daily regime
    regime_intraday_all = _combine_thu_fri_exec(exec_thu, exec_fri)
    swap_dates_all = set(idx[regime_intraday_all.ne(regime_intraday_all.shift(1))])

# Warm-up
sma_slow = ref_ac.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
rebuild_regime()

# =========================
# Risk-off momentum (GLD vs IEF)
# =========================
def rebuild_risk_off_momentum():
    global HAVE_IEF, mom_gld, mom_ief, mom_gld_use, mom_ief_use, gap_IEF, intra_IEF
    HAVE_IEF = False

    def mom(ac, lb):
        return (ac / ac.shift(lb) - 1.0).reindex(idx)

    try:
        if "IEF" in px["Adj Close"].columns and ief_ac.notna().sum() > 100:
            gap_IEF, intra_IEF = seg_returns(ief_ao, ief_ac)
            gap_IEF = gap_IEF.fillna(0.0); intra_IEF = intra_IEF.fillna(0.0)
            HAVE_IEF = True
        else:
            gap_IEF = pd.Series(0.0, index=idx)
            intra_IEF = pd.Series(0.0, index=idx)
        mom_gld = mom(gld_ac, RISK_OFF_LOOKBACK_D).fillna(-1.0)
        mom_ief = mom(ief_ac, RISK_OFF_LOOKBACK_D).fillna(-1.0) if HAVE_IEF else pd.Series(-1.0, index=idx)
    except Exception:
        HAVE_IEF = False
        mom_gld = pd.Series(-1.0, index=idx)
        mom_ief = pd.Series(-1.0, index=idx)
        gap_IEF = pd.Series(0.0, index=idx)
        intra_IEF = pd.Series(0.0, index=idx)

    # Use prior day's momentum so Friday/Monday opens don't peek
    mom_gld_use = mom_gld.shift(1)
    mom_ief_use = mom_ief.shift(1)

rebuild_risk_off_momentum()

# =========================
# Scheduler (3-state, Nasdaq lockout) with availability fallback
# =========================
def build_locked_schedule(run_idx):
    regime_intraday = regime_intraday_all.reindex(run_idx).astype(int)
    swap_set = set(d for d in swap_dates_all if d in set(run_idx))

    # Core gap/intra maps including extras
    gap_map   = {"TQQQ": gap_TQQQ, "SPXL": gap_SPXL, "QLD": gap_QLD, "SSO": gap_SSO, "GLD": gap_GLD}
    intra_map = {"TQQQ": intra_TQQQ, "SPXL": intra_SPXL, "QLD": intra_QLD, "SSO": intra_SSO, "GLD": intra_GLD}
    if globals().get("HAVE_IEF", False):
        gap_map["IEF"] = gap_IEF; intra_map["IEF"] = intra_IEF

    # extras (no proxies)
    gap_extra, intra_extra, ac_extra = {}, {}, {}
    extra_syms = sorted(lineup_syms - {"TQQQ", "SPXL", "QLD", "SSO", "GLD", "IEF"})
    for sym in extra_syms:
        try:
            ao_e, ac_e, _ = get_adj_ohlc(px, sym, idx)
            g_e, i_e = seg_returns(ao_e, ac_e)
            gap_extra[sym]   = g_e.fillna(0.0)
            intra_extra[sym] = i_e.fillna(0.0)
            ac_extra[sym]    = ac_e
        except Exception:
            gap_extra[sym]   = pd.Series(0.0, index=idx)
            intra_extra[sym] = pd.Series(0.0, index=idx)
            ac_extra[sym]    = pd.Series(index=idx, dtype=float)
    gap_map.update(gap_extra); intra_map.update(intra_extra)

    def has_data(sym, d):
        if sym in {"TQQQ","SPXL","QLD","SSO","GLD"}:
            base_ac = {"TQQQ": tqqq_ac, "SPXL": spxl_ac, "QLD": qld_ac, "SSO": sso_ac, "GLD": gld_ac}[sym]
            return pd.notna(base_ac.get(d, np.nan))
        if sym == "IEF":
            return globals().get("HAVE_IEF", False) and pd.notna(ief_ac.get(d, np.nan))
        return pd.notna(ac_extra.get(sym, pd.Series(index=idx, dtype=float)).get(d, np.nan))

    def choose_risk_off(d):
        if USE_DUAL_RISK_OFF and HAVE_IEF:
            return "GLD" if float(mom_gld_use.get(d, -1.0)) >= float(mom_ief_use.get(d, -1.0)) else "IEF"
        return "GLD"

    # Original selection (unchanged)
    def choose_asset(regime, d, lock_until):
        if regime == 2:   # 3x
            return "SPXL" if d <= lock_until else "TQQQ"
        elif regime == 1: # 2x
            return "SSO"  if d <= lock_until else "QLD"
        else:
            return choose_risk_off(d)

    # Availability fallback ONLY if the chosen symbol has no data (preserve original order)
    def pick_available_from_lineup(base, regime, d, lock_until):
        if regime == 2:
            prim_order = ["SPXL", "TQQQ"] if d <= lock_until else ["TQQQ", "SPXL"]
            tail = [s for s in RISK_ON_3X_LINEUP if s not in {"TQQQ", "SPXL"}]
            # if locked, do not allow TQQQ anywhere
            if d <= lock_until:
                tail = [s for s in tail if s != "TQQQ"]
            candidates = [base] + [p for p in prim_order if p != base] + tail
        elif regime == 1:
            prim_order = ["SSO", "QLD"] if d <= lock_until else ["QLD", "SSO"]
            tail = [s for s in RISK_ON_2X_LINEUP if s not in {"QLD", "SSO"}]
            if d <= lock_until:
                tail = [s for s in tail if s != "QLD"]
            candidates = [base] + [p for p in prim_order if p != base] + tail
        else:
            if base == "GLD":
                candidates = ["GLD"] + [s for s in RISK_OFF_GOLD_LINEUP if s != "GLD"]
            else:
                candidates = ["IEF"] + [s for s in RISK_OFF_BOND_LINEUP if s != "IEF"]
        for s in candidates:
            if has_data(s, d):
                return s
        return base  # last resort

    sched = pd.Series(index=run_idx, dtype=object)
    lockout_until = pd.Timestamp("1900-01-01")
    curr_asset = None
    entry_date = None
    leg_gross = 1.0
    # Overlay state
    overlay_on = False
    overlay_family = None
    overlay_mode = None  # "ub3_fade" or "lb3_rerisk"
    overlay_start_date = None

    for i, d in enumerate(run_idx):
        regime = int(regime_intraday.get(d, 0))

        if i == 0:
            base = choose_asset(regime, d, lockout_until)
            curr_asset = pick_available_from_lineup(base, regime, d, lockout_until)
            entry_date = d
            leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            sched.iloc[i] = curr_asset
            continue

        # Compute base asset selection on swap days; otherwise base defaults to current asset
        if d in swap_set:
            leg_gross *= (1.0 + gap_map[curr_asset].get(d, 0.0))
            # Family-wide lockout for Nasdaq legs (long or inverse) on realized loss
            if (family_of(curr_asset) == "nasdaq") and (leg_gross - 1.0 < 0.0):
                lockout_until = d + Day(LOCKOUT_DAYS)

            base = choose_asset(regime, d, lockout_until)
            next_asset = pick_available_from_lineup(base, regime, d, lockout_until)

            # Overlay sequencing on swap day
            fam = family_of(next_asset)
            # Select backbone masks
            if fam == "nasdaq":
                ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_ndx, ub2_rev_ndx, lb3_prev_ndx, lb2_rev_ndx
            elif fam == "spx":
                ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_spy, ub2_rev_spy, lb3_prev_spy, lb2_rev_spy
            else:
                if LRC_BACKBONE_FOR_RISK_OFF == "nasdaq":
                    ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_ndx, ub2_rev_ndx, lb3_prev_ndx, lb2_rev_ndx
                else:
                    ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_spy, ub2_rev_spy, lb3_prev_spy, lb2_rev_spy

            # 1) Exit overlay on reversion or gating off
            if overlay_on and (
                (overlay_mode == "ub3_fade" and bool(ub2_close_exit_ndx.get(d, False))) or
                (overlay_mode == "lb3_rerisk" and bool(lb2_rev.get(d, False)))
            ):
                # Log overlay exit
                try:
                    if overlay_mode == "ub3_fade" and bool(ub2_close_exit_ndx.get(d, False)):
                        exit_reason = "UB2"
                    elif overlay_mode == "lb3_rerisk" and bool(lb2_rev.get(d, False)):
                        exit_reason = "LB2"
                    elif regime not in LRC_APPLY_IN_STATES:
                        exit_reason = "regime_exit"
                    elif (LRC_REQUIRE_S200_POS and not bool(slope200_ok.get(d, False))):
                        exit_reason = "slope_off"
                    else:
                        exit_reason = "other"
                    if overlay_start_date is not None:
                        overlay_events.append({
                            "mode": overlay_mode,
                            "family": overlay_family,
                            "start": overlay_start_date,
                            "end": d,
                            "days": (d - overlay_start_date).days,
                            "exit_reason": exit_reason
                        })
                except Exception:
                    pass
                # After LB3→LB2 or UB3→UB2 exit, always revert to base schedule
                next_asset = pick_available_from_lineup(base, regime, d, lockout_until)
                overlay_on = False
                overlay_family = None
                overlay_mode = None
                overlay_start_date = None
            # 2) 3x UB3 fade (state=2) - ONLY for TQQQ with SPY filter
            elif (not overlay_on) and regime == 2 and overlay_allowed(d, regime):
                # Only hedge TQQQ when SPY filter is active and ^NDX closed above UB3
                if next_asset == "TQQQ" and spy_filter_active(d) and bool(ub3_close_ndx.get(d, False)):
                    if LRC_USE_INVERSE_IN_3X:
                        inv = inverse_for(next_asset)
                        if inv and has_data(inv, d):
                            next_asset = inv
                            overlay_on = True
                            overlay_family = fam
                            overlay_mode = "ub3_fade"
                            overlay_start_date = d
                        else:
                            next_asset = "GLD"
                    else:
                        next_asset = "GLD"
                        overlay_on = True
                        overlay_family = fam
                        overlay_mode = "ub3_fade"
                        overlay_start_date = d
            # 3) 0x LB3 early re-risk (state=0)
            elif (not overlay_on) and regime == 0 and overlay_allowed(d, regime):
                # Choose backbone for LB3 in 0x per config
                use_lb3 = lb3_prev_ndx if LRC_BACKBONE_FOR_RISK_OFF == "nasdaq" else lb3_prev_spy
                if bool(use_lb3.get(d, False)):
                    desired = choose_asset(2, d, lockout_until)  # prefer 3x
                    next_asset = pick_available_from_lineup(desired, 2, d, lockout_until)
                    overlay_on = True
                    overlay_family = family_of(next_asset)
                    overlay_mode = "lb3_rerisk"
                    overlay_start_date = d

            curr_asset = next_asset
            entry_date = d
            leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            sched.iloc[i] = curr_asset
        else:
            # Non-swap day: allow overlay flips at open
            base = curr_asset
            next_asset = curr_asset

            fam = family_of(base)
            if fam == "nasdaq":
                ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_ndx, ub2_rev_ndx, lb3_prev_ndx, lb2_rev_ndx
            elif fam == "spx":
                ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_spy, ub2_rev_spy, lb3_prev_spy, lb2_rev_spy
            else:
                if LRC_BACKBONE_FOR_RISK_OFF == "nasdaq":
                    ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_ndx, ub2_rev_ndx, lb3_prev_ndx, lb2_rev_ndx
                else:
                    ub3_prev, ub2_rev, lb3_prev, lb2_rev = ub3_prev_spy, ub2_rev_spy, lb3_prev_spy, lb2_rev_spy

            # 1) Exit overlay on reversion or gating off
            if overlay_on and (
                (overlay_mode == "ub3_fade" and bool(ub2_close_exit_ndx.get(d, False))) or
                (overlay_mode == "lb3_rerisk" and bool(lb2_rev.get(d, False)))
            ):
                # Log overlay exit
                try:
                    if overlay_mode == "ub3_fade" and bool(ub2_close_exit_ndx.get(d, False)):
                        exit_reason = "UB2"
                    elif overlay_mode == "lb3_rerisk" and bool(lb2_rev.get(d, False)):
                        exit_reason = "LB2"
                    elif regime not in LRC_APPLY_IN_STATES:
                        exit_reason = "regime_exit"
                    elif (LRC_REQUIRE_S200_POS and not bool(slope200_ok.get(d, False))):
                        exit_reason = "slope_off"
                    else:
                        exit_reason = "other"
                    if overlay_start_date is not None:
                        overlay_events.append({
                            "mode": overlay_mode,
                            "family": overlay_family,
                            "start": overlay_start_date,
                            "end": d,
                            "days": (d - overlay_start_date).days,
                            "exit_reason": exit_reason
                        })
                except Exception:
                    pass
                # After LB3→LB2 or UB3→UB2 exit, always revert to base schedule
                next_asset = base
                overlay_on = False
                overlay_family = None
                overlay_mode = None
                overlay_start_date = None
            # 2) 3x UB3 fade (state=2) - ONLY for TQQQ with SPY filter
            elif (not overlay_on) and regime == 2 and overlay_allowed(d, regime):
                # Only hedge TQQQ when SPY filter is active and ^NDX closed above UB3
                if base == "TQQQ" and spy_filter_active(d) and bool(ub3_close_ndx.get(d, False)):
                    if LRC_USE_INVERSE_IN_3X:
                        inv = inverse_for(base)
                        if inv and has_data(inv, d):
                            next_asset = inv
                            overlay_on = True
                            overlay_family = fam
                            overlay_mode = "ub3_fade"
                            overlay_start_date = d
                        else:
                            next_asset = "GLD"
                            overlay_on = True
                            overlay_family = fam
                            overlay_mode = "ub3_fade"
                            overlay_start_date = d
                    else:
                        next_asset = "GLD"
                        overlay_on = True
                        overlay_family = fam
                        overlay_mode = "ub3_fade"
                        overlay_start_date = d
            # 3) 0x LB3 early re-risk (state=0)
            elif (not overlay_on) and regime == 0 and overlay_allowed(d, regime):
                use_lb3 = lb3_prev_ndx if LRC_BACKBONE_FOR_RISK_OFF == "nasdaq" else lb3_prev_spy
                if bool(use_lb3.get(d, False)):
                    desired = choose_asset(2, d, lockout_until)  # prefer 3x
                    next_asset = pick_available_from_lineup(desired, 2, d, lockout_until)
                    overlay_on = True
                    overlay_family = family_of(next_asset)
                    overlay_mode = "lb3_rerisk"
                    overlay_start_date = d

            # Apply returns for the day with potential open swap
            if next_asset != curr_asset:
                # apply gap on existing, then switch at open
                leg_gross *= (1.0 + gap_map[curr_asset].get(d, 0.0))
                curr_asset = next_asset
                entry_date = d
                leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            else:
                leg_gross *= (1.0 + gap_map[curr_asset].get(d, 0.0)) * (1.0 + intra_map[curr_asset].get(d, 0.0))
            sched.iloc[i] = curr_asset

    return sched

# =========================
# Simulation engine
# =========================
# Use prior day's move for "stress slippage" toggle (no lookahead at open)
ref_ret_abs = gspc_ac.pct_change().abs().shift(1).reindex(idx).fillna(0.0)

def trade_cost(d):
    bps = SLIPPAGE_BPS_STRESS if (d in ref_ret_abs.index and ref_ret_abs.loc[d] >= STRESS_THRESHOLD) else SLIPPAGE_BPS
    return bps / 10000.0

def proxy_drag(asset, d):
    if not APPLY_PROXY_DRAGS:
        return 1.0
    if asset == "TQQQ" and bool(prox_mask_TQQQ.get(d, False)):
        if CALIBRATE_TQQQ_PROXY: return 1.0
        fin_d = LEV_EXCESS * float(rf_daily.get(d, 0.0)) if d in rf_daily.index else LEV_EXCESS * (FIN_FALLBACK_ANNUAL/252.0)
        return (1.0 - fee_daily_3x) * (1.0 - fin_d)
    if asset == "SPXL" and bool(prox_mask_SPXL.get(d, False)):
        if CALIBRATE_SPXL_PROXY: return 1.0
        fin_d = LEV_EXCESS * float(rf_daily.get(d, 0.0)) if d in rf_daily.index else LEV_EXCESS * (FIN_FALLBACK_ANNUAL/252.0)
        return (1.0 - fee_daily_3x) * (1.0 - fin_d)
    if asset == "QLD" and bool(prox_mask_QLD.get(d, False)):
        if CALIBRATE_QLD_PROXY: return 1.0
        fin_d = LEV_EXCESS_2X * float(rf_daily.get(d, 0.0)) if d in rf_daily.index else LEV_EXCESS_2X * (FIN_FALLBACK_ANNUAL/252.0)
        return (1.0 - fee_daily_2x) * (1.0 - fin_d)
    if asset == "SSO" and bool(prox_mask_SSO.get(d, False)):
        if CALIBRATE_SSO_PROXY: return 1.0
        fin_d = LEV_EXCESS_2X * float(rf_daily.get(d, 0.0)) if d in rf_daily.index else LEV_EXCESS_2X * (FIN_FALLBACK_ANNUAL/252.0)
        return (1.0 - fee_daily_2x) * (1.0 - fin_d)
    if asset == "GLD" and bool(prox_mask_GLD.get(d, False)):
        return (1.0 - gld_fee_daily)
    return 1.0

def sim_strategy(run_idx, paper=False):
    gap_map   = {"TQQQ": gap_TQQQ, "SPXL": gap_SPXL, "QLD": gap_QLD, "SSO": gap_SSO, "GLD": gap_GLD}
    intra_map = {"TQQQ": intra_TQQQ, "SPXL": intra_SPXL, "QLD": intra_QLD, "SSO": intra_SSO, "GLD": intra_GLD}
    if globals().get("HAVE_IEF", False):
        gap_map["IEF"] = gap_IEF; intra_map["IEF"] = intra_IEF

    # Add extra lineup symbols so schedule fallback (if ever used) can value correctly
    # (These extras have no proxies; used only if base/fallback lack data)
    # Build once (cheap) here:
    gap_extra, intra_extra = {}, {}
    extra_syms = sorted(lineup_syms - {"TQQQ", "SPXL", "QLD", "SSO", "GLD", "IEF"})
    for sym in extra_syms:
        try:
            ao_e, ac_e, _ = get_adj_ohlc(px, sym, idx)
            g_e, i_e = seg_returns(ao_e, ac_e)
            gap_extra[sym]   = g_e.fillna(0.0)
            intra_extra[sym] = i_e.fillna(0.0)
        except Exception:
            gap_extra[sym]   = pd.Series(0.0, index=idx)
            intra_extra[sym] = pd.Series(0.0, index=idx)
    gap_map.update(gap_extra); intra_map.update(intra_extra)

    sched = build_locked_schedule(run_idx)

    if paper:
        eq = START_CAPITAL
        equity_curve = []
        daily_index = []
        curr_asset = None

        for i, d in enumerate(run_idx):
            next_asset = sched.loc[d]

            if curr_asset is None:
                curr_asset = next_asset
                eq *= (1.0 + intra_map[curr_asset].get(d, 0.0))
                equity_curve.append(eq); daily_index.append(d)
                continue

            eq *= (1.0 + gap_map[curr_asset].get(d, 0.0))
            if next_asset != curr_asset:
                curr_asset = next_asset
            eq *= (1.0 + intra_map[curr_asset].get(d, 0.0))

            equity_curve.append(eq); daily_index.append(d)
        return {"equity_curve": pd.Series(equity_curve, index=pd.DatetimeIndex(daily_index))}

    # REAL branch (taxes)
    eq_pre   = START_CAPITAL
    eq_after = START_CAPITAL
    equity_curve = []
    daily_index = []

    curr_asset = None
    entry_date = None
    entry_eq_pre_after_buy = 0.0
    cum_div_leg_pre = 0.0

    realized_ST = defaultdict(float)
    realized_LT = defaultdict(float)
    carry_ST = 0.0; carry_LT = 0.0
    div_tax_by_year = defaultdict(float)

    eq_pre_prev_close = None
    prev_d = None

    for i, d in enumerate(run_idx):
        y = d.year
        next_asset = sched.loc[d]

        if curr_asset is None:
            curr_asset = next_asset
            c = trade_cost(d)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)
            entry_date = d
            entry_eq_pre_after_buy = eq_pre
            cum_div_leg_pre = 0.0

            drag = proxy_drag(curr_asset, d)
            eq_pre   *= drag
            eq_after *= drag
            eq_pre   *= (1.0 + intra_map[curr_asset].get(d, 0.0))
            eq_after *= (1.0 + intra_map[curr_asset].get(d, 0.0))
            equity_curve.append(eq_after); daily_index.append(d)
            eq_pre_prev_close = eq_pre
            prev_d = d
            continue

        div_amt = float(div_series.get(curr_asset, pd.Series(0.0, index=run_idx)).loc[d])
        if div_amt != 0.0 and prev_d is not None:
            prev_close_raw = raw_close[curr_asset].loc[prev_d]
            if pd.notna(prev_close_raw) and prev_close_raw != 0.0 and eq_pre_prev_close is not None:
                div_cash = float(eq_pre_prev_close) * (div_amt / float(prev_close_raw))
                div_tax  = DIV_TAX_RATE * div_cash
                eq_after -= div_tax
                div_tax_by_year[y] += div_tax
                cum_div_leg_pre += div_cash

        g = gap_map[curr_asset].get(d, 0.0)
        eq_pre   *= (1.0 + g)
        eq_after *= (1.0 + g)

        if next_asset != curr_asset:
            c = trade_cost(d)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)

            hold_days = (d - entry_date).days
            realized_pre = eq_pre - entry_eq_pre_after_buy - cum_div_leg_pre
            if hold_days > 365:
                realized_LT[y] += realized_pre
            else:
                realized_ST[y] += realized_pre

            curr_asset = next_asset
            c = trade_cost(d)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)

            entry_date = d
            entry_eq_pre_after_buy = eq_pre
            cum_div_leg_pre = 0.0

            drag = proxy_drag(curr_asset, d)
            eq_pre   *= drag
            eq_after *= drag
            eq_pre   *= (1.0 + intra_map[curr_asset].get(d, 0.0))
            eq_after *= (1.0 + intra_map[curr_asset].get(d, 0.0))
        else:
            drag = proxy_drag(curr_asset, d)
            eq_pre   *= drag
            eq_after *= drag
            eq_pre   *= (1.0 + intra_map[curr_asset].get(d, 0.0))
            eq_after *= (1.0 + intra_map[curr_asset].get(d, 0.0))

        # Year-end taxes
        is_year_end = (i == len(run_idx) - 1) or (run_idx[i+1].year != y)
        if is_year_end:
            st = realized_ST[y] + carry_ST
            lt = realized_LT[y] + carry_LT
            if st > 0 and lt < 0:
                off = min(st, -lt); st -= off; lt += off
            elif st < 0 and lt > 0:
                off = min(lt, -st); lt -= off; st += off

            tax_cap = 0.0; new_cf_st = 0.0; new_cf_lt = 0.0
            if st >= 0 and lt >= 0:
                tax_cap = ST_RATE * st + LT_RATE * lt
            elif st <= 0 and lt <= 0:
                net_loss = -(st + lt)
                deduct = min(LOSS_DED_CAP, net_loss)
                tax_cap = -ORD_RATE * deduct
                rem = net_loss - deduct
                mag_st = -st; mag_lt = -lt
                if (mag_st + mag_lt) > 0:
                    share_st = rem * (mag_st / (mag_st + mag_lt))
                    share_lt = rem - share_st
                    new_cf_st = -share_st
                    new_cf_lt = -share_lt
            else:
                if st > 0: tax_cap += ST_RATE * st
                if lt > 0: tax_cap += LT_RATE * lt

            eq_after -= tax_cap
            carry_ST, carry_LT = new_cf_st, new_cf_lt
            eq_pre = eq_after

        equity_curve.append(eq_after); daily_index.append(d)
        eq_pre_prev_close = eq_pre
        prev_d = d

    equity_curve = pd.Series(equity_curve, index=pd.DatetimeIndex(daily_index))
    return {"equity_curve": equity_curve}

# =========================
# Segment runner
# =========================
def build_run_index(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) if end_date else idx[-1]
    mask = (idx >= start_date) & (idx <= end_date) & sma_slow.notna()
    run_idx = idx[mask]
    return run_idx

def metrics_from_curve(curve):
    if curve is None or curve.empty:
        return {}
    cagr  = cagr_from_curve(curve)
    mdd   = max_drawdown_from_curve(curve)
    vol   = annualized_vol_from_curve(curve)
    shrp  = sharpe_from_curve(curve)
    sort  = sortino_from_curve(curve)
    calm  = calmar_from_curve(curve)
    yr    = year_returns_from_curve(curve)
    best_y = float(yr.max()) if not yr.empty else np.nan
    worst_y= float(yr.min()) if not yr.empty else np.nan
    win_rate = float((yr > 0).mean()) if not yr.empty else np.nan
    return {
        "Final Value": float(curve.iloc[-1]),
        "CAGR": cagr,
        "MaxDD": mdd,
        "AnnVol": vol,
        "Sharpe": shrp,
        "Sortino": sort,
        "Calmar": calm,
        "BestYear": best_y,
        "WorstYear": worst_y,
        "WinRateYears": win_rate
    }

def print_metrics(tag, m):
    if not m:
        print(f"{tag}: no data")
        return
    print(f"{tag}:")
    print(f"  Final Value: ${m['Final Value']:,.2f}")
    print(f"  CAGR: {fmt_pct(m['CAGR'])}")
    print(f"  Max Drawdown: {fmt_pct(m['MaxDD'])}")
    print(f"  Ann Vol: {fmt_pct(m['AnnVol'])}")
    print(f"  Sharpe: {m['Sharpe']:.2f}  |  Sortino: {m['Sortino']:.2f}  |  Calmar: {m['Calmar']:.2f}")
    print(f"  Best Year: {fmt_pct(m['BestYear'])}  |  Worst Year: {fmt_pct(m['WorstYear'])}  |  % Winning Years: {100.0*m['WinRateYears']:.1f}%")
    print()

def run_segment(name, start, end):
    run_idx = build_run_index(start, end)
    if len(run_idx) == 0:
        print(f"{name}: no run days (check dates/SMA warm-up).")
        return None, None, None
    res_paper = sim_strategy(run_idx, paper=True)
    res_real  = sim_strategy(run_idx, paper=False)
    curve_paper = res_paper["equity_curve"]
    curve_real  = res_real["equity_curve"]
    m_paper = metrics_from_curve(curve_paper)
    m_real  = metrics_from_curve(curve_real)

    print(f"\n--- Yearly Returns for {name} ---")
    yr_paper = year_returns_from_curve(curve_paper).apply(fmt_pct)
    yr_real  = year_returns_from_curve(curve_real).apply(fmt_pct)
    df_yr = pd.DataFrame({"Paper": yr_paper, "Real": yr_real}); df_yr.index.name = "Year"
    if not df_yr.empty:
        print(df_yr.to_string())
    else:
        print("No yearly data to display for this segment.")
    print("--------------------------------------\n")

    return (curve_paper, curve_real, (m_paper, m_real))

# =========================
# Pure 2-state baseline (dual anchor)
# =========================
def build_regime_by_SMA_baseline_dual():
    ref = px["Adj Close"]["^GSPC"].dropna().reindex(idx)
    sma = ref.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()

    def baseline_state(anchor):
        ref_w = ref.resample(anchor).last()
        sma_w = sma.resample(anchor).last()
        reg_w = pd.Series(index=ref_w.index, dtype=int)
        prev = 0
        for dt in ref_w.index:
            m = sma_w.loc[dt]
            if pd.isna(m):
                reg_w.loc[dt] = 0; continue
            p = ref_w.loc[dt]
            up = (1 + BAND_PCT) * m
            dn = (1 - BAND_PCT) * m
            curr = 1 if (prev == 0 and p > up) else (0 if (prev == 1 and p < dn) else prev)
            reg_w.loc[dt] = curr; prev = curr
        return reg_w.replace({1: 2, 0: 0}).astype(int)

    state_thu_b = baseline_state("W-THU")
    state_fri_b = baseline_state("W-FRI")
    exec_thu_b  = _exec_next_bd(state_thu_b)
    exec_fri_b  = _exec_next_bd(state_fri_b)
    sig = _combine_thu_fri_exec(exec_thu_b, exec_fri_b)
    swap = set(idx[sig.ne(sig.shift(1))])
    return swap, sig

def run_pure_baseline_segment(name, start, end):
    run_idx = build_run_index(start, end)
    if len(run_idx) == 0:
        print(f"{name} (Baseline): no run days (check dates/SMA warm-up).")
        return None, None, None

    # Save globals
    saved_swap, saved_reg = globals().get("swap_dates_all", None), globals().get("regime_intraday_all", None)

    # Build baseline dual-day signal and splice
    swap_b, sig_b = build_regime_by_SMA_baseline_dual()
    globals()["swap_dates_all"], globals()["regime_intraday_all"] = swap_b, sig_b

    try:
        res_paper = sim_strategy(run_idx, paper=True)
        res_real  = sim_strategy(run_idx, paper=False)
    finally:
        # Restore full 3-state regime
        globals()["swap_dates_all"], globals()["regime_intraday_all"] = saved_swap, saved_reg
        rebuild_regime()

    curve_paper = res_paper["equity_curve"]
    curve_real  = res_real["equity_curve"]
    m_paper = metrics_from_curve(curve_paper)
    m_real  = metrics_from_curve(curve_real)

    print(f"\n--- Yearly Returns for {name} (Baseline) ---")
    yr_paper = year_returns_from_curve(curve_paper).apply(fmt_pct)
    yr_real  = year_returns_from_curve(curve_real).apply(fmt_pct)
    df_yr = pd.DataFrame({"Paper": yr_paper, "Real": yr_real}); df_yr.index.name = "Year"
    if not df_yr.empty: print(df_yr.to_string())
    print("--------------------------------------\n")
    return (curve_paper, curve_real, (m_paper, m_real))

# =========================
# S&P 500 Buy & Hold Comparison
# =========================
def run_sp500_benchmark():
    print("\n===== S&P 500 Buy & Hold Benchmark (^GSPC) =====\n")
    try:
        holdout_idx = build_run_index(HOLDOUT[0], HOLDOUT[1])
        if not holdout_idx.empty:
            sp_ac_holdout = gspc_ac.reindex(holdout_idx)
            sp_curve_holdout = START_CAPITAL * (sp_ac_holdout / sp_ac_holdout.iloc[0])
            m_sp_holdout = metrics_from_curve(sp_curve_holdout)
            print_metrics("S&P 500 B&H (OOS/Holdout)", m_sp_holdout)
        else:
            print("S&P 500 B&H (OOS/Holdout): no data")
    except Exception as e:
        print(f"Could not run S&P 500 OOS/Holdout benchmark: {e}")
    print("================================================\n")

# =========================
# Walk-Forward Validation
# =========================
def walk_forward_validation(oos_start, oos_end, train_years, test_years, paper=True, chain_capital=True):
    oos_start = pd.to_datetime(oos_start)
    oos_end   = pd.to_datetime(oos_end)

    tests = []
    curves_scaled = []
    eq_level = START_CAPITAL

    t0 = oos_start
    while True:
        train_end = t0 + pd.DateOffset(years=train_years) - Day(1)
        test_start= train_end + Day(1)
        test_end  = test_start + pd.DateOffset(years=test_years) - Day(1)
        if test_start > oos_end:
            break
        if test_end > oos_end:
            test_end = oos_end

        run_idx = build_run_index(test_start, test_end)
        if len(run_idx) == 0:
            t0 = t0 + pd.DateOffset(years=test_years)
            continue

        res = sim_strategy(run_idx, paper=paper)
        curve_local = res["equity_curve"]

        if chain_capital:
            scale = eq_level / START_CAPITAL
            curve_scaled = curve_local * scale
            eq_level = float(curve_scaled.iloc[-1])
        else:
            curve_scaled = curve_local

        m = metrics_from_curve(curve_scaled)
        tests.append((test_start.date(), test_end.date(), m, curve_scaled))
        curves_scaled.append(curve_scaled)

        t0 = t0 + pd.DateOffset(years=test_years)

    if not curves_scaled:
        return None, []

    agg_curve = pd.concat(curves_scaled).sort_index()
    return agg_curve, tests

# =========================
# Dashboard frontend
# =========================
def build_schedule_and_legs(run_idx):
    # Build schedule via your backtest scheduler, then derive legs/returns (open->open compounding)
    sched = build_locked_schedule(run_idx)

    # Build maps as in sim_strategy for valuation
    gap_map   = {"TQQQ": gap_TQQQ, "SPXL": gap_SPXL, "QLD": gap_QLD, "SSO": gap_SSO, "GLD": gap_GLD}
    intra_map = {"TQQQ": intra_TQQQ, "SPXL": intra_SPXL, "QLD": intra_QLD, "SSO": intra_SSO, "GLD": intra_GLD}
    if globals().get("HAVE_IEF", False):
        gap_map["IEF"] = gap_IEF; intra_map["IEF"] = intra_IEF
    # extras (no proxies)
    extra_syms = sorted(lineup_syms - {"TQQQ", "SPXL", "QLD", "SSO", "GLD", "IEF"})
    for sym in extra_syms:
        try:
            ao_e, ac_e, _ = get_adj_ohlc(px, sym, idx)
            g_e, i_e = seg_returns(ao_e, ac_e)
            gap_map[sym]   = g_e.fillna(0.0)
            intra_map[sym] = i_e.fillna(0.0)
        except Exception:
            gap_map[sym]   = pd.Series(0.0, index=idx)
            intra_map[sym] = pd.Series(0.0, index=idx)

    legs = []
    if len(run_idx) == 0 or sched.empty:
        return sched, pd.DataFrame(legs)

    curr = sched.loc[run_idx[0]]
    entry = run_idx[0]
    leg_gross = 1.0 + float(intra_map[curr].get(entry, 0.0))

    for i in range(1, len(run_idx)):
        d = run_idx[i]
        nxt = sched.loc[d]
        if nxt != curr:
            # swap at open of d
            leg_gross *= (1.0 + float(gap_map[curr].get(d, 0.0)))
            legs.append({
                "Entry": entry,
                "Exit": d,
                "Asset": curr,
                "Next": nxt,
                "ReturnPct": leg_gross - 1.0,
                "Days": (d - entry).days
            })
            # new leg
            curr = nxt
            entry = d
            leg_gross = 1.0 + float(intra_map[curr].get(d, 0.0))
        else:
            # hold day
            leg_gross *= (1.0 + float(gap_map[curr].get(d, 0.0))) * (1.0 + float(intra_map[curr].get(d, 0.0)))

    return sched, pd.DataFrame(legs)

def _find_next_trading_day_with_weekday(from_date, target_weekday):
    """
    Return the next market day in idx strictly AFTER from_date whose weekday == target_weekday.
    target_weekday: Monday=0 ... Friday=4
    """
    from_date = pd.Timestamp(from_date).normalize()
    pos = idx.searchsorted(from_date, side="right")
    while pos < len(idx) and idx[pos].weekday() != target_weekday:
        pos += 1
    return idx[pos] if pos < len(idx) else None

def weekly_signal_dashboard(window_start="2019-01-01", lookback_legs=12, now_ts=None):
    # Pick "today" on the market calendar (roll back if weekend/holiday)
    now_ts = pd.Timestamp.now(tz=None) if now_ts is None else pd.Timestamp(now_ts)
    today = now_ts.normalize()
    if today not in idx:
        pos = idx.searchsorted(today, side="right") - 1
        today = idx[max(0, pos)]

    # Build schedule/legs up to today (for context and lock propagation)
    run_idx_hist = idx[(idx >= pd.to_datetime(window_start)) & (idx <= today)]
    sched_hist, legs_hist = build_schedule_and_legs(run_idx_hist)

    # Was a decision made today (Thu or Fri close)?
    made_thu = (len(state_thu) >= 2) and (state_thu.index[-1].normalize() == today)
    made_fri = (len(state_fri) >= 2) and (state_fri.index[-1].normalize() == today)

    if not (made_thu or made_fri):
        print("Decided: NO (run after-hours on Thursday or Friday).")
        curr = sched_hist.iloc[-1] if not sched_hist.empty else None
        print(f"Current holding: {curr if curr else 'n/a'}")

        # Next decision opportunities from 'today'
        pos = idx.searchsorted(today, side="right")
        next_thu = None
        next_fri = None
        while pos < len(idx) and (next_thu is None or next_fri is None):
            d = idx[pos]
            if d.weekday() == 3 and next_thu is None:  # Thursday
                next_thu = d
            if d.weekday() == 4 and next_fri is None:  # Friday
                next_fri = d
            pos += 1

        if next_thu is not None:
            print(f"Next Thu decision: {next_thu.date()}  -> Exec: {next_trading_day(next_thu).date()} (open) | Symbol: TBD")
        if next_fri is not None:
            print(f"Next Fri decision: {next_fri.date()}  -> Exec: {next_trading_day(next_fri).date()} (open) | Symbol: TBD")

        # Recent completed legs
        if legs_hist is not None and not legs_hist.empty:
            legs_tail = legs_hist.tail(lookback_legs).copy()
            legs_tail["Return%"] = (legs_tail["ReturnPct"] * 100.0).round(2)
            cols = ["Entry", "Exit", "Asset", "Next", "Days", "Return%"]
            print("\nRecent completed legs (open->open):")
            print(legs_tail[cols].to_string(index=False))
        else:
            print("\nNo completed legs in window.")
        print(f"Has today's bar? {idx[-1].normalize() == pd.Timestamp.now().normalize()}")
        return

    # Decision made today
    anchor = "Thursday" if made_thu else "Friday"
    exec_day = next_trading_day(today)

    # Build schedule through exec_day to propagate lock state and read the chosen symbol
    if exec_day is None:
        print("===== Decision/Execution Plan =====")
        print(f"Decided: YES ({anchor} close {today.date()})")
        print("Entry:   n/a (no next trading day found in data window)")
        print("Decided Exit (earliest): n/a")
        return
    run_idx_until_exec = idx[idx <= exec_day]
    sched_upto, _ = build_schedule_and_legs(run_idx_until_exec)
    suggested = sched_upto.loc[exec_day] if exec_day in sched_upto.index else None

    print("===== Decision/Execution Plan =====")
    print(f"Decided: YES ({anchor} close {today.date()})")
    print(f"Entry:   {exec_day.date()} (open) | Suggested symbol: {suggested if suggested else 'TBD'}")
    # Earliest next decision/exec window
    next_decision = _find_next_trading_day_with_weekday(today, 4) if made_thu else _find_next_trading_day_with_weekday(today, 3)
    next_exec = next_trading_day(next_decision) if next_decision is not None else None
    if next_decision is not None and next_exec is not None:
        print(f"Decided Exit (earliest): decision {next_decision.date()} (close) -> execution {next_exec.date()} (open)")
    else:
        print("Decided Exit (earliest): n/a")

    # Weekly summary and recent legs
    fridays_hist = pd.DatetimeIndex([d for d in run_idx_hist if d.weekday() == 4])
    if not fridays_hist.empty:
        applied = sched_hist.reindex(fridays_hist).dropna()
        if not applied.empty:
            print(f"\nWeek summary (from Friday open {applied.index[-1].date()}): Target was {applied.iloc[-1]}")

    if legs_hist is not None and not legs_hist.empty:
        legs_tail = legs_hist.tail(12).copy()
        legs_tail["Return%"] = (legs_tail["ReturnPct"] * 100.0).round(2)
        cols = ["Entry", "Exit", "Asset", "Next", "Days", "Return%"]
        print("\nRecent completed legs (open->open):")
        print(legs_tail[cols].to_string(index=False))
    else:
        print("\nNo completed legs in window.")

    print(f"\nHas today's bar? {idx[-1].normalize() == pd.Timestamp.now().normalize()}")
    print("\nRepainting: NO (decisions fixed at Thu/Fri close; execution next trading day).")
    print("Lagging: 1 business day by design (Thu→Fri and Fri→Mon).")
    print("==============================================\n")

# =========================
# EXECUTION LOGIC
# =========================
if not DO_WALK_FORWARD:
    # 1) Pure baseline on HOLDOUT (dual-day)
    print("\n===== Baseline Strategy (2-State, dual-day: Thu->Fri and Fri->Mon) =====")
    bp, br, bm = run_pure_baseline_segment("Holdout (Baseline)", HOLDOUT[0], HOLDOUT[1])
    if bm:
        print_metrics("Baseline Holdout (Paper)", bm[0])
        print_metrics("Baseline Holdout (Real)",  bm[1])

    # 2) S&P 500 B&H for reference
    run_sp500_benchmark()

    # Flip weekday sanity check: should be mostly Friday and Monday
    rebuild_regime()
    run_idx_chk = build_run_index("2018-01-01", None)
    sig_chk = regime_intraday_all.reindex(run_idx_chk)
    flips = sig_chk[sig_chk.ne(sig_chk.shift(1))].index
    print("\nFlip weekday counts (expect mostly Friday and Monday):")
    print(pd.Series(flips.day_name()).value_counts())

    # 3) New 3-state strategy on same window (dual-day)
    print(f"\n===== NEW STRATEGY (3-state dual-day on {SIGNAL_SYMBOL}, SMA {SMA_SLOW}/{SMA_MID}) =====")
    print(f"Up-Window: {MID_UP_WINDOW_WEEKS_BENIGN}w (benign) / {MID_UP_WINDOW_WEEKS_STRESSED}w (stressed)")
    print(f"Slope Filter: {USE_SLOPE_FILTER} ({SLOPE_LOOKBACK_W}w), Dual Risk-Off: {USE_DUAL_RISK_OFF} ({RISK_OFF_LOOKBACK_D}d)")
    print(f"Dist. Gate: {USE_DIST_GATE_200} ({DELTA_200_PCT*100:.1f}%), Adaptive: {USE_ADAPTIVE_DIST}, Vol Cap: {USE_VOL_CAP}, DD Throttle: {USE_DD_THROTTLE}")
    print(f"Wash Sale Lockout (Nasdaq family): {LOCKOUT_DAYS} days\n")

    cp_hold, cr_hold, m_hold = run_segment("Holdout", HOLDOUT[0], HOLDOUT[1])
    if m_hold:
        print_metrics("Holdout (Paper)", m_hold[0])
        print_metrics("Holdout (Paper)", m_hold[0])
        print_metrics("Holdout (Real)",  m_hold[1])

    # Overlay diagnostics summary
    if overlay_events:
        ev = pd.DataFrame(overlay_events)
        total = len(ev)
        ub3_n = int((ev["mode"] == "ub3_fade").sum())
        lb3_n = int((ev["mode"] == "lb3_rerisk").sum())
        avg_days = ev["days"].mean() if "days" in ev else float("nan")
        print("\n===== LR Overlay Summary (Holdout) =====")
        print(f"Total Overlays: {total}  |  UB3→UB2 fades: {ub3_n}  |  LB3→Mid re-risk: {lb3_n}")
        if not np.isnan(avg_days):
            print(f"Average Overlay Duration: {avg_days:.1f} days")
        exit_counts = ev["exit_reason"].value_counts(dropna=False)
        if not exit_counts.empty:
            print("Exit reasons:")
            for k, v in exit_counts.items():
                print(f"  {k}: {v}")
        
        # Show last 3 UB3 fade trades for verification
        ub3_trades = ev[ev["mode"] == "ub3_fade"].tail(3)
        if not ub3_trades.empty:
            print("\nLast 3 UB3→UB2 trades (for ThinkorSwim verification):")
            for _, t in ub3_trades.iterrows():
                signal_date = t["start"] - pd.Timedelta(days=1)  # Entry is next day, so signal was prev
                print(f"\n  Trade: Signal {signal_date.date()} → Entry {t['start'].date()} → Exit {t['end'].date()} ({t['days']} days)")
                
                # Show QQQ/^NDX channel values on signal date
                if signal_date in std_ndx.index:
                    row = std_ndx.loc[signal_date]
                    nasdaq_ticker = "QQQ" if "QQQ" in px["Close"].columns else "^NDX"
                    print(f"    {nasdaq_ticker} on {signal_date.date()}:")
                    print(f"      Close: {row['close']:.2f}")
                    print(f"      Mid:   {row['mid']:.2f}")
                    print(f"      UB1:   {row['UB1']:.2f}")
                    print(f"      UB2:   {row['UB2']:.2f}")
                    print(f"      UB3:   {row['UB3']:.2f}  ← Entry trigger")
                    print(f"      LB1:   {row['LB1']:.2f}")
                    print(f"      LB2:   {row['LB2']:.2f}")
                    print(f"      LB3:   {row['LB3']:.2f}")
                    print(f"      Close >= UB3? {row['close']:.2f} >= {row['UB3']:.2f} = {row['close'] >= row['UB3']}")
                
                # Show SPY filter on signal date
                if signal_date in std_spy.index:
                    spy_row = std_spy.loc[signal_date]
                    print(f"    SPY Filter on {signal_date.date()}:")
                    print(f"      Close: {spy_row['close']:.2f}  |  Mid: {spy_row['mid']:.2f}  |  UB2: {spy_row['UB2']:.2f}")
                    spy_ok = (spy_row['mid'] <= spy_row['close'] <= spy_row['UB2'])
                    print(f"      Filter OK? {spy_row['mid']:.2f} <= {spy_row['close']:.2f} <= {spy_row['UB2']:.2f} = {spy_ok}")
                
                # Entry/Exit open prices and PnL (short QQQ proxy)
                try:
                    base_ticker = "QQQ" if "QQQ" in px["Open"].columns else ("^NDX" if "^NDX" in px["Open"].columns else None)
                    if base_ticker is not None:
                        ent_o = float(px["Open"][base_ticker].get(t['start'], float('nan')))
                        ex_o  = float(px["Open"][base_ticker].get(t['end'], float('nan')))
                        if not (pd.isna(ent_o) or pd.isna(ex_o)):
                            short_ret = (ent_o / ex_o - 1.0) * 100.0
                            print(f"    Entry Open ({base_ticker}): {t['start'].date()}  {ent_o:.2f}")
                            print(f"    Exit  Open ({base_ticker}): {t['end'].date()}    {ex_o:.2f}")
                            print(f"    Short Return: {short_ret:.2f}%")
                except Exception:
                    pass
        print("========================================\n")

        # Full trade list (entry/exit open and PnL)
        try:
            ub3_all = ev[ev["mode"] == "ub3_fade"].copy()
            if not ub3_all.empty:
                base_ticker = "QQQ" if "QQQ" in px["Open"].columns else ("^NDX" if "^NDX" in px["Open"].columns else None)
                print("All UB3→UB2 overlay trades (entry/exit open, short PnL):")
                print(f"Ticker used for PnL: {base_ticker}")
                print("  Signal       Entry        Exit        Days   EntryOpen   ExitOpen   ShortPnL%")
                for _, t in ub3_all.iterrows():
                    signal_date = (t["start"] - pd.Timedelta(days=1)).date()
                    ent = t["start"]; ex = t["end"]; days = int(t.get("days", 0))
                    ent_o = float(px["Open"][base_ticker].get(ent, float('nan'))) if base_ticker else float('nan')
                    ex_o  = float(px["Open"][base_ticker].get(ex, float('nan'))) if base_ticker else float('nan')
                    pnl = (ent_o / ex_o - 1.0) * 100.0 if (not pd.isna(ent_o) and not pd.isna(ex_o) and ex_o != 0.0) else float('nan')
                    print(f"  {signal_date}  {ent.date()}  {ex.date()}  {days:5d}   {ent_o:10.2f}  {ex_o:9.2f}   {pnl:9.2f}")
                print()
        except Exception:
            pass

    # 4) Dashboard-style summary (intended for Thu/Fri after-hours)
    weekly_signal_dashboard(window_start="2019-01-01", lookback_legs=20)

else:
    wf_curve, wf_tests = walk_forward_validation(
        WF_START_DATE, WF_END_DATE,
        WF_TRAIN_YEARS, WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL
    )
    if wf_curve is not None and not wf_curve.empty:
        m_wf = metrics_from_curve(wf_curve)
        simMode = "Paper (No Taxes/Fees)" if WF_USE_PAPER else "Real (With Taxes/Fees)"
        print(f"===== Walk-Forward Validation ({simMode}, chained, {WF_START_DATE} to {WF_END_DATE}) =====")
        for (ts, te, m, _) in wf_tests:
            print_metrics(f"Test window {ts} → {te}", m)
        print_metrics("Aggregated WF tests (chained)", m_wf)
    else:
        print(f"===== Walk-Forward Validation ({WF_START_DATE} to {WF_END_DATE}) =====")
        print("No data available for this period or configuration.")
        print(f"Check SMA warm-up and if WF_TRAIN_YEARS ({WF_TRAIN_YEARS}) is too long for the period.")