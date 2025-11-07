# dual_day_strategy.py
# Weekly 3-state engine with dual execution days:
# - Decide Thursday close -> execute next trading day open (usually Friday)
# - Decide Friday close   -> execute next trading day open (usually Monday)
# Keeps original gates, taxes, scheduler, and Nasdaq lockout.
# Adds: dashboard-style frontend (Decided/Entry, Next Thu/Fri decision windows, recent legs).
#
# Sprint 1 (Volatility-Targeted Sizing): Implemented downside-only VT
# - Target vol: 35% annualized, 20-day lookback
# - Weekly recalculation at Thu/Fri anchors
# - Only scales down when vol exceeds target (never scales above 100%)
# - Baseline: Provides strong MaxDD protection with minimal CAGR give-up

import pandas as pd, numpy as np, yfinance as yf
from pandas.tseries.offsets import BDay, Day
from collections import defaultdict
import math

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# =========================
# Config
# =========================
DATA_START      = "1950-01-01"
DATA_END        = None  # None => today

DO_WALK_FORWARD = True

WF_START_DATE     = "1963-01-01"
WF_END_DATE       = "2017-12-31"
WF_TRAIN_YEARS    = 2
WF_TEST_YEARS     = 1
WF_USE_PAPER      = True
WF_CHAIN_CAPITAL  = True

IN_SAMPLE         = ("2000-01-01", "2000-01-01")
OOS_MAIN          = ("1999-01-01", None)
HOLDOUT           = ("2000-01-01", None)  # Test SPXL 2000-2025

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

# Tripwire demotion (Sprint 2)
USE_TRIPWIRE      = False  # Enable/disable tripwire demotion
TRIPWIRE_COOLDOWN_DAYS = 5  # Business days to stay in state 0 after tripwire triggers

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

# Prefer SPXL over TQQQ for 3x (useful for testing 1970-1989 with ^GSPC proxy)
PREFER_SPXL = False

# ============ Dashboard lineups (availability fallback only) ============
# Only used if the original choice has no data that day. Original order/lockout preserved.
RISK_ON_3X_LINEUP    = ["TQQQ", "SPXL", "UPRO", "UDOW", "TECL", "SOXL", "TNA", "FNGU"]
RISK_ON_2X_LINEUP    = ["QLD",  "SSO",  "SPUU", "DDM",  "UWM",  "ROM"]
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
print("Starting data download...", flush=True)
if DATA_END is None:
    DATA_END = (pd.Timestamp.today() + BDay(1)).date().isoformat()

# Tripwire symbols: VIX, VIX3M (fallback to ^VXV), HYG, LQD, RSP, SPY
tripwire_syms = ["^VIX", "VIX3M", "^VXV", "HYG", "LQD", "RSP", "SPY"]
tickers = sorted(set(
    [SIGNAL_SYMBOL, "^GSPC", "^NDX", "SPY", "GC=F"] + list(lineup_syms) + tripwire_syms
))
dl_lookback = max(SMA_SLOW, SMA_MID, DD_LOOKBACK_D, RISK_OFF_LOOKBACK_D) + 10
dl_start = (pd.to_datetime(DATA_START) - BDay(dl_lookback)).date().isoformat()
print(f"Downloading {len(tickers)} tickers from {dl_start} to {DATA_END}...", flush=True)
px = yf.download(tickers, start=dl_start, end=DATA_END, auto_adjust=False, progress=False)
print("Data download complete. Building asset returns...", flush=True)

# Reference series for SMA/logic
if SIGNAL_SYMBOL not in px["Adj Close"].columns or px["Adj Close"][SIGNAL_SYMBOL].isnull().all():
    print(f"Warning: SIGNAL_SYMBOL {SIGNAL_SYMBOL} not found or all NaN. Defaulting to ^GSPC.")
    SIGNAL_SYMBOL = "^GSPC"

ref_ac = px["Adj Close"][SIGNAL_SYMBOL].dropna()
idx = ref_ac.index

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
gspc_ao, gspc_ac, gspc_cl  = get_adj_ohlc(px, "^GSPC", idx)  # For SPXL/SSO proxy base
spxl_ao, spxl_ac, spxl_cl  = get_adj_ohlc(px, "SPXL", idx)
qld_ao,  qld_ac,  qld_cl   = get_adj_ohlc(px, "QLD",  idx)
sso_ao,  sso_ac,  sso_cl   = get_adj_ohlc(px, "SSO",  idx)
ief_ao,  ief_ac,  ief_cl   = get_adj_ohlc(px, "IEF",  idx)

# Tripwire data
vix_ac = px["Adj Close"]["^VIX"].reindex(idx) if "^VIX" in px["Adj Close"].columns else pd.Series(index=idx, dtype=float)
# Try VIX3M first, fallback to ^VXV (both are 3M implied vol)
vix3m_ac = px["Adj Close"]["VIX3M"].reindex(idx) if "VIX3M" in px["Adj Close"].columns else pd.Series(index=idx, dtype=float)
vxv_ac = px["Adj Close"]["^VXV"].reindex(idx) if "^VXV" in px["Adj Close"].columns else pd.Series(index=idx, dtype=float)
# Use VIX3M if available, otherwise fallback to ^VXV
vix3m_fallback_ac = vix3m_ac if vix3m_ac.notna().sum() > vxv_ac.notna().sum() else vxv_ac
hyg_ac = px["Adj Close"]["HYG"].reindex(idx) if "HYG" in px["Adj Close"].columns else pd.Series(index=idx, dtype=float)
lqd_ac = px["Adj Close"]["LQD"].reindex(idx) if "LQD" in px["Adj Close"].columns else pd.Series(index=idx, dtype=float)
rsp_ac = px["Adj Close"]["RSP"].reindex(idx) if "RSP" in px["Adj Close"].columns else pd.Series(index=idx, dtype=float)

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

def _combine_base_series(primary_ao, primary_ac, fallback_ao, fallback_ac):
    ao = primary_ao.copy()
    ac = primary_ac.copy()
    if ao is None or ac is None:
        return fallback_ao, fallback_ac
    ao = ao.where(~ao.isna(), fallback_ao)
    ac = ac.where(~ac.isna(), fallback_ac)
    return ao, ac

def build_asset_returns_TQQQ():
    ndx_ao_c, ndx_ac_c = _combine_base_series(ndx_ao, ndx_ac, gspc_ao, gspc_ac)
    base_gap, base_intra = seg_returns(ndx_ao_c, ndx_ac_c)
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
    # Use ^GSPC as base (for 1970+), fallback to SPY if ^GSPC unavailable
    if gspc_ac.notna().sum() > spy_ac.notna().sum():
        base_gap, base_intra = seg_returns(gspc_ao, gspc_ac)
    else:
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
    ndx_ao_c, ndx_ac_c = _combine_base_series(ndx_ao, ndx_ac, gspc_ao, gspc_ac)
    base_gap, base_intra = seg_returns(ndx_ao_c, ndx_ac_c)
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
    # Use ^GSPC as base (for 1970+), fallback to SPY if ^GSPC unavailable
    if gspc_ac.notna().sum() > spy_ac.notna().sum():
        base_gap, base_intra = seg_returns(gspc_ao, gspc_ac)
    else:
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

print("Building GLD returns...", flush=True)
gap_GLD,  intra_GLD,  prox_mask_GLD  = build_asset_returns_GLD()
print("Building TQQQ returns...", flush=True)
gap_TQQQ, intra_TQQQ, prox_mask_TQQQ = build_asset_returns_TQQQ()
print("Building SPXL returns...", flush=True)
gap_SPXL, intra_SPXL, prox_mask_SPXL = build_asset_returns_SPXL()
print("Building QLD returns...", flush=True)
gap_QLD,  intra_QLD,  prox_mask_QLD  = build_asset_returns_QLD()
print("Building SSO returns...", flush=True)
gap_SSO,  intra_SSO,  prox_mask_SSO  = build_asset_returns_SSO()
print("Asset returns complete. Building regime...", flush=True)

prox_mask_GLD  = prox_mask_GLD.reindex(idx).fillna(False).astype(bool)
prox_mask_TQQQ = prox_mask_TQQQ.reindex(idx).fillna(False).astype(bool)
prox_mask_SPXL = prox_mask_SPXL.reindex(idx).fillna(False).astype(bool)
prox_mask_QLD  = prox_mask_QLD.reindex(idx).fillna(False).astype(bool)
prox_mask_SSO  = prox_mask_SSO.reindex(idx).fillna(False).astype(bool)

raw_close  = {"TQQQ": tqqq_cl, "SPXL": spxl_cl, "GLD": gld_cl, "QLD": qld_cl, "SSO": sso_cl, "IEF": ief_cl}
div_series = {"TQQQ": div_tqqq, "SPXL": div_spxl, "GLD": div_gld, "QLD": div_qld, "SSO": div_sso, "IEF": div_ief}

# Add sector rotation tickers to raw_close (for scoring)
for sym in RISK_ON_3X_LINEUP + RISK_ON_2X_LINEUP:
    if sym not in raw_close:
        try:
            _, ac_sym, _ = get_adj_ohlc(px, sym, idx)
            raw_close[sym] = ac_sym
            # Create empty div series for new tickers
            div_series[sym] = pd.Series(0.0, index=idx)
        except Exception:
            raw_close[sym] = pd.Series(index=idx, dtype=float)
            div_series[sym] = pd.Series(0.0, index=idx)

# =========================
# Tripwire demotion gate (Sprint 2) - Fast crash detection
# =========================
USE_TRIPWIRE = False  # Enable/disable tripwire demotion
TRIPWIRE_COOLDOWN_DAYS = 5  # Business days to stay in state 0 after tripwire triggers

# Sector rotation (Sprint 2)
USE_SECTOR_ROTATION = False  # Enable/disable sector rotation in risk-on states
TURBO_TICKERS = {"TECL", "SOXL", "FNGU", "TNA"}  # Turbo tickers (only when tripwire clear and VIX healthy)
TURBO_VIX_THRESHOLD = 0.90  # VIX/VIX3M < this to allow turbo tickers

def compute_tripwire_flags(d):
    """Compute tripwire flags at date d. Returns (vix_bad, credit_bad, breadth_bad)"""
    if not USE_TRIPWIRE:
        return False, False, False
    
    flags = {"vix": False, "credit": False, "breadth": False}
    
    # VIX Flag: VIX/VIX3M > 1.00 (backwardation)
    # Use VIX3M if available, fallback to ^VXV
    if d in vix_ac.index and d in vix3m_fallback_ac.index:
        vix_val = vix_ac.get(d, np.nan)
        vix3m_val = vix3m_fallback_ac.get(d, np.nan)
        if pd.notna(vix_val) and pd.notna(vix3m_val) and vix3m_val > 0:
            flags["vix"] = (vix_val / vix3m_val) > 1.00
    
    # Credit Flag: HYG/LQD below 100D SMA AND 20D return < 0
    # Use HYG if available, fallback to LQD
    credit_ac = hyg_ac if hyg_ac.notna().sum() > lqd_ac.notna().sum() else lqd_ac
    if d in credit_ac.index and credit_ac.notna().sum() > 100:
        sma100 = credit_ac.rolling(100, min_periods=100).mean()
        ret20 = credit_ac.pct_change(20)
        sma_val = sma100.get(d, np.nan)
        ret_val = ret20.get(d, np.nan)
        price_val = credit_ac.get(d, np.nan)
        if pd.notna(sma_val) and pd.notna(ret_val) and pd.notna(price_val):
            flags["credit"] = (price_val < sma_val) and (ret_val < 0)
    
    # Breadth Flag: RSP/SPY below 100D SMA AND 20D return < 0
    # Use RSP if available, fallback to SPY
    breadth_ac = rsp_ac if rsp_ac.notna().sum() > spy_ac.notna().sum() else spy_ac
    if d in breadth_ac.index and breadth_ac.notna().sum() > 100:
        sma100 = breadth_ac.rolling(100, min_periods=100).mean()
        ret20 = breadth_ac.pct_change(20)
        sma_val = sma100.get(d, np.nan)
        ret_val = ret20.get(d, np.nan)
        price_val = breadth_ac.get(d, np.nan)
        if pd.notna(sma_val) and pd.notna(ret_val) and pd.notna(price_val):
            flags["breadth"] = (price_val < sma_val) and (ret_val < 0)
    
    return flags["vix"], flags["credit"], flags["breadth"]

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

    # Tripwire demotion (Sprint 2): if 2+ flags bad at anchor and state >= 1, force state 0
    if USE_TRIPWIRE:
        tripwire_cooldown_until = pd.Timestamp("1900-01-01")
        for i, d in enumerate(state_w.index):
            # Check if we're still in cooldown (compare weekly anchor dates)
            if tripwire_cooldown_until > pd.Timestamp("1900-01-01"):
                # Find the next weekly anchor date after cooldown ends
                next_anchors_after_cooldown = state_w.index[state_w.index > tripwire_cooldown_until]
                if len(next_anchors_after_cooldown) > 0 and d < next_anchors_after_cooldown[0]:
                    # Still in cooldown: force state 0
                    state_w.iloc[i] = 0
                    continue
                else:
                    # Cooldown expired
                    tripwire_cooldown_until = pd.Timestamp("1900-01-01")
            
            if state_w.iloc[i] >= 1:
                vix_bad, credit_bad, breadth_bad = compute_tripwire_flags(d)
                bad_count = sum([vix_bad, credit_bad, breadth_bad])
                
                if bad_count >= 2:
                    # Tripwire triggered: force state 0 and set cooldown
                    state_w.iloc[i] = 0
                    # Set cooldown: find the date TRIPWIRE_COOLDOWN_DAYS business days after anchor d
                    pos = idx.searchsorted(d, side="right")
                    cooldown_end_pos = min(len(idx) - 1, pos + TRIPWIRE_COOLDOWN_DAYS)
                    if cooldown_end_pos < len(idx):
                        tripwire_cooldown_until = idx[cooldown_end_pos]

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
print("Computing SMA and building regime...", flush=True)
sma_slow = ref_ac.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
rebuild_regime()
print("Regime built. Starting simulation...", flush=True)

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
        if sym == "CASH":
            return True  # CASH always available (uses rf_daily)
        if sym in {"TQQQ","SPXL","QLD","SSO","GLD"}:
            base_ac = {"TQQQ": tqqq_ac, "SPXL": spxl_ac, "QLD": qld_ac, "SSO": sso_ac, "GLD": gld_ac}[sym]
            return pd.notna(base_ac.get(d, np.nan))
        if sym == "IEF":
            return globals().get("HAVE_IEF", False) and pd.notna(ief_ac.get(d, np.nan))
        return pd.notna(ac_extra.get(sym, pd.Series(index=idx, dtype=float)).get(d, np.nan))

    def choose_risk_off(d):
        # Cash fallback: if both GLD and IEF have negative 63D momentum, use CASH
        if USE_DUAL_RISK_OFF and HAVE_IEF:
            mom_gld_val = float(mom_gld_use.get(d, -1.0))
            mom_ief_val = float(mom_ief_use.get(d, -1.0))
            if mom_gld_val < 0 and mom_ief_val < 0:
                return "CASH"  # Both negative: use cash (risk-free rate)
            return "GLD" if mom_gld_val >= mom_ief_val else "IEF"
        return "GLD"

    # Sector rotation chooser (Sprint 2)
    def choose_asset_sector_rotation(regime, d, lock_until):
        """Select asset using momentum/volatility scoring for sector rotation"""
        if regime == 0:
            return choose_risk_off(d)
        
        # Determine universe based on regime
        if regime == 2:  # 3x
            universe = RISK_ON_3X_LINEUP.copy()
        else:  # regime == 1, 2x
            universe = RISK_ON_2X_LINEUP.copy()
        
        # Filter out locked tickers (Nasdaq lockout)
        if d <= lock_until:
            universe = [s for s in universe if s not in {"TQQQ", "QLD"}]
        
        # Check tripwire status and VIX term structure for turbo tickers
        vix_bad, credit_bad, breadth_bad = compute_tripwire_flags(d)
        tripwire_active = sum([vix_bad, credit_bad, breadth_bad]) >= 2
        
        # Check VIX term structure
        vix_healthy = False
        if d in vix_ac.index and d in vix3m_fallback_ac.index:
            vix_val = vix_ac.get(d, np.nan)
            vix3m_val = vix3m_fallback_ac.get(d, np.nan)
            if pd.notna(vix_val) and pd.notna(vix3m_val) and vix3m_val > 0:
                vix_healthy = (vix_val / vix3m_val) < TURBO_VIX_THRESHOLD
        
        # Filter turbo tickers if tripwire active or VIX unhealthy
        if tripwire_active or not vix_healthy:
            universe = [s for s in universe if s not in TURBO_TICKERS]
        
        # Score each candidate
        scores = {}
        for sym in universe:
            # Check if we have enough data (≥126 trading days)
            if sym not in raw_close:
                continue
            
            sym_close = raw_close[sym]
            if sym_close.isnull().all():
                continue
            
            # Get returns up to date d (use close prices, compute daily returns)
            sym_close_to_d = sym_close[sym_close.index <= d].dropna()
            if len(sym_close_to_d) < 126:
                continue
            
            # Compute daily returns from close prices
            sym_ret_to_d = sym_close_to_d.pct_change().dropna()
            if len(sym_ret_to_d) < 63:
                continue
            
            # Momentum: 63-day total return
            mom_63 = float((1 + sym_ret_to_d.tail(63)).prod() - 1.0)
            
            # Volatility: 20-day realized vol (annualized)
            if len(sym_ret_to_d) >= 20:
                vol_20 = float(sym_ret_to_d.tail(20).std() * math.sqrt(252))
            else:
                continue
            
            # Score: momentum / volatility
            if vol_20 > 0:
                scores[sym] = mom_63 / vol_20
            else:
                scores[sym] = -999.0  # Very bad score if no volatility
        
        if not scores:
            # Fallback to original logic if no scores available
            if regime == 2:
                return "SPXL" if d <= lock_until else "TQQQ"
            else:
                return "SSO" if d <= lock_until else "QLD"
        
        # Return top-scoring asset
        return max(scores.items(), key=lambda x: x[1])[0]
    
    # Original selection (with PREFER_SPXL option)
    def choose_asset(regime, d, lock_until):
        # Use sector rotation if enabled
        if USE_SECTOR_ROTATION and regime >= 1:
            return choose_asset_sector_rotation(regime, d, lock_until)
        
        # Original logic
        if regime == 2:   # 3x
            if PREFER_SPXL:
                return "SPXL"  # Always SPXL when preferred (uses ^GSPC proxy pre-1993)
            else:
                return "SPXL" if d <= lock_until else "TQQQ"
        elif regime == 1: # 2x
            if PREFER_SPXL:
                return "SSO"   # Match 2x to SPX family
            else:
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

    for i, d in enumerate(run_idx):
        regime = int(regime_intraday.get(d, 0))

        if i == 0:
            base = choose_asset(regime, d, lockout_until)
            curr_asset = pick_available_from_lineup(base, regime, d, lockout_until)
            entry_date = d
            if curr_asset == "CASH":
                leg_gross = 1.0 + float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
            else:
                leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            sched.iloc[i] = curr_asset
            continue

        if d in swap_set:
            if curr_asset != "CASH":
                leg_gross *= (1.0 + gap_map[curr_asset].get(d, 0.0))
            if (curr_asset in {"TQQQ", "QLD"}) and (leg_gross - 1.0 < 0.0):
                lockout_until = d + Day(LOCKOUT_DAYS)

            base = choose_asset(regime, d, lockout_until)
            next_asset = pick_available_from_lineup(base, regime, d, lockout_until)

            curr_asset = next_asset
            entry_date = d
            if curr_asset == "CASH":
                leg_gross = 1.0 + float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
            else:
                leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            sched.iloc[i] = curr_asset
        else:
            if curr_asset == "CASH":
                leg_gross *= (1.0 + float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0)))
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

# =========================
# Volatility-targeted sizing (Sprint 1) - Helper functions
# =========================
USE_VOL_TARGETING   = True   # Enable/disable volatility-targeted sizing
VT_MODE             = "downside_only"  # "full", "downside_only", or "regime_dependent"
TARGET_VOL_ANNUAL   = 0.35   # sigma* (0.35 for growth, 0.30 for more protection)
VT_LOOKBACK_D       = 20     # realized vol lookback (days)
ALPHA_MIN           = 0.40   # Only used in "full" mode
ALPHA_MAX           = 1.0    # Cap at 1.0 for downside-only (never scale up)
ALPHA_UPDATE_THRESH = 0.10   # only update if change exceeds this

# Regime-dependent VT parameters (used when VT_MODE == "regime_dependent")
# Note: Tested but not used in baseline. Kept for future experimentation.
VT_TARGET_BY_REGIME = {
    2: 0.40,   # State 2 (3x allowed): higher target for growth
    1: 0.25,   # State 1 (2x allowed): lower target
    0: None,   # State 0 (risk-off): no VT
}
VT_ALPHA_MIN_BY_REGIME = {
    2: 0.50,   # State 2: higher floor
    1: 0.40,   # State 1: lower floor
    0: 0.20,   # State 0: very low floor (if VT enabled)
}

# =========================
# Drawdown-only throttle (Step 3) - Portfolio-level risk control
# Note: Tested but not used in baseline. Kept for future experimentation.
# =========================
USE_DD_THROTTLE_VT   = False  # Enable/disable drawdown throttle (portfolio-level)
DD_THROTTLE_THRESH1  = 0.10   # First threshold (10%)
DD_THROTTLE_THRESH2  = 0.20   # Second threshold (20%)
DD_THROTTLE_MULT1    = 1.00   # Multiplier when DD <= thresh1
DD_THROTTLE_MULT2    = 0.80   # Multiplier when thresh1 < DD <= thresh2
DD_THROTTLE_MULT3    = 0.60   # Multiplier when DD > thresh2
DD_THROTTLE_COOLDOWN = True   # Only increase multiplier after recovering half DD

def _build_daily_total_return_series(gap_s: pd.Series, intra_s: pd.Series) -> pd.Series:
    g = gap_s.reindex(idx).fillna(0.0)
    i = intra_s.reindex(idx).fillna(0.0)
    return (1.0 + g) * (1.0 + i) - 1.0

def _build_weekly_alpha_for_asset(asset: str, gap_map: dict, intra_map: dict, anchors=("W-THU","W-FRI")) -> pd.Series:
    if asset not in gap_map or asset not in intra_map:
        return pd.Series(1.0, index=idx)
    r = _build_daily_total_return_series(gap_map[asset], intra_map[asset])
    vol20 = r.rolling(VT_LOOKBACK_D, min_periods=max(10, VT_LOOKBACK_D//2)).std().reindex(idx)
    ann_vol = vol20 * math.sqrt(252)
    anchor_dates = []
    for a in anchors:
        snap = ref_ac.resample(a).last().dropna().index
        anchor_dates.extend(list(snap))
    anchor_dates = sorted([d for d in anchor_dates if d in idx])
    alpha_w = pd.Series(index=anchor_dates, dtype=float)
    prev_alpha = None
    
    # Get regime at anchor dates
    # At anchor date (Thu/Fri close), we need the regime that's currently active
    # regime_intraday_all is forward-filled, so at anchor date it shows current regime
    # But we need to look at what regime will be active starting next trading day
    # Actually, for VT we want the regime we're transitioning INTO at the anchor decision
    # So we look at the regime at the next trading day (execution day)
    regime_at_anchors = pd.Series(index=anchor_dates, dtype=int)
    for d in anchor_dates:
        exec_day = next_trading_day(d)
        if exec_day is not None and exec_day in regime_intraday_all.index:
            regime_at_anchors.loc[d] = int(regime_intraday_all.loc[exec_day])
        else:
            # Fallback: use regime at anchor date itself
            regime_at_anchors.loc[d] = int(regime_intraday_all.get(d, 0))
    
    for d in anchor_dates:
        sig = float(ann_vol.get(d, float("nan")))
        regime = int(regime_at_anchors.get(d, 0))
        
        if VT_MODE == "regime_dependent":
            # Regime-dependent: use different target vol and alpha_min based on regime
            target_vol = VT_TARGET_BY_REGIME.get(regime, None)
            if target_vol is None:
                # State 0 (risk-off): no VT, keep alpha = 1.0
                clamped = 1.0
            else:
                alpha_min_regime = VT_ALPHA_MIN_BY_REGIME.get(regime, 0.40)
                # Downside-only: α = min(1.0, max(α_min, σ*/σ̂))
                raw = target_vol / max(sig, 1e-6) if (sig == sig) else 1.0
                clamped = min(1.0, max(alpha_min_regime, raw))
        elif VT_MODE == "downside_only":
            # Downside-only: α = min(1.0, σ*/σ̂) - only scale down, never up
            raw = TARGET_VOL_ANNUAL / max(sig, 1e-6) if (sig == sig) else 1.0
            clamped = min(1.0, raw)  # Cap at 1.0, no minimum floor
        else:
            # Full mode: allow scaling both up and down
            raw = TARGET_VOL_ANNUAL / max(sig, 1e-6) if (sig == sig) else 1.0
            clamped = min(ALPHA_MAX, max(ALPHA_MIN, raw))
        
        if prev_alpha is None:
            # First anchor: always set
            alpha_w.loc[d] = clamped
            prev_alpha = clamped
        elif abs(clamped - prev_alpha) > ALPHA_UPDATE_THRESH:
            # Change exceeds threshold: update
            alpha_w.loc[d] = clamped
            prev_alpha = clamped
        else:
            # Change below threshold: keep previous
            alpha_w.loc[d] = prev_alpha
    # Initialize with NaN, then set values at anchor execution dates and forward-fill
    alpha_d = pd.Series(np.nan, index=idx)
    if not alpha_w.empty:
        for d in alpha_w.index:
            nxt = next_trading_day(d)
            if nxt is not None and nxt in alpha_d.index:
                alpha_d.loc[nxt] = float(alpha_w.loc[d])
        # Forward-fill from anchor dates, fill any remaining NaN (before first anchor) with 1.0
        alpha_d = alpha_d.ffill().fillna(1.0)
    else:
        # No anchor dates: default to 1.0
        alpha_d = alpha_d.fillna(1.0)
    return alpha_d

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

    # Precompute per-asset alpha series (vol-targeted sizing) using global function
    # Cache alpha series globally since it's computed on full idx and doesn't depend on run_idx
    alpha_by_asset = {}
    if USE_VOL_TARGETING:
        if not hasattr(_build_weekly_alpha_for_asset, '_cache'):
            _build_weekly_alpha_for_asset._cache = {}
        assets_for_alpha = set(list(gap_map.keys()))
        # Add CASH to alpha map (always 1.0, no vol targeting for cash)
        assets_for_alpha.add("CASH")
        for a in assets_for_alpha:
            if a == "CASH":
                alpha_by_asset["CASH"] = pd.Series(1.0, index=idx)
            elif a not in _build_weekly_alpha_for_asset._cache:
                _build_weekly_alpha_for_asset._cache[a] = _build_weekly_alpha_for_asset(a, gap_map, intra_map)
                alpha_by_asset[a] = _build_weekly_alpha_for_asset._cache[a]
            else:
                alpha_by_asset[a] = _build_weekly_alpha_for_asset._cache[a]
    else:
        # No vol targeting: set all alphas to 1.0 (full exposure)
        assets_for_alpha = set(list(gap_map.keys()))
        assets_for_alpha.add("CASH")
        alpha_by_asset = {a: pd.Series(1.0, index=idx) for a in assets_for_alpha}

    # Drawdown throttle state tracking
    dd_throttle_state = {
        'peak_equity': START_CAPITAL,
        'throttle_engaged_dd': None,  # DD level when throttle first engaged
        'throttle_engaged_peak': START_CAPITAL,  # Peak equity when throttle engaged
        'current_mult': 1.0
    }
    
    def get_dd_throttle_mult(equity, state):
        """Compute drawdown throttle multiplier with cooldown logic"""
        if not USE_DD_THROTTLE_VT:
            return 1.0
        
        # Update peak equity (always track maximum seen)
        state['peak_equity'] = max(state['peak_equity'], equity)
        peak = state['peak_equity']
        dd = 1.0 - (equity / peak) if peak > 0 else 0.0
        
        # Determine desired multiplier based on DD
        if dd <= DD_THROTTLE_THRESH1:
            desired_mult = DD_THROTTLE_MULT1
        elif dd <= DD_THROTTLE_THRESH2:
            desired_mult = DD_THROTTLE_MULT2
        else:
            desired_mult = DD_THROTTLE_MULT3
        
        # Cooldown logic: only increase multiplier after recovering half DD
        if DD_THROTTLE_COOLDOWN and state['throttle_engaged_dd'] is not None:
            # Throttle was engaged, check if we've recovered enough
            recovery_needed = state['throttle_engaged_dd'] / 2.0
            # Current DD from the peak when throttle engaged
            current_dd_from_engaged_peak = 1.0 - (equity / state['throttle_engaged_peak']) if state['throttle_engaged_peak'] > 0 else 0.0
            recovered = state['throttle_engaged_dd'] - current_dd_from_engaged_peak
            
            if recovered >= recovery_needed:
                # Recovered enough, allow multiplier to increase
                state['throttle_engaged_dd'] = None
            elif desired_mult > state['current_mult']:
                # Want to increase but haven't recovered enough, keep current multiplier
                return state['current_mult']
        
        # Track when throttle engages (first time we drop below 1.0)
        if desired_mult < 1.0 and state['throttle_engaged_dd'] is None:
            state['throttle_engaged_dd'] = dd
            state['throttle_engaged_peak'] = peak
        
        state['current_mult'] = desired_mult
        return desired_mult

    if paper:
        eq = START_CAPITAL
        equity_curve = []
        daily_index = []
        curr_asset = None

        for i, d in enumerate(run_idx):
            next_asset = sched.loc[d]

            if curr_asset is None:
                curr_asset = next_asset
                # Handle CASH: use risk-free rate
                if curr_asset == "CASH":
                    r_asset = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
                else:
                    # First day: only intraday return (open-to-open is just intraday on entry day)
                    r_asset_raw = float(intra_map[curr_asset].get(d, 0.0))
                    # Apply proxy drag to asset return, then volatility-targeted sizing
                    drag = proxy_drag(curr_asset, d)
                    r_asset = (1.0 + r_asset_raw) * drag - 1.0
                
                alpha_vt = float(alpha_by_asset.get(curr_asset, pd.Series(1.0, index=idx)).get(d, 1.0))
                rf_t    = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
                
                # Apply DD throttle multiplicatively to alpha (based on current equity before return)
                dd_mult = get_dd_throttle_mult(eq, dd_throttle_state)
                alpha_final = alpha_vt * dd_mult
                r_eff = alpha_final * r_asset + (1.0 - alpha_final) * rf_t
                
                eq *= (1.0 + r_eff)
                equity_curve.append(eq); daily_index.append(d)
                continue

            # Handle CASH: use risk-free rate
            if curr_asset == "CASH":
                r_asset = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
            else:
                # Open-to-open daily return: (1 + gap) * (1 + intra) - 1
                g = float(gap_map[curr_asset].get(d, 0.0))
                if next_asset != curr_asset:
                    i_ret = float(intra_map[next_asset].get(d, 0.0))
                    r_asset_raw = (1.0 + g) * (1.0 + i_ret) - 1.0
                    curr_asset = next_asset
                else:
                    i_ret = float(intra_map[curr_asset].get(d, 0.0))
                    r_asset_raw = (1.0 + g) * (1.0 + i_ret) - 1.0

                # Apply proxy drag to asset return, then volatility-targeted sizing
                drag = proxy_drag(curr_asset, d)
                r_asset = (1.0 + r_asset_raw) * drag - 1.0
            
            # Apply volatility-targeted sizing: r_eff = alpha * r_asset + (1-alpha) * rf
            alpha_vt = float(alpha_by_asset.get(curr_asset, pd.Series(1.0, index=idx)).get(d, 1.0))
            rf_t    = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
            
            # Apply DD throttle multiplicatively to alpha (based on equity before today's return)
            dd_mult = get_dd_throttle_mult(eq, dd_throttle_state)
            alpha_final = alpha_vt * dd_mult
            r_eff = alpha_final * r_asset + (1.0 - alpha_final) * rf_t

            eq *= (1.0 + r_eff)
            equity_curve.append(eq); daily_index.append(d)
        return {"equity_curve": pd.Series(equity_curve, index=pd.DatetimeIndex(daily_index))}

    # REAL branch (taxes) - separate DD throttle state for eq_after
    dd_throttle_state_real = {
        'peak_equity': START_CAPITAL,
        'throttle_engaged_dd': None,
        'throttle_engaged_peak': START_CAPITAL,
        'current_mult': 1.0
    }
    
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

            # Handle CASH: use risk-free rate
            if curr_asset == "CASH":
                r_asset = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
            else:
                # First day: only intraday return (open-to-open is just intraday on entry day)
                r_asset_raw = float(intra_map[curr_asset].get(d, 0.0))
                # Apply proxy drag to asset return, then volatility-targeted sizing
                drag = proxy_drag(curr_asset, d)
                r_asset = (1.0 + r_asset_raw) * drag - 1.0
            
            alpha_vt = float(alpha_by_asset.get(curr_asset, pd.Series(1.0, index=idx)).get(d, 1.0))
            rf_t = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
            
            # Apply DD throttle multiplicatively to alpha (based on eq_after before return)
            dd_mult = get_dd_throttle_mult(eq_after, dd_throttle_state_real)
            alpha_final = alpha_vt * dd_mult
            r_eff = alpha_final * r_asset + (1.0 - alpha_final) * rf_t
            
            eq_pre   *= (1.0 + r_eff)
            eq_after *= (1.0 + r_eff)
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

        # Handle CASH: use risk-free rate
        if curr_asset == "CASH":
            r_asset = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
        else:
            # Open-to-open daily return: (1 + gap) * (1 + intra) - 1
            g = float(gap_map[curr_asset].get(d, 0.0))
            if next_asset != curr_asset:
                i_ret = float(intra_map[next_asset].get(d, 0.0))
                r_asset_raw = (1.0 + g) * (1.0 + i_ret) - 1.0
                curr_asset = next_asset
            else:
                i_ret = float(intra_map[curr_asset].get(d, 0.0))
                r_asset_raw = (1.0 + g) * (1.0 + i_ret) - 1.0

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

        # Apply proxy drag to asset return, then volatility-targeted sizing
        if curr_asset != "CASH":
            drag = proxy_drag(curr_asset, d)
            r_asset = (1.0 + r_asset_raw) * drag - 1.0
        # else: r_asset already set to rf_daily above
        
        # Apply volatility-targeted sizing: r_eff = alpha * r_asset + (1-alpha) * rf
        alpha_vt = float(alpha_by_asset.get(curr_asset, pd.Series(1.0, index=idx)).get(d, 1.0))
        rf_t = float(rf_daily.get(d, FIN_FALLBACK_ANNUAL/252.0))
        
        # Apply DD throttle multiplicatively to alpha (based on eq_after before today's return)
        dd_mult = get_dd_throttle_mult(eq_after, dd_throttle_state_real)
        alpha_final = alpha_vt * dd_mult
        r_eff = alpha_final * r_asset + (1.0 - alpha_final) * rf_t
        
        eq_pre   *= (1.0 + r_eff)
        eq_after *= (1.0 + r_eff)

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
    test_num = 0
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

        test_num += 1
        print(f"Running WF test {test_num}: {test_start.date()} to {test_end.date()}...", flush=True)
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

    # 4) Dashboard-style summary (intended for Thu/Fri after-hours)
    weekly_signal_dashboard(window_start="2019-01-01", lookback_legs=20)

else:
    # Sprint 2: Test Tripwire variants
    print(f"\n{'='*80}")
    print("SPRINT 2: TRIPWIRE DEMOTION GATE TESTING")
    print(f"{'='*80}")
    print(f"\nPeriods used:")
    print(f"  Walk-Forward: {WF_START_DATE} to {WF_END_DATE}")
    print(f"  Shadow OOS: 2018-01-01 to 2020-12-31")
    print(f"  Final OOS: 2021-01-01 to present")
    print(f"\nTripwire: 2+ of 3 flags bad (VIX backwardation, Credit weak, Breadth weak)")
    print(f"  Cooldown: {TRIPWIRE_COOLDOWN_DAYS} business days\n")
    
    # Save original settings
    original_vol_targeting = USE_VOL_TARGETING
    original_vt_mode = VT_MODE
    original_tripwire = USE_TRIPWIRE
    
    # Test 1: No VT + Tripwire
    print("=" * 80)
    print("TEST 1: NO VT + TRIPWIRE")
    print("=" * 80)
    USE_VOL_TARGETING = False
    USE_TRIPWIRE = True
    VT_MODE = "downside_only"
    if hasattr(_build_weekly_alpha_for_asset, '_cache'):
        _build_weekly_alpha_for_asset._cache = {}
    rebuild_regime()
    
    wf_curve_no_vt_trip, wf_tests_no_vt_trip = walk_forward_validation(
        WF_START_DATE, WF_END_DATE,
        WF_TRAIN_YEARS, WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL
    )
    
    results_no_vt_trip = {}
    if wf_curve_no_vt_trip is not None and not wf_curve_no_vt_trip.empty:
        m_wf_no_vt_trip = metrics_from_curve(wf_curve_no_vt_trip)
        results_no_vt_trip['wf'] = m_wf_no_vt_trip
        cp_s_no_vt_trip, cr_s_no_vt_trip, m_s_no_vt_trip = run_segment("Shadow OOS", "2018-01-01", "2020-12-31")
        if m_s_no_vt_trip:
            results_no_vt_trip['shadow'] = m_s_no_vt_trip[0] if WF_USE_PAPER else m_s_no_vt_trip[1]
        cp_f_no_vt_trip, cr_f_no_vt_trip, m_f_no_vt_trip = run_segment("Final OOS", "2021-01-01", None)
        if m_f_no_vt_trip:
            results_no_vt_trip['final'] = m_f_no_vt_trip[0] if WF_USE_PAPER else m_f_no_vt_trip[1]
    
    # Test 2: Downside-VT (current) + Tripwire
    print("\n" + "=" * 80)
    print("TEST 2: DOWNSIDE-VT (BASELINE) + TRIPWIRE")
    print("=" * 80)
    USE_VOL_TARGETING = True
    USE_TRIPWIRE = True
    VT_MODE = "downside_only"
    if hasattr(_build_weekly_alpha_for_asset, '_cache'):
        _build_weekly_alpha_for_asset._cache = {}
    rebuild_regime()
    
    wf_curve_downside_trip, wf_tests_downside_trip = walk_forward_validation(
        WF_START_DATE, WF_END_DATE,
        WF_TRAIN_YEARS, WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL
    )
    
    results_downside_trip = {}
    if wf_curve_downside_trip is not None and not wf_curve_downside_trip.empty:
        m_wf_downside_trip = metrics_from_curve(wf_curve_downside_trip)
        results_downside_trip['wf'] = m_wf_downside_trip
        cp_s_downside_trip, cr_s_downside_trip, m_s_downside_trip = run_segment("Shadow OOS", "2018-01-01", "2020-12-31")
        if m_s_downside_trip:
            results_downside_trip['shadow'] = m_s_downside_trip[0] if WF_USE_PAPER else m_s_downside_trip[1]
        cp_f_downside_trip, cr_f_downside_trip, m_f_downside_trip = run_segment("Final OOS", "2021-01-01", None)
        if m_f_downside_trip:
            results_downside_trip['final'] = m_f_downside_trip[0] if WF_USE_PAPER else m_f_downside_trip[1]
    
    # Test 3: Light VT (state-2 only) + Tripwire
    print("\n" + "=" * 80)
    print("TEST 3: LIGHT VT (STATE-2 ONLY) + TRIPWIRE")
    print("=" * 80)
    USE_VOL_TARGETING = True
    USE_TRIPWIRE = True
    VT_MODE = "regime_dependent"
    # Light VT: State 2: σ*=0.40-0.45, State 1: σ*=0.25, State 0: no VT
    VT_TARGET_BY_REGIME[2] = 0.40  # Test 0.40 first
    VT_TARGET_BY_REGIME[1] = 0.25
    VT_TARGET_BY_REGIME[0] = None
    VT_ALPHA_MIN_BY_REGIME[2] = 0.60
    VT_ALPHA_MIN_BY_REGIME[1] = 0.40
    if hasattr(_build_weekly_alpha_for_asset, '_cache'):
        _build_weekly_alpha_for_asset._cache = {}
    rebuild_regime()
    
    wf_curve_light_vt_trip, wf_tests_light_vt_trip = walk_forward_validation(
        WF_START_DATE, WF_END_DATE,
        WF_TRAIN_YEARS, WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL
    )
    
    results_light_vt_trip = {}
    if wf_curve_light_vt_trip is not None and not wf_curve_light_vt_trip.empty:
        m_wf_light_vt_trip = metrics_from_curve(wf_curve_light_vt_trip)
        results_light_vt_trip['wf'] = m_wf_light_vt_trip
        cp_s_light_vt_trip, cr_s_light_vt_trip, m_s_light_vt_trip = run_segment("Shadow OOS", "2018-01-01", "2020-12-31")
        if m_s_light_vt_trip:
            results_light_vt_trip['shadow'] = m_s_light_vt_trip[0] if WF_USE_PAPER else m_s_light_vt_trip[1]
        cp_f_light_vt_trip, cr_f_light_vt_trip, m_f_light_vt_trip = run_segment("Final OOS", "2021-01-01", None)
        if m_f_light_vt_trip:
            results_light_vt_trip['final'] = m_f_light_vt_trip[0] if WF_USE_PAPER else m_f_light_vt_trip[1]
    
    # Baseline: Downside-VT (no tripwire) for comparison
    print("\n" + "=" * 80)
    print("BASELINE: DOWNSIDE-VT (NO TRIPWIRE)")
    print("=" * 80)
    USE_VOL_TARGETING = True
    USE_TRIPWIRE = False
    VT_MODE = "downside_only"
    if hasattr(_build_weekly_alpha_for_asset, '_cache'):
        _build_weekly_alpha_for_asset._cache = {}
    rebuild_regime()
    
    wf_curve_downside, wf_tests_downside = walk_forward_validation(
        WF_START_DATE, WF_END_DATE,
        WF_TRAIN_YEARS, WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL
    )
    
    results_downside = {}
    if wf_curve_downside is not None and not wf_curve_downside.empty:
        m_wf_downside = metrics_from_curve(wf_curve_downside)
        results_downside['wf'] = m_wf_downside
        cp_s_downside, cr_s_downside, m_s_downside = run_segment("Shadow OOS", "2018-01-01", "2020-12-31")
        if m_s_downside:
            results_downside['shadow'] = m_s_downside[0] if WF_USE_PAPER else m_s_downside[1]
        cp_f_downside, cr_f_downside, m_f_downside = run_segment("Final OOS", "2021-01-01", None)
        if m_f_downside:
            results_downside['final'] = m_f_downside[0] if WF_USE_PAPER else m_f_downside[1]
    
    # Original: No VT, no tripwire
    print("\n" + "=" * 80)
    print("ORIGINAL: NO VT, NO TRIPWIRE")
    print("=" * 80)
    USE_VOL_TARGETING = False
    USE_TRIPWIRE = False
    if hasattr(_build_weekly_alpha_for_asset, '_cache'):
        _build_weekly_alpha_for_asset._cache = {}
    rebuild_regime()
    
    wf_curve_no_vt, wf_tests_no_vt = walk_forward_validation(
        WF_START_DATE, WF_END_DATE,
        WF_TRAIN_YEARS, WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL
    )
    
    results_no_vt = {}
    if wf_curve_no_vt is not None and not wf_curve_no_vt.empty:
        m_wf_no_vt = metrics_from_curve(wf_curve_no_vt)
        results_no_vt['wf'] = m_wf_no_vt
        cp_s_no_vt, cr_s_no_vt, m_s_no_vt = run_segment("Shadow OOS", "2018-01-01", "2020-12-31")
        if m_s_no_vt:
            results_no_vt['shadow'] = m_s_no_vt[0] if WF_USE_PAPER else m_s_no_vt[1]
        cp_f_no_vt, cr_f_no_vt, m_f_no_vt = run_segment("Final OOS", "2021-01-01", None)
        if m_f_no_vt:
            results_no_vt['final'] = m_f_no_vt[0] if WF_USE_PAPER else m_f_no_vt[1]
    
    # Restore original settings
    USE_VOL_TARGETING = original_vol_targeting
    VT_MODE = original_vt_mode
    USE_TRIPWIRE = original_tripwire
    rebuild_regime()
    
    # Print comparison
    print("\n" + "=" * 80)
    print("TRIPWIRE COMPARISON RESULTS")
    print("=" * 80)
    
    def print_comparison_tripwire(name, m_no_vt_trip, m_downside_trip, m_light_vt_trip, m_downside_base, m_no_vt_base):
        if not all([m_no_vt_trip, m_downside_trip, m_light_vt_trip, m_downside_base, m_no_vt_base]):
            return
        print(f"\n{name}:")
        print(f"{'Metric':<20} {'NoVT+Trip':>15} {'DownVT+Trip':>15} {'LightVT+Trip':>15} {'DownVT':>15} {'NoVT':>15}")
        print("-" * 95)
        for key in ['CAGR', 'MaxDD', 'AnnVol', 'Sharpe', 'Sortino', 'Calmar']:
            no_vt_trip_val = m_no_vt_trip.get(key, np.nan)
            downside_trip_val = m_downside_trip.get(key, np.nan)
            light_vt_trip_val = m_light_vt_trip.get(key, np.nan)
            downside_base_val = m_downside_base.get(key, np.nan)
            no_vt_base_val = m_no_vt_base.get(key, np.nan)
            if not (np.isnan(no_vt_trip_val) or np.isnan(downside_trip_val) or np.isnan(light_vt_trip_val) or np.isnan(downside_base_val) or np.isnan(no_vt_base_val)):
                if key == 'MaxDD':
                    print(f"{key:<20} {fmt_pct(no_vt_trip_val):>15} {fmt_pct(downside_trip_val):>15} {fmt_pct(light_vt_trip_val):>15} {fmt_pct(downside_base_val):>15} {fmt_pct(no_vt_base_val):>15}")
                else:
                    print(f"{key:<20} {no_vt_trip_val:>15.4f} {downside_trip_val:>15.4f} {light_vt_trip_val:>15.4f} {downside_base_val:>15.4f} {no_vt_base_val:>15.4f}")
            else:
                print(f"{key:<20} {'nan':>15} {'nan':>15} {'nan':>15} {'nan':>15} {'nan':>15}")
    
    if all([results_no_vt_trip.get('wf'), results_downside_trip.get('wf'), results_light_vt_trip.get('wf'), results_downside.get('wf'), results_no_vt.get('wf')]):
        print_comparison_tripwire("Walk-Forward (1963-2017)", results_no_vt_trip['wf'], results_downside_trip['wf'], results_light_vt_trip['wf'], results_downside['wf'], results_no_vt['wf'])
    if all([results_no_vt_trip.get('shadow'), results_downside_trip.get('shadow'), results_light_vt_trip.get('shadow'), results_downside.get('shadow'), results_no_vt.get('shadow')]):
        print_comparison_tripwire("Shadow OOS (2018-2020)", results_no_vt_trip['shadow'], results_downside_trip['shadow'], results_light_vt_trip['shadow'], results_downside['shadow'], results_no_vt['shadow'])
    if all([results_no_vt_trip.get('final'), results_downside_trip.get('final'), results_light_vt_trip.get('final'), results_downside.get('final'), results_no_vt.get('final')]):
        print_comparison_tripwire("Final OOS (2021+)", results_no_vt_trip['final'], results_downside_trip['final'], results_light_vt_trip['final'], results_downside['final'], results_no_vt['final'])
    
    print("\n" + "=" * 80)