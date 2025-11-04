# dual_day_strategy.py
# Weekly 3-state engine with dual execution days:
# - Decide Thursday close -> execute next trading day open (usually Friday)
# - Decide Friday close   -> execute next trading day open (usually Monday)
# Keeps original gates, taxes, scheduler, and Nasdaq lockout.
# Adds: Linear Regression Channel hedge overlay with UB3/UB2 hysteresis bands.

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
DATA_START = "1950-01-01"
DATA_END = None  # None => today

# === Hedge overlay config ===
# Match hedge leverage to main-leg leverage
HEDGE_LEVERAGE_BY_ASSET = {
    "TQQQ": -3.0,  # hedge 3x legs with ~-3x
    "QLD":  -2.0,  # hedge 2x legs with ~-2x
}

# Min hold time to reduce hedge churn (days)
HEDGE_MIN_HOLD_ON_DAYS = (
    2  # require ≥2 consecutive days hedged before allowing turn-off
)
HEDGE_MIN_HOLD_OFF_DAYS = (
    2  # require ≥2 consecutive days unhedged before allowing turn-on
)
# --- Low-risk improvements ---
HEDGE_SD_SCALE = 0.9
HEDGE_SLOPE_ESCALATION = True
HEDGE_SLOPE_LOOKBACK_D = 5
HEDGE_SLOPE_THRESH = 0.0

HEDGE_VOL_AWARE_EPS = True
HEDGE_VOL_LOOKBACK_D = 20
HEDGE_VOL_EPS_MIN = 0.5
HEDGE_VOL_EPS_MAX = 2.0

HEDGE_CAP_AVOID_NET_SHORT = (
    True  # cap w to avoid net short in 2x regime (e.g., QLD)
)
HEDGE_SLOPE_ESCALATION = (
    True  # only escalate to 1/2 when channel slope <= threshold
)
HEDGE_SLOPE_LOOKBACK_D = 5  # slope lookback in days
HEDGE_SLOPE_THRESH = 0.0  # require midline_slope <= 0 to allow 1/2

HEDGE_VOL_LOOKBACK_D = 20  # realized vol lookback
HEDGE_VOL_EPS_MIN = 0.5  # min scaling of eps
HEDGE_VOL_EPS_MAX = 2.0  # max scaling of eps

USE_HEDGE_OVERLAY = True
HEDGE_PRICE_SERIES = "^NDX"  # compute LR channel on NDX (or "QQQ")
HEDGE_APPLIES_TO = {"TQQQ", "QLD"}  # only hedge when these are held
HEDGE_WINDOW_N = 252  # LR window (trading days)
HEDGE_WIDTHS = (1.0, 2.0, 3.0, 4.0)  # UB1..UB4
HEDGE_SMA_LENGTH = 200  # SMA gate for hedge

# Band hysteresis thresholds (lower-band crossings)
# eps = 0.001 means 0.1% around the UB level
HEDGE_EPS_UB2 = 0.0015
HEDGE_EPS_UB3 = 0.0015
HEDGE_EPS_UB4 = 0.0015
HEDGE_LEVERAGE = -3.0  # -3x leverage for hedge instrument (SQQQ)

# Step sizes
HEDGE_W_NEAR = 1.0 / 3.0  # 1/3 overlay when above UB3 lower band
HEDGE_W_HIGH = 0.50  # 1/2 overlay if above UB4 (with hysteresis)

# Charge slippage at open when the hedge weight changes
HEDGE_SLIPPAGE_ON_ADJUST = True

# WALK FORWARD BACKTEST
DO_WALK_FORWARD = True

WF_START_DATE = "2000-01-01"
WF_END_DATE = "2025-01-01"
WF_TRAIN_YEARS = 2
WF_TEST_YEARS = 1
WF_USE_PAPER = True
WF_CHAIN_CAPITAL = True

IN_SAMPLE = (None, None)
OOS_MAIN = (None, None)
HOLDOUT = ("2000-01-01", "2015-01-01")

START_CAPITAL = 21000.0

# SMA/Signal params (3-state engine uses ^GSPC by default)
SIGNAL_SYMBOL = "^GSPC"
SMA_SLOW = 200
SMA_MID = 100
BAND_PCT = 0.005

# 3-state behavior
ONE_SIDED_2X = True
ADAPTIVE_UP_WINDOW = True
MID_UP_WINDOW_WEEKS_BENIGN = 4
MID_UP_WINDOW_WEEKS_STRESSED = 6

USE_SLOPE_FILTER = True
SLOPE_LOOKBACK_W = 4

# Additional gates
USE_DIST_GATE_200 = True
DELTA_200_PCT = 0.010
USE_ADAPTIVE_DIST = True

USE_VOL_CAP = True
VOL_LOOKBACK_D = 20
VOL_TH_3X = 0.35
VOL_TH_ROFF = 0.45

USE_DD_THROTTLE = True
DD_LOOKBACK_D = 252
DD_TH_2X = 0.20

# Risk-off (GLD vs IEF momentum)
USE_DUAL_RISK_OFF = False  # GLD only by default
RISK_OFF_LOOKBACK_D = 63

# Trading frictions
SLIPPAGE_BPS = 5
SLIPPAGE_BPS_STRESS = 15
STRESS_THRESHOLD = 0.02

# Taxes (only if paper=False)
ST_RATE = 0.37
LT_RATE = 0.15
DIV_TAX_RATE = 0.34
ORD_RATE = 0.34
LOSS_DED_CAP = 3000.0

# Proxy drags
LEV_ANNUAL_FEE_3X = 0.0094
LEV_ANNUAL_FEE_2X = 0.0095
GLD_ER_ANNUAL = 0.0040
LEV_EXCESS = 2.0
LEV_EXCESS_2X = 1.0
FIN_FALLBACK_ANNUAL = 0.03
APPLY_PROXY_DRAGS = True

# Proxy calibration
CALIBRATE_TQQQ_PROXY = True
CALIBRATE_GLD_PROXY = False
CALIBRATE_SPXL_PROXY = True
CALIBRATE_QLD_PROXY = True
CALIBRATE_SSO_PROXY = True

# Wash-sale avoidance for Nasdaq legs
LOCKOUT_DAYS = 30

# Preferred gold proxy source (for pre-inception)
USE_LBMA_SPOT = True

# ============ Dashboard lineups (availability fallback only) ============
RISK_ON_3X_LINEUP = [
    "TQQQ",
    "SPXL",
    "UPRO",
    "UDOW",
    "TECL",
    "SOXL",
    "TNA",
    "FNGU",
]
RISK_ON_2X_LINEUP = ["QLD", "SSO", "SPUU", "DDM", "UWM", "ROM"]
RISK_OFF_GOLD_LINEUP = ["GLD", "IAU", "GLDM", "SGOL", "BAR", "AAAU"]
RISK_OFF_BOND_LINEUP = [
    "IEF",
    "VGIT",
    "GOVT",
    "SCHR",
    "IEI",
    "SHY",
    "TLH",
]
lineup_syms = set(
    RISK_ON_3X_LINEUP
    + RISK_ON_2X_LINEUP
    + RISK_OFF_GOLD_LINEUP
    + RISK_OFF_BOND_LINEUP
)

# =========================
# Helpers (LR channel + hedge weight)
# =========================

def hedge_series_for_asset(asset: str):
    lev = HEDGE_LEVERAGE_BY_ASSET.get(asset, HEDGE_LEVERAGE)
    if math.isclose(abs(lev), 3.0):
        return gap_INV_m3, intra_INV_m3
    if math.isclose(abs(lev), 2.0):
        return gap_INV_m2, intra_INV_m2
    g, i = build_inv_ndx_returns(lev)
    return g, i

def cap_w_for_asset(w: float, asset: str) -> float:
    if not (USE_HEDGE_OVERLAY and HEDGE_CAP_AVOID_NET_SHORT):
        return w
    L_map = {"TQQQ": 3.0, "QLD": 2.0}
    L = L_map.get(asset)
    if L is None:
        return w
    H_abs = abs(HEDGE_LEVERAGE_BY_ASSET.get(asset, HEDGE_LEVERAGE))
    w_cap = L / (L + H_abs)
    return min(w, w_cap)


def _rolling_lsma_mid_and_resid_sd(arr):
    x = np.arange(arr.size)
    m, b = np.polyfit(x, arr, 1)
    y = b + m * x
    resid = arr - y
    mid_last = y[-1]
    sd = float(np.std(resid))
    return mid_last, sd


def build_lr_channels(
    price: pd.Series, n=252, widths=(1.0, 2.0, 3.0, 4.0), sd_scale=1.0
) -> pd.DataFrame:
    def mid_func(arr):
        x = np.arange(arr.size)
        m, b = np.polyfit(x, arr, 1)
        return b + m * (arr.size - 1)

    def sd_func(arr):
        x = np.arange(arr.size)
        m, b = np.polyfit(x, arr, 1)
        resid = arr - (b + m * x)
        return float(np.std(resid))

    mid = price.rolling(n, min_periods=n).apply(mid_func, raw=True)
    sd = (
        price.rolling(n, min_periods=n).apply(sd_func, raw=True)
        * sd_scale
    )
    out = {"mid": mid}
    for i, w in enumerate(widths, start=1):
        out[f"ub{i}"] = mid + w * sd
        out[f"lb{i}"] = mid - w * sd
    return pd.DataFrame(out)


def build_band_hysteresis_hedge_weights(
    price: pd.Series,
    ub2: pd.Series,
    ub3: pd.Series,
    ub4: pd.Series,
    sma200: pd.Series,
    eps2=0.001,
    eps3=0.001,
    eps4=0.001,
    w_near=1.0 / 3.0,
    w_high=0.50,
    require_sma=True,
    allow_high=None,
    min_on_days: int = 0,
    min_off_days: int = 0,
) -> pd.Series:
    """
    State machine with hysteresis and dwell time:
    - Enter hedge (w_near) when price crosses above UB3*(1 - eps3) and (optionally) price > SMA200.
    - Escalate to w_high when price crosses above UB4*(1 + eps4) AND allow_high[d] (if provided).
    - De-escalate to w_near when price crosses below UB4*(1 - eps4) but stays >= UB3*(1 - eps3).
    - Exit hedge (0) when price crosses below UB2*(1 - eps2) or (require_sma and price <= SMA200).
    - min_on_days/min_off_days enforce minimum days “hedged vs unhedged” between flips (escalate/de-escalate within hedged is allowed).
    eps2/eps3/eps4 can be scalars or Series; allow_high can be a boolean Series.
    """

    def _val(x, d, fallback):
        if isinstance(x, pd.Series):
            v = x.get(d, np.nan)
            return fallback if pd.isna(v) else float(v)
        return float(x)

    w = pd.Series(0.0, index=price.index)
    state = 0  # 0=off, 1=near, 2=high
    on_cnt = 0
    off_cnt = 1  # start as off for 1 day

    for d in price.index:
        p = price.loc[d]
        m = sma200.loc[d] if d in sma200.index else np.nan
        u2 = ub2.loc[d] if d in ub2.index else np.nan
        u3 = ub3.loc[d] if d in ub3.index else np.nan
        u4 = ub4.loc[d] if d in ub4.index else np.nan
        if (
            np.isnan(p)
            or np.isnan(u2)
            or np.isnan(u3)
            or np.isnan(u4)
            or np.isnan(m)
        ):
            w.iloc[w.index.get_loc(d)] = 0.0
            # still count as off day
            off_cnt += 1
            on_cnt = 0
            state = 0
            continue

        e2 = _val(eps2, d, 0.001)
        e3 = _val(eps3, d, 0.001)
        e4 = _val(eps4, d, 0.001)
        can_escalate = (
            True
            if allow_high is None
            else bool(pd.Series(allow_high).get(d, False))
        )

        # Propose next state from price logic
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
                elif can_escalate and p >= u4 * (1 + e4):
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

        # Enforce dwell time only for off<->on flips (1/2 <-> 1 counted as on)
        curr_on = state != 0
        next_on = ns != 0
        if (
            (not curr_on)
            and next_on
            and (off_cnt < int(min_off_days))
        ):
            ns = 0  # too soon to turn on
        if curr_on and (not next_on) and (on_cnt < int(min_on_days)):
            ns = state  # too soon to turn off

        # Commit and update streak counters
        state = ns
        if state != 0:
            on_cnt += 1
            off_cnt = 0
        else:
            off_cnt += 1
            on_cnt = 0

        w.iloc[w.index.get_loc(d)] = (
            w_near if state == 1 else (w_high if state == 2 else 0.0)
        )

    return w


# =========================
# Helpers (metrics, math, IO)
# =========================
def cagr_from_curve(curve):
    if len(curve) < 2:
        return np.nan
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    if yrs <= 0:
        return np.nan
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1)


def max_drawdown_from_curve(curve):
    if curve.empty:
        return np.nan
    running_max = curve.cummax()
    dd = curve / running_max - 1.0
    return float(dd.min())


def annualized_vol_from_curve(curve):
    if len(curve) < 2:
        return np.nan
    rets = curve.pct_change().dropna()
    return float(rets.std() * math.sqrt(252))


def sharpe_from_curve(curve, rf=0.0):
    rets = curve.pct_change().dropna()
    if rets.empty:
        return np.nan
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std() * math.sqrt(252)
    return float((ann_ret - rf) / ann_vol) if ann_vol > 0 else np.nan


def sortino_from_curve(curve, rf=0.0):
    rets = curve.pct_change().dropna()
    if rets.empty:
        return np.nan
    downside = rets[rets < 0]
    dd = downside.std() * math.sqrt(252)
    ann_ret = (1 + rets.mean()) ** 252 - 1
    return float((ann_ret - rf) / dd) if dd > 0 else np.nan


def calmar_from_curve(curve):
    cagr = cagr_from_curve(curve)
    mdd = max_drawdown_from_curve(curve)
    return (
        float(cagr / abs(mdd))
        if (mdd is not None and mdd < 0)
        else np.nan
    )


def adj_open(open_s, close_s, adj_close_s):
    return open_s * (adj_close_s / close_s)


def seg_returns(ao, ac):
    gap = ao / ac.shift(1) - 1.0
    intra = ac / ao - 1.0
    return gap, intra


def year_returns_from_curve(curve):
    if curve.empty:
        return pd.Series(dtype=float)
    y_end = curve.resample("YE").last()
    y_start = y_end.shift(1)
    yr = (y_end / y_start - 1.0).dropna()
    yr.index = yr.index.year
    return yr


def fmt_pct(x):
    if x is None or (
        isinstance(x, float) and (np.isnan(x) or np.isinf(x))
    ):
        return "nan"
    return f"{100.0 * x:.2f}%"


def next_trading_day(date_like):
    d = pd.Timestamp(date_like).normalize()
    pos = idx.searchsorted(d, side="right")
    return idx[pos] if pos < len(idx) else None


# =========================
# Download data
# =========================
if DATA_END is None:
    DATA_END = (pd.Timestamp.today() + BDay(1)).date().isoformat()

tickers = sorted(
    set(
        [SIGNAL_SYMBOL, "^GSPC", "^NDX", "SPY", "GC=F"]
        + list(lineup_syms)
    )
)
dl_lookback = (
    max(SMA_SLOW, SMA_MID, DD_LOOKBACK_D, RISK_OFF_LOOKBACK_D) + 10
)
dl_start = (
    (pd.to_datetime(DATA_START) - BDay(dl_lookback))
    .date()
    .isoformat()
)
px = yf.download(
    tickers,
    start=dl_start,
    end=DATA_END,
    auto_adjust=False,
    progress=True,
)

# Reference series for SMA/logic
if (
    SIGNAL_SYMBOL not in px["Adj Close"].columns
    or px["Adj Close"][SIGNAL_SYMBOL].isnull().all()
):
    print(
        f"Warning: SIGNAL_SYMBOL {SIGNAL_SYMBOL} not found or all NaN. Defaulting to ^GSPC."
    )
    SIGNAL_SYMBOL = "^GSPC"

ref_ac = px["Adj Close"][SIGNAL_SYMBOL].dropna()
idx = ref_ac.index


# Build adjusted open and adjusted close for symbols
def get_adj_ohlc(px, sym, idx):
    if (
        sym not in px["Adj Close"].columns
        or px["Adj Close"][sym].isnull().all()
    ):
        ao = pd.Series(index=idx, dtype=float)
        ac = pd.Series(index=idx, dtype=float)
        cl = pd.Series(index=idx, dtype=float)
        return ao, ac, cl
    c = px["Close"][sym].dropna()
    ao = adj_open(
        px["Open"][sym].reindex(c.index),
        c,
        px["Adj Close"][sym].reindex(c.index),
    )
    ac = px["Adj Close"][sym].reindex(c.index)
    cl = px["Close"][sym].reindex(c.index)
    ao = ao.reindex(idx)
    ac = ac.reindex(idx)
    cl = cl.reindex(idx)
    return ao, ac, cl


ndx_ao, ndx_ac, ndx_cl = get_adj_ohlc(px, "^NDX", idx)
tqqq_ao, tqqq_ac, tqqq_cl = get_adj_ohlc(px, "TQQQ", idx)
gld_ao, gld_ac, gld_cl = get_adj_ohlc(px, "GLD", idx)
gcf_ao, gcf_ac, gcf_cl = get_adj_ohlc(px, "GC=F", idx)
spy_ao, spy_ac, spy_cl = get_adj_ohlc(px, "SPY", idx)
spxl_ao, spxl_ac, spxl_cl = get_adj_ohlc(px, "SPXL", idx)
qld_ao, qld_ac, qld_cl = get_adj_ohlc(px, "QLD", idx)
sso_ao, sso_ac, sso_cl = get_adj_ohlc(px, "SSO", idx)
ief_ao, ief_ac, ief_cl = get_adj_ohlc(px, "IEF", idx)
gspc_ac = px["Adj Close"]["^GSPC"].reindex(idx)


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
div_gld = get_div_series("GLD", idx)
div_spxl = get_div_series("SPXL", idx)
div_qld = get_div_series("QLD", idx)
div_sso = get_div_series("SSO", idx)
div_ief = get_div_series("IEF", idx)


# =========================
# Risk-free (daily) for financing; LBMA spot loader
# =========================
def load_rf_daily(idx):
    if pdr is None:
        return pd.Series(FIN_FALLBACK_ANNUAL / 252.0, index=idx)
    try:
        dff = pdr.DataReader(
            "DFF", "fred", idx.min(), idx.max()
        ).squeeze()
    except Exception:
        dff = None
    try:
        tb3 = pdr.DataReader(
            "TB3MS", "fred", idx.min(), idx.max()
        ).squeeze()
        tb3 = tb3.resample("B").ffill()
    except Exception:
        tb3 = None

    if dff is None and tb3 is None:
        return pd.Series(FIN_FALLBACK_ANNUAL / 252.0, index=idx)

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
        s = pdr.DataReader(
            "GOLDPMGBD228NLBM", "fred", idx.min(), idx.max()
        ).squeeze()
        s = s.reindex(
            pd.date_range(s.index.min(), s.index.max(), freq="B")
        ).ffill()
        return s.reindex(idx).ffill()
    except Exception:
        return None


rf_daily = load_rf_daily(idx)
fee_daily_3x = LEV_ANNUAL_FEE_3X / 252.0
fee_daily_2x = LEV_ANNUAL_FEE_2X / 252.0
gld_fee_daily = GLD_ER_ANNUAL / 252.0


# =========================
# Build asset returns with pre-inception proxies
# =========================
def build_asset_returns_GLD():
    real_gap, real_intra = seg_returns(gld_ao, gld_ac)
    has = (~gld_ao.isna()) & (~gld_ac.isna())

    spot = load_lbma_spot(idx)
    if spot is not None and not spot.empty:
        ao_spot = spot.shift(1)
        ac_spot = spot
        proxy_gap, proxy_intra = seg_returns(ao_spot, ac_spot)
    else:
        proxy_gap, proxy_intra = seg_returns(gcf_ao, gcf_ac)

    if CALIBRATE_GLD_PROXY and has.any():
        real_T = (1 + real_gap) * (1 + real_intra) - 1.0
        prox_T = (1 + proxy_gap) * (1 + proxy_intra) - 1.0
        overlap = real_T.dropna().index.intersection(
            prox_T.dropna().index
        )
        if len(overlap) >= 200:
            b, a = np.polyfit(
                prox_T.loc[overlap].values,
                real_T.loc[overlap].values,
                1,
            )
            pre = ~has
            T_pre = (1 + proxy_gap.loc[pre]) * (
                1 + proxy_intra.loc[pre]
            ) - 1.0
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = (1.0 + T_adj) / (
                1.0 + proxy_gap.loc[pre]
            ) - 1.0

    gap = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied


def build_asset_returns_TQQQ():
    base_gap, base_intra = seg_returns(ndx_ao, ndx_ac)
    lev = 3.0
    proxy_gap = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = (1 + base_gap).fillna(1.0) * (
        1 + base_intra
    ).fillna(1.0) - 1.0
    proxy_intra = (
        ((1.0 + lev * proxy_daily) / den - 1.0)
        .clip(lower=-0.95)
        .fillna(0.0)
    )

    real_gap, real_intra = seg_returns(tqqq_ao, tqqq_ac)
    has = (~tqqq_ao.isna()) & (~tqqq_ac.isna())

    if CALIBRATE_TQQQ_PROXY:
        real_T = (1 + real_gap) * (1 + real_intra) - 1.0
        prox_T = (1 + proxy_gap) * (1 + proxy_intra) - 1.0
        overlap = real_T.dropna().index.intersection(
            prox_T.dropna().index
        )
        if len(overlap) >= 200:
            b, a = np.polyfit(
                prox_T.loc[overlap].values,
                real_T.loc[overlap].values,
                1,
            )
            pre = ~has
            T_pre = (1 + proxy_gap.loc[pre]) * (
                1 + proxy_intra.loc[pre]
            ) - 1.0
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = (
                (1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0
            ).clip(lower=-0.95)

    gap = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied


def build_asset_returns_SPXL():
    base_gap, base_intra = seg_returns(spy_ao, spy_ac)
    lev = 3.0
    proxy_gap = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = (1 + base_gap).fillna(1.0) * (
        1 + base_intra
    ).fillna(1.0) - 1.0
    proxy_intra = (
        ((1.0 + lev * proxy_daily) / den - 1.0)
        .clip(lower=-0.95)
        .fillna(0.0)
    )

    real_gap, real_intra = seg_returns(spxl_ao, spxl_ac)
    has = (~spxl_ao.isna()) & (~spxl_ac.isna())

    if CALIBRATE_SPXL_PROXY:
        real_T = (1 + real_gap) * (1 + real_intra) - 1.0
        prox_T = (1 + proxy_gap) * (1 + proxy_intra) - 1.0
        overlap = real_T.dropna().index.intersection(
            prox_T.dropna().index
        )
        if len(overlap) >= 200:
            b, a = np.polyfit(
                prox_T.loc[overlap].values,
                real_T.loc[overlap].values,
                1,
            )
            pre = ~has
            T_pre = (1 + proxy_gap.loc[pre]) * (
                1 + proxy_intra.loc[pre]
            ) - 1.0
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = (
                (1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0
            ).clip(lower=-0.95)

    gap = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied


def build_asset_returns_QLD():
    base_gap, base_intra = seg_returns(ndx_ao, ndx_ac)
    lev = 2.0
    proxy_gap = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = (1 + base_gap).fillna(1.0) * (
        1 + base_intra
    ).fillna(1.0) - 1.0
    proxy_intra = (
        ((1.0 + lev * proxy_daily) / den - 1.0)
        .clip(lower=-0.95)
        .fillna(0.0)
    )

    real_gap, real_intra = seg_returns(qld_ao, qld_ac)
    has = (~qld_ao.isna()) & (~qld_ac.isna())

    if CALIBRATE_QLD_PROXY:
        real_T = (1 + real_gap) * (1 + real_intra) - 1.0
        prox_T = (1 + proxy_gap) * (1 + proxy_intra) - 1.0
        overlap = real_T.dropna().index.intersection(
            prox_T.dropna().index
        )
        if len(overlap) >= 200:
            b, a = np.polyfit(
                prox_T.loc[overlap].values,
                real_T.loc[overlap].values,
                1,
            )
            pre = ~has
            T_pre = (1 + proxy_gap.loc[pre]) * (
                1 + proxy_intra.loc[pre]
            ) - 1.0
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = (
                (1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0
            ).clip(lower=-0.95)

    gap = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied


def build_asset_returns_SSO():
    base_gap, base_intra = seg_returns(spy_ao, spy_ac)
    lev = 2.0
    proxy_gap = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = (1 + base_gap).fillna(1.0) * (
        1 + base_intra
    ).fillna(1.0) - 1.0
    proxy_intra = (
        ((1.0 + lev * proxy_daily) / den - 1.0)
        .clip(lower=-0.95)
        .fillna(0.0)
    )

    real_gap, real_intra = seg_returns(sso_ao, sso_ac)
    has = (~sso_ao.isna()) & (~sso_ac.isna())

    if CALIBRATE_SSO_PROXY:
        real_T = (1 + real_gap) * (1 + real_intra) - 1.0
        prox_T = (1 + proxy_gap) * (1 + proxy_intra) - 1.0
        overlap = real_T.dropna().index.intersection(
            prox_T.dropna().index
        )
        if len(overlap) >= 200:
            b, a = np.polyfit(
                prox_T.loc[overlap].values,
                real_T.loc[overlap].values,
                1,
            )
            pre = ~has
            T_pre = (1 + proxy_gap.loc[pre]) * (
                1 + proxy_intra.loc[pre]
            ) - 1.0
            T_adj = a + b * T_pre
            proxy_intra.loc[pre] = (
                (1.0 + T_adj) / (1.0 + proxy_gap.loc[pre]) - 1.0
            ).clip(lower=-0.95)

    gap = real_gap.where(has, proxy_gap).fillna(0.0)
    intra = real_intra.where(has, proxy_intra).fillna(0.0)
    proxied = ~has
    return gap, intra, proxied


def build_inv_ndx_returns(lev=-3.0):
    
    base_gap, base_intra = seg_returns(ndx_ao, ndx_ac)
    proxy_gap = (lev * base_gap).clip(lower=-0.95).fillna(0.0)
    den = (1.0 + proxy_gap).replace(-1.0, -0.999999)
    proxy_daily = (1 + base_gap).fillna(1.0) * (
        1 + base_intra
    ).fillna(1.0) - 1.0
    proxy_intra = (
        ((1.0 + lev * proxy_daily) / den - 1.0)
        .clip(lower=-0.95)
        .fillna(0.0)
    )
    # Align to idx
    proxy_gap = proxy_gap.reindex(idx).fillna(0.0)
    proxy_intra = proxy_intra.reindex(idx).fillna(0.0)
    return proxy_gap, proxy_intra


# Build all returns
gap_GLD, intra_GLD, prox_mask_GLD = build_asset_returns_GLD()
gap_TQQQ, intra_TQQQ, prox_mask_TQQQ = build_asset_returns_TQQQ()
gap_SPXL, intra_SPXL, prox_mask_SPXL = build_asset_returns_SPXL()
gap_QLD, intra_QLD, prox_mask_QLD = build_asset_returns_QLD()
gap_SSO, intra_SSO, prox_mask_SSO = build_asset_returns_SSO()

# Hedge instrument (SQQQ ~ -3x NDX)
# Hedge instruments (~inverse NDX proxies)
gap_INV_m3, intra_INV_m3 = build_inv_ndx_returns(-3.0)  # ~SQQQ
gap_INV_m2, intra_INV_m2 = build_inv_ndx_returns(-2.0)  # ~QID

prox_mask_GLD = prox_mask_GLD.reindex(idx).fillna(False).astype(bool)
prox_mask_TQQQ = (
    prox_mask_TQQQ.reindex(idx).fillna(False).astype(bool)
)
prox_mask_SPXL = (
    prox_mask_SPXL.reindex(idx).fillna(False).astype(bool)
)
prox_mask_QLD = prox_mask_QLD.reindex(idx).fillna(False).astype(bool)
prox_mask_SSO = prox_mask_SSO.reindex(idx).fillna(False).astype(bool)

# === Hedge channels and weights ===
# === Hedge channels and weights ===
if USE_HEDGE_OVERLAY:
    try:
        hedge_price = px["Adj Close"][HEDGE_PRICE_SERIES].reindex(idx)
    except Exception:
        hedge_price = ndx_ac  # fallback

    ch_df = build_lr_channels(
        hedge_price,
        n=HEDGE_WINDOW_N,
        widths=HEDGE_WIDTHS,
        sd_scale=HEDGE_SD_SCALE,
    ).reindex(idx)
    sma200_hedge = hedge_price.rolling(
        HEDGE_SMA_LENGTH, min_periods=HEDGE_SMA_LENGTH
    ).mean()

    # Vol-aware hysteresis (scale eps by recent realized vol)
    if HEDGE_VOL_AWARE_EPS:
        rv = (
            hedge_price.pct_change()
            .rolling(HEDGE_VOL_LOOKBACK_D)
            .std()
        )
        med = (
            float(rv.median(skipna=True))
            if not np.isnan(rv.median(skipna=True))
            else 1.0
        )
        scale = (rv / med).clip(
            lower=HEDGE_VOL_EPS_MIN, upper=HEDGE_VOL_EPS_MAX
        )
        eps2_s = (
            (HEDGE_EPS_UB2 * scale).reindex(idx).fillna(HEDGE_EPS_UB2)
        )
        eps3_s = (
            (HEDGE_EPS_UB3 * scale).reindex(idx).fillna(HEDGE_EPS_UB3)
        )
        eps4_s = (
            (HEDGE_EPS_UB4 * scale).reindex(idx).fillna(HEDGE_EPS_UB4)
        )
    else:
        eps2_s, eps3_s, eps4_s = (
            HEDGE_EPS_UB2,
            HEDGE_EPS_UB3,
            HEDGE_EPS_UB4,
        )

    # Slope gate for 1/2 escalation
    if HEDGE_SLOPE_ESCALATION:
        mid = ch_df["mid"]
        slope = (mid - mid.shift(HEDGE_SLOPE_LOOKBACK_D)).reindex(idx)
        allow_high = (slope <= HEDGE_SLOPE_THRESH).fillna(False)
    else:
        allow_high = None

    hedge_w_series = (
        build_band_hysteresis_hedge_weights(
            hedge_price,
            ch_df["ub2"],
            ch_df["ub3"],
            ch_df["ub4"],
            sma200_hedge,
            eps2=eps2_s,
            eps3=eps3_s,
            eps4=eps4_s,
            w_near=HEDGE_W_NEAR,
            w_high=HEDGE_W_HIGH,
            require_sma=True,
            allow_high=allow_high,
            min_on_days=HEDGE_MIN_HOLD_ON_DAYS,
            min_off_days=HEDGE_MIN_HOLD_OFF_DAYS,
        )
        .reindex(idx)
        .fillna(0.0)
    )
else:
    hedge_w_series = pd.Series(0.0, index=idx)

raw_close = {
    "TQQQ": tqqq_cl,
    "SPXL": spxl_cl,
    "GLD": gld_cl,
    "QLD": qld_cl,
    "SSO": sso_cl,
    "IEF": ief_cl,
}
div_series = {
    "TQQQ": div_tqqq,
    "SPXL": div_spxl,
    "GLD": div_gld,
    "QLD": div_qld,
    "SSO": div_sso,
    "IEF": div_ief,
}


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
        p = price_w.loc[dt]
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
    # Weekly snapshots anchored to 'anchor'
    sma200 = ref_ac.rolling(SMA_SLOW, min_periods=SMA_SLOW).mean()
    smamid = ref_ac.rolling(SMA_MID, min_periods=SMA_MID).mean()

    ref_w = ref_ac.resample(anchor).last()
    s200_w = sma200.resample(anchor).last()
    smid_w = smamid.resample(anchor).last()

    sig200_w = hysteresis_series(ref_w, s200_w, BAND_PCT)
    sigmid_w = hysteresis_series(ref_w, smid_w, BAND_PCT)

    # Mid fresh upcross (weekly)
    up_mid = (sigmid_w.diff() == 1).astype(int)

    # Stress on ^NDX
    ndx_ret = ndx_ac.pct_change()
    rv_d = ndx_ret.rolling(VOL_LOOKBACK_D).std() * math.sqrt(252)
    rv_w = rv_d.resample(anchor).last().reindex(ref_w.index).ffill()

    running_max_ndx = ndx_ac.rolling(
        DD_LOOKBACK_D, min_periods=max(1, int(DD_LOOKBACK_D * 0.8))
    ).max()
    dd_d = ndx_ac / running_max_ndx - 1.0
    dd_w = dd_d.resample(anchor).last().reindex(ref_w.index).ffill()

    stressed_w = (rv_w > VOL_TH_3X) | (dd_w <= -DD_TH_2X)

    # Base 3-state
    if ONE_SIDED_2X:
        if ADAPTIVE_UP_WINDOW:
            win_b = max(1, int(MID_UP_WINDOW_WEEKS_BENIGN))
            win_s = max(1, int(MID_UP_WINDOW_WEEKS_STRESSED))
            up_mid_win_b = (
                up_mid.rolling(win_b, min_periods=1)
                .max()
                .fillna(0)
                .astype(int)
            )
            up_mid_win_s = (
                up_mid.rolling(win_s, min_periods=1)
                .max()
                .fillna(0)
                .astype(int)
            )
            up_mid_window = up_mid_win_s.where(
                stressed_w, up_mid_win_b
            ).astype(int)
        else:
            up_mid_window = (
                up_mid.rolling(4, min_periods=1)
                .max()
                .fillna(0)
                .astype(int)
            )

        state_w = pd.Series(0, index=ref_w.index, dtype=int)
        state_w[(sig200_w == 0) & (up_mid_window == 1)] = 1
        state_w[sig200_w == 1] = 2
    else:
        state_w = pd.Series(0, index=ref_w.index, dtype=int)
        state_w[sigmid_w == 1] = 1
        state_w[sig200_w == 1] = 2

    # Slope gates
    if USE_SLOPE_FILTER:
        slope200_pos = (s200_w - s200_w.shift(SLOPE_LOOKBACK_W)) > 0
        slope_mid_pos = (smid_w - smid_w.shift(SLOPE_LOOKBACK_W)) > 0
        state_w[(state_w == 2) & (~slope200_pos)] = 1
        state_w[(state_w == 1) & (~slope_mid_pos)] = 0

    # Distance gate
    if USE_DIST_GATE_200:
        dist200 = ref_w / s200_w - 1.0
        if USE_ADAPTIVE_DIST:
            state_w[
                (state_w == 2)
                & stressed_w
                & (dist200 <= DELTA_200_PCT)
            ] = 1
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
    curr = int(
        exec_fri.iloc[0] if not pd.isna(exec_fri.iloc[0]) else 0
    )
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
    if (
        SIGNAL_SYMBOL in px["Adj Close"].columns
        and not px["Adj Close"][SIGNAL_SYMBOL].isnull().all()
    ):
        ref_ac = px["Adj Close"][SIGNAL_SYMBOL].dropna().reindex(idx)
    else:
        print(
            f"Warning: SIGNAL_SYMBOL {SIGNAL_SYMBOL} not found or NaN, using ^GSPC fallback."
        )
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
    swap_dates_all = set(
        idx[regime_intraday_all.ne(regime_intraday_all.shift(1))]
    )


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
        if (
            "IEF" in px["Adj Close"].columns
            and ief_ac.notna().sum() > 100
        ):
            gap_IEF, intra_IEF = seg_returns(ief_ao, ief_ac)
            gap_IEF = gap_IEF.fillna(0.0)
            intra_IEF = intra_IEF.fillna(0.0)
            HAVE_IEF = True
        else:
            gap_IEF = pd.Series(0.0, index=idx)
            intra_IEF = pd.Series(0.0, index=idx)
        mom_gld = mom(gld_ac, RISK_OFF_LOOKBACK_D).fillna(-1.0)
        mom_ief = (
            mom(ief_ac, RISK_OFF_LOOKBACK_D).fillna(-1.0)
            if HAVE_IEF
            else pd.Series(-1.0, index=idx)
        )
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
    gap_map = {
        "TQQQ": gap_TQQQ,
        "SPXL": gap_SPXL,
        "QLD": gap_QLD,
        "SSO": gap_SSO,
        "GLD": gap_GLD,
    }
    intra_map = {
        "TQQQ": intra_TQQQ,
        "SPXL": intra_SPXL,
        "QLD": intra_QLD,
        "SSO": intra_SSO,
        "GLD": intra_GLD,
    }
    if globals().get("HAVE_IEF", False):
        gap_map["IEF"] = gap_IEF
        intra_map["IEF"] = intra_IEF

    # extras (no proxies)
    gap_extra, intra_extra, ac_extra = {}, {}, {}
    extra_syms = sorted(
        lineup_syms - {"TQQQ", "SPXL", "QLD", "SSO", "GLD", "IEF"}
    )
    for sym in extra_syms:
        try:
            ao_e, ac_e, _ = get_adj_ohlc(px, sym, idx)
            g_e, i_e = seg_returns(ao_e, ac_e)
            gap_extra[sym] = g_e.fillna(0.0)
            intra_extra[sym] = i_e.fillna(0.0)
            ac_extra[sym] = ac_e
        except Exception:
            gap_extra[sym] = pd.Series(0.0, index=idx)
            intra_extra[sym] = pd.Series(0.0, index=idx)
            ac_extra[sym] = pd.Series(index=idx, dtype=float)
    gap_map.update(gap_extra)
    intra_map.update(intra_extra)

    def has_data(sym, d):
        if sym in {"TQQQ", "SPXL", "QLD", "SSO", "GLD"}:
            base_ac = {
                "TQQQ": tqqq_ac,
                "SPXL": spxl_ac,
                "QLD": qld_ac,
                "SSO": sso_ac,
                "GLD": gld_ac,
            }[sym]
            return pd.notna(base_ac.get(d, np.nan))
        if sym == "IEF":
            return globals().get("HAVE_IEF", False) and pd.notna(
                ief_ac.get(d, np.nan)
            )
        return pd.notna(
            ac_extra.get(sym, pd.Series(index=idx, dtype=float)).get(
                d, np.nan
            )
        )

    def choose_risk_off(d):
        if USE_DUAL_RISK_OFF and HAVE_IEF:
            return (
                "GLD"
                if float(mom_gld_use.get(d, -1.0))
                >= float(mom_ief_use.get(d, -1.0))
                else "IEF"
            )
        return "GLD"

    # Original selection (unchanged)
    def choose_asset(regime, d, lock_until):
        if regime == 2:  # 3x
            return "SPXL" if d <= lock_until else "TQQQ"
        elif regime == 1:  # 2x
            return "SSO" if d <= lock_until else "QLD"
        else:
            return choose_risk_off(d)

    # Availability fallback ONLY if the chosen symbol has no data (preserve original order)
    def pick_available_from_lineup(base, regime, d, lock_until):
        if regime == 2:
            prim_order = (
                ["SPXL", "TQQQ"]
                if d <= lock_until
                else ["TQQQ", "SPXL"]
            )
            tail = [
                s
                for s in RISK_ON_3X_LINEUP
                if s not in {"TQQQ", "SPXL"}
            ]
            # if locked, do not allow TQQQ anywhere
            if d <= lock_until:
                tail = [s for s in tail if s != "TQQQ"]
            candidates = (
                [base] + [p for p in prim_order if p != base] + tail
            )
        elif regime == 1:
            prim_order = (
                ["SSO", "QLD"] if d <= lock_until else ["QLD", "SSO"]
            )
            tail = [
                s
                for s in RISK_ON_2X_LINEUP
                if s not in {"QLD", "SSO"}
            ]
            if d <= lock_until:
                tail = [s for s in tail if s != "QLD"]
            candidates = (
                [base] + [p for p in prim_order if p != base] + tail
            )
        else:
            if base == "GLD":
                candidates = ["GLD"] + [
                    s for s in RISK_OFF_GOLD_LINEUP if s != "GLD"
                ]
            else:
                candidates = ["IEF"] + [
                    s for s in RISK_OFF_BOND_LINEUP if s != "IEF"
                ]
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
            curr_asset = pick_available_from_lineup(
                base, regime, d, lockout_until
            )
            entry_date = d
            leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            sched.iloc[i] = curr_asset
            continue

        if d in swap_set:
            leg_gross *= 1.0 + gap_map[curr_asset].get(d, 0.0)
            if (curr_asset in {"TQQQ", "QLD"}) and (
                leg_gross - 1.0 < 0.0
            ):
                lockout_until = d + Day(LOCKOUT_DAYS)

            base = choose_asset(regime, d, lockout_until)
            next_asset = pick_available_from_lineup(
                base, regime, d, lockout_until
            )

            curr_asset = next_asset
            entry_date = d
            leg_gross = 1.0 + intra_map[curr_asset].get(d, 0.0)
            sched.iloc[i] = curr_asset
        else:
            leg_gross *= (1.0 + gap_map[curr_asset].get(d, 0.0)) * (
                1.0 + intra_map[curr_asset].get(d, 0.0)
            )
            sched.iloc[i] = curr_asset

    return sched


# =========================
# Simulation engine
# =========================
# Use prior day's move for "stress slippage" toggle (no lookahead at open)
ref_ret_abs = (
    gspc_ac.pct_change().abs().shift(1).reindex(idx).fillna(0.0)
)


def trade_cost(d, is_hedge_adjust=False):
    if is_hedge_adjust and not HEDGE_SLIPPAGE_ON_ADJUST:
        return 0.0
    bps = (
        SLIPPAGE_BPS_STRESS
        if (
            d in ref_ret_abs.index
            and ref_ret_abs.loc[d] >= STRESS_THRESHOLD
        )
        else SLIPPAGE_BPS
    )
    return bps / 10000.0


def proxy_drag(asset, d):
    if not APPLY_PROXY_DRAGS:
        return 1.0
    if asset == "TQQQ" and bool(prox_mask_TQQQ.get(d, False)):
        if CALIBRATE_TQQQ_PROXY:
            return 1.0
        fin_d = (
            LEV_EXCESS * float(rf_daily.get(d, 0.0))
            if d in rf_daily.index
            else LEV_EXCESS * (FIN_FALLBACK_ANNUAL / 252.0)
        )
        return (1.0 - fee_daily_3x) * (1.0 - fin_d)
    if asset == "SPXL" and bool(prox_mask_SPXL.get(d, False)):
        if CALIBRATE_SPXL_PROXY:
            return 1.0
        fin_d = (
            LEV_EXCESS * float(rf_daily.get(d, 0.0))
            if d in rf_daily.index
            else LEV_EXCESS * (FIN_FALLBACK_ANNUAL / 252.0)
        )
        return (1.0 - fee_daily_3x) * (1.0 - fin_d)
    if asset == "QLD" and bool(prox_mask_QLD.get(d, False)):
        if CALIBRATE_QLD_PROXY:
            return 1.0
        fin_d = (
            LEV_EXCESS_2X * float(rf_daily.get(d, 0.0))
            if d in rf_daily.index
            else LEV_EXCESS_2X * (FIN_FALLBACK_ANNUAL / 252.0)
        )
        return (1.0 - fee_daily_2x) * (1.0 - fin_d)
    if asset == "SSO" and bool(prox_mask_SSO.get(d, False)):
        if CALIBRATE_SSO_PROXY:
            return 1.0
        fin_d = (
            LEV_EXCESS_2X * float(rf_daily.get(d, 0.0))
            if d in rf_daily.index
            else LEV_EXCESS_2X * (FIN_FALLBACK_ANNUAL / 252.0)
        )
        return (1.0 - fee_daily_2x) * (1.0 - fin_d)
    if asset == "GLD" and bool(prox_mask_GLD.get(d, False)):
        return 1.0 - gld_fee_daily
    return 1.0


def sim_strategy(run_idx, paper=False):
    # Core maps
    gap_map   = {"TQQQ": gap_TQQQ, "SPXL": gap_SPXL, "QLD": gap_QLD, "SSO": gap_SSO, "GLD": gap_GLD}
    intra_map = {"TQQQ": intra_TQQQ, "SPXL": intra_SPXL, "QLD": intra_QLD, "SSO": intra_SSO, "GLD": intra_GLD}
    if globals().get("HAVE_IEF", False):
        gap_map["IEF"] = gap_IEF
        intra_map["IEF"] = intra_IEF

    # Extra lineup symbols so schedule fallback can value correctly
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
    gap_map.update(gap_extra)
    intra_map.update(intra_extra)

    # Build schedule for the run
    sched = build_locked_schedule(run_idx)

    # Hedge weights for this run (use prior close -> apply at today's open)
    hw_all = hedge_w_series if (USE_HEDGE_OVERLAY and 'hedge_w_series' in globals()) else pd.Series(0.0, index=idx)

    # Always use your per-asset inverse series
    def _get_gap_intra_for_asset(asset: str):
        return hedge_series_for_asset(asset)

    if paper:
        eq = START_CAPITAL
        equity_curve = []
        daily_index = []
        curr_asset = None
        prev_asset = None
        prev_w = 0.0  # yesterday's hedge weight

        for i, d in enumerate(run_idx):
            next_asset = sched.loc[d]

            # First day: no gap; apply intra with today's hedge
            if curr_asset is None:
                curr_asset = next_asset
                # Decide today's hedge at prior close and cap it
                w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
                w_today = cap_w_for_asset(w_today, curr_asset)
                # Hedge slippage at open if entering weight
                if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, prev_w, atol=1e-12):
                    eq *= (1.0 - trade_cost(d, is_hedge_adjust=True))
                # INTRA (weighted, drag on main portion)
                i_m = float(intra_map[curr_asset].get(d, 0.0))
                drag = proxy_drag(curr_asset, d)
                if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
                    _, intra_h = _get_gap_intra_for_asset(curr_asset)
                    i_h = float(intra_h.get(d, 0.0))
                else:
                    i_h = 0.0
                intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
                eq *= intra_factor

                equity_curve.append(eq); daily_index.append(d)
                prev_w = w_today
                prev_asset = curr_asset
                continue

            # GAP leg: use yesterday's asset and yesterday's weight/hedge
            g_m = float(gap_map[curr_asset].get(d, 0.0))
            if USE_HEDGE_OVERLAY and (prev_asset in HEDGE_APPLIES_TO) and (prev_w > 0.0):
                gap_h, _ = _get_gap_intra_for_asset(prev_asset)
                g_h = float(gap_h.get(d, 0.0))
                w_gap = prev_w
            else:
                g_h = 0.0
                w_gap = 0.0
            gap_factor = (1.0 - w_gap) * (1.0 + g_m) + (w_gap) * (1.0 + g_h)
            eq *= gap_factor

            # Swap main asset at open if needed
            if next_asset != curr_asset:
                curr_asset = next_asset

            # Today's hedge weight (decided prior close), capped; slippage if changed vs w_gap
            w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
            w_today = cap_w_for_asset(w_today, curr_asset)
            if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, w_gap, atol=1e-12):
                eq *= (1.0 - trade_cost(d, is_hedge_adjust=True))

            # INTRA (weighted, drag on main portion)
            i_m = float(intra_map[curr_asset].get(d, 0.0))
            drag = proxy_drag(curr_asset, d)
            if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
                _, intra_h = _get_gap_intra_for_asset(curr_asset)
                i_h = float(intra_h.get(d, 0.0))
            else:
                i_h = 0.0
            intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
            eq *= intra_factor

            equity_curve.append(eq); daily_index.append(d)
            prev_w = w_today
            prev_asset = curr_asset

        return {"equity_curve": pd.Series(equity_curve, index=pd.DatetimeIndex(daily_index))}

    else:
        # REAL branch (taxes)
        eq_pre   = START_CAPITAL
        eq_after = START_CAPITAL
        equity_curve = []
        daily_index = []

        curr_asset = None
        prev_asset = None
        entry_date = None
        entry_eq_pre_after_buy = 0.0
        cum_div_leg_pre = 0.0

        realized_ST = defaultdict(float)
        realized_LT = defaultdict(float)
        carry_ST = 0.0; carry_LT = 0.0
        div_tax_by_year = defaultdict(float)

        eq_pre_prev_close = None
        prev_d = None
        prev_w = 0.0

        for i, d in enumerate(run_idx):
            y = d.year
            next_asset = sched.loc[d]

            if curr_asset is None:
                curr_asset = next_asset
                # Main trade cost to enter
                c = trade_cost(d)
                eq_pre   *= (1.0 - c)
                eq_after *= (1.0 - c)
                entry_date = d
                entry_eq_pre_after_buy = eq_pre
                cum_div_leg_pre = 0.0

                # Hedge weight (decided prior close), cap, slippage if entering
                w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
                w_today = cap_w_for_asset(w_today, curr_asset)
                if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, prev_w, atol=1e-12):
                    c = trade_cost(d, is_hedge_adjust=True)
                    eq_pre   *= (1.0 - c)
                    eq_after *= (1.0 - c)

                # INTRA weighted, drag on main portion
                i_m = float(intra_map[curr_asset].get(d, 0.0))
                drag = proxy_drag(curr_asset, d)
                if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
                    _, intra_h = _get_gap_intra_for_asset(curr_asset)
                    i_h = float(intra_h.get(d, 0.0))
                else:
                    i_h = 0.0
                intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
                eq_pre   *= intra_factor
                eq_after *= intra_factor

                equity_curve.append(eq_after); daily_index.append(d)
                eq_pre_prev_close = eq_pre
                prev_d = d
                prev_w = w_today
                prev_asset = curr_asset
                continue

            # Dividends (main asset only)
            div_amt = float(div_series.get(curr_asset, pd.Series(0.0, index=run_idx)).loc[d])
            if div_amt != 0.0 and prev_d is not None:
                prev_close_raw = raw_close[curr_asset].loc[prev_d]
                if pd.notna(prev_close_raw) and prev_close_raw != 0.0 and eq_pre_prev_close is not None:
                    div_cash = float(eq_pre_prev_close) * (div_amt / float(prev_close_raw))
                    div_tax  = DIV_TAX_RATE * div_cash
                    eq_after -= div_tax
                    div_tax_by_year[y] += div_tax
                    cum_div_leg_pre += div_cash

            # GAP weighted using yesterday's weight and yesterday's asset-matched hedge
            g_m = float(gap_map[curr_asset].get(d, 0.0))
            if USE_HEDGE_OVERLAY and (prev_asset in HEDGE_APPLIES_TO) and (prev_w > 0.0):
                gap_h, _ = _get_gap_intra_for_asset(prev_asset)
                g_h = float(gap_h.get(d, 0.0))
                w_gap = prev_w
            else:
                g_h = 0.0
                w_gap = 0.0
            gap_factor = (1.0 - w_gap) * (1.0 + g_m) + (w_gap) * (1.0 + g_h)
            eq_pre   *= gap_factor
            eq_after *= gap_factor

            # Swap main asset at open
            if next_asset != curr_asset:
                # Exit cost
                c = trade_cost(d)
                eq_pre   *= (1.0 - c)
                eq_after *= (1.0 - c)

                # Realize gains/losses on the leg
                hold_days = (d - entry_date).days
                realized_pre = eq_pre - entry_eq_pre_after_buy - cum_div_leg_pre
                if hold_days > 365:
                    realized_LT[y] += realized_pre
                else:
                    realized_ST[y] += realized_pre

                # Enter new leg
                curr_asset = next_asset
                c = trade_cost(d)
                eq_pre   *= (1.0 - c)
                eq_after *= (1.0 - c)

                entry_date = d
                entry_eq_pre_after_buy = eq_pre
                cum_div_leg_pre = 0.0

            # Today's hedge weight (prior close), cap, slippage if changed vs w_gap
            w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
            w_today = cap_w_for_asset(w_today, curr_asset)
            if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, w_gap, atol=1e-12):
                c = trade_cost(d, is_hedge_adjust=True)
                eq_pre   *= (1.0 - c)
                eq_after *= (1.0 - c)

            # INTRA weighted, drag on main portion
            i_m = float(intra_map[curr_asset].get(d, 0.0))
            drag = proxy_drag(curr_asset, d)
            if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
                _, intra_h = _get_gap_intra_for_asset(curr_asset)
                i_h = float(intra_h.get(d, 0.0))
            else:
                i_h = 0.0
            intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
            eq_pre   *= intra_factor
            eq_after *= intra_factor

            # Year-end taxes
            is_year_end = (i == len(run_idx) - 1) or (run_idx[i + 1].year != y)
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
            prev_w = w_today
            prev_asset = curr_asset

        equity_curve = pd.Series(equity_curve, index=pd.DatetimeIndex(daily_index))
        return {"equity_curve": equity_curve}


# Small helper: pick the correct inverse series for the asset
def _get_gap_intra_for_asset(asset: str):
    return hedge_series_for_asset(asset)

    if paper:
        eq = START_CAPITAL
        equity_curve = []
        daily_index = []
        curr_asset = None
        prev_asset = None
        prev_w = 0.0  # yesterday's hedge weight

        for i, d in enumerate(run_idx):
            next_asset = sched.loc[d]

            # First day: no gap; apply intra with today's hedge
            if curr_asset is None:
                curr_asset = next_asset
                # Decide today's hedge at prior close and cap it
                w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
                w_today = cap_w_for_asset(w_today, curr_asset)
                # Hedge slippage at open if entering weight
                if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, prev_w, atol=1e-12):
                    eq *= (1.0 - trade_cost(d, is_hedge_adjust=True))
                # INTRA (weighted, drag on main portion)
                i_m = float(intra_map[curr_asset].get(d, 0.0))
                drag = proxy_drag(curr_asset, d)
                if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO):
                    _, intra_h = _get_gap_intra_for_asset(curr_asset)
                    i_h = float(intra_h.get(d, 0.0))
                else:
                    i_h = 0.0
                intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
                eq *= intra_factor

                equity_curve.append(eq); daily_index.append(d)
                prev_w = w_today
                prev_asset = curr_asset
                continue

            # GAP leg: use yesterday's asset and yesterday's weight/hedge
            g_m = float(gap_map[curr_asset].get(d, 0.0))
            if USE_HEDGE_OVERLAY and (prev_asset in HEDGE_APPLIES_TO) and (prev_w > 0.0):
                gap_h, _ = _get_gap_intra_for_asset(prev_asset)
                g_h = float(gap_h.get(d, 0.0))
                w_gap = prev_w
            else:
                g_h = 0.0
                w_gap = 0.0
            gap_factor = (1.0 - w_gap) * (1.0 + g_m) + (w_gap) * (1.0 + g_h)
            eq *= gap_factor

            # Swap main asset at open if needed
            if next_asset != curr_asset:
                curr_asset = next_asset

            # Today's hedge weight (decided prior close), capped; slippage if changed vs w_gap
            w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
            w_today = cap_w_for_asset(w_today, curr_asset)
            if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, w_gap, atol=1e-12):
                eq *= (1.0 - trade_cost(d, is_hedge_adjust=True))

            # INTRA (weighted, drag on main portion)
            i_m = float(intra_map[curr_asset].get(d, 0.0))
            drag = proxy_drag(curr_asset, d)
            if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
                _, intra_h = _get_gap_intra_for_asset(curr_asset)
                i_h = float(intra_h.get(d, 0.0))
            else:
                i_h = 0.0
            intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
            eq *= intra_factor

            equity_curve.append(eq); daily_index.append(d)
            prev_w = w_today
            prev_asset = curr_asset

        return {"equity_curve": pd.Series(equity_curve, index=pd.DatetimeIndex(daily_index))}

    # REAL branch (taxes)
    eq_pre   = START_CAPITAL
    eq_after = START_CAPITAL
    equity_curve = []
    daily_index = []

    curr_asset = None
    prev_asset = None
    entry_date = None
    entry_eq_pre_after_buy = 0.0
    cum_div_leg_pre = 0.0

    realized_ST = defaultdict(float)
    realized_LT = defaultdict(float)
    carry_ST = 0.0; carry_LT = 0.0
    div_tax_by_year = defaultdict(float)

    eq_pre_prev_close = None
    prev_d = None
    prev_w = 0.0

    for i, d in enumerate(run_idx):
        y = d.year
        next_asset = sched.loc[d]

        if curr_asset is None:
            curr_asset = next_asset
            # Main trade cost to enter
            c = trade_cost(d)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)
            entry_date = d
            entry_eq_pre_after_buy = eq_pre
            cum_div_leg_pre = 0.0

            # Hedge weight (decided prior close), cap, slippage if entering
            w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
            w_today = cap_w_for_asset(w_today, curr_asset)
            if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, prev_w, atol=1e-12):
                c = trade_cost(d, is_hedge_adjust=True)
                eq_pre   *= (1.0 - c)
                eq_after *= (1.0 - c)

            # INTRA weighted, drag on main portion
            i_m = float(intra_map[curr_asset].get(d, 0.0))
            drag = proxy_drag(curr_asset, d)
            if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
                _, intra_h = _get_gap_intra_for_asset(curr_asset)
                i_h = float(intra_h.get(d, 0.0))
            else:
                i_h = 0.0
            intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
            eq_pre   *= intra_factor
            eq_after *= intra_factor

            equity_curve.append(eq_after); daily_index.append(d)
            eq_pre_prev_close = eq_pre
            prev_d = d
            prev_w = w_today
            prev_asset = curr_asset
            continue

        # Dividends (main asset only)
        div_amt = float(div_series.get(curr_asset, pd.Series(0.0, index=run_idx)).loc[d])
        if div_amt != 0.0 and prev_d is not None:
            prev_close_raw = raw_close[curr_asset].loc[prev_d]
            if pd.notna(prev_close_raw) and prev_close_raw != 0.0 and eq_pre_prev_close is not None:
                div_cash = float(eq_pre_prev_close) * (div_amt / float(prev_close_raw))
                div_tax  = DIV_TAX_RATE * div_cash
                eq_after -= div_tax
                div_tax_by_year[y] += div_tax
                cum_div_leg_pre += div_cash

        # GAP weighted using yesterday's weight and yesterday's asset-matched hedge
        g_m = float(gap_map[curr_asset].get(d, 0.0))
        if USE_HEDGE_OVERLAY and (prev_asset in HEDGE_APPLIES_TO) and (prev_w > 0.0):
            gap_h, _ = _get_gap_intra_for_asset(prev_asset)
            g_h = float(gap_h.get(d, 0.0))
            w_gap = prev_w
        else:
            g_h = 0.0
            w_gap = 0.0
        gap_factor = (1.0 - w_gap) * (1.0 + g_m) + (w_gap) * (1.0 + g_h)
        eq_pre   *= gap_factor
        eq_after *= gap_factor

        # Swap main asset at open
        if next_asset != curr_asset:
            # Exit cost
            c = trade_cost(d)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)

            # Realize gains/losses on the leg
            hold_days = (d - entry_date).days
            realized_pre = eq_pre - entry_eq_pre_after_buy - cum_div_leg_pre
            if hold_days > 365:
                realized_LT[y] += realized_pre
            else:
                realized_ST[y] += realized_pre

            # Enter new leg
            curr_asset = next_asset
            c = trade_cost(d)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)

            entry_date = d
            entry_eq_pre_after_buy = eq_pre
            cum_div_leg_pre = 0.0

        # Today's hedge weight (prior close), cap, slippage if changed vs w_gap
        w_today = float(hw_all.shift(1).get(d, 0.0)) if (USE_HEDGE_OVERLAY and curr_asset in HEDGE_APPLIES_TO) else 0.0
        w_today = cap_w_for_asset(w_today, curr_asset)
        if USE_HEDGE_OVERLAY and HEDGE_SLIPPAGE_ON_ADJUST and not np.isclose(w_today, w_gap, atol=1e-12):
            c = trade_cost(d, is_hedge_adjust=True)
            eq_pre   *= (1.0 - c)
            eq_after *= (1.0 - c)

        # INTRA weighted, drag on main portion
        i_m = float(intra_map[curr_asset].get(d, 0.0))
        drag = proxy_drag(curr_asset, d)
        if USE_HEDGE_OVERLAY and (curr_asset in HEDGE_APPLIES_TO) and (w_today > 0.0):
            _, intra_h = _get_gap_intra_for_asset(curr_asset)
            i_h = float(intra_h.get(d, 0.0))
        else:
            i_h = 0.0
        intra_factor = (1.0 - w_today) * (drag * (1.0 + i_m)) + (w_today) * (1.0 + i_h)
        eq_pre   *= intra_factor
        eq_after *= intra_factor

        # Year-end taxes
        is_year_end = (i == len(run_idx) - 1) or (run_idx[i + 1].year != y)
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
        prev_w = w_today
        prev_asset = curr_asset

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
    cagr = cagr_from_curve(curve)
    mdd = max_drawdown_from_curve(curve)
    vol = annualized_vol_from_curve(curve)
    shrp = sharpe_from_curve(curve)
    sort = sortino_from_curve(curve)
    calm = calmar_from_curve(curve)
    yr = year_returns_from_curve(curve)
    best_y = float(yr.max()) if not yr.empty else np.nan
    worst_y = float(yr.min()) if not yr.empty else np.nan
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
        "WinRateYears": win_rate,
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
    print(
        f"  Sharpe: {m['Sharpe']:.2f}  |  Sortino: {m['Sortino']:.2f}  |  Calmar: {m['Calmar']:.2f}"
    )
    print(
        f"  Best Year: {fmt_pct(m['BestYear'])}  |  Worst Year: {fmt_pct(m['WorstYear'])}  |  % Winning Years: {100.0*m['WinRateYears']:.1f}%"
    )
    print()


def run_segment(name, start, end):
    run_idx = build_run_index(start, end)
    if len(run_idx) == 0:
        print(f"{name}: no run days (check dates/SMA warm-up).")
        return None, None, None
    res_paper = sim_strategy(run_idx, paper=True)
    res_real = sim_strategy(run_idx, paper=False)
    curve_paper = res_paper["equity_curve"]
    curve_real = res_real["equity_curve"]
    m_paper = metrics_from_curve(curve_paper)
    m_real = metrics_from_curve(curve_real)

    print(f"\n--- Yearly Returns for {name} ---")
    yr_paper = year_returns_from_curve(curve_paper).apply(fmt_pct)
    yr_real = year_returns_from_curve(curve_real).apply(fmt_pct)
    df_yr = pd.DataFrame({"Paper": yr_paper, "Real": yr_real})
    df_yr.index.name = "Year"
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
                reg_w.loc[dt] = 0
                continue
            p = ref_w.loc[dt]
            up = (1 + BAND_PCT) * m
            dn = (1 - BAND_PCT) * m
            curr = (
                1
                if (prev == 0 and p > up)
                else (0 if (prev == 1 and p < dn) else prev)
            )
            reg_w.loc[dt] = curr
            prev = curr
        return reg_w.replace({1: 2, 0: 0}).astype(int)

    state_thu_b = baseline_state("W-THU")
    state_fri_b = baseline_state("W-FRI")
    exec_thu_b = _exec_next_bd(state_thu_b)
    exec_fri_b = _exec_next_bd(state_fri_b)
    sig = _combine_thu_fri_exec(exec_thu_b, exec_fri_b)
    swap = set(idx[sig.ne(sig.shift(1))])
    return swap, sig


def run_pure_baseline_segment(name, start, end):
    run_idx = build_run_index(start, end)
    if len(run_idx) == 0:
        print(
            f"{name} (Baseline): no run days (check dates/SMA warm-up)."
        )
        return None, None, None

    # Save globals
    saved_swap, saved_reg = globals().get(
        "swap_dates_all", None
    ), globals().get("regime_intraday_all", None)

    # Build baseline dual-day signal and splice
    swap_b, sig_b = build_regime_by_SMA_baseline_dual()
    globals()["swap_dates_all"], globals()["regime_intraday_all"] = (
        swap_b,
        sig_b,
    )

    try:
        res_paper = sim_strategy(run_idx, paper=True)
        res_real = sim_strategy(run_idx, paper=False)
    finally:
        # Restore full 3-state regime
        (
            globals()["swap_dates_all"],
            globals()["regime_intraday_all"],
        ) = (saved_swap, saved_reg)
        rebuild_regime()

    curve_paper = res_paper["equity_curve"]
    curve_real = res_real["equity_curve"]
    m_paper = metrics_from_curve(curve_paper)
    m_real = metrics_from_curve(curve_real)

    print(f"\n--- Yearly Returns for {name} (Baseline) ---")
    yr_paper = year_returns_from_curve(curve_paper).apply(fmt_pct)
    yr_real = year_returns_from_curve(curve_real).apply(fmt_pct)
    df_yr = pd.DataFrame({"Paper": yr_paper, "Real": yr_real})
    df_yr.index.name = "Year"
    if not df_yr.empty:
        print(df_yr.to_string())
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
            sp_curve_holdout = START_CAPITAL * (
                sp_ac_holdout / sp_ac_holdout.iloc[0]
            )
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
def walk_forward_validation(
    oos_start,
    oos_end,
    train_years,
    test_years,
    paper=True,
    chain_capital=True,
):
    oos_start = pd.to_datetime(oos_start)
    oos_end = pd.to_datetime(oos_end)

    tests = []
    curves_scaled = []
    eq_level = START_CAPITAL

    t0 = oos_start
    while True:
        train_end = t0 + pd.DateOffset(years=train_years) - Day(1)
        test_start = train_end + Day(1)
        test_end = (
            test_start + pd.DateOffset(years=test_years) - Day(1)
        )
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
        tests.append(
            (test_start.date(), test_end.date(), m, curve_scaled)
        )
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
    gap_map = {
        "TQQQ": gap_TQQQ,
        "SPXL": gap_SPXL,
        "QLD": gap_QLD,
        "SSO": gap_SSO,
        "GLD": gap_GLD,
    }
    intra_map = {
        "TQQQ": intra_TQQQ,
        "SPXL": intra_SPXL,
        "QLD": intra_QLD,
        "SSO": intra_SSO,
        "GLD": intra_GLD,
    }
    if globals().get("HAVE_IEF", False):
        gap_map["IEF"] = gap_IEF
        intra_map["IEF"] = intra_IEF
    # extras (no proxies)
    extra_syms = sorted(
        lineup_syms - {"TQQQ", "SPXL", "QLD", "SSO", "GLD", "IEF"}
    )
    for sym in extra_syms:
        try:
            ao_e, ac_e, _ = get_adj_ohlc(px, sym, idx)
            g_e, i_e = seg_returns(ao_e, ac_e)
            gap_map[sym] = g_e.fillna(0.0)
            intra_map[sym] = i_e.fillna(0.0)
        except Exception:
            gap_map[sym] = pd.Series(0.0, index=idx)
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
            leg_gross *= 1.0 + float(gap_map[curr].get(d, 0.0))
            legs.append(
                {
                    "Entry": entry,
                    "Exit": d,
                    "Asset": curr,
                    "Next": nxt,
                    "ReturnPct": leg_gross - 1.0,
                    "Days": (d - entry).days,
                }
            )
            # new leg
            curr = nxt
            entry = d
            leg_gross = 1.0 + float(intra_map[curr].get(d, 0.0))
        else:
            # hold day
            leg_gross *= (1.0 + float(gap_map[curr].get(d, 0.0))) * (
                1.0 + float(intra_map[curr].get(d, 0.0))
            )

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


def weekly_signal_dashboard(
    window_start="2019-01-01", lookback_legs=12, now_ts=None
):
    # Pick "today" on the market calendar (roll back if weekend/holiday)
    now_ts = (
        pd.Timestamp.now(tz=None)
        if now_ts is None
        else pd.Timestamp(now_ts)
    )
    today = now_ts.normalize()
    if today not in idx:
        pos = idx.searchsorted(today, side="right") - 1
        today = idx[max(0, pos)]

    # Build schedule/legs up to today (for context and lock propagation)
    run_idx_hist = idx[
        (idx >= pd.to_datetime(window_start)) & (idx <= today)
    ]
    sched_hist, legs_hist = build_schedule_and_legs(run_idx_hist)

    # Was a decision made today (Thu or Fri close)?
    made_thu = (len(state_thu) >= 2) and (
        state_thu.index[-1].normalize() == today
    )
    made_fri = (len(state_fri) >= 2) and (
        state_fri.index[-1].normalize() == today
    )

    if not (made_thu or made_fri):
        print("Decided: NO (run after-hours on Thursday or Friday).")
        curr = sched_hist.iloc[-1] if not sched_hist.empty else None
        print(f"Current holding: {curr if curr else 'n/a'}")

        # Next decision opportunities from 'today'
        pos = idx.searchsorted(today, side="right")
        next_thu = None
        next_fri = None
        while pos < len(idx) and (
            next_thu is None or next_fri is None
        ):
            d = idx[pos]
            if d.weekday() == 3 and next_thu is None:  # Thursday
                next_thu = d
            if d.weekday() == 4 and next_fri is None:  # Friday
                next_fri = d
            pos += 1

        if next_thu is not None:
            print(
                f"Next Thu decision: {next_thu.date()}  -> Exec: {next_trading_day(next_thu).date()} (open) | Symbol: TBD"
            )
        if next_fri is not None:
            print(
                f"Next Fri decision: {next_fri.date()}  -> Exec: {next_trading_day(next_fri).date()} (open) | Symbol: TBD"
            )

        # Recent completed legs
        if legs_hist is not None and not legs_hist.empty:
            legs_tail = legs_hist.tail(lookback_legs).copy()
            legs_tail["Return%"] = (
                legs_tail["ReturnPct"] * 100.0
            ).round(2)
            cols = [
                "Entry",
                "Exit",
                "Asset",
                "Next",
                "Days",
                "Return%",
            ]
            print("\nRecent completed legs (open->open):")
            print(legs_tail[cols].to_string(index=False))
        else:
            print("\nNo completed legs in window.")
        print(
            f"Has today’s bar? {idx[-1].normalize() == pd.Timestamp.now().normalize()}"
        )
        return

    # Decision made today
    anchor = "Thursday" if made_thu else "Friday"
    exec_day = next_trading_day(today)

    # Build schedule through exec_day to propagate lock state and read the chosen symbol
    run_idx_until_exec = idx[idx <= exec_day]
    sched_upto, _ = build_schedule_and_legs(run_idx_until_exec)
    suggested = (
        sched_upto.loc[exec_day]
        if exec_day in sched_upto.index
        else None
    )

    print("===== Decision/Execution Plan =====")
    print(f"Decided: YES ({anchor} close {today.date()})")
    print(
        f"Entry:   {exec_day.date()} (open) | Suggested symbol: {suggested if suggested else 'TBD'}"
    )
    # Earliest next decision/exec window
    next_decision = (
        _find_next_trading_day_with_weekday(today, 4)
        if made_thu
        else _find_next_trading_day_with_weekday(today, 3)
    )
    next_exec = (
        next_trading_day(next_decision)
        if next_decision is not None
        else None
    )
    if next_decision is not None and next_exec is not None:
        print(
            f"Decided Exit (earliest): decision {next_decision.date()} (close) -> execution {next_exec.date()} (open)"
        )
    else:
        print("Decided Exit (earliest): n/a")

    # Weekly summary and recent legs
    fridays_hist = pd.DatetimeIndex(
        [d for d in run_idx_hist if d.weekday() == 4]
    )
    if not fridays_hist.empty:
        applied = sched_hist.reindex(fridays_hist).dropna()
        if not applied.empty:
            print(
                f"\nWeek summary (from Friday open {applied.index[-1].date()}): Target was {applied.iloc[-1]}"
            )

    if legs_hist is not None and not legs_hist.empty:
        legs_tail = legs_hist.tail(12).copy()
        legs_tail["Return%"] = (legs_tail["ReturnPct"] * 100.0).round(
            2
        )
        cols = ["Entry", "Exit", "Asset", "Next", "Days", "Return%"]
        print("\nRecent completed legs (open->open):")
        print(legs_tail[cols].to_string(index=False))
    else:
        print("\nNo completed legs in window.")

    print(
        f"\nHas today’s bar? {idx[-1].normalize() == pd.Timestamp.now().normalize()}"
    )
    print(
        "\nRepainting: NO (decisions fixed at Thu/Fri close; execution next trading day)."
    )
    print("Lagging: 1 business day by design (Thu→Fri and Fri→Mon).")
    print("==============================================\n")


# =========================
# EXECUTION LOGIC
# =========================

# ===== OOS comparison (2016–today) =====
print("\n===== OOS comparison (2016–today) =====")
OOS = ("2016-01-01", None)
run_idx_oos = build_run_index(OOS[0], OOS[1])

def print_hedge_stats(run_idx, tag):
    try:
        w = hedge_w_series.reindex(run_idx).fillna(0.0)
        on_pct = 100.0 * float((w > 0).mean())
        at_half = 100.0 * float((w >= 0.5).mean())
        flips = int(w.ne(w.shift(1)).sum())
        print(f"{tag} | Hedge on: {on_pct:.2f}% | At 1/2: {at_half:.2f}% | Flips: {flips}")
    except Exception as e:
        print(f"{tag} | Hedge stats error: {e}")

# With overlay
USE_HEDGE_OVERLAY = True
res_on_p = sim_strategy(run_idx_oos, paper=True)["equity_curve"]
res_on_r = sim_strategy(run_idx_oos, paper=False)["equity_curve"]
m_on_p = metrics_from_curve(res_on_p); m_on_r = metrics_from_curve(res_on_r)
print_metrics("OOS (Paper, overlay ON)", m_on_p)
print_metrics("OOS (Real, overlay ON)",  m_on_r)
print_hedge_stats(run_idx_oos, "Overlay ON")

# Without overlay
USE_HEDGE_OVERLAY = False
res_off_p = sim_strategy(run_idx_oos, paper=True)["equity_curve"]
res_off_r = sim_strategy(run_idx_oos, paper=False)["equity_curve"]
m_off_p = metrics_from_curve(res_off_p); m_off_r = metrics_from_curve(res_off_r)
print_metrics("OOS (Paper, overlay OFF)", m_off_p)
print_metrics("OOS (Real, overlay OFF)",  m_off_r)

# Restore flag for any later runs
USE_HEDGE_OVERLAY = True
# After a run (for quick diagnostics)
try:
    print("Hedge on % of days:", float((hedge_w_series > 0).mean())*100, "%")
    print("At 1/2 % of days:", float((hedge_w_series >= 0.5).mean())*100, "%")
    flips = (hedge_w_series.ne(hedge_w_series.shift(1))).sum()
    print("Hedge flips (any weight change):", int(flips))
except Exception:
    pass

if not DO_WALK_FORWARD:
    # 1) Pure baseline on HOLDOUT (dual-day)
    print(
        "\n===== Baseline Strategy (2-State, dual-day: Thu->Fri and Fri->Mon) ====="
    )
    bp, br, bm = run_pure_baseline_segment(
        "Holdout (Baseline)", HOLDOUT[0], HOLDOUT[1]
    )
    if bm:
        print_metrics("Baseline Holdout (Paper)", bm[0])
        print_metrics("Baseline Holdout (Real)", bm[1])

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
    print(
        f"\n===== NEW STRATEGY (3-state dual-day on {SIGNAL_SYMBOL}, SMA {SMA_SLOW}/{SMA_MID}) ====="
    )
    print(
        f"Up-Window: {MID_UP_WINDOW_WEEKS_BENIGN}w (benign) / {MID_UP_WINDOW_WEEKS_STRESSED}w (stressed)"
    )
    print(
        f"Slope Filter: {USE_SLOPE_FILTER} ({SLOPE_LOOKBACK_W}w), Dual Risk-Off: {USE_DUAL_RISK_OFF} ({RISK_OFF_LOOKBACK_D}d)"
    )
    print(
        f"Dist. Gate: {USE_DIST_GATE_200} ({DELTA_200_PCT*100:.1f}%), Adaptive: {USE_ADAPTIVE_DIST}, Vol Cap: {USE_VOL_CAP}, DD Throttle: {USE_DD_THROTTLE}"
    )
    print(f"Wash Sale Lockout (Nasdaq family): {LOCKOUT_DAYS} days\n")

    cp_hold, cr_hold, m_hold = run_segment(
        "Holdout", HOLDOUT[0], HOLDOUT[1]
    )
    if m_hold:
        print_metrics("Holdout (Paper)", m_hold[0])
        print_metrics("Holdout (Paper)", m_hold[0])
        print_metrics("Holdout (Real)", m_hold[1])

    # 4) Dashboard-style summary (intended for Thu/Fri after-hours)
    weekly_signal_dashboard(
        window_start="2019-01-01", lookback_legs=20
    )

else:
    wf_curve, wf_tests = walk_forward_validation(
        WF_START_DATE,
        WF_END_DATE,
        WF_TRAIN_YEARS,
        WF_TEST_YEARS,
        paper=WF_USE_PAPER,
        chain_capital=WF_CHAIN_CAPITAL,
    )
    if wf_curve is not None and not wf_curve.empty:
        m_wf = metrics_from_curve(wf_curve)
        simMode = (
            "Paper (No Taxes/Fees)"
            if WF_USE_PAPER
            else "Real (With Taxes/Fees)"
        )
        print(
            f"===== Walk-Forward Validation ({simMode}, chained, {WF_START_DATE} to {WF_END_DATE}) ====="
        )
        for ts, te, m, _ in wf_tests:
            print_metrics(f"Test window {ts} → {te}", m)
        print_metrics("Aggregated WF tests (chained)", m_wf)
    else:
        print(
            f"===== Walk-Forward Validation ({WF_START_DATE} to {WF_END_DATE}) ====="
        )
        print("No data available for this period or configuration.")
        print(
            f"Check SMA warm-up and if WF_TRAIN_YEARS ({WF_TRAIN_YEARS}) is too long for the period."
        )
        
print("run_idx_oos len:", len(run_idx_oos))
res = sim_strategy(run_idx_oos, paper=True)
print("sim_strategy(paper) is None?", res is None)
if res:
    print("curve len:", len(res["equity_curve"]))