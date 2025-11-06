# Dual Ticker Hedge Strategy
# Use one ticker (e.g., SPY) as confirmation filter for another (e.g., TQQQ)

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc, compute_std_lines_strict,
    N, STDEV_PERIOD, PRICE_SRC, DDOF, SLIPPAGE_BPS
)

def get_band_value(df: pd.DataFrame, date, band_spec):
    """
    Get band value, supporting fractional bands like UB3.5, LB2.5, etc.
    
    Args:
        df: DataFrame with band columns
        date: Date index
        band_spec: Band specification (e.g., "UB3", "UB3.5", "mid", "LB2.5")
    
    Returns:
        Float value of the band
    """
    if band_spec == "mid":
        return float(df.loc[date, "mid"])
    
    # Check if it's a fractional band (e.g., "UB3.5" or "LB2.5")
    if "." in band_spec:
        # Parse the band type and level
        if band_spec.startswith("UB"):
            band_type = "UB"
            level = float(band_spec[2:])
        elif band_spec.startswith("LB"):
            band_type = "LB"
            level = float(band_spec[2:])
        else:
            raise ValueError(f"Invalid band specification: {band_spec}")
        
        # Get the two integer bands to interpolate between
        lower_level = int(np.floor(level))
        upper_level = int(np.ceil(level))
        
        if lower_level == upper_level:
            # It's actually an integer
            return float(df.loc[date, f"{band_type}{lower_level}"])
        
        # Interpolate between the two bands
        lower_band = float(df.loc[date, f"{band_type}{lower_level}"])
        upper_band = float(df.loc[date, f"{band_type}{upper_level}"])
        
        # Linear interpolation
        fraction = level - lower_level
        interpolated = lower_band + fraction * (upper_band - lower_band)
        
        return interpolated
    else:
        # Standard integer band (e.g., "UB3", "LB2")
        return float(df.loc[date, band_spec])

def calculate_position_drawdown(std_df: pd.DataFrame, entry_date, exit_date, entry_price, side):
    """Calculate max drawdown for a single position during its holding period."""
    try:
        entry_idx = std_df.index.get_loc(pd.Timestamp(entry_date))
        exit_idx = std_df.index.get_loc(pd.Timestamp(exit_date))
        holding_period = std_df.iloc[entry_idx:exit_idx+1]
        
        if side == "long":
            worst_price = holding_period["low"].min()
            drawdown = (worst_price - entry_price) / entry_price
        else:
            worst_price = holding_period["high"].max()
            drawdown = (entry_price - worst_price) / entry_price
        
        return drawdown
    except:
        return 0.0

def build_dual_ticker_hedge_trades(
    filter_df: pd.DataFrame,  # e.g., SPY
    trade_df: pd.DataFrame,   # e.g., TQQQ
    filter_condition: dict,   # {"min_band": "mid", "max_band": "UB2"}
    trade_entry_band: str,    # e.g., "UB3" or "UB3.5"
    trade_exit_band: str,     # e.g., "UB2" or "UB1" or "UB2.5"
    side: str = "short",      # "short" for hedging (selling)
    slippage_bps: float = SLIPPAGE_BPS
) -> pd.DataFrame:
    """
    Build trades based on dual ticker confirmation.
    
    Args:
        filter_df: DataFrame with channels for filter ticker (e.g., SPY)
        trade_df: DataFrame with channels for trade ticker (e.g., TQQQ)
        filter_condition: Dict with "min_band" and "max_band" for filter ticker
                         e.g., {"min_band": "mid", "max_band": "UB2"} means SPY between Midline and UB2
        trade_entry_band: Band name for entry signal on trade ticker (e.g., "UB3")
        trade_exit_band: Band name for exit signal on trade ticker (e.g., "UB2")
        side: "short" for hedging (selling), "long" for buying
    
    Returns:
        DataFrame with trades
    """
    # Align the two dataframes by date
    common_dates = filter_df.index.intersection(trade_df.index)
    filter_aligned = filter_df.loc[common_dates].copy()
    trade_aligned = trade_df.loc[common_dates].copy()
    
    # Drop rows with missing data
    filter_aligned = filter_aligned.dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "UB4", "LB1", "LB2", "LB3", "LB4"])
    trade_aligned = trade_aligned.dropna(subset=["open", "close", "mid", "UB1", "UB2", "UB3", "UB4", "LB1", "LB2", "LB3", "LB4"])
    
    # Re-align after dropping
    common_dates = filter_aligned.index.intersection(trade_aligned.index)
    filter_aligned = filter_aligned.loc[common_dates]
    trade_aligned = trade_aligned.loc[common_dates]
    
    idx = list(common_dates)
    trades = []
    bps = slippage_bps / 10000.0
    
    in_trade = False
    i = 0
    
    while i < len(idx) - 1:
        if in_trade:
            i += 1
            continue
        
        current_date = idx[i]
        
        # Check filter condition (e.g., SPY between mid and UB2)
        filter_close = float(filter_aligned.loc[current_date, "close"])
        filter_min = get_band_value(filter_aligned, current_date, filter_condition["min_band"])
        filter_max = get_band_value(filter_aligned, current_date, filter_condition["max_band"])
        
        filter_ok = (filter_close >= filter_min) and (filter_close <= filter_max)
        
        if not filter_ok:
            i += 1
            continue
        
        # Check trade entry condition (e.g., TQQQ closes above UB3 or UB3.5)
        trade_close = float(trade_aligned.loc[current_date, "close"])
        trade_entry_level = get_band_value(trade_aligned, current_date, trade_entry_band)
        
        if side == "short":
            entry_signal = (trade_close >= trade_entry_level)
        else:
            entry_signal = (trade_close <= trade_entry_level)
        
        if not entry_signal:
            i += 1
            continue
        
        # Enter trade at next day's open
        j = i + 1
        entry_date = idx[j]
        entry_open = float(trade_aligned.loc[entry_date, "open"])
        
        if side == "short":
            entry_price = entry_open * (1.0 - bps)
        else:
            entry_price = entry_open * (1.0 + bps)
        
        in_trade = True
        
        # Search for exit (e.g., TQQQ closes at or below UB2)
        exit_date = None
        exit_price = None
        exit_reason = None
        
        for k in range(j, len(idx)):
            exit_check_date = idx[k]
            trade_close_k = float(trade_aligned.loc[exit_check_date, "close"])
            trade_exit_level = get_band_value(trade_aligned, exit_check_date, trade_exit_band)
            
            if side == "short":
                # Exit when price drops to or below exit band
                if trade_close_k <= trade_exit_level:
                    if k + 1 < len(idx):
                        exit_date = idx[k+1]
                        exit_open = float(trade_aligned.loc[exit_date, "open"])
                        exit_price = exit_open * (1.0 + bps)
                        i = k + 1
                    else:
                        exit_date = exit_check_date
                        exit_price = trade_close_k * (1.0 + bps)
                        i = k
                    exit_reason = f"close_reach_{trade_exit_band}"
                    break
            else:
                # Exit when price rises to or above exit band
                if trade_close_k >= trade_exit_level:
                    if k + 1 < len(idx):
                        exit_date = idx[k+1]
                        exit_open = float(trade_aligned.loc[exit_date, "open"])
                        exit_price = exit_open * (1.0 - bps)
                        i = k + 1
                    else:
                        exit_date = exit_check_date
                        exit_price = trade_close_k * (1.0 - bps)
                        i = k
                    exit_reason = f"close_reach_{trade_exit_band}"
                    break
        
        # Force exit if never found
        if exit_date is None:
            last_date = idx[-1]
            exit_date = last_date
            if side == "short":
                exit_price = float(trade_aligned.loc[last_date, "close"]) * (1.0 + bps)
            else:
                exit_price = float(trade_aligned.loc[last_date, "close"]) * (1.0 - bps)
            exit_reason = "force_exit"
            i = len(idx) - 1
        
        in_trade = False
        
        # Calculate return
        ret = (entry_price / exit_price - 1.0) if side == "short" else (exit_price / entry_price - 1.0)
        position_dd = calculate_position_drawdown(trade_aligned, entry_date, exit_date, entry_price, side)
        
        # Capture filter ticker state at entry
        filter_close_entry = float(filter_aligned.loc[entry_date, "close"])
        filter_mid_entry = float(filter_aligned.loc[entry_date, "mid"])
        
        # Capture trade ticker state at entry
        trade_mid_entry = float(trade_aligned.loc[entry_date, "mid"])
        trade_slope_entry = float(trade_aligned.loc[entry_date, "mid_slope"]) if pd.notna(trade_aligned.loc[entry_date, "mid_slope"]) else float("nan")
        
        trades.append({
            "signal_date": current_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "ret": ret,
            "position_max_dd": position_dd,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
            "filter_close_entry": filter_close_entry,
            "filter_mid_entry": filter_mid_entry,
            "trade_mid_entry": trade_mid_entry,
            "trade_slope_entry": trade_slope_entry,
            "exit_reason": exit_reason
        })
        
        i += 1
    
    return pd.DataFrame(trades)

def analyze_dual_ticker_strategy(
    filter_ticker: str,
    trade_ticker: str,
    start_date: str,
    end_date: str,
    filter_condition: dict,
    strategies: list
) -> pd.DataFrame:
    """
    Analyze dual ticker hedge strategies.
    
    Args:
        filter_ticker: Symbol for filter ticker (e.g., "SPY")
        trade_ticker: Symbol for trade ticker (e.g., "TQQQ")
        start_date: Start date for analysis
        end_date: End date for analysis
        filter_condition: Dict with min_band and max_band for filter
        strategies: List of tuples (name, entry_band, exit_band, side)
    
    Returns:
        DataFrame with results
    """
    print(f"Loading {filter_ticker} data...")
    filter_df = fetch_ohlc(filter_ticker, start_date, end_date, auto_adjust=False)
    filter_std = compute_std_lines_strict(filter_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    
    print(f"Loading {trade_ticker} data...")
    trade_df = fetch_ohlc(trade_ticker, start_date, end_date, auto_adjust=False)
    trade_std = compute_std_lines_strict(trade_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    
    results = []
    
    for name, entry_band, exit_band, side in strategies:
        print(f"  Running: {name}...")
        trades = build_dual_ticker_hedge_trades(
            filter_std, trade_std, filter_condition, entry_band, exit_band, side
        )
        
        if trades.empty:
            continue
        
        wins = (trades["ret"] > 0).sum()
        win_rate = wins / len(trades) * 100
        avg_ret = trades["ret"].mean() * 100
        total_ret = trades["ret"].sum() * 100
        avg_days = trades["duration_days"].mean()
        avg_position_dd = trades["position_max_dd"].mean() * 100
        worst_position_dd = trades["position_max_dd"].min() * 100
        
        calmar = total_ret / abs(worst_position_dd) if worst_position_dd != 0 else 0
        
        results.append({
            "Strategy": name,
            "Trades": len(trades),
            "Win%": win_rate,
            "Avg_Ret%": avg_ret,
            "Total_Ret%": total_ret,
            "Avg_Days": avg_days,
            "Avg_PosDD%": avg_position_dd,
            "Worst_PosDD%": worst_position_dd,
            "Calmar": calmar
        })
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values("Total_Ret%", ascending=False)

if __name__ == "__main__":
    # ========================================
    # CONFIGURATION
    # ========================================
    
    FILTER_TICKER = "SPY"   # The confirmation ticker
    TRADE_TICKER = "TQQQ"   # The ticker to trade
    START = "2010-01-01"
    END = "2020-12-31"
    
    # Filter condition: SPY must be between these bands
    FILTER_CONDITION = {
        "min_band": "mid",    # SPY must be above midline
        "max_band": "UB2"     # SPY must be below UB2
    }
    
    # Strategies to test (name, entry_band, exit_band, side)
    STRATEGIES = [
        # Hedge TQQQ when it closes above UB3.5 (between UB3 and UB4)
        ("HEDGE_TQQQ_UB3.5→UB3", "UB3.5", "UB3", "short"),
        ("HEDGE_TQQQ_UB3.5→UB2.5", "UB3.5", "UB2.5", "short"),
        ("HEDGE_TQQQ_UB3.5→UB2", "UB3.5", "UB2", "short"),
        ("HEDGE_TQQQ_UB3.5→UB1", "UB3.5", "UB1", "short"),
        ("HEDGE_TQQQ_UB3.5→Mid", "UB3.5", "mid", "short"),
        
        # Hedge TQQQ when it closes above UB2.5 (between UB2 and UB3)
        ("HEDGE_TQQQ_UB2.5→UB2", "UB2.5", "UB2", "short"),
        ("HEDGE_TQQQ_UB2.5→UB1.5", "UB2.5", "UB1.5", "short"),
        ("HEDGE_TQQQ_UB2.5→UB1", "UB2.5", "UB1", "short"),
        ("HEDGE_TQQQ_UB2.5→Mid", "UB2.5", "mid", "short"),
        
        # Hedge TQQQ when it closes above UB3, exit at different levels
        ("HEDGE_TQQQ_UB3→UB2", "UB3", "UB2", "short"),
        ("HEDGE_TQQQ_UB3→UB1", "UB3", "UB1", "short"),
        ("HEDGE_TQQQ_UB3→Mid", "UB3", "mid", "short"),
        
        # Hedge TQQQ when it closes above UB4, exit at different levels
        ("HEDGE_TQQQ_UB4→UB3", "UB4", "UB3", "short"),
        ("HEDGE_TQQQ_UB4→UB2", "UB4", "UB2", "short"),
        ("HEDGE_TQQQ_UB4→UB1", "UB4", "UB1", "short"),
        ("HEDGE_TQQQ_UB4→Mid", "UB4", "mid", "short"),
        
        # Hedge TQQQ when it closes above UB2, exit at different levels
        ("HEDGE_TQQQ_UB2→UB1", "UB2", "UB1", "short"),
        ("HEDGE_TQQQ_UB2→Mid", "UB2", "mid", "short"),
    ]
    
    # ========================================
    # RUN ANALYSIS
    # ========================================
    
    print(f"\n{'='*100}")
    print(f"DUAL TICKER HEDGE STRATEGY ANALYSIS")
    print(f"{'='*100}")
    print(f"Filter Ticker: {FILTER_TICKER} (must be between {FILTER_CONDITION['min_band']} and {FILTER_CONDITION['max_band']})")
    print(f"Trade Ticker: {TRADE_TICKER}")
    print(f"Period: {START} to {END}")
    print(f"STDEV_PERIOD: {STDEV_PERIOD} | N: {N} | SLIPPAGE: {SLIPPAGE_BPS} bps")
    print(f"{'='*100}\n")
    
    results = analyze_dual_ticker_strategy(
        FILTER_TICKER, TRADE_TICKER, START, END, FILTER_CONDITION, STRATEGIES
    )
    
    print(f"\n{'='*100}")
    print("RESULTS (Sorted by Total Return)")
    print(f"{'='*100}")
    print(results.to_string(index=False))
    print(f"{'='*100}\n")
    
    # Best by Calmar
    by_calmar = results.sort_values("Calmar", ascending=False)
    print(f"\n{'='*100}")
    print("SORTED BY CALMAR RATIO (Best Risk-Adjusted)")
    print(f"{'='*100}")
    print(by_calmar[['Strategy', 'Trades', 'Total_Ret%', 'Worst_PosDD%', 'Calmar']].to_string(index=False))
    print(f"{'='*100}\n")
    
    # Winner
    if not results.empty:
        print(f"\n{'='*100}")
        print("🏆 BEST STRATEGY")
        print(f"{'='*100}")
        winner = results.iloc[0]
        print(f"Strategy: {winner['Strategy']}")
        print(f"Total Trades: {int(winner['Trades'])}")
        print(f"Win Rate: {winner['Win%']:.1f}%")
        print(f"Avg Return per Trade: {winner['Avg_Ret%']:.2f}%")
        print(f"Total Return: {winner['Total_Ret%']:.2f}%")
        print(f"Avg Duration: {winner['Avg_Days']:.1f} days")
        print(f"Avg Position Drawdown: {winner['Avg_PosDD%']:.2f}%")
        print(f"Worst Position Drawdown: {winner['Worst_PosDD%']:.2f}%")
        print(f"Calmar Ratio: {winner['Calmar']:.2f}")
        print(f"{'='*100}\n")

