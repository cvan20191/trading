# Sequential (Non-Overlapping) Trades Analysis
# Pyramiding DISABLED: Exit must occur before new entry is recognized

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc, compute_std_lines_strict,
    N, STDEV_PERIOD, PRICE_SRC, DDOF, SLIPPAGE_BPS
)

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

def build_sequential_trades(std_df: pd.DataFrame, side: str, entry_band_k: int, exit_target: str, slippage_bps: float = SLIPPAGE_BPS) -> pd.DataFrame:
    """
    Build SEQUENTIAL (non-overlapping) trades.
    Once in a trade, ignore all new signals until exit occurs.
    """
    d = std_df.dropna(subset=["open","high","low","close","mid","UB1","UB2","UB3","UB4","LB1","LB2","LB3","LB4"]).copy()
    idx = d.index.to_list()
    trades = []
    bps = slippage_bps / 10000.0
    
    in_trade = False
    i = 0
    
    while i < len(d) - 1:
        if in_trade:
            i += 1
            continue
            
        # Check for entry signal
        close_today = float(d["close"].iloc[i])
        
        if side == "short":
            entry_band = float(d[f"UB{entry_band_k}"].iloc[i])
            entry_signal = (close_today >= entry_band)
        else:
            entry_band = float(d[f"LB{entry_band_k}"].iloc[i])
            entry_signal = (close_today <= entry_band)
        
        if not entry_signal:
            i += 1
            continue
        
        # Enter trade at next day's open
        j = i + 1
        entry_date = idx[j]
        entry_open = float(d["open"].iloc[j])
        
        if side == "short":
            entry_price = entry_open * (1.0 - bps)
        else:
            entry_price = entry_open * (1.0 + bps)
        
        in_trade = True
        
        # Search for exit
        exit_date = None
        exit_price = None
        exit_reason = None
        
        for k in range(j, len(d)):
            close_k = float(d["close"].iloc[k])
            exit_level = float(d[exit_target].iloc[k])
            
            if side == "short":
                if close_k <= exit_level:
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        exit_open = float(d["open"].iloc[k+1])
                        exit_price = exit_open * (1.0 + bps)
                        i = k + 1
                    else:
                        exit_date = idx[k]
                        exit_price = close_k * (1.0 + bps)
                        i = k
                    exit_reason = f"close_reach_{exit_target}"
                    break
            else:
                if close_k >= exit_level:
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        exit_open = float(d["open"].iloc[k+1])
                        exit_price = exit_open * (1.0 - bps)
                        i = k + 1
                    else:
                        exit_date = idx[k]
                        exit_price = close_k * (1.0 - bps)
                        i = k
                    exit_reason = f"close_reach_{exit_target}"
                    break
        
        # Force exit if never found
        if exit_date is None:
            last_i = len(d) - 1
            exit_date = idx[last_i]
            if side == "short":
                exit_price = float(d["close"].iloc[last_i]) * (1.0 + bps)
            else:
                exit_price = float(d["close"].iloc[last_i]) * (1.0 - bps)
            exit_reason = "force_exit"
            i = last_i
        
        in_trade = False
        
        # Capture metrics
        mid_entry = float(d["mid"].iloc[j]) if pd.notna(d["mid"].iloc[j]) else float("nan")
        slope_entry = float(d["mid_slope"].iloc[j]) if pd.notna(d["mid_slope"].iloc[j]) else float("nan")
        
        ret = (entry_price / exit_price - 1.0) if side == "short" else (exit_price / entry_price - 1.0)
        position_dd = calculate_position_drawdown(d, entry_date, exit_date, entry_price, side)
        
        trades.append({
            "side": side,
            "signal_date": idx[i-1] if i > 0 else idx[0],
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "ret": ret,
            "position_max_dd": position_dd,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
            "mid_entry": mid_entry,
            "mid_slope_entry": slope_entry,
            "exit_reason": exit_reason
        })
        
        i += 1
    
    return pd.DataFrame(trades)

def analyze_strategies(std_df: pd.DataFrame, strategies: list) -> pd.DataFrame:
    """
    Analyze a list of strategies and return results DataFrame.
    
    Args:
        std_df: DataFrame with OHLC and channel data
        strategies: List of tuples (name, side, entry_band_k, exit_target)
                   Example: [("SHORT_UB4→UB2", "short", 4, "UB2")]
    
    Returns:
        DataFrame with results sorted by Total Return
    """
    results = []
    
    for name, side, entry_k, exit_target in strategies:
        trades = build_sequential_trades(std_df, side=side, entry_band_k=entry_k, exit_target=exit_target)
        
        if trades.empty:
            continue
        
        wins = (trades["ret"] > 0).sum()
        win_rate = wins / len(trades) * 100
        avg_ret = trades["ret"].mean() * 100
        total_ret = trades["ret"].sum() * 100
        avg_days = trades["duration_days"].mean()
        avg_position_dd = trades["position_max_dd"].mean() * 100
        worst_position_dd = trades["position_max_dd"].min() * 100
        
        # Calculate Calmar proxy
        calmar_proxy = total_ret / abs(worst_position_dd) if worst_position_dd != 0 else 0
        
        results.append({
            "Strategy": name,
            "Trades": len(trades),
            "Win%": win_rate,
            "Avg_Ret%": avg_ret,
            "Total_Ret%": total_ret,
            "Avg_Days": avg_days,
            "Avg_PosDD%": avg_position_dd,
            "Worst_PosDD%": worst_position_dd,
            "Calmar": calmar_proxy
        })
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values("Total_Ret%", ascending=False)

def print_results(results_df: pd.DataFrame, title: str = "RESULTS"):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(title)
    print(f"{'='*100}")
    print(results_df.to_string(index=False))
    print(f"{'='*100}\n")

if __name__ == "__main__":
    SYMBOL = "SPY"
    START = "2010-01-01"
    END = "2020-12-31"
    
    print(f"\n{'='*100}")
    print(f"SEQUENTIAL (NON-OVERLAPPING) TRADES ANALYSIS - PYRAMIDING DISABLED")
    print(f"{'='*100}")
    print(f"Period: {START} to {END}")
    print(f"Rule: Exit must occur before new entry is recognized")
    print(f"STDEV_PERIOD: {STDEV_PERIOD} | N: {N} | SLIPPAGE: {SLIPPAGE_BPS} bps")
    print(f"{'='*100}\n")
    
    # Load data once
    df = fetch_ohlc(SYMBOL, START, END, auto_adjust=False)
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna()
    
    # ========================================
    # CONFIGURE YOUR STRATEGIES HERE
    # ========================================
    
    # SHORT STRATEGIES - Change these as desired
    short_strategies = [
        ("SHORT_UB4→Mid", "short", 4, "mid"),
        ("SHORT_UB4→UB3", "short", 4, "UB3"),
        ("SHORT_UB4→UB2", "short", 4, "UB2"),
        ("SHORT_UB4→UB1", "short", 4, "UB1"),
        
        ("SHORT_UB3→Mid", "short", 3, "mid"),
        ("SHORT_UB3→UB2", "short", 3, "UB2"),
        ("SHORT_UB3→UB1", "short", 3, "UB1"),
        
        ("SHORT_UB2→Mid", "short", 2, "mid"),
        ("SHORT_UB2→UB1", "short", 2, "UB1"),
        
        ("SHORT_UB1→Mid", "short", 1, "mid"),
    ]
    
    # LONG STRATEGIES - Change these as desired
    long_strategies = [
        ("LONG_LB4→Mid", "long", 4, "mid"),
        ("LONG_LB4→LB3", "long", 4, "LB3"),
        ("LONG_LB4→LB2", "long", 4, "LB2"),
        ("LONG_LB4→LB1", "long", 4, "LB1"),
        
        ("LONG_LB3→Mid", "long", 3, "mid"),
        ("LONG_LB3→LB2", "long", 3, "LB2"),
        ("LONG_LB3→LB1", "long", 3, "LB1"),
        
        ("LONG_LB2→Mid", "long", 2, "mid"),
        ("LONG_LB2→LB1", "long", 2, "LB1"),
        
        ("LONG_LB1→Mid", "long", 1, "mid"),
    ]
    
    # ========================================
    # RUN ANALYSIS
    # ========================================
    
    print("Running SHORT strategies...")
    short_results = analyze_strategies(std_df, short_strategies)
    print_results(short_results, "SHORT STRATEGIES (Sorted by Total Return)")
    
    print("\nRunning LONG strategies...")
    long_results = analyze_strategies(std_df, long_strategies)
    print_results(long_results, "LONG STRATEGIES (Sorted by Total Return)")
    
    # Combined analysis
    all_strategies = short_strategies + long_strategies
    all_results = analyze_strategies(std_df, all_strategies)
    
    print_results(all_results, "ALL STRATEGIES (Sorted by Total Return)")
    
    # Best by Calmar
    by_calmar = all_results.sort_values("Calmar", ascending=False)
    print_results(by_calmar[['Strategy', 'Trades', 'Total_Ret%', 'Worst_PosDD%', 'Calmar']].head(10), 
                  "TOP 10 BY CALMAR RATIO (Best Risk-Adjusted)")
    
    # Safest
    by_safety = all_results.sort_values("Worst_PosDD%", ascending=False)
    print_results(by_safety[['Strategy', 'Trades', 'Total_Ret%', 'Avg_PosDD%', 'Worst_PosDD%']].head(10),
                  "TOP 10 SAFEST (Lowest Worst Position Drawdown)")
    
    # Winner
    print(f"\n{'='*100}")
    print("🏆 WINNER: MOST PROFITABLE STRATEGY (Sequential/Non-Overlapping)")
    print(f"{'='*100}")
    winner = all_results.iloc[0]
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
