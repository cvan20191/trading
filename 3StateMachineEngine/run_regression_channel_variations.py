#!/usr/bin/env python3
"""
Run regression channel analysis with all entry/exit band combinations.

Tests:
- LB3 -> LB2, LB1, Midline, UB1, UB2, UB3
- LB2 -> LB1, Midline, UB1, UB2, UB3
- UB3 -> UB2, UB1, Midline, LB1, LB2, LB3
- UB2 -> UB1, Midline, LB1, LB2, LB3
- UB1 -> Midline, LB1, LB2, LB3
"""

import pandas as pd
import numpy as np
import sys
from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc,
    compute_std_lines_strict,
    build_pyramiding_trades,
    N, STDEV_PERIOD, PRICE_SRC, DDOF, SLIPPAGE_BPS
)

def build_non_overlapping_trades(std_df: pd.DataFrame,
                                  side: str,
                                  entry_band_k: int,
                                  exit_target: str,
                                  slippage_bps: float = SLIPPAGE_BPS) -> pd.DataFrame:
    """
    Build trades WITHOUT overlapping positions - only one position at a time.
    Each entry signal is ignored if already in a position.
    """
    d = std_df.dropna(subset=["open","high","low","close","mid","UB1","UB2","UB3","UB4","LB1","LB2","LB3","LB4"]).copy()
    idx = d.index.to_list()
    trades = []
    bps = slippage_bps / 10000.0
    
    in_position = False
    current_entry_idx = None
    
    for i in range(len(d)):
        if i >= len(d) - 1:
            break  # need next-day open to fill
        
        # If in position, check for exit first (on day i, after entry has occurred)
        if in_position and current_entry_idx is not None and i >= current_entry_idx:
            exit_level = float(d[exit_target].iloc[i])
            
            if side == "short":
                # Short exit: Close <= exit_target
                if float(d["close"].iloc[i]) <= exit_level:
                    exit_date = idx[i]
                    exit_price = exit_level * (1.0 + bps)  # buy to close at band
                    
                    # Get entry details
                    entry_date = idx[current_entry_idx]
                    entry_price = float(d["open"].iloc[current_entry_idx]) * (1.0 - bps)
                    
                    ret = (entry_price / exit_price - 1.0)
                    mid_entry = float(d["mid"].iloc[current_entry_idx]) if pd.notna(d["mid"].iloc[current_entry_idx]) else float("nan")
                    slope_entry = float(d["mid_slope"].iloc[current_entry_idx]) if pd.notna(d["mid_slope"].iloc[current_entry_idx]) else float("nan")
                    
                    trades.append({
                        "side": side,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "ret": ret,
                        "pnl": ret,
                        "entry_band_k": entry_band_k,
                        "exit_target": exit_target,
                        "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
                        "mid_entry": mid_entry,
                        "mid_slope_entry": slope_entry,
                        "exit_reason": f"reach_{exit_target}"
                    })
                    
                    in_position = False
                    current_entry_idx = None
                    continue  # Skip entry check for this day
            else:
                # Long exit: Close >= exit_target
                if float(d["close"].iloc[i]) >= exit_level:
                    exit_date = idx[i]
                    exit_price = exit_level * (1.0 - bps)  # sell to close at band
                    
                    # Get entry details
                    entry_date = idx[current_entry_idx]
                    entry_price = float(d["open"].iloc[current_entry_idx]) * (1.0 + bps)
                    
                    ret = (exit_price / entry_price - 1.0)
                    mid_entry = float(d["mid"].iloc[current_entry_idx]) if pd.notna(d["mid"].iloc[current_entry_idx]) else float("nan")
                    slope_entry = float(d["mid_slope"].iloc[current_entry_idx]) if pd.notna(d["mid_slope"].iloc[current_entry_idx]) else float("nan")
                    
                    trades.append({
                        "side": side,
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": exit_date,
                        "exit_price": exit_price,
                        "ret": ret,
                        "pnl": ret,
                        "entry_band_k": entry_band_k,
                        "exit_target": exit_target,
                        "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
                        "mid_entry": mid_entry,
                        "mid_slope_entry": slope_entry,
                        "exit_reason": f"reach_{exit_target}"
                    })
                    
                    in_position = False
                    current_entry_idx = None
                    continue  # Skip entry check for this day
        
        # If not in position, check for entry signal (on day i, enter at day i+1)
        if not in_position:
            if side == "short":
                entry_band = d[f"UB{entry_band_k}"]
                entry_signal = (d["high"].iloc[i] >= float(entry_band.iloc[i]))
            else:
                entry_band = d[f"LB{entry_band_k}"]
                entry_signal = (d["low"].iloc[i] <= float(entry_band.iloc[i]))
            
            if entry_signal:
                # Entry at next day's open (day i+1)
                if i + 1 < len(d):
                    current_entry_idx = i + 1
                    in_position = True
    
    # Force exit on last bar if still in position
    if in_position and current_entry_idx is not None:
        last_i = len(d) - 1
        exit_date = idx[last_i]
        entry_date = idx[current_entry_idx]
        
        if side == "short":
            entry_price = float(d["open"].iloc[current_entry_idx]) * (1.0 - bps)
            exit_price = float(d["close"].iloc[last_i]) * (1.0 + bps)
            ret = (entry_price / exit_price - 1.0)
        else:
            entry_price = float(d["open"].iloc[current_entry_idx]) * (1.0 + bps)
            exit_price = float(d["close"].iloc[last_i]) * (1.0 - bps)
            ret = (exit_price / entry_price - 1.0)
        
        mid_entry = float(d["mid"].iloc[current_entry_idx]) if pd.notna(d["mid"].iloc[current_entry_idx]) else float("nan")
        slope_entry = float(d["mid_slope"].iloc[current_entry_idx]) if pd.notna(d["mid_slope"].iloc[current_entry_idx]) else float("nan")
        
        trades.append({
            "side": side,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "ret": ret,
            "pnl": ret,
            "entry_band_k": entry_band_k,
            "exit_target": exit_target,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
            "mid_entry": mid_entry,
            "mid_slope_entry": slope_entry,
            "exit_reason": "force_exit"
        })
    
    return pd.DataFrame(trades)

def calculate_metrics(trades_df: pd.DataFrame, name: str) -> dict:
    """Calculate performance metrics from trades DataFrame."""
    if trades_df is None or trades_df.empty:
        return {
            "name": name,
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "avg_days": 0.0,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    
    n_trades = len(trades_df)
    wins = (trades_df["ret"] > 0).sum()
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    avg_return = trades_df["ret"].mean()
    total_return = (1 + trades_df["ret"]).prod() - 1.0
    avg_days = trades_df["duration_days"].mean()
    
    # Sharpe ratio (annualized)
    if n_trades > 1:
        returns = trades_df["ret"].values
        sharpe = np.sqrt(252 / avg_days) * (returns.mean() / returns.std()) if returns.std() > 0 else np.nan
    else:
        sharpe = np.nan
    
    # Max drawdown (cumulative)
    cumret = (1 + trades_df["ret"]).cumprod()
    running_max = cumret.cummax()
    drawdown = (cumret / running_max - 1.0)
    max_drawdown = drawdown.min()
    
    return {
        "name": name,
        "trades": n_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_return": total_return,
        "avg_days": avg_days,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

def print_metrics_summary(metrics: dict):
    """Print formatted metrics summary."""
    print(f"\n{'='*80}")
    print(f"Strategy: {metrics['name']}")
    print(f"{'='*80}")
    print(f"Total Trades: {metrics['trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Average Return per Trade: {metrics['avg_return']*100:.4f}%")
    print(f"Total Cumulative Return: {metrics['total_return']*100:.2f}%")
    print(f"Average Duration (days): {metrics['avg_days']:.1f}")
    print(f"Sharpe Ratio (annualized): {metrics['sharpe']:.4f}" if not np.isnan(metrics['sharpe']) else "Sharpe Ratio (annualized): N/A")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"{'='*80}\n")

def run_all_variations(allow_overlapping=True):
    """Run all regression channel entry/exit combinations.
    
    Args:
        allow_overlapping: If True, use pyramiding (overlapping trades).
                          If False, only one position at a time.
    """
    
    # Configuration
    SYMBOL = "SPY"
    START = "2015-09-01"
    END = "2020-10-01"
    
    print(f"Loading data for {SYMBOL} from {START} to {END}...")
    df = fetch_ohlc(SYMBOL, START, END, auto_adjust=False)
    
    print("Computing regression channels...")
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna()
    
    print(f"Data ready: {len(std_df)} trading days\n")
    
    # Ensure Midline column exists (map from "mid")
    if "Midline" not in std_df.columns and "mid" in std_df.columns:
        std_df["Midline"] = std_df["mid"]
    
    all_results = []
    
    # Map exit target names to actual column names
    exit_target_map = {
        "Midline": "mid",  # Use "mid" column for midline
        "LB1": "LB1",
        "LB2": "LB2",
        "LB3": "LB3",
        "UB1": "UB1",
        "UB2": "UB2",
        "UB3": "UB3",
    }
    
    # Define all combinations
    combinations = [
        # LB3 variations (long entries)
        ("LB3", "long", 3, "LB2"),
        ("LB3", "long", 3, "LB1"),
        ("LB3", "long", 3, "Midline"),
        ("LB3", "long", 3, "UB1"),
        ("LB3", "long", 3, "UB2"),
        ("LB3", "long", 3, "UB3"),
        
        # LB2 variations (long entries)
        ("LB2", "long", 2, "LB1"),
        ("LB2", "long", 2, "Midline"),
        ("LB2", "long", 2, "UB1"),
        ("LB2", "long", 2, "UB2"),
        ("LB2", "long", 2, "UB3"),
        
        # UB3 variations (short entries)
        ("UB3", "short", 3, "UB2"),
        ("UB3", "short", 3, "UB1"),
        ("UB3", "short", 3, "Midline"),
        ("UB3", "short", 3, "LB1"),
        ("UB3", "short", 3, "LB2"),
        ("UB3", "short", 3, "LB3"),
        
        # UB2 variations (short entries)
        ("UB2", "short", 2, "UB1"),
        ("UB2", "short", 2, "Midline"),
        ("UB2", "short", 2, "LB1"),
        ("UB2", "short", 2, "LB2"),
        ("UB2", "short", 2, "LB3"),
        
        # UB1 variations (short entries)
        ("UB1", "short", 1, "Midline"),
        ("UB1", "short", 1, "LB1"),
        ("UB1", "short", 1, "LB2"),
        ("UB1", "short", 1, "LB3"),
    ]
    
    mode_str = "PYRAMIDING (overlapping trades allowed)" if allow_overlapping else "NON-OVERLAPPING (one position at a time)"
    print(f"Running {len(combinations)} strategy variations...")
    print(f"Mode: {mode_str}\n")
    
    for entry_band, side, entry_k, exit_target in combinations:
        strategy_name = f"{entry_band} -> {exit_target}"
        print(f"Running: {strategy_name}...")
        
        # Map exit target to actual column name
        exit_col = exit_target_map.get(exit_target, exit_target)
        
        # Verify column exists
        if exit_col not in std_df.columns:
            print(f"  WARNING: Column '{exit_col}' not found. Skipping...\n")
            all_results.append({
                "name": strategy_name,
                "trades": 0,
                "error": f"Column '{exit_col}' not found"
            })
            continue
        
        try:
            if allow_overlapping:
                trades_df = build_pyramiding_trades(
                    std_df,
                    side=side,
                    entry_band_k=entry_k,
                    exit_target=exit_col,  # Use mapped column name
                    slippage_bps=SLIPPAGE_BPS
                )
            else:
                trades_df = build_non_overlapping_trades(
                    std_df,
                    side=side,
                    entry_band_k=entry_k,
                    exit_target=exit_col,  # Use mapped column name
                    slippage_bps=SLIPPAGE_BPS
                )
            
            metrics = calculate_metrics(trades_df, strategy_name)
            all_results.append(metrics)
            print_metrics_summary(metrics)
            
        except Exception as e:
            print(f"ERROR running {strategy_name}: {e}\n")
            all_results.append({
                "name": strategy_name,
                "trades": 0,
                "error": str(e)
            })
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE - All Variations")
    print("="*100)
    
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df[["name", "trades", "win_rate", "avg_return", "total_return", "avg_days", "sharpe", "max_drawdown"]]
    summary_df["win_rate"] = (summary_df["win_rate"] * 100).round(2)
    summary_df["avg_return"] = (summary_df["avg_return"] * 100).round(4)
    summary_df["total_return"] = (summary_df["total_return"] * 100).round(2)
    summary_df["avg_days"] = summary_df["avg_days"].round(1)
    summary_df["sharpe"] = summary_df["sharpe"].round(4)
    summary_df["max_drawdown"] = (summary_df["max_drawdown"] * 100).round(2)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    # Save detailed results to CSV
    suffix = "pyramiding" if allow_overlapping else "non_overlapping"
    output_file = f"regression_channel_variations_results_{suffix}.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run regression channel variations")
    parser.add_argument("--non-overlapping", action="store_true", 
                       help="Use non-overlapping trades (one position at a time)")
    args = parser.parse_args()
    
    allow_overlapping = not args.non_overlapping
    results = run_all_variations(allow_overlapping=allow_overlapping)
