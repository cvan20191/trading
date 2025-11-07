# Parameter Sensitivity Analysis - Dual Ticker Hedge Strategies
# Test how dual ticker hedge strategy performance changes with different parameter values

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc, compute_std_lines_strict,
    PRICE_SRC, DDOF, SLIPPAGE_BPS
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

def get_band_value(df: pd.DataFrame, idx: int, band_spec) -> float:
    """Get band value supporting fractional bands like UB3.5, LB2.5."""
    if band_spec == "mid":
        return float(df["mid"].iloc[idx])
    
    # Parse band specification
    band_str = str(band_spec)
    
    # Check if it's a fractional band (e.g., "UB3.5" or "LB2.5")
    if "." in band_str:
        if band_str.startswith("UB"):
            band_type = "UB"
            level = float(band_str[2:])
        elif band_str.startswith("LB"):
            band_type = "LB"
            level = float(band_str[2:])
        else:
            # Fallback: assume UB if no prefix
            band_type = "UB"
            level = float(band_str)
        
        lower_level = int(np.floor(level))
        upper_level = int(np.ceil(level))
        
        if lower_level == upper_level:
            return float(df[f"{band_type}{lower_level}"].iloc[idx])
        
        # Interpolate between bands
        lower_band = float(df[f"{band_type}{lower_level}"].iloc[idx])
        upper_band = float(df[f"{band_type}{upper_level}"].iloc[idx])
        fraction = level - lower_level
        return lower_band + fraction * (upper_band - lower_band)
    
    # Standard band (e.g., "UB3", "LB2")
    return float(df[band_spec].iloc[idx])

def build_dual_ticker_hedge_trades(
    filter_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    filter_condition: dict,
    trade_entry_band: str,
    trade_exit_band: str,
    side: str = "short",
    slippage_bps: float = SLIPPAGE_BPS
) -> pd.DataFrame:
    """Build dual ticker hedge trades with filter condition."""
    # Align dataframes
    common_dates = filter_df.index.intersection(trade_df.index)
    filter_aligned = filter_df.loc[common_dates].dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "UB4"])
    trade_aligned = trade_df.loc[common_dates].dropna(subset=["open", "close", "mid", "UB1", "UB2", "UB3", "UB4"])
    
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
        
        # Check filter condition (SPY between mid and UB2)
        filter_close = float(filter_aligned.loc[current_date, "close"])
        filter_min_idx = filter_aligned.index.get_loc(current_date)
        filter_max_idx = filter_min_idx
        filter_min = get_band_value(filter_aligned, filter_min_idx, filter_condition["min_band"])
        filter_max = get_band_value(filter_aligned, filter_max_idx, filter_condition["max_band"])
        
        filter_ok = (filter_close >= filter_min) and (filter_close <= filter_max)
        
        if not filter_ok:
            i += 1
            continue
        
        # Check trade entry condition (TQQQ closes above entry band)
        trade_close = float(trade_aligned.loc[current_date, "close"])
        trade_idx = trade_aligned.index.get_loc(current_date)
        trade_entry_level = get_band_value(trade_aligned, trade_idx, trade_entry_band)
        
        if side == "short":
            entry_signal = (trade_close >= trade_entry_level)
        else:
            entry_signal = (trade_close <= trade_entry_level)
        
        if not entry_signal:
            i += 1
            continue
        
        # Enter at next day's open
        j = i + 1
        entry_date = idx[j]
        entry_open = float(trade_aligned.loc[entry_date, "open"])
        
        if side == "short":
            entry_price = entry_open * (1.0 - bps)
        else:
            entry_price = entry_open * (1.0 + bps)
        
        in_trade = True
        
        # Search for exit
        exit_date = None
        exit_price = None
        
        for k in range(j, len(idx)):
            exit_check_date = idx[k]
            trade_close_k = float(trade_aligned.loc[exit_check_date, "close"])
            trade_exit_idx = trade_aligned.index.get_loc(exit_check_date)
            trade_exit_level = get_band_value(trade_aligned, trade_exit_idx, trade_exit_band)
            
            if side == "short":
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
                    break
            else:
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
                    break
        
        if exit_date is None:
            last_date = idx[-1]
            exit_date = last_date
            if side == "short":
                exit_price = float(trade_aligned.loc[last_date, "close"]) * (1.0 + bps)
            else:
                exit_price = float(trade_aligned.loc[last_date, "close"]) * (1.0 - bps)
            i = len(idx) - 1
        
        in_trade = False
        
        ret = (entry_price / exit_price - 1.0) if side == "short" else (exit_price / entry_price - 1.0)
        position_dd = calculate_position_drawdown(trade_aligned, entry_date, exit_date, entry_price, side)
        
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "ret": ret,
            "position_max_dd": position_dd,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
        })
        
        i += 1
    
    return pd.DataFrame(trades)

def analyze_dual_ticker_with_params(filter_ticker: str, trade_ticker: str, start: str, end: str, 
                                   n: int, stdev_len: int, filter_condition: dict,
                                   entry_band: str, exit_band: str, side: str = "short") -> dict:
    """Analyze dual ticker hedge strategy with specific parameters."""
    try:
        # Load both tickers
        filter_df = fetch_ohlc(filter_ticker, start, end, auto_adjust=False)
        filter_std = compute_std_lines_strict(filter_df, n=n, stdev_len=stdev_len, price_src=PRICE_SRC, ddof=DDOF)
        
        trade_df = fetch_ohlc(trade_ticker, start, end, auto_adjust=False)
        trade_std = compute_std_lines_strict(trade_df, n=n, stdev_len=stdev_len, price_src=PRICE_SRC, ddof=DDOF)
        
        trades = build_dual_ticker_hedge_trades(filter_std, trade_std, filter_condition, entry_band, exit_band, side)
        
        if trades.empty:
            return {
                "trades": 0,
                "total_ret": 0,
                "avg_ret": 0,
                "win_rate": 0,
                "sharpe": 0,
                "avg_dd": 0,
                "worst_dd": 0,
                "calmar": 0
            }
        
        total_ret = trades["ret"].sum() * 100
        avg_ret = trades["ret"].mean() * 100
        win_rate = (trades["ret"] > 0).sum() / len(trades) * 100
        avg_dd = trades["position_max_dd"].mean() * 100
        worst_dd = trades["position_max_dd"].min() * 100
        
        # Simple Sharpe approximation
        ret_std = trades["ret"].std()
        sharpe = (trades["ret"].mean() / ret_std) if ret_std > 0 else 0
        
        # Calmar
        calmar = total_ret / abs(worst_dd) if worst_dd != 0 else 0
        
        return {
            "trades": len(trades),
            "total_ret": total_ret,
            "avg_ret": avg_ret,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "avg_dd": avg_dd,
            "worst_dd": worst_dd,
            "calmar": calmar
        }
    except Exception as e:
        print(f"Error with N={n}, StDev={stdev_len}: {e}")
        return {
            "trades": 0,
            "total_ret": 0,
            "avg_ret": 0,
            "win_rate": 0,
            "sharpe": 0,
            "avg_dd": 0,
            "worst_dd": 0,
            "calmar": 0
        }

def run_sensitivity_analysis(filter_ticker: str, trade_ticker: str, start: str, end: str, 
                             n_values: list, stdev_values: list, filter_condition: dict,
                             strategies: list) -> pd.DataFrame:
    """
    Run sensitivity analysis across parameter ranges for dual ticker hedge strategies.
    
    Args:
        filter_ticker: Filter ticker (e.g., SPY)
        trade_ticker: Trade ticker (e.g., TQQQ)
        start: Start date
        end: End date
        n_values: List of N (regression window) values to test
        stdev_values: List of StDev lookback values to test
        filter_condition: Dict with min_band and max_band for filter
        strategies: List of (name, entry_band, exit_band, side) tuples
    
    Returns:
        DataFrame with results for each parameter combination
    """
    results = []
    
    total_combos = len(n_values) * len(stdev_values) * len(strategies)
    current = 0
    
    for strategy_name, entry_band, exit_band, side in strategies:
        print(f"\n{'='*80}")
        print(f"Testing Strategy: {strategy_name}")
        print(f"{'='*80}")
        
        for n in n_values:
            for stdev_len in stdev_values:
                current += 1
                print(f"  [{current}/{total_combos}] N={n}, StDev={stdev_len}...", end=" ")
                
                metrics = analyze_dual_ticker_with_params(
                    filter_ticker, trade_ticker, start, end, n, stdev_len, 
                    filter_condition, entry_band, exit_band, side
                )
                
                print(f"Trades={metrics['trades']}, Total Ret={metrics['total_ret']:.1f}%, Calmar={metrics['calmar']:.2f}")
                
                results.append({
                    "Strategy": strategy_name,
                    "N": n,
                    "StDev_Len": stdev_len,
                    "Trades": metrics["trades"],
                    "Total_Ret%": metrics["total_ret"],
                    "Avg_Ret%": metrics["avg_ret"],
                    "Win%": metrics["win_rate"],
                    "Sharpe": metrics["sharpe"],
                    "Avg_DD%": metrics["avg_dd"],
                    "Worst_DD%": metrics["worst_dd"],
                    "Calmar": metrics["calmar"]
                })
    
    return pd.DataFrame(results)

def analyze_sensitivity_results(results_df: pd.DataFrame):
    """Analyze and print sensitivity analysis results."""
    
    print(f"\n\n{'='*100}")
    print("PARAMETER SENSITIVITY ANALYSIS RESULTS")
    print(f"{'='*100}\n")
    
    for strategy in results_df["Strategy"].unique():
        strat_data = results_df[results_df["Strategy"] == strategy].copy()
        
        print(f"\n{'='*100}")
        print(f"Strategy: {strategy}")
        print(f"{'='*100}")
        
        # Calculate coefficient of variation (CV) for key metrics
        # CV = std / mean - lower is better (more stable)
        total_ret_cv = strat_data["Total_Ret%"].std() / strat_data["Total_Ret%"].mean() if strat_data["Total_Ret%"].mean() != 0 else np.inf
        avg_ret_cv = strat_data["Avg_Ret%"].std() / strat_data["Avg_Ret%"].mean() if strat_data["Avg_Ret%"].mean() != 0 else np.inf
        win_rate_cv = strat_data["Win%"].std() / strat_data["Win%"].mean() if strat_data["Win%"].mean() != 0 else np.inf
        
        print(f"\nStability Metrics (Coefficient of Variation - Lower is Better):")
        print(f"  Total Return CV: {total_ret_cv:.3f}")
        print(f"  Avg Return CV: {avg_ret_cv:.3f}")
        print(f"  Win Rate CV: {win_rate_cv:.3f}")
        
        print(f"\nPerformance Range:")
        print(f"  Total Return: {strat_data['Total_Ret%'].min():.1f}% to {strat_data['Total_Ret%'].max():.1f}% (Range: {strat_data['Total_Ret%'].max() - strat_data['Total_Ret%'].min():.1f}%)")
        print(f"  Avg Return: {strat_data['Avg_Ret%'].min():.2f}% to {strat_data['Avg_Ret%'].max():.2f}%")
        print(f"  Win Rate: {strat_data['Win%'].min():.1f}% to {strat_data['Win%'].max():.1f}%")
        print(f"  Trade Count: {strat_data['Trades'].min()} to {strat_data['Trades'].max()}")
        
        # Best and worst parameter combinations
        best_idx = strat_data["Total_Ret%"].idxmax()
        worst_idx = strat_data["Total_Ret%"].idxmin()
        
        print(f"\nBest Parameters:")
        print(f"  N={strat_data.loc[best_idx, 'N']}, StDev={strat_data.loc[best_idx, 'StDev_Len']}")
        print(f"  Total Return: {strat_data.loc[best_idx, 'Total_Ret%']:.1f}%")
        
        print(f"\nWorst Parameters:")
        print(f"  N={strat_data.loc[worst_idx, 'N']}, StDev={strat_data.loc[worst_idx, 'StDev_Len']}")
        print(f"  Total Return: {strat_data.loc[worst_idx, 'Total_Ret%']:.1f}%")
        
        # Detailed results table
        print(f"\nDetailed Results:")
        print(strat_data.to_string(index=False))
        
        # Robustness assessment
        print(f"\n{'='*100}")
        print("ROBUSTNESS ASSESSMENT:")
        print(f"{'='*100}")
        
        # Check if performance degrades gracefully
        ret_range_pct = (strat_data['Total_Ret%'].max() - strat_data['Total_Ret%'].min()) / abs(strat_data['Total_Ret%'].mean()) * 100 if strat_data['Total_Ret%'].mean() != 0 else np.inf
        
        if total_ret_cv < 0.2:
            robustness = "🟢 HIGHLY ROBUST - Performance is very stable across parameters"
        elif total_ret_cv < 0.5:
            robustness = "🟡 MODERATELY ROBUST - Performance varies but remains reasonable"
        else:
            robustness = "🔴 PARAMETER SENSITIVE - Performance varies significantly (RED FLAG)"
        
        print(f"{robustness}")
        print(f"Return Range: {ret_range_pct:.1f}% of mean")
        
        if total_ret_cv < 0.5:
            print("✓ Strategy shows graceful degradation - good sign of robustness")
        else:
            print("⚠ Strategy performance 'falls off a cliff' with parameter changes - major red flag")

if __name__ == "__main__":
    FILTER_TICKER = "SPY"
    TRADE_TICKER = "TQQQ"
    START = "2010-01-01"
    END = "2020-12-31"
    
    # Filter condition
    FILTER_CONDITION = {
        "min_band": "mid",
        "max_band": "UB2"
    }
    
    # Parameter ranges to test
    N_VALUES = [200, 220, 252, 280, 300]  # Regression window
    STDEV_VALUES = [10, 12, 15, 20, 25]   # StDev lookback period
    
    # Strategies to test - the ones you want to verify for robustness
    STRATEGIES = [
        ("HEDGE_TQQQ_UB2.5→UB1.5", "UB2.5", "UB1.5", "short"),  # Tighter range
        ("HEDGE_TQQQ_UB2.5→UB2", "UB2.5", "UB2", "short"),      # Very tight exit
    ]
    
    print(f"\n{'='*100}")
    print(f"DUAL TICKER HEDGE - PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*100}")
    print(f"Filter Ticker: {FILTER_TICKER} (between {FILTER_CONDITION['min_band']} and {FILTER_CONDITION['max_band']})")
    print(f"Trade Ticker: {TRADE_TICKER}")
    print(f"Period: {START} to {END}")
    print(f"N (Regression Window) values: {N_VALUES}")
    print(f"StDev Lookback values: {STDEV_VALUES}")
    print(f"Total parameter combinations: {len(N_VALUES) * len(STDEV_VALUES)}")
    print(f"Strategies to test: {len(STRATEGIES)}")
    print(f"Total tests: {len(N_VALUES) * len(STDEV_VALUES) * len(STRATEGIES)}")
    print(f"{'='*100}\n")
    
    # Run analysis
    results = run_sensitivity_analysis(FILTER_TICKER, TRADE_TICKER, START, END, 
                                      N_VALUES, STDEV_VALUES, FILTER_CONDITION, STRATEGIES)
    
    # Analyze results
    analyze_sensitivity_results(results)
    
    # Summary comparison
    print(f"\n\n{'='*100}")
    print("CROSS-STRATEGY ROBUSTNESS COMPARISON")
    print(f"{'='*100}\n")
    
    summary = []
    for strategy in results["Strategy"].unique():
        strat_data = results[results["Strategy"] == strategy]
        total_ret_cv = strat_data["Total_Ret%"].std() / strat_data["Total_Ret%"].mean() if strat_data["Total_Ret%"].mean() != 0 else np.inf
        avg_total_ret = strat_data["Total_Ret%"].mean()
        
        summary.append({
            "Strategy": strategy,
            "Avg_Total_Ret%": avg_total_ret,
            "Ret_CV": total_ret_cv,
            "Robustness": "High" if total_ret_cv < 0.2 else ("Medium" if total_ret_cv < 0.5 else "Low")
        })
    
    summary_df = pd.DataFrame(summary).sort_values("Ret_CV")
    print(summary_df.to_string(index=False))
    print(f"\n{'='*100}\n")

