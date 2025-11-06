# Realistic Conservative Fading Backtest
# "Closes Beyond" logic: Signal when CLOSE crosses band, execute at next day's OPEN

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc, compute_std_lines_strict,
    N, STDEV_PERIOD, PRICE_SRC, DDOF, SLIPPAGE_BPS, INITIAL_EQUITY
)

def calculate_position_drawdown(std_df: pd.DataFrame, entry_date, exit_date, entry_price, side):
    """Calculate max drawdown for a single position during its holding period."""
    try:
        entry_idx = std_df.index.get_loc(pd.Timestamp(entry_date))
        exit_idx = std_df.index.get_loc(pd.Timestamp(exit_date))
        
        # Get price data during holding period
        holding_period = std_df.iloc[entry_idx:exit_idx+1]
        
        if side == "long":
            # For longs: drawdown is (low - entry) / entry
            worst_price = holding_period["low"].min()
            drawdown = (worst_price - entry_price) / entry_price
        else:
            # For shorts: drawdown is (high - entry) / entry (negative when price rises)
            worst_price = holding_period["high"].max()
            drawdown = (entry_price - worst_price) / entry_price
        
        return drawdown
    except:
        return 0.0

def build_realistic_fading_trades(std_df: pd.DataFrame,
                                  side: str,
                                  entry_band_k: int,
                                  exit_target: str,
                                  slippage_bps: float = SLIPPAGE_BPS) -> pd.DataFrame:
    """
    Realistic backtest: 
    - Entry signal: CLOSE beyond band at EOD -> enter at NEXT day's OPEN
    - Exit signal: CLOSE reaches target at EOD -> exit at NEXT day's OPEN
    
    This simulates real trading where you:
    1. Check signals after market close (4:01 PM)
    2. Place orders for next day's open (9:30 AM)
    """
    d = std_df.dropna(subset=["open","high","low","close","mid","UB1","UB2","UB3","UB4","LB1","LB2","LB3","LB4"]).copy()
    idx = d.index.to_list()
    trades = []
    bps = slippage_bps / 10000.0

    for i in range(len(d)):
        if i >= len(d) - 1:
            break  # need next-day open to fill

        # DAY i: Check if CLOSE is beyond entry band (signal generation at 4:01 PM)
        close_today = float(d["close"].iloc[i])
        
        if side == "short":
            entry_band = float(d[f"UB{entry_band_k}"].iloc[i])
            entry_signal = (close_today >= entry_band)  # Close ABOVE UBk
        else:
            entry_band = float(d[f"LB{entry_band_k}"].iloc[i])
            entry_signal = (close_today <= entry_band)  # Close BELOW LBk

        if not entry_signal:
            continue

        # DAY i+1: Execute entry at OPEN (9:30 AM next day)
        j = i + 1
        entry_date = idx[j]
        entry_open = float(d["open"].iloc[j])
        
        if side == "short":
            entry_price = entry_open * (1.0 - bps)  # sell to open
        else:
            entry_price = entry_open * (1.0 + bps)  # buy to open

        # Search for exit: CLOSE reaches target band
        exit_date = None
        exit_price = None
        exit_reason = None
        
        for k in range(j, len(d)):
            close_k = float(d["close"].iloc[k])
            exit_level = float(d[exit_target].iloc[k])
            
            if side == "short":
                # Short exit signal: CLOSE drops to/below exit target
                if close_k <= exit_level:
                    # Exit at NEXT day's open (if available)
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        exit_open = float(d["open"].iloc[k+1])
                        exit_price = exit_open * (1.0 + bps)  # buy to close
                    else:
                        # Last bar - exit at close
                        exit_date = idx[k]
                        exit_price = close_k * (1.0 + bps)
                    exit_reason = f"close_reach_{exit_target}"
                    break
            else:
                # Long exit signal: CLOSE rises to/above exit target
                if close_k >= exit_level:
                    # Exit at NEXT day's open (if available)
                    if k + 1 < len(d):
                        exit_date = idx[k+1]
                        exit_open = float(d["open"].iloc[k+1])
                        exit_price = exit_open * (1.0 - bps)  # sell to close
                    else:
                        # Last bar - exit at close
                        exit_date = idx[k]
                        exit_price = close_k * (1.0 - bps)
                    exit_reason = f"close_reach_{exit_target}"
                    break

        # If never found exit, force exit on last bar
        if exit_date is None:
            last_i = len(d) - 1
            exit_date = idx[last_i]
            if side == "short":
                exit_price = float(d["close"].iloc[last_i]) * (1.0 + bps)
            else:
                exit_price = float(d["close"].iloc[last_i]) * (1.0 - bps)
            exit_reason = "force_exit"

        # Capture mid and slope at entry
        mid_entry = float(d["mid"].iloc[j]) if pd.notna(d["mid"].iloc[j]) else float("nan")
        slope_entry = float(d["mid_slope"].iloc[j]) if pd.notna(d["mid_slope"].iloc[j]) else float("nan")

        ret = (entry_price / exit_price - 1.0) if side == "short" else (exit_price / entry_price - 1.0)
        pnl = ret  # 1 unit
        
        # Calculate position-level max drawdown
        position_dd = calculate_position_drawdown(d, entry_date, exit_date, entry_price, side)
        
        trades.append({
            "side": side,
            "signal_date": idx[i],  # Day signal was generated
            "entry_date": entry_date,  # Day trade was entered
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "ret": ret,
            "pnl": pnl,
            "position_max_dd": position_dd,  # Max drawdown during this trade
            "entry_band_k": entry_band_k,
            "exit_target": exit_target,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
            "mid_entry": mid_entry,
            "mid_slope_entry": slope_entry,
            "exit_reason": exit_reason
        })

    return pd.DataFrame(trades)

def build_equity_curve(trades_df: pd.DataFrame, initial_equity: float = 100000.0) -> pd.Series:
    """Build equity curve from trades assuming full capital deployment per trade."""
    if trades_df.empty:
        return pd.Series(dtype=float)
    
    trades = trades_df.sort_values('entry_date').copy()
    all_dates = pd.date_range(trades['entry_date'].min(), trades['exit_date'].max(), freq='D')
    equity = pd.Series(initial_equity, index=all_dates)
    
    for idx, trade in trades.iterrows():
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        ret = trade['ret']
        
        mask = (equity.index >= entry_date) & (equity.index <= exit_date)
        if mask.any():
            equity.loc[equity.index >= exit_date] *= (1 + ret)
    
    return equity

def calculate_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame) -> dict:
    """Calculate comprehensive performance metrics."""
    if equity_curve.empty or len(equity_curve) < 2:
        return {}
    
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = len(equity_curve) / 252.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr / annual_vol) if annual_vol > 0 else 0
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (cagr / downside_std) if downside_std > 0 else 0
    
    if not trades_df.empty:
        win_rate = (trades_df['ret'] > 0).sum() / len(trades_df)
        avg_win = trades_df[trades_df['ret'] > 0]['ret'].mean() if (trades_df['ret'] > 0).any() else 0
        avg_loss = trades_df[trades_df['ret'] < 0]['ret'].mean() if (trades_df['ret'] < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
    
    equity_curve.index = pd.to_datetime(equity_curve.index)
    yearly_returns = equity_curve.resample('YE').last().pct_change().dropna()
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Annual Vol': annual_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': profit_factor,
        'Best Year': yearly_returns.max() if not yearly_returns.empty else 0,
        'Worst Year': yearly_returns.min() if not yearly_returns.empty else 0,
        'Winning Years': (yearly_returns > 0).sum() if not yearly_returns.empty else 0,
        'Total Years': len(yearly_returns) if not yearly_returns.empty else 0
    }

if __name__ == "__main__":
    SYMBOL = "SPY"
    START = "2010-01-01"
    END = "2020-12-31"
    
    print(f"\n{'='*80}")
    print(f"LONG & SHORT STRATEGIES COMPARISON (2010-2020)")
    print(f"Signal: CLOSE beyond band (4:01 PM) -> Execute: NEXT day OPEN (9:30 AM)")
    print(f"Period: {START} to {END} (Full Decade)")
    print(f"STDEV_PERIOD: {STDEV_PERIOD} | N: {N} | SLIPPAGE: {SLIPPAGE_BPS} bps")
    print(f"{'='*80}\n")
    
    # Load data
    df = fetch_ohlc(SYMBOL, START, END, auto_adjust=False)
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna()
    
    # Define strategies - BOTH LONG AND SHORT
    strategies = [
        # LONG FADES
        ("LONG_LB4→Mid", "long", 4, "mid"),
        ("LONG_LB3→Mid", "long", 3, "mid"),
        ("LONG_LB2→Mid", "long", 2, "mid"),
        ("LONG_LB1→Mid", "long", 1, "mid"),
        
        # SHORT FADES
        ("SHORT_UB4→Mid", "short", 4, "mid"),
        ("SHORT_UB3→Mid", "short", 3, "mid"),
        ("SHORT_UB2→Mid", "short", 2, "mid"),
        ("SHORT_UB1→Mid", "short", 1, "mid"),
    ]
    
    results = []
    all_trades = []
    
    print("Running realistic backtests...")
    for name, side, entry_k, exit_target in strategies:
        trades = build_realistic_fading_trades(std_df, side=side, entry_band_k=entry_k, exit_target=exit_target)
        
        if trades.empty:
            results.append({
                "Strategy": name,
                "Trades": 0,
                "Win%": 0.0,
                "Avg_Ret%": 0.0,
                "Total_Ret%": 0.0,
                "Avg_Days": 0.0,
                "Best%": 0.0,
                "Worst%": 0.0
            })
        else:
            trades["strategy"] = name
            all_trades.append(trades)
            
            wins = (trades["ret"] > 0).sum()
            win_rate = wins / len(trades) * 100
            avg_ret = trades["ret"].mean() * 100
            total_ret = trades["ret"].sum() * 100
            avg_days = trades["duration_days"].mean()
            best = trades["ret"].max() * 100
            worst = trades["ret"].min() * 100
            
            # Position drawdown stats
            avg_position_dd = trades["position_max_dd"].mean() * 100
            worst_position_dd = trades["position_max_dd"].min() * 100
            
            results.append({
                "Strategy": name,
                "Trades": len(trades),
                "Win%": win_rate,
                "Avg_Ret%": avg_ret,
                "Total_Ret%": total_ret,
                "Avg_Days": avg_days,
                "Avg_PosDD%": avg_position_dd,
                "Worst_PosDD%": worst_position_dd,
                "Best%": best,
                "Worst%": worst
            })
    
    # Display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Total_Ret%", ascending=False)
    
    print("\n" + "="*120)
    print("REALISTIC FADING RESULTS (Sorted by Total Return)")
    print("="*120)
    print(results_df.to_string(index=False))
    print("="*120)
    
    # Top 5 with full metrics
    print("\n" + "="*80)
    print("TOP 5 STRATEGIES - FULL METRICS")
    print("="*80)
    
    for idx, row in results_df.head(5).iterrows():
        strat_name = row['Strategy']
        print(f"\n{'='*80}")
        print(f"{strat_name}")
        print(f"{'='*80}")
        
        strat_trades = None
        for trades in all_trades:
            if not trades.empty and trades['strategy'].iloc[0] == strat_name:
                strat_trades = trades
                break
        
        if strat_trades is not None and not strat_trades.empty:
            equity = build_equity_curve(strat_trades, INITIAL_EQUITY)
            metrics = calculate_metrics(equity, strat_trades)
            
            print(f"Trades: {row['Trades']}")
            print(f"Win Rate: {row['Win%']:.1f}%")
            print(f"Avg Return: {row['Avg_Ret%']:.2f}%")
            print(f"Total Return: {row['Total_Ret%']:.2f}%")
            print(f"Avg Duration: {row['Avg_Days']:.1f} days")
            print(f"\nPosition Drawdown (Intra-Trade):")
            print(f"  Avg Position DD: {row['Avg_PosDD%']:.2f}%")
            print(f"  Worst Position DD: {row['Worst_PosDD%']:.2f}%")
            print(f"\nPortfolio Metrics:")
            print(f"  CAGR: {metrics.get('CAGR', 0)*100:.2f}%")
            print(f"  Portfolio Max Drawdown: {metrics.get('Max Drawdown', 0)*100:.2f}%")
            print(f"  Annual Vol: {metrics.get('Annual Vol', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics.get('Sharpe', 0):.2f}")
            print(f"  Sortino Ratio: {metrics.get('Sortino', 0):.2f}")
            print(f"  Calmar Ratio: {metrics.get('Calmar', 0):.2f}")
            print(f"\nTrade Quality:")
            print(f"  Avg Win: {metrics.get('Avg Win', 0)*100:.2f}%")
            print(f"  Avg Loss: {metrics.get('Avg Loss', 0)*100:.2f}%")
            print(f"  Profit Factor: {metrics.get('Profit Factor', 0):.2f}")
            print(f"  Best Year: {metrics.get('Best Year', 0)*100:.2f}%")
            print(f"  Worst Year: {metrics.get('Worst Year', 0)*100:.2f}%")
            print(f"  Winning Years: {int(metrics.get('Winning Years', 0))}/{int(metrics.get('Total Years', 0))}")
    
    # Print trade dates for safest SHORT strategies
    print(f"\n\n{'='*80}")
    print("TRADE DATES FOR SAFEST SHORT STRATEGIES")
    print(f"{'='*80}")
    
    for trades in all_trades:
        if not trades.empty:
            strat_name = trades['strategy'].iloc[0]
            if strat_name in ['SHORT_UB3→Mid', 'SHORT_UB4→Mid']:
                print(f"\n{strat_name} ({len(trades)} trades):")
                print("-" * 80)
                for idx, trade in trades.iterrows():
                    print(f"Signal: {trade['signal_date']} | Entry: {trade['entry_date']} | Exit: {trade['exit_date']} | "
                          f"Return: {trade['ret']*100:6.2f}% | PosDD: {trade['position_max_dd']*100:6.2f}% | "
                          f"Days: {int(trade['duration_days'])}")

