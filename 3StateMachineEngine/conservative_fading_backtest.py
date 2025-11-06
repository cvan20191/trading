# Conservative Fading Backtest
# Testing partial reversion strategies: extreme bands fade back toward mean

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc, compute_std_lines_strict, build_pyramiding_trades,
    N, STDEV_PERIOD, PRICE_SRC, DDOF, SLIPPAGE_BPS, INITIAL_EQUITY
)

def build_equity_curve(trades_df: pd.DataFrame, initial_equity: float = 100000.0) -> pd.Series:
    """Build equity curve from trades assuming full capital deployment per trade."""
    if trades_df.empty:
        return pd.Series(dtype=float)
    
    # Sort by entry date
    trades = trades_df.sort_values('entry_date').copy()
    
    # Create daily equity series
    all_dates = pd.date_range(trades['entry_date'].min(), trades['exit_date'].max(), freq='D')
    equity = pd.Series(initial_equity, index=all_dates)
    
    # Track active trades and their returns
    for idx, trade in trades.iterrows():
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        ret = trade['ret']
        
        # Apply return on exit date
        mask = (equity.index >= entry_date) & (equity.index <= exit_date)
        if mask.any():
            # Compound the return
            equity.loc[equity.index >= exit_date] *= (1 + ret)
    
    return equity

def calculate_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame) -> dict:
    """Calculate comprehensive performance metrics."""
    if equity_curve.empty or len(equity_curve) < 2:
        return {}
    
    # Returns
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Time period
    years = len(equity_curve) / 252.0
    
    # CAGR
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe (assuming 0% risk-free rate)
    sharpe = (cagr / annual_vol) if annual_vol > 0 else 0
    
    # Max Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    
    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (cagr / downside_std) if downside_std > 0 else 0
    
    # Trade stats
    if not trades_df.empty:
        win_rate = (trades_df['ret'] > 0).sum() / len(trades_df)
        avg_win = trades_df[trades_df['ret'] > 0]['ret'].mean() if (trades_df['ret'] > 0).any() else 0
        avg_loss = trades_df[trades_df['ret'] < 0]['ret'].mean() if (trades_df['ret'] < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
    
    # Yearly returns
    equity_curve.index = pd.to_datetime(equity_curve.index)
    yearly_returns = equity_curve.resample('Y').last().pct_change().dropna()
    
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
    START = "2015-09-01"
    END = "2020-10-01"
    
    print(f"\n{'='*80}")
    print(f"CONSERVATIVE FADING BACKTEST")
    print(f"Testing partial reversion from extreme bands")
    print(f"Period: {START} to {END}")
    print(f"STDEV_PERIOD: {STDEV_PERIOD} | N: {N} | SLIPPAGE: {SLIPPAGE_BPS} bps")
    print(f"{'='*80}\n")
    
    # Load data
    df = fetch_ohlc(SYMBOL, START, END, auto_adjust=False)
    
    # Compute channels
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna()
    
    # Define conservative fading strategies
    strategies = [
        # ===== LONG FADES (from extreme lows back up) =====
        ("LONG_LB4→LB3", "long", 4, "LB3"),
        ("LONG_LB4→LB2", "long", 4, "LB2"),
        ("LONG_LB4→LB1", "long", 4, "LB1"),
        ("LONG_LB4→Mid", "long", 4, "mid"),
        
        ("LONG_LB3→LB2", "long", 3, "LB2"),
        ("LONG_LB3→LB1", "long", 3, "LB1"),
        ("LONG_LB3→Mid", "long", 3, "mid"),
        
        # ===== SHORT FADES (from extreme highs back down) =====
        ("SHORT_UB4→UB3", "short", 4, "UB3"),
        ("SHORT_UB4→UB2", "short", 4, "UB2"),
        ("SHORT_UB4→UB1", "short", 4, "UB1"),
        ("SHORT_UB4→Mid", "short", 4, "mid"),
        
        ("SHORT_UB3→UB2", "short", 3, "UB2"),
        ("SHORT_UB3→UB1", "short", 3, "UB1"),
        ("SHORT_UB3→Mid", "short", 3, "mid"),
    ]
    
    results = []
    all_trades = []
    
    print("Running backtests...")
    for name, side, entry_k, exit_target in strategies:
        trades = build_pyramiding_trades(std_df, side=side, entry_band_k=entry_k, exit_target=exit_target)
        
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
            
            results.append({
                "Strategy": name,
                "Trades": len(trades),
                "Win%": win_rate,
                "Avg_Ret%": avg_ret,
                "Total_Ret%": total_ret,
                "Avg_Days": avg_days,
                "Best%": best,
                "Worst%": worst
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Total_Ret%", ascending=False)
    
    print("\n" + "="*120)
    print("CONSERVATIVE FADING RESULTS (Sorted by Total Return)")
    print("="*120)
    print(results_df.to_string(index=False))
    print("="*120)
    
    # Separate long and short results
    print("\n" + "="*80)
    print("LONG FADES (Buy extreme lows, sell on bounce)")
    print("="*80)
    long_results = results_df[results_df["Strategy"].str.startswith("LONG")]
    print(long_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("SHORT FADES (Sell extreme highs, cover on pullback)")
    print("="*80)
    short_results = results_df[results_df["Strategy"].str.startswith("SHORT")]
    print(short_results.to_string(index=False))
    
    # Top performers with full metrics
    print("\n" + "="*80)
    print("TOP 5 CONSERVATIVE FADING STRATEGIES - FULL METRICS")
    print("="*80)
    top5 = results_df.head(5)
    
    for idx, row in top5.iterrows():
        strat_name = row['Strategy']
        print(f"\n{'='*80}")
        print(f"{strat_name}")
        print(f"{'='*80}")
        
        # Find trades for this strategy
        strat_trades = None
        for trades in all_trades:
            if not trades.empty and trades['strategy'].iloc[0] == strat_name:
                strat_trades = trades
                break
        
        if strat_trades is not None and not strat_trades.empty:
            # Build equity curve and calculate metrics
            equity = build_equity_curve(strat_trades, INITIAL_EQUITY)
            metrics = calculate_metrics(equity, strat_trades)
            
            print(f"Trades: {row['Trades']}")
            print(f"Win Rate: {row['Win%']:.1f}%")
            print(f"Avg Return: {row['Avg_Ret%']:.2f}%")
            print(f"Total Return: {row['Total_Ret%']:.2f}%")
            print(f"Avg Duration: {row['Avg_Days']:.1f} days")
            print(f"\nRisk-Adjusted Metrics:")
            print(f"  CAGR: {metrics.get('CAGR', 0)*100:.2f}%")
            print(f"  Max Drawdown: {metrics.get('Max Drawdown', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics.get('Sharpe', 0):.2f}")
            print(f"  Sortino Ratio: {metrics.get('Sortino', 0):.2f}")
            print(f"  Calmar Ratio: {metrics.get('Calmar', 0):.2f}")
            print(f"  Annual Volatility: {metrics.get('Annual Vol', 0)*100:.2f}%")
            print(f"\nTrade Quality:")
            print(f"  Avg Win: {metrics.get('Avg Win', 0)*100:.2f}%")
            print(f"  Avg Loss: {metrics.get('Avg Loss', 0)*100:.2f}%")
            print(f"  Profit Factor: {metrics.get('Profit Factor', 0):.2f}")
            print(f"  Best Trade: {row['Best%']:.2f}%")
            print(f"  Worst Trade: {row['Worst%']:.2f}%")
            print(f"\nYearly Performance:")
            print(f"  Best Year: {metrics.get('Best Year', 0)*100:.2f}%")
            print(f"  Worst Year: {metrics.get('Worst Year', 0)*100:.2f}%")
            print(f"  Winning Years: {int(metrics.get('Winning Years', 0))}/{int(metrics.get('Total Years', 0))}")
        else:
            print(f"Trades: {row['Trades']}")
            print(f"Win Rate: {row['Win%']:.1f}%")
            print(f"Avg Return: {row['Avg_Ret%']:.2f}%")
            print(f"Total Return: {row['Total_Ret%']:.2f}%")
    
    # Export all trades
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        combined_trades = combined_trades.sort_values("entry_date")
        
        csv_path = f"conservative_fading_trades_{df.index[0].date()}_{df.index[-1].date()}.csv"
        combined_trades.to_csv(csv_path, index=False)
        print(f"\n✓ Exported {len(combined_trades)} trades to: {csv_path}")
        
        # Key insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        # Best strategy by category
        best_long = long_results.iloc[0] if not long_results.empty else None
        best_short = short_results.iloc[0] if not short_results.empty else None
        
        if best_long is not None:
            print(f"\n🏆 Best Long Fade: {best_long['Strategy']}")
            print(f"   {best_long['Trades']} trades | {best_long['Win%']:.1f}% win rate | {best_long['Total_Ret%']:.2f}% total return")
        
        if best_short is not None:
            print(f"\n🏆 Best Short Fade: {best_short['Strategy']}")
            print(f"   {best_short['Trades']} trades | {best_short['Win%']:.1f}% win rate | {best_short['Total_Ret%']:.2f}% total return")
        
        # Compare aggressive vs conservative exits
        print("\n📊 Exit Target Comparison:")
        print("   Conservative exits (closer to entry) = faster, smaller gains, higher frequency")
        print("   Aggressive exits (toward mean) = slower, larger gains, same frequency")

