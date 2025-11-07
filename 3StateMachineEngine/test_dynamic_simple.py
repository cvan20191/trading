# Simple test of dynamic rebalancing with clear P&L tracking

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import fetch_ohlc, compute_std_lines_strict, N, STDEV_PERIOD, PRICE_SRC, DDOF
from dynamic_short_rebalancer import DynamicShortRebalancer, compute_z_from_channels, compute_realized_vol_annualized


def simple_dynamic_backtest():
    """
    Simple test: just short TQQQ when z > 3, exit when z < 2.
    Track P&L clearly.
    """
    print("Loading TQQQ data...")
    df = fetch_ohlc("TQQQ", "2010-01-01", "2020-12-31", auto_adjust=False)
    std_df = compute_std_lines_strict(df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    std_df = std_df.dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "UB4"])
    
    # Compute z-scores
    ub_cols = {'UB1': std_df['UB1'], 'UB2': std_df['UB2'], 'UB3': std_df['UB3'], 'UB4': std_df['UB4']}
    z_scores = compute_z_from_channels(std_df['close'], std_df['mid'], ub_cols)
    
    # Simple strategy: short when z > 3, cover when z < 2
    cash = 100000.0
    position_shares = 0.0  # Negative when short
    
    trades = []
    equity_curve = []
    
    for date in std_df.index:
        price = float(std_df.loc[date, 'close'])
        z = z_scores.loc[date] if pd.notna(z_scores.loc[date]) else 0.0
        
        # Calculate equity
        position_value = position_shares * price
        equity = cash + position_value
        
        # Trading logic
        if abs(position_shares) < 1e-6:
            # Not in position
            if z > 3.0:
                # Enter short: sell 100 shares
                position_shares = -100.0
                cash += -position_shares * price  # Add proceeds from sale
                trades.append({
                    'date': date,
                    'action': 'SHORT',
                    'shares': position_shares,
                    'price': price,
                    'z': z,
                    'cash_after': cash
                })
        else:
            # In position
            if z < 2.0:
                # Exit: buy back shares
                cash -= -position_shares * price  # Pay to buy back
                trades.append({
                    'date': date,
                    'action': 'COVER',
                    'shares': -position_shares,
                    'price': price,
                    'z': z,
                    'cash_after': cash,
                    'pnl': equity - 100000.0
                })
                position_shares = 0.0
        
        # Recalculate equity after trade
        position_value = position_shares * price
        equity = cash + position_value
        
        equity_curve.append({
            'date': date,
            'equity': equity,
            'cash': cash,
            'position_shares': position_shares,
            'position_value': position_value,
            'z': z
        })
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"\nFirst 10 trades:")
    print(trades_df.head(10).to_string(index=False))
    
    print(f"\nFinal equity: ${equity_df['equity'].iloc[-1]:,.2f}")
    print(f"Total return: {(equity_df['equity'].iloc[-1] / 100000.0 - 1) * 100:.2f}%")
    
    # Show equity curve stats
    running_max = equity_df['equity'].expanding().max()
    drawdown = (equity_df['equity'] - running_max) / running_max
    max_dd = drawdown.min() * 100
    print(f"Max drawdown: {max_dd:.2f}%")


if __name__ == "__main__":
    simple_dynamic_backtest()

