# Dual Ticker Hedge with Dynamic Rebalancing - Clean Implementation

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import (
    fetch_ohlc, compute_std_lines_strict,
    N, STDEV_PERIOD, PRICE_SRC, DDOF, SLIPPAGE_BPS
)
from dynamic_short_rebalancer import (
    DynamicShortRebalancer,
    compute_z_from_channels,
    compute_realized_vol_annualized
)


def get_band_value(df, date, band_spec):
    """Get band value supporting fractional bands."""
    if band_spec == "mid":
        return float(df.loc[date, "mid"])
    
    if "." in band_spec:
        if band_spec.startswith("UB"):
            band_type = "UB"
            level = float(band_spec[2:])
        elif band_spec.startswith("LB"):
            band_type = "LB"
            level = float(band_spec[2:])
        else:
            raise ValueError(f"Invalid band specification: {band_spec}")
        
        lower_level = int(np.floor(level))
        upper_level = int(np.ceil(level))
        
        if lower_level == upper_level:
            return float(df.loc[date, f"{band_type}{lower_level}"])
        
        lower_band = float(df.loc[date, f"{band_type}{lower_level}"])
        upper_band = float(df.loc[date, f"{band_type}{upper_level}"])
        
        fraction = level - lower_level
        return lower_band + fraction * (upper_band - lower_band)
    else:
        return float(df.loc[date, band_spec])


def backtest_dynamic_hedge(
    filter_df,
    trade_df,
    filter_condition,
    rebalancer,
    initial_capital=100000.0
):
    """
    Backtest dynamic hedging strategy with proper P&L tracking.
    
    Returns:
        equity_curve, trades_log, metrics
    """
    # Align dataframes
    common_dates = filter_df.index.intersection(trade_df.index)
    filter_df = filter_df.loc[common_dates].dropna(subset=["close", "mid", "UB1", "UB2"])
    trade_df = trade_df.loc[common_dates].dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "UB4"])
    
    common_dates = filter_df.index.intersection(trade_df.index)
    filter_df = filter_df.loc[common_dates]
    trade_df = trade_df.loc[common_dates]
    
    # Compute z-scores
    ub_cols = {f'UB{i}': trade_df[f'UB{i}'] for i in range(1, 5)}
    z_scores = compute_z_from_channels(trade_df['close'], trade_df['mid'], ub_cols)
    
    # Compute volatility
    returns = trade_df['close'].pct_change()
    ann_vol = compute_realized_vol_annualized(returns, window=20)
    
    # Initialize
    cash = initial_capital
    short_shares = 0.0  # Number of shares short (stored as positive number)
    short_entry_price = 0.0  # Average entry price for short
    position_max_dd = 0.0
    days_in_trade = 0
    
    equity_curve = []
    trades_log = []
    bps = SLIPPAGE_BPS / 10000.0
    
    for date in trade_df.index:
        # Get current prices and indicators
        price = float(trade_df.loc[date, 'close'])
        z = z_scores.loc[date] if pd.notna(z_scores.loc[date]) else 0.0
        vol = ann_vol.loc[date] if pd.notna(ann_vol.loc[date]) else 0.5
        
        # Check filter condition
        filter_close = float(filter_df.loc[date, 'close'])
        filter_min = get_band_value(filter_df, date, filter_condition['min_band'])
        filter_max = get_band_value(filter_df, date, filter_condition['max_band'])
        filter_ok = (filter_close >= filter_min) and (filter_close <= filter_max)
        
        # Calculate current equity
        # Short position value: we owe short_shares * price
        # Equity = cash - (short_shares * price)
        if short_shares > 0:
            short_market_value = short_shares * price
            equity = cash - short_market_value
            
            days_in_trade += 1
            
            # Track drawdown
            # P&L = (entry_price - current_price) * shares
            # DD% = worst P&L / (entry_price * shares)
            pnl = (short_entry_price - price) * short_shares
            dd_pct = pnl / (short_entry_price * short_shares) if short_shares > 0 else 0
            position_max_dd = min(position_max_dd, dd_pct)
        else:
            equity = cash
            position_max_dd = 0.0
            days_in_trade = 0
        
        # Calculate current weight
        current_weight = (short_shares * price) / equity if equity > 0 and short_shares > 0 else 0.0
        
        # Determine rebalancing action
        shares_to_trade, target_w = rebalancer.rebalance(
            equity=equity,
            price=price,
            z=z,
            ann_vol=vol,
            filter_ok=filter_ok,
            current_weight=current_weight,
            pos_max_drawdown=position_max_dd,
            days_in_trade=days_in_trade
        )
        
        # Execute trade
        if abs(shares_to_trade) > 1e-6:
            if shares_to_trade < 0:
                # Adding to short (selling more)
                shares_to_sell = -shares_to_trade
                fill_price = price * (1.0 - bps)
                cash_received = shares_to_sell * fill_price
                cash += cash_received
                
                # Update weighted average entry price
                if short_shares > 0:
                    total_cost = short_shares * short_entry_price + shares_to_sell * fill_price
                    short_shares += shares_to_sell
                    short_entry_price = total_cost / short_shares
                else:
                    short_shares = shares_to_sell
                    short_entry_price = fill_price
                
                trades_log.append({
                    'date': date,
                    'action': 'ADD_SHORT',
                    'shares': shares_to_sell,
                    'price': fill_price,
                    'z': z,
                    'target_weight': target_w,
                    'equity': equity
                })
            else:
                # Covering short (buying back)
                shares_to_buy = shares_to_trade
                fill_price = price * (1.0 + bps)
                cash_paid = shares_to_buy * fill_price
                cash -= cash_paid
                
                # Realize P&L on covered portion
                covered_pnl = (short_entry_price - fill_price) * shares_to_buy
                
                short_shares -= shares_to_buy
                
                if short_shares < 1e-6:
                    short_shares = 0.0
                    short_entry_price = 0.0
                    position_max_dd = 0.0
                    days_in_trade = 0
                
                trades_log.append({
                    'date': date,
                    'action': 'COVER_SHORT',
                    'shares': shares_to_buy,
                    'price': fill_price,
                    'z': z,
                    'target_weight': target_w,
                    'equity': equity,
                    'realized_pnl': covered_pnl
                })
        
        # Record equity curve
        if short_shares > 0:
            final_equity = cash - (short_shares * price)
        else:
            final_equity = cash
        
        equity_curve.append({
            'date': date,
            'equity': final_equity,
            'cash': cash,
            'short_shares': short_shares,
            'short_value': short_shares * price if short_shares > 0 else 0,
            'z': z
        })
    
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades_log)
    
    # Calculate metrics
    if len(equity_df) > 0:
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_capital - 1.0) * 100
        
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        metrics = {
            'Final_Equity': final_equity,
            'Total_Return%': total_return,
            'Max_DD%': max_dd,
            'Num_Rebalances': len(trades_df),
            'Calmar': total_return / abs(max_dd) if max_dd != 0 else 0
        }
    else:
        metrics = {}
    
    return equity_df, trades_df, metrics


def main():
    print("\n" + "="*100)
    print("DUAL TICKER DYNAMIC HEDGE BACKTEST")
    print("="*100)
    
    # Load data
    print("\nLoading SPY data...")
    spy_df = fetch_ohlc("SPY", "2010-01-01", "2020-12-31", auto_adjust=False)
    spy_std = compute_std_lines_strict(spy_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    
    print("Loading TQQQ data...")
    tqqq_df = fetch_ohlc("TQQQ", "2010-01-01", "2020-12-31", auto_adjust=False)
    tqqq_std = compute_std_lines_strict(tqqq_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    
    # Filter condition
    filter_condition = {
        'min_band': 'mid',
        'max_band': 'UB2'
    }
    
    # Create rebalancer
    rebalancer = DynamicShortRebalancer(
        w_beta=1.00,        # Match static strategies (100% capital)
        w_max=1.00,         # Allow full position
        vol_target=0.60,
        drift_thresh=0.05,
        dd_steps=(-0.10, -0.15, -0.20),  # More tolerant of drawdowns
        tmax1=50,           # Increased from 25
        tmax2=100           # Increased from 50
    )
    
    print(f"\nRebalancer config:")
    print(f"  w_beta={rebalancer.w_beta}, w_max={rebalancer.w_max}")
    print(f"  vol_target={rebalancer.vol_target}, drift_thresh={rebalancer.drift_thresh}")
    print(f"  DD stops={rebalancer.dd_steps}, Time stops=({rebalancer.tmax1}, {rebalancer.tmax2})")
    
    # Run backtest
    print("\nRunning backtest...")
    equity_curve, trades_log, metrics = backtest_dynamic_hedge(
        spy_std, tqqq_std, filter_condition, rebalancer
    )
    
    # Print results
    print("\n" + "="*100)
    print("RESULTS")
    print("="*100)
    
    if metrics:
        print(f"\nFinal Equity: ${metrics['Final_Equity']:,.2f}")
        print(f"Total Return: {metrics['Total_Return%']:.2f}%")
        print(f"Max Drawdown: {metrics['Max_DD%']:.2f}%")
        print(f"Num Rebalances: {metrics['Num_Rebalances']}")
        print(f"Calmar Ratio: {metrics['Calmar']:.2f}")
        
        if len(trades_log) > 0:
            print(f"\nFirst 10 rebalances:")
            print(trades_log.head(10)[['date', 'action', 'shares', 'price', 'z', 'target_weight']].to_string(index=False))
            
            print(f"\nLast 10 rebalances:")
            print(trades_log.tail(10)[['date', 'action', 'shares', 'price', 'z', 'target_weight']].to_string(index=False))
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()

