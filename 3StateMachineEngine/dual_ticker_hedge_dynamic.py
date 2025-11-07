# Dual Ticker Hedge Strategy with Dynamic Rebalancing
# Compare static discrete trades vs dynamic continuous rebalancing

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


def get_band_value(df: pd.DataFrame, date, band_spec):
    """
    Get band value, supporting fractional bands like UB3.5, LB2.5, etc.
    """
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
        interpolated = lower_band + fraction * (upper_band - lower_band)
        
        return interpolated
    else:
        return float(df.loc[date, band_spec])


def build_static_trades(
    filter_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    filter_condition: dict,
    trade_entry_band: str,
    trade_exit_band: str,
    side: str = "short",
    slippage_bps: float = SLIPPAGE_BPS
):
    """
    Build static discrete trades (original logic).
    """
    common_dates = filter_df.index.intersection(trade_df.index)
    filter_aligned = filter_df.loc[common_dates].copy()
    trade_aligned = trade_df.loc[common_dates].copy()
    
    filter_aligned = filter_aligned.dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "UB4"])
    trade_aligned = trade_aligned.dropna(subset=["open", "close", "mid", "UB1", "UB2", "UB3", "UB4"])
    
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
        
        # Check filter condition
        filter_close = float(filter_aligned.loc[current_date, "close"])
        filter_min = get_band_value(filter_aligned, current_date, filter_condition["min_band"])
        filter_max = get_band_value(filter_aligned, current_date, filter_condition["max_band"])
        
        filter_ok = (filter_close >= filter_min) and (filter_close <= filter_max)
        
        if not filter_ok:
            i += 1
            continue
        
        # Check trade entry condition
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
        
        # Search for exit
        exit_date = None
        exit_price = None
        position_max_dd = 0.0
        
        for k in range(j, len(idx)):
            exit_check_date = idx[k]
            trade_close_k = float(trade_aligned.loc[exit_check_date, "close"])
            trade_exit_level = get_band_value(trade_aligned, exit_check_date, trade_exit_band)
            
            # Track position drawdown
            if side == "short":
                worst_price = float(trade_aligned.loc[exit_check_date, "high"])
                dd = (entry_price - worst_price) / entry_price
                position_max_dd = min(position_max_dd, dd)
            else:
                worst_price = float(trade_aligned.loc[exit_check_date, "low"])
                dd = (worst_price - entry_price) / entry_price
                position_max_dd = min(position_max_dd, dd)
            
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
        
        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "ret": ret,
            "position_max_dd": position_max_dd,
            "duration_days": (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days,
        })
        
        i += 1
    
    return pd.DataFrame(trades)


def build_dynamic_trades(
    filter_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    filter_condition: dict,
    rebalancer: DynamicShortRebalancer,
    initial_equity: float = 100000.0,
    slippage_bps: float = SLIPPAGE_BPS
):
    """
    Build dynamic continuously rebalanced trades.
    
    Returns:
        (equity_curve, trades_log, metrics)
    """
    common_dates = filter_df.index.intersection(trade_df.index)
    filter_aligned = filter_df.loc[common_dates].copy()
    trade_aligned = trade_df.loc[common_dates].copy()
    
    filter_aligned = filter_aligned.dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "UB4"])
    trade_aligned = trade_aligned.dropna(subset=["open", "close", "mid", "UB1", "UB2", "UB3", "UB4"])
    
    common_dates = filter_aligned.index.intersection(trade_aligned.index)
    filter_aligned = filter_aligned.loc[common_dates]
    trade_aligned = trade_aligned.loc[common_dates]
    
    # Compute z-scores for trade ticker
    ub_cols = {
        'UB1': trade_aligned['UB1'],
        'UB2': trade_aligned['UB2'],
        'UB3': trade_aligned['UB3'],
        'UB4': trade_aligned['UB4']
    }
    z_scores = compute_z_from_channels(trade_aligned['close'], trade_aligned['mid'], ub_cols)
    
    # Compute returns and volatility for trade ticker
    returns = trade_aligned['close'].pct_change()
    ann_vol = compute_realized_vol_annualized(returns, window=20)
    
    # Initialize state
    cash = initial_equity  # Cash account
    current_shares = 0.0   # Negative for short positions
    current_weight = 0.0
    position_entry_value = 0.0  # Total value when position was opened
    position_max_dd = 0.0
    days_in_trade = 0
    
    equity_curve = []
    trades_log = []
    bps = slippage_bps / 10000.0
    
    idx = list(common_dates)
    
    for i, date in enumerate(idx):
        # Check filter condition
        filter_close = float(filter_aligned.loc[date, "close"])
        filter_min = get_band_value(filter_aligned, date, filter_condition["min_band"])
        filter_max = get_band_value(filter_aligned, date, filter_condition["max_band"])
        filter_ok = (filter_close >= filter_min) and (filter_close <= filter_max)
        
        # Get current state
        price = float(trade_aligned.loc[date, "close"])
        z = z_scores.loc[date] if pd.notna(z_scores.loc[date]) else 0.0
        vol = ann_vol.loc[date] if pd.notna(ann_vol.loc[date]) else 0.5
        
        # Calculate current equity
        # For short positions: current_shares is negative
        # Equity = cash + value_of_short_position
        # value_of_short_position = -abs(shares) * price (it's a liability)
        if abs(current_shares) > 1e-6:
            position_market_value = current_shares * price  # Negative for short
            equity = cash + position_market_value  # Cash + (negative liability) = cash - liability
            
            days_in_trade += 1
            
            # Track drawdown: compare current position value to entry value
            # For short: we want (entry_value - current_market_value) / entry_value
            # If we sold at $100 (entry_value = -100 shares * $50 = -$5000)
            # And price goes to $60 (current = -100 * $60 = -$6000)
            # Loss = -$6000 - (-$5000) = -$1000
            # DD% = -$1000 / $5000 = -20%
            position_pnl_dollars = position_entry_value - abs(position_market_value)
            position_dd_pct = position_pnl_dollars / position_entry_value if position_entry_value > 0 else 0
            position_max_dd = min(position_max_dd, position_dd_pct)
        else:
            equity = cash
            position_max_dd = 0.0
            days_in_trade = 0
        
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
        
        # Execute trade if needed
        if abs(shares_to_trade) > 1e-6:
            # Apply slippage
            if shares_to_trade < 0:
                # Adding to short (selling) - we receive cash
                fill_price = price * (1.0 - bps)
                cash_flow = -shares_to_trade * fill_price  # Positive (we receive cash)
                cash += cash_flow
            else:
                # Covering short (buying) - we pay cash
                fill_price = price * (1.0 + bps)
                cash_flow = shares_to_trade * fill_price  # Positive (we pay cash)
                cash -= cash_flow
            
            # Log trade
            trades_log.append({
                "date": date,
                "action": "add_short" if shares_to_trade < 0 else "cover_short",
                "shares": shares_to_trade,
                "price": fill_price,
                "z_score": z,
                "target_weight": target_w,
                "current_weight": current_weight,
                "equity": equity,
                "cash_flow": cash_flow if shares_to_trade < 0 else -cash_flow
            })
            
            # Update position
            old_shares = current_shares
            current_shares += shares_to_trade
            
            # Track entry value for new/added positions
            if abs(old_shares) < 1e-6 and abs(current_shares) > 1e-6:
                # Opening new position
                position_entry_value = abs(current_shares * fill_price)
            elif abs(current_shares) > abs(old_shares):
                # Adding to position - update weighted average entry value
                added_value = abs(shares_to_trade * fill_price)
                position_entry_value += added_value
            
            current_weight = target_w
            
            # If fully closed, reset
            if abs(current_shares) < 1e-6:
                current_shares = 0.0
                current_weight = 0.0
                position_entry_value = 0.0
                position_max_dd = 0.0
                days_in_trade = 0
        
        # Recalculate equity for curve
        if abs(current_shares) > 1e-6:
            position_value = current_shares * price
            equity_now = cash - position_value
        else:
            equity_now = cash
        
        equity_curve.append({
            "date": date,
            "equity": equity_now,
            "position_weight": current_weight,
            "z_score": z,
            "cash": cash,
            "position_value": current_shares * price if abs(current_shares) > 1e-6 else 0
        })
    
    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades_log)
    
    # Calculate metrics
    if len(equity_df) > 0:
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1.0) * 100
        
        # Calculate drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        # Count rebalances
        num_rebalances = len(trades_df)
        
        metrics = {
            "Final_Equity": final_equity,
            "Total_Return%": total_return,
            "Max_DD%": max_dd,
            "Num_Rebalances": num_rebalances
        }
    else:
        metrics = {}
    
    return equity_df, trades_df, metrics


def compare_static_vs_dynamic(
    filter_ticker: str,
    trade_ticker: str,
    start_date: str,
    end_date: str,
    filter_condition: dict,
    static_strategies: list,
    dynamic_config: dict = None
):
    """
    Compare static discrete trades vs dynamic rebalancing.
    """
    print(f"\n{'='*100}")
    print(f"DUAL TICKER HEDGE: STATIC vs DYNAMIC COMPARISON")
    print(f"{'='*100}")
    print(f"Filter: {filter_ticker} (between {filter_condition['min_band']} and {filter_condition['max_band']})")
    print(f"Trade: {trade_ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*100}\n")
    
    # Load data
    print(f"Loading {filter_ticker} data...")
    filter_df = fetch_ohlc(filter_ticker, start_date, end_date, auto_adjust=False)
    filter_std = compute_std_lines_strict(filter_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    
    print(f"Loading {trade_ticker} data...")
    trade_df = fetch_ohlc(trade_ticker, start_date, end_date, auto_adjust=False)
    trade_std = compute_std_lines_strict(trade_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC, ddof=DDOF)
    
    # Run static strategies
    print(f"\n{'='*100}")
    print("STATIC STRATEGIES (Discrete Trades)")
    print(f"{'='*100}\n")
    
    static_results = []
    for name, entry_band, exit_band, side in static_strategies:
        print(f"  Running: {name}...")
        trades = build_static_trades(
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
        
        static_results.append({
            "Strategy": name,
            "Trades": len(trades),
            "Win%": win_rate,
            "Total_Ret%": total_ret,
            "Avg_Days": avg_days,
            "Worst_PosDD%": worst_position_dd,
            "Calmar": calmar
        })
    
    static_df = pd.DataFrame(static_results)
    if not static_df.empty:
        static_df = static_df.sort_values("Calmar", ascending=False)
        print(f"\n{static_df.to_string(index=False)}\n")
    
    # Run dynamic strategy
    print(f"\n{'='*100}")
    print("DYNAMIC STRATEGY (Continuous Rebalancing)")
    print(f"{'='*100}\n")
    
    if dynamic_config is None:
        dynamic_config = {}
    
    rebalancer = DynamicShortRebalancer(
        w_beta=dynamic_config.get('w_beta', 0.35),
        w_max=dynamic_config.get('w_max', 0.40),
        vol_target=dynamic_config.get('vol_target', 0.60),
        drift_thresh=dynamic_config.get('drift_thresh', 0.05),
        dd_steps=dynamic_config.get('dd_steps', (-0.07, -0.10, -0.12)),
        tmax1=dynamic_config.get('tmax1', 25),
        tmax2=dynamic_config.get('tmax2', 50)
    )
    
    print(f"Config: w_beta={rebalancer.w_beta}, w_max={rebalancer.w_max}, vol_target={rebalancer.vol_target}")
    print(f"        DD steps={rebalancer.dd_steps}, Time stops=({rebalancer.tmax1}, {rebalancer.tmax2})\n")
    
    equity_curve, trades_log, metrics = build_dynamic_trades(
        filter_std, trade_std, filter_condition, rebalancer
    )
    
    if metrics:
        print(f"Final Equity: ${metrics['Final_Equity']:,.2f}")
        print(f"Total Return: {metrics['Total_Return%']:.2f}%")
        print(f"Max Drawdown: {metrics['Max_DD%']:.2f}%")
        print(f"Num Rebalances: {metrics['Num_Rebalances']}")
        print(f"Calmar Ratio: {metrics['Total_Return%'] / abs(metrics['Max_DD%']):.2f}")
        
        # Debug: show first few trades
        if len(trades_log) > 0:
            print(f"\nFirst 10 rebalances:")
            print(trades_log.head(10).to_string(index=False))
    
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")
    
    if not static_df.empty:
        best_static = static_df.iloc[0]
        print(f"Best Static:  {best_static['Strategy']}")
        print(f"              Total Return: {best_static['Total_Ret%']:.2f}%, Calmar: {best_static['Calmar']:.2f}")
        print(f"              Trades: {int(best_static['Trades'])}, Worst DD: {best_static['Worst_PosDD%']:.2f}%\n")
    
    if metrics:
        print(f"Dynamic:      Continuous Rebalancing")
        print(f"              Total Return: {metrics['Total_Return%']:.2f}%, Calmar: {metrics['Total_Return%'] / abs(metrics['Max_DD%']):.2f}")
        print(f"              Rebalances: {metrics['Num_Rebalances']}, Max DD: {metrics['Max_DD%']:.2f}%\n")
    
    return static_df, equity_curve, trades_log, metrics


if __name__ == "__main__":
    # Configuration
    FILTER_TICKER = "SPY"
    TRADE_TICKER = "TQQQ"
    START = "2010-01-01"
    END = "2020-12-31"
    
    FILTER_CONDITION = {
        "min_band": "mid",
        "max_band": "UB2"
    }
    
    # Static strategies to compare
    STATIC_STRATEGIES = [
        ("STATIC_UB3.5→UB2", "UB3.5", "UB2", "short"),
        ("STATIC_UB3→UB2", "UB3", "UB2", "short"),
        ("STATIC_UB2.5→UB1.5", "UB2.5", "UB1.5", "short"),
        ("STATIC_UB2.5→UB2", "UB2.5", "UB2", "short"),
    ]
    
    # Dynamic config
    DYNAMIC_CONFIG = {
        'w_beta': 0.35,
        'w_max': 0.40,
        'vol_target': 0.60,
        'drift_thresh': 0.05,
        'dd_steps': (-0.07, -0.10, -0.12),
        'tmax1': 25,
        'tmax2': 50
    }
    
    static_results, equity_curve, trades_log, dynamic_metrics = compare_static_vs_dynamic(
        FILTER_TICKER, TRADE_TICKER, START, END,
        FILTER_CONDITION, STATIC_STRATEGIES, DYNAMIC_CONFIG
    )

