#!/usr/bin/env python3
"""
Show NDX channel values on overlay trade dates for ThinkorSwim verification.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/Users/christian/Documents/Projects/trading/3StateMachineEngine')

from TOS_LR_Channels_Calc_v2 import fetch_ohlc, compute_std_lines_strict

# Match dualday.py params
N = 252
STDEV_PERIOD = 12
PRICE_SRC = "hl2"

print("Loading ^NDX data...")
ndx_df = fetch_ohlc("^NDX", "2000-01-01", "2025-01-01", auto_adjust=False)

print("Computing channels with strict rolling window...")
print(f"Parameters: N={N}, STDEV_PERIOD={STDEV_PERIOD}, PRICE_SRC={PRICE_SRC}")
print()

ndx_std = compute_std_lines_strict(ndx_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC)
ndx_std = ndx_std.dropna(subset=["close", "mid", "UB1", "UB2", "UB3", "LB1", "LB2", "LB3"])

# Load SPY for filter
spy_df = fetch_ohlc("SPY", "2000-01-01", "2025-01-01", auto_adjust=False)
spy_std = compute_std_lines_strict(spy_df, n=N, stdev_len=STDEV_PERIOD, price_src=PRICE_SRC)
spy_std = spy_std.dropna(subset=["close", "mid", "UB2"])

# Create signals
ub3_close_ndx = (ndx_std["close"] >= ndx_std["UB3"]).shift(1).fillna(False)
ub2_close_exit_ndx = (ndx_std["close"] <= ndx_std["UB2"]).fillna(False)

# SPY filter: close between mid and UB2
spy_filter = ((spy_std["close"] >= spy_std["mid"]) & 
              (spy_std["close"] <= spy_std["UB2"])).reindex(ndx_std.index).fillna(False)

# Find trades
common = ndx_std.index.intersection(spy_std.index)
trades = []
in_trade = False
i = 0
idx_list = list(common)

while i < len(idx_list) - 1:
    if in_trade:
        i += 1
        continue
    
    d = idx_list[i]
    
    # Check filter and entry
    if spy_filter.get(d, False) and ub3_close_ndx.get(d, False):
        # Enter next day
        entry_date = idx_list[i+1]
        in_trade = True
        signal_date = d
        
        # Find exit
        for j in range(i+1, len(idx_list)):
            exit_check = idx_list[j]
            if ub2_close_exit_ndx.get(exit_check, False):
                if j+1 < len(idx_list):
                    exit_date = idx_list[j+1]
                else:
                    exit_date = exit_check
                
                days = (exit_date - entry_date).days
                trades.append({
                    "signal": signal_date,
                    "entry": entry_date,
                    "exit": exit_date,
                    "days": days
                })
                in_trade = False
                i = j + 1
                break

print(f"Found {len(trades)} overlay trades")
print(f"Showing first 5 trades with detailed channel values for ThinkorSwim verification:\n")
print("=" * 100)

for idx, t in enumerate(trades[:5], 1):
    signal_date = t["signal"]
    entry_date = t["entry"]
    exit_date = t["exit"]
    
    print(f"\nTRADE #{idx}")
    print("=" * 100)
    print(f"Signal Date: {signal_date.date()} (Close triggers entry at next open)")
    print(f"Entry Date:  {entry_date.date()}")
    print(f"Exit Date:   {exit_date.date()}")
    print(f"Duration:    {t['days']} days")
    print()
    
    # Show signal day details
    if signal_date in ndx_std.index:
        row = ndx_std.loc[signal_date]
        spy_row = spy_std.loc[signal_date] if signal_date in spy_std.index else None
        
        print(f"^NDX on SIGNAL DAY ({signal_date.date()}):")
        print(f"  Open:  {row['open']:.2f}")
        print(f"  High:  {row['high']:.2f}")
        print(f"  Low:   {row['low']:.2f}")
        print(f"  Close: {row['close']:.2f} ← Closed >= UB3 (triggers entry)")
        print(f"  HL/2:  {(row['high'] + row['low']) / 2:.2f}")
        print()
        print(f"  Channel values:")
        print(f"    Midline: {row['mid']:.2f}")
        print(f"    UB1:     {row['UB1']:.2f}")
        print(f"    UB2:     {row['UB2']:.2f}")
        print(f"    UB3:     {row['UB3']:.2f} ← Entry threshold")
        print(f"    UB4:     {row['UB4']:.2f}")
        print(f"    LB1:     {row['LB1']:.2f}")
        print(f"    LB2:     {row['LB2']:.2f}")
        print(f"    LB3:     {row['LB3']:.2f}")
        print()
        
        if spy_row is not None:
            print(f"SPY FILTER on SIGNAL DAY ({signal_date.date()}):")
            print(f"  Close:   {spy_row['close']:.2f}")
            print(f"  Midline: {spy_row['mid']:.2f}")
            print(f"  UB2:     {spy_row['UB2']:.2f}")
            print(f"  Filter OK: {spy_row['mid']:.2f} <= {spy_row['close']:.2f} <= {spy_row['UB2']:.2f} = TRUE ✓")
            print()
    
    # Show exit signal day
    exit_signal_date = None
    for j in range(len(idx_list)):
        if idx_list[j] >= entry_date and ub2_close_exit_ndx.get(idx_list[j], False):
            exit_signal_date = idx_list[j]
            break
    
    if exit_signal_date and exit_signal_date in ndx_std.index:
        row = ndx_std.loc[exit_signal_date]
        print(f"^NDX on EXIT SIGNAL DAY ({exit_signal_date.date()}):")
        print(f"  Open:  {row['open']:.2f}")
        print(f"  High:  {row['high']:.2f}")
        print(f"  Low:   {row['low']:.2f}")
        print(f"  Close: {row['close']:.2f} ← Closed <= UB2 (triggers exit)")
        print()
        print(f"  Channel values:")
        print(f"    UB2: {row['UB2']:.2f} ← Exit threshold")
        print(f"    UB3: {row['UB3']:.2f}")
        print()
    
    print("=" * 100)

print(f"\n\nTO VERIFY IN THINKORSWIM:")
print("1. Load ^NDX chart")
print("2. Add Mobius Linear Regression Channels study")
print("3. Set parameters: Length=252, StdLen=12, Price=HL/2")
print("4. Check the Close, UB2, UB3 values on the dates above")
print("5. Verify: Signal day Close >= UB3, Exit signal day Close <= UB2")

