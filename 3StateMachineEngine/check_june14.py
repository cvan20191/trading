#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import numpy as np

print("Fetching SPY data...")
spy = yf.download("SPY", start="2022-01-01", end="2023-12-31", progress=False)

# Compute HL/2
spy['hl2'] = (spy['High'] + spy['Low']) / 2

# Linear regression midline (n=252)
def lr_mid_at_bar(series, n=252):
    result = []
    for i in range(len(series)):
        if i < n - 1:
            result.append(np.nan)
        else:
            window = series.iloc[i-n+1:i+1].values
            x = np.arange(n)
            coeffs = np.polyfit(x, window, 1)
            result.append(coeffs[0] * (n - 1) + coeffs[1])
    return pd.Series(result, index=series.index)

print("Computing midline...")
spy['mid'] = lr_mid_at_bar(spy['hl2'], n=252)

# StDev(12) rolling
spy['stdev12'] = spy['hl2'].rolling(12, min_periods=12).std()

# HighestAll over rolling 241 window (252-12+1)
spy['sigma'] = spy['stdev12'].rolling(241, min_periods=241).max()

# Bands
for k in range(1, 5):
    spy[f'UB{k}'] = spy['mid'] + k * spy['sigma']
    spy[f'LB{k}'] = spy['mid'] - k * spy['sigma']

# Check June 14, 2023
date = pd.Timestamp('2023-06-14')
if date in spy.index:
    row = spy.loc[date]
    print()
    print("=" * 80)
    print(f"SPY on {date.date()} - STANDALONE ORIGINAL CODE")
    print("=" * 80)
    print(f"Close:   {float(row['Close']):.2f}")
    print(f"HL/2:    {float(row['hl2']):.2f}")
    print(f"Midline: {float(row['mid']):.2f}")
    print(f"Sigma:   {float(row['sigma']):.2f}")
    print(f"UB1:     {float(row['UB1']):.2f}")
    print(f"UB2:     {float(row['UB2']):.2f}")
    print(f"UB3:     {float(row['UB3']):.2f}")
    print(f"LB1:     {float(row['LB1']):.2f}")
    print(f"LB2:     {float(row['LB2']):.2f}")
    print(f"LB3:     {float(row['LB3']):.2f}")
    print()
    print("COMPARISON TO TOS:")
    print(f"  Midline: TOS=411.56 | Python={float(row['mid']):.2f} | Diff={float(row['mid'])-411.56:.2f}")
    print(f"  Sigma:   TOS=18.91  | Python={float(row['sigma']):.2f} | Diff={float(row['sigma'])-18.91:.2f}")
    print(f"  UB2:     TOS=449.38 | Python={float(row['UB2']):.2f} | Diff={float(row['UB2'])-449.38:.2f}")
    print(f"  LB2:     TOS=373.74 | Python={float(row['LB2']):.2f} | Diff={float(row['LB2'])-373.74:.2f}")
    print("=" * 80)

