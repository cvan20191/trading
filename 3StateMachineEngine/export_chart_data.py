#!/usr/bin/env python3
"""
Convert ThinkorSwim strategy export CSV to OHLC data.
Extracts Open, High, Low, Close prices from the Strategy column.
"""

import csv
import re
import sys
from datetime import datetime
from typing import List, Dict

def convert_currency_to_decimal(value: str) -> float:
    """Remove currency symbols and convert to float."""
    cleaned = value.replace('$', '').replace('(', '-').replace(')', '').strip()
    cleaned = re.sub(r'[^-0-9.]', '', cleaned)
    return float(cleaned) if cleaned else 0.0


def extract_ohlc_data(csv_file: str) -> List[Dict]:
    """Extract OHLC data from ThinkorSwim strategy export CSV."""
    ohlc_data = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines and find lines with semicolon-delimited data
    lines = content.strip().split('\n')
    csv_lines = [line for line in lines if line.count(';') > 1]
    
    # Parse as CSV with semicolon delimiter
    reader = csv.DictReader(csv_lines, delimiter=';')
    
    for row in reader:
        strategy = row.get('Strategy', '')
        
        # Look for SOHLCP pattern: (SOHLCP|symbol|open|high|low|close|price)
        match = re.search(r'\(SOHLCP\|([^)]+)\)', strategy)
        
        if match:
            values = match.group(1).split('|')
            
            if len(values) >= 5:
                try:
                    # values[0] is symbol (e.g., SPY)
                    open_price = convert_currency_to_decimal(values[1])
                    high_price = convert_currency_to_decimal(values[2])
                    low_price = convert_currency_to_decimal(values[3])
                    close_price = convert_currency_to_decimal(values[4])
                    
                    # Parse date/time
                    date_str = row.get('Date/Time', '')
                    try:
                        dt = datetime.strptime(date_str.strip(), '%m/%d/%y')
                    except:
                        try:
                            dt = datetime.strptime(date_str.strip(), '%m/%d/%Y')
                        except:
                            print(f"Warning: Could not parse date '{date_str}', skipping row")
                            continue
                    
                    ohlc_data.append({
                        'DateTime': dt,
                        'Open': open_price,
                        'High': high_price,
                        'Low': low_price,
                        'Close': close_price
                    })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse OHLC values: {e}")
                    continue
    
    # Sort by date
    ohlc_data.sort(key=lambda x: x['DateTime'])
    
    return ohlc_data


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_chart_data.py <input_csv_file> [output_csv_file]")
        print("Example: python export_chart_data.py Chart_Data_SPY_11525.csv SPY_OHLC_Data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Reading data from: {input_file}")
    ohlc_data = extract_ohlc_data(input_file)
    
    if not ohlc_data:
        print("No OHLC data found in the file!")
        sys.exit(1)
    
    print(f"Extracted {len(ohlc_data)} rows of OHLC data")
    
    if output_file:
        # Write to CSV file
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['DateTime', 'Open', 'High', 'Low', 'Close']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in ohlc_data:
                writer.writerow({
                    'DateTime': row['DateTime'].strftime('%Y-%m-%d'),
                    'Open': f"{row['Open']:.2f}",
                    'High': f"{row['High']:.2f}",
                    'Low': f"{row['Low']:.2f}",
                    'Close': f"{row['Close']:.2f}"
                })
        
        print(f"Data written to: {output_file}")
    else:
        # Print to console
        print("\nFirst 10 rows:")
        print(f"{'DateTime':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}")
        print("-" * 55)
        for row in ohlc_data[:10]:
            print(f"{row['DateTime'].strftime('%Y-%m-%d'):<12} "
                  f"{row['Open']:>10.2f} {row['High']:>10.2f} "
                  f"{row['Low']:>10.2f} {row['Close']:>10.2f}")
        
        if len(ohlc_data) > 10:
            print(f"\n... and {len(ohlc_data) - 10} more rows")


if __name__ == '__main__':
    main()

