#!/bin/bash
# Activate the project venv and run the momentum backtest script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
else
    echo "Error: No virtual environment found (venv or env)"
    exit 1
fi

python momentum.py
