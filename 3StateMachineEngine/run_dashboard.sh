#!/bin/bash
# Run dashboard and dual strategy dashboard
# This script activates the virtual environment and runs the dashboard

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment (try venv first, then env)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
else
    echo "Error: No virtual environment found (venv or env)"
    exit 1
fi

# Run the dashboard
python dashboardAndDual

