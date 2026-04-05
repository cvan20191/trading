#!/bin/bash
# Run Roth or NonRoth dashboard with a simple menu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
else
    echo "Error: No virtual environment found (venv or env)"
    exit 1
fi

# Fix curl_cffi SSL cert path (used by yfinance 0.2.50+)
export CURL_CA_BUNDLE="$(python -c 'import certifi; print(certifi.where())')"

echo ""
echo "=============================="
echo "  Dashboard Launcher"
echo "=============================="
echo "  1)  Roth"
echo "  2)  NonRoth"
echo "=============================="
echo ""
read -p "Select [1/2]: " choice

case "$choice" in
    1)
        echo ""
        echo ">>> Running Roth Dashboard..."
        echo ""
        python "Roth_Dash&Dual.py"
        ;;
    2)
        echo ""
        echo ">>> Running NonRoth Dashboard..."
        echo ""
        python "NonRoth_Dashboard&Dual"
        ;;
    *)
        echo "Invalid choice. Enter 1 or 2."
        exit 1
        ;;
esac
