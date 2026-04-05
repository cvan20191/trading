#!/usr/bin/env bash
# Start macro-dashboard: FastAPI backend (8000) + Vite frontend (5173).
# API calls from the UI go to /api → proxied to localhost:8000 (see vite.config.ts).
#
# Prerequisites:
#   - backend: Python 3.11+, dependencies (pip install -r requirements.txt), backend/.env
#   - frontend: npm install in frontend/
#
# Usage:
#   ./run-dev.sh              # from macro-dashboard/
#   bash macro-dashboard/run-dev.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"

# Must match frontend/vite.config.ts proxy target (default 8000).
BACKEND_PORT=8000
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

die() { echo "error: $*" >&2; exit 1; }

[[ -d "$BACKEND" ]] || die "missing backend dir: $BACKEND"
[[ -d "$FRONTEND" ]] || die "missing frontend dir: $FRONTEND"

if [[ ! -f "$BACKEND/.env" ]]; then
  echo "warning: $BACKEND/.env not found — copy .env.example and set at least OPENAI_API_KEY" >&2
fi

# Prefer project venv if present
if [[ -f "$BACKEND/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$BACKEND/.venv/bin/activate"
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v npm >/dev/null 2>&1 || die "npm not found"

cd "$BACKEND"
python3 -c "import uvicorn" 2>/dev/null || die "uvicorn not installed — run: cd backend && pip install -r requirements.txt"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo ""
    echo "Stopping backend (pid $BACKEND_PID)..."
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting API at http://127.0.0.1:${BACKEND_PORT}  (docs: http://127.0.0.1:${BACKEND_PORT}/docs)"
uvicorn app.main:app --reload --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Give uvicorn a moment to bind
sleep 0.5

cd "$FRONTEND"
if [[ ! -d node_modules ]]; then
  echo "Installing frontend dependencies (npm install)..."
  npm install
fi

echo "Starting UI at http://127.0.0.1:${FRONTEND_PORT}"
echo "Open the UI in your browser. Ctrl+C stops both servers."
echo ""

npm run dev -- --port "$FRONTEND_PORT"
