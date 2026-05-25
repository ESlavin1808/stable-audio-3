#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$ROOT_DIR/venv" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "  Run setup_portable.sh first."
    exit 1
fi

source "$ROOT_DIR/venv/bin/activate"

# ─── Show GPU info ──────────────────────────────────
echo "=== System Info ==="
python "$ROOT_DIR/app/check_torch.py"

echo ""
echo "  Starting server..."
echo "  URL: http://localhost:8765"
echo ""
echo "  Close this window to stop the server."
echo "  (or Ctrl+C)"
echo ""

python -m uvicorn app.server:app --host 127.0.0.1 --port 8765 --log-level info
