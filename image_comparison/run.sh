#!/usr/bin/env bash
set -euo pipefail

# Run this script from an allocated Zhores compute node.
# Optional:
#   RESULTS_DIR=/absolute/path/to/PSPStain/results
#   PORT=8501
#   ADDRESS=0.0.0.0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8501}"
ADDRESS="${ADDRESS:-0.0.0.0}"

if ! python -c "import streamlit, PIL" >/dev/null 2>&1; then
    echo "Streamlit/Pillow are not available in the active Python environment."
    echo "Install them with:"
    echo "  python -m pip install -r \"$SCRIPT_DIR/requirements.txt\""
    exit 1
fi

if [[ -n "${RESULTS_DIR:-}" ]]; then
    export RESULTS_DIR
fi

echo "Starting Image Generation Comparison"
echo "App directory: $SCRIPT_DIR"
echo "Results directory: ${RESULTS_DIR:-$SCRIPT_DIR/../results}"
echo "Listening address: $ADDRESS"
echo "Port: $PORT"
echo "Stop with Ctrl+C"

exec python -m streamlit run "$SCRIPT_DIR/app.py" \
    --server.address "$ADDRESS" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
