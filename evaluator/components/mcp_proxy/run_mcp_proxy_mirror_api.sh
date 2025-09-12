#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."

# Set MirrorAPI service URL from environment variable
if [ -z "$MIRROR_API_URL" ]; then
    echo "❌ MIRROR_API_URL environment variable is not set"
    echo "Please set it to your MirrorAPI URL, for example:"
    echo "  export MIRROR_API_URL=\"https://your-mirrorapi.example.com\""
    echo "  export MIRROR_API_URL=\"http://localhost:8000\""
    exit 1
fi

echo "Using MirrorAPI URL: $MIRROR_API_URL"

# Check if MirrorAPI is accessible before running the test
echo "Checking MirrorAPI connectivity..."
if ! curl -s --connect-timeout 5 "$MIRROR_API_URL/health" >/dev/null 2>&1 && ! curl -s --connect-timeout 5 "$MIRROR_API_URL" >/dev/null 2>&1; then
    echo "❌ MirrorAPI is not accessible at $MIRROR_API_URL"
    echo "Please ensure MirrorAPI is running and accessible."
    exit 1
fi
echo "✅ MirrorAPI is accessible"

echo "Running mcp_tool_test.py..."
PYTHONPATH="$PROJECT_ROOT" MIRROR_API_BASE_URL="$MIRROR_API_URL" python "$SCRIPT_DIR/mcp_tool_test.py"