#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../../.."
TOOLS_DIR="$SCRIPT_DIR/toolenv2404_filtered"
TOOLS_TAR="$SCRIPT_DIR/toolenv2404_filtered.tar.gz"
TOOLS_URL="https://huggingface.co/datasets/stabletoolbench/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz"

if [ ! -d "$TOOLS_DIR" ]; then
    echo "toolenv2404_filtered directory not found."
    if [ ! -f "$TOOLS_TAR" ]; then
        echo "Downloading toolenv2404_filtered.tar.gz from HuggingFace..."
        curl -L -o "$TOOLS_TAR" "$TOOLS_URL"
    else
        echo "Found $TOOLS_TAR, skipping download."
    fi
    echo "Unpacking $TOOLS_TAR ..."
    tar -xzf "$TOOLS_TAR" -C "$SCRIPT_DIR"
    echo "Unpacked toolenv2404_filtered."
else
    echo "toolenv2404_filtered directory already exists."
fi

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