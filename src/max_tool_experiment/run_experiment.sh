#!/bin/bash

# LangChain Max Tool Experiment Runner
# This script starts the MCP tool server and runs the experiment

set -e  # Exit on any error

echo "🚀 Starting LangChain Max Tool Experiment..."
echo "=========================================="

# Check LLM provider availability
echo "🔍 Checking LLM provider availability..."

# Check if vLLM environment variables are set (cluster deployment)
if [ ! -z "$VLLM_BASE_URL" ] || [ ! -z "$VLLM_MODEL" ]; then
    echo "✅ vLLM environment variables detected - will use cluster deployment"
    # Test vLLM connection
    if curl -s "${VLLM_BASE_URL:-http://localhost:8000/v1}/models" > /dev/null 2>&1; then
        echo "✅ vLLM server is accessible"
    else
        echo "⚠️  Warning: vLLM server not accessible, will fall back to Ollama if available"
    fi
else
    # Check if Ollama is running (local testing)
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running - will use local deployment"
        # Check if the required model is available
        if ! ollama list | grep -q "${OLLAMA_MODEL:-llama3.2:3b-instruct-fp16}"; then
            echo "📥 Pulling required model: ${OLLAMA_MODEL:-llama3.2:3b-instruct-fp16}"
            ollama pull "${OLLAMA_MODEL:-llama3.2:3b-instruct-fp16}"
        fi
    else
        echo "❌ Error: No LLM provider available"
        echo "   For local testing: start Ollama with 'ollama serve'"
        echo "   For cluster deployment: set VLLM_BASE_URL and VLLM_MODEL environment variables"
        exit 1
    fi
fi

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up..."
    if [ ! -z "$MCP_PID" ]; then
        kill $MCP_PID 2>/dev/null || true
        echo "   Stopped MCP server (PID: $MCP_PID)"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start MCP server in background
echo "🔧 Starting MCP tool server..."
python mcp_tool_server.py &
MCP_PID=$!

# Wait a moment for the server to start
sleep 3

# Check if MCP server started successfully
if ! curl -s http://127.0.0.1:8000/mcp/ > /dev/null 2>&1; then
    echo "❌ Error: MCP server failed to start"
    cleanup
    exit 1
fi

echo "✅ MCP server started successfully (PID: $MCP_PID)"
echo "🧪 Running experiment..."

# Run the experiment
python ollama_maxtool.py

echo "✅ Experiment completed!"
echo "📊 Results saved to: experiment_results_langchain_ollama.csv"

# Cleanup
cleanup 