#!/bin/bash

# This script starts the MCP tool server and runs the experiment

set -e  # Exit on any error

echo "🚀 Starting Tool RAG Experiments..."
echo "=========================================="

# Check LLM provider availability
echo "🔍 Checking LLM provider availability..."

# Check if provider is explicitly set
if [ ! -z "$LLM_PROVIDER" ]; then
    echo "✅ LLM_PROVIDER explicitly set to: $LLM_PROVIDER"
    
    # Provider-specific checks
    case "$LLM_PROVIDER" in
        "ollama")
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "✅ Ollama is running"
                # Check if the required model is available
                MODEL_NAME="${LLM_MODEL:-llama3.2:3b-instruct-fp16}"
                if ! ollama list | grep -q "$MODEL_NAME"; then
                    echo "📥 Pulling required model: $MODEL_NAME"
                    ollama pull "$MODEL_NAME"
                fi
            else
                echo "❌ Error: Ollama is not running. Please start Ollama with 'ollama serve'"
                exit 1
            fi
            ;;
        "vllm")
            BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
            if curl -s "$BASE_URL/models" > /dev/null 2>&1; then
                echo "✅ vLLM server is accessible at $BASE_URL"
            else
                echo "❌ Error: vLLM server not accessible at $BASE_URL"
                exit 1
            fi
            ;;
        "openai")
            echo "✅ OpenAI provider configured (will use API key from environment)"
            ;;
        *)
            echo "❌ Error: Unsupported LLM_PROVIDER: $LLM_PROVIDER"
            echo "   Supported providers: ollama, vllm, openai"
            exit 1
            ;;
    esac
else
    # Auto-detect provider
    echo "🔍 Auto-detecting LLM provider..."
    
    # Check if base URL is set (implies remote deployment)
    if [ ! -z "$LLM_BASE_URL" ]; then
        if [[ "$LLM_BASE_URL" == *"ollama"* ]]; then
            echo "✅ Auto-detected Ollama from LLM_BASE_URL"
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo "✅ Ollama is running"
            else
                echo "❌ Error: Ollama is not running"
                exit 1
            fi
        else
            echo "✅ Auto-detected vLLM from LLM_BASE_URL"
            if curl -s "$LLM_BASE_URL/models" > /dev/null 2>&1; then
                echo "✅ vLLM server is accessible"
            else
                echo "❌ Error: vLLM server not accessible"
                exit 1
            fi
        fi
    else
        # Check if Ollama is running locally
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Auto-detected Ollama (running locally)"
            # Check if the required model is available
            MODEL_NAME="${LLM_MODEL:-llama3.2:3b-instruct-fp16}"
            if ! ollama list | grep -q "$MODEL_NAME"; then
                echo "📥 Pulling required model: $MODEL_NAME"
                ollama pull "$MODEL_NAME"
            fi
        else
            echo "❌ Error: No LLM provider available"
            echo "   Set LLM_PROVIDER environment variable or ensure Ollama is running"
            exit 1
        fi
    fi
fi

# Set up signal handlers
trap cleanup SIGINT SIGTERM
echo "🧪 Running experiments..."

# Run the experiment
python run_experiments.py

echo "✅ Experiment completed!"
