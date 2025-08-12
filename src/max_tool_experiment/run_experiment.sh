#!/bin/bash

# Enhanced Max Tool Experiment with ToolBench-style evaluation
# This script runs the enhanced experiment with ToolBench metrics

set -e

echo "🚀 Starting Enhanced Max Tool Experiment with ToolBench-style Evaluation"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# After set -e and color funcs add arg parsing
# Check if we're in the right directory
if [ ! -f "mcp_tool_server.py" ]; then
    print_error "Please run this script from the src/max_tool_experiment directory"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Ollama is running
print_info "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_warning "Ollama service is not running. Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Check if required model is available
print_info "Checking required model..."
if ! ollama list | grep -q "llama3.2:3b-instruct-fp16"; then
    print_warning "Required model not found. Pulling llama3.2:3b-instruct-fp16..."
    ollama pull llama3.2:3b-instruct-fp16
fi

# Install dependencies
print_info "Installing dependencies..."
uv sync

# Prepare Python runner (avoid manual venv activation; use project env)
print_info "Preparing Python environment..."
PYTHON_CMD="uv run python"

# Function to cleanup background processes
cleanup() {
    print_info "Cleaning up background processes..."
    if [ ! -z "$MCP_PID" ]; then
        kill $MCP_PID 2>/dev/null || true
    fi
    if [ ! -z "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Start MCP tool server in background
print_info "Starting MCP tool server..."
$PYTHON_CMD mcp_tool_server.py &
MCP_PID=$!

# Wait for MCP server to start
print_info "Waiting for MCP server to start..."
sleep 3

# Check if MCP server is running
if ! curl -s http://localhost:8000/mcp/ > /dev/null 2>&1; then
    print_error "MCP server failed to start"
    exit 1
fi

print_status "MCP server is running on http://localhost:8000/mcp/"

# Run the enhanced experiment
print_info "Running enhanced experiment with ToolBench-style evaluation..."
$PYTHON_CMD enhanced_maxtool_experiment.py

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    print_status "Enhanced experiment completed successfully!"
    
    # Check if results files were created
    if [ -f "toolbench_agent_evaluation_results.json" ]; then
        print_status "ToolBench evaluation results saved to toolbench_agent_evaluation_results.json"
    fi
    
    if [ -f "enhanced_experiment_results.json" ]; then
        print_status "Enhanced experiment results saved to enhanced_experiment_results.json"
    fi
    
    # Display summary if results exist
    if [ -f "enhanced_experiment_results.json" ]; then
        print_info "=== Experiment Summary ==="
        $PYTHON_CMD - <<'PY'
import json
report = json.load(open('enhanced_experiment_results.json'))
m = report.get('agent_metrics', {})
print("Tool Execution Rate : {:.2%}".format(m.get("tool_execution_rate", 0)))
print("Correct Tool Rate   : {:.2%}".format(m.get("correct_tool_rate", 0)))
print("Irrelevant Tool Rate: {:.2%}".format(m.get("irrelevant_tool_rate", 0)))
print("Average Latency     : {:.2f}s".format(m.get("average_latency", 0)))
print("ToolBench Pass Rate : {:.2f}%".format(report.get('toolbench_pass_rate_evaluation', {}).get('overall_statistics', {}).get('pass_rate', 0)))
PY
    fi
else
    print_error "Enhanced experiment failed"
    exit 1
fi

print_status "Enhanced experiment completed!"
print_info "Check the following files for detailed results:"
echo "   - enhanced_experiment_results.json (Combined Report)" 

# Cleanup and exit
cleanup
exit 0 