# LangChain Max Tool Experiment

This experiment tests how well LangChain handles increasing numbers of tools using either vllm or local Ollama models by measuring **tool selection accuracy, execution success, and latency**.

## Overview

The experiment consists of two main components:
1. **MCP Tool Server** (`mcp_tool_server.py`) - Provides 5 real tools via MCP protocol
2. **Experiment Client** (`ollama_maxtool.py`) - Tests tool selection and execution with LangChain

### Tools Available
- `weather_info` - Fetches weather for a location
- `word_count` - Counts words in text
- `reverse_string` - Reverses text
- `uppercase` - Converts text to uppercase
- `insurance_scorer` - Generates random insurance scores

### Test Queries
The experiment runs 5 fixed queries, each mapped to a ground truth tool:
1. "What is the weather in New York?" → `weather_info`
2. "How many words are in 'Hello World, this is a test sentence'?" → `word_count`
3. "Reverse this text: Python Experiment" → `reverse_string`
4. "Convert this to uppercase: llamastack" → `uppercase`
5. "Give me an insurance evaluation score" → `insurance_scorer`

### Metrics Measured
- **Tool Execution Rate** - How many times tools are actually executed out of 5 queries
- **Correct Tool Selection Rate** - How many times the correct tool is selected out of 5 queries
- **Irrelevant Tool Rate** - How many time irrelevant tool is selected out of 5 queries
- **Average Latency** - Average response time for all queries

## Prerequisites

### 0. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

### 2. Install Python Dependencies
```bash
# From the project root (using uv)
uv sync

# Or install individually:
uv add langchain-mcp-adapters langgraph langchain-ollama langchain-community python-dotenv fastmcp requests
```

**Note**: This project uses `uv` for dependency management. If you don't have `uv` installed, you can install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Start Ollama Service
```bash
ollama serve
```

### 4. Pull Required Model
```bash
ollama pull llama3.2:3b-instruct-fp16
```

**Note**: For cluster deployments with vLLM, the model will be automatically loaded by the vLLM server.

## Running the Experiment

### Option 1: Automated Script (Recommended)
```bash
cd src/max_tool_experiment
./run_experiment.sh
```

The script will:
- ✅ Check if Ollama is running
- ✅ Verify the required model is available (pull if needed)
- ✅ Start the MCP server in the background
- ✅ Run the experiment
- ✅ Clean up processes automatically

### Option 2: Manual Execution

#### Terminal 1: Start MCP Server
```bash
cd src/max_tool_experiment
python mcp_tool_server.py
```

#### Terminal 2: Run Experiment
```bash
cd src/max_tool_experiment
python ollama_maxtool.py
```

## Output

The experiment generates:
- **Console output** showing real-time progress and results
- **CSV file** (`experiment_results_langchain_ollama.csv`) with detailed metrics

### Sample Output
```
🚀 Starting LangChain Max Tool Experiment...
==========================================
✅ MCP server started successfully (PID: 12345)
🧪 Running experiment...

Testing with 5 tools from MCP server...

User: What is the weather in New York?
Response: Weather in New York is sunny.
Executed Tools: ['weather_info']
Ground Truth Tool: weather_info

...

Total Tools: 5, Tool Execution Rate: 100.00%, Correct Tool Rate: 100.00%, Avg Latency: 2.3456s
✅ Experiment completed!
📊 Results saved to: experiment_results_langchain_ollama.csv
```

## Troubleshooting

### Common Issues

#### 1. Ollama Not Running
```
❌ Error: Ollama is not running. Please start Ollama first:
   ollama serve
```
**Solution**: Start Ollama with `ollama serve`

#### 2. Model Not Found
```
📥 Pulling required model: llama3.2:3b-instruct-fp16
```
**Solution**: Wait for the model to download, or manually run `ollama pull llama3.2:3b-instruct-fp16`

#### 3. MCP Server Failed to Start
```
❌ Error: MCP server failed to start
```
**Solution**: 
- Check if port 8000 is available
- Ensure all dependencies are installed
- Check Python environment

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'langchain_mcp_adapters'
```
**Solution**: Install missing dependencies:
```bash
uv add langchain-mcp-adapters langgraph langchain-ollama python-dotenv fastmcp
```

### Environment Variables

The experiment uses these environment variables (optional):

#### For Ollama (Local Testing)
- `OLLAMA_URL` - Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` - Model name (default: `llama3.2:3b-instruct-fp16`)

#### For vLLM (Cluster Deployment)
- `VLLM_BASE_URL` - vLLM server URL (default: `http://localhost:8000/v1`)
- `VLLM_MODEL` - Model name (default: `meta-llama/Llama-2-7b-chat-hf`)

#### Auto-Detection
The experiment automatically detects the best available provider:
1. If `VLLM_BASE_URL` or `VLLM_MODEL` is set → uses vLLM
2. If Ollama is running locally → uses Ollama
3. Defaults to Ollama for local development

## Experiment Customization

### Adding More Tools
To add more tools to the MCP server, edit `mcp_tool_server.py`:

```python
@mcp.tool()
def your_new_tool(param: str) -> str:
    """Description of your tool."""
    return f"Result: {param}"
```

### Modifying Test Queries
Edit the `queries` list in `ollama_maxtool.py`:

```python
queries = [
    ("Your new query?", "your_new_tool"),
    # ... existing queries
]
```

### Changing the Model
The experiment uses an LLM provider abstraction that supports both Ollama and vLLM. You can change models by:

#### Using Environment Variables (Recommended)
```bash
# For Ollama
export OLLAMA_MODEL="your-ollama-model"

# For vLLM
export VLLM_MODEL="your-vllm-model"
export VLLM_BASE_URL="http://your-cluster:8000/v1"
```

#### Using the LLM Provider API
```python
from llm_provider import get_llm_provider

# For local testing
llm = get_llm_provider("ollama", model="your-model")

# For cluster deployment
llm = get_llm_provider("vllm", model="your-model", base_url="http://cluster:8000/v1")
```

## Architecture

```
┌─────────────────┐    HTTP/MCP    ┌─────────────────┐
│   Experiment    │ ──────────────▶ │   MCP Server    │
│   Client        │                │                 │
│                 │                │ ┌─────────────┐ │
│ ┌─────────────┐ │                │ │ Tool 1      │ │
│ │ LangChain   │ │                │ │ Tool 2      │ │
│ │ Agent       │ │                │ │ Tool 3      │ │
│ │             │ │                │ │ Tool 4      │ │
│ │ ┌─────────┐ │ │                │ │ Tool 5      │ │
│ │ │ LLM     │ │ │                │ └─────────────┘ │
│ │ │Provider │ │ │                └─────────────────┘
│ │ │(Ollama/ │ │ │
│ │ │ vLLM)   │ │ │
│ │ └─────────┘ │ │
│ └─────────────┘ │
└─────────────────┘
```

## LLM Provider Abstraction

The experiment uses a flexible LLM provider abstraction (`llm_provider.py`) that supports:

### Local Testing (Ollama)
- **Use case**: Development and testing on local machines
- **Setup**: Install Ollama and pull models locally
- **Configuration**: Set `OLLAMA_URL` and `OLLAMA_MODEL` environment variables

### Cluster Deployment (vLLM)
- **Use case**: Large-scale experiments on compute clusters
- **Setup**: Connect to existing vLLM deployments
- **Configuration**: Set `VLLM_BASE_URL` and `VLLM_MODEL` environment variables

### Auto-Detection
The system automatically detects the best available provider:
1. **vLLM**: If `VLLM_BASE_URL` or `VLLM_MODEL` environment variables are set
2. **Ollama**: If Ollama is running locally and accessible
3. **Fallback**: Defaults to Ollama for local development

### Usage Examples

```python
from llm_provider import get_llm_provider, get_local_llm, get_cluster_llm

# Auto-detect (recommended)
llm = get_llm_provider()

# Explicit provider selection
llm = get_llm_provider("ollama", model="llama3.2:3b-instruct-fp16")
llm = get_llm_provider("vllm", model="meta-llama/Llama-2-7b-chat-hf")

# Convenience functions
llm = get_local_llm()  # For local testing
llm = get_cluster_llm()  # For cluster deployment
```

## Contributing

To extend this experiment:
1. Add new tools to `mcp_tool_server.py`
2. Update test queries in `ollama_maxtool.py`
3. Modify metrics calculation as needed
4. Update this README with new instructions

## License

This experiment is part of the small-model-experiments project. 