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
cd src/evaluator
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
cd src/evaluator
python mcp_tool_server.py
```

#### Terminal 2: Run Experiment
```bash
cd src/evaluator
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

The experiment uses these unified environment variables:

- `LLM_PROVIDER` - Explicitly set the provider ("ollama", "vllm", "openai")
- `LLM_MODEL` - Model name/identifier
- `LLM_BASE_URL` - Base URL for the provider (helps auto-detect remote vs local)

#### Auto-Detection Logic
If `LLM_PROVIDER` is not set, the system automatically detects the best available provider:
1. If `LLM_BASE_URL` is set → infers remote deployment (vLLM for non-Ollama URLs)
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
The experiment uses an LLM provider abstraction that automatically detects the best available provider. You can configure it using environment variables:

#### Using Environment Variables (Recommended)
```bash
# Explicitly set provider
export LLM_PROVIDER="ollama"
export LLM_MODEL="your-ollama-model"

# Or for vLLM
export LLM_PROVIDER="vllm"
export LLM_MODEL="your-vllm-model"
export LLM_BASE_URL="http://your-cluster:8000/v1"

# Or let auto-detection work
export LLM_MODEL="your-model"
export LLM_BASE_URL="http://your-cluster:8000/v1"  # Will auto-detect vLLM
```

#### Using the LLM Provider API
```python
from llm_provider import _get_provider

# Provider is automatically determined from environment
llm = _get_provider()

# Override temperature if needed
llm = _get_provider(temperature=0.1)
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

The experiment uses a flexible LLM provider abstraction (`llm_provider.py`) that automatically detects the best available provider based on environment variables:

### Supported Providers
- **Ollama**: For local development and testing
- **vLLM**: For high-performance serving on clusters
- **OpenAI**: For cloud API access

### Environment-Based Configuration
The provider is automatically determined from environment variables:
- `LLM_PROVIDER`: Explicitly set the provider ("ollama", "vllm", "openai")
- `LLM_MODEL`: Model name/identifier
- `LLM_BASE_URL`: Base URL for the provider (helps auto-detect remote vs local)

### Auto-Detection Logic
If `LLM_PROVIDER` is not set, the system automatically detects the best available provider:
1. If `LLM_BASE_URL` is set → infers remote deployment (vLLM for non-Ollama URLs)
2. If Ollama is running locally → uses Ollama
3. Defaults to Ollama for local development

### Usage Examples

```python
from llm_provider import _get_provider

# Provider is automatically determined from environment
llm = _get_provider()

# Override temperature if needed
llm = _get_provider(temperature=0.1)

# Check provider setup
status = validate_provider_setup()
print(f"Using {status['provider']} with model {status['model']}")
```

## Contributing

To extend this experiment:
1. Add new tools to `mcp_tool_server.py`
2. Update test queries in `ollama_maxtool.py`
3. Modify metrics calculation as needed
4. Update this README with new instructions

## License

This experiment is part of the small-model-experiments project. 