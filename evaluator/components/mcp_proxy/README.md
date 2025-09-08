# MCP Proxy: ToolBench Integration and Automation

This directory contains scripts and utilities for running MCP (Model Context Protocol) servers and testing tool integration with the MirrorAPI and ToolBench tool datasets.

**Required:** Retrieve login token from OpenShift and run: `oc login --token=<TOKEN> --server=<server>`.

## Key Components

### 1. `run_mcp_proxy_mirror_api.sh`
- **Purpose:** Automates the setup and testing workflow with smart MirrorAPI connectivity.
- **What it does:**
  - Downloads the `toolenv2404_filtered` tool dataset from HuggingFace if not already present.
  - Unpacks the dataset for use.
  - **Smart MirrorAPI Connection:**
    - Automatically sets up kubectl port-forwarding for OpenShift/Kubernetes clusters
    - Supports custom MirrorAPI URLs via `MIRROR_API_URL` environment variable
    - Verifies MirrorAPI connectivity before proceeding
  - Runs `mcp_tool_test.py` to register tools and test LLM-agent integration.
  - Cleans up the port-forward process when done.
- **How to run:**
  - From the project root:
    ```sh
    bash src/max_tool_experiment/mcp_proxy/run_mcp_proxy_mirror_api.sh
    ```
  - Or from this directory (recommended):
    ```sh
    ./run_mcp_proxy_mirror_api.sh
    ```
  - With custom MirrorAPI URL:
    ```sh
    MIRROR_API_URL="https://your-mirrorapi.example.com" ./run_mcp_proxy_mirror_api.sh
    ```

### 2. `mcp_tool_test.py`
- **Purpose:** Registers tools from the ToolBench dataset, starts an MCP server, connects to the LLM, and tests tool invocation via a LangGraph agent.
- **Key steps:**
  - Gets MirrorAPI URL from environment variable or uses default
  - Registers the first 10 tools from the first category in `toolenv2404_filtered`.
  - Starts the MCP server in the background (for demo/testing).
  - Connects to the MCP server and lists available tools.
  - Connects to the LLM (Ollama, vllm, or OpenAI, as configured).
  - Creates a LangGraph agent with the tools.
  - Sends a simple query for each tool and prints the response.
- **How to run:**
  - From the project root:
    ```sh
    PYTHONPATH=. python src/max_tool_experiment/mcp_proxy/mcp_tool_test.py
    ```
  - Or from this directory:
    ```sh
    PYTHONPATH=../.. python mcp_tool_test.py
    ```

### 3. `mcp_proxy_setup.py`
- **Purpose:** Core utility functions for MCP proxy setup and tool registration.
- **Key functions:**
  - `setup_mcp_server_with_tools()`: Main function to set up MCP server with tools
  - `register_tools_from_dir()`: Registers tools from dataset directory
  - `make_mcp_proxy_tool()`: Creates proxy tool functions for MirrorAPI
  - `check_mirror_api_health()`: Verifies MirrorAPI connectivity
  - `download_and_unpack_dataset()`: Downloads and extracts tool datasets
- **Features:**
  - Automatic MirrorAPI URL construction (appends `/predict` internally)
  - Smart dataset downloading and extraction
  - Health checks for MirrorAPI connectivity
  - Configurable parameters for different environments

### 4. `toolenv2404_filtered/`
- **What it is:**
  - A directory of tool JSONs from the [StableToolBench ToolEnv2404 dataset](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404).
  - Downloaded and unpacked automatically by the setup script if not present.

## **Workflow Overview**
1. The setup script ensures the tool dataset is present and unpacks it if needed.
2. **Smart MirrorAPI Connection:**
   - Automatically detects if kubectl is available and sets up port-forwarding for OpenShift/Kubernetes
   - Supports custom MirrorAPI URLs via environment variables
   - Verifies connectivity before proceeding
3. The MCP server is started and tools are registered from the dataset.
4. The script connects to the MCP server and the LLM, creates an agent, and tests tool invocation.
5. Cleanup of port-forward processes when done.

## **Environment and Path Notes**
- Always run scripts from the project root or use the provided robust path handling.
- Set `PYTHONPATH` to the project root for all Python invocations to ensure imports work.
- The scripts are robust to being run from different directories by resolving paths relative to their own location.
- **MirrorAPI Configuration:**
  - Default: Uses kubectl port-forwarding for OpenShift/Kubernetes clusters
  - Custom: Set `MIRROR_API_URL` environment variable for external URLs
  - Health checks ensure connectivity before proceeding

## **Configuration Options**

### MirrorAPI URL Configuration
```bash
# Use default port-forwarding (recommended for OpenShift/Kubernetes)
./run_mcp_proxy_mirror_api.sh

# Use custom external URL
MIRROR_API_URL="https://your-mirrorapi.example.com" ./run_mcp_proxy_mirror_api.sh

# Use local development server
MIRROR_API_URL="http://localhost:8000" ./run_mcp_proxy_mirror_api.sh
```

### MCP Server Configuration
The `mcp_proxy_setup.py` module provides configurable functions:
```python
from mcp_proxy_setup import setup_mcp_server_with_tools

# Basic setup
mcp = setup_mcp_server_with_tools()

# Custom configuration
mcp = setup_mcp_server_with_tools(
    mcp_port=9001,
    mirror_api_base_url="http://localhost:8000",
    dataset_path="my_dataset",
    n=5,
    server_name="MyCustomServer"
)
```

## **References**
- [StableToolBench ToolEnv2404 on HuggingFace](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404)

---

**For any issues, check the printed error messages for missing directories or import errors, and ensure you are running from the correct directory with the correct environment.**
