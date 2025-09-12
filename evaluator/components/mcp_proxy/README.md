# MCP Proxy: ToolBench Integration and Automation

This directory contains scripts and utilities for running MCP (Model Context Protocol) servers and testing tool integration with the MirrorAPI and ToolBench tool datasets.

## Key Components

### 1. `run_mcp_proxy_mirror_api.sh`
- **Purpose:** Automates the setup and testing workflow with MirrorAPI connectivity.
- **What it does:**
  - Checks for the `MIRROR_API_URL` environment variable and verifies MirrorAPI connectivity.
  - Runs `mcp_tool_test.py` to register tools and test LLM-agent integration.
- **How to run:**
  - From the project root:
    ```sh
    bash evaluator/components/mcp_proxy/run_mcp_proxy_mirror_api.sh
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
- **Purpose:** Registers tools from `example_tools.json`, starts an MCP server, connects to the LLM, and tests tool invocation via a LangGraph agent.
- **Key steps:**
  - Gets MirrorAPI URL from environment variable or uses default.
  - Loads tool definitions from `example_tools.json` (shared with the main server).
  - Starts the MCP server in the background (for demo/testing).
  - Connects to the MCP server and lists available tools.
  - Connects to the LLM (Ollama, vllm, or OpenAI, as configured).
  - Creates a LangGraph agent with the tools.
  - Sends a simple query for each tool and prints the response.
- **How to run:**
  - From the project root:
    ```sh
    PYTHONPATH=. python evaluator/components/mcp_proxy/mcp_tool_test.py
    ```
  - Or from this directory:
    ```sh
    PYTHONPATH=../.. python mcp_tool_test.py
    ```

### 3. `mcp_proxy.py`
- **Purpose:** Core utility for MCP proxy setup and tool registration.
- **Key features:**
  - Loads tool definitions from `example_tools.json`.
  - Cleans tool names (removes spaces, dots, unicode, and leading underscores).
  - Registers each tool with a proxy function that forwards queries to MirrorAPI.
  - Prints the payload for each tool as it is registered (for debugging).
  - Starts the MCP server and makes tools available for agent queries.

### 4. `example_tools.json`
- **What it is:**
  - A JSON file containing the canonical list of example tool definitions used for both server and test registration.
  - Easy to update and extend with new tools for future experiments.

## **Workflow Overview**
1. The setup script checks MirrorAPI connectivity.
2. The MCP server is started and tools are registered from `example_tools.json`.
3. Each tool is registered with a proxy function that forwards queries to MirrorAPI.
4. The script connects to the MCP server and the LLM, creates an agent, and tests tool invocation.

## **Environment and Path Notes**
- Always run scripts from the project root or use the provided robust path handling.
- Set `PYTHONPATH` to the project root for all Python invocations to ensure imports work.
- The scripts are robust to being run from different directories by resolving paths relative to their own location.
- **MirrorAPI Configuration:**
  - Set `MIRROR_API_URL` environment variable for your MirrorAPI endpoint.
  - Health checks ensure connectivity before proceeding.

## **Configuration Options**

### MirrorAPI URL Configuration
```bash
# Use custom external URL
MIRROR_API_URL="https://your-mirrorapi.example.com" ./run_mcp_proxy_mirror_api.sh

# Use local development server
MIRROR_API_URL="http://localhost:8000" ./run_mcp_proxy_mirror_api.sh
```

## **Tool Name Cleaning and Payload Debugging**
- Tool names are cleaned to remove spaces, dots, unicode, and leading underscores for safe registration.
- The payload for each tool is printed during registration for debugging and transparency.

## **References**
- [StableToolBench ToolEnv2404 on HuggingFace](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404)

---

**For any issues, check the printed error messages for missing directories or import errors, and ensure you are running from the correct directory with the correct environment.**
