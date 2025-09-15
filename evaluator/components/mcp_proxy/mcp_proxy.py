import asyncio
import threading

from mcp.server.fastmcp import FastMCP
import requests
import os
import json

from evaluator.components.data_provider import ToolSet
from evaluator.utils.tool_logger import log_tool


def make_mcp_proxy_tool(tool_name, payload_details, base_url="http://localhost:8000"):
    """
    Returns a function that can be registered as an MCP tool.
    - tool_name: Name of the tool (string)
    - payload_details: Dict with the payload to send to /predict (should include api_doc, request, mode, etc.)
    - base_url: Base URL for the MirrorAPI service (default: http://localhost:8000)
    """
    def tool_func():
        predict_url = f"{base_url.rstrip('/')}/predict"
        resp = requests.post(predict_url, json=payload_details)
        return resp.json()
    tool_func.__name__ = tool_name
    return log_tool(tool_name)(tool_func)


def check_mirror_api_health(base_url="http://localhost:8000", timeout=5):
    """
    Check if MirrorAPI is running and accessible.
    
    Args:
        base_url: Base URL for the MirrorAPI service
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if MirrorAPI is accessible, False otherwise
    """
    for endpoint in [f"{base_url.rstrip('/')}/health", base_url]:
        try:
            response = requests.get(endpoint, timeout=timeout)
            if response.status_code in [200, 404, 405]:
                print(f"✅ MirrorAPI is running at {base_url}")
                return True
        except requests.exceptions.RequestException:
            continue
    
    print(f"❌ MirrorAPI is not accessible at {base_url}")
    print("Please ensure MirrorAPI is running before starting the MCP server.")
    return False


async def run_mcp_proxy(tools: ToolSet, mcp_port=9000, server_name="General", mirror_api_base_url=None, run_detached=False):
    """
    Set up an MCP server and register proxy tools from a list of tool dictionaries, connecting to MirrorAPI.
    Args:
        tools: A dictionary of tool specifications. Each value is a dict that should have an 'api_list' key (list of API specs)
        mcp_port: Port for the MCP server (default: 9000)
        server_name: Name for the MCP server (default: 'General')
        mirror_api_base_url: Base URL for the MirrorAPI service (default: env MIRROR_API_BASE_URL or http://localhost:8000)
        run_detached: If True, run the server in a background thread
    Returns:
        If run_detached: List[str] of registered tool names
        Else: The MCP server instance (to be run synchronously)
    """
    if mirror_api_base_url is None:
        mirror_api_base_url = os.getenv("MIRROR_API_BASE_URL", "http://localhost:8000")
    mcp = FastMCP(server_name, port=mcp_port)
    registered_tool_names = []

    for tool_mcp_name, tool_dict in tools.items():
        payload = {
            "api_doc": tool_dict,
            "request": {
                "category": tool_dict.get("category_name"),
                "tool_name": tool_dict.get("tool_name"),
                "api_name": tool_dict.get("api_name"),
                "tool_input": "{}",
                "strip": "filter"
            },
            "mode": "sft"
        }
        tool_func = make_mcp_proxy_tool(tool_mcp_name, payload, mirror_api_base_url)
        mcp.tool(tool_mcp_name)(tool_func) # (tool_func) can be removed , left here for testing mirrorapi responses
        print(f"Registered proxy tool: {tool_mcp_name}")
        registered_tool_names.append(tool_mcp_name)

    print(f"\nSummary: Registered {len(registered_tool_names)} proxy tools:")
    for t in registered_tool_names:
        print(f"- {t}")

    print(f"MCP server '{server_name}' configured on port {mcp_port}")
    print(f"MirrorAPI base URL: {mirror_api_base_url}")

    print(f"Starting the MCP server...")
    if run_detached:
        threading.Thread(target=lambda: mcp.run(transport="streamable-http"), daemon=True).start()
        await asyncio.sleep(2)  # Give server time to start
    else:
        return mcp


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tools_path = os.path.join(script_dir, "example_tools.json")
    with open(tools_path, "r") as f:
        example_tools = json.load(f)
    mcp = asyncio.run(run_mcp_proxy(example_tools))
    # Now run the server synchronously (no nested event loop)
    mcp.run(transport="streamable-http")
