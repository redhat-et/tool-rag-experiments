import asyncio
import threading

from mcp.server.fastmcp import FastMCP
import requests
import os
import json

from evaluator.eval_spec import DATASET_SETTINGS
from evaluator.utils.file_downloader import fetch_remote_paths
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


def _process_tool_file(file_path, category, base_url):
    """
    Process a single tool file and return tool registration info.
    
    Args:
        file_path: Path to the tool JSON file
        category: Category name for the tool
        base_url: Base URL for the MirrorAPI service
        
    Returns:
        tuple: (tool_name, tool_func) if successful, (None, None) if failed
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        tool_name = data.get('tool_name')
        api_list = data.get('api_list', [])
        
        if not tool_name or not api_list:
            return None, None
            
        api_doc = api_list[0]
        payload = {
            "api_doc": api_doc,
            "request": {
                "category": category,
                "tool_name": tool_name,
                "api_name": api_doc.get("api_name", tool_name),
                "tool_input": "{}",
                "strip": "filter"
            },
            "mode": "sft"
        }
        
        tool_func = make_mcp_proxy_tool(tool_name, payload, base_url)
        return tool_name, tool_func
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None


def _get_categories_to_process(base_dir):
    """
    Get the list of categories to process based on the evaluation parameters.
    
    Args:
        base_dir: Base directory containing category subdirectories,
        this is the path to the toolenv2404_filtered directory where the tool json files are located.
        
    Returns:
        list: List of category names to process
    """
    categories_to_include = DATASET_SETTINGS["tool_categories"]
    if categories_to_include is not None:
        categories = [c for c in categories_to_include if os.path.isdir(os.path.join(base_dir, c))]
        if len(categories) < len(categories_to_include):
            print(f"Warning: Requested categories {categories_to_include}, but only {categories} are valid categories.")
        return categories

    return [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]


def register_tools_from_dir(mcp, local_tool_dir_paths, base_url="http://localhost:8000"):
    """
    Registers NUMBER_OF_TOOLS_TO_REGISTER tools (or all tools if not set) from the categories defined
    in CATEGORIES_TO_INCLUDE (or all categories if not set). If the specified number of tools is smaller than
    the number of available tools in the selected categories, only the first NUMBER_OF_TOOLS_TO_REGISTER tools
    (in alphabetical order of categories and tools) will be registered.
    Prints a clear error if the directory does not exist.
    """
    if len(local_tool_dir_paths) == 0:
        raise ValueError("No tool files provided")
    if len(local_tool_dir_paths) > 1:
        raise ValueError(f"Multiple tool files provided: {local_tool_dir_paths}\nMultiple tool files are not yet supported.")

    base_dir = local_tool_dir_paths[0]
    if not os.path.isabs(base_dir):
        base_dir = os.path.abspath(base_dir)
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory '{base_dir}' does not exist. Please check the path.")
        return
    
    registered_tool_names = []
    categories_to_process = _get_categories_to_process(base_dir)
    remaining_tools_number = DATASET_SETTINGS["max_tools_num"]
    
    for category in categories_to_process:
        category_path = str(os.path.join(base_dir, category))
        print(f"Processing category: {category_path}")
        
        json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]

        if remaining_tools_number is None or remaining_tools_number >= len(json_files):
            files_to_process = json_files
        else:
            files_to_process = json_files[:remaining_tools_number]
        
        for json_file in files_to_process:
            print(f"Processing file: {json_file}")
            file_path = os.path.join(category_path, json_file)
            
            tool_name, tool_func = _process_tool_file(file_path, category, base_url)
            
            if tool_name and tool_func:
                mcp.tool()(tool_func)
                print(f"Registered tool: {tool_name}")
                registered_tool_names.append(tool_name)

        if remaining_tools_number is not None:
            remaining_tools_number -= len(files_to_process)
            if remaining_tools_number <= 0:
                break
    
    print(f"\nSummary: Registered {len(registered_tool_names)} tools:")
    for t in registered_tool_names:
        print(f"- {t}")

    return registered_tool_names


def setup_mcp_server_with_tools(
    mcp_port,
    mirror_api_base_url,
    root_dataset_path,
    server_name="General"
):
    """
    Set up an MCP server with tools registered from a dataset.
    
    Args:
        mcp_port: Port for the MCP server
        mirror_api_base_url: Base URL for the MirrorAPI service
        root_dataset_path: Path to the root dataset directory
        server_name: Name for the MCP server (default: General)
        
    Returns:
        FastMCP: The configured MCP server instance
        List[str]: A list of registered tool names
    """
    # Download dataset files if needed
    remote_dataset_paths = DATASET_SETTINGS["tool_files"]
    local_paths = fetch_remote_paths(remote_dataset_paths, root_dataset_path)
    
    # Create MCP server
    mcp = FastMCP(server_name, port=mcp_port)
    
    # Register tools from the dataset
    registered_tool_names = register_tools_from_dir(mcp, local_paths, base_url=mirror_api_base_url)
    
    print(f"MCP server '{server_name}' configured on port {mcp_port}")
    print(f"MirrorAPI base URL: {mirror_api_base_url}")
    
    return mcp, registered_tool_names


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


async def run_mcp_proxy(tool_dicts, mcp_port=9000, server_name="General", mirror_api_base_url=None, run_detached=False):
    """
    Set up an MCP server and register proxy tools from a list of tool dictionaries, connecting to MirrorAPI.
    Args:
        tool_dicts: List[dict], each dict should have an 'api_list' key (list of API specs)
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

    for tool_dict in tool_dicts:
        api_list = tool_dict.get("api_list", [])
        # Try to get category from the first API, fallback to 'General'
        category = api_list[0].get("category_name", "General") if api_list else "General"
        for api_doc in api_list:
            tool_name = api_doc.get("tool_name")
            payload = {
                "api_doc": api_doc,
                "request": {
                    "category": category,
                    "tool_name": tool_name,
                    "api_name": api_doc.get("api_name", tool_name),
                    "tool_input": "{}",
                    "strip": "filter"
                },
                "mode": "sft"
            }
            tool_func = make_mcp_proxy_tool(tool_name, payload, mirror_api_base_url)
            mcp.tool(tool_name)(tool_func) # (tool_func) can be removed , left here for testing mirrorapi responses
            print(f"Registered proxy tool: {tool_name}")
            registered_tool_names.append(tool_name)

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
    import json
    import asyncio
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tools_path = os.path.join(script_dir, "example_tools.json")
    with open(tools_path, "r") as f:
        example_tools = json.load(f)
    mcp = asyncio.run(run_mcp_proxy(example_tools))
    # Now run the server synchronously (no nested event loop)
    mcp.run(transport="streamable-http")
