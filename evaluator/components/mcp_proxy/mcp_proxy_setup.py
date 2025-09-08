from mcp.server.fastmcp import FastMCP
import requests
import os
import json
import tarfile
from pathlib import Path

def download_and_unpack_dataset(dataset_url, output_dir="toolenv2404_filtered"):
    """
    Download and unpack a dataset from a URL.
    
    Args:
        dataset_url: URL to download the dataset from
        output_dir: Directory to extract the dataset to
        
    Returns:
        str: Path to the extracted dataset directory
    """
    print(f"Downloading dataset from: {dataset_url}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Download the dataset
    response = requests.get(dataset_url, stream=True)
    response.raise_for_status()
    
    # Save to temporary file
    temp_file = "temp_dataset.tar.gz"
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded dataset to {temp_file}")
    
    # Extract the dataset
    print(f"Extracting dataset to {output_dir}")
    with tarfile.open(temp_file, 'r:gz') as tar:
        tar.extractall(output_dir)
    
    # Clean up temporary file
    os.remove(temp_file)
    
    print(f"Dataset extracted successfully to {output_dir}")
    return str(output_path)


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
    return tool_func


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


def _get_categories_to_process(base_dir, n):
    """
    Get the list of categories to process based on the n parameter.
    
    Args:
        base_dir: Base directory containing category subdirectories,
        this is the path to the toolenv2404_filtered directory where the tool json files are located.
        n: Number of tools to register (None for all)
        
    Returns:
        list: List of category names to process
    """
    categories = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]
    
    if n is not None and n > 0:
        # Only process the first category
        return categories[:1]
    else:
        # Process all categories
        return categories


def register_tools_from_dir(mcp, base_dir, n=10, base_url="http://localhost:8000"):
    """
    If n is set, only registers the first n tools from the first subdirectory (category) found in base_dir.
    If n is None, registers all tools from all categories as before.
    Prints a clear error if the directory does not exist.
    """
    if not os.path.isabs(base_dir):
        base_dir = os.path.abspath(base_dir)
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory '{base_dir}' does not exist. Please check the path.")
        return
    
    registered_tools = []
    categories_to_process = _get_categories_to_process(base_dir, n)
    
    for category in categories_to_process:
        category_path = os.path.join(base_dir, category)
        print(f"Processing category: {category_path}")
        
        json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
        
        # Determine how many files to process
        files_to_process = json_files[:n] if n is not None and n > 0 else json_files
        
        for json_file in files_to_process:
            print(f"Processing file: {json_file}")
            file_path = os.path.join(category_path, json_file)
            
            tool_name, tool_func = _process_tool_file(file_path, category, base_url)
            
            if tool_name and tool_func:
                mcp.tool()(tool_func)
                print(f"Registered tool: {tool_name}")
                registered_tools.append(tool_name)
    
    print(f"\nSummary: Registered {len(registered_tools)} tools:")
    for t in registered_tools:
        print(f"- {t}")



def setup_mcp_server_with_tools(
    mcp_port=9000,
    mirror_api_base_url="http://localhost:8000",
    dataset_path="toolenv2404_filtered",
    n=10,
    dataset_url=None,
    server_name="General"
):
    """
    Set up an MCP server with tools registered from a dataset.
    
    Args:
        mcp_port: Port for the MCP server (default: 9000)
        mirror_api_base_url: Base URL for the MirrorAPI service (default: http://localhost:8000)
        dataset_path: Path to the dataset directory (default: toolenv2404_filtered)
        n: Number of tools to register (default: 10)
        dataset_url: Optional URL to download dataset from (default: None)
        server_name: Name for the MCP server (default: General)
        
    Returns:
        FastMCP: The configured MCP server instance
    """
    # Download dataset if URL is provided
    if dataset_url:
        print(f"Downloading dataset from: {dataset_url}")
        dataset_path = download_and_unpack_dataset(dataset_url, dataset_path)
    
    # Create MCP server
    mcp = FastMCP(server_name, port=mcp_port)
    
    # Register tools from the dataset
    register_tools_from_dir(mcp, dataset_path, n=n, base_url=mirror_api_base_url)
    
    print(f"MCP server '{server_name}' configured on port {mcp_port}")
    print(f"MirrorAPI base URL: {mirror_api_base_url}")
    print(f"Dataset path: {dataset_path}")
    print(f"Number of tools to register: {n}")
    
    return mcp


def run_mcp_server(mcp, transport="streamable-http"):
    """
    Run the MCP server.
    
    Args:
        mcp: FastMCP server instance
        transport: Transport type (default: streamable-http)
    """
    print(f"Starting MCP server with {transport} transport...")
    mcp.run(transport=transport)


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


# Example usage:
if __name__ == "__main__":
    # Get MirrorAPI URL from environment variable or use default
    mirror_api_url = os.getenv("MIRROR_API_BASE_URL", "http://localhost:8000")
    
    # Check if MirrorAPI is running
    if not check_mirror_api_health(mirror_api_url):
        print("Exiting due to MirrorAPI connectivity issues.")
        exit(1)
    
    # Example 1: Basic setup with OpenShift route
    mcp = setup_mcp_server_with_tools(
        mcp_port=9000,
        mirror_api_base_url=mirror_api_url,
        dataset_path="toolenv2404_filtered",
        n=10
    )
    run_mcp_server(mcp)
    
