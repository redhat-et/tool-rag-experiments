import asyncio
import keyword
import re
import threading

from mcp.server.fastmcp import FastMCP
import requests
import os
import json

from evaluator.components.data_provider import ToolSet
from evaluator.utils.tool_logger import log_tool

from inspect import Signature, Parameter
from typing import Any, Annotated

from evaluator.utils.utils import print_verbose, print_iterable_verbose

_TYPE_MAP = {
    "STRING": str,
    "NUMBER": float,
    "BOOLEAN": bool,
    "OBJECT": dict,
    "ARRAY": list,
}


def _annotated(py_type: type, desc: str | None, required_flag: bool):
    meta = {"description": desc or "", "required": required_flag}
    try:
        from pydantic import Field
        return Annotated[py_type, Field(description=desc or ""), meta]
    except Exception:
        return Annotated[py_type, meta]  # generic metadata fallback


def _sanitize(name: str, used: set[str]) -> str:
    """
    Convert a given arbitrary name into a valid, unique Python identifier.
    """
    base = re.sub(r'\W+', '_', name).strip('_') or "param"
    if base[0].isdigit():
        base = "_" + base
    if keyword.iskeyword(base):
        base = base + "_"
    cand, i = base, 2
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


def _register_mcp_proxy_tool(mcp_instance, tool_name, tool_dict, base_url):

    required = tool_dict.get("required_parameters", [])
    optional = tool_dict.get("optional_parameters", [])

    parameters = []
    doc_lines = [tool_dict.get("api_description"), "", "Parameters:"]

    # we need these to sanitize tool parameters that are not valid Python identifiers
    used = set()
    sanitized_to_original_name = {}

    def _make_param(entry: dict, required_flag: bool) -> Parameter:
        original_name = entry["name"]
        sanitized = _sanitize(original_name, used)
        if sanitized != original_name:
            sanitized_to_original_name[sanitized] = original_name

        py_t = _TYPE_MAP.get(entry.get("type", "").upper(), Any)
        desc = (entry.get("description") or "").strip()
        default = entry.get("default", None)

        # Best-effort default coercion
        if default is not None and py_t is not str:
            try:
                default = py_t(default)
            except Exception:
                pass

        # Signature: required => no default; optional => has default
        default_for_sig = Parameter.empty if required_flag else default
        anno = _annotated(py_t, desc, required_flag)

        # Docstring: include required/optional + default
        type_name = getattr(py_t, "__name__", str(py_t))
        req_tag = "required" if required_flag else "optional"
        default_tag = f", default={default!r}" if default_for_sig is not Parameter.empty else ""
        doc_lines.append(f"  :param {sanitized}: ({req_tag}{default_tag}) {desc}")
        doc_lines.append(f"  :type {sanitized}: {type_name}")

        return Parameter(sanitized, kind=Parameter.KEYWORD_ONLY, annotation=anno, default=default_for_sig)

    for e in required:
        parameters.append(_make_param(e, required_flag=True))
    for e in optional:
        parameters.append(_make_param(e, required_flag=False))

    signature = Signature(parameters)
    docstring = "\n".join(doc_lines)

    def tool_func(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        # bound.apply_defaults()
        params = dict(bound.arguments)

        # Translate sanitized names back to original names for MirrorAPI
        outgoing = {}
        for param_name, param_value in params.items():
            original = sanitized_to_original_name.get(param_name, param_name)
            outgoing[original] = param_value

        predict_url = f"{base_url.rstrip('/')}/predict"
        payload = {
            "api_doc": tool_dict,
            "request": {
                "category": tool_dict.get("category_name"),
                "tool_name": tool_dict.get("tool_name"),
                "api_name": tool_dict.get("api_name"),
                "tool_input": str(outgoing),
                "strip": "filter"
            },
            "mode": "sft"
        }

        print_verbose(f"Sending payload to MirrorAPI: {payload}")
        response = requests.post(predict_url, json=payload).json()
        print_verbose(f"Tool execution response from MirrorAPI: {response}")
        return response

    final_tool_func = log_tool(tool_name)(tool_func)
    final_tool_func.__signature__ = signature
    final_tool_func.__doc__ = docstring

    mcp_instance.tool(name=tool_name, description=docstring)(final_tool_func)


async def run_mcp_proxy(tools: ToolSet, run_detached=False):
    """
    Set up an MCP server and register proxy tools from a list of tool dictionaries, connecting to MirrorAPI.
    Args:
        tools: A dictionary of tool specifications. Each value is a dict that should have an 'api_list' key (list of API specs)
        run_detached: If True, run the server in a background thread
    Returns:
        If run_detached: List[str] of registered tool names
        Else: The MCP server instance (to be run synchronously)
    """
    mirror_api_base_url = os.getenv("MIRROR_API_BASE_URL", "http://localhost:8000")
    mcp_port = int(os.getenv("MCP_PROXY_LOCAL_PORT", 9000))
    mcp_instance = FastMCP("General", port=mcp_port)
    registered_tool_names = []

    for tool_mcp_name, tool_dict in tools.items():
        _register_mcp_proxy_tool(mcp_instance, tool_mcp_name, tool_dict, mirror_api_base_url)
        registered_tool_names.append(tool_mcp_name)

    print_iterable_verbose(f"\nSummary: Registered {len(registered_tool_names)} tools:", registered_tool_names)

    print_verbose(f"MCP server configured on port {mcp_port}")
    print_verbose(f"MirrorAPI base URL: {mirror_api_base_url}")

    print(f"Starting the MCP server...")
    if run_detached:
        threading.Thread(target=lambda: mcp_instance.run(transport="streamable-http"), daemon=True).start()
        await asyncio.sleep(2)  # Give server time to start
    else:
        return mcp_instance


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tools_path = os.path.join(script_dir, "example_tools.json")
    with open(tools_path, "r") as f:
        example_tools = json.load(f)
    mcp = asyncio.run(run_mcp_proxy(example_tools))
    # Now run the server synchronously (no nested event loop)
    mcp.run(transport="streamable-http")
