import itertools
import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pydantic import BaseModel, Field

from evaluator.eval_spec import DATASET_SETTINGS
from evaluator.utils.file_downloader import fetch_remote_paths


ToolSet = Dict[str, Dict[str, Any]]


class QuerySpecification(BaseModel):
    """
    A query specification contains the following:
    - the query text
    - optionally, the reference answer
    - a set of golden tools that have to be invoked in order to solve the query
    - optionally, additional tools to be made available to the model at evaluation time
    """
    query: str
    reference_answer: Optional[str] = None
    golden_tools: ToolSet = Field(default_factory=dict)
    additional_tools: Optional[ToolSet] = None


def create_unique_mcp_tool_name(category_name: str, tool_name: str, api_name: str) -> str:
    """
    Create a unique MCP tool name from a ToolBench API identifiers.
    A function (analogous to an MCP tool) in ToolBench is uniquely identified by a category name, a tool name,
    and an API name.
    This function unifies and normalizes these three identifiers, returning a unique and valid MCP tool name.
    """
    unique_name = '.'.join([category_name, tool_name, api_name])
    return unique_name.replace(' ', '_').lower()


def tool_api_list_to_tool_set(tool_api_list: List[Dict[str, Any]]) -> ToolSet:
    tool_set = {}

    for tool_api in tool_api_list:
        mcp_tool_name = create_unique_mcp_tool_name(tool_api["category_name"],
                                                    tool_api["tool_name"],
                                                    tool_api["api_name"])
        tool_set[mcp_tool_name] = tool_api

    return tool_set


def _tool_file_content_to_api_list(tool_file_content: Dict[str, Any], category_name: str) -> List[Dict[str, Any]]:
    api_list = []

    for partial_api_spec in tool_file_content["api_list"]:
        new_api_spec = {
            "category_name": category_name,
            "tool_name": tool_file_content["tool_name"],
            "api_name": partial_api_spec["name"],
            "api_description": f"{tool_file_content['tool_description']}\n{partial_api_spec['description']}",
            "required_parameters": partial_api_spec["required_parameters"],
            "optional_parameters": partial_api_spec["optional_parameters"],
        }
        api_list.append(new_api_spec)

    return api_list


def _load_tools_from_dir(root_dir: str or Path, categories: List[str], tool_num: int) -> List[Dict[str, Any]]:
    if tool_num <= 0:
        raise ValueError(f"`tool_num` must be a positive integer; got {tool_num}.")

    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"`root_dir` does not exist or is not a directory: {root}")

    # Collect tool JSON paths from the specified categories
    candidates = []
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            raise ValueError(f"Category directory not found: {cat_dir}")

        for p in cat_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".json":
                candidates.append(p)

    rng = random.Random()
    rng.shuffle(candidates)

    tool_dicts = []
    for path in candidates:
        if len(tool_dicts) == tool_num:
            break

        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            # Corrupted/invalid or unreadable file -> skip
            continue

        try:
            new_tool_dicts = _tool_file_content_to_api_list(loaded, path.parent.name)
        except Exception:
            # Defensive: if the parser itself crashes on this input, treat as invalid
            new_tool_dicts = None

        if new_tool_dicts is None:
            continue

        # Add objects but cap at tool_num
        needed = tool_num - len(tool_dicts)
        tool_dicts.extend(new_tool_dicts[:needed])

    if len(tool_dicts) < tool_num:
        raise ValueError(
            f"Requested {tool_num} tools, but only found {len(tool_dicts)} valid "
            f"JSONs across categories {list(categories)} (searched {len(candidates)} files)."
        )

    return tool_dicts


def _load_random_tools(categories: List[str], tool_num: int) -> ToolSet:
    """
    Load tool_num random tools from the given categories.
    """
    # Download dataset files if needed
    root_dataset_path = Path(os.getenv("ROOT_DATASET_PATH"))
    remote_dataset_paths = DATASET_SETTINGS["tool_files"]
    local_paths = fetch_remote_paths(remote_dataset_paths, root_dataset_path)

    if len(local_paths) == 0:
        raise ValueError("No tool files provided")
    if len(local_paths) > 1:
        raise ValueError(f"Multiple tool files provided: {local_paths}\nMultiple tool files are not yet supported.")

    root_dir = local_paths[0]
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        raise ValueError(f"ERROR: Directory '{root_dir}' does not exist. Please check the path.")

    loaded_tools = _load_tools_from_dir(root_dir, categories, tool_num)
    return tool_api_list_to_tool_set(loaded_tools)


def _parse_raw_query_tool_definitions(query: Dict[str, Any]) -> Tuple[ToolSet or None, ToolSet or None]:
    """
    This method receives the query dict in the ToolBench dataset format.
    It returns the golden set of tools for this query. It is returned as a dictionary where
    each key is a unique MCP name of the tool and the corresponding value is a ToolBench dictionary of that tool.
    If required and available, it returns additional tools to use for baseline evaluation.
    """
    relevant_apis = query.get("relevant APIs")
    if not relevant_apis:
        print(f"No relevant APIs provided in the query specification:\n{query}")
        return None, None

    # extract the golden tools definitions
    golden_tools = {}
    for api in relevant_apis:
        tool_name = api[0]
        api_name = api[1]

        for tool_api in query["api_list"]:
            if tool_api["tool_name"] == tool_name and tool_api["api_name"] == api_name:
                category_name = tool_api["category_name"]
                tool_spec = tool_api
                break
        else:
            print(f"Bad query specification: the api doesn't appear in the API list:\n{query}")
            return None, None
        mcp_tool_name = create_unique_mcp_tool_name(category_name, tool_name, api_name)
        golden_tools[mcp_tool_name] = tool_spec

    # extract the rest of the available tools, if available
    additional_tool_apis = [tool_api for tool_api in query["api_list"] if tool_api not in golden_tools.values()]
    additional_tools = tool_api_list_to_tool_set(additional_tool_apis)

    # extract additional irrelevant tools if needed
    required_number_of_additional_tools = DATASET_SETTINGS["irrelevant_tools_ratio"] * len(golden_tools)
    if required_number_of_additional_tools == 0:
        return golden_tools, None
    if required_number_of_additional_tools <= len(additional_tools):
        # only return a subset of additional_tools
        additional_tools = dict(itertools.islice(additional_tools.items(), required_number_of_additional_tools))
        return golden_tools, additional_tools

    # if we reached this point, more tools are needed
    category_names = list(set([tool_api["category_name"] for tool_api in golden_tools.values()]))
    random_toolset = _load_random_tools(category_names, required_number_of_additional_tools - len(additional_tools))
    additional_tools.update(random_toolset)
    return golden_tools, additional_tools


def _load_queries_from_single_file(query_file_path: str or Path) -> List[QuerySpecification]:
    with open(query_file_path, 'r') as f:
        data = json.load(f)

    queries = []
    for raw_query_spec in data:
        query = raw_query_spec.get("query", "")
        golden_tools, additional_tools = _parse_raw_query_tool_definitions(raw_query_spec)
        if golden_tools is not None:
            queries.append(
                QuerySpecification(query=query, golden_tools=golden_tools, additional_tools=additional_tools or None)
            )
        else:
            print(f"Couldn't extract the tool definitions, skipping this query.")

    return queries


def get_queries() -> List[QuerySpecification]:
    """Load queries from the dataset."""
    try:
        root_dataset_path = Path(os.getenv("ROOT_DATASET_PATH"))
        if not root_dataset_path:
            print(f"⚠️ Root dataset folder not configured, using fallback queries.")
            return get_fallback_queries()

        remote_query_files = DATASET_SETTINGS["query_files"]
        if not remote_query_files:
            print(f"⚠️ Query files not configured properly, using fallback queries.")
            return get_fallback_queries()

        # Download the query files if needed
        local_paths = fetch_remote_paths(remote_query_files, root_dataset_path)

        # Actually load the queries
        queries = []
        for path in local_paths:
            new_queries = _load_queries_from_single_file(path)
            if len(queries) + len(new_queries) > DATASET_SETTINGS["queries_num"]:
                num_new_queries_to_add = DATASET_SETTINGS["queries_num"] - len(queries)
                queries.extend(new_queries[:num_new_queries_to_add])
                break
            queries.extend(new_queries)
        return queries

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Using fallback queries")
        return get_fallback_queries()


def get_tools_from_queries(queries: List[QuerySpecification]) -> ToolSet:
    tools = {}

    for query_spec in queries:
        tools.update(query_spec.golden_tools)
        if query_spec.additional_tools:
            tools.update(query_spec.additional_tools)

    return tools


def get_fallback_queries() -> List[QuerySpecification]:
    """Fallback queries if dataset loading fails."""
    return [
        QuerySpecification(query="What is the weather in New York?", golden_tools=["weather_info"]),
        QuerySpecification(query="How many words are in 'Hello World, this is a test sentence'?", golden_tools=["word_count"]),
        QuerySpecification(query="Reverse this text: Python Experiment", golden_tools=["reverse_string"]),
        QuerySpecification(query="Convert this to uppercase: llamastack", golden_tools=["uppercase"]),
        QuerySpecification(query="Give me an insurance evaluation score", golden_tools=["insurance_scorer"])
    ]
