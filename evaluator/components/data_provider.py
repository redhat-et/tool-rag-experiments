import itertools
import json
import math
import os
import random
from json import JSONDecodeError
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pydantic import BaseModel, Field

from evaluator.eval_spec import DATASET_SETTINGS, EvaluationEnvSpec
from evaluator.utils.file_downloader import fetch_remote_paths

ToolSet = Dict[str, Dict[str, Any]]


class QuerySpecification(BaseModel):
    """
    A query specification contains the following:
    - the unique query ID
    - the query text
    - optionally, the reference answer
    - a set of golden tools that have to be invoked in order to solve the query
    - optionally, additional tools to be made available to the model at evaluation time
    """
    id: int
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


def _load_random_tools(categories: List[str] or None, tool_num: int) -> ToolSet:
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

    if categories is None:
        # if not given, fetch from all possible categories
        categories = [
            name for name in os.listdir(root_dir)
            if not name.startswith(".") and os.path.isdir(os.path.join(root_dir, name))
        ]
    loaded_tools = _load_tools_from_dir(root_dir, categories, tool_num)
    return tool_api_list_to_tool_set(loaded_tools)


def _parse_raw_query_tool_definitions(query: Dict[str, Any], experiment_environment: EvaluationEnvSpec) -> Tuple[ToolSet or None, ToolSet or None]:
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
    required_number_of_additional_tools = math.ceil(experiment_environment.irrelevant_tools_ratio * len(golden_tools))
    if required_number_of_additional_tools == 0:
        return golden_tools, None
    if experiment_environment.irrelevant_tools_from_same_categories and required_number_of_additional_tools <= len(additional_tools):
        # only return a subset of additional_tools
        additional_tools = dict(itertools.islice(additional_tools.items(), required_number_of_additional_tools))
        return golden_tools, additional_tools

    # if we reached this point, more tools are needed
    if experiment_environment.irrelevant_tools_from_same_categories:
        category_names = list(set([tool_api["category_name"] for tool_api in golden_tools.values()]))
        random_toolset = _load_random_tools(category_names, required_number_of_additional_tools - len(additional_tools))
        additional_tools.update(random_toolset)
        return golden_tools, additional_tools

    random_toolset = _load_random_tools(None, required_number_of_additional_tools)
    return golden_tools, random_toolset


def _load_reference_answer(root_dir: Path, model_name: str or None, query_id: int) -> str or None:
    """
    Load the final answer for a given query produced by a given model.

    Directory layout:
      root_dir/
        └── {model_name}/
            ├── subdir_a/
            │   ├── {query_id}_*.json
            │   └── ...
            └── subdir_b/
                └── ...

    Each matching JSON file has the structure:
      {
        "answer_generation": {
          "final_answer": "<JSON string>"
        }
      }
    where the string parses to:
      { "final_answer": <the value to return> }

    Args:
        root_dir: Path to the root answers directory.
        model_name: Name of the model (must match the directory name under root_dir).
        query_id: Integer query ID. The JSON filename begins with "{query_id}_".

    Returns:
        The value stored under the nested "final_answer" key (representing the final answer to the query).

    Raises:
        FileNotFoundError: If the model directory or the query file cannot be found.
        ValueError: If the JSON structure is malformed or missing required keys.
        json.JSONDecodeError: If any JSON parsing fails.
    """
    if model_name is None:
        # this evaluation will proceed without reference answers
        return None

    model_dir = root_dir / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    prefix = f"{query_id}_"

    # Search all subdirectories (any depth) under the model directory
    for dirpath, _, filenames in os.walk(model_dir):
        for fname in filenames:
            if fname.startswith(prefix) and fname.endswith(".json"):
                candidate_path = os.path.join(dirpath, fname)
                # Found the first matching file; return immediately after parsing
                with open(candidate_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                try:
                    outer_final = data["answer_generation"]["final_answer"]
                except (KeyError, TypeError):
                    raise ValueError(
                        f"File {candidate_path} is missing "
                        '["answer_generation"]["final_answer"].'
                    )

                if not isinstance(outer_final, str):
                    raise ValueError(
                        f'In {candidate_path}, "final_answer" should be a JSON string.'
                    )

                if not outer_final:
                    # the reference answer for this query is not available
                    return None

                try:
                    inner = json.loads(outer_final)  # parse the stringified JSON
                except JSONDecodeError:
                    # the final answer JSON is invalid
                    return None
                if "final_answer" not in inner:
                    raise ValueError(
                        f'Inner JSON in {candidate_path} is missing "final_answer".'
                    )
                return inner["final_answer"]

    # If we get here, nothing matched
    raise FileNotFoundError(
        f"No JSON starting with '{prefix}' found under: {model_dir}"
    )


def _load_queries_from_single_file(
        query_file_path: str or Path,
        max_queries_num: int or None,
        root_dataset_path: str or Path,
        experiment_environment: EvaluationEnvSpec,
) -> List[QuerySpecification]:
    with open(query_file_path, 'r') as f:
        data = json.load(f)

    if DATASET_SETTINGS["reference_model_id"] is None:
        # judge-based evaluation is disabled - no need to load reference answers
        reference_answers_local_dir = None
    else:
        reference_answers_local_dir = fetch_remote_paths(
            [DATASET_SETTINGS["reference_answers_path"]],
            root_dataset_path
        )[0]

    queries = []
    for raw_query_spec in data:
        if "query" not in raw_query_spec or "query_id" not in raw_query_spec:
            print(f"Invalid query spec, skipping this query.")
        else:
            query = raw_query_spec.get("query")
            query_id = int(raw_query_spec.get("query_id"))
            golden_tools, additional_tools = _parse_raw_query_tool_definitions(raw_query_spec, experiment_environment)
            if golden_tools is not None:
                reference_answer = _load_reference_answer(
                    reference_answers_local_dir,
                    DATASET_SETTINGS["reference_model_id"],
                    query_id
                )
                queries.append(
                    QuerySpecification(
                        id=query_id,
                        query=query,
                        reference_answer=reference_answer,
                        golden_tools=golden_tools,
                        additional_tools=additional_tools or None
                    )
                )
            else:
                print(f"Couldn't extract the tool definitions, skipping this query.")
        if max_queries_num is not None and len(queries) >= max_queries_num:
            break

    return queries


def get_queries(experiment_environment: EvaluationEnvSpec, fine_tuning_mode=False) -> List[QuerySpecification]:
    """Load queries from the dataset."""
    root_dataset_path = Path(os.getenv("ROOT_DATASET_PATH"))
    if not root_dataset_path:
        raise ValueError(f"⚠️ Root dataset folder not configured, using fallback queries.")

    remote_query_files = DATASET_SETTINGS["fine_tuning_query_files"] if fine_tuning_mode else DATASET_SETTINGS["query_files"]
    if not remote_query_files:
        raise ValueError(f"⚠️ Query files not configured properly, using fallback queries.")

    # Download the query files if needed
    local_paths = fetch_remote_paths(remote_query_files, root_dataset_path)

    # Actually load the queries
    queries_num = None if fine_tuning_mode else DATASET_SETTINGS["queries_num"]
    queries = []
    for path in local_paths:
        remaining_queries_num = None if queries_num is None else queries_num - len(queries)
        if remaining_queries_num == 0:
            break
        new_queries = _load_queries_from_single_file(path,
                                                     remaining_queries_num,
                                                     root_dataset_path,
                                                     experiment_environment)
        queries.extend(new_queries)

    return queries


def get_tools_from_queries(queries: List[QuerySpecification]) -> ToolSet:
    tools = {}

    for query_spec in queries:
        tools.update(query_spec.golden_tools)
        if query_spec.additional_tools:
            tools.update(query_spec.additional_tools)

    return tools
