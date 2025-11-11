import itertools
import json
import math
import os
import random
from json import JSONDecodeError
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from numpy import str_
from pydantic import BaseModel, Field
from evaluator.components.llm_provider import query_llm, get_llm
from evaluator.config.schema import EnvironmentConfig, DatasetConfig
from evaluator.utils.file_downloader import fetch_remote_paths
from evaluator.utils.utils import log
from evaluator.config.config_io import load_config
from tqdm import tqdm
import re
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
    examples: Optional[Dict[str, Any]] = None
    reference_answer: Optional[str] = None
    golden_tools: ToolSet = Field(default_factory=dict)
    additional_tools: Optional[ToolSet] = None
    demo_mode: Optional[bool] = False


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


def _load_random_tools(categories: List[str] or None, tool_num: int, dataset_config: DatasetConfig) -> ToolSet:
    """
    Load tool_num random tools from the given categories.
    """
    # Download dataset files if needed
    root_dataset_path = Path(os.getenv("ROOT_DATASET_PATH"))
    local_paths = fetch_remote_paths(dataset_config.tool_file_paths, root_dataset_path)

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


def _parse_raw_query_tool_definitions(
        query: Dict[str, Any],
        experiment_environment: EnvironmentConfig,
        dataset_config: DatasetConfig,
) -> Tuple[ToolSet or None, ToolSet or None]:
    """
    This method receives the query dict in the ToolBench dataset format.
    It returns the golden set of tools for this query. It is returned as a dictionary where
    each key is a unique MCP name of the tool and the corresponding value is a ToolBench dictionary of that tool.
    If required and available, it returns additional tools to use for baseline evaluation.
    """
    relevant_apis = query.get("relevant APIs")
    if not relevant_apis:
        log(f"No relevant APIs provided in the query specification:\n{query}")
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
            log(f"Bad query specification: the api doesn't appear in the API list:\n{query}")
            return None, None
        mcp_tool_name = create_unique_mcp_tool_name(category_name, tool_name, api_name)
        golden_tools[mcp_tool_name] = tool_spec

    return golden_tools, None

    # -------------- Deprecated code - soon to be removed --------------

    # extract the rest of the available tools, if available
    additional_tool_apis = [tool_api for tool_api in query["api_list"] if tool_api not in golden_tools.values()]
    additional_tools = tool_api_list_to_tool_set(additional_tool_apis)

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
        random_toolset = _load_random_tools(
            category_names, required_number_of_additional_tools - len(additional_tools), dataset_config)
        additional_tools.update(random_toolset)
        return golden_tools, additional_tools

    random_toolset = _load_random_tools(None, required_number_of_additional_tools, dataset_config)
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
        experiment_environment: EnvironmentConfig,
        dataset_config: DatasetConfig,
) -> List[QuerySpecification]:
    with open(query_file_path, 'r') as f:
        data = json.load(f)

    if dataset_config.reference_model_id is None:
        # judge-based evaluation is disabled - no need to load reference answers
        reference_answers_local_dir = None
    else:
        reference_answers_local_dir = fetch_remote_paths(
            [dataset_config.reference_answers_path],
            root_dataset_path
        )[0]

    queries = []
    for raw_query_spec in data:
        if "query" not in raw_query_spec or "query_id" not in raw_query_spec:
            log(f"Invalid query spec, skipping this query.")
        else:
            query = raw_query_spec.get("query")
            query_id = int(raw_query_spec.get("query_id"))
            golden_tools, additional_tools = (
                _parse_raw_query_tool_definitions(raw_query_spec, experiment_environment, dataset_config))
            if golden_tools is not None:
                reference_answer = _load_reference_answer(
                    reference_answers_local_dir,
                    dataset_config.reference_model_id,
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
                log(f"Couldn't extract the tool definitions, skipping this query.")
        if max_queries_num is not None and len(queries) >= max_queries_num:
            break

    return queries


def get_queries(
        experiment_environment: EnvironmentConfig,
        dataset_config: DatasetConfig,
        fine_tuning_mode=False
) -> List[QuerySpecification]:
    """Load queries from the dataset."""
    root_dataset_path = Path(os.getenv("ROOT_DATASET_PATH"))
    if not root_dataset_path:
        raise ValueError(f"⚠️ Root dataset folder not configured, using fallback queries.")

    remote_query_files = dataset_config.fine_tuning_query_file_paths if fine_tuning_mode else dataset_config.query_file_paths
    if not remote_query_files:
        raise ValueError(f"⚠️ Query files not configured properly, using fallback queries.")

    # Download the query files if needed
    local_paths = fetch_remote_paths(remote_query_files, root_dataset_path)

    # Actually load the queries
    queries_num = None if fine_tuning_mode else dataset_config.queries_num
    queries = []
    for path in local_paths:
        remaining_queries_num = None if queries_num is None else queries_num - len(queries)
        if remaining_queries_num == 0:
            break
        new_queries = _load_queries_from_single_file(path,
                                                     remaining_queries_num,
                                                     root_dataset_path,
                                                     experiment_environment,
                                                     dataset_config)
        queries.extend(new_queries)

    return queries


def get_tools_from_queries(queries: List[QuerySpecification]) -> ToolSet:
    tools = {}

    cfg_path = "evaluator/config/yaml/tool_rag_experiments.yaml"
    cfg = load_config(cfg_path, use_defaults=True)
    examples = cfg.data.generate_examples

    model_id = cfg.data.additional_examples_model_id
    # Base tools from the dataset
    for query_spec in tqdm(queries, desc="Getting tools from queries"):
        tools.update(query_spec.golden_tools)
        if query_spec.additional_tools:
            tools.update(query_spec.additional_tools)

        #Getting or generating additional examples for tools that don't have them
        if examples:
            golden_tools = query_spec.golden_tools
            for tool in golden_tools:
                examples_exists = is_tool_in_additional_store(tool, query_spec.id)
                if not examples_exists:
                    # TODO: get the model id from the config file, This doesnt work
                    llm = get_llm(model_id=model_id, model_config=cfg.models)
                    tools[tool]["examples"] = generate_and_save_examples(llm, tool, query_spec)
                else:
                    aq = get_additional_query(query_spec.id)
                    tools[tool]["examples"] = aq[tool]
    return tools


def load_examples_store(path: str | None = None) -> List[Dict[str, Any]]:
    """
    Load the centralized additional queries store.
    Expected format: a JSON list of objects {"query_id": int, "examples": {...}}.
    Returns an empty list if the file doesn't exist or cannot be parsed.
    """
    try:
        store_path = Path(path) if path else Path(os.getenv("EXAMPLES_STORE_PATH"))
        if not store_path.exists():
            return []
        with store_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, list) else []
    except Exception:
        return []


def get_additional_query(query_id: int) -> Dict[str, Any] | None:
    """
    Return ALL examples for the given query_id by merging entries
    from data/examples.json (supports multiple records per query_id).
    """
    store = load_examples_store()
    if not store:
        return None
    for item in store:
        next_query_id = item.get("query_id")
        if next_query_id == query_id:
            return item.get('examples')
    return None


def get_examples_by_tool_name(tool_name: str) -> Dict[str, Any] | None:
    """
    Look through data/examples.json (list of entries with an
    "examples" mapping) and return the queries map for the specified
    tool_name if present. Tries exact match first, then tolerant variants that
    add/remove a trailing period.
    """
    store = load_examples_store()

    for item in store:
        if not isinstance(item, dict):
            continue
        aq_map = item.get("examples")
        if not isinstance(aq_map, dict):
            continue
        block = aq_map.get(tool_name)
        if isinstance(block, dict):
            return block
    return None

def generate_and_save_examples(llm, tool_name, query_spec, store_path: Path | None = None):
    """
    For each query in queries, use the provided LLM to generate examples if not present,
    and save to the appropriate JSON file for that query (matching by query_id).
    """

    # system_prompt = '''You create 5 additional queries for each tool and only return the additional queries information, given the query implemented, return in the following format as a JSON string:
    #             {tool_name: {"query1": "", "query2": "", "query3": "", "query4": "", "query5": ""}}  '''
    
    system_prompt = '''You are a tool query generator. For each specified tool, create EXACTLY 5 concise, natural-language user queries suitable for invoking that tool.

    Output requirements:
    - Return ONLY a single JSON object (no explanations, no code fences).
    - Shape:
    {
        "query1",
        "query2",
        "query3",
        "query4",
        "query5"
    }
    - Each query must be ≤ 20 words and phrased as a natural request (not SQL), ending with a question mark when appropriate.
    - Preserve any placeholder tokens as given (e.g., {id}, tt1234567) without inventing new identifiers.
    - Avoid near-duplicates; vary phrasing and sub-intents across the 5 queries.
    - Use English.

    Context you will receive:
    - tool_name(s): a set of tool identifiers
    - original user query: the initial task description
    '''     
    
    out_path = store_path or Path(os.getenv("EXAMPLES_STORE_PATH"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        try:
            with out_path.open('w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"error creating central_out_path: {e}")
            pass

    example = _generate_additional_query_for_tool(
            llm,
            system_prompt,
            query_spec.query,
            tool_name,
        )

    examples = {}
    examples[tool_name] = example
    query_spec.examples = examples
    # Save additional queries to centralized file

    if examples is not None:
        append_examples_entry(query_spec.id, query_spec.examples, out_path)
    return examples

def _generate_additional_query_for_tool(llm, system_prompt: str, query_text: str, tool_name: str) -> Dict[str, Any] | None:
    """
    Call the LLM to generate additional queries for a single tool, retrying until
    a mapping with query1..query5 is produced or max attempts are reached.
    Returns the parsed dict (queries map or tool->queries map) or None.
    """
    correct_response = False
    iteration = 0
    additional_query = None
    while correct_response is False:
        user_prompt = f"tool_name = {tool_name}, Query= {query_text}"
        result = query_llm(llm, system_prompt, user_prompt)
        model_id = str(getattr(llm, "model", "") or getattr(llm, "model_name", "") or "")
        if "llama3.1:8b" in model_id:
            additional_query = lama_model_parsing(result)
        else:
            additional_query = qwen_model_parsing(result)
        correct_response = has_required_query_keys(additional_query)
        iteration += 1
        if iteration > 10:
            log(f"Failed to generate additional queries for tool {tool_name} after 5 iterations")
            break
    return additional_query

def lama_model_parsing(response: str):
    """
    Parse the response from the Lama model and return the additional queries.
    """
    if not response:
        return None
    text = response.strip()
    quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if not quoted:
        return None
    return {f"query{i}": v for i, v in enumerate(quoted[:5], start=1)}

def qwen_model_parsing(response: str):
    """
    Parse the response from the Qwen model and return the additional queries.
    """
    # Remove markdown/code block wrappers if present
    match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    response_text = match.group(1).strip() if match else response
    # Try to extract the 'examples' dict block
    additional = None
    response_text = response_text.strip()
    try:
        additional = json.loads(response_text)
    except Exception as e:
        additional = None
    return additional


def has_required_query_keys(response: Any) -> bool:
    """
    Return True iff the response contains all of the keys Query1..Query5 (case-insensitive)
    under at least one mapping block. Accepts either a parsed dict or a JSON/string response
    (optionally wrapped in ```json fences).
    """
    required = {"query1", "query2", "query3", "query4", "query5"}

    def _check_dict(d: Dict[str, Any]) -> bool:
        if not isinstance(d, dict):
            return False
        # Case 1: top-level is the queries map
        keys_lc = {k.lower() for k in d.keys() if isinstance(k, str)}
        if required.issubset(keys_lc):
            return True
        # Case 2: top-level is tool_name -> queries_map
        for _, v in d.items():
            if isinstance(v, dict):
                inner_keys = {k.lower() for k in v.keys() if isinstance(k, str)}
                if required.issubset(inner_keys):
                    return True
        return False

    # If it's already a dict, check directly
    if isinstance(response, dict):
        return _check_dict(response)

    # If it's a string, strip fences and try JSON
    if isinstance(response, str):
        text = response.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        try:
            obj = json.loads(text)
            return _check_dict(obj)
        except Exception:
            # Heuristic fallback: ensure all tokens appear (weak check)
            found = {tok for tok in required if re.search(fr"\b{tok}\b", text, flags=re.IGNORECASE)}
            return required.issubset(found)

    return False


def is_tool_in_additional_store(tool_name: str, query_id: int, store_path: Path | None = None) -> bool:
    """
    Return True if any entry in the centralized store has examples containing this tool_name
    (tolerates trailing-dot variants).
    """

    try:
        out_path = store_path or Path(os.getenv("EXAMPLES_STORE_PATH"))
        if not out_path.exists():
            log(f"examples.json not found, creating empty file")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("[]", encoding="utf-8")
            return False

        with out_path.open('r', encoding='utf-8') as f:
            loaded = json.load(f)

        for item in loaded:
            next_query_id = item.get("query_id")
            if next_query_id == query_id:
                aq = item.get("examples")
                if tool_name in aq:
                    return True
        return False
    except Exception:
        return False


def append_examples_entry(query_id: int, examples: Dict[str, Any], store_path: Path | None = None) -> None:
    """
    Append a new entry to data/examples.json in the list-of-dicts format:
      { "query_id": <int>, "examples": <dict> }
    """
    out_path = store_path or Path(os.getenv("EXAMPLES_STORE_PATH"))
    store_list: List[Dict[str, Any]] = []
    try:
        if out_path.exists():
            with out_path.open('r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    store_list = loaded
    except Exception:
        store_list = []

    # Upsert: if entry for query_id exists, merge/overwrite per tool; otherwise append new
    idx = None
    for i, item in enumerate(store_list):
        if isinstance(item, dict) and item.get("query_id") == query_id:
            idx = i
            break

    if idx is None:
        store_list.append({
            "query_id": query_id,
            "examples": examples or {},
        })
    else:
        existing_block = store_list[idx].get("examples")
        if not isinstance(existing_block, dict):
            existing_block = {}
        for tool_name, qmap in (examples or {}).items():
            if isinstance(qmap, dict):
                existing_block[tool_name] = qmap
        store_list[idx]["examples"] = existing_block
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(store_list, f, indent=2, ensure_ascii=False)


