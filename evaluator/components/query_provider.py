import json
import os
from typing import Tuple, List
from pathlib import Path

from evaluator.eval_spec import DATASET_SETTINGS
from evaluator.utils.file_downloader import fetch_remote_paths


def _load_queries_from_single_file(query_file_path: str or Path) -> List[Tuple[str, str]]:
    with open(query_file_path, 'r') as f:
        data = json.load(f)

    queries = []
    for item in data:
        query = item.get("query", "")
        # Extract the primary tool from the API list
        api_list = item.get("api_list", [])
        if api_list:
            primary_tool = api_list[0].get("api_name", "unknown")
        else:
            primary_tool = "unknown"

        queries.append((query, primary_tool))

    return queries


def _filter_queries(initial_queries: List[Tuple[str, str]], relevant_tool_names: List[str]) -> List[Tuple[str, str]]:
    """
    Filter the given list of queries to only those that use tools in the given tool list.
    """
    # a temporary hack until the filtering starts working
    return initial_queries

    filtered_queries = []
    for query in initial_queries:
        # TODO: this is wrong and will not work. To fix the tool-name-to-function mapping, a large update to the MCP proxy is needed. This code will be updated afterwards.
        if query[1] in relevant_tool_names:
            filtered_queries.append(query)
    return filtered_queries


def get_queries(relevant_tool_names: List[str]) -> List[Tuple[str, str]]:
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
            new_queries = _filter_queries(_load_queries_from_single_file(path), relevant_tool_names)
            if len(queries) + len(new_queries) > DATASET_SETTINGS["max_queries_num"]:
                num_new_queries_to_add = DATASET_SETTINGS["max_queries_num"] - len(queries)
                queries.extend(new_queries[:num_new_queries_to_add])
                break
            queries.extend(new_queries)
        return queries

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Using fallback queries")
        return get_fallback_queries()


def get_fallback_queries() -> List[Tuple[str, str]]:
    """Fallback queries if dataset loading fails."""
    return [
        ("What is the weather in New York?", "weather_info"),
        ("How many words are in 'Hello World, this is a test sentence'?", "word_count"),
        ("Reverse this text: Python Experiment", "reverse_string"),
        ("Convert this to uppercase: llamastack", "uppercase"),
        ("Give me an insurance evaluation score", "insurance_scorer")
    ]
