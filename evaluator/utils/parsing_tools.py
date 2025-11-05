from evaluator.components.llm_provider import query_llm, get_llm
from evaluator.components.data_provider import get_queries
from evaluator.config.config_io import load_config
from pathlib import Path
import re
import json
from evaluator.utils.utils import print_iterable_verbose, log
from typing import Dict, List, Any
from copy import deepcopy
import os


def generate_and_save_additional_queries(llm, queries):
    """
    For each query in queries, use the provided LLM to generate additional_queries if not present,
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
    
    root_dir = Path(os.getenv("ROOT_DATASET_PATH", "data"))
    store_rel = os.getenv("ADDITIONAL_QUERIES_STORE_PATH", "additional_queries.json")
    central_out_path = root_dir / store_rel
    central_out_path.parent.mkdir(parents=True, exist_ok=True)
    if not central_out_path.exists():
        try:
            with central_out_path.open('w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    # generate additional queries for each query in queries
    for i, query_spec in enumerate(queries):
        path = Path(query_spec.path)
    
        # Skip generation if central store already contains this query_id
        if is_query_id_in_additional_store(query_spec.id, store_path=central_out_path):
            continue
        golden_tools = list((getattr(query_spec, 'golden_tools', {}) or {}).keys())

        existing = getattr(query_spec, 'additional_queries', None)
        # normalize wrapper {"additional_queries": {...}} if present
        if isinstance(existing, dict) and set(existing.keys()) == {"additional_queries"}:
            existing = existing.get("additional_queries")
        
        tools = [t for t in golden_tools if not _has_queries_for_tool(existing, t)]
        additional_queries = {}
        # for each tool, generate additional queries
        for tool in tools:
            additional_query = _generate_additional_query_for_tool(
                llm,
                system_prompt,
                getattr(query_spec, 'query', None),
                tool,
            )
            additional_queries[tool] = additional_query
        query_spec.additional_queries = additional_queries
        # Save additional queries to centralized file
        if additional_queries is not None:
            append_additional_queries_entry(query_spec.id, query_spec.additional_queries, central_out_path)
            log(f"Saved additional queries for query_id={query_spec.id} to {central_out_path}")

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
    # Try to extract the 'additional_queries' dict block
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


def is_query_id_in_additional_store(query_id: int, store_path: Path | None = None) -> bool:
    """
    Return True if data/additional_queries.json (list format) already contains an entry with this query_id.
    """
    try:
        out_path = store_path or (Path("data") / "additional_queries.json")
        if not out_path.exists():
            return False
        with out_path.open('r', encoding='utf-8') as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            return any(isinstance(item, dict) and item.get("query_id") == query_id for item in loaded)
    except Exception:
        return False
    return False


def append_additional_queries_entry(query_id: int, additional_queries: Dict[str, Any], store_path: Path | None = None) -> None:
    """
    Append a new entry to data/additional_queries.json in the list-of-dicts format:
      { "query_id": <int>, "additional_queries": <dict> }
    """
    out_path = store_path or (Path("data") / "additional_queries.json")
    store_list: List[Dict[str, Any]] = []
    try:
        if out_path.exists():
            with out_path.open('r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    store_list = loaded
    except Exception:
        store_list = []
    store_list.append({
        "query_id": query_id,
        "additional_queries": additional_queries,
    })
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(store_list, f, indent=2, ensure_ascii=False)

def _has_queries_for_tool(aq: dict | None, tool_name: str) -> bool:
            if not isinstance(aq, dict):
                return False
            block = aq.get(tool_name)
            if not isinstance(block, dict):
                return False
            # any non-empty string value counts as present
            return any(isinstance(v, str) and v.strip() for v in block.values())
