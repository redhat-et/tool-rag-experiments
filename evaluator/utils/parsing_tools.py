from evaluator.components.llm_provider import query_llm
from pathlib import Path
import re
import json
from evaluator.utils.utils import print_iterable_verbose, log

def generate_and_save_additional_queries(llm, queries):
    """
    For each query in queries, use the provided LLM to generate additional_queries if not present,
    and save to the appropriate JSON file for that query (matching by query_id).
    """

    system_prompt = '''You create 5 additional queries for each tool and only return the additional queries information, given the query implemented, return in the following format as a JSON string:
                {tool_name: {"query1": "", "query2": "", "query3": "", "query4": "", "query5": ""}}  '''
    curr_file = None
    for i, query_spec in enumerate(queries):
        # If additional_queries already present, skip generating and saving
        path = Path(query_spec.path)
        if getattr(query_spec, 'additional_queries', None) or curr_file == path:
            log(f"Skipping query_id {getattr(query_spec, 'id', '<N/A>')} because additional_queries is present.")
            continue
        user_prompt = f"tool_name = {getattr(query_spec, 'golden_tools', {}).keys()}, Query= {getattr(query_spec, 'query', None)}"
        result = query_llm(llm, system_prompt, user_prompt)
        # Remove markdown/code block wrappers if present
        additional = qwen_model_parsing(result)
        query_spec.additional_queries = additional
        # Saving additional queries to the original query JSON file
        if path and additional is not None:
            if path.exists():
                import json as _json
                with path.open('r', encoding='utf-8') as f:
                    orig_queries = _json.load(f)
                for item in orig_queries:
                    if (
                        (item.get("query_id") == query_spec.id)
                        or (str(item.get("query_id")) == str(query_spec.id))
                    ):
                        item["additional_queries"] = additional
                with path.open('w', encoding='utf-8') as f:
                    _json.dump(orig_queries, f, indent=2, ensure_ascii=False)
                log(f"Successfully added additional queries to original file {path}")
                curr_file = path

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


