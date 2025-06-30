import time
from llama_stack.apis.vector_io import QueryChunksResponse
from typing import Dict, List, Any, Optional, Union
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from src.tools.info import available_client_tools
from src.tools.metrics import add_metric

def get_tool_names(response: QueryChunksResponse) -> list[str]:
    tool_names = {
        response.chunks[0].metadata["tool_name"],
        response.chunks[1].metadata["tool_name"],
        response.chunks[2].metadata["tool_name"]
    }


    return list(tool_names)

def get_tool_embedding(vector_db_id: str, client: LlamaStackClient, query: str):
    return client.vector_io.query(vector_db_id=vector_db_id, query=query)

def get_query_id(query_obj):
    """Extract an ID from a query object for better test identification."""
    if isinstance(query_obj, dict) and 'id' in query_obj:
        return query_obj['id']
    elif isinstance(query_obj, dict) and 'query' in query_obj:
        # Use first few words of query if no ID is available
        words = query_obj['query'].split()[:5]
        return '_'.join(words).lower().replace(',', '').replace('.', '')
    return "unknown_query"

def execute_query(
    client: LlamaStackClient,
    prompt: str,
    model: str,
    tools: Union[List[str], List[Any]], # list of toolgroup_ids or tool objects
    instructions: Optional[str] = None,
    max_tokens: int = 4096
) -> Dict[str, Any]:
    """Execute a single query with a given set of tools."""

    if instructions is None:
        # Default instructions for general tool use
        instructions = """
            You MUST always use a tool.
            """

    agent = Agent(
        client,
        model=model,
        instructions=instructions,
        tools=tools,
        tool_config={"tool_choice": "auto"},
        sampling_params={"max_tokens": max_tokens}
    )

    session_id = agent.create_session(session_name=f"Test_{int(time.time())}")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})" if not all(isinstance(t, str) for t in tools) else "")

    turn_response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        session_id=session_id,
        stream=False
    )
    return turn_response

def run_client_tool_test(model, vector_db_id, query_obj, llama_client, logger):
    """Run a single test for a specific server type, model, and query."""
    query_id = get_query_id(query_obj)
    prompt = query_obj['query']
    expected_tool_call = query_obj['tool_call']

    tool_candidates = get_tool_names(get_tool_embedding(
        query=prompt,
        vector_db_id=vector_db_id,
        client=llama_client
    ))

    tools = []
    for candidate in tool_candidates:
        tools.append(available_client_tools.get(candidate))

    logger.info(f"Testing query '{query_id}' with model {model}")
    logger.info(f"Query: {prompt[:50]}...")
    logger.info(f"Tool Candidates: {tool_candidates}.")

    try:
        response = execute_query(
            client=llama_client,
            prompt=prompt,
            model=model,
            tools=tools,
        )
        steps = response.steps

        try:
            tools_used = steps[1].tool_calls[0].tool_name
        except Exception as e:
            logger.error(f"Error extracting tool name: {e}")
            tools_used = None
        tool_call_match = True if tools_used == expected_tool_call else False
        logger.info(f"Tool used: {tools_used} Tool expected: {expected_tool_call} match: {tool_call_match} ")

        final_response = ""
        try:
            final_response = steps[2].api_model_response.content.strip()
            inference_not_empty = True if final_response != '' else False
        except Exception as e:
            logger.error(f"Error checking inference content: {e}")
            inference_not_empty = False
        logger.info(f'Inference not empty: {inference_not_empty}')
        logger.info(f"Query '{query_id}' succeeded with model {model} and the response \n {final_response}")

        add_metric(
            model=model,
            query_id=query_id,
            status="SUCCESS",
            tool_call_match=tool_call_match,
            inference_not_empty=inference_not_empty,
            expected_tool_call=expected_tool_call
        )

        return True

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Query '{query_id}' failed with model {model}: {error_msg}")

        add_metric(
            model=model,
            query_id=query_id,
            status="FAILURE",
            tool_call_match=False,
            inference_not_empty=False,
            expected_tool_call=expected_tool_call,
            error=error_msg
        )

        return False