import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

from evaluator.components.mcp_proxy.mcp_proxy import run_mcp_proxy
from evaluator.interfaces.metric_collector import MetricCollector
from evaluator.components.data_provider import get_queries, QuerySpecification, ToolSet, get_tools_from_queries
from evaluator.utils.module_extractor import create_algorithms, create_metric_collectors
from evaluator.eval_spec import EVALUATED_ALGORITHMS, METRIC_COLLECTORS, DATASET_SETTINGS, VERBOSE
from evaluator.interfaces.tool_rag_algorithm import ToolRagAlgorithm
from evaluator.utils.csv_logger import CSVLogger
from evaluator.components.llm_provider import get_llm
from dotenv import load_dotenv

from evaluator.utils.tool_logger import ToolLogger
from evaluator.utils.utils import print_iterable_verbose

load_dotenv()

if not VERBOSE:
    # in non-verbose mode we want to suppress the excessive output from MCP server and client
    logging.disable(logging.WARNING)

MAX_RETRIES = 3
RETRY_DELAY = 5


async def run_all_experiments():

    # Set up the necessary components for the experiments:
    # - the language model
    # - the data to evaluate on
    # - the tools to use
    # - the evaluation metrics to collect and calculate
    # - the algorithms to be evaluated

    print(f"Launching evaluation setup with the following parameters:")
    for key, value in DATASET_SETTINGS.items():
        print(f"{key}: {value}")
    print(f"verbose: {VERBOSE}\n")

    provider_id = os.getenv("LLM_PROVIDER_ID")
    model_id = os.getenv("MODEL_ID")
    base_url = os.getenv("INFERENCE_SERVER_BASE_URL")

    print(f"Connecting to {provider_id} server on {base_url} serving {model_id}...")
    llm = get_llm(provider_id=provider_id, model_id=model_id, base_url=base_url)
    print("Connection established successfully.\n")

    print("Fetching query dataset...")
    queries = get_queries()
    print(f"Successfully loaded {len(queries)} queries.\n")
    print_iterable_verbose("The following queries will be executed:\n", queries)

    print("Retrieving available tool definitions...")
    tool_specs = get_tools_from_queries(queries)
    tools = await set_up_mcp(tool_specs)
    print_iterable_verbose("The following tools will be available during evaluation:\n", tools)
    print(f"Evaluation will proceed with {len(tools)} tools.\n")

    print("Loading metric collectors...")
    metric_collectors = create_metric_collectors(METRIC_COLLECTORS)
    print(f"The following metric collectors will be active during evaluation:\n{METRIC_COLLECTORS}\n")

    print("Loading experimental configurations...")
    algorithms_to_compare = create_algorithms(EVALUATED_ALGORITHMS)
    print(f"The following configurations will be executed:\n{EVALUATED_ALGORITHMS}\n")

    # Actually run the experiments
    with CSVLogger(metric_collectors, Path(os.getenv("OUTPUT_PATH")), metadata_columns=["Experiment ID", "Algorithm"]) as logger:
        for i, algo in enumerate(algorithms_to_compare):
            print(f"{'-' * 60}\nRunning Experiment {i+1} of {len(algorithms_to_compare)}: {algo.get_unique_id()}...")
            await run_experiment(algo, llm, tools, queries, metric_collectors)
            print(f"{'-' * 60}\nSummary of Experiment {i+1} - {algo.get_unique_id()}\n{'-' * 60}")
            logger.log_experiment(meta_values={"Experiment ID": i+1, "Algorithm": algo.get_unique_id()})


async def set_up_mcp(tools_to_provide: ToolSet) -> List[BaseTool]:
    await run_mcp_proxy(tools_to_provide, run_detached=True)
    mcp_proxy_port = os.getenv("MCP_PROXY_LOCAL_PORT", 9000)
    client = MultiServerMCPClient({
        "general": {
            "transport": "streamable_http",
            "url": f"http://127.0.0.1:{mcp_proxy_port}/mcp/"
        }
    })
    return await client.get_tools()


async def run_experiment(algo: ToolRagAlgorithm,
                         llm: BaseChatModel,
                         tools: List[BaseTool],
                         queries: List[QuerySpecification],
                         metric_collectors: List[MetricCollector]
                         ):

    tool_logger = ToolLogger(os.getenv("TOOL_LOG_PATH"))
    algo.set_up(llm, tools)
    for mc in metric_collectors:
        mc.set_up()

    for i, query_spec in enumerate(queries):
        print(f"Processing query {i+1} of {len(queries)}...")

        for mc in metric_collectors:
            mc.prepare_for_measurement(query_spec)

        response = None
        retrieved_tools = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response, retrieved_tools = await algo.process_query(query_spec)
                break  # success, go to next query
            except (GraphRecursionError, ValidationError) as e:
                # if we hit it, the model obviously failed to adequately address the query
                # TODO: execution errors must be tracked as a separate metric and categorized according to the error type
                print(f"Exception while processing query {i+1}: {e}")
                if attempt < MAX_RETRIES:
                    print(f"Retrying query {i+1}...")
                    continue
                else:
                    print(f"All {MAX_RETRIES} retried failed, marking query {i+1} as failed.")
                    response = "Query execution failed."
                    break
            except openai.InternalServerError as e:
                # detect gateway timeout (504) specifically
                if "504" in str(e) or "Gateway Time-out" in str(e):
                    print(f"Timeout on query {i+1} (attempt {attempt}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        print(f"All {MAX_RETRIES} retried failed, aborting the experiment.")
                # it's not a 504, or max retries reached
                raise

        executed_tools = tool_logger.get_executed_tools()

        for mc in metric_collectors:
            mc.register_measurement(
                query_spec,
                response=response,
                executed_tools=executed_tools,
                retrieved_tools=retrieved_tools
            )

    algo.tear_down()
    for mc in metric_collectors:
        mc.tear_down()


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
